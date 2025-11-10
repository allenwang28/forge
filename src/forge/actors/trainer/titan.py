# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Callable

import torch
import torch.distributed.checkpoint as dcp
import torchstore as ts

from monarch.actor import endpoint
from torch import Tensor
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Checkpoint,
    Comm,
    Compile,
    Job,
    LRScheduler,
    MemoryEstimation,
    Model,
    Optimizer,
    Parallelism,
    Quantize,
    Training,
)
from torchtitan.experiments.forge import StepwiseTrainer
from torchtitan.experiments.forge.job_config import ForgeJobConfig

from forge.actors._torchstore_utils import (
    DcpHandle,
    get_dcp_whole_state_dict_key,
    get_param_key,
    rdma_available,
)

from forge.controller import ForgeActor
from forge.data.utils import batch_to_device
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class TitanTrainer(ForgeActor):
    """A generic trainer actor implementation built on top of TorchTitan.

    Built on top of TorchTitan's StepwiseTrainer, this actor provides a complete training
    loop for reinforcement learning. It performs forward and backward passes with gradient
    computation, optimization steps, and checkpoint management. Unlike the ReferenceModel
    actor which only runs forward passes, TitanTrainer actively updates the policy model
    parameters through gradient descent.

    The trainer supports the same distributed training strategies that TorchTitan does,
    including but not limited to, tensor parallelism, data parallelism, and FSDP
    (Fully Sharded Data Parallel). It is typically used in conjunction with ReferenceModel
    for policy optimization algorithms like GRPO (Group Relative Policy Optimization),
    where it optimizes the policy against a loss that includes KL divergence penalties
    from the reference model.

    The trainer handles:
    - Forward and backward propagation with automatic mixed precision (AMP)
    - Optimizer steps with learning rate scheduling
    """

    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    quantize: Quantize = field(default_factory=Quantize)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)
    # Non JobConfig-related fields
    loss: Callable = lambda logits, **targets: logits
    state_dict_key: str = "model_state_dict"
    use_dcp: bool = not rdma_available()
    dcp_path: str = "forge_dcp_tmp"

    def __post_init__(self):
        super().__init__()
        if self.use_dcp:
            torch.serialization.set_crc32_options(False)

        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        self.step = 1  # fragile contract.
        self.num_training_steps = self.training.steps
        self.gradient_accumulation_steps = 1
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Compiling loss")
        self.loss = torch.compile(self.loss)

    @endpoint
    async def setup(self):
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        for key in {
            "loss",
            "state_dict_key",
            "use_dcp",
            "dcp_path",
        }:
            engine_config.pop(key)  # Not part of job config
        self.engine = StepwiseTrainer(
            ForgeJobConfig(**engine_config), loss_fn=self.loss
        )
        self.engine.setup()
        # StepwiseTrainer's setup() already loads checkpoint and zeros gradients

    def _forward_backward(
        self, inputs: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> float:
        """Internal implementation: Execute forward and backward pass.

        Args:
            inputs: Global batch inputs - will be sliced per DP rank
            targets: Global batch targets - will be sliced per DP rank

        Returns:
            Loss value (float)
        """
        self.engine.gc_handler.run(self.step)

        # Slice global batch for this DP rank
        rank = self.engine.dp_rank
        dp_degree = self.engine.dp_degree
        local_batch_size = self.training.local_batch_size

        start_idx = rank * local_batch_size
        end_idx = (rank + 1) * local_batch_size

        # Slice inputs for this rank
        local_inputs = {
            k: v[start_idx:end_idx] if isinstance(v, Tensor) else v
            for k, v in inputs.items()
        }

        # Slice targets for this rank
        local_targets = {
            k: v[start_idx:end_idx] if isinstance(v, Tensor) else v
            for k, v in targets.items()
        }

        # Move to device
        batch_to_device(local_inputs, self.engine.device)
        batch_to_device(local_targets, self.engine.device)

        # Forward + backward
        loss = self.engine.forward_backward(local_inputs, local_targets)
        torch.distributed.all_reduce(loss)

        # Return loss as float
        return loss.detach().item()

    @endpoint
    async def forward_backward(
        self, inputs: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> float:
        """Protocol: Execute forward and backward pass.

        Args:
            inputs: Global batch inputs - will be sliced per DP rank
            targets: Global batch targets - will be sliced per DP rank

        Returns:
            Loss value (float)
        """
        return self._forward_backward(inputs, targets)

    def _optim_step(self) -> dict:
        """Internal implementation: Apply optimizer step.

        Returns:
            dict with keys: step, learning_rate, accumulated_microbatches
        """
        step_info = self.engine.optim_step()
        self.step = step_info["step"]  # Sync step counter with StepwiseTrainer

        # Save checkpoint
        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=self.step == self.num_training_steps,
        )

        return step_info

    @endpoint
    async def optim_step(self) -> dict:
        """Protocol: Apply optimizer step.

        Returns:
            dict with keys: step, learning_rate, accumulated_microbatches
        """
        return self._optim_step()

    @endpoint
    async def clear_gradients(self) -> None:
        """Protocol: Clear accumulated gradients without applying them."""
        self.engine.clear_gradients()

    @endpoint
    async def get_info(self) -> dict:
        """Protocol: Get static trainer and model metadata.

        Returns:
            dict with keys:
                - model_name: str
                - step: int
                - config: dict (model config)
                - parallelism: dict (DP/TP/PP config)
        """
        return {
            "model_name": f"{self.model.name}/{self.model.flavor}",
            "step": self.step,
            "config": {
                "vocab_size": self.engine.model_args.vocab_size
                if hasattr(self.engine.model_args, "vocab_size")
                else None,
                "hidden_size": self.engine.model_args.dim
                if hasattr(self.engine.model_args, "dim")
                else None,
                "num_layers": self.engine.model_args.n_layers
                if hasattr(self.engine.model_args, "n_layers")
                else None,
                "num_attention_heads": self.engine.model_args.n_heads
                if hasattr(self.engine.model_args, "n_heads")
                else None,
                "max_seq_len": self.engine.model_args.max_seq_len
                if hasattr(self.engine.model_args, "max_seq_len")
                else None,
            },
            "parallelism": {
                "dp_degree": self.engine.dp_degree,
                "dp_rank": self.engine.dp_rank,
                "tp_degree": self.engine.parallel_dims.tp
                if hasattr(self.engine.parallel_dims, "tp")
                else 1,
                "pp_degree": self.engine.parallel_dims.pp
                if hasattr(self.engine.parallel_dims, "pp")
                else 1,
                "device": str(self.engine.device),
                "local_batch_size": self.training.local_batch_size,
            },
        }

    @endpoint
    async def get_status(self) -> dict:
        """Protocol: Get current runtime status.

        Returns:
            dict with keys:
                - step: int
                - accumulated_microbatches: int
        """
        return {
            "step": self.step,
            "accumulated_microbatches": self.engine._accumulated_microbatches,
        }

    @endpoint
    async def train_step(
        self, inputs: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]
    ) -> float:
        """Legacy method: Combined forward_backward + optim_step.

        This method is kept for backward compatibility with GRPO.
        New code should use forward_backward() and optim_step() separately.
        """
        t = Tracer("rl_trainer_perf/step", timer="gpu", track_memory=True)
        t.start()

        # Use internal implementation (not endpoint)
        loss = self._forward_backward(inputs, targets)
        t.step("forward_backward")

        # Use internal implementation (not endpoint)
        step_info = self._optim_step()
        current_lr = step_info["learning_rate"]
        record_metric("rl_trainer/learning_rate", current_lr, Reduce.MIN)
        record_metric("rl_trainer/avg_loss", loss, Reduce.MEAN)

        t.step("optimizer_step")
        t.stop()
        return loss

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        """Push weights to torchstore in HF format."""
        t = Tracer("rl_trainer_perf/push_weights", timer="gpu", track_memory=True)
        t.start()
        logger.info(f"Pushing weights for policy version {policy_version}")

        start_time = time.perf_counter()
        if "model" not in self.engine.checkpointer.states:
            raise RuntimeError("Model state not found in checkpointer state")

        sd = self.engine.checkpointer.states["model"].state_dict()
        flattened_state_dict, _ = flatten_state_dict(sd)
        t.step("flatten_state_dict")
        if self.engine.checkpointer.sd_adapter is None:
            raise RuntimeError(
                "Trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            )
        hf_state_dict = self.engine.checkpointer.sd_adapter.to_hf(flattened_state_dict)
        t.step("to_hf")
        if self.use_dcp:
            key = get_dcp_whole_state_dict_key(policy_version)
            dcp_id = f"{self.dcp_path}/{key}"
            storage_writer = torch.distributed.checkpoint.FileSystemWriter(
                dcp_id, single_file_per_rank=False, thread_count=8
            )
            metadata = dcp.save(storage_writer=storage_writer, state_dict=hf_state_dict)
            dcp_handle = DcpHandle(
                checkpoint_id=dcp_id,
                metadata=metadata,
                param_names=hf_state_dict.keys(),
            )
            await ts.put(key, dcp_handle)
            t.step("dcp_save")
        else:
            for name, param in hf_state_dict.items():
                key = get_param_key(policy_version, name)
                await ts.put(key, param)
            t.step("ts_save")
        t.stop()
        end_time = time.perf_counter()
        logger.info("Completed weights push in %.2f seconds", end_time - start_time)

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()
