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

# Import directly from torchtitan
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import TORCH_DTYPE_MAP
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
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools import utils

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
    """A generic trainer actor built directly on TorchTitan's composable APIs.

    This trainer uses TorchTitan's components directly instead of the ForgeEngine wrapper,
    giving full control over the setup flow and making customization easy. It provides a
    complete training loop for reinforcement learning with distributed training support.

    Supports:
    - Tensor parallelism (TP), FSDP, and data parallelism (DP)
    - Activation checkpointing and torch.compile
    - Automatic mixed precision (AMP)
    - Checkpoint management with HuggingFace format export
    - Weight export to TorchStore or local filesystem

    The trainer composes TorchTitan components in setup():
    1. Initialize distributed training
    2. Build model from train spec
    3. Apply parallelism (TP, FSDP, activation checkpointing)
    4. Initialize weights
    5. Build optimizer and LR scheduler
    6. Setup checkpointing
    7. Setup training contexts (AMP, loss parallel)
    """

    # TorchTitan config components
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

    # Forge-specific fields
    loss: Callable = lambda logits, **targets: logits
    use_dcp: bool = not rdma_available()
    dcp_path: str = "forge_dcp_tmp"

    def __post_init__(self):
        super().__init__()

        # Set DCP options if needed
        if self.use_dcp:
            torch.serialization.set_crc32_options(False)

        # Convert dict fields to proper types
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        # Initialize runtime state
        self.step = 1
        self.num_training_steps = self.training.steps

        # These will be set in setup()
        self.device = None
        self.parallel_dims = None
        self.world_mesh = None
        self.dp_degree = None
        self.dp_rank = None
        self.model_parts = None
        self.optimizers = None
        self.lr_schedulers = None
        self.checkpointer = None
        self.train_context = None
        self.amp_context = None
        self.gc_handler = None
        self.model_args = None

        # Compile loss function
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Compiling loss function")
        self.loss = torch.compile(self.loss)

    @endpoint
    async def setup(self):
        """Setup trainer using TorchTitan's composable APIs.

        This method orchestrates TorchTitan components to build the training
        environment. The flow is explicit and customizable.
        """

        # ===== Phase 1: Distributed Setup =====

        # Determine device
        device_module = utils.device_module
        device_type = utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # Initialize distributed
        dist_utils.init_distributed(
            self.comm,
            enable_cpu_backend=self.training.enable_cpu_offload,
        )

        # Build parallel dimensions
        world_size = int(os.environ["WORLD_SIZE"])
        self.parallel_dims = ParallelDims(
            dp_shard=self.parallelism.data_parallel_shard_degree,
            dp_replicate=self.parallelism.data_parallel_replicate_degree,
            cp=self.parallelism.context_parallel_degree,
            tp=self.parallelism.tensor_parallel_degree,
            pp=self.parallelism.pipeline_parallel_degree,
            ep=self.parallelism.expert_parallel_degree,
            etp=self.parallelism.expert_tensor_parallel_degree,
            world_size=world_size,
        )
        self.world_mesh = self.parallel_dims.world_mesh

        # Extract DP info for data loading
        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            self.dp_degree = dp_mesh.size()
            self.dp_rank = dp_mesh.get_local_rank()
        else:
            self.dp_degree = 1
            self.dp_rank = 0

        # Setup determinism (using default seed and deterministic=False)
        dist_utils.set_determinism(self.world_mesh, self.device)

        # Setup garbage collection
        self.gc_handler = utils.GarbageCollection(
            gc_freq=self.training.gc_freq,
            debug=self.training.gc_debug,
        )

        # ===== Phase 2: Model Building =====

        # Get train spec from registry
        train_spec = get_train_spec(self.model.name)

        # Get model args for this flavor
        self.model_args = train_spec.model_args[self.model.flavor]

        # Update model args from config
        self.model_args.update_from_config(self)

        # Build model on meta device
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[self.training.dtype]),
        ):
            model = train_spec.model_cls(self.model_args)

        # Calculate model size
        param_count, flops = self.model_args.get_nparams_and_flops(
            model, self.training.seq_len
        )
        logger.info(f"Model has {param_count:,} parameters")
        logger.info(f"Model FLOPs per token: {flops:,}")

        # ===== Phase 3: Apply Parallelism =====

        # Check for pipeline parallelism
        if self.parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallelism not yet supported. "
                "Set parallelism.pipeline_parallel_degree=1"
            )

        # Apply TP, FSDP, activation checkpointing, compile
        model = train_spec.parallelize_fn(model, self.parallel_dims, self)

        # ===== Phase 4: Initialize Weights =====

        # Determine init device
        if self.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        # Move off meta device and initialize
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.init_weights(buffer_device=buffer_device)
        model.train()

        self.model_parts = [model]

        # ===== Phase 5: Build Training Components =====

        # Build optimizers
        self.optimizers = train_spec.build_optimizers_fn(
            self.model_parts,
            self.optimizer,
            self.parallel_dims,
        )

        # Build LR schedulers
        self.lr_schedulers = train_spec.build_lr_schedulers_fn(
            self.optimizers,
            self.lr_scheduler,
            self.training.steps,
        )

        # ===== Phase 6: Setup Checkpointing =====

        self.checkpointer = CheckpointManager(
            dataloader=None,  # No dataloader in RL training
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            checkpoint_config=self.checkpoint,
            sd_adapter=(
                train_spec.state_dict_adapter(
                    self.model_args,
                    self.model.hf_assets_path,
                )
                if train_spec.state_dict_adapter
                else None
            ),
        )

        # Load checkpoint if exists
        self.checkpointer.load(step=self.step)

        # ===== Phase 7: Setup Training Contexts =====

        # Loss parallel context
        loss_parallel_enabled = (
            self.parallel_dims.tp_enabled and not self.parallelism.disable_loss_parallel
        )
        enable_compiled_autograd = self.compile.enable
        self.train_context = dist_utils.get_train_context(
            loss_parallel_enabled, enable_compiled_autograd
        )

        # AMP context
        self.amp_context = dist_utils.maybe_enable_amp(
            self.parallel_dims,
            self.training.mixed_precision_param,
            device_type,
        )

        # ===== Phase 8: Initialize Gradient State =====

        self.optimizers.zero_grad()

        logger.info("TitanTrainer setup complete")

    def forward_backward(
        self, inputs: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> Tensor:
        """Execute forward and backward pass.

        Args:
            inputs: Dictionary containing model inputs (e.g., input_ids)
            targets: Dictionary containing targets for loss computation

        Returns:
            Loss tensor
        """
        assert len(self.model_parts) == 1, "Pipeline parallelism not supported"
        model = self.model_parts[0]

        if self.parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")

        with self.train_context():
            with self.amp_context:
                # Forward pass
                logits = model(**inputs)

                # Compute loss
                loss = self.loss(logits, **targets)

            # Free logits before backward to avoid memory peak
            del logits

            # Backward pass (gradients accumulate)
            loss.backward()

        return loss

    @endpoint
    async def train_step(
        self, inputs: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]
    ) -> float:
        """Execute one complete training step.

        This method performs forward pass, backward pass, optimizer step,
        and checkpointing in one call.

        Args:
            inputs: List of input dicts, one per DP rank
            targets: List of target dicts, one per DP rank

        Returns:
            Loss value (float)
        """
        t = Tracer("rl_trainer_perf/step", timer="gpu", track_memory=True)
        t.start()

        # Run GC if needed
        self.gc_handler.run(self.step)

        # Get data for this DP rank
        local_inputs = inputs[self.dp_rank]
        local_targets = targets[self.dp_rank]

        # Move to device
        batch_to_device(local_inputs, self.device)
        batch_to_device(local_targets, self.device)

        # Forward + backward
        loss = self.forward_backward(local_inputs, local_targets)

        # All-reduce loss for logging
        torch.distributed.all_reduce(loss)

        t.step("forward_backward")

        # Get current LR
        current_lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        record_metric("rl_trainer/learning_rate", current_lr, Reduce.MIN)

        # Optimizer step
        self.optimizers.step()
        self.optimizers.zero_grad()
        self.lr_schedulers.step()

        t.step("optimizer_step")

        # Extract loss value
        loss = loss.detach().item()
        record_metric("rl_trainer/avg_loss", loss, Reduce.MEAN)

        # Increment step
        self.step += 1

        # Checkpoint
        self.checkpointer.save(
            curr_step=self.step,
            last_step=self.step == self.num_training_steps,
        )

        t.step("save_checkpoint")
        t.stop()

        return loss

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        """Push weights to TorchStore in HuggingFace format.

        Args:
            policy_version: Version number for this policy checkpoint
        """
        t = Tracer("rl_trainer_perf/push_weights", timer="gpu", track_memory=True)
        t.start()
        logger.info(f"Pushing weights for policy version {policy_version}")

        start_time = time.perf_counter()

        # Get model state dict
        if "model" not in self.checkpointer.states:
            raise RuntimeError("Model state not found in checkpointer state")

        sd = self.checkpointer.states["model"].state_dict()
        flattened_state_dict, _ = flatten_state_dict(sd)
        t.step("flatten_state_dict")

        # Convert to HF format
        if self.checkpointer.sd_adapter is None:
            raise RuntimeError(
                "Trying to save checkpoint in HF safetensors format, "
                "but sd_adapter is not provided."
            )

        hf_state_dict = self.checkpointer.sd_adapter.to_hf(flattened_state_dict)
        t.step("to_hf")

        # Save using DCP or direct TorchStore
        if self.use_dcp:
            # Save via distributed checkpoint
            key = get_dcp_whole_state_dict_key(policy_version)
            dcp_id = f"{self.dcp_path}/{key}"

            storage_writer = dcp.FileSystemWriter(
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
            # Save directly to TorchStore
            for name, param in hf_state_dict.items():
                key = get_param_key(policy_version, name)
                await ts.put(key, param)
            t.step("ts_save")

        t.stop()
        end_time = time.perf_counter()
        logger.info("Completed weights push in %.2f seconds", end_time - start_time)

    @endpoint
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.checkpointer:
            self.checkpointer.close()
