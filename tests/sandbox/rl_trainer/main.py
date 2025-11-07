# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m tests.sandbox.rl_trainer.main --config apps/grpo/qwen3_1_7b.yaml

import asyncio

import torch
import torchstore as ts
from forge.actors.trainer import TitanTrainer
from forge.controller.provisioner import init_provisioner, shutdown
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.perf_tracker import Tracer
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from omegaconf import DictConfig


def simple_grpo_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Simplified loss function for memory/CPU profiling purposes.
    Just performs basic tensor operations to simulate memory usage.
    """
    # Extract dimensions
    local_batch_size, response_len = response.shape
    vocab_size = logits.size(-1)
    full_seq_len = logits.size(1)

    # Extract only the response portion from logits
    # logits shape: [local_batch_size, request_len + response_len, vocab_size]
    # We want the last response_len tokens
    request_len = full_seq_len - response_len
    response_logits = logits[
        :, request_len:, :
    ]  # [local_batch_size, response_len, vocab_size]

    # Flatten logits and response for cross-entropy
    logits_flat = response_logits.reshape(-1, vocab_size)
    response_flat = response.reshape(-1)

    # Basic cross-entropy loss (simplified)
    loss = torch.nn.functional.cross_entropy(
        logits_flat, response_flat, reduction="none"
    ).view(local_batch_size, response_len)

    # Apply padding mask and reduce
    masked_loss = loss * padding_mask
    loss = masked_loss.sum() / padding_mask.sum().clamp(min=1.0)

    return loss


def generate_random_batch(
    local_batch_size: int,
    request_len: int,
    response_len: int,
    vocab_size: int = 32000,
    device: str = "cuda",
    dp_size: int = 1,
):
    """
    Generate random input and target tensors matching GRPO data format.
    Creates one batch per data parallel rank.
    """
    inputs = []
    targets = []

    # Create one batch for each data parallel rank
    for _ in range(dp_size):
        request = torch.randint(
            1,
            vocab_size,
            (local_batch_size, request_len),
            dtype=torch.long,
            device=device,
        )
        response = torch.randint(
            1,
            vocab_size,
            (local_batch_size, response_len),
            dtype=torch.long,
            device=device,
        )

        # Create padding mask (randomly mask some tokens as padding)
        padding_mask = torch.rand((local_batch_size, response_len), device=device) > 0.1

        ref_logprobs = (
            -torch.abs(torch.randn((local_batch_size, response_len), device=device))
            - 1.0
        )
        advantages = torch.randn((local_batch_size, 1), device=device)
        input_tokens = torch.cat([request, response], dim=1)
        inputs.append({"tokens": input_tokens})
        targets.append(
            {
                "response": response,
                "ref_logprobs": ref_logprobs,
                "advantages": advantages,
                "padding_mask": padding_mask,
            }
        )

    return inputs, targets


async def main(cfg: DictConfig):
    """
    Trainer simulation app for memory/CPU profiling and system usage analysis.

    This app initializes only the TitanTrainer component and runs a training loop with
    synthetic random data to simulate real trainer system usage patterns. It is
    designed for:

    - Memory profiling of trainer infrastructure
    - CPU usage analysis during training steps
    - System resource monitoring (GPU memory, network, etc.)
    - Performance benchmarking of trainer components
    - Testing trainer stability under load

    The app uses the same configuration format as GRPO but bypasses policy generation,
    replay buffers, and reward computation, focusing purely on the trainer's
    computational and memory characteristics with realistic data shapes.
    """

    # Extract training parameters from existing GRPO config fields
    local_batch_size = cfg.get("local_batch_size", None)
    assert local_batch_size is not None, "local_batch_size must be specified"

    request_len = cfg.get("max_req_tokens", 128)
    response_len = cfg.get("max_res_tokens", 128)
    max_training_steps = cfg.trainer.training.get("steps", 100)

    # Use hardcoded vocab size for testing (avoids network dependency)
    model_name = cfg.get("model")
    print(f"Using model config: {model_name}")
    # Common vocab sizes: Qwen=151936, Llama3=128256
    vocab_size = 151936  # Qwen vocab size
    pad_id = 0
    print(f"Using vocab size: {vocab_size}, pad token ID: {pad_id}")

    # Get data parallel size from config
    dp_size = cfg.trainer.parallelism.get("data_parallel_shard_degree", 1)
    if dp_size == -1:
        dp_size = 1

    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # Initialize trainer FIRST (this creates the mesh)
    print("Initializing trainer...")
    trainer = await TitanTrainer.options(**cfg.actors.trainer).as_actor(
        **cfg.trainer, loss=simple_grpo_loss
    )
    print("Trainer initialized successfully!")

    # NOW initialize torchstore (after trainer mesh is created)
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")
    print("Trainer initialized successfully with following configs!")
    print(f"  - Local batch size: {local_batch_size}")
    print(f"  - Request length: {request_len}")
    print(f"  - Response length: {response_len}")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Data parallel size: {dp_size}")
    print(f"  - Max training steps: {max_training_steps}")

    async def continuous_training():
        training_step = 0

        print("Starting training loop with random data...")
        while training_step < max_training_steps:
            t = Tracer("trainer/continuous_training")
            t.start()

            inputs, targets = generate_random_batch(
                local_batch_size=local_batch_size,
                request_len=request_len,
                response_len=response_len,
                vocab_size=vocab_size,
                dp_size=dp_size,
            )
            t.step("generate_random_data")

            # Perform training step
            await trainer.train_step.call(inputs, targets)
            training_step += 1
            t.step("train_step")

            await trainer.push_weights.call(training_step)
            t.step("push_weights")
            t.stop()

            # Flush metrics
            await mlogger.flush.call_one(training_step)

            print(f"Completed training step {training_step}/{max_training_steps}")

            # Sleep between steps to avoid overwhelming the system
            await asyncio.sleep(1.0)

        print(f"Reached training limit ({max_training_steps} steps). Exiting.")

    try:
        await continuous_training()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Shutting down...")
        await shutdown()
        print("Trainer shutdown complete.")


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
