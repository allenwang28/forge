# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:

python -m apps.sft.main --config apps/sft/llama3_8b.yaml

"""

import asyncio
import logging
import os
import sys
from functools import partial

import torch

from forge.actors.trainer import TitanTrainer
from forge.data.collate import collate_packed
from forge.data.datasets.packed import PackedDataset, TextPacker
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
from forge.observability import get_or_create_metric_logger, record_metric, Reduce
from forge.util.config import parse

from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cross_entropy_loss(logits: torch.Tensor, **targets) -> torch.Tensor:
    """Cross-entropy loss for SFT."""
    labels = targets["labels"]

    # Flatten for cross-entropy
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute loss
    loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat)
    return loss


def create_dataloader(cfg: DictConfig):
    """Create SFT dataloader.

    Note: Dataloader keeps data on CPU. Trainer will move to device.
    """
    print(os.path.join(cfg.trainer.model.hf_assets_path, "tokenizer.json"))
    tokenizer = HuggingFaceModelTokenizer(
        tokenizer_json_path=os.path.join(
            cfg.trainer.model.hf_assets_path, "tokenizer.json"
        ),
        tokenizer_config_json_path=os.path.join(
            cfg.trainer.model.hf_assets_path, "tokenizer_config.json"
        ),
        generation_config_path=os.path.join(
            cfg.trainer.model.hf_assets_path, "generation_config.json"
        ),
        chat_template_path=(
            path
            if os.path.exists(
                path := os.path.join(
                    cfg.trainer.model.hf_assets_path, "chat_template.jinja"
                )
            )
            else None
        ),
    )

    dataset = sft_iterable_dataset(
        model_transform=tokenizer,
        message_transform=AlpacaToMessages(),
        path="yahma/alpaca-cleaned",
        split="train",
    )
    packer = TextPacker(padding_idx=0)
    dataset = PackedDataset(
        dataset=dataset,
        packer=packer,
        target_tokens_per_pack=cfg.trainer.training.seq_len,
    )
    # Keep data on CPU - trainer will move to device
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=cfg.trainer.training.local_batch_size,
        collate_fn=partial(
            collate_packed, mask_fn=packer.create_block_mask, device="cpu"
        ),
    )
    return dataloader


async def main(cfg: DictConfig):
    """SFT training loop using Trainer protocol."""
    logger.info("Starting SFT training...")

    # Initialize metric logger in main process
    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # Create trainer actor (as_actor automatically calls setup)
    logger.info("Creating trainer...")
    trainer = await TitanTrainer.options(**cfg.processes.trainer).as_actor(
        **cfg.trainer, loss=cross_entropy_loss
    )

    # Get trainer info to understand DP setup
    trainer_info = await trainer.get_info.choose()
    dp_degree = trainer_info["parallelism"]["dp_degree"]
    local_batch_size = trainer_info["parallelism"]["local_batch_size"]
    global_batch_size = dp_degree * local_batch_size

    logger.info(f"Trainer info: {trainer_info['model_name']}")
    logger.info(
        f"DP degree: {dp_degree}, local batch: {local_batch_size}, global batch: {global_batch_size}"
    )

    # Create dataloader with global batch size
    # Update config to use global batch size
    cfg.trainer.training.local_batch_size = global_batch_size
    dataloader = create_dataloader(cfg)

    # Training loop
    num_training_steps = cfg.trainer.training.steps
    current_step = 0

    logger.info(f"Starting training for {num_training_steps} steps...")
    for batch in dataloader:
        if current_step >= num_training_steps:
            break

        # Pop and record dataset metrics
        if "metrics" in batch:
            for metric in batch.pop("metrics"):
                record_metric(metric.key, metric.value, metric.reduction)

        # Prepare global batch (trainer will slice per rank)
        labels = batch.pop("labels")

        # Remove non-model fields (metadata from packed dataset)
        batch.pop("document_ids", None)

        # Now batch only contains model inputs (input_ids, attention_mask, position_ids, etc.)
        # Protocol: forward_backward with global batch
        loss = await trainer.forward_backward.call(batch, {"labels": labels})

        # Protocol: optim_step
        step_info = await trainer.optim_step.call()
        current_step = step_info["step"]

        # Log metrics
        record_metric("sft/train_step/loss", loss, Reduce.MEAN)
        record_metric("sft/train_step/step", current_step, Reduce.MEAN)
        logger.info(f"{current_step} / {num_training_steps} | Loss: {loss:.4f}")

        # Flush metrics
        await mlogger.flush.call_one(global_step=current_step)

    logger.info("Training complete!")
    await trainer.cleanup.call()
    await mlogger.shutdown.call_one()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(main(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())
