"""SFT fine-tuning of gemma3-4b-it for cooperative behavior in GovSim.

Supervised fine-tuning on ideal traces (prompt -> ideal_reasoning) to teach
the model the correct output distribution (harvest 5-15) and universalization
reasoning pattern. This serves as a warm-start before GRPO.

Supports mixed datasets: GovSim traces + general cooperative traces for
improved generalization. Use --general_traces_path and --mix_ratio to control.

Usage:
    python -m training.train_sft \
        --dataset_path training/traces/generated_traces_1k.json \
        --output_dir models/gemma3-4b-sft-v1

    # With mixed dataset (30% general, 70% GovSim):
    accelerate launch --config_file training/accelerate_3gpu.yaml \
        -m training.train_sft \
        --dataset_path training/traces/generated_traces_1k.json \
        --general_traces_path training/traces/general_traces.json \
        --mix_ratio 0.3 \
        --output_dir models/gemma3-4b-sft-v3
"""
import argparse
import json
import logging
import os
import random

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from .config import (
    SCENARIO_CONFIGS,
    GameState,
    HistoryPattern,
    Memory,
    Scenario,
)
from .generate_traces import make_date

logger = logging.getLogger(__name__)


class MetricsLoggingCallback(TrainerCallback):
    """Log training metrics via Python logging (visible in SLURM logs)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss", None)
        lr = logs.get("learning_rate", None)
        epoch = logs.get("epoch", None)
        parts = [f"step={step}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        if epoch is not None:
            parts.append(f"epoch={epoch:.2f}")
        logger.info(f"[METRICS] {' | '.join(parts)}")


def load_dataset_from_traces(path: str, tokenizer) -> Dataset:
    """Load traces JSON and convert to chat-format dataset for SFTTrainer.

    Each row is a full conversation: system + user + assistant (ideal_reasoning).
    SFTTrainer will only compute loss on the assistant tokens.
    """
    with open(path) as f:
        traces = json.load(f)

    rows = []
    for t in traces:
        state = GameState(
            scenario=Scenario(t["scenario"]),
            agent_name=t["agent_name"],
            other_agent_names=t.get("other_agent_names", ["Kate", "Jack", "Emma", "Luke"]),
            resource_in_pool=t["resource_in_pool"],
            carrying_capacity=t["carrying_capacity"],
            num_agents=t["num_agents"],
            current_round=t["current_round"],
            date=t.get("date", make_date(t["current_round"])),
            history_pattern=HistoryPattern(t["history_pattern"]),
            memories=[Memory(**m) for m in t["memories"]],
            inject_universalization=t.get("inject_universalization", True),
            agreed_limit=t.get("agreed_limit"),
        )

        from .prompt_builder import build_prompt_messages
        messages = build_prompt_messages(state)

        # Add the ideal response as the assistant turn
        messages.append({
            "role": "assistant",
            "content": t["ideal_reasoning"],
        })

        rows.append({"messages": messages})

    return Dataset.from_list(rows)


def load_general_traces(path: str) -> Dataset:
    """Load general cooperative traces and convert to chat-format dataset.

    General traces have format:
        {"prompt": "...", "ideal_response": "...", "persona": "...", "scenario_type": "..."}

    These are converted to simple user/assistant chat pairs.
    """
    with open(path) as f:
        traces = json.load(f)

    rows = []
    for t in traces:
        messages = [
            {"role": "user", "content": t["prompt"]},
            {"role": "assistant", "content": t["ideal_response"]},
        ]
        rows.append({"messages": messages})

    return Dataset.from_list(rows)


def mix_datasets(
    govsim_ds: Dataset,
    general_ds: Dataset,
    mix_ratio: float,
    seed: int = 42,
) -> Dataset:
    """Interleave GovSim and general datasets at the specified ratio.

    Args:
        govsim_ds: GovSim traces dataset.
        general_ds: General cooperative traces dataset.
        mix_ratio: Fraction of final dataset that should be general traces
                   (e.g. 0.3 = 30% general, 70% GovSim).
        seed: Random seed for shuffling.

    Returns:
        Combined and shuffled dataset.
    """
    n_govsim = len(govsim_ds)
    # Compute how many general samples to include based on the ratio:
    # mix_ratio = n_general / (n_govsim + n_general)
    # => n_general = n_govsim * mix_ratio / (1 - mix_ratio)
    n_general = int(n_govsim * mix_ratio / (1.0 - mix_ratio))
    n_general = min(n_general, len(general_ds))

    logger.info(
        f"Mixing datasets: {n_govsim} GovSim + {n_general} general "
        f"(target ratio={mix_ratio:.0%} general)"
    )

    # Sample from general dataset if we need fewer than available
    if n_general < len(general_ds):
        rng = random.Random(seed)
        indices = rng.sample(range(len(general_ds)), n_general)
        general_subset = general_ds.select(indices)
    else:
        general_subset = general_ds

    # Concatenate and shuffle
    combined = concatenate_datasets([govsim_ds, general_subset])
    combined = combined.shuffle(seed=seed)

    actual_ratio = n_general / len(combined)
    logger.info(
        f"Combined dataset: {len(combined)} examples "
        f"(actual general ratio={actual_ratio:.1%})"
    )

    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--general_traces_path", default=None,
        help="Path to general cooperative traces JSON for mixed training",
    )
    parser.add_argument(
        "--mix_ratio", type=float, default=0.3,
        help="Fraction of general traces in the mix (default: 0.3 = 30%% general, 70%% GovSim)",
    )
    parser.add_argument("--output_dir", default="models/gemma3-4b-sft-v1")
    parser.add_argument("--model_name", default="google/gemma-3-4b-it")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    logger.info(f"Loading tokenizer: {args.model_name} (local_only={local_only})")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=local_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Gemma 3: use <end_of_turn> as EOS for proper chat format
    if "gemma" in args.model_name.lower():
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if end_of_turn_id is not None and end_of_turn_id != tokenizer.unk_token_id:
            logger.info(
                f"Setting eos_token to <end_of_turn> (id={end_of_turn_id}) "
                f"for proper chat response stopping"
            )
            tokenizer.eos_token = "<end_of_turn>"
            tokenizer.eos_token_id = end_of_turn_id

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        local_files_only=local_only,
        attn_implementation="eager",
    )

    # Gemma 3: disable hybrid cache to avoid torch.compile + PEFT conflict
    model.generation_config.cache_implementation = None

    # LoRA config — same as GRPO for compatibility
    lora_target = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Gemma 3 multimodal: restrict to language model to skip vision tower
    if hasattr(getattr(model, "model", None), "language_model"):
        pattern = "|".join(lora_target)
        lora_target = f"model\\.language_model.*({pattern})"
        logger.info(f"Multimodal model detected, LoRA target regex: {lora_target}")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=lora_target,
        task_type="CAUSAL_LM",
    )

    # Load dataset(s)
    logger.info(f"Loading GovSim dataset: {args.dataset_path}")
    govsim_dataset = load_dataset_from_traces(args.dataset_path, tokenizer)
    logger.info(f"GovSim dataset loaded: {len(govsim_dataset)} examples")

    if args.general_traces_path is not None:
        logger.info(f"Loading general traces: {args.general_traces_path}")
        general_dataset = load_general_traces(args.general_traces_path)
        logger.info(f"General dataset loaded: {len(general_dataset)} examples")

        dataset = mix_datasets(
            govsim_dataset, general_dataset,
            mix_ratio=args.mix_ratio,
            seed=args.seed,
        )
    else:
        dataset = govsim_dataset
        logger.info("No general traces provided, using GovSim-only dataset")

    # SFT config
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_length=args.max_length,
        # Training settings
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        # Logging and saving
        logging_steps=10,
        save_steps=250,
        save_total_limit=5,
        report_to="none",
        seed=args.seed,
    )

    # Create trainer
    logger.info("Creating SFTConfig... done")
    import sys; sys.stderr.flush(); sys.stdout.flush()
    logger.info("Creating SFTTrainer...")
    sys.stderr.flush()
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MetricsLoggingCallback()],
    )
    logger.info("SFTTrainer created successfully")

    # Log trainable parameters
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    logger.info(
        f"Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # Check for existing checkpoints to resume from
    resume_from = None
    if os.path.isdir(args.output_dir):
        checkpoints = [
            d for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))
        ]
        if checkpoints:
            resume_from = True
            latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            logger.info(f"Resuming from checkpoint: {latest}")

    # Train
    logger.info(
        f"Starting SFT training: epochs={args.num_train_epochs}, "
        f"lr={args.learning_rate}, batch={args.batch_size}x{args.grad_accum}"
    )
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    logger.info(f"Model saved to {args.output_dir}/final_adapter")


if __name__ == "__main__":
    main()
