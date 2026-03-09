"""GRPO fine-tuning of gemma3-4b-it for cooperative behavior in GovSim.

Usage:
    python -m training.train_grpo \
        --dataset_path training/traces/generated_traces.json \
        --output_dir models/gemma3-4b-cooperative \
        --max_steps 1000
"""
import argparse
import json
import logging
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .config import (
    SCENARIO_CONFIGS,
    GameState,
    HistoryPattern,
    Memory,
    Scenario,
)
from .generate_traces import make_date
from .reward_functions import format_reward, game_mechanics_reward

logger = logging.getLogger(__name__)


class MetricsLoggingCallback(TrainerCallback):
    """Log training metrics via Python logging (visible in SLURM logs)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss", None)
        reward = logs.get("reward", None)
        reward_std = logs.get("reward_std", None)
        lr = logs.get("learning_rate", None)
        compl_len = logs.get("mean_completion_length", logs.get("completion_length", None))
        parts = [f"step={step}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if reward is not None:
            parts.append(f"reward={reward:.3f}")
        if reward_std is not None:
            parts.append(f"reward_std={reward_std:.3f}")
        if compl_len is not None:
            parts.append(f"compl_len={compl_len:.0f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        logger.info(f"[METRICS] {' | '.join(parts)}")


def patch_gemma3_for_text_only_training():
    """Patch Gemma 3's causal mask to not require token_type_ids for text-only.

    Gemma 3's create_causal_mask_mapping() raises ValueError when
    model.training=True and token_type_ids is None. TRL's GRPOTrainer
    generates in train mode, so this check triggers even for text-only inputs.
    """
    import transformers.models.gemma3.modeling_gemma3 as gemma3_mod

    original_fn = gemma3_mod.create_causal_mask_mapping

    def patched_fn(
        config, inputs_embeds, attention_mask, cache_position,
        past_key_values, position_ids, token_type_ids, pixel_values,
        is_training=False, is_first_iteration=None,
    ):
        if is_training and token_type_ids is None and pixel_values is None:
            token_type_ids = torch.zeros(
                inputs_embeds.shape[:2], dtype=torch.long,
                device=inputs_embeds.device,
            )
        return original_fn(
            config, inputs_embeds, attention_mask, cache_position,
            past_key_values, position_ids, token_type_ids, pixel_values,
            is_training=is_training, is_first_iteration=is_first_iteration,
        )

    gemma3_mod.create_causal_mask_mapping = patched_fn
    logger.info("Patched Gemma 3 create_causal_mask_mapping for text-only training")


def load_dataset_from_traces(path: str) -> Dataset:
    """Load traces JSON and convert to HuggingFace Dataset for GRPOTrainer."""
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

        from .prompt_builder import build_prompt_messages, build_prompt_metadata
        messages = build_prompt_messages(state)
        metadata = build_prompt_metadata(state)

        # Convert None agreed_limit to -1 for HF Dataset (no mixed int/None)
        if metadata.get("agreed_limit") is None:
            metadata["agreed_limit"] = -1
        rows.append({"prompt": messages, **metadata})

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="models/gemma3-4b-cooperative")
    parser.add_argument("--model_name", default="google/gemma-3-4b-it")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=2048)
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

    # CRITICAL: Gemma 3 ends chat responses with <end_of_turn> (token 106),
    # NOT <eos> (token 1). If we don't set this, generation runs to
    # max_completion_length every time, producing 0 reward.
    if "gemma" in args.model_name.lower():
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if end_of_turn_id is not None and end_of_turn_id != tokenizer.unk_token_id:
            logger.info(
                f"Setting eos_token to <end_of_turn> (id={end_of_turn_id}) "
                f"for proper chat response stopping (was <eos> id={tokenizer.eos_token_id})"
            )
            tokenizer.eos_token = "<end_of_turn>"
            tokenizer.eos_token_id = end_of_turn_id

    # NOTE: Gemma 3 token_type_ids patch no longer needed in transformers >= 4.57
    # The create_causal_mask_mapping function was removed; token_type_ids=None
    # is handled gracefully in newer versions.

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        local_files_only=local_only,
        attn_implementation="eager",
    )

    # Gemma 3 defaults to cache_implementation='hybrid', which causes
    # torch.compile() issues with PEFT's make_inputs_require_grads hook.
    model.generation_config.cache_implementation = None

    # LoRA config — restrict to attention projections only
    lora_target = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Gemma 3 multimodal: restrict to language model to avoid vision tower
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

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset_from_traces(args.dataset_path)
    logger.info(f"Dataset loaded: {len(dataset)} examples")

    # GRPO config — informed by DAPO, Dr. GRPO, and increase-syn reference
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        generation_batch_size=args.num_generations,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # GRPO loss and clipping
        beta=0.001,                         # small KL penalty (0.04 found unstable)
        epsilon=0.2,                        # clip lower bound
        epsilon_high=0.28,                  # DAPO clip-higher (asymmetric)
        loss_type="bnpo",                   # token-level loss, no length bias
        scale_rewards="group",              # normalize rewards within group
        mask_truncated_completions=True,    # DAPO: exclude truncated from loss
        reward_weights=[1.0, 0.25],         # [universalization, format]
        # Generation sampling
        temperature=1.0,                    # >= 1.0 for diverse exploration
        # Training stability
        max_grad_norm=0.1,                  # aggressive clipping
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging and saving
        logging_steps=1,
        save_steps=250,
        save_total_limit=5,
        report_to="none",                  # use MetricsLoggingCallback instead
        seed=args.seed,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[game_mechanics_reward, format_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MetricsLoggingCallback()],
    )

    # Ensure generation config has both <eos> and <end_of_turn> as stop tokens
    if "gemma" in args.model_name.lower():
        eos_id = tokenizer.convert_tokens_to_ids("<eos>")
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        all_eos = list({eos_id, end_of_turn_id} - {None, tokenizer.unk_token_id})
        if all_eos:
            trainer.generation_config.eos_token_id = all_eos
            logger.info(f"Generation config eos_token_id set to: {all_eos}")

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

    # Log trainable parameters
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    logger.info(
        f"Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # Train
    logger.info(
        f"Starting GRPO training: max_steps={args.max_steps}, "
        f"num_generations={args.num_generations}, lr={args.learning_rate}"
    )
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    logger.info(f"Model saved to {args.output_dir}/final_adapter")


if __name__ == "__main__":
    main()
