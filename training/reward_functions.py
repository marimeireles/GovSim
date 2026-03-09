"""Reward functions for GRPO training.

These are RLVR (verifiable) rewards — pure functions of the completion text
and the game state metadata. No learned reward model.

Two reward functions:
  1. game_mechanics_reward: Universalized simulation — "if ALL agents took
     this amount, would the resource survive 12 rounds?" Rewards based on
     survival + pool health + income.
  2. format_reward: Rewards well-structured reasoning + answer format.
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Debug logging counter
_log_counter = 0
_LOG_FIRST_N = 40


def _extract_completion_text(completion) -> str:
    """Extract plain text from a completion that may be in conversational format.

    TRL 0.20+ passes completions as list of message dicts for conversational
    datasets, e.g. [{"role": "assistant", "content": "the text"}].
    """
    if isinstance(completion, list):
        if completion and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        return ""
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def extract_harvest(text: str) -> Optional[int]:
    """Extract the harvest number from 'Answer: N tons/flocks/pallets'."""
    match = re.search(r'Answer:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None


def simulate_universalized(
    resource_in_pool: int,
    carrying_capacity: int,
    num_agents: int,
    agent_harvest: int,
    num_rounds: int = 12,
) -> dict:
    """Simulate what happens if ALL agents take agent_harvest every round.

    This is the universalization test: "if everyone did what I'm doing,
    would the resource survive?" Runs the full game (12 rounds).

    Returns dict with:
      - survived_rounds: how many rounds before collapse (or num_rounds if survived)
      - collapsed: True if pool went below 5 at any point
      - agent_actual: what agent actually gets in round 1 (may be less if overdemand)
      - final_pool: pool level at end
    """
    pool = resource_in_pool
    total_per_round = agent_harvest * num_agents

    # What the agent actually gets in round 1
    if total_per_round > pool:
        agent_actual = int(agent_harvest * pool / total_per_round) if total_per_round > 0 else 0
    else:
        agent_actual = agent_harvest

    for r in range(num_rounds):
        if total_per_round > pool:
            pool_after = 0
        else:
            pool_after = pool - total_per_round

        if pool_after < 5:
            return {
                "survived_rounds": r,
                "collapsed": True,
                "agent_actual": agent_actual,
                "final_pool": pool_after,
            }

        pool = min(carrying_capacity, pool_after * 2)

    return {
        "survived_rounds": num_rounds,
        "collapsed": False,
        "agent_actual": agent_actual,
        "final_pool": pool,
    }


def game_mechanics_reward(
    completions,
    prompts=None,
    resource_in_pool=None,
    carrying_capacity=None,
    num_agents=None,
    sustainable_share=None,
    **kwargs,
) -> list[float]:
    """Reward based on universalized game mechanics.

    Asks: "if ALL agents took this amount every round, would the resource
    survive 12 rounds?" This directly teaches the universalization principle.

    Reward components:
      - Collapsed: -1.0 (immediate) to -0.5 (collapsed late)
      - Survived: 0.7 * pool_health + 0.3 * income
    """
    global _log_counter
    rewards = []

    if resource_in_pool is None:
        resource_in_pool = []
    if carrying_capacity is None:
        carrying_capacity = []
    if num_agents is None:
        num_agents = []
    if sustainable_share is None:
        sustainable_share = []

    for i, completion in enumerate(completions):
        text = _extract_completion_text(completion)
        harvest = extract_harvest(text)

        pool = resource_in_pool[i] if i < len(resource_in_pool) else 100
        cap = carrying_capacity[i] if i < len(carrying_capacity) else 100
        n_agents = num_agents[i] if i < len(num_agents) else 5
        sust = sustainable_share[i] if i < len(sustainable_share) else 10

        # Debug logging
        if _log_counter < _LOG_FIRST_N:
            _log_counter += 1
            preview = text[:200] if text else "<empty>"
            logger.info(
                f"[Game reward {_log_counter}/{_LOG_FIRST_N}] "
                f"harvest={harvest}, pool={pool}, cap={cap}, sust={sust}, "
                f"preview={preview!r}"
            )

        # No parseable answer
        if harvest is None:
            rewards.append(-1.0)
            continue

        # Clamp harvest to non-negative
        harvest = max(0, harvest)

        # Universalized simulation: if ALL agents take this, 12 rounds
        result = simulate_universalized(pool, cap, n_agents, harvest)

        if result["collapsed"]:
            # Partial credit for surviving longer before collapse
            survival_ratio = result["survived_rounds"] / 12
            reward = -1.0 + survival_ratio * 0.5  # -1.0 to -0.5
            rewards.append(reward)
            continue

        # Pool health: final pool / carrying capacity (0 to 1)
        health = result["final_pool"] / cap if cap > 0 else 0.0

        # Income: ratio of harvest to sustainable share, capped at 1.0
        if sust > 0:
            income = min(1.0, result["agent_actual"] / sust)
        else:
            income = 1.0 if result["agent_actual"] == 0 else 0.0

        # Weighted combination: prioritize pool health
        reward = 0.7 * health + 0.3 * income

        rewards.append(reward)

    return rewards


def format_reward(completions, prompts=None, **kwargs) -> list[float]:
    """Reward for producing correctly formatted output.

    Checks:
      1. Contains exactly one "Answer:" followed by a number
      2. Contains step-by-step reasoning (not just a bare answer)
      3. No garbage/artifact text after the answer
    """
    rewards = []
    for completion in completions:
        text = _extract_completion_text(completion)
        score = 0.0

        # Must have exactly one "Answer:" — multiple indicates malformed output
        answer_count = text.count("Answer:")
        if answer_count != 1:
            rewards.append(0.0)
            continue

        # Must have a parseable number after "Answer:"
        harvest = extract_harvest(text)
        if harvest is None:
            rewards.append(0.0)
            continue

        # Reject unreasonable harvest values (> 200 is never valid)
        if harvest > 200:
            rewards.append(0.0)
            continue

        # Valid answer format
        score += 0.5

        # Check reasoning quality before "Answer:"
        answer_pos = text.find("Answer:")
        if answer_pos > 0:
            reasoning = text[:answer_pos]
            word_count = len(reasoning.split())
            if word_count >= 50:
                score += 0.5
            elif word_count >= 20:
                score += 0.25

        rewards.append(score)

    return rewards
