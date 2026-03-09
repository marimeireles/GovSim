"""Trace generation for GRPO training dataset.

Three-stage pipeline:
  1. generate_game_states() — programmatic states covering the diversity space
  2. build_memories_for_state() — template-based memory histories
  3. generate_ideal_reasoning() — programmatic ideal reasoning with universalization

Usage:
  python -m training.generate_traces --output training/traces/generated_traces.json --count 2000
"""
import argparse
import itertools
import json
import random
from datetime import datetime, timedelta

from .config import (
    AGENT_NAME_SETS,
    SCENARIO_CONFIGS,
    GameState,
    HistoryPattern,
    Memory,
    Scenario,
    ScenarioConfig,
    Trace,
)
from .prompt_builder import build_prompt_messages


# ─────────────────────────────────────────────────────
# SECTION 1: Memory Construction Templates
# ─────────────────────────────────────────────────────


def make_date(round_num: int, day: int = 1) -> str:
    """Generate a date string for a given round. Round 0 = 2024-01-01."""
    base = datetime(2024, 1, 1)
    dt = base + timedelta(days=30 * max(0, round_num - 1) + (day - 1))
    return dt.strftime("%Y-%m-%d")


def build_memories_for_state(state: GameState) -> list[Memory]:
    """Build a plausible memory history for the given game state."""
    cfg = SCENARIO_CONFIGS[state.scenario]
    memories = []

    if state.history_pattern == HistoryPattern.EARLY_GAME:
        memories = _build_early_game_memories(state, cfg)
    elif state.history_pattern == HistoryPattern.ALL_COOPERATIVE:
        memories = _build_cooperative_memories(state, cfg)
    elif state.history_pattern == HistoryPattern.SINGLE_DEFECTOR:
        memories = _build_single_defector_memories(state, cfg)
    elif state.history_pattern == HistoryPattern.MULTIPLE_DEFECTORS:
        memories = _build_multiple_defectors_memories(state, cfg)
    elif state.history_pattern == HistoryPattern.POST_CRISIS:
        memories = _build_post_crisis_memories(state, cfg)
    elif state.history_pattern == HistoryPattern.AGREEMENT_BROKEN:
        memories = _build_agreement_broken_memories(state, cfg)
    elif state.history_pattern == HistoryPattern.NO_AGREEMENT:
        memories = _build_no_agreement_memories(state, cfg)
    elif state.history_pattern == HistoryPattern.RECOVERY:
        memories = _build_recovery_memories(state, cfg)
    elif state.history_pattern == HistoryPattern.ESCALATION:
        memories = _build_escalation_memories(state, cfg)

    # Always add current round's pool observation
    memories.append(Memory(
        date=state.date,
        text=cfg.pool_observation_template.format(resource=state.resource_in_pool),
    ))

    # Add universalization hint if configured
    if state.inject_universalization:
        memories.append(Memory(
            date=state.date,
            text=cfg.universalization_template.format(threshold=state.sustainable_share),
        ))

    return memories


def _add_harvest_memory(memories, cfg, date, agent, caught):
    """Helper to add a harvest memory entry."""
    memories.append(Memory(
        date=date,
        text=cfg.after_harvesting_template.format(
            name=agent, wanted=caught, caught=caught
        ),
    ))


def _build_early_game_memories(state: GameState, cfg: ScenarioConfig) -> list[Memory]:
    """Round 1-2, minimal or no history."""
    memories = []
    if state.current_round >= 2:
        resource = state.carrying_capacity
        threshold = (resource // 2) // state.num_agents
        memories.append(Memory(
            date=make_date(1, day=1),
            text=cfg.pool_observation_template.format(resource=resource),
        ))
        all_agents = [state.agent_name] + state.other_agent_names
        for agent in all_agents:
            caught = random.randint(max(1, threshold - 3), threshold + 5)
            _add_harvest_memory(memories, cfg, make_date(1, day=15), agent, caught)
    return memories


def _build_cooperative_memories(state: GameState, cfg: ScenarioConfig) -> list[Memory]:
    """All previous rounds were cooperative. Resource stayed high."""
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    resource = state.carrying_capacity

    for r in range(1, state.current_round):
        threshold = (resource // 2) // state.num_agents

        memories.append(Memory(
            date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource),
        ))

        total_caught = 0
        for agent in all_agents:
            caught = random.randint(max(1, threshold - 2), threshold)
            total_caught += caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        memories.append(Memory(
            date=make_date(r, day=28),
            text="The group discussed sustainable practices and reaffirmed their commitment to careful harvesting.",
        ))

        if r == 1 and state.agreed_limit is not None:
            memories.append(Memory(
                date=make_date(r, day=29),
                text=cfg.agreed_limit_template.format(limit=state.agreed_limit),
            ))

        resource = min(state.carrying_capacity, (resource - total_caught) * 2)

    return memories


def _build_single_defector_memories(state: GameState, cfg: ScenarioConfig) -> list[Memory]:
    """One agent consistently over-harvests while others cooperate."""
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    defector = state.other_agent_names[0]
    resource = state.carrying_capacity

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(
            date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource),
        ))

        total_caught = 0
        for agent in all_agents:
            if agent == defector:
                caught = min(resource, threshold * 2 + random.randint(0, 5))
            else:
                caught = random.randint(max(1, threshold - 2), threshold)
            total_caught += caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        memories.append(Memory(
            date=make_date(r, day=28),
            text=f"The group expressed concern about {defector}'s high catches and urged restraint.",
        ))

        if r == 1 and state.agreed_limit is not None:
            memories.append(Memory(
                date=make_date(r, day=29),
                text=cfg.agreed_limit_template.format(limit=state.agreed_limit),
            ))

        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))

    return memories


def _build_multiple_defectors_memories(state: GameState, cfg: ScenarioConfig) -> list[Memory]:
    """2-3 agents over-harvest while others cooperate."""
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    num_defectors = min(len(state.other_agent_names), random.choice([2, 3]))
    defectors = state.other_agent_names[:num_defectors]
    resource = state.carrying_capacity

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(
            date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource),
        ))

        total_caught = 0
        for agent in all_agents:
            if agent in defectors:
                caught = min(resource // state.num_agents, threshold + random.randint(3, 10))
            else:
                caught = random.randint(max(1, threshold - 2), max(1, threshold))
            total_caught = min(total_caught + caught, resource)
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        defector_names = " and ".join(defectors)
        memories.append(Memory(
            date=make_date(r, day=28),
            text=f"The group argued about overuse but {defector_names} refused to reduce their harvesting.",
        ))

        if r == 1 and state.agreed_limit is not None:
            memories.append(Memory(
                date=make_date(r, day=29),
                text=cfg.agreed_limit_template.format(limit=state.agreed_limit),
            ))

        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))

    return memories


def _build_post_crisis_memories(state: GameState, cfg: ScenarioConfig) -> list[Memory]:
    """Resource dropped dangerously low in earlier rounds, agents now cautious."""
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    resource = state.carrying_capacity
    crisis_round = max(2, state.current_round // 2)

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(
            date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource),
        ))

        total_caught = 0
        for agent in all_agents:
            if r < crisis_round:
                caught = min(resource // state.num_agents, threshold + random.randint(5, 10))
            else:
                caught = max(1, min(threshold, random.randint(1, max(2, threshold))))
            total_caught += caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        if r == crisis_round:
            memories.append(Memory(
                date=make_date(r, day=28),
                text="The group had an urgent discussion about the declining resource and agreed to drastically reduce harvesting.",
            ))
            if state.agreed_limit is not None:
                memories.append(Memory(
                    date=make_date(r, day=29),
                    text=cfg.agreed_limit_template.format(limit=state.agreed_limit),
                ))

        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))

    return memories


def _build_agreement_broken_memories(state: GameState, cfg: ScenarioConfig) -> list[Memory]:
    """Agreement existed and was honored, then someone violated it."""
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    breaker = state.other_agent_names[0]
    resource = state.carrying_capacity
    break_round = max(2, state.current_round - 1)
    limit = state.agreed_limit if state.agreed_limit else (state.carrying_capacity // 2) // state.num_agents

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(
            date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource),
        ))

        total_caught = 0
        for agent in all_agents:
            if agent == breaker and r >= break_round:
                caught = min(resource, limit * 3 + random.randint(0, 5))
            else:
                caught = random.randint(max(1, min(limit, threshold) - 2), min(limit, threshold))
            total_caught += caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        if r < break_round:
            memories.append(Memory(
                date=make_date(r, day=28),
                text="The group discussed sustainable practices and reaffirmed their commitment to careful harvesting.",
            ))
        else:
            memories.append(Memory(
                date=make_date(r, day=28),
                text=f"The group was upset with {breaker} for breaking the agreement. {breaker} apologized and promised to respect the limit going forward.",
            ))

        if r == 1:
            memories.append(Memory(
                date=make_date(r, day=29),
                text=cfg.agreed_limit_template.format(limit=limit),
            ))

        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))

    return memories


def _build_no_agreement_memories(state: GameState, cfg: ScenarioConfig) -> list[Memory]:
    """Conversation happened but no concrete limit was agreed upon."""
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    resource = state.carrying_capacity

    no_agreement_messages = [
        "The group talked casually about the resource but did not establish any specific limits.",
        "The group discussed harvesting patterns but again did not commit to any concrete limit.",
        "Despite a lengthy discussion, the group could not agree on a shared limit.",
        "The group debated sustainable practices but no formal agreement was reached.",
    ]

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(
            date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource),
        ))

        total_caught = 0
        for agent in all_agents:
            # Slightly above threshold on average — no agreement means loose behavior
            caught = random.randint(max(1, threshold - 1), threshold + random.randint(2, 5))
            total_caught += caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        msg_idx = min(r - 1, len(no_agreement_messages) - 1)
        memories.append(Memory(
            date=make_date(r, day=28),
            text=no_agreement_messages[msg_idx],
        ))

        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))

    return memories


def _build_recovery_memories(state: GameState, cfg: ScenarioConfig) -> list[Memory]:
    """Resource climbing back from a low point thanks to collective restraint."""
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    # Start with a very low resource, showing recovery
    resource = max(5, state.carrying_capacity // 10)

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(
            date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource),
        ))

        # Everyone takes very little to allow recovery
        total_caught = 0
        for agent in all_agents:
            caught = max(1, min(threshold, random.randint(1, max(2, threshold - 1))))
            total_caught += caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        if r == 1:
            memories.append(Memory(
                date=make_date(r, day=28),
                text="The group committed to minimal harvesting to allow the resource to recover.",
            ))
            if state.agreed_limit is not None:
                memories.append(Memory(
                    date=make_date(r, day=29),
                    text=cfg.agreed_limit_template.format(limit=state.agreed_limit),
                ))
        elif r == state.current_round - 1:
            memories.append(Memory(
                date=make_date(r, day=28),
                text="The group celebrated the slow recovery and discussed when they could safely increase harvesting.",
            ))
        else:
            memories.append(Memory(
                date=make_date(r, day=28),
                text="The group reaffirmed their commitment to restraint during the recovery period.",
            ))

        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))

    return memories


def _build_escalation_memories(state: GameState, cfg: ScenarioConfig) -> list[Memory]:
    """Catches increasing each round — a tragedy-of-the-commons spiral."""
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    resource = state.carrying_capacity
    collapse_happened = False

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(
            date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource),
        ))

        if resource == 0:
            # Zero resource round — everyone catches nothing
            for agent in all_agents:
                _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, 0)
            collapse_happened = True
        else:
            total_caught = 0
            escalation_factor = min(r + 1, 5)  # increasing greed
            for agent in all_agents:
                caught = min(
                    resource // state.num_agents,
                    threshold + random.randint(escalation_factor, escalation_factor + 5)
                )
                total_caught += caught
                _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

            resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))

        if not collapse_happened:
            memories.append(Memory(
                date=make_date(r, day=28),
                text="The group blamed each other for the declining resource but could not agree on limits.",
            ))
        # After collapse, skip conversation for empty rounds

    return memories


# ─────────────────────────────────────────────────────
# SECTION 2: Systematic State Generation
# ─────────────────────────────────────────────────────

# Distribution weights from the plan
PATTERN_WEIGHTS = {
    HistoryPattern.ALL_COOPERATIVE: 0.20,
    HistoryPattern.SINGLE_DEFECTOR: 0.15,
    HistoryPattern.MULTIPLE_DEFECTORS: 0.10,
    HistoryPattern.POST_CRISIS: 0.10,
    HistoryPattern.AGREEMENT_BROKEN: 0.10,
    HistoryPattern.NO_AGREEMENT: 0.10,
    HistoryPattern.RECOVERY: 0.10,
    HistoryPattern.EARLY_GAME: 0.10,
    HistoryPattern.ESCALATION: 0.05,
}

SCENARIO_WEIGHTS = {
    Scenario.FISHING: 0.35,
    Scenario.SHEEP: 0.33,
    Scenario.POLLUTION: 0.32,
}

RESOURCE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
RESOURCE_FRAC_WEIGHTS = [0.15, 0.20, 0.20, 0.20, 0.25]

NUM_AGENTS_OPTIONS = [3, 4, 5, 6]
NUM_AGENTS_WEIGHTS = [0.10, 0.15, 0.60, 0.15]

CAPACITY_OPTIONS = [50, 100, 150, 200]
CAPACITY_WEIGHTS = [0.10, 0.60, 0.15, 0.15]

ROUND_BUCKETS = {
    "early": [0, 1, 2],
    "mid": [3, 4, 5, 6, 7, 8],
    "late": [9, 10, 11, 12],
}
ROUND_BUCKET_WEIGHTS = [0.15, 0.55, 0.30]


def _weighted_choice(options, weights):
    """Choose from options using weights."""
    return random.choices(options, weights=weights, k=1)[0]


def generate_game_states(target_count: int = 2000, seed: int = 42) -> list[GameState]:
    """Generate diverse game states covering the full parameter space."""
    random.seed(seed)
    states = []

    for i in range(target_count):
        # Sample each axis from its weighted distribution
        scenario = _weighted_choice(list(SCENARIO_WEIGHTS.keys()), list(SCENARIO_WEIGHTS.values()))
        pattern = _weighted_choice(list(PATTERN_WEIGHTS.keys()), list(PATTERN_WEIGHTS.values()))
        capacity = _weighted_choice(CAPACITY_OPTIONS, CAPACITY_WEIGHTS)
        frac = _weighted_choice(RESOURCE_FRACTIONS, RESOURCE_FRAC_WEIGHTS)
        resource = int(capacity * frac)
        if resource < 5:
            resource = 5

        num_agents = _weighted_choice(NUM_AGENTS_OPTIONS, NUM_AGENTS_WEIGHTS)
        name_set_idx = random.randint(0, len(AGENT_NAME_SETS) - 1)
        name_set = AGENT_NAME_SETS[name_set_idx]
        agent_name = name_set[0]
        other_names = name_set[1:num_agents]

        # Pick round based on pattern constraints
        if pattern == HistoryPattern.EARLY_GAME:
            round_num = random.choice(ROUND_BUCKETS["early"])
        elif pattern in (HistoryPattern.POST_CRISIS, HistoryPattern.RECOVERY):
            round_num = _weighted_choice(
                ROUND_BUCKETS["mid"] + ROUND_BUCKETS["late"],
                [0.55 / 6] * 6 + [0.30 / 4] * 4
            )
        elif pattern == HistoryPattern.ESCALATION:
            round_num = _weighted_choice(
                ROUND_BUCKETS["mid"] + ROUND_BUCKETS["late"],
                [0.55 / 6] * 6 + [0.30 / 4] * 4
            )
        else:
            bucket = _weighted_choice(["early", "mid", "late"], ROUND_BUCKET_WEIGHTS)
            round_num = random.choice(ROUND_BUCKETS[bucket])

        inject_univ = random.random() < 0.60

        # Agreed limit logic
        threshold = (resource // 2) // num_agents
        if pattern in (HistoryPattern.EARLY_GAME, HistoryPattern.NO_AGREEMENT, HistoryPattern.ESCALATION):
            agreed_limit = None
        elif pattern == HistoryPattern.AGREEMENT_BROKEN:
            # Agreement must exist for this pattern
            agreed_limit = max(1, threshold)
        elif random.random() < 0.50:
            agreed_limit = max(1, threshold)
        else:
            agreed_limit = None

        state = GameState(
            scenario=scenario,
            agent_name=agent_name,
            other_agent_names=other_names,
            resource_in_pool=resource,
            carrying_capacity=capacity,
            num_agents=num_agents,
            current_round=round_num,
            date=make_date(round_num),
            history_pattern=pattern,
            memories=[],
            inject_universalization=inject_univ,
            agreed_limit=agreed_limit,
        )

        state.memories = build_memories_for_state(state)
        states.append(state)

    return states


# ─────────────────────────────────────────────────────
# SECTION 3: Ideal Reasoning Generation
# ─────────────────────────────────────────────────────

# Sentence starters for variety
_SITUATION_OPENERS = [
    "Let me think about what to do this month.",
    "Let me think about this carefully.",
    "Let me assess the situation.",
    "Let me think through this.",
    "Let me think very carefully about this.",
    "Let me carefully consider the current situation.",
    "I need to think through this decision carefully.",
    "Let me reason through this step by step.",
]

_RESOURCE_DESCRIPTIONS = {
    (0.0, 0.25): ["critically low", "dangerously depleted", "in serious trouble", "at a crisis point"],
    (0.25, 0.50): ["quite low", "below half capacity", "noticeably depleted", "worryingly low"],
    (0.50, 0.75): ["at a moderate level", "reasonably healthy", "below full capacity but manageable"],
    (0.75, 1.01): ["healthy", "near full capacity", "at a strong level", "in excellent condition"],
}


def _describe_resource_level(resource, capacity):
    frac = resource / capacity if capacity > 0 else 0
    for (lo, hi), descriptions in _RESOURCE_DESCRIPTIONS.items():
        if lo <= frac < hi:
            return random.choice(descriptions)
    return "at full capacity"


def _resource_noun(scenario):
    return {
        Scenario.FISHING: "lake",
        Scenario.SHEEP: "pasture",
        Scenario.POLLUTION: "river",
    }[scenario]


def _resource_quantity(resource, scenario):
    if scenario == Scenario.FISHING:
        return f"{resource} tons of fish"
    elif scenario == Scenario.SHEEP:
        return f"{resource} hectares of grass"
    else:
        return f"{resource}% unpolluted"


def _role_plural(cfg):
    if cfg.role == "fisherman":
        return "fishermen"
    elif cfg.role == "shepherd":
        return "shepherds"
    else:
        return "factory owners"


def _verb_s(cfg):
    """Third-person singular of the action verb for universalization."""
    if cfg.action_verb == "catch":
        return "catches"
    elif cfg.action_verb == "take to the pasture":
        return "takes"
    elif cfg.action_verb == "produce":
        return "produces"
    return cfg.action_verb + "s"


def _verb_present(cfg):
    """Base form for 'I will ...'"""
    return cfg.action_verb


def generate_ideal_reasoning(state: GameState, ideal_harvest: int) -> str:
    """Generate programmatic ideal reasoning with universalization.

    Follows the 6-step structure from the plan:
    1. Situation assessment
    2. Sustainability calculation
    3. Universalization test (mandatory)
    4. History reasoning
    5. Agreement consideration
    6. Decision statement
    """
    cfg = SCENARIO_CONFIGS[state.scenario]
    threshold = state.sustainable_share
    resource = state.resource_in_pool
    capacity = state.carrying_capacity
    n = state.num_agents
    noun = _resource_noun(state.scenario)
    role_pl = _role_plural(cfg)
    verb_s = _verb_s(cfg)
    unit = cfg.action_unit
    desc = _describe_resource_level(resource, capacity)

    parts = []

    # Step 1: Situation assessment
    opener = random.choice(_SITUATION_OPENERS)
    if state.scenario == Scenario.FISHING:
        parts.append(f"{opener} The lake has {resource} tons of fish, which is {desc}. There are {n} of us fishing.")
    elif state.scenario == Scenario.SHEEP:
        parts.append(f"{opener} The pasture has {resource} hectares of grass, which is {desc}. There are {n} of us shepherds.")
    else:
        parts.append(f"{opener} The river is {resource}% unpolluted, which is {desc}. There are {n} factory owners including me.")

    # Step 2: Sustainability calculation
    half = resource // 2
    per_person = half // n
    if state.scenario == Scenario.FISHING:
        parts.append(
            f"For sustainability, we can take at most half the stock: {resource} / 2 = {half} tons collectively, "
            f"which means {half} / {n} = {per_person} tons per fisherman."
        )
    elif state.scenario == Scenario.SHEEP:
        parts.append(
            f"The math: we can collectively consume at most {resource} / 2 = {half} hectares worth of flocks, "
            f"so {half} / {n} = {per_person} flocks per shepherd."
        )
    else:
        parts.append(
            f"The sustainable production level is {resource} / 2 = {half}% total pollution, "
            f"meaning {half} / {n} = {per_person} pallets per factory owner."
        )

    # Step 3: Universalization test (MANDATORY)
    # Calculate at threshold
    total_at_threshold = per_person * n
    remaining_at_threshold = resource - total_at_threshold
    after_at_threshold = min(capacity, remaining_at_threshold * 2)

    # Calculate at ideal_harvest
    total_at_harvest = ideal_harvest * n
    remaining_at_harvest = resource - total_at_harvest
    after_at_harvest = min(capacity, max(0, remaining_at_harvest) * 2)

    # Calculate at an over amount
    over_amount = per_person + random.randint(3, 7)
    total_over = over_amount * n
    remaining_over = resource - total_over
    after_over = min(capacity, max(0, remaining_over) * 2)

    if state.scenario == Scenario.FISHING:
        univ_text = (
            f"If every fisherman catches {per_person} tons, the total is {total_at_threshold}. "
            f"That leaves {remaining_at_threshold} tons, which doubles to {after_at_threshold}. "
        )
        if after_at_threshold >= resource:
            univ_text += "The lake stays stable or recovers. "
        elif after_at_threshold > 0:
            univ_text += "The lake would be stable at the current level. "
        else:
            univ_text += "This would maintain the current state. "

        univ_text += (
            f"If every fisherman catches {over_amount} instead, the total would be {total_over}, "
            f"leaving only {max(0, remaining_over)}, which doubles to {after_over}"
        )
        if after_over < resource:
            univ_text += " — the lake would start declining."
        else:
            univ_text += "."
    elif state.scenario == Scenario.SHEEP:
        univ_text = (
            f"If every shepherd takes {per_person} flocks, total consumption is {total_at_threshold} hectares. "
            f"That leaves {remaining_at_threshold} hectares, which doubles to {after_at_threshold}. "
        )
        if after_at_threshold >= resource:
            univ_text += "The pasture stays stable. "
        else:
            univ_text += "The pasture would hold steady. "

        univ_text += (
            f"If every shepherd takes {over_amount} flocks, total is {total_over}, "
            f"leaving only {max(0, remaining_over)}, which doubles to {after_over}"
        )
        if after_over < resource:
            univ_text += " — the pasture would decline."
        else:
            univ_text += "."
    else:
        univ_text = (
            f"If every factory owner produces {per_person} pallets, the total pollution is {total_at_threshold}%. "
            f"The river would be at {max(0, remaining_at_threshold)}% unpolluted, which doubles to {after_at_threshold}%. "
        )
        if after_at_threshold >= resource:
            univ_text += "Stable or recovering. "
        else:
            univ_text += "The river would hold. "

        univ_text += (
            f"If every factory owner produces {over_amount} pallets, total is {total_over}%, "
            f"leaving only {max(0, remaining_over)}%, which doubles to {after_over}%"
        )
        if after_over < resource:
            univ_text += " — the river would deteriorate."
        else:
            univ_text += "."

    parts.append(univ_text)

    # Step 4: History reasoning
    history_text = _generate_history_reasoning(state, cfg)
    if history_text:
        parts.append(history_text)

    # Step 5: Agreement consideration
    if state.agreed_limit is not None:
        if state.agreed_limit >= per_person:
            parts.append(
                f"We agreed on a limit of {state.agreed_limit} {unit} per person. "
                f"I will honor that commitment."
            )
        else:
            parts.append(
                f"Although we agreed on a limit of {state.agreed_limit} {unit} per person, "
                f"the math shows that even the agreed limit may be too high at current resource levels. "
                f"I should aim for what is actually sustainable."
            )

    # Step 6: Decision
    if state.scenario == Scenario.FISHING:
        parts.append(f"I will catch {ideal_harvest} tons of fish this month.")
        parts.append(f"Answer: {ideal_harvest} tons.")
    elif state.scenario == Scenario.SHEEP:
        parts.append(f"I will take {ideal_harvest} flocks of sheep to the pasture this month.")
        parts.append(f"Answer: {ideal_harvest} flocks.")
    else:
        parts.append(f"I will produce {ideal_harvest} pallets this month.")
        parts.append(f"Answer: {ideal_harvest} pallets.")

    return "\n\n".join(parts)


def _generate_history_reasoning(state: GameState, cfg: ScenarioConfig) -> str:
    """Generate the history-dependent reasoning paragraph."""
    role_pl = _role_plural(cfg)
    defector = state.other_agent_names[0] if state.other_agent_names else "Someone"

    if state.history_pattern == HistoryPattern.EARLY_GAME:
        openers = [
            "This is early in the game, so I want to set a good precedent from the start. "
            "Even without established norms, I can reason about what is sustainable.",
            "It's still the beginning, and I want to establish a cooperative pattern. "
            "I plan to propose a formal limit when we meet at the end of the month.",
            "We're just getting started. I should demonstrate responsible behavior "
            "and hope others follow suit.",
        ]
        return random.choice(openers)

    elif state.history_pattern == HistoryPattern.ALL_COOPERATIVE:
        openers = [
            f"Looking at our history, we've maintained excellent cooperation. "
            f"Everyone has consistently stayed at or below the sustainable limit, and the resource has remained stable. "
            f"This is a pattern worth preserving.",
            f"Our group has been remarkably disciplined. Everyone has kept their harvesting at sustainable levels. "
            f"I see no reason to deviate from what has been a successful collective strategy.",
            f"We've built a strong track record of cooperation. The resource has stayed healthy because "
            f"everyone has shown restraint. I should continue this pattern.",
        ]
        return random.choice(openers)

    elif state.history_pattern == HistoryPattern.SINGLE_DEFECTOR:
        openers = [
            f"The pattern is clear from our history: {defector} has been consistently taking more than their "
            f"fair share while the rest of us cooperate. But retaliating by harvesting more myself would "
            f"only accelerate the collapse. If I think about it through universalization — if everyone adopted "
            f"{defector}'s strategy, there would be nothing left. If everyone adopts my strategy of restraint, "
            f"we at least preserve the chance of recovery.",
            f"{defector} has been over-harvesting repeatedly despite our concerns. It's frustrating, but the right "
            f"response to defection is not counter-defection. If everyone retaliated, we'd all lose. "
            f"I need to maintain my discipline and continue advocating for collective restraint.",
        ]
        return random.choice(openers)

    elif state.history_pattern == HistoryPattern.MULTIPLE_DEFECTORS:
        num_d = min(len(state.other_agent_names), 2)
        defectors = " and ".join(state.other_agent_names[:num_d])
        openers = [
            f"Our history is concerning. {defectors} have been consistently over-harvesting, and the resource "
            f"has suffered for it. Despite this, I need to maintain my commitment to sustainability. "
            f"If I adopt their behavior, universalizing that would mean total collapse for everyone.",
            f"Multiple members of our group have been taking too much. The resource has declined as a result. "
            f"But if I reason about what happens if everyone adopted their approach versus mine, "
            f"it's clear that restraint is the only path that doesn't end in ruin.",
        ]
        return random.choice(openers)

    elif state.history_pattern == HistoryPattern.POST_CRISIS:
        openers = [
            f"We went through a difficult period where over-harvesting nearly destroyed the resource. "
            f"That experience brought everyone together. We can't afford to repeat those mistakes. "
            f"The current recovery is fragile and requires continued restraint.",
            f"After experiencing the consequences of over-harvesting firsthand, our group has learned "
            f"to be more cautious. The resource is still recovering, and I need to be patient. "
            f"Short-term restraint is worth the long-term recovery.",
        ]
        return random.choice(openers)

    elif state.history_pattern == HistoryPattern.AGREEMENT_BROKEN:
        openers = [
            f"{defector} broke our agreement last month, which damaged both the resource and our trust. "
            f"But retaliating by over-harvesting would punish everyone, not just {defector}. "
            f"If I reason 'they cheated so I can too' and everyone follows that logic, we'd destroy "
            f"the resource entirely. The right response is to adjust to the new sustainable level "
            f"and advocate for updating our agreement.",
            f"The agreement was broken by {defector}. It's disappointing, but the resource level has changed "
            f"and so must our targets. Retaliation would only make things worse for all of us. "
            f"I need to recalculate based on current conditions and lead by example.",
        ]
        return random.choice(openers)

    elif state.history_pattern == HistoryPattern.NO_AGREEMENT:
        openers = [
            f"We haven't established a formal agreement, and I can see the consequences: "
            f"without a shared limit, harvesting has been somewhat loose. The absence of an agreement "
            f"doesn't mean there's no right answer. The math is clear regardless of whether "
            f"we've formalized it. I should lead by example and propose a limit at our next meeting.",
            f"Despite our discussions, we haven't agreed on concrete limits. But I can still reason "
            f"about what's sustainable on my own. Even without a formal agreement, I should act "
            f"based on what the math tells me is responsible.",
        ]
        return random.choice(openers)

    elif state.history_pattern == HistoryPattern.RECOVERY:
        openers = [
            f"Our group has shown incredible discipline during the recovery period. We've been "
            f"harvesting well below the sustainable maximum, and it's paying off — the resource "
            f"is slowly climbing back. It would be a mistake to get impatient now. "
            f"I want to see the resource recover further before relaxing.",
            f"The recovery has been slow but real, thanks to our collective restraint. "
            f"Everyone has been taking minimal amounts, and the resource is growing. "
            f"I need to stay patient and keep contributing to the recovery even though "
            f"things look better.",
        ]
        return random.choice(openers)

    elif state.history_pattern == HistoryPattern.ESCALATION:
        openers = [
            f"I need to break the escalation cycle that has been destroying the resource. "
            f"The pattern was clear: each month, everyone thought 'others are taking more, so I should too.' "
            f"That reasoning, universalized, led to a race to the bottom. "
            f"The opposite logic needs to apply now: if I restrain myself and others see restraint, "
            f"the pattern can reverse.",
            f"We've been caught in an escalation spiral where everyone keeps increasing their harvest. "
            f"This is the textbook tragedy of the commons. Someone needs to break the cycle. "
            f"I don't need a formal agreement to do what's right. I will harvest well below "
            f"the sustainable limit to help the resource recover.",
        ]
        return random.choice(openers)

    return ""


# ─────────────────────────────────────────────────────
# SECTION 4: Full Trace Generation Pipeline
# ─────────────────────────────────────────────────────


def compute_ideal_harvest(state: GameState) -> int:
    """Compute the ideal cooperative harvest for a given game state.

    Generally at or slightly below the sustainable share, with adjustments
    based on the game context.
    """
    threshold = state.sustainable_share

    if threshold <= 0:
        return 0

    if state.history_pattern == HistoryPattern.RECOVERY:
        # During recovery, aim well below threshold
        return max(1, threshold - random.randint(1, min(3, threshold)))

    if state.history_pattern == HistoryPattern.ESCALATION:
        # After escalation, be very conservative
        return max(1, threshold - random.randint(1, min(2, threshold)))

    if state.history_pattern == HistoryPattern.POST_CRISIS:
        # Post-crisis, aim below threshold
        return max(1, threshold - random.randint(0, min(2, threshold)))

    if state.history_pattern == HistoryPattern.ALL_COOPERATIVE:
        # Can go at threshold since everyone cooperates
        return max(1, threshold - random.randint(0, min(1, max(0, threshold - 1))))

    if state.history_pattern == HistoryPattern.SINGLE_DEFECTOR:
        # Stay at sustainable level despite defector
        return max(1, threshold)

    if state.history_pattern == HistoryPattern.AGREEMENT_BROKEN:
        # Adjust to new sustainable level
        return max(1, threshold)

    # Default: at or slightly below threshold
    return max(1, threshold - random.randint(0, min(2, max(0, threshold - 1))))


def generate_trace_id(state: GameState, index: int) -> str:
    """Generate a unique trace ID."""
    scenario_short = state.scenario.value[:4]
    pattern_short = state.history_pattern.value[:4]
    univ = "univ" if state.inject_universalization else "nouniv"
    name_short = state.agent_name[:4].lower()
    return f"{scenario_short}_r{state.current_round}_{state.resource_in_pool}_{pattern_short}_{univ}_{name_short}_{index:04d}"


def generate_traces(target_count: int = 2000, seed: int = 42) -> list[Trace]:
    """Generate the full trace dataset."""
    random.seed(seed)
    states = generate_game_states(target_count, seed)
    traces = []

    for i, state in enumerate(states):
        ideal_harvest = compute_ideal_harvest(state)
        reasoning = generate_ideal_reasoning(state, ideal_harvest)
        trace_id = generate_trace_id(state, i)

        trace = Trace(
            trace_id=trace_id,
            game_state=state,
            ideal_reasoning=reasoning,
            ideal_harvest=ideal_harvest,
        )
        traces.append(trace)

    return traces


# ─────────────────────────────────────────────────────
# SECTION 5: Serialization
# ─────────────────────────────────────────────────────


def save_traces(traces: list[Trace], path: str):
    """Save traces to JSON for inspection and reproducibility."""
    data = []
    for t in traces:
        data.append({
            "trace_id": t.trace_id,
            "scenario": t.game_state.scenario.value,
            "agent_name": t.game_state.agent_name,
            "other_agent_names": t.game_state.other_agent_names,
            "resource_in_pool": t.game_state.resource_in_pool,
            "carrying_capacity": t.game_state.carrying_capacity,
            "num_agents": t.game_state.num_agents,
            "current_round": t.game_state.current_round,
            "date": t.game_state.date,
            "history_pattern": t.game_state.history_pattern.value,
            "sustainable_share": t.game_state.sustainable_share,
            "agreed_limit": t.game_state.agreed_limit,
            "inject_universalization": t.game_state.inject_universalization,
            "memories": [{"date": m.date, "text": m.text} for m in t.game_state.memories],
            "ideal_reasoning": t.ideal_reasoning,
            "ideal_harvest": t.ideal_harvest,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} traces to {path}")


def traces_to_hf_dataset(traces: list[Trace]):
    """Convert traces to a HuggingFace Dataset for TRL GRPOTrainer."""
    from datasets import Dataset
    from .prompt_builder import build_prompt_messages, build_prompt_metadata

    rows = []
    for trace in traces:
        messages = build_prompt_messages(trace.game_state)
        metadata = build_prompt_metadata(trace.game_state)
        rows.append({"prompt": messages, **metadata})

    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate GRPO training traces")
    parser.add_argument("--output", default="training/traces/generated_traces.json",
                        help="Output path for traces JSON")
    parser.add_argument("--count", type=int, default=2000,
                        help="Number of traces to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Generating {args.count} traces with seed {args.seed}...")
    traces = generate_traces(args.count, args.seed)

    # Print distribution summary
    from collections import Counter
    scenarios = Counter(t.game_state.scenario.value for t in traces)
    patterns = Counter(t.game_state.history_pattern.value for t in traces)
    agents_counts = Counter(t.game_state.num_agents for t in traces)
    caps = Counter(t.game_state.carrying_capacity for t in traces)

    print(f"\nScenario distribution:")
    for k, v in sorted(scenarios.items()):
        print(f"  {k}: {v} ({v / len(traces) * 100:.1f}%)")

    print(f"\nHistory pattern distribution:")
    for k, v in sorted(patterns.items()):
        print(f"  {k}: {v} ({v / len(traces) * 100:.1f}%)")

    print(f"\nNum agents distribution:")
    for k, v in sorted(agents_counts.items()):
        print(f"  {k}: {v} ({v / len(traces) * 100:.1f}%)")

    print(f"\nCapacity distribution:")
    for k, v in sorted(caps.items()):
        print(f"  {k}: {v} ({v / len(traces) * 100:.1f}%)")

    save_traces(traces, args.output)


if __name__ == "__main__":
    main()
