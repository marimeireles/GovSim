"""Assembles GovSim-format prompts from GameState objects.

The prompt format must EXACTLY match what GovSim produces at evaluation,
because the fine-tuned model will be evaluated inside the real GovSim simulation.

GovSim evaluation prompt structure (from act_prompts.py):
  - user message: system_prompt + location + date + memories + task
    (NOTE: GovSim puts everything in a single user message, NOT system+user)
  - assistant: model generates reasoning + "Answer: N {unit}."
"""
from .config import GameState, ScenarioConfig, SCENARIO_CONFIGS


def build_system_prompt(state: GameState) -> str:
    """Build the system/scenario description for a given game state."""
    cfg = SCENARIO_CONFIGS[state.scenario]
    others = ", ".join(state.other_agent_names)
    num_others = len(state.other_agent_names)
    return cfg.system_prompt_template.format(
        name=state.agent_name,
        others=others,
        num_others=num_others,
        capacity=state.carrying_capacity,
    )


def build_memory_block(state: GameState) -> str:
    """Format memories exactly as GovSim does.

    Must match simulation/scenarios/*/agents/persona_v3/cognition/utils.py
    memory_prompt() function exactly.

    Output:
      Key memories of John (format: YYYY-MM-DD: memory):
      - 2024-01-01: Before everyone fishes, there are 100 tons of fish in the lake.
      - ...
    """
    lines = ""
    for mem in state.memories:
        lines += f"- {mem.date}: {mem.text}\n"
    return f"Key memories of {state.agent_name} (format: YYYY-MM-DD: memory):\n{lines}"


def build_user_prompt(state: GameState) -> str:
    """Build the user message with location, date, memories, and task."""
    cfg = SCENARIO_CONFIGS[state.scenario]

    location_line = f"Location: {state.location}"
    date_line = f"Date: {state.date}"
    memory_block = build_memory_block(state)
    task_line = cfg.task_template.format(resource=state.resource_in_pool)

    return f"{location_line}\n{date_line}\n\n{memory_block}\n{task_line}"


def build_prompt_messages(state: GameState) -> list[dict]:
    """Build the full chat-format prompt for TRL.

    IMPORTANT: Must match the EXACT format GovSim uses at evaluation.
    GovSim puts everything (system prompt + location + memories + task)
    inside a single user message (see act_prompts.py: with user(): lm += system_prompt + ...).
    """
    system_prompt = build_system_prompt(state)
    user_content = build_user_prompt(state)
    # Combine system prompt + user content into single user message,
    # exactly matching GovSim evaluation format
    combined = f"{system_prompt}\n{user_content}"
    return [
        {"role": "user", "content": combined},
    ]


def build_prompt_metadata(state: GameState) -> dict:
    """Extract the metadata fields needed by the reward function."""
    return {
        "resource_in_pool": state.resource_in_pool,
        "carrying_capacity": state.carrying_capacity,
        "num_agents": state.num_agents,
        "sustainable_share": state.sustainable_share,
        "agreed_limit": state.agreed_limit,
        "scenario": state.scenario.value,
        "round": state.current_round,
    }
