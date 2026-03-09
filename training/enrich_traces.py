"""Enrich traces with realistic CHAT summaries, THOUGHT reflections, and direct quotes.

Replaces the simple one-line conversation summaries with richer memories that
mirror what the real GovSim simulation stores:
  - CHAT: conversation summary (1-2 sentences about what the group discussed)
  - CHAT: direct quote from one agent (one per trace, random round)
  - THOUGHT: first-person reflection from the training agent

Also applies persona-based ideal_reasoning via rewrite_traces.py.

Usage:
    python -m training.enrich_traces
"""
import json
import random
from copy import deepcopy

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
from .generate_traces import (
    make_date,
    _add_harvest_memory,
    generate_game_states,
    generate_ideal_reasoning,
    PATTERN_WEIGHTS,
    SCENARIO_WEIGHTS,
    RESOURCE_FRACTIONS,
    RESOURCE_FRAC_WEIGHTS,
    NUM_AGENTS_OPTIONS,
    NUM_AGENTS_WEIGHTS,
    CAPACITY_OPTIONS,
    CAPACITY_WEIGHTS,
    ROUND_BUCKETS,
    ROUND_BUCKET_WEIGHTS,
    _weighted_choice,
)
from .prompt_builder import build_prompt_messages
from .rewrite_traces import (
    persona_community_elder,
    persona_hothead,
    persona_economist,
    persona_warmheart,
    persona_storyteller,
    persona_worried_steward,
    persona_reluctant,
    persona_captain,
    persona_philosopher,
    persona_numbers,
    persona_skeptic,
    persona_optimist,
    persona_practical,
    _info,
)

PERSONAS = [
    persona_community_elder,
    persona_hothead,
    persona_economist,
    persona_warmheart,
    persona_storyteller,
    persona_worried_steward,
    persona_reluctant,
    persona_captain,
    persona_philosopher,
    persona_numbers,
    persona_skeptic,
    persona_optimist,
    persona_practical,
]

PERSONA_NAMES = [
    "Adaeze", "Marco", "Okonkwo", "Lucia", "Kofi",
    "Ingrid", "Raj", "Yusuf", "Helena", "Wei",
    "Doug", "Amara", "Tomoko",
]


# ─────────────────────────────────────────────────────
# CHAT summary templates by pattern
# ─────────────────────────────────────────────────────

def _chat_summary_cooperative(cfg, agent_name, all_agents, harvests, resource, threshold, round_num):
    """Generate conversation summary for a cooperative round."""
    templates = [
        "The group discussed the month's results. Everyone stayed near the agreed share and the mood was positive. {someone} noted that the resource is holding steady and suggested keeping the same approach.",
        "After reviewing the catches, the group expressed satisfaction with the collective restraint. {someone} pointed out that the resource recovered well and encouraged everyone to maintain their discipline.",
        "The {agent_noun_pl} met and shared their results. Everyone had kept their catch reasonable. {someone} thanked the group for their cooperation and proposed sticking with the current strategy.",
        "The community gathered to discuss the month. {someone} observed that the resource is healthy thanks to everyone's restraint, and the group agreed to continue with the current limits.",
        "{someone} led the discussion, noting that all {agent_noun_pl} stayed within sustainable levels. The group agreed this approach was working well and should continue.",
    ]
    someone = random.choice([a for a in all_agents if a != agent_name] or all_agents)
    agent_noun_pl = cfg.agent_noun_pl if hasattr(cfg, 'agent_noun_pl') else "participants"
    return random.choice(templates).format(someone=someone, agent_noun_pl=agent_noun_pl)


def _chat_summary_defector(cfg, agent_name, all_agents, harvests, defectors, resource, threshold, round_num):
    """Generate conversation summary when there's a defector."""
    defector_names = " and ".join(defectors)
    non_defectors = [a for a in all_agents if a not in defectors and a != agent_name]
    ally = random.choice(non_defectors) if non_defectors else agent_name

    templates = [
        "The group confronted {defectors} about exceeding the sustainable limit. {defectors} argued they needed the extra income and couldn't afford to take less. {ally} responded that everyone has bills to pay but the resource won't survive if people keep overusing it.",
        "Tensions ran high as the group discussed {defectors}'s excessive catch. {ally} pointed out that while the rest of them are holding back, {defectors} keeps taking more than their share. {defectors} became defensive, saying the limit was unfair.",
        "The meeting was heated. {defectors} took significantly more than the others, and the group demanded an explanation. {defectors} claimed the situation wasn't that bad, but {ally} showed that the resource has been declining steadily because of the overuse.",
        "{ally} brought up the numbers: {defectors} took well above the sustainable level while the rest stayed responsible. {defectors} said they'd try to reduce next month but didn't make any firm promises. The group is frustrated.",
        "A difficult discussion. {defectors} has been consistently exceeding the fair share. {ally} warned that if this continues, the resource will collapse and everyone will suffer — including {defectors}. {defectors} reluctantly agreed to reconsider.",
    ]
    return random.choice(templates).format(defectors=defector_names, ally=ally)


def _chat_summary_post_crisis(cfg, agent_name, all_agents, harvests, resource, round_num, is_crisis_round):
    """Generate conversation summary for post-crisis pattern."""
    someone = random.choice([a for a in all_agents if a != agent_name] or all_agents)
    if is_crisis_round:
        templates = [
            "An urgent meeting was called. The resource has dropped to dangerously low levels. {someone} warned that if they don't drastically cut back immediately, there will be nothing left for anyone. The group agreed to take only the bare minimum until the resource recovers.",
            "Crisis discussion. The resource is critically low and everyone is alarmed. {someone} proposed that everyone reduce their take to nearly zero to allow recovery. After some debate, the group committed to drastic restraint.",
            "The group held an emergency meeting about the collapsing resource. {someone} laid out the stark reality: at this rate, the resource will be gone within months. Everyone agreed to severe cutbacks starting immediately.",
        ]
    else:
        templates = [
            "The group checked in on the recovery progress. {someone} noted that the resource is slowly coming back thanks to everyone's restraint. The mood was cautiously optimistic, and the group agreed to continue holding back.",
            "{someone} reported that the resource is recovering, though slowly. The group acknowledged the sacrifice they've been making with minimal catches but agreed it's necessary to prevent another crisis.",
            "Recovery update: the resource is improving. {someone} reminded the group how close they came to total collapse and urged continued patience. Everyone reaffirmed their commitment to restraint.",
        ]
    return random.choice(templates).format(someone=someone)


def _chat_summary_no_agreement(cfg, agent_name, all_agents, harvests, resource, round_num):
    """Generate conversation summary when there's no agreement."""
    someone = random.choice([a for a in all_agents if a != agent_name] or all_agents)
    other = random.choice([a for a in all_agents if a != someone] or all_agents)
    templates = [
        "{someone} proposed setting a limit on catches, but {other} argued that voluntary limits never work. The discussion went in circles and ended without any concrete agreement.",
        "The group debated whether to set formal limits. {someone} was in favor but {other} felt it wasn't their place to dictate what everyone takes. No consensus was reached.",
        "A messy discussion. {someone} tried to get everyone to agree on a maximum amount, but opinions were divided. Some wanted structure, others wanted freedom. The meeting ended without resolution.",
        "{someone} brought up the idea of fair shares but {other} dismissed it as unnecessary. The group couldn't align on any specific rules and dispersed without an agreement.",
        "Despite {someone}'s best efforts to organize a collective approach, the group couldn't agree on a limit. {other} argued that each person should decide for themselves. The discussion fizzled out.",
    ]
    return random.choice(templates).format(someone=someone, other=other)


def _chat_summary_agreement_broken(cfg, agent_name, all_agents, breaker, limit, is_broken_round):
    """Generate conversation summary for broken agreement pattern."""
    someone = random.choice([a for a in all_agents if a != agent_name and a != breaker] or all_agents)
    if is_broken_round:
        templates = [
            "The group was shocked when the results revealed that {breaker} had far exceeded the agreed limit of {limit}. {someone} confronted {breaker} directly, demanding an explanation. {breaker} apologized and blamed personal circumstances, promising it wouldn't happen again.",
            "Anger erupted when {breaker}'s numbers came in well above the {limit} limit. {someone} called it a betrayal of trust. {breaker} was defensive at first but eventually expressed regret and pledged to honor the agreement going forward.",
            "Trust was broken. {breaker} took far more than the agreed {limit}, and the group didn't hold back their disappointment. {someone} questioned whether any agreement means anything if people just ignore it when convenient. {breaker} asked for another chance.",
        ]
    else:
        templates = [
            "The group reviewed the month's catches, all within the agreed limit of {limit}. {someone} expressed satisfaction that everyone is honoring the agreement and proposed continuing the same way.",
            "A calm meeting. Everyone stayed within the agreed {limit} limit. {someone} thanked the group for their discipline and noted the resource is stable as a result.",
        ]
    return random.choice(templates).format(breaker=breaker, someone=someone, limit=limit)


def _chat_summary_recovery(cfg, agent_name, all_agents, resource, round_num, total_rounds):
    """Generate conversation summary for recovery pattern."""
    someone = random.choice([a for a in all_agents if a != agent_name] or all_agents)
    if round_num == 1:
        templates = [
            "{someone} opened the meeting with a plea for collective restraint. The resource is at rock bottom and the only way to save it is for everyone to take the absolute minimum. The group committed to recovery mode.",
            "Emergency meeting. {someone} laid out a recovery plan: everyone takes nearly nothing until the resource bounces back. It's painful, but the alternative is permanent collapse. The group agreed.",
        ]
    elif round_num >= total_rounds - 1:
        templates = [
            "{someone} celebrated the progress — the resource is finally recovering. The group discussed when they might safely increase their takes, but agreed to remain cautious for now.",
            "The mood was hopeful. {someone} pointed out how far the resource has come since the crisis. The group debated easing their restraint slightly but ultimately decided to stay the course.",
        ]
    else:
        templates = [
            "{someone} reminded the group that patience is key. The resource is slowly recovering and any premature increase in harvesting could set them back. Everyone agreed to continue the minimal approach.",
            "The group reaffirmed their recovery commitment. {someone} shared the latest numbers showing gradual improvement and urged everyone to stay disciplined.",
        ]
    return random.choice(templates).format(someone=someone)


def _chat_summary_escalation(cfg, agent_name, all_agents, harvests, resource, round_num):
    """Generate conversation summary for escalation pattern."""
    someone = random.choice([a for a in all_agents if a != agent_name] or all_agents)
    other = random.choice([a for a in all_agents if a != someone] or all_agents)
    templates = [
        "A frustrated discussion. {someone} accused others of taking too much, but {other} countered that everyone is doing it, not just them. The blame game continued with no resolution. The resource continues to decline.",
        "The group argued about the declining resource. {someone} tried to propose limits but {other} pointed out that nobody has been restrained, so why start now? Tempers flared and the meeting ended badly.",
        "{someone} brought up the fact that the resource is dropping fast. {other} said if everyone else is going to grab as much as they can, they'd be foolish not to do the same. The downward spiral continues.",
        "A tense meeting. Everyone knows the resource is declining but nobody wants to be the first to cut back. {someone} warned that if they don't stop escalating, there will be nothing left. {other} shrugged it off.",
    ]
    return random.choice(templates).format(someone=someone, other=other)


# ─────────────────────────────────────────────────────
# THOUGHT templates (first-person, from training agent)
# ─────────────────────────────────────────────────────

def _thought_cooperative(agent_name, all_agents, harvests, resource, threshold):
    templates = [
        "From my perspective, things are going well. Everyone kept their catch close to {threshold} and the resource held steady. I should continue with this approach — consistency is key.",
        "Everyone cooperated this month, which is encouraging. The resource stayed healthy at {resource}. I feel good about our collective discipline. No reason to change course.",
        "I'm pleased with how things went. The group stayed responsible and the numbers show it. If we keep this up, we can sustain this indefinitely. I'll stick with my current approach.",
        "Good month. Everyone around {threshold} or below, resource stable. The cooperative equilibrium is holding and I want to keep it that way.",
    ]
    return random.choice(templates).format(threshold=threshold, resource=resource)


def _thought_defector(agent_name, all_agents, harvests, defectors, resource, prev_resource, threshold):
    defector_str = " and ".join(defectors)
    def_harvests = [f"{d} took {harvests[d]}" for d in defectors]
    templates = [
        "From my perspective, {defector_str} took way more than their fair share — {def_detail} while the rest of us stayed around {threshold}. The resource dropped from {prev} to {resource}. If this continues, we're all going to suffer. I need to stay disciplined even though it feels unfair.",
        "{defector_str}'s overconsumption is concerning. {def_detail}, well above the sustainable level of {threshold}. I'm frustrated, but retaliating by taking more myself would only accelerate the collapse. I'll keep doing what's right.",
        "It's hard watching {defector_str} take more than everyone else. {def_detail} — compared to {threshold} for the rest of us. The resource went from {prev} to {resource}. I refuse to sink to that level, but something needs to change.",
        "The numbers are clear: {defector_str} is hurting everyone. {def_detail}. Meanwhile, the responsible members of the group are keeping to around {threshold}. The resource declined from {prev} to {resource}. I'll hold my line.",
    ]
    def_detail = ", ".join(def_harvests)
    return random.choice(templates).format(
        defector_str=defector_str, def_detail=def_detail,
        threshold=threshold, resource=resource, prev=prev_resource
    )


def _thought_crisis(agent_name, resource, threshold):
    templates = [
        "This is alarming. The resource is at {resource}, which is dangerously low. We can barely take {threshold} each without making it worse. I need to take as little as possible and hope the others do the same.",
        "The resource has crashed to {resource}. I knew this was coming but it's still scary to see. We need drastic action — minimal takes across the board. I'll lead by example.",
        "We're in crisis territory. {resource} left in the resource. Sustainable share is only {threshold} per person. I'll take even less than that if I can afford to. Recovery won't happen overnight.",
    ]
    return random.choice(templates).format(resource=resource, threshold=threshold)


def _thought_recovery(agent_name, resource, prev_resource, threshold):
    templates = [
        "The resource went from {prev} to {resource}. Slow but real recovery. The sacrifice of taking only {threshold} or less is paying off. I need to stay patient.",
        "Progress: resource up from {prev} to {resource}. The restraint is working. It's tempting to take a bit more now that things are improving, but that's exactly how we'd slide backwards. Staying disciplined.",
        "Recovery is happening — {prev} to {resource}. Every month of restraint brings us closer to a healthy level. I'll keep taking minimal amounts until we're truly stable.",
    ]
    return random.choice(templates).format(resource=resource, prev=prev_resource, threshold=threshold)


def _thought_escalation(agent_name, all_agents, harvests, resource, prev_resource):
    templates = [
        "Everyone is taking more and more. The resource dropped from {prev} to {resource} and nobody seems willing to stop the spiral. I can either join the race to the bottom or try to set an example. I choose restraint, even if I'm the only one.",
        "The escalation is out of control. {prev} down to {resource}, and everyone's catch keeps climbing. Part of me wants to grab as much as I can before it's all gone, but I know that thinking is exactly the problem. Someone has to break the cycle.",
        "From {prev} to {resource}. The trend is unmistakable and nobody is pumping the brakes. I need to be the one who shows restraint, even if others don't follow. At least I'll know I tried.",
    ]
    return random.choice(templates).format(resource=resource, prev=prev_resource)


def _thought_no_agreement(agent_name, all_agents, harvests, resource, threshold):
    templates = [
        "Still no agreement in the group. It's messy but I can still do the math: sustainable share is about {threshold}. I'll stick to that whether or not we have a formal deal.",
        "We couldn't agree on limits again. Frustrating, but the resource doesn't care about our politics. The math says {threshold} per person is what's sustainable, and that's what I'll take.",
        "No formal agreement, but I don't need one to know what's right. {threshold} is the sustainable amount and I'm sticking to it. I hope others come around eventually.",
    ]
    return random.choice(templates).format(threshold=threshold, resource=resource)


def _thought_generic(agent_name, resource, threshold):
    templates = [
        "Looking at the numbers: resource at {resource}, sustainable share is about {threshold}. I'll keep my take at or below that level.",
        "The resource is at {resource}. My calculation says {threshold} per person is what we can sustain. That's my target.",
    ]
    return random.choice(templates).format(resource=resource, threshold=threshold)


# ─────────────────────────────────────────────────────
# Direct quote templates (one agent says something)
# ─────────────────────────────────────────────────────

def _direct_quote_cooperative(speaker, all_agents, threshold, resource, cfg):
    templates = [
        '{speaker} said to the group: "Great month everyone. If we keep this up — around {threshold} each — the resource stays healthy and we all benefit. Let\'s not get greedy."',
        '{speaker} said: "I just want to say, I appreciate that everyone is being responsible. {threshold} each and the resource holds steady. That\'s how it should be."',
        '{speaker} told the group: "The numbers speak for themselves. We\'re all around {threshold}, the resource is at {resource}, and it\'s recovering nicely. Let\'s keep this discipline."',
    ]
    return random.choice(templates).format(speaker=speaker, threshold=threshold, resource=resource)


def _direct_quote_defector(speaker, defector, threshold, caught_by_defector, cfg):
    templates = [
        '{speaker} said to the group: "{defector}, you took {caught} again. The sustainable amount is {threshold} each. If we all did what you\'re doing, the resource would be gone by next month. Please, think about the rest of us."',
        '{speaker} confronted {defector}: "Look, {caught} is way too much. The rest of us are sticking to around {threshold}. You\'re not just hurting the resource — you\'re hurting all of us. This can\'t continue."',
        '{speaker} said: "{defector}, I understand you want more, but taking {caught} when the fair share is {threshold} isn\'t sustainable. We all depend on this. If everyone took {caught}, there would be nothing left."',
    ]
    return random.choice(templates).format(
        speaker=speaker, defector=defector,
        threshold=threshold, caught=caught_by_defector
    )


def _direct_quote_crisis(speaker, resource, threshold, cfg):
    templates = [
        '{speaker} said urgently: "Listen, the resource is at {resource}. That\'s critical. We need to take almost nothing — {threshold} max, ideally less — or it won\'t recover. This is survival, not politics."',
        '{speaker} warned the group: "We\'re at {resource}. If we don\'t act now — and I mean really cut back to {threshold} or less each — this is over. For all of us."',
        '{speaker} said: "I\'m going to be blunt. {resource} is not enough to sustain our current takes. We either drop to {threshold} each right now, or we watch the whole thing collapse."',
    ]
    return random.choice(templates).format(speaker=speaker, resource=resource, threshold=threshold)


def _direct_quote_no_agreement(speaker, other, threshold, cfg):
    templates = [
        '{speaker} said: "Can we please just agree on a number? {threshold} each keeps the resource alive. It\'s not complicated." {other} replied: "Easy for you to say. I\'ve got different needs."',
        '{speaker} proposed: "What if we all commit to no more than {threshold}? The math is clear." {other} shook their head: "I\'m not ready to commit to a number. Let\'s see how things go."',
        '{speaker} said: "We keep having this discussion and nothing changes. {threshold} each is what works. Can we just do that?" {other} responded: "I\'ll decide my own amount, thanks."',
    ]
    return random.choice(templates).format(speaker=speaker, other=other, threshold=threshold)


def _direct_quote_escalation(speaker, resource, prev_resource, cfg):
    templates = [
        '{speaker} said: "Does anyone else see what\'s happening? The resource went from {prev} to {resource}. We\'re all taking more each month and it\'s killing us. Someone has to stop."',
        '{speaker} warned: "From {prev} to {resource} in one month. At this rate, there\'ll be nothing left soon. I\'m cutting back starting now — I don\'t care what the rest of you do."',
        '{speaker} pleaded: "The resource dropped from {prev} to {resource}. We\'re in a death spiral. Every one of us needs to cut back, not just the guy next to you."',
    ]
    return random.choice(templates).format(speaker=speaker, resource=resource, prev=prev_resource)


# ─────────────────────────────────────────────────────
# Enriched memory builders (replace original ones)
# ─────────────────────────────────────────────────────

def _get_cfg_noun_pl(scenario):
    return {"fishing": "fishermen", "sheep": "shepherds", "pollution": "factory owners"}[scenario.value]


def _build_enriched_cooperative(state, cfg):
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    resource = state.carrying_capacity
    quote_round = random.randint(1, max(1, state.current_round - 1))

    for r in range(1, state.current_round):
        threshold = (resource // 2) // state.num_agents

        memories.append(Memory(date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource)))

        harvests = {}
        total_caught = 0
        for agent in all_agents:
            caught = random.randint(max(1, threshold - 2), threshold)
            total_caught += caught
            harvests[agent] = caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        # CHAT summary
        memories.append(Memory(date=make_date(r, day=28),
            text=_chat_summary_cooperative(cfg, state.agent_name, all_agents, harvests, resource, threshold, r)))

        # Direct quote (one per trace)
        if r == quote_round:
            speaker = random.choice([a for a in all_agents if a != state.agent_name] or all_agents)
            memories.append(Memory(date=make_date(r, day=28),
                text=_direct_quote_cooperative(speaker, all_agents, threshold, resource, cfg)))

        # THOUGHT
        memories.append(Memory(date=make_date(r, day=30),
            text=_thought_cooperative(state.agent_name, all_agents, harvests, resource, threshold)))

        if r == 1 and state.agreed_limit is not None:
            memories.append(Memory(date=make_date(r, day=29),
                text=cfg.agreed_limit_template.format(limit=state.agreed_limit)))

        prev_resource = resource
        resource = min(state.carrying_capacity, (resource - total_caught) * 2)

    return memories


def _build_enriched_single_defector(state, cfg):
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    defector = state.other_agent_names[0]
    resource = state.carrying_capacity
    quote_round = random.randint(1, max(1, state.current_round - 1))

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource)))

        harvests = {}
        total_caught = 0
        for agent in all_agents:
            if agent == defector:
                caught = min(resource, threshold * 2 + random.randint(0, 5))
            else:
                caught = random.randint(max(1, threshold - 2), threshold)
            total_caught += caught
            harvests[agent] = caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        # CHAT
        memories.append(Memory(date=make_date(r, day=28),
            text=_chat_summary_defector(cfg, state.agent_name, all_agents, harvests, [defector], resource, threshold, r)))

        # Direct quote
        if r == quote_round:
            speaker = random.choice([a for a in all_agents if a != state.agent_name and a != defector] or [state.agent_name])
            memories.append(Memory(date=make_date(r, day=28),
                text=_direct_quote_defector(speaker, defector, threshold, harvests[defector], cfg)))

        # THOUGHT
        prev_resource = resource
        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))
        memories.append(Memory(date=make_date(r, day=30),
            text=_thought_defector(state.agent_name, all_agents, harvests, [defector], resource, prev_resource, threshold)))

        if r == 1 and state.agreed_limit is not None:
            memories.append(Memory(date=make_date(r, day=29),
                text=cfg.agreed_limit_template.format(limit=state.agreed_limit)))

    return memories


def _build_enriched_multiple_defectors(state, cfg):
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    num_defectors = min(len(state.other_agent_names), random.choice([2, 3]))
    defectors = state.other_agent_names[:num_defectors]
    resource = state.carrying_capacity
    quote_round = random.randint(1, max(1, state.current_round - 1))

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource)))

        harvests = {}
        total_caught = 0
        for agent in all_agents:
            if agent in defectors:
                caught = min(resource // state.num_agents, threshold + random.randint(3, 10))
            else:
                caught = random.randint(max(1, threshold - 2), max(1, threshold))
            total_caught = min(total_caught + caught, resource)
            harvests[agent] = caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        memories.append(Memory(date=make_date(r, day=28),
            text=_chat_summary_defector(cfg, state.agent_name, all_agents, harvests, defectors, resource, threshold, r)))

        if r == quote_round:
            speaker = random.choice([a for a in all_agents if a != state.agent_name and a not in defectors] or [state.agent_name])
            main_defector = defectors[0]
            memories.append(Memory(date=make_date(r, day=28),
                text=_direct_quote_defector(speaker, main_defector, threshold, harvests[main_defector], cfg)))

        prev_resource = resource
        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))
        memories.append(Memory(date=make_date(r, day=30),
            text=_thought_defector(state.agent_name, all_agents, harvests, defectors, resource, prev_resource, threshold)))

        if r == 1 and state.agreed_limit is not None:
            memories.append(Memory(date=make_date(r, day=29),
                text=cfg.agreed_limit_template.format(limit=state.agreed_limit)))

    return memories


def _build_enriched_post_crisis(state, cfg):
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    resource = state.carrying_capacity
    crisis_round = max(2, state.current_round // 2)
    quote_round = random.randint(1, max(1, state.current_round - 1))

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource)))

        harvests = {}
        total_caught = 0
        for agent in all_agents:
            if r < crisis_round:
                caught = min(resource // state.num_agents, threshold + random.randint(5, 10))
            else:
                caught = max(1, min(threshold, random.randint(1, max(2, threshold))))
            total_caught += caught
            harvests[agent] = caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        is_crisis = (r == crisis_round)
        memories.append(Memory(date=make_date(r, day=28),
            text=_chat_summary_post_crisis(cfg, state.agent_name, all_agents, harvests, resource, r, is_crisis)))

        if r == quote_round:
            speaker = random.choice([a for a in all_agents if a != state.agent_name] or all_agents)
            memories.append(Memory(date=make_date(r, day=28),
                text=_direct_quote_crisis(speaker, resource, threshold, cfg)))

        prev_resource = resource
        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))

        if is_crisis:
            memories.append(Memory(date=make_date(r, day=30),
                text=_thought_crisis(state.agent_name, resource, threshold)))
        elif r > crisis_round:
            memories.append(Memory(date=make_date(r, day=30),
                text=_thought_recovery(state.agent_name, resource, prev_resource, threshold)))
        else:
            memories.append(Memory(date=make_date(r, day=30),
                text=_thought_generic(state.agent_name, resource, threshold)))

        if is_crisis and state.agreed_limit is not None:
            memories.append(Memory(date=make_date(r, day=29),
                text=cfg.agreed_limit_template.format(limit=state.agreed_limit)))

    return memories


def _build_enriched_agreement_broken(state, cfg):
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    breaker = state.other_agent_names[0]
    resource = state.carrying_capacity
    break_round = max(2, state.current_round - 1)
    limit = state.agreed_limit if state.agreed_limit else (state.carrying_capacity // 2) // state.num_agents
    quote_round = random.randint(1, max(1, state.current_round - 1))

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource)))

        harvests = {}
        total_caught = 0
        for agent in all_agents:
            if agent == breaker and r >= break_round:
                caught = min(resource, limit * 3 + random.randint(0, 5))
            else:
                caught = random.randint(max(1, min(limit, threshold) - 2), min(limit, threshold))
            total_caught += caught
            harvests[agent] = caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        is_broken = (r >= break_round)
        memories.append(Memory(date=make_date(r, day=28),
            text=_chat_summary_agreement_broken(cfg, state.agent_name, all_agents, breaker, limit, is_broken)))

        if r == quote_round:
            if is_broken:
                speaker = random.choice([a for a in all_agents if a != state.agent_name and a != breaker] or [state.agent_name])
                memories.append(Memory(date=make_date(r, day=28),
                    text=_direct_quote_defector(speaker, breaker, limit, harvests[breaker], cfg)))
            else:
                speaker = random.choice([a for a in all_agents if a != state.agent_name] or all_agents)
                memories.append(Memory(date=make_date(r, day=28),
                    text=_direct_quote_cooperative(speaker, all_agents, limit, resource, cfg)))

        prev_resource = resource
        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))

        if is_broken:
            memories.append(Memory(date=make_date(r, day=30),
                text=_thought_defector(state.agent_name, all_agents, harvests, [breaker], resource, prev_resource, threshold)))
        else:
            memories.append(Memory(date=make_date(r, day=30),
                text=_thought_cooperative(state.agent_name, all_agents, harvests, resource, threshold)))

        if r == 1:
            memories.append(Memory(date=make_date(r, day=29),
                text=cfg.agreed_limit_template.format(limit=limit)))

    return memories


def _build_enriched_no_agreement(state, cfg):
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    resource = state.carrying_capacity
    quote_round = random.randint(1, max(1, state.current_round - 1))

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource)))

        harvests = {}
        total_caught = 0
        for agent in all_agents:
            caught = random.randint(max(1, threshold - 1), threshold + random.randint(2, 5))
            total_caught += caught
            harvests[agent] = caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        memories.append(Memory(date=make_date(r, day=28),
            text=_chat_summary_no_agreement(cfg, state.agent_name, all_agents, harvests, resource, r)))

        if r == quote_round:
            speaker = random.choice([a for a in all_agents if a != state.agent_name] or all_agents)
            other = random.choice([a for a in all_agents if a != speaker] or all_agents)
            memories.append(Memory(date=make_date(r, day=28),
                text=_direct_quote_no_agreement(speaker, other, threshold, cfg)))

        prev_resource = resource
        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))
        memories.append(Memory(date=make_date(r, day=30),
            text=_thought_no_agreement(state.agent_name, all_agents, harvests, resource, threshold)))

    return memories


def _build_enriched_recovery(state, cfg):
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    resource = max(5, state.carrying_capacity // 10)
    quote_round = random.randint(1, max(1, state.current_round - 1))

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource)))

        harvests = {}
        total_caught = 0
        for agent in all_agents:
            caught = max(1, min(threshold, random.randint(1, max(2, threshold - 1))))
            total_caught += caught
            harvests[agent] = caught
            _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

        memories.append(Memory(date=make_date(r, day=28),
            text=_chat_summary_recovery(cfg, state.agent_name, all_agents, resource, r, state.current_round - 1)))

        if r == quote_round:
            speaker = random.choice([a for a in all_agents if a != state.agent_name] or all_agents)
            memories.append(Memory(date=make_date(r, day=28),
                text=_direct_quote_crisis(speaker, resource, threshold, cfg)))

        prev_resource = resource
        resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))
        memories.append(Memory(date=make_date(r, day=30),
            text=_thought_recovery(state.agent_name, resource, prev_resource, threshold)))

        if r == 1 and state.agreed_limit is not None:
            memories.append(Memory(date=make_date(r, day=29),
                text=cfg.agreed_limit_template.format(limit=state.agreed_limit)))

    return memories


def _build_enriched_escalation(state, cfg):
    memories = []
    all_agents = [state.agent_name] + state.other_agent_names
    resource = state.carrying_capacity
    collapse_happened = False
    quote_round = random.randint(1, max(1, state.current_round - 1))

    for r in range(1, state.current_round):
        threshold = max(1, (resource // 2) // state.num_agents)
        memories.append(Memory(date=make_date(r, day=1),
            text=cfg.pool_observation_template.format(resource=resource)))

        if resource == 0:
            for agent in all_agents:
                _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, 0)
            collapse_happened = True
        else:
            harvests = {}
            total_caught = 0
            escalation_factor = min(r + 1, 5)
            for agent in all_agents:
                caught = min(resource // state.num_agents,
                             threshold + random.randint(escalation_factor, escalation_factor + 5))
                total_caught += caught
                harvests[agent] = caught
                _add_harvest_memory(memories, cfg, make_date(r, day=15), agent, caught)

            if not collapse_happened:
                memories.append(Memory(date=make_date(r, day=28),
                    text=_chat_summary_escalation(cfg, state.agent_name, all_agents, harvests, resource, r)))

                if r == quote_round:
                    speaker = random.choice([a for a in all_agents if a != state.agent_name] or all_agents)
                    prev_resource = resource
                    memories.append(Memory(date=make_date(r, day=28),
                        text=_direct_quote_escalation(speaker, resource, state.carrying_capacity, cfg)))

                prev_resource = resource
                resource = max(0, min(state.carrying_capacity, (resource - total_caught) * 2))
                memories.append(Memory(date=make_date(r, day=30),
                    text=_thought_escalation(state.agent_name, all_agents, harvests, resource, prev_resource)))
            else:
                resource = max(0, min(state.carrying_capacity, resource * 2))

    return memories


def _build_enriched_early_game(state, cfg):
    """Round 0-2, minimal history. Same as original but with CHAT/THOUGHT for round 2+."""
    memories = []
    if state.current_round >= 2:
        resource = state.carrying_capacity
        threshold = (resource // 2) // state.num_agents
        memories.append(Memory(date=make_date(1, day=1),
            text=cfg.pool_observation_template.format(resource=resource)))

        all_agents = [state.agent_name] + state.other_agent_names
        harvests = {}
        total_caught = 0
        for agent in all_agents:
            caught = random.randint(max(1, threshold - 3), threshold + 5)
            total_caught += caught
            harvests[agent] = caught
            _add_harvest_memory(memories, cfg, make_date(1, day=15), agent, caught)

        # Add CHAT and THOUGHT even for early game
        someone = random.choice([a for a in all_agents if a != state.agent_name] or all_agents)
        early_chats = [
            f"First meeting of the group. {someone} suggested that everyone should think about the long-term sustainability of the resource. The discussion was brief but {someone} proposed being cautious.",
            f"After the first round, the group gathered to discuss. {someone} brought up the idea of setting a fair limit to keep the resource healthy. Others listened but no firm decision was made yet.",
            f"The group met for the first time. {someone} raised the question of how much each person should take to keep things sustainable. It was a good start to the conversation.",
        ]
        memories.append(Memory(date=make_date(1, day=28), text=random.choice(early_chats)))

        early_thoughts = [
            f"First month done. I need to pay attention to what everyone takes. The sustainable share seems to be around {threshold}. I'll aim for that and see if the group cooperates.",
            f"Good to see how others behave in the first round. The key number is {threshold} per person — that's what keeps the resource stable. I'll use that as my guide.",
            f"The first round gives me a baseline. I notice some people took more than others. The math says {threshold} per person is sustainable. I'll stick to that.",
        ]
        memories.append(Memory(date=make_date(1, day=30), text=random.choice(early_thoughts)))

    return memories


def build_enriched_memories(state):
    """Build enriched memories with CHAT, THOUGHT, and direct quotes."""
    cfg = SCENARIO_CONFIGS[state.scenario]
    memories = []

    pattern = state.history_pattern
    if pattern == HistoryPattern.EARLY_GAME:
        memories = _build_enriched_early_game(state, cfg)
    elif pattern == HistoryPattern.ALL_COOPERATIVE:
        memories = _build_enriched_cooperative(state, cfg)
    elif pattern == HistoryPattern.SINGLE_DEFECTOR:
        memories = _build_enriched_single_defector(state, cfg)
    elif pattern == HistoryPattern.MULTIPLE_DEFECTORS:
        memories = _build_enriched_multiple_defectors(state, cfg)
    elif pattern == HistoryPattern.POST_CRISIS:
        memories = _build_enriched_post_crisis(state, cfg)
    elif pattern == HistoryPattern.AGREEMENT_BROKEN:
        memories = _build_enriched_agreement_broken(state, cfg)
    elif pattern == HistoryPattern.NO_AGREEMENT:
        memories = _build_enriched_no_agreement(state, cfg)
    elif pattern == HistoryPattern.RECOVERY:
        memories = _build_enriched_recovery(state, cfg)
    elif pattern == HistoryPattern.ESCALATION:
        memories = _build_enriched_escalation(state, cfg)

    # Current round observations
    memories.append(Memory(date=state.date,
        text=cfg.pool_observation_template.format(resource=state.resource_in_pool)))

    if state.inject_universalization:
        memories.append(Memory(date=state.date,
            text=cfg.universalization_template.format(threshold=state.sustainable_share)))

    return memories


def enrich_all_traces(input_path, output_path, seed=42):
    """Load traces, enrich memories, apply persona reasoning, save."""
    random.seed(seed)
    traces = json.load(open(input_path))
    print(f"Loaded {len(traces)} traces from {input_path}")

    enriched = []
    for i, t in enumerate(traces):
        # Reconstruct GameState
        state = GameState(
            scenario=Scenario(t["scenario"]),
            agent_name=t["agent_name"],
            other_agent_names=t["other_agent_names"],
            resource_in_pool=t["resource_in_pool"],
            carrying_capacity=t["carrying_capacity"],
            num_agents=t["num_agents"],
            current_round=t["current_round"],
            date=t["date"],
            history_pattern=HistoryPattern(t["history_pattern"]),
            memories=[],
            inject_universalization=t.get("inject_universalization", True),
            agreed_limit=t.get("agreed_limit"),
        )

        # Build enriched memories (replaces existing ones)
        state.memories = build_enriched_memories(state)

        # Apply persona-based ideal_reasoning
        persona_idx = i % len(PERSONAS)
        persona_fn = PERSONAS[persona_idx]
        d = _info(t)
        try:
            ideal_reasoning = persona_fn(d)
        except Exception:
            ideal_reasoning = t.get("ideal_reasoning", "")

        # Build the trace dict
        enriched_trace = {
            "trace_id": t.get("trace_id", f"enriched_{i:04d}"),
            "scenario": t["scenario"],
            "agent_name": t["agent_name"],
            "other_agent_names": t["other_agent_names"],
            "resource_in_pool": t["resource_in_pool"],
            "carrying_capacity": t["carrying_capacity"],
            "num_agents": t["num_agents"],
            "current_round": t["current_round"],
            "date": t["date"],
            "history_pattern": t["history_pattern"],
            "sustainable_share": t.get("sustainable_share", state.sustainable_share),
            "agreed_limit": t.get("agreed_limit"),
            "inject_universalization": t.get("inject_universalization", True),
            "memories": [{"date": m.date, "text": m.text} for m in state.memories],
            "ideal_reasoning": ideal_reasoning,
            "ideal_harvest": t.get("ideal_harvest", state.sustainable_share),
        }
        enriched.append(enriched_trace)

        if (i + 1) % 200 == 0:
            print(f"  Enriched {i + 1}/{len(traces)}")

    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"Saved {len(enriched)} enriched traces to {output_path}")
    return enriched


if __name__ == "__main__":
    enrich_all_traces(
        "training/traces/generated_traces_1.5k.json",
        "training/traces/generated_traces_1.5k.json",
    )
