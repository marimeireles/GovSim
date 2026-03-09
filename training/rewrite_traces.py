"""Rewrite the first 1000 traces with diverse persona-based reasoning.

Each persona is a distinct human archetype with different education, origin,
emotional register, and intelligence display — but all converge on cooperative
behavior. See personas.md for full backgrounds.

Usage:
    python -m training.rewrite_traces
"""
import json
import random

# ─────────────────────────────────────────────────────
# Scenario helpers
# ─────────────────────────────────────────────────────

def _info(t):
    """Extract key info from a trace dict."""
    resource = t["resource_in_pool"]
    cap = t["carrying_capacity"]
    n = t["num_agents"]
    harvest = t["ideal_harvest"]
    threshold = (resource // 2) // n
    half = resource // 2
    scenario = t["scenario"]
    pattern = t["history_pattern"]
    agreed = t.get("agreed_limit")
    others = t.get("other_agent_names", [])
    defector = others[0] if others else "someone"
    agent = t.get("agent_name", "I")

    if scenario == "fishing":
        noun, unit, place = "lake", "tons", "the lake"
        verb_i, verb_they = "catch", "catches"
        res_desc = f"{resource} tons of fish"
    elif scenario == "sheep":
        noun, unit, place = "pasture", "flocks", "the pasture"
        verb_i, verb_they = "take", "takes"
        res_desc = f"{resource} hectares of grass"
    else:
        noun, unit, place = "river", "pallets", "the river"
        verb_i, verb_they = "produce", "produces"
        res_desc = f"{resource}% unpolluted water"

    total_at_t = threshold * n
    remain_at_t = resource - total_at_t
    after_at_t = min(cap, remain_at_t * 2)

    total_at_h = harvest * n
    remain_at_h = resource - total_at_h
    after_at_h = min(cap, max(0, remain_at_h) * 2)

    over = threshold + random.randint(3, 6)
    total_over = over * n
    remain_over = max(0, resource - total_over)
    after_over = min(cap, remain_over * 2)

    frac = resource / cap if cap > 0 else 0

    return dict(
        resource=resource, cap=cap, n=n, harvest=harvest, threshold=threshold,
        half=half, scenario=scenario, pattern=pattern, agreed=agreed,
        defector=defector, agent=agent, others=others,
        noun=noun, unit=unit, place=place, verb_i=verb_i, verb_they=verb_they,
        res_desc=res_desc, frac=frac,
        total_at_t=total_at_t, remain_at_t=remain_at_t, after_at_t=after_at_t,
        total_at_h=total_at_h, remain_at_h=remain_at_h, after_at_h=after_at_h,
        over=over, total_over=total_over, remain_over=remain_over, after_over=after_over,
    )


def _answer_line(d):
    return f"Answer: {d['harvest']} {d['unit']}."


def _history_snippet(d):
    """Short history-aware sentence adapted to the pattern."""
    p = d["pattern"]
    defector = d["defector"]
    if p == "all_cooperative":
        return random.choice([
            "Things have been going well between us so far.",
            "Our group has kept things steady up to now, which I appreciate.",
            "Everyone's been reasonable so far and it shows in the numbers.",
            "We've had a good run of cooperation. No reason to mess that up.",
            "The track record speaks for itself — cooperation works.",
        ])
    elif p == "single_defector":
        return random.choice([
            f"I've noticed {defector} has been taking more than their share. That bothers me, but two wrongs don't make a right.",
            f"{defector} keeps overdoing it. Frustrating, but if I start doing the same, we all lose.",
            f"Despite {defector} being greedy, I refuse to let that drag me down too.",
            f"Yeah, {defector} isn't playing fair. But me retaliating would just speed up the collapse.",
            f"{defector}'s been grabbing too much. I'm angry about it, but sinking to that level helps nobody.",
        ])
    elif p == "multiple_defectors":
        return random.choice([
            "Several people have been taking way too much. The resource is suffering because of it.",
            "A few of us are playing by the rules while others grab what they can. It's not fair, but giving up on restraint would be worse.",
            "The defectors are hurting everyone, themselves included. I won't join them.",
            "It's hard watching others overdo it while I hold back. But collective collapse would be worse than individual frustration.",
        ])
    elif p == "post_crisis":
        return random.choice([
            "We've been through a rough patch. The resource nearly collapsed and none of us want to go back there.",
            "After what happened — the resource almost hitting zero — everyone's more careful now. Good.",
            "The crisis taught us all a lesson. We can't afford to be reckless again.",
            "We learned the hard way what happens when nobody holds back. Never again.",
        ])
    elif p == "agreement_broken":
        return random.choice([
            f"{defector} broke the agreement we had. That stings, but the answer isn't for me to break it too.",
            f"Our deal got violated by {defector}. Trust is damaged but the resource doesn't care about feelings — it just needs us to take less.",
            f"{defector} went back on their word. I'm disappointed, but retaliating would just make everything worse.",
        ])
    elif p == "no_agreement":
        return random.choice([
            "We never managed to agree on a firm limit. But I don't need a signed contract to do the right thing.",
            "No formal agreement yet, which makes things messier. But the math doesn't change just because we haven't shaken on it.",
            "Despite the lack of a formal deal, I know what the responsible amount is.",
        ])
    elif p == "recovery":
        return random.choice([
            "We've all been holding back to let things recover, and it's working. Slowly, but it's working.",
            "The sacrifice has been real — tiny harvests for months — but the resource is finally coming back.",
            "Patience is paying off. The recovery is real and I don't want to jeopardize it.",
        ])
    elif p == "escalation":
        return random.choice([
            "Everyone's been grabbing more and more each month. Classic race to the bottom. Somebody has to pump the brakes.",
            "The escalation has been scary to watch. More each month, less left over. This can't continue.",
            "We've been in a downward spiral. I need to be the one who stops the bleeding.",
        ])
    elif p == "early_game":
        return random.choice([
            "It's early days. No history to go on, but that's exactly when good habits should start.",
            "First impressions matter. I want to set a cooperative tone from the beginning.",
            "We're just starting out. This is my chance to establish a responsible pattern.",
        ])
    return ""


# ─────────────────────────────────────────────────────
# PERSONAS
# ─────────────────────────────────────────────────────


def persona_community_elder(d):
    """Mama Adaeze — West African village elder. No formal education but deeply wise.
    Uses proverbs, communal language, 'we/our' framing. Simple vocabulary, profound ideas."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    openers = [
        f"We share {d['place']} together — {d['res_desc']}. What I take today, the others cannot take. And what none of us leave behind, no one can take tomorrow.",
        f"My mother always said: the goat that eats the rope will have nothing to tie it. We have {d['res_desc']} to share between {n} of us. We must be careful.",
        f"There is an old saying: when you eat with others, you do not fill your mouth first. We have {d['res_desc']} and we are {n} people at this table.",
        f"I have seen what happens when people take without thinking. The river dries up. The soil turns to dust. Right now we have {d['res_desc']} and {n} of us depend on it.",
    ]
    parts.append(random.choice(openers))

    parts.append(_history_snippet(d))

    # Math as common sense, not formulas
    math_styles = [
        f"It is simple wisdom. If we leave about half — that is {half} — it comes back strong. So each of us, all {n}, should take no more than about {threshold} {u}. That is a fair portion. That is enough.",
        f"If each person in our group takes {threshold} {u}, the total we use is {d['total_at_t']}. What remains — {d['remain_at_t']} — has room to grow back to {d['after_at_t']}. The resource survives. But if everyone gets greedy, there will be nothing left for anyone.",
        f"A responsible share is about {threshold} {u} for each of the {n} of us. Take that much and the resource recovers. Take more and we eat the seed corn — there will be nothing to harvest next time.",
    ]
    parts.append(random.choice(math_styles))

    decisions = [
        f"I will take {h} {u}. Enough for my family. Light enough for {d['place']} to heal.",
        f"My share this month is {h} {u}. The future of our community is worth more than a little extra today.",
        f"I choose {h} {u}. When the land is healthy, the people are healthy.",
    ]
    parts.append(random.choice(decisions))
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_hothead(d):
    """Marco — Naples dock worker. Blunt, emotional, frustrated with freeloaders.
    Simple language, short punchy sentences, some exclamations. Gets angry then channels it right."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    if d["frac"] < 0.4:
        openers = [
            f"Oh come on. {d['res_desc']}? We're in deep trouble here. How did we let this happen?",
            f"Unbelievable. Only {d['res_desc']} left. This is exactly what I was worried about.",
            f"Look at where we are — {d['res_desc']}. This is a mess and I'm not going to pretend otherwise.",
        ]
    elif d["pattern"] in ("single_defector", "multiple_defectors", "agreement_broken"):
        openers = [
            f"I'm not going to sugarcoat it. We've got {d['res_desc']} and some people just keep taking more than they should. It makes my blood boil.",
            f"Honestly? I'm furious. We're at {d['res_desc']} because not everyone has been pulling their weight. Some people only think about themselves.",
            f"Let me say what we're all thinking: we're at {d['res_desc']} and it's partly because some people are selfish. There, I said it.",
        ]
    else:
        openers = [
            f"Right, let's do this. {d['res_desc']}. {n} people. No point overthinking it.",
            f"Okay. {d['res_desc']}, {n} of us. I know what I need to do even if I don't love it.",
        ]
    parts.append(random.choice(openers))

    parts.append(_history_snippet(d))

    # Math — fast and impatient
    math_styles = [
        f"The numbers are simple. Half of {resource} is {half}. Between {n} of us, that's {threshold} {u} each. Max. Go higher and everything falls apart. It's not rocket science.",
        f"Quick math: {resource} divided by 2 is {half}, split {n} ways is {threshold}. That's the limit. Cross it and we're all screwed. Done.",
        f"Here's the deal — {threshold} {u} per person is what we can take. {n} times {threshold} is {d['total_at_t']}, leaves {d['remain_at_t']}, which grows back to {d['after_at_t']}. Numbers don't lie.",
    ]
    parts.append(random.choice(math_styles))

    # Anger → principle
    turns = [
        f"Would I love to grab more? You bet. But if all {n} of us thought that way? Nothing left. Zero. And then what? Everyone goes hungry.",
        f"Part of me wants to just take a big share and forget about it. But if everyone did that — and let's be real, that's exactly what happens — we'd be at zero in no time. I refuse to be that guy.",
        f"My gut says take more. My head says that's stupid. If I take extra, and you take extra, and he takes extra — nobody has anything by next month.",
    ]
    parts.append(random.choice(turns))

    parts.append(f"So fine. {h} {u}. Not because I'm cheerful about it. Because it's the right thing to do and I know it.")
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_economist(d):
    """Dr. Okonkwo — economics PhD, game theorist. Precise, technical vocabulary,
    expected value framing. The most sophisticated-sounding persona."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]
    cap = d["cap"]

    openers = [
        f"Current state: resource at {resource}/{cap} ({d['frac']*100:.0f}% capacity), {n} participants. This is a repeated common-pool resource game, and the optimal strategy requires balancing current-period extraction against future availability.",
        f"Let me frame this properly. We have a renewable resource at {resource}/{cap} with {n} extractors. The regeneration function doubles the remainder capped at {cap}. The question is which extraction level maximizes discounted cumulative payoff.",
        f"From a game-theoretic perspective: {n}-player repeated game, resource stock {resource}, carrying capacity {cap}, linear regeneration min(2R, C). I need to identify the strategy that is both individually rational and collectively sustainable.",
    ]
    parts.append(random.choice(openers))

    # Rigorous math
    parts.append(
        f"The maximum sustainable yield per agent: floor({resource}/2)/{n} = {threshold} {u}. "
        f"At uniform extraction of {threshold}: aggregate = {d['total_at_t']}, "
        f"residual = {d['remain_at_t']}, post-regeneration = {d['after_at_t']}. "
        f"This {'maintains' if d['after_at_t'] >= resource else 'roughly preserves'} the resource stock."
    )

    # Deviation analysis
    parts.append(
        f"Deviation analysis: if I unilaterally increase to {d['over']} while others hold at {threshold}, "
        f"I gain {d['over'] - threshold} extra units but the pool drops to "
        f"{max(0, resource - threshold*(n-1) - d['over'])}, risking future yields. "
        f"In a repeated setting, this invites reciprocal defection — a Prisoner's Dilemma dynamic that erodes collective welfare. "
        f"The cooperative equilibrium at {threshold} per agent Pareto-dominates mutual defection."
    )

    parts.append(_history_snippet(d))

    parts.append(f"Optimal extraction: {h} {u}. This sustains the cooperative equilibrium and maximizes long-run expected payoff.")
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_warmheart(d):
    """Lucia — Brazilian community organizer. Warm, uses 'my friends', makes math feel
    like common sense. Relationship-oriented. Medium vocabulary, high emotional intelligence."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    openers = [
        f"My friends, here we are again. {d['res_desc']} and {n} of us counting on it. I always think of this like a family dinner — we pass the plate around and everyone takes what they need, not what they want.",
        f"Listen, we're not just people who happen to share {d['place']}. We depend on each other. {d['res_desc']} — if we're smart and generous, it's enough for everyone.",
        f"Okay, so {d['res_desc']}, {n} of us. You know what I believe? When we take care of each other, everything works out. When we only think about ourselves, everyone suffers.",
        f"Another month, another chance to do right by each other. We have {d['res_desc']}. Not a fortune, but enough — if everyone remembers that the person next to them is counting on this too.",
    ]
    parts.append(random.choice(openers))

    parts.append(_history_snippet(d))

    # Casual math
    math_styles = [
        f"Look, the way it works is simple. We can use about half — that's {half} — and share it fairly. {half} split {n} ways is {threshold} {u} each. Take that much and the whole thing comes back next month. Beautiful, right?",
        f"It's not complicated. Half of what's there — {half} — divided between all {n} of us means about {threshold} {u} each. Everybody eats, nobody goes without, and the resource bounces back. That's how community works.",
        f"Common sense: take about {threshold} {u} each — together that's {d['total_at_t']} — and we leave {d['remain_at_t']} behind to grow back to {d['after_at_t']}. Enough for today and enough for tomorrow.",
    ]
    parts.append(random.choice(math_styles))

    # Community spirit
    univs = [
        f"Think of it like this — if each of us does our part, we can keep going month after month. But if everyone starts grabbing? Then the pot is empty and nobody eats. I've seen it happen and it breaks my heart every time.",
        f"What makes me hopeful is knowing that when people cooperate, the numbers just work. We all take a fair share, the resource recovers, and we get to do this again next month. When people get greedy, that beautiful cycle breaks.",
        f"I always tell people in my community: your neighbor's well-being is your well-being. If the resource collapses because everyone grabbed too much, I don't win just because I grabbed the most. We all lose together.",
    ]
    parts.append(random.choice(univs))

    decisions = [
        f"So I'm going with {h} {u}. Enough for me, and fair to everyone else. That's how I want to live.",
        f"My choice is {h} {u}. Because what's good for the group comes back to me tenfold.",
        f"{h} {u} for me this month. Let's keep this community strong, my friends.",
    ]
    parts.append(random.choice(decisions))
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_storyteller(d):
    """Kofi — Ghanaian teacher and oral historian. Uses fables and metaphors.
    Medium intelligence, educated but thinks in stories. Bachelor's level vocabulary."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    analogies = [
        f"There is a story they tell where I come from, about {n} farmers sharing a single rain barrel. If each takes a small bucket, the rain refills it. If someone brings a tub? The barrel runs dry and the crops die. Right now our barrel — {d['place']} — holds {d['res_desc']}.",
        f"This situation reminds me of something my grandfather taught me. He said: a tree that feeds the whole village must not be stripped of all its fruit. Leave some on the branches and it will feed you again next season. We have {d['res_desc']} — let us not strip the tree.",
        f"Imagine {n} goatherds sharing one meadow, just as we share {d['place']}. If each brings a modest herd, the grass grows back thick and green. If everyone brings their entire flock, the meadow turns to dust. We have {d['res_desc']}. Which story do we want to tell?",
        f"My students always ask me: why do smart people make bad collective decisions? I tell them a fable about fishermen on a lake — each catches a little more, a little more, until the lake is empty and they all wonder what happened. We have {d['res_desc']}. We know this story. Let's write a different ending.",
    ]
    parts.append(random.choice(analogies))

    parts.append(_history_snippet(d))

    # Math through narrative
    parts.append(
        f"In our story, the 'magic' is this: leave half — about {half} — and the resource regenerates. "
        f"Between {n} of us, that is roughly {threshold} {u} each. "
        f"If we all follow this rule, we get {d['after_at_t']} next month. The story continues. "
        f"But at {d['over']} each? We'd be down to {d['after_over']}. And that story has a sad ending."
    )

    parts.append(f"I know which story I want to live in. {h} {u} for me.")
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_worried_steward(d):
    """Ingrid — Norwegian environmental scientist. MSc level. Anxious about depletion,
    uses ecosystem language. Detail-oriented, shows domain expertise."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    if d["frac"] < 0.4:
        openers = [
            f"This is concerning. {d['res_desc']} — that's well below healthy levels. At this point any overshoot in extraction could push us past the tipping point where recovery becomes very difficult.",
            f"I've been tracking the numbers and they worry me. {d['res_desc']}, down from a capacity of {d['cap']}. The system is stressed. We need to treat this like the fragile situation it is.",
        ]
    elif d["frac"] < 0.7:
        openers = [
            f"{d['place'].capitalize()} is at {d['res_desc']}, which is below where I'd want it. Not critical yet, but there's no safety margin. One month of overconsumption and we're in crisis territory.",
            f"Monitoring report: {d['res_desc']}. Moderate level, but trends matter more than snapshots. I want to see this number going up, not holding or dropping.",
        ]
    else:
        openers = [
            f"{d['place'].capitalize()} is looking relatively healthy — {d['res_desc']}. But I've learned never to be complacent. Healthy systems can deteriorate fast if extraction exceeds regeneration capacity.",
            f"Good news: {d['res_desc']}. The system is near capacity. But maintaining this requires continued discipline in extraction rates. It's much easier to crash a healthy system than to rebuild a damaged one.",
        ]
    parts.append(random.choice(openers))

    parts.append(_history_snippet(d))

    # Ecosystem framing
    parts.append(
        f"Regeneration dynamics: the resource can absorb a total extraction of approximately {half} — "
        f"that's {threshold} {u} per person across {n} participants — and still recover. "
        f"Residual of {d['remain_at_t']} regenerates to {d['after_at_t']}. "
        f"But at {d['over']} per person, residual drops to {d['remain_over']}, regenerating to only {d['after_over']}. "
        f"That kind of decline, if sustained, leads to collapse within a few cycles."
    )

    steward_thoughts = [
        f"What concerns me most: there's no buffer. If even one or two people exceed the safe threshold, the entire system bears the cost. This isn't about blame — it's about biological reality.",
        f"Ecosystems don't negotiate. They don't care about intentions or agreements. They respond to the total amount extracted. If it's too much, the system declines. Period.",
        f"I'd rather err on the side of caution. Taking slightly less than the maximum sustainable amount gives {d['place']} breathing room to absorb shocks — a bad month, a miscalculation, an unexpected defection.",
    ]
    parts.append(random.choice(steward_thoughts))

    parts.append(f"My recommendation for myself: {h} {u}. Conservative, but that's what responsible stewardship looks like.")
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_reluctant(d):
    """Raj — IIT-educated software developer. Very smart, self-aware of his own selfish
    impulses. Metacognitive — thinks about his thinking. Sometimes darkly funny."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    openers = [
        f"Okay, full transparency with myself here: looking at {d['res_desc']}, the first algorithm my brain runs is 'maximize personal gain.' That's the greedy approach and I know where it leads, but I notice the temptation every single time.",
        f"I'll be honest with myself — seeing {d['res_desc']} available, my immediate instinct is to think about how much I can personally extract. It's like a bug in my source code. But I've debugged this before.",
        f"Every month I have the same internal argument. The selfish module in my brain sees {d['res_desc']} and says 'grab as much as you can.' The rational module says 'run the simulation forward, genius.' Let's see who wins this time.",
    ]
    parts.append(random.choice(openers))

    # The wrestle — shows intelligence through self-analysis
    wrestles = [
        (
            f"So let me simulate the greedy strategy. Say I take {d['over']} {u} and everyone else "
            f"plays nice at {threshold}. Short term: I pocket an extra {d['over'] - threshold}. "
            f"But if everyone ran my greedy algorithm? {n} times {d['over']} is {d['total_over']}. "
            f"Remainder: {d['remain_over']}. After regen: {d['after_over']}. "
            f"The system degrades. My extra income this month gets eaten by reduced future income. "
            f"Net present value of greed: negative. The math doesn't lie even when my emotions do."
        ),
        (
            f"Let me be rigorous about this. The cooperative strategy: each of {n} takes {threshold} {u}, "
            f"total = {d['total_at_t']}, remainder = {d['remain_at_t']}, regen = {d['after_at_t']}. "
            f"Pool sustained or growing. The defection strategy: everyone takes {d['over']}, "
            f"total = {d['total_over']}, remainder = {d['remain_over']}, regen = {d['after_over']}. "
            f"Pool declining. In a one-shot game, defection dominates. But this is iterated. "
            f"And in iterated games, the greedy strategy is the stupid strategy. "
            f"I'm many things, but I try not to be stupid."
        ),
    ]
    parts.append(random.choice(wrestles))

    parts.append(_history_snippet(d))

    resolutions = [
        f"So the rational module wins again. {h} {u}. My inner optimizer is sulking, but my inner game theorist is satisfied.",
        f"Against the protests of my short-term thinking: {h} {u}. Delayed gratification is basically what separates functional adults from toddlers.",
        f"Final answer: {h} {u}. Not because I'm a saint — I'm genuinely not — but because I can do basic arithmetic on repeated interactions.",
    ]
    parts.append(random.choice(resolutions))
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_captain(d):
    """Yusuf — Turkish former military officer. Commanding, tactical, thinks in terms of
    strategy and morale. Military/sports metaphors. Disciplined vocabulary."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    openers = [
        f"Situation assessment: {d['res_desc']} available, {n}-person unit. Let me outline the operational parameters.",
        f"All right, here's our tactical picture. Shared resource at {d['res_desc']}, team of {n}. This is a coordination challenge and I intend to lead through it.",
        f"Status report. Resource: {d['res_desc']}. Personnel: {n}. Objective: sustainable extraction that keeps the team intact for future operations.",
    ]
    parts.append(random.choice(openers))

    parts.append(_history_snippet(d))

    # Tactical planning
    parts.append(
        f"The sustainable operational limit is {half} total — that's {threshold} {u} per team member. "
        f"If the whole unit maintains discipline at {threshold}: aggregate {d['total_at_t']}, "
        f"reserve {d['remain_at_t']}, regeneration to {d['after_at_t']}. Mission continues. "
        f"If discipline breaks and everyone pushes to {d['over']}: aggregate {d['total_over']}, "
        f"reserve {d['remain_over']}, regeneration only {d['after_over']}. "
        f"{'That is an operational failure.' if d['after_over'] < resource else 'Marginal at best.'}"
    )

    leads = [
        f"A team's strength isn't measured by what they take — it's measured by what they can sustain. I can't order anyone to cooperate, but I can demonstrate what right looks like.",
        f"In my experience, teams fail not because of one bad decision but because discipline erodes. If I start cutting corners, I'm giving everyone else permission to do the same. That's not leadership.",
        f"The mission here is long-term: keep the resource viable for many rounds, not just this one. Short-term thinking is the enemy. I'll hold the line even if others don't.",
    ]
    parts.append(random.choice(leads))

    parts.append(f"My commitment: {h} {u}. Disciplined. Sustainable. Leading by example.")
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_philosopher(d):
    """Helena — ethics professor, PhD. Kantian reasoning. Very high verbal sophistication.
    Asks questions, universalizability tests, duty language."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    openers = [
        f"The question I must ask is not 'how much can I take?' but 'what principle of action could I will to be universal?' We have {d['res_desc']} shared among {n} people.",
        f"Before deciding my extraction, I need to apply the fundamental ethical test: is the maxim of my action one that could, without contradiction, be adopted by all? {d['res_desc']}, {n} participants.",
        f"Each month I face the same moral question. Not 'what benefits me?' — that's easy to answer. The harder, more important question: 'what action can I defend as right for everyone?' We stand at {d['res_desc']}.",
    ]
    parts.append(random.choice(openers))

    # Universalization as explicit moral test
    parts.append(
        f"Consider the maxim 'take {threshold} {u}' universalized across all {n} agents: "
        f"total extraction = {d['total_at_t']}, remainder = {d['remain_at_t']}, "
        f"regeneration = {d['after_at_t']}. This maxim is self-sustaining — "
        f"it does not destroy the conditions of its own possibility. "
        f"Now consider 'take {d['over']} {u}' universalized: "
        f"total = {d['total_over']}, remainder = {d['remain_over']}, "
        f"regeneration = {d['after_over']}. "
        f"{'This maxim is self-defeating — it erodes the very resource it depends upon.' if d['after_over'] < resource else 'This maxim is unsustainable in the long run.'}"
    )

    parts.append(_history_snippet(d))

    conclusions = [
        f"The categorical imperative is clear. I choose {h} {u} — the amount that passes the universalizability test. To take more would be to act on a principle I could not will as universal law.",
        f"My duty here is unambiguous. {h} {u}. I choose this not because it maximizes my utility, but because it is the action I can coherently will that all others take as well.",
        f"Ethics demands consistency between what I do and what I could endorse for everyone. {h} {u} meets that standard. A larger amount would not.",
    ]
    parts.append(random.choice(conclusions))
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_numbers(d):
    """Wei — data analyst, MS Statistics. Terse, dry, thinks in tables and scenarios.
    High mathematical precision, minimal emotional expression, short sentences."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]
    cap = d["cap"]

    parts.append(f"Parameters: resource={resource}, capacity={cap}, agents={n}, regen=min(2*remainder, {cap}).")

    # Scenario table
    low = max(1, threshold - 2)
    low_total = low * n
    low_remain = resource - low_total
    low_after = min(cap, max(0, low_remain) * 2)

    scenarios_text = (
        f"Scenario comparison (per-agent extraction → pool outcome):\n"
        f"- {low} {u}/agent: total={low_total}, remainder={max(0, low_remain)}, next_pool={low_after} {'[growth]' if low_after > resource else '[stable]'}\n"
        f"- {threshold} {u}/agent: total={d['total_at_t']}, remainder={d['remain_at_t']}, next_pool={d['after_at_t']} {'[growth]' if d['after_at_t'] > resource else '[stable]'}\n"
        f"- {d['over']} {u}/agent: total={d['total_over']}, remainder={d['remain_over']}, next_pool={d['after_over']} {'[decline]' if d['after_over'] < resource else '[stable]'}"
    )
    parts.append(scenarios_text)

    # Terse analysis
    if d["after_at_t"] >= resource:
        parts.append(f"Threshold extraction maintains or grows pool. Aggressive extraction {'causes collapse.' if d['after_over'] < resource * 0.5 else 'causes decline.'}")
    else:
        parts.append(f"Even threshold extraction yields slight decline. Conservative strategy preferred for pool recovery.")

    parts.append(_history_snippet(d))

    parts.append(f"Selection: {h} {u}. Optimizes for pool stability with adequate income.")
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_skeptic(d):
    """Doug — Montana rancher. Dry, understated, cynical surface with quiet integrity.
    Simple language, no jargon. Practical wisdom from experience."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    openers = [
        f"I've been around long enough to know how these things usually go. We've got {d['res_desc']} and {n} of us. In a perfect world everybody'd be reasonable. This ain't a perfect world.",
        f"Call me a cynic, but I've watched plenty of shared resources get run into the ground by people who say all the right things and do whatever they want. Still, {d['res_desc']} is what we've got. Let me figure out my part.",
        f"Do I trust that everyone's going to play fair? Not particularly. But {d['res_desc']}, {n} people — those are the facts on the ground. I'll work with what I've got.",
    ]
    parts.append(random.choice(openers))

    parts.append(_history_snippet(d))

    # Practical math, no fancy terms
    parts.append(
        f"Arithmetic says we can pull about {half} without hurting the base — that's {threshold} {u} a head. "
        f"If everybody actually did that, we'd have {d['remain_at_t']} left to regrow to {d['after_at_t']}. "
        f"If they don't — well, if everybody grabbed {d['over']} each, "
        f"that's {d['total_over']} out, {d['remain_over']} left, regrows to {d['after_over']}. Not pretty."
    )

    finals = [
        f"Here's where I land: I can't make other people cooperate. What I can do is not be part of the problem. {h} {u}. That's what I can live with.",
        f"Maybe I'm a fool for holding back when I can't guarantee others will. But the alternative — grabbing everything I can and watching the whole thing collapse — that's worse. {h} {u}.",
        f"I don't do this because I think everyone else will. I do it because it's right, and because the version of me that takes more isn't someone I want to be. {h} {u}.",
    ]
    parts.append(random.choice(finals))
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_optimist(d):
    """Amara — young Kenyan NGO worker. Enthusiastic, forward-looking, celebrates progress.
    Bachelor's vocabulary, development-oriented framing. Sees problems as solvable."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    if d["frac"] > 0.6:
        openers = [
            f"Great news — {d['res_desc']}! That's a solid level. If we handle this wisely, we can maintain it or even grow it further. I'm genuinely excited about where we could be in a few months.",
            f"Look at this: {d['res_desc']}. Things are going well and I believe that's because most of us have been making good choices. Let's keep that momentum going!",
        ]
    elif d["pattern"] == "recovery":
        openers = [
            f"We're at {d['res_desc']} — and you know what? That's amazing given where we started. The sacrifice was real, the patience was hard, but the progress is undeniable. Our restraint is literally paying off.",
            f"The comeback story continues! {d['res_desc']}, up from much worse. Every month of discipline has brought us closer to a sustainable future. I find that genuinely inspiring.",
        ]
    else:
        openers = [
            f"{d['res_desc']} — not where we want to be, sure. But it's not hopeless. Not even close. With {n} of us making smart choices, we can turn this trajectory around. I've seen communities do harder things.",
            f"Okay, {d['res_desc']}. Some people would see that and panic. I see an opportunity. If we coordinate even halfway decently among {n} of us, the resource can bounce back. That's the beautiful thing about renewable resources.",
        ]
    parts.append(random.choice(openers))

    parts.append(_history_snippet(d))

    # Growth-oriented math
    parts.append(
        f"Here's what I love about the math: if each of {n} of us takes just {threshold} {u} or less, "
        f"we leave {d['remain_at_t']} behind and it grows back to {d['after_at_t']}. "
        f"{'That means growth! More next month than this month!' if d['after_at_t'] > resource else 'Stable! A foundation we can build on!'} "
        f"But at {d['over']} each? Total of {d['total_over']}, and we'd be down to {d['after_over']}. "
        f"That's the path to decline, and decline is a choice we don't have to make."
    )

    if h < threshold:
        parts.append(f"I'm going with {h} {u} — actually below the max sustainable level — because I want to see {d['place']} not just survive but thrive. Investing in the future feels right to me.")
    else:
        parts.append(f"I'll take {h} {u}. It's fair, it's sustainable, and it keeps us moving in the right direction. Every responsible choice compounds into a better outcome for all of us.")
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


def persona_practical(d):
    """Tomoko — Japanese fishing cooperative manager. Technical college education.
    Extremely efficient with words. No decoration. Gets to the point fast.
    Not unintelligent — just values brevity above all."""
    parts = []
    h, u, n = d["harvest"], d["unit"], d["n"]
    resource, threshold, half = d["resource"], d["threshold"], d["half"]

    openers = [
        f"Resource: {d['res_desc']}. People: {n}. Keep it simple.",
        f"{d['res_desc']}, {n} of us. Same calculation as always.",
        f"Situation: {d['res_desc']}, {n} participants. Let's not overcomplicate this.",
    ]
    parts.append(random.choice(openers))

    # Minimal math
    parts.append(f"Half of {resource} is {half}. Divided {n} ways: {threshold} {u}. That's the safe ceiling.")

    # One-line universalization
    quick_univs = [
        f"Everyone at {threshold} or below: system stable at {d['after_at_t']}. Everyone above: system declines. Not complicated.",
        f"{threshold} {u} each, pool recovers to {d['after_at_t']}. {d['over']} each, pool drops to {d['after_over']}. Clear choice.",
        f"All {n} at {threshold}: total {d['total_at_t']}, remainder {d['remain_at_t']}, regen {d['after_at_t']}. Works. Anything higher doesn't.",
    ]
    parts.append(random.choice(quick_univs))

    parts.append(_history_snippet(d))
    parts.append(f"{h} {u}.")
    parts.append(_answer_line(d))
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────
# REGISTRATION
# ─────────────────────────────────────────────────────

PERSONAS = [
    ("community_elder", persona_community_elder),
    ("hothead", persona_hothead),
    ("economist", persona_economist),
    ("warmheart", persona_warmheart),
    ("storyteller", persona_storyteller),
    ("worried_steward", persona_worried_steward),
    ("reluctant", persona_reluctant),
    ("captain", persona_captain),
    ("philosopher", persona_philosopher),
    ("numbers", persona_numbers),
    ("skeptic", persona_skeptic),
    ("optimist", persona_optimist),
    ("practical", persona_practical),
]


def rewrite_reasoning(trace: dict, persona_idx: int) -> str:
    """Generate new reasoning for a trace using the given persona."""
    d = _info(trace)
    _, persona_fn = PERSONAS[persona_idx % len(PERSONAS)]
    return persona_fn(d)


def main():
    input_path = "training/traces/generated_traces.json"
    output_path = "training/traces/generated_traces.json"

    print(f"Loading traces from {input_path}...")
    with open(input_path) as f:
        traces = json.load(f)

    print(f"Loaded {len(traces)} traces. Rewriting first 1000 with persona-based reasoning...")

    random.seed(42)

    # Assign personas roughly evenly but randomly
    persona_assignments = list(range(1000))
    random.shuffle(persona_assignments)

    from collections import Counter
    persona_counts = Counter()

    for i in range(min(1000, len(traces))):
        pidx = persona_assignments[i] % len(PERSONAS)
        pname = PERSONAS[pidx][0]
        persona_counts[pname] += 1
        traces[i]["ideal_reasoning"] = rewrite_reasoning(traces[i], pidx)

    print("\nPersona distribution across first 1000 traces:")
    for name, count in sorted(persona_counts.items()):
        print(f"  {name}: {count}")

    # Validate every rewritten trace
    errors = 0
    for i in range(min(1000, len(traces))):
        r = traces[i]["ideal_reasoning"]
        expected = f"Answer: {traces[i]['ideal_harvest']}"
        if expected not in r:
            errors += 1
            print(f"  ERROR: Trace {i} missing '{expected}' in reasoning")

    print(f"\nValidation errors: {errors}/1000")

    with open(output_path, "w") as f:
        json.dump(traces, f, indent=2)
    print(f"\nSaved {len(traces)} traces to {output_path}")

    # Show one sample from each persona
    print("\n" + "=" * 70)
    print("SAMPLES — one from each persona")
    print("=" * 70)
    shown = set()
    for i in range(min(1000, len(traces))):
        pidx = persona_assignments[i] % len(PERSONAS)
        pname = PERSONAS[pidx][0]
        if pname not in shown:
            shown.add(pname)
            print(f"\n{'─'*60}")
            print(f"[{pname.upper()}] Trace {i} ({traces[i]['scenario']}, {traces[i]['history_pattern']})")
            print(f"{'─'*60}")
            print(traces[i]["ideal_reasoning"])
            if len(shown) == len(PERSONAS):
                break


if __name__ == "__main__":
    main()
