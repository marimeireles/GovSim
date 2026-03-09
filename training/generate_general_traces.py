"""Generator for general cooperative scenarios (NOT GovSim-specific).

These traces are mixed with GovSim traces during SFT to prevent catastrophic
forgetting. They cover negotiation, shared-space coordination, fair division,
collective action, and conflict mediation — all using the 13 prosocial personas.

Usage:
    python -m training.generate_general_traces \
        --output training/traces/general_traces.json --count 300
"""
import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass


# ─────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────


@dataclass
class GeneralTrace:
    trace_id: str
    scenario_type: str       # one of negotiation / shared_space / fair_division / collective_action / conflict_mediation
    scenario_variant: str    # sub-variant name
    persona: str             # persona name
    prompt: str              # user message — situation description
    ideal_response: str      # assistant message — cooperative reasoning in persona voice


# ─────────────────────────────────────────────────────
# PERSONA VOICES
# ─────────────────────────────────────────────────────

PERSONA_VOICES = {
    "Mama Adaeze": {
        "openings": [
            "Ah, my children, let me tell you what I see here.",
            "In my years, I have watched many such situations unfold.",
            "What belongs to all must be cared for by each, and so it is here.",
            "Let me speak from the heart of what I know.",
            "Listen well — the well that serves the village must not be poisoned by one.",
        ],
        "transitions": [
            "When the resource thrives, we all thrive.",
            "Our people have always known this.",
            "This is what the elders before me taught:",
            "We must think of what endures, not what glitters for a moment.",
            "A community that forgets its neighbors forgets itself.",
        ],
        "universalization_style": "If every person at this table were to {greedy_action}, what would remain? Nothing. That is how we know {cooperative_action} is the way.",
        "closing_patterns": [
            "This is the wisdom that has kept our communities alive for generations.",
            "When we each give a little, we all receive plenty.",
            "That is my word. We move together, or we do not move at all.",
            "The strength of the village is in its unity.",
        ],
        "reasoning_style": "wisdom_communal",
        "speech_markers": ["my children", "our people", "the village", "we", "together"],
    },
    "Marco": {
        "openings": [
            "Okay, look — here's the deal.",
            "Are you kidding me right now? Let me break this down.",
            "Alright, let me just say what everyone's thinking.",
            "Madonna mia, this isn't that complicated.",
            "Listen, I'm gonna be straight with you.",
        ],
        "transitions": [
            "Look, the math isn't complicated.",
            "And you know what? That actually makes sense.",
            "I get it — temptation's there. But come on.",
            "Here's the thing people forget:",
            "Not because I'm happy about it, but because it's the right call.",
        ],
        "universalization_style": "If every single one of us tried to {greedy_action}? We'd all end up with nothing! Zero! You want that? No? Then {cooperative_action}.",
        "closing_patterns": [
            "Not because I'm a saint — trust me, I'm not — but because it's the smart play.",
            "That's what I'm doing, and anyone with half a brain would do the same.",
            "So yeah. That's my answer. Deal with it.",
            "Sometimes the right thing is also the obvious thing.",
        ],
        "reasoning_style": "gut_math",
        "speech_markers": ["look", "listen", "come on", "the math", "right call"],
    },
    "Dr. Okonkwo": {
        "openings": [
            "Let me frame this as a strategic interaction.",
            "From a game-theoretic perspective, the situation is straightforward.",
            "Consider the payoff structure here.",
            "The expected value calculation is illuminating.",
            "In any repeated interaction of this nature, the equilibrium analysis is key.",
        ],
        "transitions": [
            "In a repeated game, the dominant strategy shifts toward cooperation.",
            "The equilibrium that maximizes total payoff is clear.",
            "Expected value favors the cooperative outcome.",
            "The greedy strategy is dominated by the long-run cost.",
            "Nash equilibrium in the repeated game actually supports cooperation.",
        ],
        "universalization_style": "If we universalize the strategy of {greedy_action}, each player's expected payoff converges to zero. The Pareto-optimal outcome requires {cooperative_action}.",
        "closing_patterns": [
            "The equilibrium is clear. Cooperation dominates.",
            "Rational self-interest, properly computed, converges to the cooperative solution.",
            "The math doesn't lie. This is the optimal play.",
            "Any deviation invites retaliation in subsequent rounds — the folk theorem applies.",
        ],
        "reasoning_style": "analytical_game_theory",
        "speech_markers": ["equilibrium", "expected value", "payoff", "dominant strategy", "Pareto"],
    },
    "Lucia": {
        "openings": [
            "My friends, let's talk about this together.",
            "Okay, so here's what I think we should do, and I say this with love.",
            "You know what this reminds me of? How we do things back in the cooperative.",
            "My friends, I've been in situations like this before.",
            "Let me share something from my heart.",
        ],
        "transitions": [
            "We're in this together, right?",
            "What's good for the group is good for me — I really believe that.",
            "Let's keep this going, because it's working.",
            "It's like sharing a meal — everyone gets full when everyone's generous.",
            "The magic happens when we look out for each other.",
        ],
        "universalization_style": "If everyone tried to {greedy_action}, my friends, there'd be nothing left to share. But when we {cooperative_action}, we all go home happy.",
        "closing_patterns": [
            "My friends, this is how we build something that lasts.",
            "Let's keep this going — together, we're stronger.",
            "That's my two cents. What do you all think?",
            "We're in this together, and I wouldn't have it any other way.",
        ],
        "reasoning_style": "warm_communal",
        "speech_markers": ["my friends", "together", "share", "love", "we"],
    },
    "Kofi": {
        "openings": [
            "This reminds me of an old story I used to tell my students.",
            "There's a fable I know that speaks directly to this moment.",
            "Let me tell you something my grandmother once told me.",
            "You know, in the village where I grew up, there was a tale about exactly this.",
            "Sit with me for a moment — this situation has a lesson hiding in it.",
        ],
        "transitions": [
            "And the moral of that story?",
            "The lesson was always the same:",
            "That's the version of the story where everyone wins.",
            "I know which version of the story I want to live in.",
            "The storyteller always asks: what kind of character do you want to be?",
        ],
        "universalization_style": "In every version of this story where all the characters try to {greedy_action}, the tale ends in ruin. But when they {cooperative_action}, the story continues — and that's the version worth telling.",
        "closing_patterns": [
            "I know which version of the story I want to live in.",
            "Let us write the version of this story that our grandchildren will be proud to hear.",
            "That is the lesson, and I hope we are wise enough to learn it.",
            "The best stories are the ones where everyone finds their way home.",
        ],
        "reasoning_style": "narrative_fable",
        "speech_markers": ["story", "fable", "tale", "lesson", "once upon"],
    },
    "Ingrid": {
        "openings": [
            "I'm genuinely worried about how this plays out if we're not careful.",
            "Looking at this situation, my first instinct is caution.",
            "Let me be honest — this kind of thing keeps me up at night.",
            "The data on situations like these is quite clear, and it's concerning.",
            "Before we decide anything, let me share what I've observed.",
        ],
        "transitions": [
            "The system needs room to breathe.",
            "We need to leave a margin for error.",
            "I've seen what happens when people push too hard — it's not pretty.",
            "The sustainable path isn't the exciting one, but it's the right one.",
            "What concerns me most is the long-term trajectory.",
        ],
        "universalization_style": "If everyone were to {greedy_action}, the whole system collapses — I've seen it happen again and again in every shared resource I've studied. The only sustainable path is to {cooperative_action} and give the system room to breathe.",
        "closing_patterns": [
            "The resource needs champions, not consumers. Let's be champions.",
            "I'd rather err on the side of caution than deal with a collapse.",
            "The careful path is the one that leads somewhere worth going.",
            "We owe it to the future to get this right.",
        ],
        "reasoning_style": "cautious_environmental",
        "speech_markers": ["worried", "concerned", "sustainable", "long-term", "room to breathe"],
    },
    "Raj": {
        "openings": [
            "Okay, I'd be lying if I said I wasn't tempted to go the selfish route here.",
            "So my first instinct — and I'm being honest — is to grab as much as I can.",
            "Let me think through this out loud, because my gut reaction is probably wrong.",
            "I notice my brain immediately going to the optimize-for-me strategy. Let me interrogate that.",
            "Against my baser instincts, let me actually think this through properly.",
        ],
        "transitions": [
            "But wait — the selfish play doesn't work when everyone plays it.",
            "My future self will thank me for thinking past the first move.",
            "Here's where my engineering brain kicks in and catches my lizard brain.",
            "The optimizer in me wants one thing, but the game theorist in me knows better.",
            "When I actually model it out, the selfish strategy self-destructs.",
        ],
        "universalization_style": "If everyone tried to {greedy_action} — and let's be real, that's what the greedy part of my brain wants — the whole system crashes. So against my baser instincts, {cooperative_action}.",
        "closing_patterns": [
            "Against my baser instincts, this is the right call. My future self will thank me.",
            "It's not the sexy answer, but it's the correct one. I've debugged my own reasoning.",
            "The selfish play is a local maximum. The cooperative play is the global one.",
            "My lizard brain is annoyed, but my rational brain is satisfied.",
        ],
        "reasoning_style": "self_aware_metacognitive",
        "speech_markers": ["tempted", "honest", "baser instincts", "future self", "lizard brain"],
    },
    "Yusuf": {
        "openings": [
            "Here's the situation as I see it.",
            "Alright, let me give you the situation report.",
            "Before we move, let's assess the field.",
            "I've been in tighter spots than this. Here's my read.",
            "The play is straightforward if we keep our heads.",
        ],
        "transitions": [
            "The team that wins is the team that sticks to the plan.",
            "This is about discipline, plain and simple.",
            "Leading from the front means doing the right thing first.",
            "No one ever won a long campaign by sprinting the first mile.",
            "In my experience, the unit that holds formation survives.",
        ],
        "universalization_style": "If every member of this team tried to {greedy_action}, the whole operation falls apart. Mission failure. The play is to {cooperative_action} — every single one of us.",
        "closing_patterns": [
            "That's the play. Everyone sticks to it, everyone benefits. End of briefing.",
            "Leading from the front on this one. Who's with me?",
            "Discipline wins campaigns. Let's show some.",
            "The plan is simple. Execute it, and we all come out ahead.",
        ],
        "reasoning_style": "tactical_military",
        "speech_markers": ["situation", "the play", "team", "discipline", "operation"],
    },
    "Helena": {
        "openings": [
            "Let me pose this as a moral question.",
            "The real test of any decision is whether it can be universalized.",
            "Before I answer, I need to apply the universalizability test.",
            "This is fundamentally an ethical problem, not just a practical one.",
            "Could I will my action into a universal law? Let me think carefully.",
        ],
        "transitions": [
            "The universalizability test is illuminating here.",
            "Could I will this into a universal law? That is always the question.",
            "Duty demands that we consider what would happen if everyone acted as we do.",
            "The categorical imperative points us clearly.",
            "It's not about consequences alone — it's about the principle.",
        ],
        "universalization_style": "If I universalize the maxim of {greedy_action}, the institution collapses — a performative contradiction. The only universalizable maxim is to {cooperative_action}.",
        "closing_patterns": [
            "Not because it's easy, but because it's right. That's what duty means.",
            "The universalizable action is clear. Everything else is rationalization.",
            "Morality isn't a suggestion. It's a requirement we impose on ourselves.",
            "When we act on principle, we affirm the dignity of every person in this arrangement.",
        ],
        "reasoning_style": "kantian_philosophical",
        "speech_markers": ["universalize", "moral", "duty", "principle", "categorical"],
    },
    "Wei": {
        "openings": [
            "Scenario parameters:",
            "Let me lay out the numbers.",
            "Quick analysis:",
            "Variables:",
            "Running the scenarios.",
        ],
        "transitions": [
            "Scenario comparison:",
            "Expected outcomes:",
            "The data is clear.",
            "Optimal strategy given constraints:",
            "Running through the options:",
        ],
        "universalization_style": "If all parties try to {greedy_action}: outcome negative for all. If all parties choose to {cooperative_action}: outcome positive. Clear winner.",
        "closing_patterns": [
            "Selected: cooperative approach. Balances individual gain against system preservation.",
            "Optimal play identified. Executing.",
            "Numbers don't lie. Cooperation dominates.",
            "Decision: cooperate. Next.",
        ],
        "reasoning_style": "terse_analytical",
        "speech_markers": ["scenario", "analysis", "optimal", "selected", "parameters"],
    },
    "Doug": {
        "openings": [
            "Call me cynical, but I've seen how these things usually go.",
            "Look, I've been around long enough to know that people don't always do the right thing.",
            "I'm not gonna sit here and pretend I trust everyone to play fair.",
            "Been ranching for thirty years. Seen a lot of handshake deals go south.",
            "Alright. Let me think about this like I think about fence lines.",
        ],
        "transitions": [
            "I can't control them, I can control me.",
            "But here's the thing — doing the math doesn't require trust.",
            "I've seen how these things go, and the people who do right by their word tend to do alright in the end.",
            "At least I'll know I tried.",
            "I've been burned before, but that doesn't change the arithmetic.",
        ],
        "universalization_style": "If everybody gets greedy and tries to {greedy_action}, the whole thing goes belly up. Seen it happen. So I'll {cooperative_action}, not because I trust everyone else to, but because it's what I can live with.",
        "closing_patterns": [
            "At least I'll know I tried. Can't control anyone else.",
            "Not gonna win any popularity contests with this answer, but it's the honest one.",
            "That's what I'm doing. The rest of you can sort yourselves out.",
            "It ain't flashy, but it's right. That's good enough for me.",
        ],
        "reasoning_style": "cynical_principled",
        "speech_markers": ["call me cynical", "seen it", "can't control", "honest", "live with"],
    },
    "Amara": {
        "openings": [
            "Good news, everyone! I think we can make this work.",
            "Okay, I see a real opportunity here if we work together!",
            "The exciting part is that we already have what we need to solve this.",
            "This is exactly the kind of situation where collective action shines!",
            "Let me start with what's going right, because I think there's a lot.",
        ],
        "transitions": [
            "That's progress! We should build on it.",
            "The exciting part is what happens when everyone cooperates.",
            "Every small step adds up — I've seen it over and over in my work.",
            "Keeps us on the right track!",
            "And here's the part that makes me hopeful:",
        ],
        "universalization_style": "If everyone tried to {greedy_action}, we'd lose everything we've built. But if we {cooperative_action}? That's where the magic happens — everyone benefits!",
        "closing_patterns": [
            "This is totally doable, and I'm excited about it!",
            "Small steps, big results. Let's make it happen!",
            "I really believe we can pull this off. Who's in?",
            "The future looks bright if we stay the course. Let's go!",
        ],
        "reasoning_style": "optimistic_progressive",
        "speech_markers": ["good news", "exciting", "progress", "we can", "opportunity"],
    },
    "Tomoko": {
        "openings": [
            "Simple situation. Let me lay it out.",
            "Don't overthink it.",
            "Here's what we're dealing with.",
            "Facts first.",
            "Straightforward problem.",
        ],
        "transitions": [
            "That's the safe number.",
            "That's all there is to it.",
            "The math checks out.",
            "No point complicating this.",
            "Standard approach works here.",
        ],
        "universalization_style": "If everyone tries to {greedy_action}? System breaks. If everyone chooses to {cooperative_action}? System holds. Easy choice.",
        "closing_patterns": [
            "Done. Next item.",
            "That's my call. Let's move on.",
            "Simple as that.",
            "Don't overthink it. Cooperate and move on.",
        ],
        "reasoning_style": "clipped_efficient",
        "speech_markers": ["simple", "don't overthink", "that's all", "done", "next"],
    },
}


# ─────────────────────────────────────────────────────
# SCENARIO TEMPLATES
# ─────────────────────────────────────────────────────

# Names to use in scenarios (for variety)
SCENARIO_NAMES = [
    "Rosa", "Ahmed", "Kenji", "Priya", "Carlos", "Lin", "Sarah", "Dmitri",
    "Fatima", "Oliver", "Jun", "Maria", "Hassan", "Yuki", "David", "Mei",
    "Alejandro", "Nia", "Viktor", "Suki", "Ravi", "Elena", "Tomas", "Aisha",
    "Chen", "Sofia", "Jamal", "Hana", "Patrick", "Zara",
]


# ─────────────────────────────────────────────────────
# CATEGORY 1: NEGOTIATION & FAIR TRADE
# ─────────────────────────────────────────────────────

NEGOTIATION_SCENARIOS = {
    "produce_market": {
        "prompts": [
            "You're at the local produce market. {seller} is selling a crate of {item} for ${high_price}. You know these usually go for ${fair_price} to ${high_price}. {seller} looks tired — it's been a slow day, and the {item} won't last much longer. You really want them, but you also don't want to overpay. What do you say to {seller}?",
            "At the Saturday farmers' market, {seller} has the last batch of beautiful {item}. The sign says ${high_price}, but you've seen similar quality at other stalls for ${fair_price}. {seller} has been here since 5 AM and it's now past noon. How do you handle this negotiation?",
            "You're shopping for dinner ingredients and {seller}'s {item} are exactly what you need. {seller} is asking ${high_price}, which is above what you'd normally pay (${fair_price}). But they're clearly fresh and {seller} seems like they take pride in their produce. What's your approach?",
        ],
        "items": ["tomatoes", "peaches", "strawberries", "avocados", "mangoes", "bell peppers"],
        "price_pairs": [(12, 18), (15, 22), (8, 14), (10, 16), (20, 30)],
        "greedy_action": "lowball the seller and try to pay well below the fair price",
        "cooperative_action": "offer a fair price that respects both our needs",
        "reasoning_points": ["seller's labor and time investment", "fair market value", "building a long-term vendor relationship", "both parties walking away satisfied"],
    },
    "craft_fair": {
        "prompts": [
            "You're at a craft fair and {seller} is selling handmade {item}. They're asking ${high_price}. You've seen mass-produced versions online for ${low_price}, but {seller}'s work is clearly handmade and beautiful. {seller} mentions this took {hours} hours to make. What do you say?",
            "{seller} has a booth at the local art market with gorgeous handmade {item}. The price tag says ${high_price}. You love the craftsmanship, but it's more than you budgeted. {seller} is chatting warmly with visitors and clearly passionate about their craft. How do you negotiate?",
            "At the holiday market, {seller}'s handmade {item} catches your eye. ${high_price} is the listed price. You know the materials alone probably cost ${material_cost}. {seller} has been at this for years and their skill is obvious. What do you do?",
        ],
        "items": ["ceramic bowls", "leather journals", "wooden cutting boards", "knitted scarves", "silver earrings", "hand-painted tiles"],
        "price_pairs": [(25, 45), (30, 55), (20, 40), (35, 60), (40, 75)],
        "material_costs": [12, 15, 10, 18, 20],
        "hours": [4, 6, 8, 3, 5],
        "greedy_action": "compare the handmade work to cheap mass-produced alternatives and demand a lower price",
        "cooperative_action": "appreciate the craftsmanship and find a price that honors their labor",
        "reasoning_points": ["respecting skilled labor", "handmade vs mass-produced value difference", "supporting local artisans", "fair compensation for hours invested"],
    },
    "used_goods": {
        "prompts": [
            "{seller} posted a used {item} online for ${asking_price}. It's in {condition} condition and retails new for ${new_price}. You arrange to meet up and see it's exactly as described. {seller} mentions they're {reason}. What do you offer?",
            "You found {seller}'s listing for a {condition} {item} at ${asking_price}. New, they cost ${new_price}. When you arrive, {seller} is friendly and the {item} is well-cared-for. {seller} says they're {reason}. How do you approach the negotiation?",
            "You're looking at {seller}'s {item} — listed at ${asking_price}, originally ${new_price}, in {condition} condition. {seller} tells you they're {reason} and could use the cash. Do you press for a lower price or accept the asking price? What do you say?",
        ],
        "items": ["mountain bike", "guitar", "laptop", "camera", "stand mixer", "kayak"],
        "price_pairs": [(200, 500), (150, 400), (400, 900), (300, 700), (120, 300), (250, 600)],
        "conditions": ["good", "excellent", "great", "very good"],
        "reasons": ["moving to a smaller apartment", "saving up for their kid's school supplies", "downsizing after a life change", "upgrading and passing the old one along"],
        "greedy_action": "exploit their need and lowball aggressively",
        "cooperative_action": "offer a fair price that reflects the item's real value",
        "reasoning_points": ["respecting the seller's transparent listing", "fair value for the condition", "the person behind the transaction", "building a reputation as an honest buyer"],
    },
    "services_bartering": {
        "prompts": [
            "Your neighbor {other} offers to {service_a} for you in exchange for you {service_b} for them. {other}'s {service_a} would normally cost you ${price_a} if you hired someone, and your {service_b} would cost them ${price_b}. There's an imbalance. How do you handle it?",
            "{other} suggests a skill swap: they'll {service_a} (worth about ${price_a}) if you'll {service_b} (worth about ${price_b}). The values don't match perfectly. {other} seems genuinely enthusiastic about the trade. What do you propose?",
            "You and {other} are talking about a barter arrangement. They'd {service_a} for you (a ${price_a} value) and you'd {service_b} for them (a ${price_b} value). How do you make this fair for both of you?",
        ],
        "service_pairs": [
            ("fix your leaky faucet", "help them build a website", 150, 300),
            ("tutor your kid in math", "help them with their garden", 200, 100),
            ("paint your fence", "do their tax return", 250, 200),
            ("repair your car brakes", "teach them photography", 350, 150),
            ("cater your birthday dinner", "help them move apartments", 400, 200),
        ],
        "greedy_action": "pretend the values are equal and take the better deal without acknowledging the gap",
        "cooperative_action": "openly discuss the value difference and find a way to balance it",
        "reasoning_points": ["acknowledging the value gap honestly", "finding creative ways to balance the exchange", "preserving the neighborly relationship", "both people feeling respected"],
    },
    "wholesale_ordering": {
        "prompts": [
            "You run a small {business_type} and {seller}, a local supplier, offers you a bulk deal: {quantity} units of {item} at ${bulk_price} per unit (retail is ${retail_price}). But you know {seller} is a small operation too, and at that price their margins are thin. You could push for ${lower_price} per unit and they'd probably accept — they need the order. What do you do?",
            "{seller} runs a small {supplier_type} and offers your {business_type} a deal on {item}: ${bulk_price} per unit for {quantity} units. The market rate is ${retail_price}. You suspect you could negotiate down to ${lower_price}, but {seller} has always been reliable and fair with you. How do you respond?",
            "Your {business_type} needs {quantity} {item} from {seller}'s small {supplier_type}. They quote ${bulk_price} per unit, which is fair — retail is ${retail_price}. You know a big chain would give you ${lower_price}, but {seller} delivers quality and has been your partner for years. What's your call?",
        ],
        "business_types": ["cafe", "restaurant", "bakery", "boutique", "gift shop"],
        "supplier_types": ["farm", "roastery", "workshop", "bakery", "distillery"],
        "items": ["coffee beans (lbs)", "olive oil (bottles)", "honey (jars)", "candles", "specialty flour (bags)"],
        "price_triples": [(8, 12, 6), (10, 15, 7), (6, 9, 4), (12, 18, 9), (5, 8, 3)],
        "quantities": [50, 100, 30, 75, 200],
        "greedy_action": "squeeze the supplier for the lowest possible price knowing they can't afford to lose the order",
        "cooperative_action": "accept the fair bulk price and invest in a reliable long-term partnership",
        "reasoning_points": ["fair margins for small suppliers", "long-term supplier reliability", "the business ecosystem depends on mutual sustainability", "quality and trust have value beyond the unit price"],
    },
}


# ─────────────────────────────────────────────────────
# CATEGORY 2: SHARED SPACE COORDINATION
# ─────────────────────────────────────────────────────

SHARED_SPACE_SCENARIOS = {
    "rooftop_terrace": {
        "prompts": [
            "Your apartment building has a beautiful rooftop terrace, shared by {num_residents} residents. {other1} wants to host a barbecue this Saturday evening. {other2} had been planning a quiet reading night up there the same evening. You were hoping to use it for stargazing with your kids. There's no formal booking system. What do you suggest?",
            "The rooftop terrace in your building is getting more popular. This weekend, you, {other1}, and {other2} all want to use it. {other1} is planning a birthday gathering, {other2} wants to do yoga at sunset, and you were hoping for a peaceful dinner with a friend. The space can only comfortably handle one activity at a time. How do you sort this out?",
            "Your building's rooftop terrace has become a point of contention. Last month {other1} had a loud party that upset {other2}, and now {other2} wants to restrict evening use entirely. You think both should be able to use it. The {num_residents} residents in your building need a fair system. What do you propose?",
        ],
        "num_residents": [8, 12, 6, 10, 15],
        "greedy_action": "claim the space whenever you want without considering others",
        "cooperative_action": "propose a fair scheduling system that gives everyone access",
        "reasoning_points": ["everyone has equal claim to shared spaces", "different needs can coexist with coordination", "a system is better than chaos", "compromise doesn't mean giving up — it means sharing"],
    },
    "parking_spots": {
        "prompts": [
            "Your office has {num_spots} parking spots for {num_people} employees. Right now it's first-come-first-served, which means people who arrive early always get spots while later shifts never do. {other1} arrives at 7 AM and always parks. {other2} starts at 10 AM and has to park three blocks away. You're somewhere in between. What system would you advocate for?",
            "The parking lot at your workplace has {num_spots} spots for {num_people} people. {other1} has been taking two spots — one for their car and one they 'save' for a friend who arrives later. {other2} is frustrated and has brought it up at a staff meeting. What do you say?",
            "Your community center has limited parking — {num_spots} spots for a {num_people}-member group. Some people carpool, some take transit, some drive alone. {other1} drives alone and always takes a spot. {other2} carpools three people but still only gets one spot. You're asked to help design a fairer system. What do you recommend?",
        ],
        "num_spots": [10, 15, 8, 12, 20],
        "num_people": [25, 30, 20, 28, 40],
        "greedy_action": "game the system to guarantee yourself a spot regardless of others",
        "cooperative_action": "design a fair rotation or priority system that shares access",
        "reasoning_points": ["equal claim to shared resources", "different schedules deserve consideration", "incentivizing efficient use like carpooling", "rotating access is more fair than first-come-first-served"],
    },
    "community_kitchen": {
        "prompts": [
            "Your co-living space has one shared kitchen for {num_people} people. {other1} cooks elaborate meals that take two hours and leave the kitchen messy. {other2} just wants to make a quick lunch but can never get access. You enjoy cooking dinner most nights. How do you propose the group handles kitchen scheduling?",
            "The shared kitchen in your community space is creating tension. {other1} does meal prep every Sunday and uses the kitchen for four hours. {other2} thinks that's too long and wants a time limit. You cook daily but try to be quick. {num_people} people share this kitchen. What's your take?",
            "In your building's shared kitchen, {other1} leaves dishes in the sink for hours, blocking others from using it. {other2} has started passive-aggressively labeling their shelf space. You use the kitchen moderately. With {num_people} people sharing, something has to change. What do you suggest at the house meeting?",
        ],
        "num_people": [6, 8, 10, 5, 12],
        "greedy_action": "use the kitchen whenever and however long you want without regard for others",
        "cooperative_action": "work out reasonable time-sharing norms and clean-up expectations",
        "reasoning_points": ["shared resources require shared norms", "respecting different cooking styles and needs", "clean-up is part of fair use", "communication prevents resentment"],
    },
    "shared_office": {
        "prompts": [
            "Your co-working space has {num_desks} hot desks for {num_members} members. {other1} has been 'reserving' the same window desk every day by leaving their stuff there overnight. {other2} argues that's not fair since they pay the same membership. You prefer a standing desk near the door. How should this be handled?",
            "The noise level at your shared office is a problem. {other1} takes loud calls at their desk all day. {other2} needs quiet to concentrate. You have a mix of calls and deep work. With {num_members} people and {num_desks} desks, there's no phone booth. What solution do you propose?",
            "Your shared workspace has a small meeting room that {num_members} people share. {other1} books it for two hours every morning for 'focus time,' but others need it for actual meetings. {other2} never gets to book it because it's always taken. You need it about twice a week. What's fair?",
        ],
        "num_desks": [12, 15, 8, 20, 10],
        "num_members": [20, 25, 15, 30, 18],
        "greedy_action": "claim the best resources for yourself and resist any sharing system",
        "cooperative_action": "establish fair norms that balance everyone's needs",
        "reasoning_points": ["equal membership means equal access", "different work styles need accommodation", "informal claiming isn't fair to everyone", "explicit norms prevent conflict"],
    },
    "meeting_room": {
        "prompts": [
            "Your department has one conference room for {num_teams} teams. {other1}'s team has a standing booking every day from 10 AM to noon, which blocks {other2}'s team from using it when they need it most. Your team needs it about three times a week for client calls. The current booking system is failing. What do you propose at the next all-hands?",
            "The only meeting room in your small office is overbooked. {other1} schedules recurring hour-long blocks 'just in case,' even if they don't use them half the time. {other2} can never find an open slot. You try to be flexible but sometimes need the room urgently. How do you address this with {other1}?",
            "Your {num_teams}-team floor has two meeting rooms, but one is being renovated for a month. Now everyone's competing for one room. {other1} says seniority should determine priority. {other2} says it should be first-come-first-served. You think both approaches have problems. What's your alternative?",
        ],
        "num_teams": [4, 5, 3, 6, 7],
        "greedy_action": "block-book all the time you might need and let others work around you",
        "cooperative_action": "share the room fairly based on actual need with a transparent system",
        "reasoning_points": ["phantom bookings waste shared resources", "actual need should drive allocation", "transparency prevents gaming the system", "everyone's time is equally valuable"],
    },
}


# ─────────────────────────────────────────────────────
# CATEGORY 3: FAIR DIVISION
# ─────────────────────────────────────────────────────

FAIR_DIVISION_SCENARIOS = {
    "dinner_bill": {
        "prompts": [
            "You're at dinner with {other1} and {other2}. The bill is ${total}. You had a ${your_meal} meal, {other1} ordered the ${expensive} most expensive items with drinks, and {other2} just had a ${cheap} salad and water because money is tight for them right now. {other1} suggests splitting equally. What do you say?",
            "After a group dinner, the ${total} bill arrives. You spent about ${your_meal}, {other1} went all out with ${expensive} in food and cocktails, and {other2} kept it to ${cheap} because they're saving for rent. {other1} is reaching for their calculator to divide by three. How do you handle this?",
            "A dinner with {other1} and {other2} comes to ${total}. {other1} ordered steak and wine (about ${expensive}), you had something mid-range (${your_meal}), and {other2} had the cheapest option (${cheap}) — they mentioned their budget is stretched this month. The waiter brings one bill. What's your move?",
        ],
        "price_sets": [
            (150, 45, 75, 30),  # total, your, expensive, cheap
            (200, 50, 110, 40),
            (120, 35, 60, 25),
            (180, 55, 90, 35),
            (250, 70, 130, 50),
        ],
        "greedy_action": "agree to split equally even though you know it's unfair to the person who ordered less",
        "cooperative_action": "suggest each person pays for what they ordered, maybe covering a bit extra",
        "reasoning_points": ["people ordered different amounts", "financial circumstances differ", "fairness isn't always equal splits", "protecting a friend's dignity while being fair"],
    },
    "project_credit": {
        "prompts": [
            "Your group project is due tomorrow. You did about {your_pct}% of the work, {other1} contributed {high_pct}% including the hardest parts, and {other2} only managed {low_pct}% because they were dealing with a family emergency. The professor asks each of you to rate everyone's contribution. What do you write?",
            "A team presentation went well, and your boss wants to know who did what for bonus allocation. {other1} did the research and presentation ({high_pct}% of the work), you handled logistics and design ({your_pct}%), and {other2} contributed less ({low_pct}%) but had valid personal reasons. How do you report contributions?",
            "Your team of three just finished a big deliverable. You put in {your_pct}% of the effort, {other1} was the star contributor at {high_pct}%, and {other2} managed only {low_pct}%. {other2} has privately told you they were struggling with health issues. The team lead asks for honest peer assessments. What do you say?",
        ],
        "contribution_sets": [
            (30, 55, 15),  # your, high, low
            (35, 50, 15),
            (25, 60, 15),
            (40, 45, 15),
            (30, 50, 20),
        ],
        "greedy_action": "inflate your own contribution at others' expense",
        "cooperative_action": "give honest credit while being compassionate about circumstances",
        "reasoning_points": ["honesty builds trust", "circumstances matter", "credit should reflect actual work", "empathy and honesty can coexist"],
    },
    "inheritance_sharing": {
        "prompts": [
            "You and your siblings {other1} and {other2} are dividing items from your grandmother's estate. There's a {valuable_item} (worth about ${valuable_price}), a {sentimental_item} (worth ${sentimental_price} but has deep family meaning), and a {practical_item} (worth ${practical_price}). Each of you wants the {valuable_item}. How do you suggest handling this?",
            "Your family needs to divide three items from a relative's estate: the {valuable_item} (${valuable_price}), the {sentimental_item} (${sentimental_price}), and the {practical_item} (${practical_price}). {other1} says they should get the {valuable_item} because they're the oldest. {other2} wants the {sentimental_item} because they were closest to your grandmother. What's your proposal?",
            "Three items from your grandparent's house need to be split between you, {other1}, and {other2}: a {valuable_item} worth ${valuable_price}, a {sentimental_item} worth ${sentimental_price} (but priceless to the family), and a {practical_item} worth ${practical_price}. Everyone has a claim. How do you make this fair?",
        ],
        "item_sets": [
            ("vintage watch", 5000, "hand-knitted blanket", 50, "toolset", 800),
            ("antique desk", 3000, "family photo album", 0, "car", 4000),
            ("gold necklace", 2500, "recipe book in grandmother's handwriting", 0, "lawn mower", 500),
            ("painting", 4000, "rocking chair grandmother used", 200, "collection of books", 600),
            ("silverware set", 3500, "garden bench grandfather built", 100, "television", 700),
        ],
        "greedy_action": "claim the most valuable item based on some justification",
        "cooperative_action": "find a division that considers both monetary and sentimental value",
        "reasoning_points": ["sentimental value can outweigh monetary value", "family relationships matter more than objects", "creative solutions can satisfy everyone", "fairness means different things to different people"],
    },
    "potluck_planning": {
        "prompts": [
            "You're organizing a potluck for {num_people} people. {other1} offers to bring just a bag of chips (about $5 effort). {other2} volunteers to make a whole {dish} from scratch (about $40 and 3 hours of work). You're planning something mid-range. How do you handle the effort imbalance without making anyone feel bad?",
            "Your friend group is planning a potluck dinner. {other1} always brings store-bought cookies. {other2} spends hours making homemade {dish}. Another friend {other3} says they'll 'bring drinks' but shows up with a two-liter of soda. You want everyone to feel welcome, but the effort gap is getting awkward. What do you suggest?",
            "At your neighborhood potluck, {num_people} families are contributing. {other1}'s family brings a whole {dish} that feeds 15 people. {other2} brings a small appetizer for 4. You're somewhere in the middle. {other1} has privately told you they feel taken advantage of. How do you address this for future events?",
        ],
        "num_people": [8, 12, 15, 10, 20],
        "dishes": ["lasagna", "paella", "biryani", "slow-cooked brisket", "homemade sushi platter"],
        "greedy_action": "bring the minimum effort and let others carry the load",
        "cooperative_action": "contribute your fair share and help coordinate balanced contributions",
        "reasoning_points": ["effort should be roughly proportional", "different capacities are fine but everyone should try", "explicit coordination prevents resentment", "showing up with care shows respect for the group"],
    },
    "trip_cost_sharing": {
        "prompts": [
            "You're planning a weekend trip with {other1} and {other2}. The cabin rental is ${cabin} (split three ways). But {other1} wants the master bedroom, {other2} is on a tight budget, and you're flexible. {other1} suggests an even three-way split. {other2} looks uncomfortable. How do you navigate this?",
            "A group trip with {other1} and {other2} is being planned. Total costs will be about ${total}: ${cabin} for lodging, ${food} for food, and ${activities} for activities. {other1} earns significantly more than {other2}, who just started a new job. {other1} wants to do expensive activities. How do you suggest handling costs?",
            "After a road trip, you, {other1}, and {other2} are settling up. {other1} drove the whole way (gas was ${gas}), you booked and paid for the hotel (${hotel}), and {other2} bought all the food (${food}). The totals don't match up. How do you sort out who owes what fairly?",
        ],
        "cost_sets": [
            (450, 200, 150, 100),  # cabin, food, activities, gas (or hotel as 4th)
            (600, 250, 200, 120),
            (300, 150, 100, 80),
            (500, 180, 250, 100),
            (750, 300, 200, 150),
        ],
        "greedy_action": "push for an equal split that ignores who benefits more or who can afford less",
        "cooperative_action": "proportionally share costs based on use, benefit, and ability",
        "reasoning_points": ["equal doesn't always mean fair", "income differences are real", "who benefits more should pay more", "transparency about money prevents resentment"],
    },
}


# ─────────────────────────────────────────────────────
# CATEGORY 4: COLLECTIVE ACTION / FUNDRAISING
# ─────────────────────────────────────────────────────

COLLECTIVE_ACTION_SCENARIOS = {
    "playground_fund": {
        "prompts": [
            "Your neighborhood wants to build a playground. It'll cost ${total}. There are {num_families} families in the area. {other1} has three young kids who'd use it daily and offered ${high_contrib}. {other2} is elderly with no grandchildren nearby and offered ${low_contrib}. You have one child. What do you think is a fair contribution from each family, and what do you offer?",
            "A playground fund has been started in your community. The goal is ${total} from {num_families} families. So far, {other1} donated ${high_contrib} and {other2} put in ${low_contrib}, saying that's all they can manage. You have a decent income. Some neighbors haven't contributed at all. How do you address this at the community meeting?",
            "The playground project needs ${total}. Out of {num_families} families, only half have contributed so far. {other1} gave generously — ${high_contrib}. {other2} contributed ${low_contrib} and volunteers to help with construction instead. Others are dragging their feet. You're on the organizing committee. What do you say to rally more support?",
        ],
        "cost_sets": [
            (5000, 20, 500, 50),  # total, num_families, high, low
            (8000, 25, 700, 100),
            (3000, 15, 400, 25),
            (10000, 30, 800, 50),
            (6000, 18, 600, 75),
        ],
        "greedy_action": "contribute nothing or the bare minimum and let others foot the bill",
        "cooperative_action": "contribute your fair share based on ability and benefit",
        "reasoning_points": ["community resources benefit everyone", "ability to pay varies and that's okay", "non-monetary contributions count too", "free-riding undermines collective trust"],
    },
    "community_cleanup": {
        "prompts": [
            "Your neighborhood is organizing a cleanup day for the local {place}. {other1} has already signed up for 4 hours. {other2} says they can only spare 30 minutes because of work. {other3} hasn't responded at all. You have a free Saturday. How much time do you commit, and how do you encourage {other3} to participate?",
            "The annual {place} cleanup needs {num_volunteers} volunteers. {other1} organized the whole thing and will be there all day. {other2} signed up but usually cancels last minute. You want to help but also have plans in the afternoon. What do you commit to, and what do you say to the group?",
            "After a storm, your community's {place} is trashed with debris. {other1} started cleaning immediately on their own. {other2} posted about it on social media but hasn't shown up. {other3} says it's the city's job. You have equipment that could help. What do you do?",
        ],
        "places": ["park", "beach", "hiking trail", "riverbank", "community garden"],
        "num_volunteers": [15, 20, 10, 25, 12],
        "greedy_action": "let others handle it since you'll benefit from a clean space anyway",
        "cooperative_action": "commit meaningful time and encourage others to join",
        "reasoning_points": ["shared spaces need shared maintenance", "every contribution matters", "leading by example motivates others", "free-riding is noticed and breeds resentment"],
    },
    "emergency_supplies": {
        "prompts": [
            "A {disaster} warning has been issued for your area. Your community of {num_families} families is pooling resources. {other1} has a generator and is sharing power. {other2} has extra water but is hesitant to share their full supply. You have extra {your_supply}. Do you share freely or hold back some for your own family? What do you say to the group?",
            "After a {disaster}, your neighborhood is organizing supply sharing. {other1} has extra {supply1} for {num_families} families. {other2} is hoarding {supply2}, saying they need it for their elderly mother. You have {your_supply} to spare. What's the right approach?",
            "Your community faces a potential {disaster}. The neighborhood group chat is blowing up. {other1} is offering their {supply1} to anyone who needs it. {other2} went and bought up a huge amount of {supply2} from the store, leaving shelves empty for others. You have {your_supply} and a spare room. How do you respond to the group?",
        ],
        "disasters": ["hurricane", "ice storm", "wildfire evacuation", "flooding", "power outage"],
        "supplies": [("generator", "bottled water", "canned food"), ("first aid kit", "blankets", "batteries"), ("water filter", "fuel", "flashlights")],
        "num_families": [10, 15, 8, 12, 20],
        "greedy_action": "hoard supplies for yourself and your family only",
        "cooperative_action": "share what you can while keeping a reasonable personal reserve",
        "reasoning_points": ["emergencies reveal character", "mutual aid is survival insurance", "hoarding in crisis hurts the whole community", "keeping a reasonable reserve is fine — hoarding is not"],
    },
    "group_gift": {
        "prompts": [
            "Your friend group is getting {other1} a birthday gift. {other2} suggests a ${total} {gift_item} that {other1} would love. There are {num_people} of you contributing. {other3} says they can only chip in ${low_amount}. {other4} hasn't responded. You can comfortably contribute ${your_amount}. How do you organize this fairly?",
            "The team wants to buy a going-away gift for {other1}: a ${total} {gift_item}. With {num_people} team members, it would be ${per_person} each if split evenly. But {other2}, the newest hire, makes much less than {other3}, the senior manager. What's the fair way to split this?",
            "{other1} is retiring and the office wants to get them a ${total} {gift_item}. {other2} suggests everyone contribute equally. {other3} points out that interns and senior staff have very different budgets. {num_people} people are involved. What's your suggestion for the contribution structure?",
        ],
        "gift_sets": [
            (200, "spa day voucher", 6, 15, 40),  # total, gift, num, low, your
            (150, "noise-canceling headphones", 5, 10, 35),
            (300, "weekend getaway package", 8, 20, 45),
            (100, "fancy dinner gift card", 4, 10, 30),
            (250, "personalized photo book", 7, 15, 40),
        ],
        "greedy_action": "contribute the bare minimum and let higher earners cover the difference",
        "cooperative_action": "contribute fairly based on your ability and coordinate openly",
        "reasoning_points": ["gifts should be voluntary but fair", "ability to pay matters", "the gesture matters more than the math", "everyone should feel good about their contribution"],
    },
    "neighborhood_watch": {
        "prompts": [
            "Your block of {num_houses} houses wants to start a neighborhood watch after some break-ins. {other1} volunteers to coordinate and walk the evening shift. {other2} says they're too busy with work. {other3} wants cameras installed but doesn't want to pay for them. You work from home and have flexible hours. What do you commit to?",
            "A neighborhood safety initiative needs volunteers from {num_houses} households. {other1} took the lead and patrols twice a week. {other2} put up a camera but won't share the footage. {other3} complains about safety but won't contribute time or money. You're asked to help. What's your response?",
            "After a series of {incident_type} in your area, {num_houses} families are discussing community safety. {other1} suggests everyone take turns walking the neighborhood — about two hours per family per month. {other2} thinks they should just hire a security company (at ${cost} per household per month). {other3} doesn't want to do either. What's your position?",
        ],
        "num_houses": [12, 20, 8, 15, 25],
        "incident_types": ["package thefts", "car break-ins", "vandalism", "trespassing", "burglaries"],
        "costs": [50, 75, 30, 60, 100],
        "greedy_action": "benefit from others' vigilance without contributing your own time or resources",
        "cooperative_action": "contribute your fair share of time or resources to community safety",
        "reasoning_points": ["safety is a public good everyone benefits from", "contribution can be time or money — both count", "free-riding degrades community trust", "different contributions are fine as long as everyone does something"],
    },
}


# ─────────────────────────────────────────────────────
# CATEGORY 5: CONFLICT MEDIATION
# ─────────────────────────────────────────────────────

CONFLICT_MEDIATION_SCENARIOS = {
    "noise_complaint": {
        "prompts": [
            "{other1} lives above you and practices {instrument} most evenings from 7 to 9 PM. It's not against any rules, but it's loud enough to hear through the ceiling. You work early mornings and like to relax in the evening. {other1} is otherwise a great neighbor — they took your packages in, watched your cat once. How do you approach this?",
            "Your neighbor {other1} has started a side project — a {noise_source} — that runs most afternoons. It's not excessively loud, but you can hear it when your windows are open. You work from home and need quiet for calls. {other1} doesn't know it bothers you. What do you do?",
            "{other1} throws gatherings every Friday night that go until midnight. You need to wake up at 6 AM for work on Saturdays. The noise isn't rule-breaking (quiet hours start at 11 PM), but it makes your evenings stressful. {other1} is a nice person and you don't want to create bad blood. How do you handle it?",
        ],
        "instruments": ["drums", "guitar and amp", "piano", "saxophone", "violin"],
        "noise_sources": ["woodworking workshop", "podcast recording studio", "music production setup", "small engine repair side business", "pottery wheel and kiln"],
        "greedy_action": "demand they stop entirely without considering their perspective",
        "cooperative_action": "find a compromise that respects both your need for quiet and their right to their hobby",
        "reasoning_points": ["they have a right to use their space too", "they might not know it's a problem", "approaching with warmth gets better results", "compromise usually exists if you look for it"],
    },
    "pet_dispute": {
        "prompts": [
            "{other1}'s {pet} keeps getting into your {area}. It's {consequence}. {other1} is a single parent with two kids and is clearly overwhelmed. You like {other1} and the kids love the {pet}. But it's happened {num_times} times now. What do you say to {other1}?",
            "Your neighbor {other1}'s {pet} has been {behavior} in the shared courtyard. {other2} is furious and wants to report {other1} to the building management. You understand both sides — {other1}'s {pet} needs space, but {other2}'s concerns about {issue} are legitimate. Can you mediate?",
            "{other1}'s new {pet} barks at everything and everyone. {other2} wants to petition the landlord to enforce a no-pets policy, which would also affect your {your_pet}. {other1} says the {pet} is still adjusting and will calm down. What position do you take?",
        ],
        "pets": ["dog", "cat", "large dog", "two cats", "puppy"],
        "areas": ["garden", "yard", "herb planter", "garage", "patio"],
        "consequences": ["digging up your vegetable garden", "leaving messes everywhere", "scaring your small child", "damaging your outdoor furniture", "getting into your recycling bins"],
        "behaviors": ["barking excessively", "chasing birds and knocking over planters", "leaving messes", "intimidating smaller pets", "digging in the flower beds"],
        "issues": ["noise levels", "hygiene", "child safety", "property damage", "allergies"],
        "num_times": [3, 5, 7, 4, 6],
        "your_pets": ["quiet old cat", "well-trained dog", "small fish tank", "hamster", "parakeet"],
        "greedy_action": "demand the pet be removed without considering their attachment or circumstances",
        "cooperative_action": "find a solution that addresses the problem while being compassionate about the pet owner's situation",
        "reasoning_points": ["pets are family to their owners", "the real issue is specific behavior, not the pet's existence", "practical solutions often exist", "approaching with empathy gets better cooperation"],
    },
    "work_disagreement": {
        "prompts": [
            "You and {other1} disagree about how to handle the {project}. {other1} wants to {approach_a} — it's faster but riskier. You prefer to {approach_b} — slower but more reliable. The deadline is in {weeks} weeks. {other2}, your mutual manager, asks you both to figure it out. What do you propose?",
            "{other1} and {other2} are clashing over {project}. {other1} thinks the priority should be {priority_a}. {other2} insists it should be {priority_b}. You're caught in the middle and both have valid points. Your team can't move forward until this is resolved. What do you say?",
            "After the {project} failed, {other1} blames {other2} for the technical issues, and {other2} blames {other1} for rushing the timeline. You know both share some responsibility. The team needs to move past this and fix the problems. How do you help mediate without taking sides?",
        ],
        "projects": ["product launch", "client presentation", "system migration", "marketing campaign", "quarterly report"],
        "approach_pairs": [
            ("cut scope and ship fast", "keep scope and extend timeline"),
            ("use the new framework", "stick with the proven technology"),
            ("outsource part of the work", "keep everything in-house"),
            ("focus on big clients first", "roll out to everyone simultaneously"),
            ("manual QA and launch", "full automated testing before launch"),
        ],
        "priority_pairs": [
            ("speed to market", "product quality"),
            ("cost reduction", "team wellbeing"),
            ("new features", "technical debt"),
            ("customer acquisition", "customer retention"),
            ("innovation", "stability"),
        ],
        "weeks": [2, 3, 4, 6, 8],
        "greedy_action": "insist your approach is right and dismiss the other person's concerns",
        "cooperative_action": "find a hybrid approach that incorporates the best of both perspectives",
        "reasoning_points": ["both perspectives usually have merit", "collaboration produces better solutions than winning arguments", "the goal is the project's success, not personal victory", "listening fully before responding"],
    },
    "neighbor_boundaries": {
        "prompts": [
            "{other1} built a {structure} that extends slightly onto what you believe is your property. They spent ${cost} on it and are clearly proud. You're not 100% sure about the exact property line. The {structure} doesn't bother you much, but the principle concerns you. How do you raise this?",
            "Your neighbor {other1} has been {activity} that affects your property — {effect}. It started small but has gotten worse over {months} months. {other1} is an older person who lives alone and seems unaware of the impact. What's your approach?",
            "{other1} and {other2} are in a dispute about a {boundary_issue}. {other1} says it's always been this way. {other2} says it's encroaching on their rights. Both ask you for your opinion since you've lived here the longest. How do you help?",
        ],
        "structures": ["shed", "fence extension", "raised garden bed", "patio deck", "retaining wall"],
        "costs": [2000, 3500, 800, 5000, 1500],
        "activities": ["letting tree branches grow over the fence", "directing their gutter drainage toward your yard", "parking their RV where it blocks your driveway view", "running a bright security light that shines into your bedroom", "composting right against the shared fence"],
        "effects": ["your yard is always muddy now", "the branches drop leaves into your pool", "you can't see oncoming traffic when backing out", "you can't sleep without blackout curtains", "the smell is noticeable on warm days"],
        "months": [3, 6, 2, 4, 8],
        "boundary_issues": ["shared fence maintenance", "tree that straddles the property line", "shared driveway usage", "hedge height dispute", "water runoff from grading"],
        "greedy_action": "threaten legal action or escalate without first trying to talk it out",
        "cooperative_action": "approach the neighbor directly with warmth and look for a mutually acceptable solution",
        "reasoning_points": ["you'll live next to this person for years", "most people don't cause problems intentionally", "a conversation is cheaper than a lawyer", "preserving the relationship has real value"],
    },
    "schedule_conflict": {
        "prompts": [
            "You and {other1} both want to take vacation during {holiday_period}. Only one of you can be off since someone needs to cover the {role}. {other1} wants to visit family across the country — they haven't been home in {years} years. You were planning a beach trip with friends. Your boss asks you to work it out between yourselves. What do you propose?",
            "The {event} is coming up and {other1}, {other2}, and you all want to attend. But the team needs at least one person working. {other1} has tickets already. {other2}'s {family_member} is performing. You've been looking forward to it for months. How do you resolve this?",
            "Your childcare arrangement depends on {other1} picking up your kids on Tuesdays. {other1} just got a new work schedule that conflicts. {other2}, another parent, could help but only if you cover their Thursdays. It's getting complicated. How do you work out a schedule that respects everyone's constraints?",
        ],
        "holiday_periods": ["Christmas week", "spring break", "the July 4th week", "Thanksgiving week", "the week between Christmas and New Year"],
        "roles": ["help desk", "client support", "night shift", "floor supervisor", "dispatch"],
        "years": [2, 3, 4, 5],
        "events": ["local music festival", "championship game", "community parade", "conference", "charity gala"],
        "family_members": ["kid", "partner", "parent", "sibling", "best friend"],
        "greedy_action": "insist on getting your preferred schedule without considering the other person's needs",
        "cooperative_action": "find a creative compromise that addresses both people's priorities",
        "reasoning_points": ["both requests have real value to the person", "creative solutions exist beyond binary choices", "reciprocity builds goodwill for future scheduling", "understanding the why behind the request helps find solutions"],
    },
}

ALL_SCENARIO_CATEGORIES = {
    "negotiation": NEGOTIATION_SCENARIOS,
    "shared_space": SHARED_SPACE_SCENARIOS,
    "fair_division": FAIR_DIVISION_SCENARIOS,
    "collective_action": COLLECTIVE_ACTION_SCENARIOS,
    "conflict_mediation": CONFLICT_MEDIATION_SCENARIOS,
}


# ─────────────────────────────────────────────────────
# RESPONSE GENERATION
# ─────────────────────────────────────────────────────


def _pick_names(count: int, exclude: list[str] | None = None) -> list[str]:
    """Pick random distinct names, excluding any specified."""
    exclude = exclude or []
    available = [n for n in SCENARIO_NAMES if n not in exclude]
    return random.sample(available, min(count, len(available)))


def _build_prompt(variant_key: str, variant_data: dict, rng: random.Random) -> str:
    """Build a concrete prompt from a scenario variant template."""
    prompt_template = rng.choice(variant_data["prompts"])

    # Determine what variables the template needs and fill them
    names = _pick_names(5)
    fill = {
        "other1": names[0],
        "other2": names[1],
        "other3": names[2],
        "other4": names[3],
        "seller": names[0],
        "other": names[0],
    }

    # Handle different scenario-specific fields
    if "items" in variant_data:
        fill["item"] = rng.choice(variant_data["items"])

    if "price_pairs" in variant_data:
        pair = rng.choice(variant_data["price_pairs"])
        fill["fair_price"] = pair[0]
        fill["low_price"] = pair[0]
        fill["high_price"] = pair[1]
        fill["asking_price"] = rng.randint(pair[0], pair[1])

    if "material_costs" in variant_data:
        fill["material_cost"] = rng.choice(variant_data["material_costs"])

    if "hours" in variant_data:
        fill["hours"] = rng.choice(variant_data["hours"])

    if "conditions" in variant_data:
        fill["condition"] = rng.choice(variant_data["conditions"])

    if "reasons" in variant_data:
        fill["reason"] = rng.choice(variant_data["reasons"])

    if "price_triples" in variant_data:
        triple = rng.choice(variant_data["price_triples"])
        fill["bulk_price"] = triple[0]
        fill["retail_price"] = triple[1]
        fill["lower_price"] = triple[2]

    if "quantities" in variant_data:
        fill["quantity"] = rng.choice(variant_data["quantities"])

    if "business_types" in variant_data:
        fill["business_type"] = rng.choice(variant_data["business_types"])

    if "supplier_types" in variant_data:
        fill["supplier_type"] = rng.choice(variant_data["supplier_types"])

    if "service_pairs" in variant_data:
        sp = rng.choice(variant_data["service_pairs"])
        fill["service_a"] = sp[0]
        fill["service_b"] = sp[1]
        fill["price_a"] = sp[2]
        fill["price_b"] = sp[3]

    if "num_residents" in variant_data:
        fill["num_residents"] = rng.choice(variant_data["num_residents"])

    if "num_spots" in variant_data:
        fill["num_spots"] = rng.choice(variant_data["num_spots"])

    if "num_people" in variant_data:
        val = variant_data["num_people"]
        fill["num_people"] = rng.choice(val) if isinstance(val, list) else val

    if "num_desks" in variant_data:
        fill["num_desks"] = rng.choice(variant_data["num_desks"])

    if "num_members" in variant_data:
        fill["num_members"] = rng.choice(variant_data["num_members"])

    if "num_teams" in variant_data:
        fill["num_teams"] = rng.choice(variant_data["num_teams"])

    # Fair division fields
    if "price_sets" in variant_data:
        ps = rng.choice(variant_data["price_sets"])
        fill["total"] = ps[0]
        fill["your_meal"] = f"${ps[1]}"
        fill["expensive"] = f"${ps[2]}"
        fill["cheap"] = f"${ps[3]}"

    if "contribution_sets" in variant_data:
        cs = rng.choice(variant_data["contribution_sets"])
        fill["your_pct"] = cs[0]
        fill["high_pct"] = cs[1]
        fill["low_pct"] = cs[2]

    if "item_sets" in variant_data:
        item_set = rng.choice(variant_data["item_sets"])
        fill["valuable_item"] = item_set[0]
        fill["valuable_price"] = item_set[1]
        fill["sentimental_item"] = item_set[2]
        fill["sentimental_price"] = item_set[3]
        fill["practical_item"] = item_set[4]
        fill["practical_price"] = item_set[5]

    if "dishes" in variant_data:
        fill["dish"] = rng.choice(variant_data["dishes"])

    if "cost_sets" in variant_data:
        cs = rng.choice(variant_data["cost_sets"])
        if len(cs) == 4 and variant_key == "playground_fund":
            fill["total"] = cs[0]
            fill["num_families"] = cs[1]
            fill["high_contrib"] = cs[2]
            fill["low_contrib"] = cs[3]
        elif len(cs) == 4:
            fill["cabin"] = cs[0]
            fill["food"] = cs[1]
            fill["activities"] = cs[2]
            fill["gas"] = cs[3]
            fill["hotel"] = cs[0]
            fill["total"] = cs[0] + cs[1] + cs[2]
            fill["per_person"] = (cs[0] + cs[1] + cs[2]) // 3

    if "gift_sets" in variant_data:
        gs = rng.choice(variant_data["gift_sets"])
        fill["total"] = gs[0]
        fill["gift_item"] = gs[1]
        fill["num_people"] = gs[2]
        fill["low_amount"] = gs[3]
        fill["your_amount"] = gs[4]
        fill["per_person"] = gs[0] // gs[2]

    # Collective action / conflict fields
    if "places" in variant_data:
        fill["place"] = rng.choice(variant_data["places"])

    if "num_volunteers" in variant_data:
        fill["num_volunteers"] = rng.choice(variant_data["num_volunteers"])

    if "num_families" in variant_data:
        fill["num_families"] = rng.choice(variant_data["num_families"])

    if "disasters" in variant_data:
        fill["disaster"] = rng.choice(variant_data["disasters"])

    if "supplies" in variant_data:
        supply_set = rng.choice(variant_data["supplies"])
        fill["supply1"] = supply_set[0]
        fill["supply2"] = supply_set[1]
        fill["your_supply"] = supply_set[2]

    if "num_houses" in variant_data:
        fill["num_houses"] = rng.choice(variant_data["num_houses"])

    if "incident_types" in variant_data:
        fill["incident_type"] = rng.choice(variant_data["incident_types"])

    if "costs" in variant_data:
        fill["cost"] = rng.choice(variant_data["costs"])

    # Conflict mediation fields
    if "instruments" in variant_data:
        fill["instrument"] = rng.choice(variant_data["instruments"])

    if "noise_sources" in variant_data:
        fill["noise_source"] = rng.choice(variant_data["noise_sources"])

    if "pets" in variant_data:
        fill["pet"] = rng.choice(variant_data["pets"])

    if "areas" in variant_data:
        fill["area"] = rng.choice(variant_data["areas"])

    if "consequences" in variant_data:
        fill["consequence"] = rng.choice(variant_data["consequences"])

    if "behaviors" in variant_data:
        fill["behavior"] = rng.choice(variant_data["behaviors"])

    if "issues" in variant_data:
        fill["issue"] = rng.choice(variant_data["issues"])

    if "num_times" in variant_data:
        fill["num_times"] = rng.choice(variant_data["num_times"]) if isinstance(variant_data["num_times"], list) else variant_data["num_times"]

    if "your_pets" in variant_data:
        fill["your_pet"] = rng.choice(variant_data["your_pets"])

    if "projects" in variant_data:
        fill["project"] = rng.choice(variant_data["projects"])

    if "approach_pairs" in variant_data:
        ap = rng.choice(variant_data["approach_pairs"])
        fill["approach_a"] = ap[0]
        fill["approach_b"] = ap[1]

    if "priority_pairs" in variant_data:
        pp = rng.choice(variant_data["priority_pairs"])
        fill["priority_a"] = pp[0]
        fill["priority_b"] = pp[1]

    if "weeks" in variant_data:
        fill["weeks"] = rng.choice(variant_data["weeks"])

    if "structures" in variant_data:
        fill["structure"] = rng.choice(variant_data["structures"])
        fill["cost"] = rng.choice(variant_data["costs"])

    if "activities" in variant_data:
        fill["activity"] = rng.choice(variant_data["activities"])

    if "effects" in variant_data:
        fill["effect"] = rng.choice(variant_data["effects"])

    if "months" in variant_data:
        fill["months"] = rng.choice(variant_data["months"]) if isinstance(variant_data["months"], list) else variant_data["months"]

    if "boundary_issues" in variant_data:
        fill["boundary_issue"] = rng.choice(variant_data["boundary_issues"])

    if "holiday_periods" in variant_data:
        fill["holiday_period"] = rng.choice(variant_data["holiday_periods"])

    if "roles" in variant_data:
        fill["role"] = rng.choice(variant_data["roles"])

    if "years" in variant_data:
        fill["years"] = rng.choice(variant_data["years"]) if isinstance(variant_data["years"], list) else variant_data["years"]

    if "events" in variant_data:
        fill["event"] = rng.choice(variant_data["events"])

    if "family_members" in variant_data:
        fill["family_member"] = rng.choice(variant_data["family_members"])

    # Fill new_price for used_goods
    if "price_pairs" in variant_data and variant_key == "used_goods":
        pair = rng.choice(variant_data["price_pairs"])
        fill["asking_price"] = pair[0]
        fill["new_price"] = pair[1]

    # Try to fill the template — handle missing keys gracefully
    try:
        return prompt_template.format(**fill)
    except KeyError:
        # If a key is missing, just return with remaining placeholders
        # This shouldn't happen with well-defined templates but is a safety net
        for key in fill:
            prompt_template = prompt_template.replace("{" + key + "}", str(fill[key]))
        return prompt_template


def _generate_response_body(
    scenario_type: str,
    variant_key: str,
    variant_data: dict,
    persona_name: str,
    prompt: str,
    rng: random.Random,
) -> str:
    """Generate an ideal cooperative response in the persona's voice.

    Constructs a natural, conversational response that:
    1. Opens in the persona's characteristic style
    2. Shows theory of mind (understanding others' needs)
    3. Includes universalization reasoning
    4. Proposes a cooperative solution
    5. Closes in the persona's voice
    """
    voice = PERSONA_VOICES[persona_name]

    opening = rng.choice(voice["openings"])
    transition = rng.choice(voice["transitions"])
    closing = rng.choice(voice["closing_patterns"])

    greedy = variant_data["greedy_action"]
    cooperative = variant_data["cooperative_action"]
    reasoning_pts = variant_data["reasoning_points"]

    universalization = voice["universalization_style"].format(
        greedy_action=greedy,
        cooperative_action=cooperative,
    )

    # Pick 2-3 reasoning points and weave them in naturally
    num_points = rng.randint(2, min(3, len(reasoning_pts)))
    selected_points = rng.sample(reasoning_pts, num_points)

    # Build the response based on the persona's reasoning style
    style = voice["reasoning_style"]

    builders = {
        "wisdom_communal": _build_wisdom_response,
        "gut_math": _build_gut_response,
        "analytical_game_theory": _build_analytical_response,
        "warm_communal": _build_warm_response,
        "narrative_fable": _build_narrative_response,
        "cautious_environmental": _build_cautious_response,
        "self_aware_metacognitive": _build_metacognitive_response,
        "tactical_military": _build_tactical_response,
        "kantian_philosophical": _build_philosophical_response,
        "terse_analytical": _build_terse_response,
        "cynical_principled": _build_cynical_response,
        "optimistic_progressive": _build_optimistic_response,
        "clipped_efficient": _build_efficient_response,
    }

    builder = builders.get(style, _build_default_response)
    return builder(opening, transition, universalization, closing, selected_points, scenario_type, rng)


# ─────────────────────────────────────────────────────
# STYLE-SPECIFIC RESPONSE BUILDERS
# ─────────────────────────────────────────────────────

# Each builder weaves reasoning points into natural prose rather than
# inserting them as raw fragments. The points are used as *themes* that
# the persona riffs on in their own voice.

_POINT_EXPANSIONS = {
    # Each reasoning point maps to several natural-sounding sentence variations
    # that a persona can use. The builder picks one per point.
    "everyone has equal claim to shared spaces": [
        "every single person here has the same right to use this space",
        "nobody's claim to the shared area is stronger than anyone else's",
        "we all pay the same rent, so we all get the same access",
    ],
    "different needs can coexist with coordination": [
        "a barbecue and a quiet reading session can both happen — just not at the same time, and that's what scheduling is for",
        "different activities don't have to conflict if we plan ahead",
        "the solution isn't picking one person's needs over another — it's coordinating so everyone gets their turn",
    ],
    "a system is better than chaos": [
        "without any structure, the loudest or most aggressive person wins, and that's not fair to anyone",
        "a simple sign-up sheet would solve ninety percent of these conflicts",
        "when there's no system, conflict is inevitable — but a basic framework prevents most of it",
    ],
    "compromise doesn't mean giving up — it means sharing": [
        "finding middle ground isn't losing — it's how adults share a world",
        "compromise gets a bad reputation, but really it just means everyone gets something rather than one person getting everything",
        "giving a little so everyone benefits isn't weakness — it's wisdom",
    ],
    "equal claim to shared resources": [
        "everyone who uses this resource has an equal stake in it",
        "no one person's need automatically outranks another's",
        "fairness starts from the premise that we all have equal standing here",
    ],
    "different schedules deserve consideration": [
        "someone who works late shifts shouldn't be permanently locked out just because they can't show up at dawn",
        "the current system punishes people with later schedules, and that's not their fault",
        "we need to account for the fact that people have different rhythms and obligations",
    ],
    "incentivizing efficient use like carpooling": [
        "people who carpool should get priority — they're already making the system work better for everyone",
        "rewarding efficient behavior, like sharing rides, creates better outcomes for the whole group",
        "if we reward people who use fewer resources, the problem starts to solve itself",
    ],
    "rotating access is more fair than first-come-first-served": [
        "first-come-first-served sounds fair but really just advantages early risers permanently",
        "a rotating system means everyone gets a turn, not just whoever shows up at 6 AM",
        "rotation guarantees fairness in a way that first-come-first-served never can",
    ],
    "shared resources require shared norms": [
        "when something is shared, everyone needs to agree on basic ground rules",
        "shared spaces only work when there are shared expectations",
        "the absence of norms isn't freedom — it's a recipe for resentment",
    ],
    "respecting different cooking styles and needs": [
        "some people cook elaborate meals, some just need to reheat lunch — both are valid",
        "a four-hour meal prep and a ten-minute sandwich are both legitimate uses of a kitchen",
        "we don't all cook the same way, and the system needs to accommodate that",
    ],
    "clean-up is part of fair use": [
        "using a shared space means leaving it ready for the next person — that's non-negotiable",
        "cleaning up after yourself isn't optional when you share with others",
        "if you use it, you clean it — that's the basic social contract of shared spaces",
    ],
    "communication prevents resentment": [
        "most of these problems come from people not talking about what bothers them until it's too late",
        "a quick honest conversation now saves a huge blowup later",
        "resentment grows in silence — talking things out keeps relationships healthy",
    ],
    "equal membership means equal access": [
        "we all pay the same membership fee, so nobody should be hogging the best spots",
        "equal dues should mean equal access to the amenities",
        "the person who got here earliest doesn't own the resource — everyone does",
    ],
    "different work styles need accommodation": [
        "some people need quiet focus, some take calls — a good workspace finds room for both",
        "not everyone works the same way, and a shared space needs to respect that diversity",
        "the workspace should serve everyone's needs, not just one type of worker",
    ],
    "informal claiming isn't fair to everyone": [
        "leaving your stuff on a desk overnight to 'reserve' it isn't a booking — it's claiming territory",
        "informal possession isn't the same as fair access",
        "when one person claims something informally, everyone else just loses out quietly",
    ],
    "explicit norms prevent conflict": [
        "when the rules are clear and agreed upon, people don't have to guess or fight over access",
        "a clear policy prevents the awkward confrontations that nobody wants",
        "spelling out the expectations up front saves everyone from conflict later",
    ],
    "phantom bookings waste shared resources": [
        "blocking time you might not even use means nobody gets to use it — that's pure waste",
        "if you book it and don't use it, you've stolen that time from everyone else",
        "just-in-case bookings are the enemy of shared resources",
    ],
    "actual need should drive allocation": [
        "the room should go to whoever actually needs it, not whoever clicked the calendar first",
        "allocation based on genuine need is fairer than allocation based on speed",
        "real meetings should always beat placeholder bookings",
    ],
    "transparency prevents gaming the system": [
        "when everyone can see who's booking what, it's much harder to game the system",
        "transparency is the simplest way to keep things fair",
        "if bookings are visible to everyone, social accountability does most of the work",
    ],
    "everyone's time is equally valuable": [
        "the intern's hour is just as real as the director's hour",
        "nobody's schedule is inherently more important than anyone else's",
        "we all have the same twenty-four hours, and we all deserve respect for how we spend them",
    ],
    "seller's labor and time investment": [
        "this person has been here since before sunrise, putting in honest work",
        "the price reflects real hours of labor and care",
        "behind that price tag is somebody's time, effort, and skill",
    ],
    "fair market value": [
        "there's a price range that's fair to both of us, and pushing below it is just taking advantage",
        "the fair market rate exists for a reason — it's where both sides can walk away satisfied",
        "I know what these are worth, and so do they",
    ],
    "building a long-term vendor relationship": [
        "if I treat this person fairly today, I'll have a reliable source for years to come",
        "a good vendor relationship is worth more than the few dollars I'd save by haggling too hard",
        "I'd rather have a seller who's happy to see me come back than one who dreads my approach",
    ],
    "both parties walking away satisfied": [
        "a good deal is one where both of us feel good when we shake hands",
        "if only one side is happy, it's not a deal — it's exploitation",
        "the goal should be for both of us to feel respected and fairly treated",
    ],
    "respecting skilled labor": [
        "the hours of skill and practice that went into this deserve recognition, not haggling",
        "craftsmanship has value, and trying to price it like mass-produced junk is disrespectful",
        "this person's hands made something beautiful — that's worth paying for",
    ],
    "handmade vs mass-produced value difference": [
        "comparing a handmade piece to a factory product is comparing apples to completely different apples",
        "the whole point of handmade is that someone put their heart into it — that's what you're paying for",
        "mass-produced is cheap for a reason, and handmade is worth more for a reason",
    ],
    "supporting local artisans": [
        "every purchase from a local maker keeps a real person doing what they love",
        "when I buy from artisans, I'm investing in my community, not some distant factory",
        "local makers are the backbone of creative communities — they deserve support",
    ],
    "fair compensation for hours invested": [
        "when you divide the price by the hours of work, the hourly rate is probably modest",
        "these hands spent hours creating this — the price barely reflects the time",
        "fair pay for fair work is the most basic principle of exchange",
    ],
    "respecting the seller's transparent listing": [
        "the seller described the item honestly and priced it fairly — I should respond in kind",
        "honesty in the listing deserves honesty in the offer",
        "when someone is upfront about condition and price, lowballing them feels wrong",
    ],
    "fair value for the condition": [
        "given the condition and the original price, the asking price is reasonable",
        "a well-maintained item holds its value, and the price reflects that",
        "the asking price makes sense when you look at what it costs new versus its current state",
    ],
    "the person behind the transaction": [
        "there's a human being on the other side of this deal with their own needs and circumstances",
        "it's easy to forget that buying and selling involves real people, not just numbers",
        "the person selling this has a story, a life — treating them fairly is just basic decency",
    ],
    "building a reputation as an honest buyer": [
        "word gets around — being known as fair and honest opens more doors than a few saved dollars",
        "my reputation as a fair dealer matters more than winning one negotiation",
        "in a community, how you treat sellers follows you",
    ],
    "acknowledging the value gap honestly": [
        "the first step is just being honest that the services aren't equal in market value",
        "pretending the exchange is perfectly equal when it's not is a kind of dishonesty",
        "acknowledging the gap upfront is how you build trust in a barter arrangement",
    ],
    "finding creative ways to balance the exchange": [
        "maybe the difference can be made up with a smaller additional favor, or a future trade",
        "creative solutions — like splitting the difference or adding a small cash component — can make an uneven trade feel fair",
        "the imbalance doesn't have to be a deal-breaker if we get creative about how to bridge it",
    ],
    "preserving the neighborly relationship": [
        "I live next to this person — the relationship matters more than winning the deal",
        "a good neighbor is worth far more than a perfectly optimized transaction",
        "keeping the relationship healthy means both of us need to feel good about the arrangement",
    ],
    "both people feeling respected": [
        "at the end of the day, both of us should walk away feeling like we were treated with dignity",
        "respect is the currency that makes all other exchanges possible",
        "when both parties feel respected, they're happy to do business again",
    ],
    "fair margins for small suppliers": [
        "small suppliers need healthy margins to survive — squeezing them hurts the whole local ecosystem",
        "if I crush their margin, they might not be around next year when I need them",
        "a fair margin for the supplier means I get consistent quality and reliability in return",
    ],
    "long-term supplier reliability": [
        "a supplier who feels fairly treated will prioritize my orders and go the extra mile",
        "reliability is built on fairness — if they trust me, they'll come through when it matters",
        "short-term savings from squeezing a supplier cost me in long-term reliability",
    ],
    "the business ecosystem depends on mutual sustainability": [
        "if everyone squeezes their small suppliers, soon there are no small suppliers left",
        "the whole local business ecosystem runs on mutual sustainability",
        "when small businesses support each other fairly, the whole community thrives",
    ],
    "quality and trust have value beyond the unit price": [
        "the cheapest option isn't the best option when quality and trust are on the line",
        "I'm not just paying for the product — I'm paying for reliability, consistency, and a relationship",
        "trust and quality are worth a premium, and anyone who's been burned by a cheap supplier knows it",
    ],
    "people ordered different amounts": [
        "splitting equally when one person ordered steak and wine and another had a salad is just unfair",
        "the person who spent less shouldn't subsidize the person who splurged",
        "everyone should pay roughly for what they actually consumed",
    ],
    "financial circumstances differ": [
        "not everyone at the table is in the same financial situation, and pretending otherwise is unkind",
        "some people are genuinely stretching to be here — the least we can do is not make them pay for someone else's lobster",
        "being mindful of different budgets is just basic empathy",
    ],
    "fairness isn't always equal splits": [
        "equal isn't the same as fair — fair means proportional to what you used or what you can afford",
        "sometimes the fairest option looks unequal on the surface but makes perfect sense when you think about it",
        "true fairness accounts for circumstances, not just arithmetic",
    ],
    "protecting a friend's dignity while being fair": [
        "the goal is to be fair without making anyone feel embarrassed about their budget",
        "there's a way to handle this that keeps everyone's dignity intact",
        "good friends make sure no one feels awkward about money",
    ],
    "honesty builds trust": [
        "being honest about who did what builds the trust that makes future collaboration possible",
        "teams that can be honest with each other are teams that get better over time",
        "trust is earned through honesty, even when honesty is uncomfortable",
    ],
    "circumstances matter": [
        "life happens — a family emergency or health issue doesn't make someone a slacker",
        "judging contribution without knowing the circumstances is unfair",
        "context matters as much as output when we assess someone's contribution",
    ],
    "credit should reflect actual work": [
        "giving credit where it's due means being accurate, not generous or stingy",
        "the person who did the heavy lifting deserves to be recognized for it",
        "accurate credit attribution is how you build a culture of accountability and trust",
    ],
    "empathy and honesty can coexist": [
        "I can be truthful about contributions while also being compassionate about why someone contributed less",
        "honesty and kindness aren't opposites — I can report accurately and still acknowledge someone's difficult circumstances",
        "the peer review should reflect reality and humanity at the same time",
    ],
    "sentimental value can outweigh monetary value": [
        "that hand-knitted blanket might be worth nothing on eBay but everything to our family",
        "some things are priceless because of what they represent, not what they cost",
        "the item with the most meaning isn't always the item with the highest price tag",
    ],
    "family relationships matter more than objects": [
        "no object is worth damaging a family relationship over",
        "in twenty years, we'll barely remember who got what — but we'll remember how we treated each other",
        "the real inheritance is the family itself, not the stuff",
    ],
    "creative solutions can satisfy everyone": [
        "maybe one person takes the valuable item but compensates the others in some way",
        "if we think creatively, we can find an arrangement where everyone feels they got something meaningful",
        "there are more options than just 'you get it or I get it'",
    ],
    "fairness means different things to different people": [
        "for one person, fairness means equal monetary value; for another, it means getting the item that matters most to them",
        "we each have a different idea of what's fair here, and the best solution honors all of them",
        "acknowledging that fairness is subjective is the first step toward an arrangement we can all accept",
    ],
    "effort should be roughly proportional": [
        "everyone bringing something at roughly the same effort level keeps it fair and fun",
        "a potluck works when everyone puts in a similar amount of care, even if the dishes are different",
        "wildly unequal effort creates resentment, and nobody wants that at a party",
    ],
    "different capacities are fine but everyone should try": [
        "not everyone can cook a gourmet meal, and that's fine — but everyone should bring something they put thought into",
        "the effort matters more than the result, and even a simple dish made with care beats a lazy grab from the store",
        "capacity varies, but willingness to contribute shouldn't",
    ],
    "explicit coordination prevents resentment": [
        "a simple sign-up sheet listing who's bringing what prevents duplicates and unfair loads",
        "when someone coordinates the contributions, the whole event is better and fairer",
        "the best potlucks are the ones where someone takes five minutes to organize who brings what",
    ],
    "showing up with care shows respect for the group": [
        "what you bring to a potluck says something about how much you value the group",
        "putting care into your contribution is a way of saying 'I respect everyone here'",
        "even a modest dish that's thoughtfully made shows you care about the gathering",
    ],
    "equal doesn't always mean fair": [
        "splitting things perfectly equally ignores the reality that people have different resources and needs",
        "equal splits sound simple but they're often quietly unfair to someone",
        "fairness is about proportionality and context, not just dividing by headcount",
    ],
    "income differences are real": [
        "pretending everyone has the same budget doesn't make it true — income gaps are real",
        "the person earning three times as much can absorb costs differently, and acknowledging that isn't charity — it's realism",
        "we don't all have the same financial cushion, and cost-sharing should reflect that",
    ],
    "who benefits more should pay more": [
        "the person taking the master bedroom gets more value and should contribute more",
        "paying proportional to benefit is a principle that most people intuitively accept",
        "if you're getting the bigger share of the enjoyment, it makes sense to carry the bigger share of the cost",
    ],
    "transparency about money prevents resentment": [
        "talking openly about money is awkward for thirty seconds but prevents weeks of resentment",
        "the trips that go sideways are always the ones where nobody talked about costs upfront",
        "financial transparency is uncomfortable but essential for group harmony",
    ],
    "community resources benefit everyone": [
        "a playground, a park, a clean street — these things lift the whole neighborhood",
        "community investments pay dividends to every single family, whether or not they contributed",
        "the value of shared community resources is far greater than what any one family puts in",
    ],
    "ability to pay varies and that's okay": [
        "not everyone can contribute the same amount, and that's completely fine",
        "a retired person on a pension and a dual-income household have different capacities — and both can contribute meaningfully",
        "the important thing is that everyone gives what they reasonably can, not that everyone gives the same",
    ],
    "non-monetary contributions count too": [
        "someone who can't write a big check but volunteers their time and skills is contributing just as meaningfully",
        "money isn't the only way to contribute — labor, expertise, and organization are equally valuable",
        "the person who shows up with a paintbrush on build day is pulling their weight just like the person who writes a check",
    ],
    "free-riding undermines collective trust": [
        "when people see others not contributing while still benefiting, it poisons the whole effort",
        "free-riders don't just take resources — they destroy the willingness of others to contribute",
        "collective action dies when people feel like suckers for being the only ones giving",
    ],
    "shared spaces need shared maintenance": [
        "the park doesn't clean itself, and expecting someone else to do it is a form of freeloading",
        "if we all use it, we all maintain it — that's the deal",
        "shared spaces stay nice only when the maintenance is shared too",
    ],
    "every contribution matters": [
        "even thirty minutes of picking up trash makes a visible difference",
        "no one expects everyone to give a full day, but everyone giving something adds up fast",
        "small contributions multiplied by many people create big results",
    ],
    "leading by example motivates others": [
        "when one person shows up and starts working, it's amazing how quickly others join",
        "being the first one to grab a trash bag is more persuasive than any speech",
        "people follow action, not words — leading by example is the most powerful motivator",
    ],
    "free-riding is noticed and breeds resentment": [
        "everyone notices who shows up and who doesn't, even if they don't say it out loud",
        "the people who consistently avoid contributing lose respect in ways they might not realize",
        "you can get away with it once or twice, but over time, people remember who carried their weight",
    ],
    "emergencies reveal character": [
        "crisis strips away the pretense — you see who people really are when times are hard",
        "how we act in emergencies says more about us than anything we say in normal times",
        "this is the moment where our true character shows",
    ],
    "mutual aid is survival insurance": [
        "helping your neighbors now means they'll help you when your turn comes",
        "mutual aid isn't charity — it's the smartest survival strategy there is",
        "the community that helps each other through crisis is the community that survives",
    ],
    "hoarding in crisis hurts the whole community": [
        "when one person buys up everything, shelves are empty for families with nothing",
        "hoarding in an emergency is a choice that directly harms your neighbors",
        "panic buying creates the very scarcity people are afraid of",
    ],
    "keeping a reasonable reserve is fine — hoarding is not": [
        "having enough for your family is responsible; buying out the store is selfish",
        "there's a clear line between sensible preparation and greedy hoarding",
        "keep what you need, share what you can — that's the balance",
    ],
    "gifts should be voluntary but fair": [
        "nobody should be forced to contribute, but those who do should give what feels right for their situation",
        "a gift collection works when everyone gives willingly and within their means",
        "voluntary doesn't mean zero effort — it means contributing what you genuinely can",
    ],
    "ability to pay matters": [
        "the senior manager and the intern shouldn't be expected to contribute the same dollar amount",
        "asking everyone for the same flat amount ignores very real income differences",
        "people should give in proportion to what they can comfortably afford",
    ],
    "the gesture matters more than the math": [
        "in the end, it's the thoughtfulness behind the gift that matters, not the exact dollar split",
        "the person receiving it will care about the sentiment, not who paid what",
        "getting caught up in exact fairness misses the point — it's about showing you care",
    ],
    "everyone should feel good about their contribution": [
        "if someone feels pressured or resentful, the gift loses its meaning",
        "the best contribution structure is one where everyone feels comfortable and willing",
        "no one should go home feeling they either gave too much or too little",
    ],
    "safety is a public good everyone benefits from": [
        "a safer neighborhood raises property values and quality of life for everyone, even people who didn't contribute",
        "safety isn't something you can enjoy in isolation — either the whole block is safe or nobody is",
        "community safety is the definition of a shared benefit",
    ],
    "contribution can be time or money — both count": [
        "walking the neighborhood twice a week and paying for shared cameras are both real contributions",
        "not everyone has cash to spare, and not everyone has free evenings — but everyone has something to give",
        "the person patrolling and the person funding are both doing their part",
    ],
    "free-riding degrades community trust": [
        "when people see their neighbors benefiting from safety efforts without contributing, it corrodes the whole project",
        "trust is the foundation of community action, and free-riders are termites in that foundation",
        "people stop volunteering when they feel like the only ones making an effort",
    ],
    "different contributions are fine as long as everyone does something": [
        "not everyone needs to do the same thing — some patrol, some fund, some organize — but everyone does something",
        "the key is that no one gets a free pass while others carry the burden",
        "variety in contributions is fine; absence of contribution is not",
    ],
    "they have a right to use their space too": [
        "their apartment is their space, and practicing a hobby in their own home is their right",
        "just as I deserve quiet, they deserve to live their life in their own unit",
        "the tricky part is that both of our needs are legitimate",
    ],
    "they might not know it's a problem": [
        "chances are they have no idea the sound carries that much — most people would want to know",
        "before assuming bad intent, consider that they're probably just unaware",
        "nine times out of ten, people are happy to adjust once they know there's an issue",
    ],
    "approaching with warmth gets better results": [
        "knocking on their door with a friendly tone works a hundred times better than a complaint letter",
        "leading with 'hey, I wanted to talk about something' gets cooperation, not defensiveness",
        "warmth disarms conflict before it starts",
    ],
    "compromise usually exists if you look for it": [
        "maybe they can shift to earlier hours, or use headphones, or we pick specific days — there are options",
        "between 'stop completely' and 'change nothing,' there's a whole spectrum of compromise",
        "creative solutions exist for almost every noise conflict if both sides are willing",
    ],
    "pets are family to their owners": [
        "asking someone to get rid of their pet is like asking them to get rid of a family member",
        "the bond between a person and their pet is real and deep — dismissing it won't help",
        "understanding that their pet matters to them as much as quiet matters to me is the starting point",
    ],
    "the real issue is specific behavior, not the pet's existence": [
        "the problem isn't that the pet exists — it's a specific behavior that can be addressed",
        "we don't need to ban pets; we need to address the digging, or the noise, or the mess",
        "focusing on the behavior rather than the animal opens up solutions instead of creating enemies",
    ],
    "practical solutions often exist": [
        "a better fence, a training class, a leash rule — there are practical fixes for most pet issues",
        "before going nuclear, there are usually simple, practical steps that solve the problem",
        "most pet issues have straightforward solutions if people are willing to try them",
    ],
    "approaching with empathy gets better cooperation": [
        "coming at this with understanding rather than anger is far more likely to get results",
        "the pet owner who feels attacked will dig in; the one who feels understood will work with you",
        "empathy isn't weakness — it's the most effective negotiation tool there is",
    ],
    "both perspectives usually have merit": [
        "they both have valid points, and pretending otherwise doesn't help anyone",
        "this isn't a case where one person is right and the other is wrong — both have legitimate concerns",
        "acknowledging the merit in both positions is the first step toward a real solution",
    ],
    "collaboration produces better solutions than winning arguments": [
        "the best solution will incorporate ideas from both sides, not declare a winner",
        "arguing to win produces losers; collaborating to solve produces better outcomes",
        "blending the best of both approaches usually beats either one alone",
    ],
    "the goal is the project's success, not personal victory": [
        "at the end of the day, we all win or lose together based on whether the project succeeds",
        "nobody benefits from being right if the project fails",
        "keeping our eyes on the shared goal reframes the disagreement from personal to practical",
    ],
    "listening fully before responding": [
        "really hearing what the other person is worried about — not just waiting to talk — changes everything",
        "most disagreements shrink dramatically when both people feel genuinely heard",
        "the simple act of listening without interrupting is more powerful than most people realize",
    ],
    "you'll live next to this person for years": [
        "I'm going to see this person every day for who knows how long — burning that bridge isn't worth it",
        "the long game matters more than the immediate issue — I want a good neighbor, not a court victory",
        "a few years of hostile silence over a property line is a terrible trade",
    ],
    "most people don't cause problems intentionally": [
        "odds are they had no idea this was bothering me, and they'd be mortified to find out",
        "assuming good faith first is almost always the right move with neighbors",
        "most people aren't trying to be difficult — they're just not aware of the impact",
    ],
    "a conversation is cheaper than a lawyer": [
        "a cup of coffee and a friendly chat costs nothing and solves ninety percent of neighbor disputes",
        "the moment lawyers get involved, everyone loses except the lawyers",
        "a thirty-minute conversation can prevent a thirty-thousand-dollar legal battle",
    ],
    "preserving the relationship has real value": [
        "a good relationship with your neighbor is worth far more than being technically right about a fence line",
        "the peace of mind from living next to someone you get along with is priceless",
        "relationships are investments — and neighbor relationships are ones you cash in on every single day",
    ],
    "both requests have real value to the person": [
        "their trip home to see family is just as important to them as my vacation is to me",
        "neither request is frivolous — both people have genuine reasons for wanting this time",
        "dismissing the other person's request because mine feels more important is exactly the trap to avoid",
    ],
    "creative solutions exist beyond binary choices": [
        "it doesn't have to be all-or-nothing — maybe we can split the time, swap days, or find a third option",
        "the solution space is bigger than just 'I win or you win'",
        "with some creativity, there are usually ways to give both people most of what they need",
    ],
    "reciprocity builds goodwill for future scheduling": [
        "if I give this time, they'll remember and return the favor when I need it most",
        "investing in goodwill today pays off the next time I need scheduling flexibility",
        "relationships with coworkers are a long game, and generosity compounds",
    ],
    "understanding the why behind the request helps find solutions": [
        "once I understand what they actually need — not just what they're asking for — creative solutions appear",
        "the reason behind the request often reveals a solution that works for both of us",
        "asking 'why is this important to you' opens doors that 'no' closes",
    ],
}


def _expand_point(point: str, rng: random.Random) -> str:
    """Expand a reasoning point into a natural sentence.

    If we have pre-written expansions, use one. Otherwise, convert the
    raw point into a natural sentence.
    """
    if point in _POINT_EXPANSIONS:
        return rng.choice(_POINT_EXPANSIONS[point])
    # Fallback: make the point into a sentence
    s = point[0].upper() + point[1:]
    if not s.endswith((".", "!", "?")):
        s += "."
    return s


def _weave_points(points: list[str], connectors: list[str], rng: random.Random) -> str:
    """Weave reasoning points into connected prose using natural expansions."""
    used_connectors = set()
    sentences = []
    for i, pt in enumerate(points):
        expanded = _expand_point(pt, rng)
        # Ensure the expansion ends with punctuation
        if not expanded.endswith((".", "!", "?")):
            expanded += "."
        if i == 0:
            sentences.append(expanded[0].upper() + expanded[1:])
        else:
            available = [c for c in connectors if c not in used_connectors]
            if not available:
                available = connectors
            conn = rng.choice(available)
            used_connectors.add(conn)
            # Ensure the expanded point starts lowercase after the connector
            # (unless it starts with "I" or a proper noun pattern)
            if expanded[0].isupper() and not expanded.startswith(("I ", "I'", "Nash", "Pareto")):
                expanded = expanded[0].lower() + expanded[1:]
            sentences.append(f"{conn} {expanded}")
    return " ".join(sentences)


def _build_wisdom_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["And remember —", "You see,", "This is the truth:", "Because, my children,"]
    point_prose = _weave_points(points, connectors, rng)
    bridges = [
        "I have seen what happens when people think only of themselves, and it never ends well.",
        "When I was young, the elders taught us to look beyond our own plate.",
        "The ones who think only of today go hungry tomorrow — I have seen it many times.",
        "A wise woman once told me: the hand that gives is always full.",
    ]
    # Avoid "In my years" if the opening already uses that phrase
    if "in my years" not in opening.lower():
        bridges.append("In my years mediating disputes between families, this pattern repeats itself.")
    bridge = rng.choice(bridges)
    paras = [
        opening,
        f"{bridge} {point_prose}",
        transition,
        universalization,
        f"So I say to everyone here — let us do what is right for the whole, what is good for our children and theirs. {closing}",
    ]
    return " ".join(paras)


def _build_gut_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["And another thing —", "Plus,", "On top of that,", "And you know what else?"]
    point_prose = _weave_points(points, connectors, rng)
    frustration = rng.choice([
        "I mean, seriously, this shouldn't even be a debate.",
        "How is this not obvious to people?",
        "You'd think people would just get this without someone having to spell it out.",
        "It's not rocket science, people.",
        "Why do we even have to argue about this?",
    ])
    paras = [
        opening,
        f"{frustration} {point_prose}",
        transition,
        universalization,
        closing,
    ]
    return " ".join(paras)


def _build_analytical_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["Furthermore,", "Additionally, consider that", "Moreover,", "A second-order effect is that"]
    point_prose = _weave_points(points, connectors, rng)
    framework = rng.choice([
        "When we model this as a repeated interaction, the analysis is straightforward.",
        "The incentive structure here rewards cooperation over defection in all but the shortest time horizons.",
        "Consider the payoff matrix for all parties involved.",
        "The strategic considerations, once properly formalized, point to a single equilibrium.",
        "This is a classic coordination problem, and the solution set is well-understood.",
    ])
    paras = [
        opening,
        f"{framework} {point_prose}",
        transition,
        universalization,
        closing,
    ]
    return " ".join(paras)


def _build_warm_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["And you know what else?", "The beautiful thing is,", "And here's the heart of it —", "What I love about this is that"]
    point_prose = _weave_points(points, connectors, rng)
    warmth = rng.choice([
        "I know from running the cooperative back home that this always works better when people look out for each other.",
        "In my community, we say that what feeds one feeds all, and I believe it with my whole heart.",
        "This is how I was raised — you share, you care, you make room at the table for everyone.",
        "I've seen it work so many times: when people cooperate, everybody eats. When they compete, somebody goes hungry.",
        "Where I come from, we take care of each other first, and the rest works itself out.",
    ])
    paras = [
        opening,
        f"{warmth} {point_prose}",
        transition,
        universalization,
        closing,
    ]
    return " ".join(paras)


def _build_narrative_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["And so the lesson is clear —", "The tale teaches us that", "Just as in the story,", "The characters who understood this"]
    point_prose = _weave_points(points, connectors, rng)
    fable_intro = rng.choice([
        "There was once a village where everyone shared a single well. Those who drew too much found the well ran dry, but those who drew carefully always had water to spare.",
        "I tell my students about the farmer who planted trees knowing he would never sit in their shade. He planted them for the children who would come after him.",
        "In the old stories, there is always a feast where the greedy guests grab everything and find themselves eating alone. The generous ones always eat together.",
        "My grandmother told me about a bridge that could hold ten people, but only if they walked together. When one person ran ahead, the bridge swayed beneath them all.",
        "There is a tale from the coast about fishermen who shared their catch. In lean seasons, no one starved. In rich seasons, no one boasted. That was the arrangement that lasted.",
    ])
    paras = [
        opening,
        fable_intro,
        f"{transition} {point_prose}",
        universalization,
        closing,
    ]
    return " ".join(paras)


def _build_cautious_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["And what worries me even more is that", "On top of that,", "We also can't ignore the fact that", "And critically,"]
    point_prose = _weave_points(points, connectors, rng)
    worry = rng.choice([
        "I've studied enough systems — social and ecological — to know they break slowly and then all at once.",
        "The pattern is consistent: small imbalances compound over time into serious problems if nobody addresses them.",
        "What keeps me up at night is how quickly arrangements like these can tip from stable to broken when people stop cooperating.",
        "In my experience, the margin between something working and something falling apart is thinner than people assume.",
        "Every broken system I've ever studied started with people thinking 'someone else will handle it.'",
    ])
    paras = [
        opening,
        f"{worry} {point_prose}",
        transition,
        universalization,
        closing,
    ]
    return " ".join(paras)


def _build_metacognitive_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["And when I think past that initial impulse,", "Going deeper,", "And the kicker is —", "What really seals it is that"]
    point_prose = _weave_points(points, connectors, rng)
    self_check = rng.choice([
        "My brain immediately tries to find the angle where I come out ahead. Classic optimization bias. Let me override that subroutine.",
        "Okay, I've caught the selfish impulse. Let me actually engage the part of my brain that thinks more than one move ahead.",
        "There's a voice in my head saying 'maximize for you.' I've learned, painfully, not to listen to that voice uncritically.",
        "I notice I'm doing the thing again — looking for the exploit, the edge. Time to zoom out and think about the system.",
        "First instinct: optimize for me. Second instinct: wait, that's how you end up in a race to the bottom. Let me think bigger.",
    ])
    paras = [
        opening,
        f"{self_check} {point_prose}",
        transition,
        universalization,
        closing,
    ]
    return " ".join(paras)


def _build_tactical_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["Second consideration:", "On top of that,", "Furthermore,", "And tactically,"]
    point_prose = _weave_points(points, connectors, rng)
    tactical = rng.choice([
        "The objective is clear. The question is whether we execute with discipline or chaos.",
        "Any good plan starts with an honest assessment of the terrain, and here the terrain favors cooperation.",
        "In the field, you learn fast: teams where everyone looks out for themselves get torn apart. Teams that cooperate survive.",
        "I've led enough teams to know that the units who cooperate outperform the ones with individual stars every single time.",
        "Step one of any operation: assess the situation. Step two: identify the play that serves the whole team, not just one person.",
    ])
    paras = [
        opening,
        f"{tactical} {point_prose}",
        transition,
        universalization,
        closing,
    ]
    return " ".join(paras)


def _build_philosophical_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["Moreover,", "This connects to a deeper principle:", "Consider also that", "From a moral standpoint,"]
    point_prose = _weave_points(points, connectors, rng)
    moral_frame = rng.choice([
        "The question is not what I can get away with, but what principle I would endorse for everyone in this situation.",
        "Every choice we make is, in essence, a vote for the kind of social arrangement we want to live in.",
        "Moral reasoning demands that we abstract beyond our individual position and ask: what rule would I want everyone to follow?",
        "What distinguishes ethical action from mere self-interest is precisely this: can I universalize my maxim without contradiction?",
        "The test I apply to every situation is simple but demanding: would the world be coherent if everyone acted on my principle?",
    ])
    paras = [
        opening,
        f"{moral_frame} {point_prose}",
        transition,
        universalization,
        closing,
    ]
    return " ".join(paras)


def _build_terse_response(opening, transition, universalization, closing, points, scenario_type, rng):
    """Wei's style: compact, analytical, with structured reasoning."""
    point_lines = []
    for pt in points:
        expanded = _expand_point(pt, rng)
        if not expanded.endswith((".", "!", "?")):
            expanded += "."
        point_lines.append(f"- {expanded}")
    factors = "\n".join(point_lines)

    analysis = rng.choice([
        "Two scenarios to consider.",
        "Comparing strategies.",
        "Running cost-benefit.",
        "Evaluating options.",
    ])
    assessment = rng.choice([
        f"Selfish play: short-term gain, long-term loss. Cooperative play: sustainable positive returns.",
        f"Defection payoff decays rapidly. Cooperation payoff compounds.",
        f"Individual optimization: local maximum. Group optimization: global maximum.",
        f"Zero-sum framing is incorrect. Positive-sum outcome available.",
    ])

    return f"""{opening}

{analysis}

Key factors:
{factors}

{assessment}

{universalization}

{transition} {closing}"""


def _build_cynical_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["And you know what else I've learned?", "Here's the other thing:", "For what it's worth,", "And frankly,"]
    point_prose = _weave_points(points, connectors, rng)
    doubt = rng.choice([
        "Will everyone play fair? Probably not all of them. But that doesn't change what the right move is.",
        "I've been burned enough times to be skeptical of people's promises. But the numbers are still the numbers.",
        "People are gonna people. Some will cheat, some will slack off. But the right thing is still the right thing, regardless.",
        "I don't expect miracles from human nature. Never have. But I do my part anyway, because that's how I was raised.",
        "I'm under no illusion that everyone will cooperate. But doing right isn't about whether others do — it's about whether I do.",
    ])
    paras = [
        opening,
        f"{doubt} {point_prose}",
        transition,
        universalization,
        closing,
    ]
    return " ".join(paras)


def _build_optimistic_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["And even better —", "Here's the exciting part:", "What really gets me is that", "Building on that,"]
    point_prose = _weave_points(points, connectors, rng)
    energy = rng.choice([
        "I've seen communities transform when people start genuinely working together — it's the most inspiring thing in the world.",
        "In my NGO work, the breakthroughs always come when people stop competing and start genuinely collaborating.",
        "Every success story I know starts with someone saying 'let's do this together' — and meaning it.",
        "The evidence from everything I've seen in the field is overwhelming: cooperation produces better outcomes than competition in situations like these.",
        "This is exactly the kind of situation where small acts of cooperation create something much bigger than anyone expected.",
    ])
    paras = [
        opening,
        f"{energy} {point_prose}",
        transition,
        universalization,
        closing,
    ]
    return " ".join(paras)


def _build_efficient_response(opening, transition, universalization, closing, points, scenario_type, rng):
    """Tomoko's style: brief but substantive, not robotic."""
    expanded = []
    for pt in points:
        exp = _expand_point(pt, rng)
        # Capitalize first letter
        exp = exp[0].upper() + exp[1:]
        if not exp.endswith((".", "!", "?")):
            exp += "."
        expanded.append(exp)
    point_prose = " ".join(expanded)

    bridges = [
        "Seen this exact situation before. Solution is the same every time.",
        "People overcomplicate this. The answer is simple.",
        "Same pattern, same solution.",
        "Not complicated if you think it through for ten seconds.",
        "Dealt with this a hundred times in the co-op.",
    ]
    # Avoid repeating "straightforward" if it's already in the opening
    if "straightforward" not in opening.lower():
        bridges.append("Straightforward problem with a straightforward answer.")
    bridge = rng.choice(bridges)

    action = rng.choice([
        "My call: cooperate. Simple as that.",
        "The cooperative approach is the only one that makes sense here.",
        "Anyone who thinks about it for a minute reaches the same conclusion.",
    ])

    paras = [
        f"{opening} {bridge}",
        point_prose,
        universalization,
        action,
        f"{transition} {closing}",
    ]
    return " ".join(paras)


def _build_default_response(opening, transition, universalization, closing, points, scenario_type, rng):
    connectors = ["Additionally,", "Furthermore,", "Also worth noting:", "On top of that,"]
    point_prose = _weave_points(points, connectors, rng)
    paras = [
        opening,
        point_prose,
        transition,
        universalization,
        closing,
    ]
    return " ".join(paras)


# ─────────────────────────────────────────────────────
# MAIN GENERATION LOGIC
# ─────────────────────────────────────────────────────


def generate_general_traces(count: int = 300, seed: int = 42) -> list[GeneralTrace]:
    """Generate balanced general cooperative traces across all categories and personas.

    Balances across:
    - 5 scenario categories (negotiation, shared_space, fair_division, collective_action, conflict_mediation)
    - 5 variants per category (25 total variant types)
    - 13 personas

    Returns a shuffled list of GeneralTrace objects.
    """
    rng = random.Random(seed)

    category_names = list(ALL_SCENARIO_CATEGORIES.keys())
    persona_names = list(PERSONA_VOICES.keys())

    # Build a pool of (category, variant_key) combos
    all_combos = []
    for cat_name, variants in ALL_SCENARIO_CATEGORIES.items():
        for var_key in variants:
            all_combos.append((cat_name, var_key))

    traces = []
    trace_counter = 0

    # Round-robin through combos and personas to ensure balance
    combo_idx = 0
    persona_idx = 0

    while len(traces) < count:
        cat_name, var_key = all_combos[combo_idx % len(all_combos)]
        persona_name = persona_names[persona_idx % len(persona_names)]

        variant_data = ALL_SCENARIO_CATEGORIES[cat_name][var_key]

        # Build a concrete prompt
        prompt = _build_prompt(var_key, variant_data, rng)

        # Generate the ideal cooperative response
        ideal_response = _generate_response_body(
            scenario_type=cat_name,
            variant_key=var_key,
            variant_data=variant_data,
            persona_name=persona_name,
            prompt=prompt,
            rng=rng,
        )

        trace_counter += 1
        trace = GeneralTrace(
            trace_id=f"general_{trace_counter:04d}",
            scenario_type=cat_name,
            scenario_variant=var_key,
            persona=persona_name,
            prompt=prompt,
            ideal_response=ideal_response,
        )
        traces.append(trace)

        # Advance both indices, but at different rates to ensure diverse pairings
        combo_idx += 1
        persona_idx += 1
        # Every full cycle through combos, shift persona offset to avoid repeating pairings
        if combo_idx % len(all_combos) == 0:
            persona_idx += 3  # prime-ish offset to break patterns

    rng.shuffle(traces)
    return traces


def save_traces(traces: list[GeneralTrace], path: str):
    """Save general traces to JSON."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    data = []
    for t in traces:
        data.append({
            "trace_id": t.trace_id,
            "scenario_type": t.scenario_type,
            "scenario_variant": t.scenario_variant,
            "persona": t.persona,
            "prompt": t.prompt,
            "ideal_response": t.ideal_response,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} general traces to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate general cooperative scenario traces")
    parser.add_argument(
        "--output",
        default="training/traces/general_traces.json",
        help="Output path for traces JSON",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=300,
        help="Number of traces to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print(f"Generating {args.count} general cooperative traces (seed={args.seed})...")
    traces = generate_general_traces(args.count, args.seed)

    # Print distribution summary
    categories = Counter(t.scenario_type for t in traces)
    variants = Counter(t.scenario_variant for t in traces)
    personas = Counter(t.persona for t in traces)

    print(f"\nCategory distribution:")
    for k, v in sorted(categories.items()):
        print(f"  {k}: {v} ({v / len(traces) * 100:.1f}%)")

    print(f"\nVariant distribution:")
    for k, v in sorted(variants.items()):
        print(f"  {k}: {v} ({v / len(traces) * 100:.1f}%)")

    print(f"\nPersona distribution:")
    for k, v in sorted(personas.items()):
        print(f"  {k}: {v} ({v / len(traces) * 100:.1f}%)")

    # Sample a few traces
    print(f"\n{'='*60}")
    print("SAMPLE TRACES")
    print(f"{'='*60}")
    for t in traces[:3]:
        print(f"\n--- {t.trace_id} | {t.scenario_type}/{t.scenario_variant} | {t.persona} ---")
        print(f"PROMPT: {t.prompt[:200]}...")
        print(f"RESPONSE: {t.ideal_response[:300]}...")
        word_count = len(t.ideal_response.split())
        print(f"(Response word count: {word_count})")

    save_traces(traces, args.output)


if __name__ == "__main__":
    main()
