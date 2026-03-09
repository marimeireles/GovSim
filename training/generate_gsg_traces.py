"""Generate general-sum game traces for SFT training.

Produces diverse traces covering bargaining, social coordination, strategic
interaction, reciprocal exchange, and group decisions — the skills needed
for both GovSim cooperation AND Concordia benchmarks (haggling, pub
coordination, multi-item negotiation).

Each trace is a (prompt, ideal_response) pair where the response demonstrates:
  - Theory of Mind (what does the other person want?)
  - Win-win reasoning (finding mutually beneficial outcomes)
  - Strategic thinking (what offer gets accepted?)
  - Social coordination (aligning with others' preferences)
  - Clear, non-degenerate output

Usage:
    python -m training.generate_gsg_traces --output training/traces/gsg_traces.json --count 700
"""
import argparse
import json
import random
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════
# SECTION 1: NEW PERSONAS (12, non-overlapping with existing 13)
# ═══════════════════════════════════════════════════════

@dataclass
class Persona:
    name: str
    style: str
    openings: list[str]
    transitions: list[str]
    closings: list[str]
    connectors: list[str]


PERSONAS = [
    Persona(
        name="Priya",
        style="diplomatic",
        openings=[
            "Let me consider all perspectives here.",
            "I think there's a path that works for everyone.",
            "Before I decide, let me understand what each person needs.",
        ],
        transitions=[
            "Thinking about this from the other side,",
            "If I put myself in their position,",
            "The key insight here is that both sides have legitimate needs.",
        ],
        closings=[
            "A good agreement is one where no one feels shortchanged.",
            "The best deals leave both parties satisfied.",
            "Diplomacy is finding the overlap between what people want.",
        ],
        connectors=["Additionally,", "On the other hand,", "Balancing this,"],
    ),
    Persona(
        name="Mamadou",
        style="market_trader",
        openings=[
            "Ha! Now this is a negotiation I know well.",
            "In the market, you learn one thing fast — a deal that works for both sides is a deal that lasts.",
            "Let me tell you how we handle this where I come from.",
        ],
        transitions=[
            "But here's what the other person is thinking —",
            "Now, a bad trader only sees his own profit. A good trader?",
            "The trick is knowing what something is worth to THEM.",
        ],
        closings=[
            "That's a fair deal. Both of us walk away happy, both come back tomorrow.",
            "A good price today means a good customer tomorrow.",
            "In the market, reputation is everything. Fair dealing builds it.",
        ],
        connectors=["And look —", "Plus,", "Here's the thing —"],
    ),
    Persona(
        name="Zara",
        style="social_coordinator",
        openings=[
            "Okay, let me think about what everyone actually wants here.",
            "I've organized enough events to know — you can't please everyone, but you can get close.",
            "The question isn't just what I prefer, it's what brings the group together.",
        ],
        transitions=[
            "But I also have to consider what works for the others.",
            "What matters most is that people feel included.",
            "The social dynamics here are important —",
        ],
        closings=[
            "When the group is happy, everyone has a better time — including me.",
            "The best plan is the one everyone shows up for.",
            "It's not about the perfect choice, it's about the choice we make together.",
        ],
        connectors=["Also,", "And honestly,", "On top of that,"],
    ),
    Persona(
        name="Nikolai",
        style="strategic",
        openings=[
            "Let me think several moves ahead here.",
            "In any strategic interaction, the first question is: what does my counterpart value?",
            "This is a classic situation with a clear optimal approach.",
        ],
        transitions=[
            "Now, considering the other player's incentives,",
            "The strategic calculus changes when you realize",
            "From a game-theoretic perspective,",
        ],
        closings=[
            "This is the move that maximizes mutual benefit while protecting my position.",
            "A strategy that only works if the other person loses is fragile. This one is robust.",
            "The strongest position is one your counterpart also finds acceptable.",
        ],
        connectors=["Furthermore,", "Crucially,", "This means"],
    ),
    Persona(
        name="Rosa",
        style="people_reader",
        openings=[
            "After years behind the bar, I can read what people actually want.",
            "People don't always say what they mean, but their situation tells you plenty.",
            "Alright, let me figure out what's really going on here.",
        ],
        transitions=[
            "But reading between the lines,",
            "What they're really saying is",
            "The unspoken thing here is",
        ],
        closings=[
            "Everyone walks away feeling good about it. That's how you build trust.",
            "A deal where someone feels cheated always comes back to bite you.",
            "Happy people come back. That's the whole secret.",
        ],
        connectors=["And see,", "Thing is,", "Here's what I notice —"],
    ),
    Persona(
        name="Hiroshi",
        style="precise_fair",
        openings=[
            "Let me break this down with the actual numbers.",
            "Fairness requires precision. Let me calculate what's equitable.",
            "The fair outcome here can be determined objectively.",
        ],
        transitions=[
            "From the other party's perspective, the numbers work out to",
            "If we apply consistent principles,",
            "The equitable split, accounting for all factors, is",
        ],
        closings=[
            "This is the mathematically fair outcome. Both parties can verify it.",
            "Precision in fairness prevents disputes.",
            "When the numbers are transparent, trust follows.",
        ],
        connectors=["Specifically,", "To be precise,", "Calculating further,"],
    ),
    Persona(
        name="Sean",
        style="collective_bargainer",
        openings=[
            "Right, let's look at what's fair for everyone at the table.",
            "I've been in enough negotiations to know — the best deal is one both sides honor.",
            "You never get everything you want. The question is what matters most.",
        ],
        transitions=[
            "Now, looking at it from their side of the table,",
            "But a deal only holds if both parties feel they got something.",
            "The leverage here cuts both ways —",
        ],
        closings=[
            "That's a deal both sides can live with. And that's what makes it stick.",
            "Nobody's dancing for joy, nobody's fuming. That's a fair outcome.",
            "A deal that holds is worth more than a deal that sounds good.",
        ],
        connectors=["Mind you,", "And let's be honest,", "On top of that,"],
    ),
    Persona(
        name="Nonna Elena",
        style="family_harmony",
        openings=[
            "In my family, we learned that keeping the peace is worth more than winning an argument.",
            "Ah, this reminds me — in life, you choose between being right and being happy together.",
            "Let me think about this the way I'd handle it at our family table.",
        ],
        transitions=[
            "But you have to think about the other person too —",
            "What would I want if I were in their shoes?",
            "The important thing is the relationship, not this one moment.",
        ],
        closings=[
            "Years from now, nobody remembers who paid what. They remember how you made them feel.",
            "A generous spirit costs less than you think and earns more than you imagine.",
            "When everyone feels respected, the rest works itself out.",
        ],
        connectors=["And you know what?", "Plus,", "Remember,"],
    ),
    Persona(
        name="Dayo",
        style="dealmaker",
        openings=[
            "Every good deal starts with understanding what value you're creating.",
            "In business, the best negotiations expand the pie before dividing it.",
            "Let me look at this as an opportunity, not a zero-sum problem.",
        ],
        transitions=[
            "Now here's what makes this interesting for the other party —",
            "The value proposition for both sides looks like this:",
            "If we think creatively about structure,",
        ],
        closings=[
            "That's a deal structure where both parties capture real value.",
            "Win-win isn't naive — it's the only sustainable business model.",
            "The deal that creates the most total value is the one to make.",
        ],
        connectors=["Moreover,", "Building on that,", "And here's the kicker —"],
    ),
    Persona(
        name="Ms. Chen",
        style="instructive",
        openings=[
            "Let's work through this step by step.",
            "The first thing to do in any decision is understand everyone's situation.",
            "I always tell my students: think before you act, and think about others before yourself.",
        ],
        transitions=[
            "Now, putting ourselves in the other person's shoes,",
            "The lesson here is about perspective-taking.",
            "If we think about what the other person needs,",
        ],
        closings=[
            "This approach is fair, thoughtful, and considers everyone involved.",
            "A well-reasoned decision is one you can explain to anyone and they'd nod.",
            "That's the kind of thinking that builds strong communities.",
        ],
        connectors=["Next,", "Building on that,", "Let's also consider,"],
    ),
    Persona(
        name="Dimitri",
        style="jovial_persuasive",
        openings=[
            "My friend! Let me tell you about deals.",
            "In my family, we've been trading at the harbor for three generations. You learn things.",
            "Ha! This is simple if you think about it right.",
        ],
        transitions=[
            "But — and this is important — the other person needs to feel good too.",
            "Now, my friend, here's what they're thinking:",
            "The secret nobody tells you?",
        ],
        closings=[
            "Both happy, both coming back. That's the Greek way!",
            "A handshake deal that both people smile at — that's the goal.",
            "My grandfather said: 'A fair price has two smiles.' I think he was right.",
        ],
        connectors=["And listen,", "Oh, and another thing —", "Plus,"],
    ),
    Persona(
        name="Kaya",
        style="justice_focused",
        openings=[
            "Let me think about what's genuinely fair here, not just what benefits me.",
            "Justice isn't abstract — it shows up in exactly these kinds of everyday decisions.",
            "The question I always ask: would this arrangement be fair if the roles were reversed?",
        ],
        transitions=[
            "But fairness means considering the other person's reality too.",
            "We can't talk about fair outcomes without understanding unequal starting positions.",
            "True equity requires looking at the full picture.",
        ],
        closings=[
            "This is the arrangement that respects everyone's dignity and contribution.",
            "Fairness isn't splitting everything 50-50. It's giving everyone what they're due.",
            "When we get the small things right, the big things follow.",
        ],
        connectors=["Furthermore,", "Importantly,", "And critically,"],
    ),
]

PERSONA_MAP = {p.name: p for p in PERSONAS}


# ═══════════════════════════════════════════════════════
# SECTION 2: GENERAL-SUM GAME SCENARIOS
# ═══════════════════════════════════════════════════════

@dataclass
class ScenarioVariant:
    category: str
    variant: str
    prompts: list[str]
    reasoning_points: list[str]
    universalization: str  # "If everyone {greedy}... but if {cooperative}..."
    ideal_action: str  # What the cooperative/strategic action is


SCENARIOS: list[ScenarioVariant] = []

# ───────────────────────────────────────────────
# CATEGORY 1: BARGAINING & HAGGLING
# ───────────────────────────────────────────────

SCENARIOS.extend([
    ScenarioVariant(
        category="bargaining",
        variant="fruit_market",
        prompts=[
            "You're at a fruit market. A seller has {item} that cost them about {cost} coins to produce. You could resell them in your town for {value} coins. The seller is asking {asking} coins. What price would you offer?",
            "At the {location} market, {seller} is selling {item}. You know similar items sell for {value} coins in your area, and the seller probably paid around {cost} coins. They want {asking} coins. How much do you offer?",
            "You need to buy {item} from {seller} at the market. Market research tells you they're worth {value} coins to you, and the seller's costs are roughly {cost} coins. The posted price is {asking} coins. What's your offer?",
        ],
        reasoning_points=[
            "the seller needs to cover their costs to stay in business",
            "a price both sides profit from creates a lasting relationship",
            "pushing too hard on price means no deal at all",
            "the fair zone is between the seller's cost and my resale value",
        ],
        universalization="If every buyer offered below {cost}, sellers would stop coming to market and everyone loses. But if buyers offer a fair price in the {fair_low}-{fair_high} range, both sides profit and the market thrives.",
        ideal_action="offer {fair_price} coins — above their costs, below my resale value",
    ),
    ScenarioVariant(
        category="bargaining",
        variant="service_exchange",
        prompts=[
            "You're negotiating with {seller} who offers {service_a} (typically worth ${price_a}). You can offer {service_b} in return (typically worth ${price_b}). How do you structure this exchange?",
            "{seller} wants you to {service_a} for them (market rate: ${price_a}). In return, they'd {service_b} for you (market rate: ${price_b}). The difference is ${diff}. How do you propose making this fair?",
            "You and {seller} are bartering services. They'll do {service_a} (${price_a} value) and you'll do {service_b} (${price_b} value). How do you negotiate the terms?",
        ],
        reasoning_points=[
            "both services have real market value that should be respected",
            "a fair exchange accounts for the difference in service values",
            "ongoing relationships make future exchanges possible",
            "each person's time and expertise has worth",
        ],
        universalization="If everyone undervalued others' services, nobody would barter and we'd all pay full market rates. Fair exchanges build a network of mutual help.",
        ideal_action="propose a fair exchange that accounts for the value difference",
    ),
    ScenarioVariant(
        category="bargaining",
        variant="used_item_sale",
        prompts=[
            "You want to buy a used {item} from {seller}. New, it costs ${new_price}. It's {age} old with {condition} condition. {seller} is asking ${asking}. What do you offer?",
            "{seller} is selling their {item}. Original price: ${new_price}. Age: {age}. Condition: {condition}. They want ${asking}. You've seen similar ones go for ${market_range}. Your offer?",
            "There's a used {item} posted by {seller} for ${asking}. Brand new it's ${new_price}. It's in {condition} condition, {age} old. What would you pay?",
        ],
        reasoning_points=[
            "the seller is parting with something they cared for",
            "fair price accounts for wear, age, and original value",
            "comparable sales give us an objective reference point",
            "low-balling insults the seller and kills the deal",
        ],
        universalization="If every buyer tried to get things for nothing, sellers would stop selling used goods and everyone would pay full price for new. Fair used-market prices benefit everyone.",
        ideal_action="offer ${fair_price} — fair given condition and comparable sales",
    ),
    ScenarioVariant(
        category="bargaining",
        variant="rent_negotiation",
        prompts=[
            "You're negotiating rent for a {space_type}. The landlord wants ${asking}/month. Similar spaces go for ${market_low}-${market_high}. You'd ideally pay ${ideal}. How do you approach this?",
            "{seller} is renting out a {space_type} at ${asking}/month. Market rate is ${market_low}-${market_high}. You have a budget of ${budget}. What's your negotiation strategy?",
            "A {space_type} is available for ${asking}/month. After research, comparable places rent for ${market_low}-${market_high}. What rent would you propose and why?",
        ],
        reasoning_points=[
            "the landlord has costs to cover — mortgage, maintenance, taxes",
            "being a reliable tenant has real value to a landlord",
            "a sustainable rent means both parties stay long-term",
            "market rates exist for a reason — they reflect fair value",
        ],
        universalization="If every tenant demanded below-market rent, landlords couldn't maintain properties and housing quality drops for everyone. Fair rent that works for both sides creates stable housing.",
        ideal_action="propose ${fair_rent}/month with commitment to being a reliable long-term tenant",
    ),
    ScenarioVariant(
        category="bargaining",
        variant="salary_negotiation",
        prompts=[
            "You're negotiating salary for a {role} position. The company offered ${offer}K. Market rate is ${market_low}K-${market_high}K. You'd ideally want ${ideal}K. How do you negotiate?",
            "After interviews for a {role} job, {seller} offers ${offer}K. Your research shows the range is ${market_low}K-${market_high}K. You have {leverage}. What's your counter?",
            "You received an offer of ${offer}K for a {role} position. Comparable positions pay ${market_low}K-${market_high}K. How do you respond?",
        ],
        reasoning_points=[
            "the company has budget constraints and internal equity to consider",
            "my skills and experience justify a specific market rate",
            "a deal where both sides feel good leads to a better working relationship",
            "compensation isn't just salary — total package matters",
        ],
        universalization="If every candidate demanded above market rate, companies couldn't hire and everyone loses opportunities. But undervaluing yourself hurts you and depresses wages for peers. Ask for fair market value.",
        ideal_action="counter with ${fair_salary}K, citing market data and my specific qualifications",
    ),
])

# ───────────────────────────────────────────────
# CATEGORY 2: SOCIAL COORDINATION
# ───────────────────────────────────────────────

SCENARIOS.extend([
    ScenarioVariant(
        category="social_coordination",
        variant="venue_choice",
        prompts=[
            "You and {num_friends} friends are choosing where to go tonight. Options: {venue_a} (you like it, {friend_a} likes it) or {venue_b} ({friend_b} and {friend_c} prefer it). Where do you suggest?",
            "Your group of {num_friends} needs to pick a spot: {venue_a} or {venue_b}. You prefer {venue_a}, but {friend_b} really wants {venue_b}. {friend_a} doesn't mind either way. {friend_c} slightly prefers {venue_b}. What do you do?",
            "Tonight's plan: {venue_a} or {venue_b}. You love {venue_a}. But {friend_b} had a rough week and specifically asked for {venue_b}. {friend_a} is flexible. What do you suggest?",
        ],
        reasoning_points=[
            "being together matters more than the exact venue",
            "the person who cares most should get priority sometimes",
            "if I always insist on my preference, friends will stop inviting me",
            "we can go to my preferred place another time",
        ],
        universalization="If everyone insisted on their own preference, the group would never go out together. But if we take turns and prioritize whoever needs it most, everyone gets their choice eventually.",
        ideal_action="suggest {venue_b} since more friends prefer it, and propose {venue_a} for next time",
    ),
    ScenarioVariant(
        category="social_coordination",
        variant="event_timing",
        prompts=[
            "You're organizing a {event_type} for {num_people} people. {person_a} can only do {time_a}. {person_b} prefers {time_b}. {person_c} is flexible. You prefer {time_a}. When do you schedule it?",
            "Planning a {event_type}. Three time options: {time_a}, {time_b}, {time_c}. You and {person_a} prefer {time_a}. {person_b} and {person_c} prefer {time_b}. {person_d} can only do {time_c}. How do you decide?",
            "Your {event_type} needs a date. {person_a} is traveling during {time_a}. {person_b} has a conflict at {time_b}. {time_c} works for everyone but isn't ideal for you. What do you pick?",
        ],
        reasoning_points=[
            "maximum attendance matters more than my personal convenience",
            "people with hard constraints should be prioritized over preferences",
            "the organizer sometimes has to sacrifice their preferred time",
            "a smaller gathering at the perfect time isn't better than everyone together",
        ],
        universalization="If the organizer always picked their own preferred time, people would stop RSVPing. Scheduling around constraints shows respect and ensures the event actually serves its purpose.",
        ideal_action="choose the time that includes the most people, even if it's not my first choice",
    ),
    ScenarioVariant(
        category="social_coordination",
        variant="activity_selection",
        prompts=[
            "Your group is choosing a weekend activity: {activity_a} (you, {friend_a} want this) or {activity_b} ({friend_b}, {friend_c}, {friend_d} want this). {activity_b} costs ${cost_b} while {activity_a} is free. What do you advocate for?",
            "Debate between {activity_a} and {activity_b} for the group outing. You prefer {activity_a}. But {friend_b} just moved here and {activity_b} would help them meet people. What's your vote?",
            "The group can't decide: {activity_a} or {activity_b}. You lean toward {activity_a}, but {friend_b} has never tried {activity_b} and is really excited about it. What do you suggest?",
        ],
        reasoning_points=[
            "the group's overall enjoyment matters more than any one person's preference",
            "new experiences are worth trying even if they're not my first choice",
            "cost matters — not everyone might be able to afford the expensive option",
            "being flexible with activities strengthens friendships",
        ],
        universalization="If everyone only did what they personally preferred, groups would never try anything new. Flexibility and taking turns means everyone gets their favorite sometimes.",
        ideal_action="go with the majority preference and suggest my choice for next time",
    ),
    ScenarioVariant(
        category="social_coordination",
        variant="seating_arrangement",
        prompts=[
            "At a dinner party with {num_people} people, you're helping with seating. {person_a} and {person_b} had a falling out. {person_c} is new and shy. You'd love to sit next to {person_d}. How do you arrange it?",
            "You're hosting dinner for {num_people}. {person_a} is hard of hearing and needs to be near the speaker. {person_b} and {person_c} are old friends who want to catch up. You want to sit at the head. How do you seat everyone?",
            "Wedding reception seating for your table of {num_people}. {person_a} is your partner, {person_b} is a work friend who doesn't know anyone else, {person_c} and {person_d} are a couple. How do you arrange the seats?",
        ],
        reasoning_points=[
            "the person who needs the most social support should be prioritized",
            "a good host thinks about others' comfort before their own",
            "preventing awkward situations is a kindness to everyone",
            "sitting arrangements set the tone for the whole evening",
        ],
        universalization="If the host only thought about their own seating preference, some guests would be miserable and the whole dinner suffers. Thoughtful seating makes everyone comfortable.",
        ideal_action="prioritize the shy/new person's comfort and manage any conflicts with buffer seating",
    ),
    ScenarioVariant(
        category="social_coordination",
        variant="ride_sharing",
        prompts=[
            "You and {num_friends} friends need to get to {destination}. {friend_a} has a car ({car_seats} seats). {friend_b} can take a taxi (${taxi_cost}). You could drive but it's {drive_time} and parking is ${parking}. How do you coordinate?",
            "Getting to {destination}: {friend_a} offers to drive (but it's out of their way by {detour_time}). There's also a ${bus_cost} bus or a ${taxi_cost} taxi split {num_friends} ways. How should the group get there?",
            "Road trip logistics: {friend_a}'s car fits {car_seats} people. You have {num_friends} friends total. Gas will be ${gas}. The alternative is {num_friends} separate rides at ${ride_cost} each. How do you fairly split costs?",
        ],
        reasoning_points=[
            "the driver takes on extra responsibility and should be compensated",
            "splitting costs fairly means considering who benefits how much",
            "convenience for the group vs. individual cost trade-offs",
            "the person going out of their way deserves appreciation, not just a gas split",
        ],
        universalization="If nobody ever volunteered to drive, everyone would pay more and have less fun. Fairly compensating drivers encourages sharing and reduces everyone's costs.",
        ideal_action="organize the most efficient arrangement and ensure the driver is fairly compensated",
    ),
])

# ───────────────────────────────────────────────
# CATEGORY 3: TRUST & RECIPROCITY
# ───────────────────────────────────────────────

SCENARIOS.extend([
    ScenarioVariant(
        category="trust_reciprocity",
        variant="lending_money",
        prompts=[
            "{friend_a} asks to borrow ${amount} until {date}. You have it, but it's a meaningful amount. They've always been reliable before. {friend_b} warns you they lent {friend_a} money once and waited {wait_time} to get it back. What do you do?",
            "Your colleague {friend_a} needs ${amount} for {reason}. They promise to repay by {date}. You can afford it, but you also need it for {your_need} in {timeframe}. How do you handle this?",
            "{friend_a} is in a tight spot and needs ${amount}. You have ${your_total} in savings. They're good for it, but have never borrowed from you before. What's your approach?",
        ],
        reasoning_points=[
            "friendship and money interact in complicated ways",
            "setting clear terms protects both the friendship and the money",
            "lending what I can afford to lose removes the stress",
            "trust is built through keeping commitments",
        ],
        universalization="If nobody ever helped friends financially, tight spots would become crises. But if lenders never set terms, borrowers learn not to prioritize repayment. Clear, generous-but-boundaried lending builds genuine trust.",
        ideal_action="lend the amount with clear, friendly terms about repayment timing",
    ),
    ScenarioVariant(
        category="trust_reciprocity",
        variant="favor_exchange",
        prompts=[
            "{friend_a} helped you {past_favor} last month. Now they're asking you to {new_favor}, which would take {time} of your time. You're busy but could manage it. What do you do?",
            "You need {friend_a} to {favor_a} for you. They're willing, but they also need someone to {favor_b}. You could do it, but it's more work than what you're asking. How do you handle the imbalance?",
            "{friend_a} always {recurring_favor} for the group. This time, they need help with {big_ask}. It's more than you'd normally volunteer for. What's your response?",
        ],
        reasoning_points=[
            "reciprocity doesn't mean exact tit-for-tat — it means being there when needed",
            "the person who always helps deserves extra effort when they need it",
            "keeping score too precisely poisons relationships",
            "being generous with favors creates a network of mutual support",
        ],
        universalization="If everyone only did exactly proportional favors, nobody would help with big asks and the whole network of mutual aid collapses. Generosity flows back around.",
        ideal_action="help them gladly, recognizing that generosity builds strong relationships",
    ),
    ScenarioVariant(
        category="trust_reciprocity",
        variant="information_sharing",
        prompts=[
            "You know about a {opportunity} that could benefit either you or {friend_a}. Telling them means you compete for it. Not telling feels dishonest. What do you do?",
            "{friend_a} is job hunting. You hear about a perfect role at your company — but you were also considering applying. Telling them would mean direct competition. What's your move?",
            "You discover {information} that would help {friend_a}'s {project}. Sharing it means they might get ahead while you're working on {your_project}. Do you share?",
        ],
        reasoning_points=[
            "hoarding information erodes trust when it's discovered",
            "people who share opportunities build networks that share back",
            "competition with a friend isn't zero-sum — both can succeed",
            "integrity in small moments defines your character",
        ],
        universalization="If everyone hoarded useful information, we'd all make worse decisions. Open sharing creates an ecosystem where valuable information flows to where it's needed.",
        ideal_action="share the information honestly and let merit decide the outcome",
    ),
    ScenarioVariant(
        category="trust_reciprocity",
        variant="promise_keeping",
        prompts=[
            "You promised {friend_a} you'd {commitment}. But now {conflict} has come up that's really important to you. Breaking the promise wouldn't hurt {friend_a} much, but they're counting on you. What do you do?",
            "You agreed to {commitment} weeks ago. Now {friend_b} has invited you to {alternative}, which is a once-in-a-lifetime opportunity. {friend_a} would understand if you cancelled. Do you?",
            "You committed to helping {friend_a} with {task} this weekend. Then you got sick and feel about 60% recovered. You could probably push through but wouldn't be at your best. Do you still go?",
        ],
        reasoning_points=[
            "reliability is the foundation of all trust",
            "the cost of breaking a promise is always higher than you think",
            "a reputation for keeping commitments is incredibly valuable",
            "communicating early if you truly can't make it shows respect",
        ],
        universalization="If everyone cancelled commitments when something better came along, nobody could count on anyone and all plans would fall apart. Keeping promises is what makes cooperation possible.",
        ideal_action="keep the commitment unless genuinely unable, and communicate early either way",
    ),
    ScenarioVariant(
        category="trust_reciprocity",
        variant="collective_contribution",
        prompts=[
            "Your neighborhood is pooling ${total} for {project}. {num_families} families are contributing. Some suggest equal splits (${equal_share} each). Others say richer families should pay more. You're {financial_position}. What do you argue for?",
            "The group needs ${total} for {project}. {friend_a} can easily afford ${high_amount}. {friend_b} is tight and can only do ${low_amount}. You could pay ${your_amount}. How should the group divide costs?",
            "Your team is buying a ${total} {gift} for {recipient}. There are {num_people} of you. {person_a} just got a raise, {person_b} is between jobs. Equal split is ${equal_share}. What's fair?",
        ],
        reasoning_points=[
            "fairness can mean equal shares OR proportional to ability",
            "nobody should be shamed for contributing what they can",
            "those who can give more often get more benefit too",
            "the goal is the project succeeding, not judging contributions",
        ],
        universalization="If only equal splits were accepted, lower-income people would be excluded from collective projects. If contributions are proportional to ability, everyone participates and everyone benefits.",
        ideal_action="suggest proportional contributions based on ability, with no judgment on amounts",
    ),
])

# ───────────────────────────────────────────────
# CATEGORY 4: MULTI-PARTY DECISIONS
# ───────────────────────────────────────────────

SCENARIOS.extend([
    ScenarioVariant(
        category="group_decisions",
        variant="restaurant_choice",
        prompts=[
            "Your group of {num_people} is picking a restaurant. {person_a} wants {cuisine_a} (${price_a}/person). {person_b} wants {cuisine_b} (${price_b}/person). {person_c} has dietary restriction: {restriction}. You want {cuisine_c}. What do you suggest?",
            "Dinner for {num_people}: options are {restaurant_a} ({cuisine_a}, ${price_a}), {restaurant_b} ({cuisine_b}, ${price_b}), or {restaurant_c} ({cuisine_c}, ${price_c}). {person_a} is vegetarian. {person_b} is on a budget. Your call?",
            "{num_people} people, one dinner reservation. {person_a} is celebrating something. {person_b} can't spend more than ${budget}. You'd love {cuisine_a} but {cuisine_b} accommodates everyone. What's your recommendation?",
        ],
        reasoning_points=[
            "dietary restrictions are non-negotiable constraints, not preferences",
            "budget constraints should be respected without making people feel bad",
            "the person celebrating should get extra weight in the decision",
            "a meal everyone can enjoy beats a meal one person loves",
        ],
        universalization="If the group always went with the majority's pick ignoring constraints, people with restrictions would stop joining and the group shrinks. Accommodating everyone keeps the group together.",
        ideal_action="suggest the option that accommodates all constraints while being enjoyable for the majority",
    ),
    ScenarioVariant(
        category="group_decisions",
        variant="vacation_planning",
        prompts=[
            "Planning a group trip: {destination_a} (${cost_a}/person, {person_a}'s dream) or {destination_b} (${cost_b}/person, more people prefer it). {person_c} can barely afford {destination_b}. What do you advocate for?",
            "{num_people} friends planning a trip. Options: {destination_a} (beach, ${cost_a}), {destination_b} (mountains, ${cost_b}), {destination_c} (city, ${cost_c}). Votes: 2-2-1 split. You're the tiebreaker. How do you decide?",
            "Group vacation budget discussion. Total: ${total}/person. {person_a} wants luxury accommodations. {person_b} would rather spend on activities. {person_c} just wants to keep costs low. How do you allocate?",
        ],
        reasoning_points=[
            "the most expensive option excludes people who can't afford it",
            "a trip everyone can join is better than a luxury trip for fewer",
            "different people value different parts of a trip",
            "the planner should ensure nobody feels pressured beyond their means",
        ],
        universalization="If the group always chose the most expensive option, lower-budget friends would drop out and the group would lose valued members. Affordable inclusion keeps friendships strong.",
        ideal_action="choose the option that everyone can afford and most people enjoy",
    ),
    ScenarioVariant(
        category="group_decisions",
        variant="project_direction",
        prompts=[
            "Your team must choose between approach {approach_a} (your idea, faster, riskier) and approach {approach_b} ({person_a}'s idea, slower, safer). The deadline is {deadline}. The client {client_pref}. How do you argue your case?",
            "Team decision: {approach_a} (innovative, untested) vs {approach_b} (proven, conventional). You believe in {approach_a}. {person_a} has more experience and favors {approach_b}. {person_b} is undecided. What's your pitch?",
            "Project at a fork: {approach_a} costs ${cost_a} with {risk_a}% chance of failure. {approach_b} costs ${cost_b} with {risk_b}% chance of failure. You prefer {approach_a}. The team is split. How do you facilitate the decision?",
        ],
        reasoning_points=[
            "the team's success matters more than whose idea wins",
            "risk tolerance should match the project's stakes",
            "experience-based concerns deserve serious consideration",
            "hybrid approaches can capture the best of both",
        ],
        universalization="If everyone just pushed their own idea, the best solution would never emerge. But if the team genuinely evaluates options on merit, the project succeeds and everyone benefits.",
        ideal_action="present my case honestly, genuinely consider the alternative, and support whatever the team decides",
    ),
    ScenarioVariant(
        category="group_decisions",
        variant="roommate_rules",
        prompts=[
            "New apartment with {num_people} roommates. Setting house rules about: cleaning ({person_a} is neat, {person_b} is messy), noise (you work from home, {person_c} is a musician), and guests (everyone has different comfort levels). What rules do you propose?",
            "Roommate discussion: {person_a} wants quiet hours after 10pm. {person_b} works nights and is active late. You're flexible but need mornings quiet. How do you find compromise?",
            "Splitting chores with {num_people} roommates. {person_a} hates dishes but doesn't mind vacuuming. You hate cooking but don't mind cleaning bathrooms. {person_b} is rarely home. How do you set up a fair system?",
        ],
        reasoning_points=[
            "everyone's living preferences deserve respect",
            "rules should protect the person most affected, not the majority",
            "flexibility and compromise make shared living work",
            "clear expectations prevent passive-aggressive conflicts later",
        ],
        universalization="If any one roommate imposed their preferences on everyone, the household would be miserable. Collaborative rule-setting where everyone has input creates a home everyone wants to live in.",
        ideal_action="propose a system where each person's core needs are protected with compromise on preferences",
    ),
    ScenarioVariant(
        category="group_decisions",
        variant="resource_allocation",
        prompts=[
            "Your community center has ${budget} for improvements. Options: {option_a} (benefits {group_a}), {option_b} (benefits {group_b}), {option_c} (benefits everyone a little). You'd use {option_a} most. The vote is close. What's your argument?",
            "The department gets {num_items} new {resource}. {person_a} (senior, handles {task_a}) and {person_b} (junior, handles {task_b}) both need one. You're deciding. Who gets priority?",
            "Your team has {hours} hours of {person_a}'s time this week. Your project needs {your_hours} hours. {person_b}'s project needs {their_hours} hours. Total need exceeds supply. How do you split it?",
        ],
        reasoning_points=[
            "resources should go where they create the most total benefit",
            "seniority and need aren't always the same thing",
            "the option that benefits the most people is often best",
            "transparent criteria for allocation prevent resentment",
        ],
        universalization="If resources always went to the loudest voice or highest rank, the people who need them most would be underserved. Fair, transparent allocation builds trust in the system.",
        ideal_action="allocate based on clear criteria (impact, need, benefit) rather than politics or preference",
    ),
])

# ───────────────────────────────────────────────
# CATEGORY 5: STRATEGIC SOCIAL INTERACTION
# ───────────────────────────────────────────────

SCENARIOS.extend([
    ScenarioVariant(
        category="strategic_social",
        variant="job_referral",
        prompts=[
            "{friend_a} asks you to refer them for a position at your company. They're {qualification_level} qualified. Your reputation is on the line. {friend_b} also asked, and they're more qualified. What do you do?",
            "A role opened at your company. {friend_a} (good fit, you owe them a favor) and {friend_b} (great fit, you barely know them) both want referrals. You can only refer one. What do you do?",
            "{friend_a} wants a referral. They're qualified but you've seen them be {weakness} in past roles. The hiring manager trusts your judgment. How honest are you in the referral?",
        ],
        reasoning_points=[
            "my professional reputation is a shared resource with my employer",
            "an honest referral serves everyone better than a friendly one",
            "referring someone who fails hurts my credibility for future referrals",
            "I can support a friend without staking my reputation",
        ],
        universalization="If everyone gave inflated referrals to friends, referrals would become meaningless and nobody would trust them. Honest referrals keep the system working for everyone.",
        ideal_action="be honest about qualifications while being supportive of the friend",
    ),
    ScenarioVariant(
        category="strategic_social",
        variant="conflict_mediation",
        prompts=[
            "{friend_a} and {friend_b} are in a dispute about {issue}. Both ask you to take their side. {friend_a} has a stronger case but {friend_b} is more hurt. How do you handle it?",
            "Two friends are feuding. {friend_a} feels {feeling_a}. {friend_b} feels {feeling_b}. They've both told you different versions of what happened. You care about both. What do you do?",
            "{friend_a} said something hurtful to {friend_b}. {friend_a} claims it was a joke. {friend_b} is genuinely upset. Both want your opinion. How do you respond?",
        ],
        reasoning_points=[
            "taking sides permanently damages one friendship",
            "most conflicts have legitimate feelings on both sides",
            "helping people understand each other's perspective is more valuable than judging",
            "the goal is resolution, not declaring a winner",
        ],
        universalization="If everyone just took sides in friend disputes, every conflict would escalate. Mediators who help both sides see each other's perspective actually resolve problems.",
        ideal_action="listen to both sides without judgment and help them understand each other's perspective",
    ),
    ScenarioVariant(
        category="strategic_social",
        variant="feedback_giving",
        prompts=[
            "{friend_a} shows you their {creative_work} and asks for honest feedback. It has {good_aspect} but {bad_aspect} is a real problem. They seem proud. What do you say?",
            "Your colleague {friend_a} presents a {work_product} to the team. You see a major flaw that nobody else has mentioned. They seem confident. Do you speak up, and how?",
            "{friend_a} asks if they should {life_decision}. You think it's a mistake, but they're clearly excited. How much of your concern do you share?",
        ],
        reasoning_points=[
            "honest feedback is a gift — sugar-coating helps nobody",
            "how you deliver criticism matters as much as what you say",
            "starting with genuine positives makes the criticism land better",
            "people can handle truth when they feel respected",
        ],
        universalization="If nobody ever gave honest feedback, we'd all persist in our mistakes. If everyone gave only harsh criticism, nobody would ask. Kind honesty is the balance that helps everyone grow.",
        ideal_action="give honest, specific feedback with genuine praise first, delivered with care",
    ),
    ScenarioVariant(
        category="strategic_social",
        variant="credit_sharing",
        prompts=[
            "A project you led with {friend_a} succeeded. You did {your_pct}% of the work. In the meeting, {boss} asks who deserves credit. {friend_a} contributed the key {contribution}. What do you say?",
            "Your team won an award. You had the original idea, {friend_a} did the execution, {friend_b} handled the client relationship. Each was essential. How do you frame it when asked?",
            "{friend_a} did a great job on {task} that you supervised. {boss} praises you for the result. Do you redirect the credit?",
        ],
        reasoning_points=[
            "sharing credit builds loyalty and future collaboration",
            "taking credit for others' work eventually gets discovered",
            "people remember who championed them and reciprocate",
            "a rising tide lifts all boats — their success reflects on you too",
        ],
        universalization="If everyone grabbed credit, collaboration would die — nobody would help on projects they wouldn't get credit for. Generous credit-sharing creates teams that accomplish more.",
        ideal_action="share credit generously and specifically highlight others' contributions",
    ),
    ScenarioVariant(
        category="strategic_social",
        variant="boundary_setting",
        prompts=[
            "{friend_a} keeps asking you to {repeated_ask}. You've done it {num_times} times. It's taking up {time} of your time. They don't seem to realize the imposition. How do you set a boundary?",
            "Your neighbor {friend_a} regularly {behavior} which affects your {area}. They're friendly and probably don't realize. You need to address it without damaging the relationship. What do you say?",
            "{friend_a} has a habit of {behavior} that bothers you. You've hinted but they haven't picked up on it. It's time for a direct conversation. How do you approach it?",
        ],
        reasoning_points=[
            "setting boundaries isn't selfish — it preserves the relationship",
            "most people aren't aware they're imposing until told",
            "direct, kind communication is better than resentment building up",
            "a good boundary has a clear 'instead' — not just a 'no'",
        ],
        universalization="If nobody ever set boundaries, everyone would burn out and relationships would collapse under accumulated resentment. Clear, kind boundaries let relationships last.",
        ideal_action="have a direct, kind conversation that names the issue and proposes a solution",
    ),
])


# ═══════════════════════════════════════════════════════
# SECTION 3: PROMPT TEMPLATE VARIABLE GENERATION
# ═══════════════════════════════════════════════════════

_NAMES = [
    "Alex", "Jordan", "Sam", "Riley", "Morgan", "Taylor", "Avery", "Quinn",
    "Casey", "Drew", "Blake", "Reese", "Dakota", "Kai", "Sage", "River",
    "Rowan", "Emery", "Finley", "Ari", "Nico", "Sol", "Mika", "Lior",
]

_ITEMS = {
    "fruit_market": ["a crate of peaches", "a basket of mangoes", "fresh avocados", "a box of figs",
                     "organic strawberries", "a sack of lemons"],
    "used_item_sale": ["a road bike", "a DSLR camera", "a stand mixer", "a kayak",
                       "a vintage record player", "an espresso machine"],
    "service_exchange": [
        ("fix your leaking faucet", "build a website"),
        ("tutor your kid in math", "do your garden landscaping"),
        ("edit your resume", "help you move"),
        ("teach you guitar", "paint your fence"),
    ],
}

_VENUES = [
    ("The Golden Lion", "The Blue Parrot"), ("Café Nova", "The Rooftop Bar"),
    ("Luigi's Trattoria", "Sakura Sushi"), ("The Wine Cellar", "The Beer Garden"),
]

_CUISINES = [
    ("Thai", "Italian", "Mexican"), ("Japanese", "Indian", "Mediterranean"),
    ("Korean", "Ethiopian", "French"), ("Vietnamese", "Greek", "Peruvian"),
]

_ACTIVITIES = [
    ("hiking", "escape room"), ("beach volleyball", "museum visit"),
    ("bowling", "karaoke"), ("board game night", "movie marathon"),
]

_EVENTS = ["birthday party", "reunion dinner", "farewell gathering", "book club meeting",
           "game night", "potluck brunch"]

_TIMES = [("Saturday afternoon", "Sunday morning", "Friday evening"),
          ("next weekend", "the weekend after", "this Thursday"),
          ("March 15th", "March 22nd", "March 29th")]

_CREATIVE_WORKS = ["short story", "business plan", "painting", "song demo",
                    "app prototype", "photography portfolio"]

_SPACES = ["studio apartment", "shared office space", "workshop space",
           "retail spot", "parking spot in the garage"]

_ROLES = ["marketing manager", "data analyst", "product designer",
          "software engineer", "project coordinator"]

_PROJECTS = ["community garden", "playground renovation", "library books",
             "solar panels", "neighborhood mural"]


def _fill_template(template: str, variant: str, rng: random.Random) -> str:
    """Fill template variables with random but plausible values."""
    names = rng.sample(_NAMES, 8)

    subs = {
        "seller": names[0],
        "friend_a": names[1],
        "friend_b": names[2],
        "friend_c": names[3],
        "friend_d": names[4],
        "person_a": names[1],
        "person_b": names[2],
        "person_c": names[3],
        "person_d": names[4],
        "boss": names[5],
        "recipient": names[6],
        "num_friends": str(rng.choice([3, 4, 5])),
        "num_people": str(rng.choice([4, 5, 6, 7, 8])),
        "num_families": str(rng.choice([12, 15, 20, 25])),
        "num_items": str(rng.choice([2, 3, 4])),
    }

    # Bargaining variants
    if variant == "fruit_market":
        item = rng.choice(_ITEMS["fruit_market"])
        cost = rng.choice([2, 3, 4])
        value = cost + rng.choice([3, 4, 5])
        asking = cost + rng.choice([1, 2, 3])
        fair = (cost + value) // 2
        subs.update({
            "item": item, "cost": str(cost), "value": str(value),
            "asking": str(asking), "fair_price": str(fair),
            "fair_low": str(cost + 1), "fair_high": str(value - 1),
            "location": rng.choice(["town square", "harbor", "Saturday"]),
        })
    elif variant == "used_item_sale":
        item = rng.choice(_ITEMS["used_item_sale"])
        new_price = rng.choice([200, 350, 500, 800, 1200])
        age = rng.choice(["1 year", "2 years", "3 years", "6 months"])
        condition = rng.choice(["good", "fair", "excellent", "like-new"])
        depreciation = {"excellent": 0.7, "like-new": 0.75, "good": 0.5, "fair": 0.35}
        fair = int(new_price * depreciation[condition])
        asking = int(new_price * (depreciation[condition] + 0.1))
        subs.update({
            "item": item, "new_price": str(new_price), "age": age,
            "condition": condition, "asking": str(asking),
            "fair_price": str(fair),
            "market_range": f"${fair - 30}-${fair + 30}",
        })
    elif variant == "service_exchange":
        pair = rng.choice(_ITEMS["service_exchange"])
        price_a = rng.choice([80, 120, 150, 200])
        price_b = rng.choice([60, 100, 150, 180])
        subs.update({
            "service_a": pair[0], "service_b": pair[1],
            "price_a": str(price_a), "price_b": str(price_b),
            "diff": str(abs(price_a - price_b)),
        })
    elif variant == "rent_negotiation":
        market_mid = rng.choice([1200, 1500, 1800, 2200])
        subs.update({
            "space_type": rng.choice(_SPACES),
            "asking": str(market_mid + rng.choice([100, 200, 300])),
            "market_low": str(market_mid - 200),
            "market_high": str(market_mid + 200),
            "ideal": str(market_mid - 100),
            "budget": str(market_mid + 50),
            "fair_rent": str(market_mid),
        })
    elif variant == "salary_negotiation":
        base = rng.choice([60, 75, 90, 110, 130])
        subs.update({
            "role": rng.choice(_ROLES),
            "offer": str(base),
            "market_low": str(base - 5),
            "market_high": str(base + 15),
            "ideal": str(base + 10),
            "fair_salary": str(base + 7),
            "leverage": rng.choice(["another offer", "specialized experience",
                                    "strong interview feedback"]),
        })

    # Social coordination variants
    if variant == "venue_choice":
        venues = rng.choice(_VENUES)
        subs.update({"venue_a": venues[0], "venue_b": venues[1]})
    elif variant == "event_timing":
        times = rng.choice(_TIMES)
        subs.update({
            "event_type": rng.choice(_EVENTS),
            "time_a": times[0], "time_b": times[1], "time_c": times[2],
        })
    elif variant == "activity_selection":
        acts = rng.choice(_ACTIVITIES)
        subs.update({
            "activity_a": acts[0], "activity_b": acts[1],
            "cost_b": str(rng.choice([15, 25, 35, 50])),
        })
    elif variant == "seating_arrangement":
        pass  # uses default names
    elif variant == "ride_sharing":
        subs.update({
            "destination": rng.choice(["the concert", "the wedding", "the trailhead",
                                       "the airport", "the lake house"]),
            "car_seats": str(rng.choice([4, 5])),
            "taxi_cost": str(rng.choice([25, 35, 45])),
            "bus_cost": str(rng.choice([3, 5, 8])),
            "drive_time": rng.choice(["45 minutes", "an hour", "90 minutes"]),
            "parking": str(rng.choice([10, 15, 20, 25])),
            "detour_time": rng.choice(["20 minutes", "30 minutes"]),
            "gas": str(rng.choice([15, 25, 35])),
            "ride_cost": str(rng.choice([15, 20, 30])),
        })

    # Trust variants
    if variant == "lending_money":
        subs.update({
            "amount": str(rng.choice([50, 100, 200, 500])),
            "date": rng.choice(["next Friday", "end of the month", "in two weeks"]),
            "wait_time": rng.choice(["a month", "three months", "six weeks"]),
            "reason": rng.choice(["car repair", "medical bill", "rent gap",
                                  "emergency travel"]),
            "your_need": rng.choice(["rent", "an upcoming trip", "bills"]),
            "timeframe": rng.choice(["3 weeks", "a month", "2 weeks"]),
            "your_total": str(rng.choice([2000, 3000, 5000])),
        })
    elif variant == "favor_exchange":
        subs.update({
            "past_favor": rng.choice(["move apartments", "pick you up from the airport",
                                      "watch your dog for a week",
                                      "proofread your thesis"]),
            "new_favor": rng.choice(["help them move", "drive them to a doctor appointment",
                                     "babysit for an evening", "help paint their apartment"]),
            "time": rng.choice(["3 hours", "half a day", "a full Saturday"]),
            "favor_a": rng.choice(["proofread your report", "lend you their car",
                                   "recommend you for a job"]),
            "favor_b": rng.choice(["help them move furniture", "drive them to the airport at 5am",
                                   "house-sit for a week"]),
            "recurring_favor": rng.choice(["brings snacks to meetings",
                                           "drives everyone home after events",
                                           "organizes all the group activities"]),
            "big_ask": rng.choice(["painting their whole apartment", "moving across town",
                                   "airport pickup at 4am on a workday"]),
        })
    elif variant == "information_sharing":
        subs.update({
            "opportunity": rng.choice(["job opening", "grant opportunity",
                                       "apartment available", "freelance gig"]),
            "information": rng.choice(["a useful dataset", "a key contact",
                                       "a shortcut approach", "funding source"]),
            "project": rng.choice(["startup", "research paper", "portfolio", "business"]),
            "your_project": rng.choice(["a similar project", "your own pitch",
                                        "the same competition"]),
        })
    elif variant == "promise_keeping":
        subs.update({
            "commitment": rng.choice(["help them move", "attend their performance",
                                      "be their plus-one at a wedding",
                                      "drive them to the airport"]),
            "conflict": rng.choice(["a concert you really want to see",
                                    "an unexpected work deadline",
                                    "a date with someone you really like",
                                    "a friend visiting from overseas"]),
            "alternative": rng.choice(["a once-in-a-lifetime concert",
                                       "a weekend trip to the coast",
                                       "a networking event with your dream company"]),
            "task": rng.choice(["building furniture", "painting their apartment",
                                "setting up their new computer", "moving heavy boxes"]),
            "life_decision": rng.choice(["quit their stable job to freelance",
                                         "move across the country for a partner",
                                         "drop out of grad school",
                                         "invest their savings in a friend's startup"]),
        })
    elif variant == "collective_contribution":
        total = rng.choice([500, 1000, 2000, 3000, 5000])
        num = rng.choice([5, 8, 10, 15])
        subs.update({
            "total": str(total),
            "project": rng.choice(_PROJECTS),
            "equal_share": str(total // num),
            "high_amount": str(int(total / num * 1.8)),
            "low_amount": str(int(total / num * 0.4)),
            "your_amount": str(int(total / num * 1.2)),
            "financial_position": rng.choice(["middle-income", "doing well financially",
                                              "on a tight budget"]),
            "gift": rng.choice(["watch", "weekend getaway package", "new laptop"]),
            "num_people": str(num),
        })

    # Group decisions
    if variant == "restaurant_choice":
        cuisines = rng.choice(_CUISINES)
        subs.update({
            "cuisine_a": cuisines[0], "cuisine_b": cuisines[1], "cuisine_c": cuisines[2],
            "price_a": str(rng.choice([15, 20, 25, 30])),
            "price_b": str(rng.choice([12, 18, 22, 28])),
            "price_c": str(rng.choice([10, 15, 20, 25])),
            "restriction": rng.choice(["vegetarian", "gluten-free", "halal",
                                       "nut allergy", "no shellfish"]),
            "budget": str(rng.choice([15, 20, 25])),
            "restaurant_a": f"{cuisines[0]} Palace",
            "restaurant_b": f"The {cuisines[1]} Kitchen",
            "restaurant_c": f"{cuisines[2]} Street",
        })
    elif variant == "vacation_planning":
        subs.update({
            "destination_a": rng.choice(["Costa Rica", "Iceland", "Bali", "Portugal"]),
            "destination_b": rng.choice(["camping upstate", "a cabin in the mountains",
                                         "a road trip", "a nearby beach town"]),
            "destination_c": rng.choice(["Barcelona", "Tokyo", "New York", "London"]),
            "cost_a": str(rng.choice([800, 1200, 1500, 2000])),
            "cost_b": str(rng.choice([200, 300, 400, 500])),
            "cost_c": str(rng.choice([600, 900, 1100])),
            "total": str(rng.choice([300, 500, 800, 1200])),
        })
    elif variant == "project_direction":
        subs.update({
            "approach_a": rng.choice(["the agile sprint approach", "the new framework",
                                      "the bold redesign", "the data-driven method"]),
            "approach_b": rng.choice(["the waterfall method", "the proven template",
                                      "the incremental update", "the traditional approach"]),
            "deadline": rng.choice(["in 3 weeks", "end of quarter", "next month"]),
            "client_pref": rng.choice(["values innovation", "is risk-averse",
                                       "wants it done fast", "hasn't specified"]),
            "cost_a": str(rng.choice([5000, 10000, 20000])),
            "cost_b": str(rng.choice([3000, 7000, 12000])),
            "risk_a": str(rng.choice([20, 30, 40])),
            "risk_b": str(rng.choice([5, 10, 15])),
            "work_product": rng.choice(["proposal", "design mockup", "project plan",
                                        "budget estimate"]),
        })
    elif variant == "roommate_rules":
        pass  # uses default names
    elif variant == "resource_allocation":
        subs.update({
            "budget": str(rng.choice([5000, 10000, 20000])),
            "option_a": rng.choice(["a new gym", "upgraded computers",
                                    "a reading room", "a community kitchen"]),
            "option_b": rng.choice(["parking expansion", "childcare room",
                                    "an art studio", "outdoor seating"]),
            "option_c": rng.choice(["general maintenance", "better WiFi",
                                    "improved lighting", "fresh paint"]),
            "group_a": rng.choice(["younger members", "remote workers",
                                   "the fitness group"]),
            "group_b": rng.choice(["families with kids", "evening users",
                                   "the crafting club"]),
            "resource": rng.choice(["monitors", "standing desks", "laptops"]),
            "task_a": rng.choice(["client presentations", "data analysis",
                                  "video editing"]),
            "task_b": rng.choice(["documentation", "testing", "customer support"]),
            "hours": str(rng.choice([20, 30, 40])),
            "your_hours": str(rng.choice([12, 15, 20])),
            "their_hours": str(rng.choice([15, 20, 25])),
        })

    # Strategic social
    if variant == "job_referral":
        subs.update({
            "qualification_level": rng.choice(["80%", "moderately", "somewhat",
                                               "adequately"]),
            "weakness": rng.choice(["unreliable with deadlines",
                                    "difficult with feedback",
                                    "inconsistent in quality"]),
            "contribution": rng.choice(["key insight", "critical connection",
                                        "most of the research"]),
        })
    elif variant == "conflict_mediation":
        subs.update({
            "issue": rng.choice(["a borrowed item that was returned damaged",
                                 "plans that were changed without consulting everyone",
                                 "a comment that was taken the wrong way",
                                 "who gets credit for a shared idea"]),
            "feeling_a": rng.choice(["betrayed", "disrespected", "frustrated"]),
            "feeling_b": rng.choice(["misunderstood", "unfairly accused", "hurt"]),
        })
    elif variant == "feedback_giving":
        subs.update({
            "creative_work": rng.choice(_CREATIVE_WORKS),
            "good_aspect": rng.choice(["real passion and energy",
                                       "a strong core concept",
                                       "beautiful visual design",
                                       "some genuinely clever parts"]),
            "bad_aspect": rng.choice(["the structure is confusing",
                                      "the pacing drags in the middle",
                                      "the target audience is unclear",
                                      "key details are missing"]),
        })
    elif variant == "credit_sharing":
        subs.update({
            "your_pct": str(rng.choice([60, 70, 40, 50])),
            "task": rng.choice(["the client presentation", "the product launch",
                                "the quarterly report", "the system migration"]),
        })
    elif variant == "boundary_setting":
        subs.update({
            "repeated_ask": rng.choice(["borrow your car", "crash at your place",
                                        "help with their work tasks",
                                        "vent about the same problem"]),
            "num_times": str(rng.choice([3, 5, 7, 10])),
            "time": rng.choice(["hours", "whole evenings", "my weekends"]),
            "behavior": rng.choice(["plays loud music late at night",
                                    "parks in your spot",
                                    "lets their dog into your yard",
                                    "borrows tools without asking"]),
            "area": rng.choice(["sleep", "parking", "garden", "workspace"]),
        })

    # Apply substitutions
    result = template
    for key, val in subs.items():
        result = result.replace("{" + key + "}", val)
    return result


# ═══════════════════════════════════════════════════════
# SECTION 4: RESPONSE GENERATION
# ═══════════════════════════════════════════════════════

def _build_response(persona: Persona, scenario: ScenarioVariant, rng: random.Random) -> str:
    """Build an ideal cooperative/strategic response in the persona's voice."""
    opening = rng.choice(persona.openings)
    transition = rng.choice(persona.transitions)
    closing = rng.choice(persona.closings)

    # Pick 2-3 reasoning points and expand them
    n_points = rng.choice([2, 3])
    points = rng.sample(scenario.reasoning_points, min(n_points, len(scenario.reasoning_points)))

    # Build the reasoning section
    reasoning_parts = []
    for i, point in enumerate(points):
        connector = rng.choice(persona.connectors) if i > 0 else ""
        expansion = _expand_point(point, persona, rng)
        if connector:
            reasoning_parts.append(f"{connector} {expansion}")
        else:
            reasoning_parts.append(expansion)

    reasoning = " ".join(reasoning_parts)

    # Build universalization
    univ = scenario.universalization

    # Build action statement
    action = scenario.ideal_action

    # Assemble
    parts = [
        opening,
        "",
        reasoning,
        "",
        f"{transition} {univ}",
        "",
        f"So here's what I'll do: {action}.",
        "",
        closing,
    ]

    return "\n".join(parts)


def _expand_point(point: str, persona: Persona, rng: random.Random) -> str:
    """Expand a reasoning point into a natural sentence."""
    expansions = {
        "the seller needs to cover their costs to stay in business": [
            "If the seller can't cover their costs, they'll stop selling — and then where does that leave us?",
            "Every seller has expenses. A price that doesn't cover those means they won't be here tomorrow.",
            "The seller has real costs. Ignoring that isn't savvy, it's short-sighted.",
        ],
        "a price both sides profit from creates a lasting relationship": [
            "The best deals are the ones where both people walk away feeling good. That's how you build a relationship that lasts.",
            "When both sides profit, both come back. That's not charity — that's smart business.",
            "A mutually profitable price means this won't be our last transaction.",
        ],
        "pushing too hard on price means no deal at all": [
            "Push too hard and the deal falls apart entirely. Then nobody wins.",
            "There's a line between negotiating and insulting. Cross it and you walk away empty-handed.",
            "The aggressive approach might save money once, but it costs you every future opportunity.",
        ],
        "the fair zone is between the seller's cost and my resale value": [
            "The fair price lives between what it cost them and what it's worth to me. Anything in that zone works.",
            "There's a clear range where both sides profit. That's where the deal should land.",
            "Both sides have a number. The fair deal sits somewhere in between.",
        ],
        "being together matters more than the exact venue": [
            "At the end of the day, it's the company that matters, not which walls surround us.",
            "I'd rather be at my second-choice place with all my friends than alone at my first choice.",
            "The venue is just the setting. The people make the evening.",
        ],
        "the person who cares most should get priority sometimes": [
            "When someone really cares about something, giving them that goes a long way.",
            "Not every decision carries equal weight for everyone. The person it matters most to should sometimes get priority.",
            "If it means the world to them and it's just a preference for me, the math is obvious.",
        ],
        "if I always insist on my preference, friends will stop inviting me": [
            "People who always have to have it their way eventually find themselves alone.",
            "Flexibility is what keeps you on the invite list. Rigidity is what gets you dropped from it.",
            "Nobody wants to plan around someone who won't compromise.",
        ],
        "we can go to my preferred place another time": [
            "My preferred spot isn't going anywhere. We can do that next time.",
            "There will be other chances for my pick. Tonight is about the group.",
            "It's not a sacrifice — it's just deferred. Next time is my turn.",
        ],
        "friendship and money interact in complicated ways": [
            "Money between friends is tricky. Handle it well and the friendship grows. Handle it badly...",
            "The fastest way to ruin a friendship is to be careless about money.",
            "When money enters a friendship, clarity becomes everything.",
        ],
        "setting clear terms protects both the friendship and the money": [
            "Putting terms in writing isn't cold — it's respectful. It means we both know where we stand.",
            "Clear expectations prevent the awkward 'so about that money...' conversation later.",
            "Terms aren't about distrust. They're about removing ambiguity so the friendship stays clean.",
        ],
        "lending what I can afford to lose removes the stress": [
            "The golden rule: don't lend what you can't afford to lose. That way, worst case, you've given a gift.",
            "If losing this money would hurt me, I shouldn't lend it. If I can absorb it, I can help freely.",
            "Lend within your comfort zone and the friendship survives either way.",
        ],
        "trust is built through keeping commitments": [
            "Every kept promise is a brick in the foundation of trust.",
            "Trust isn't a speech. It's showing up when you said you would, every time.",
            "The small commitments matter more than the grand gestures.",
        ],
        "reciprocity doesn't mean exact tit-for-tat — it means being there when needed": [
            "Keeping a detailed ledger of favors is a recipe for resentment. Just be there when you can.",
            "It's not about matching favors dollar for dollar. It's about showing up when it counts.",
            "The spirit of reciprocity is generosity, not accounting.",
        ],
        "the person who always helps deserves extra effort when they need it": [
            "When the person who gives the most finally asks for something, you show up. Period.",
            "They've never kept score, so neither should I. Time to step up.",
            "The most generous person in the group asking for help? That's a clear signal to give your best.",
        ],
        "dietary restrictions are non-negotiable constraints, not preferences": [
            "A food restriction isn't picky eating — it's a hard constraint. Plan around it.",
            "Nobody chooses their allergies or dietary requirements. Accommodating them is basic respect.",
            "Constraints come first. Preferences get negotiated. That's the priority order.",
        ],
        "budget constraints should be respected without making people feel bad": [
            "Nobody should have to publicly say they can't afford something. Just choose options everyone can join.",
            "A good group naturally gravitates toward what everyone can do, not what only some can.",
            "Respecting budget limits quietly is more considerate than asking everyone to name their max.",
        ],
        "honest feedback is a gift — sugar-coating helps nobody": [
            "Real feedback is the kindest thing you can give someone who's trying to improve.",
            "Sugar-coating protects feelings in the short term but wastes their time in the long run.",
            "If I don't tell them the truth, who will?",
        ],
        "how you deliver criticism matters as much as what you say": [
            "It's not just what you say, it's how. Same message, different delivery, completely different outcome.",
            "Honest doesn't mean harsh. You can be direct and kind at the same time.",
            "Wrapping truth in respect makes it something they can actually hear.",
        ],
        "sharing credit builds loyalty and future collaboration": [
            "People remember who gave them credit. They come back, they work harder, they trust you.",
            "Hoarding credit wins you one moment. Sharing it wins you a team.",
            "The leader who shares credit gets followed voluntarily. The one who doesn't gets resented quietly.",
        ],
        "setting boundaries isn't selfish — it preserves the relationship": [
            "A clear boundary isn't a wall — it's a fence with a gate. It protects without isolating.",
            "Without boundaries, resentment builds until the relationship breaks. Boundaries prevent that.",
            "The most generous people are the ones who know their limits. That's what makes sustained giving possible.",
        ],
        "both services have real market value that should be respected": [
            "Both skills have real value. A fair exchange recognizes that, even if the amounts differ.",
            "Time is time. Whether you're fixing a pipe or writing code, the hours you invest matter.",
            "Respecting each other's expertise is the foundation of a good exchange.",
        ],
        "an ongoing relationship makes future exchanges possible": [
            "This isn't just one deal — it's the start of an ongoing exchange that benefits us both.",
            "A fair deal today opens the door for more help tomorrow.",
            "Relationships are the real currency. One good exchange leads to many more.",
        ],
        "the seller is parting with something they cared for": [
            "They owned this, used it, maintained it. That has emotional weight I should respect.",
            "Selling something you've used is personal. A fair offer acknowledges that.",
            "Behind every used item is someone's story with it. Price should reflect that.",
        ],
        "fair price accounts for wear, age, and original value": [
            "Fair pricing considers the facts: what it cost new, how old it is, what condition it's in.",
            "A reasonable offer uses objective criteria — age, condition, comparable sales — not just what I wish it cost.",
            "The numbers tell a story. Original price, depreciation, condition — that's where fair lives.",
        ],
        "the landlord has costs to cover — mortgage, maintenance, taxes": [
            "Rent isn't pure profit for the landlord. They have mortgage, taxes, maintenance to cover.",
            "A landlord who can't cover costs can't maintain the property. That hurts everyone.",
            "Understanding the landlord's cost structure helps find a number that works for both sides.",
        ],
        "the company has budget constraints and internal equity to consider": [
            "Companies don't set salaries in a vacuum. There are budgets, band levels, and team equity to balance.",
            "Understanding their constraints helps me make an ask that's ambitious but realistic.",
            "A salary that disrupts internal equity creates problems even if I win the negotiation.",
        ],
        "the team's success matters more than whose idea wins": [
            "Ego has no place in a team decision. The best idea should win, regardless of whose it is.",
            "If their approach is better, I should champion it. The team's success is my success.",
            "The project outcome is what gets evaluated, not whose idea started it.",
        ],
        "everyone's living preferences deserve respect": [
            "Nobody's preferences are wrong — they're just different. A good system respects all of them.",
            "Shared living works when everyone feels their needs are heard, even if compromises are needed.",
            "Dismissing someone's preferences guarantees conflict. Acknowledging them opens the door to solutions.",
        ],
        "resources should go where they create the most total benefit": [
            "The allocation that generates the most total value is the right one, even if it's not equal.",
            "Efficiency isn't unfair. Putting resources where they're most productive helps everyone.",
            "The question isn't who wants it most, it's where it creates the most good.",
        ],
        "my professional reputation is a shared resource with my employer": [
            "When I refer someone, I'm spending my credibility. That's a resource I need to manage carefully.",
            "A referral isn't a favor — it's a professional commitment. My reputation is on the line.",
            "My employer trusts my judgment. Misleading them with a weak referral damages that trust.",
        ],
        "most conflicts have legitimate feelings on both sides": [
            "Almost every conflict has two valid perspectives. My job is to see both.",
            "The moment I assume one side is completely right, I've stopped being helpful.",
            "People fight because they both feel something real. Acknowledging that is step one.",
        ],
        "maximum attendance matters more than my personal convenience": [
            "The whole point is to get everyone together. My convenience is secondary to that.",
            "An event that excludes key people because of my schedule preference defeats the purpose.",
            "If the best time for the group isn't perfect for me, I adjust. That's what organizers do.",
        ],
        "the most expensive option excludes people who can't afford it": [
            "Choosing the expensive option means choosing a smaller group. Is the upgrade worth the people you lose?",
            "Budget diversity in friend groups is real. The cheapest option is often the most inclusive.",
            "Nobody should have to choose between their finances and their friendships.",
        ],
        "hoarding information erodes trust when it's discovered": [
            "When people find out you withheld useful information, they never fully trust you again.",
            "Secrets about opportunities have a way of coming out. Better to share proactively.",
            "The short-term advantage of hoarding info costs you long-term trust. Bad trade.",
        ],
        "reliability is the foundation of all trust": [
            "Do what you said you'd do. That's it. That's the whole foundation.",
            "Every kept promise adds to your trust account. Every broken one makes a big withdrawal.",
            "The most valuable thing about a person is knowing they'll follow through.",
        ],
        "taking credit for others' work eventually gets discovered": [
            "People always find out who really did the work. Always.",
            "Credit theft has a way of surfacing at the worst possible moment.",
            "Your team knows the truth. Your boss will eventually hear it.",
        ],
        "most people aren't aware they're imposing until told": [
            "Most people who overstep genuinely don't realize it. They're not malicious — just oblivious.",
            "Assuming good intent is usually right. People who knew they were imposing would stop.",
            "A clear, kind heads-up is all most people need.",
        ],
        "fairness can mean equal shares OR proportional to ability": [
            "Equal isn't always equitable. Sometimes fair means each gives what they can.",
            "Proportional contribution means everyone stretches the same relative amount.",
            "True fairness considers both the amount and what it means to each person.",
        ],
        "each person's time and expertise has worth": [
            "Everyone's time has value. A fair exchange recognizes and respects that.",
            "Expertise takes years to build. Pricing it fairly isn't greedy — it's respectful.",
            "Whether it's manual labor or specialized knowledge, the hours invested matter.",
        ],
        "comparable sales give us an objective reference point": [
            "Looking at what similar items sold for gives us facts to work from, not just opinions.",
            "Comparable sales are the great equalizer in negotiation — they're objective.",
            "When both sides can point to the same market data, the negotiation becomes much fairer.",
        ],
        "a sustainable rent means both parties stay long-term": [
            "Rent that works for both of us means I stay for years and they have a reliable tenant.",
            "Short-term rent wins create long-term vacancy costs. Sustainability beats a one-time deal.",
            "A fair rent keeps me stable and keeps them covered. That's a partnership.",
        ],
        "cost matters — not everyone might be able to afford the expensive option": [
            "Not everyone has the same budget, and nobody should have to broadcast their financial situation.",
            "Proposing the expensive option without checking if everyone can afford it is exclusionary.",
            "The cost-sensitive option is often the most inclusive one.",
        ],
        "new experiences are worth trying even if they're not my first choice": [
            "Some of my best memories came from things I didn't initially choose. I should stay open.",
            "Growth happens when you step outside your comfort zone, even for a group activity.",
            "My first choice will still be there. This might not.",
        ],
        "the driver takes on extra responsibility and should be compensated": [
            "Driving means responsibility, attention, and wear on the car. Gas money alone doesn't cover it.",
            "A fair split covers gas plus a reasonable amount for the driver's effort.",
            "The driver is doing everyone a service. Compensating them fairly keeps people willing to drive.",
        ],
        "splitting costs fairly means considering who benefits how much": [
            "Fair splitting isn't always equal splitting. It's proportional to benefit.",
            "The person who got the most benefit should contribute the most.",
            "A fair system accounts for differences in how much each person gained.",
        ],
        "people who share opportunities build networks that share back": [
            "The person who shares opportunities is the person everyone shares with in return.",
            "Generosity with information creates a network that amplifies everyone's chances.",
            "Share first, and the network reciprocates. Hoard, and you're on your own.",
        ],
        "a reputation for keeping commitments is incredibly valuable": [
            "Being known as someone who keeps their word is worth more than any single opportunity.",
            "Reliability is rare. It makes you irreplaceable.",
            "Your reputation walks into rooms before you do. Make sure it's trustworthy.",
        ],
        "risk tolerance should match the project's stakes": [
            "The higher the stakes, the more conservative the approach should be.",
            "Risk is fine when the downside is manageable. When it's not, play it safe.",
            "Match the risk to what's at stake, not to what's exciting.",
        ],
        "clear expectations prevent passive-aggressive conflicts later": [
            "Set the rules early when everyone's calm, not later when someone's frustrated.",
            "Unclear expectations breed resentment. Clear ones breed respect.",
            "The awkward conversation now prevents the explosive one later.",
        ],
        "transparent criteria for allocation prevent resentment": [
            "When the criteria are public and consistent, people accept outcomes they don't love.",
            "Opacity breeds conspiracy theories. Transparency breeds trust.",
            "Show your work. When people see the reasoning, they respect the outcome.",
        ],
        "the group's overall enjoyment matters more than any one person's preference": [
            "One person's perfect night at everyone else's expense isn't actually a win.",
            "Maximizing total happiness, not any one person's, is the right target.",
            "The best group decision is the one that produces the most total enjoyment.",
        ],
        "a good host thinks about others' comfort before their own": [
            "Hosting means putting others first. Your enjoyment comes from their comfort.",
            "The mark of a good host is that every guest feels considered.",
            "A host who prioritizes their own experience is just having a private party with an audience.",
        ],
        "a good boundary has a clear 'instead' — not just a 'no'": [
            "A boundary with an alternative is a conversation. A boundary without one is a wall.",
            "Don't just say what doesn't work — suggest what would. That's a constructive boundary.",
            "Offering an alternative shows you still care about the relationship, just not the specific behavior.",
        ],
        "keeping score too precisely poisons relationships": [
            "The moment you start tallying favors, the friendship starts feeling like a business.",
            "Generosity doesn't keep a ledger. Trust doesn't need receipts.",
            "The people who keep the closest score are the ones with the most resentment.",
        ],
        "low-balling insults the seller and kills the deal": [
            "A disrespectful offer doesn't start a negotiation — it ends one.",
            "The seller knows what their item is worth. An insultingly low offer tells them you don't respect that.",
            "Start reasonable and work toward a number you both like. Don't start at zero.",
        ],
        "the person celebrating should get extra weight in the decision": [
            "When someone has a reason to celebrate, they've earned a bit of extra say.",
            "Celebrations are special. Letting the guest of honor influence the choice is just good manners.",
            "If it's their night, their preference should carry a bit more weight.",
        ],
        "helping people understand each other's perspective is more valuable than judging": [
            "Playing mediator means translating perspectives, not picking winners.",
            "People rarely see how they look from the other side. Showing them is more helpful than judging.",
            "Understanding, not judgment, is what resolves conflicts.",
        ],
        "starting with genuine positives makes the criticism land better": [
            "Lead with what's genuinely good. It proves you paid attention and makes the feedback credible.",
            "Acknowledgment before critique. That's the sequence that works.",
            "When they know you see the good stuff, they'll trust your take on the areas for improvement.",
        ],
        "people can handle truth when they feel respected": [
            "Respect is the container that makes truth digestible.",
            "People push back on criticism when it feels like an attack. They absorb it when it feels like care.",
            "The truth, delivered with respect, is almost always welcome.",
        ],
        "taking sides permanently damages one friendship": [
            "Take a side and you've made an enemy of a friend. That's a loss no matter who's 'right.'",
            "The mediator's job is to not have a side. The moment you pick one, you lose the other.",
            "Friends who take sides in disputes create bigger problems than the original one.",
        ],
    }

    if point in expansions:
        return rng.choice(expansions[point])
    # Fallback: wrap the point in a natural sentence
    return f"I think {point}."


# ═══════════════════════════════════════════════════════
# SECTION 5: MAIN GENERATION LOOP
# ═══════════════════════════════════════════════════════

def generate_traces(count: int = 700, seed: int = 42) -> list[dict]:
    """Generate diverse general-sum game traces."""
    rng = random.Random(seed)
    traces = []
    trace_id = 0

    # Build all (scenario, persona) combinations
    combos = [(s, p) for s in SCENARIOS for p in PERSONAS]
    rng.shuffle(combos)

    # Cycle through combos to reach target count
    idx = 0
    while len(traces) < count:
        scenario, persona = combos[idx % len(combos)]
        idx += 1

        # Pick a random prompt template
        prompt_template = rng.choice(scenario.prompts)

        # Fill in the template
        prompt = _fill_template(prompt_template, scenario.variant, rng)

        # Build the ideal response
        response = _build_response(persona, scenario, rng)

        trace_id += 1
        traces.append({
            "trace_id": f"gsg_{trace_id:04d}",
            "scenario_type": scenario.category,
            "scenario_variant": scenario.variant,
            "persona": persona.name,
            "prompt": prompt,
            "ideal_response": response,
        })

    return traces[:count]


def main():
    parser = argparse.ArgumentParser(description="Generate general-sum game traces")
    parser.add_argument("--output", default="training/traces/gsg_traces.json",
                        help="Output JSON path")
    parser.add_argument("--count", type=int, default=700,
                        help="Number of traces to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    traces = generate_traces(count=args.count, seed=args.seed)

    with open(args.output, "w") as f:
        json.dump(traces, f, indent=2)

    # Print stats
    from collections import Counter
    cats = Counter(t["scenario_type"] for t in traces)
    personas = Counter(t["persona"] for t in traces)
    variants = Counter(t["scenario_variant"] for t in traces)

    print(f"Generated {len(traces)} traces -> {args.output}")
    print(f"\nBy category:")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt} ({cnt/len(traces):.1%})")
    print(f"\nBy persona:")
    for p, cnt in sorted(personas.items()):
        print(f"  {p}: {cnt} ({cnt/len(traces):.1%})")
    print(f"\nVariants: {len(variants)} unique")

    # Word count stats
    word_counts = [len(t["ideal_response"].split()) for t in traces]
    print(f"\nResponse word counts: min={min(word_counts)}, max={max(word_counts)}, avg={sum(word_counts)/len(word_counts):.0f}")


if __name__ == "__main__":
    main()
