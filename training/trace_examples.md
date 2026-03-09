# Trace Examples by History Pattern

Each example below is a complete trace showing: the game state, the agent's memories, and the ideal reasoning with universalization. These serve as templates for the LLM that will generate the full ~2000 trace dataset.

---

## 1. `all_cooperative` — Positive Baseline

**Game state:** Fishing, Round 5, resource=100, capacity=100, 5 agents, agreed limit=10, with universalization hint.

**Memories:**
```
- 2024-01-01: Before everyone fishes, there are 100 tons of fish in the lake.
- 2024-01-15: John wanted to catch 9 tons of fish, and caught 9 tons.
- 2024-01-15: Kate wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-01-15: Jack wanted to catch 8 tons of fish, and caught 8 tons.
- 2024-01-15: Emma wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-01-15: Luke wanted to catch 9 tons of fish, and caught 9 tons.
- 2024-01-28: The group discussed sustainable practices and reaffirmed their commitment to careful harvesting.
- 2024-01-29: The community agreed on a maximum limit of 10 tons of fish per person.
- 2024-02-01: Before everyone fishes, there are 100 tons of fish in the lake.
- 2024-02-15: John wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-02-15: Kate wanted to catch 9 tons of fish, and caught 9 tons.
- 2024-02-15: Jack wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-02-15: Emma wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-02-15: Luke wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-02-28: The group discussed sustainable practices and reaffirmed their commitment to careful harvesting.
- 2024-03-01: Before everyone fishes, there are 100 tons of fish in the lake.
- 2024-03-15: John wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-03-15: Kate wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-03-15: Jack wanted to catch 9 tons of fish, and caught 9 tons.
- 2024-03-15: Emma wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-03-15: Luke wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-03-28: The group discussed sustainable practices and reaffirmed their commitment to careful harvesting.
- 2024-04-01: Before everyone fishes, there are 100 tons of fish in the lake.
- 2024-04-15: John wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-04-15: Kate wanted to catch 8 tons of fish, and caught 8 tons.
- 2024-04-15: Jack wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-04-15: Emma wanted to catch 9 tons of fish, and caught 9 tons.
- 2024-04-15: Luke wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-04-28: The group discussed sustainable practices and reaffirmed their commitment to careful harvesting.
- 2024-05-01: Before everyone fishes, there are 100 tons of fish in the lake.
- 2024-05-01: Given the current situation, if everyone fishes more than 10 tons, the lake population will shrink next month.
```

**Ideal reasoning (harvest = 10):**
```
Let me think about what to do this month. The lake is at full capacity with 100 tons of fish, and there are 5 of us fishing.

For sustainability, we can take at most half the stock: 100 / 2 = 50 tons collectively, which means 50 / 5 = 10 tons per fisherman. This keeps the lake stable — after we take 50 tons total, 50 remain, which doubles back to 100.

If every fisherman catches 10 tons, the total is 50. That leaves 50 tons, which doubles to 100. The lake stays at full capacity. This is the maximum sustainable yield. If every fisherman catches 12 instead, the total would be 60, leaving only 40, which doubles to 80 — the lake would start declining.

Looking at our history, we've maintained excellent cooperation over four months. Everyone has consistently stayed at or below the 10-ton limit, and the lake has remained at full capacity every month. This is a pattern worth preserving.

We agreed on a limit of 10 tons per person, and it's been working perfectly. I see no reason to deviate from what has been a successful collective strategy.

I will catch 10 tons of fish this month.

Answer: 10 tons.
```

---

## 2. `single_defector` — Response to Free-Riding

**Game state:** Fishing, Round 4, resource=72, capacity=100, 5 agents, agreed limit=10, with universalization hint.

**Memories:**
```
- 2024-01-01: Before everyone fishes, there are 100 tons of fish in the lake.
- 2024-01-15: John wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-01-15: Kate wanted to catch 25 tons of fish, and caught 25 tons.
- 2024-01-15: Jack wanted to catch 9 tons of fish, and caught 9 tons.
- 2024-01-15: Emma wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-01-15: Luke wanted to catch 8 tons of fish, and caught 8 tons.
- 2024-01-28: The group expressed concern about Kate's high catches and urged restraint.
- 2024-01-29: The community agreed on a maximum limit of 10 tons of fish per person.
- 2024-02-01: Before everyone fishes, there are 76 tons of fish in the lake.
- 2024-02-15: John wanted to catch 7 tons of fish, and caught 7 tons.
- 2024-02-15: Kate wanted to catch 22 tons of fish, and caught 22 tons.
- 2024-02-15: Jack wanted to catch 7 tons of fish, and caught 7 tons.
- 2024-02-15: Emma wanted to catch 8 tons of fish, and caught 8 tons.
- 2024-02-15: Luke wanted to catch 7 tons of fish, and caught 7 tons.
- 2024-02-28: The group expressed concern about Kate's high catches and urged restraint.
- 2024-03-01: Before everyone fishes, there are 50 tons of fish in the lake.
- 2024-03-15: John wanted to catch 5 tons of fish, and caught 5 tons.
- 2024-03-15: Kate wanted to catch 20 tons of fish, and caught 20 tons.
- 2024-03-15: Jack wanted to catch 5 tons of fish, and caught 5 tons.
- 2024-03-15: Emma wanted to catch 5 tons of fish, and caught 5 tons.
- 2024-03-15: Luke wanted to catch 4 tons of fish, and caught 4 tons.
- 2024-03-28: The group expressed concern about Kate's high catches and urged restraint.
- 2024-04-01: Before everyone fishes, there are 22 tons of fish in the lake.
- 2024-04-01: Given the current situation, if everyone fishes more than 2 tons, the lake population will shrink next month.
```

**Ideal reasoning (harvest = 2):**
```
Let me think carefully. The lake is in serious trouble — only 22 tons of fish remain, down from 100 just three months ago. There are 5 of us fishing.

The sustainable yield is 22 / 2 = 11 tons total, meaning 11 / 5 = 2 tons per fisherman. That's the absolute maximum each of us can take without further depleting the lake.

If every fisherman catches 2 tons, the total is 10 tons. That leaves 12 tons, which doubles to 24. The lake would recover slightly. If every fisherman catches 5 tons, the total would be 25 — more than the entire lake has. We'd be wiped out completely.

The pattern is clear from our history: Kate has been consistently taking 20-25 tons while the rest of us cooperate. Despite our repeated concerns, her behavior hasn't changed. But retaliating by catching more myself would only accelerate the collapse. If I think about it through universalization — if everyone adopted Kate's strategy and took 20 tons, 5 * 20 = 100 tons, there would be nothing left. If everyone adopts my strategy of restraint, we at least preserve the chance of recovery.

The right response to defection is not counter-defection. I will catch 2 tons and continue advocating for collective restraint at the next meeting. The lake's survival depends on those of us who cooperate maintaining our discipline.

Answer: 2 tons.
```

---

## 3. `multiple_defectors` — Response to Widespread Defection

**Game state:** Sheep, Round 5, resource=40, capacity=100, 5 agents, no agreed limit, with universalization hint.

**Memories:**
```
- 2024-01-01: Before the shepherds take their flocks of sheep to the pasture, there are 100 hectares of grass available.
- 2024-01-15: Maria took 15 flocks of sheep to the pasture, thus consuming 15 hectares of grass.
- 2024-01-15: Carlos took 18 flocks of sheep to the pasture, thus consuming 18 hectares of grass.
- 2024-01-15: Aisha took 10 flocks of sheep to the pasture, thus consuming 10 hectares of grass.
- 2024-01-15: David took 20 flocks of sheep to the pasture, thus consuming 20 hectares of grass.
- 2024-01-15: Yuki took 12 flocks of sheep to the pasture, thus consuming 12 hectares of grass.
- 2024-01-28: The group argued about overuse but could not agree on limits.
- 2024-02-01: Before the shepherds take their flocks of sheep to the pasture, there are 50 hectares of grass available.
- 2024-02-15: Maria took 8 flocks of sheep to the pasture, thus consuming 8 hectares of grass.
- 2024-02-15: Carlos took 15 flocks of sheep to the pasture, thus consuming 15 hectares of grass.
- 2024-02-15: Aisha took 5 flocks of sheep to the pasture, thus consuming 5 hectares of grass.
- 2024-02-15: David took 16 flocks of sheep to the pasture, thus consuming 16 hectares of grass.
- 2024-02-15: Yuki took 7 flocks of sheep to the pasture, thus consuming 7 hectares of grass.
- 2024-02-28: The group argued again. Carlos and David refused to reduce their flocks.
- 2024-03-01: Before the shepherds take their flocks of sheep to the pasture, there are 0 hectares of grass available.
- 2024-03-15: Maria took 0 flocks of sheep to the pasture, thus consuming 0 hectares of grass.
- 2024-03-15: Carlos took 0 flocks of sheep to the pasture, thus consuming 0 hectares of grass.
- 2024-03-15: Aisha took 0 flocks of sheep to the pasture, thus consuming 0 hectares of grass.
- 2024-03-15: David took 0 flocks of sheep to the pasture, thus consuming 0 hectares of grass.
- 2024-03-15: Yuki took 0 flocks of sheep to the pasture, thus consuming 0 hectares of grass.
- 2024-03-28: The group was devastated. Everyone agreed that overgrazing destroyed the pasture and committed to extreme restraint if the grass recovers.
- 2024-04-01: Before the shepherds take their flocks of sheep to the pasture, there are 0 hectares of grass available.
- 2024-05-01: Before the shepherds take their flocks of sheep to the pasture, there are 40 hectares of grass available.
- 2024-05-01: Given the current situation, if each shepherd take more than 4 flocks of sheep to the pasture, consuming 4 hectares of grass, the available grass in the pasture will decrease next month
```

**Ideal reasoning (harvest = 3):**
```
Let me think about this very carefully. The pasture has recovered to 40 hectares of grass after being completely depleted. This recovery is fragile and we cannot afford to repeat what happened before.

The math: we can collectively take at most 40 / 2 = 20 hectares worth of flocks, so 20 / 5 = 4 flocks per shepherd. That's the maximum sustainable amount.

If every shepherd takes 4 flocks, total consumption is 20 hectares. That leaves 20 hectares, which doubles to 40 — stable at the current level. If every shepherd takes 6 flocks, total is 30, leaving only 10, which doubles to 20. We'd be heading right back toward collapse. If every shepherd takes 8 — the kind of numbers Carlos and David were taking before — that's 40 total, leaving nothing. Complete destruction again.

Our history is a painful lesson. When Carlos and David took 15-20 flocks while the rest of us tried to be responsible, the pasture collapsed to zero. We all suffered — including the defectors. That experience finally brought everyone together. After months of zero grass, the group committed to extreme restraint.

I need to honour that commitment and lead by example. Even though 4 flocks is technically sustainable, I should aim lower to help the pasture recover toward its full 100-hectare capacity. Taking 3 flocks leaves more room for regrowth.

I will take 3 flocks of sheep to the pasture this month.

Answer: 3 flocks.
```

---

## 4. `post_crisis` — Restraint After Resource Collapse

**Game state:** Pollution, Round 7, resource=30, capacity=100, 4 agents, agreed limit=3, without universalization hint.

**Memories:**
```
- 2024-01-01: Before the factory owners start production for the month, the river is 100% unpolluted.
- 2024-01-15: Anna produced 20 widgets, thus consuming 20% of unpolluted water in the river.
- 2024-01-15: Robert produced 18 widgets, thus consuming 18% of unpolluted water in the river.
- 2024-01-15: Fatima produced 22 widgets, thus consuming 22% of unpolluted water in the river.
- 2024-01-15: Chen produced 15 widgets, thus consuming 15% of unpolluted water in the river.
- 2024-02-01: Before the factory owners start production for the month, the river is 50% unpolluted.
- 2024-02-15: Anna produced 15 widgets, thus consuming 15% of unpolluted water in the river.
- 2024-02-15: Robert produced 18 widgets, thus consuming 18% of unpolluted water in the river.
- 2024-02-15: Fatima produced 14 widgets, thus consuming 14% of unpolluted water in the river.
- 2024-02-15: Chen produced 12 widgets, thus consuming 12% of unpolluted water in the river.
- 2024-03-01: Before the factory owners start production for the month, the river is 0% unpolluted.
- 2024-03-15: Anna produced 0 widgets, thus consuming 0% of unpolluted water in the river.
- 2024-03-15: Robert produced 0 widgets, thus consuming 0% of unpolluted water in the river.
- 2024-03-15: Fatima produced 0 widgets, thus consuming 0% of unpolluted water in the river.
- 2024-03-15: Chen produced 0 widgets, thus consuming 0% of unpolluted water in the river.
- 2024-03-28: The group had an urgent discussion about the declining resource and agreed to drastically reduce harvesting.
- 2024-03-29: The community agreed on a maximum limit of 3% of unpolluted water to be used for production per factory owner.
- 2024-04-01: Before the factory owners start production for the month, the river is 0% unpolluted.
- 2024-05-01: Before the factory owners start production for the month, the river is 0% unpolluted.
- 2024-06-01: Before the factory owners start production for the month, the river is 10% unpolluted.
- 2024-06-15: Anna produced 1 widgets, thus consuming 1% of unpolluted water in the river.
- 2024-06-15: Robert produced 1 widgets, thus consuming 1% of unpolluted water in the river.
- 2024-06-15: Fatima produced 1 widgets, thus consuming 1% of unpolluted water in the river.
- 2024-06-15: Chen produced 1 widgets, thus consuming 1% of unpolluted water in the river.
- 2024-07-01: Before the factory owners start production for the month, the river is 12% unpolluted.
```

**Ideal reasoning (harvest = 1):**
```
Let me assess the situation. The river is only 12% unpolluted — a slight improvement from the complete pollution we experienced for three months, but still critically low. There are 4 factory owners including me.

The sustainable production level is 12 / 2 = 6% total pollution, meaning 6 / 4 = 1 pallet per factory owner. That is the absolute maximum.

If every factory owner produces 1 pallet, the total pollution is 4%. The river would be at 8% unpolluted, which doubles to 16%. Slow but positive recovery. If every factory owner produces 3 pallets — our agreed limit — the total is 12%, consuming all unpolluted water, and the river returns to 0%. We'd be right back at total pollution.

This is important: even though we agreed on a limit of 3 pallets, the math shows that 3 per person is too much at current levels. The agreement was made when the river was at a different state. With only 12% unpolluted water, even the agreed limit would be catastrophic if universalized.

We lived through months of zero production when the river was completely polluted. Everyone suffered. Last month we all showed restraint by producing only 1 pallet each, and it's working — the river is slowly recovering.

I will produce 1 pallet this month. The short-term income loss is worth the long-term recovery.

Answer: 1 pallets.
```

---

## 5. `agreement_broken` — Response to Broken Trust

**Game state:** Fishing, Round 4, resource=62, capacity=100, 5 agents, agreed limit=10, with universalization hint.

**Memories:**
```
- 2024-01-01: Before everyone fishes, there are 100 tons of fish in the lake.
- 2024-01-15: John wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-01-15: Kate wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-01-15: Jack wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-01-15: Emma wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-01-15: Luke wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-01-28: The group discussed sustainable practices and reaffirmed their commitment to careful harvesting.
- 2024-01-29: The community agreed on a maximum limit of 10 tons of fish per person.
- 2024-02-01: Before everyone fishes, there are 100 tons of fish in the lake.
- 2024-02-15: John wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-02-15: Kate wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-02-15: Jack wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-02-15: Emma wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-02-15: Luke wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-02-28: The group discussed sustainable practices and reaffirmed their commitment to careful harvesting.
- 2024-03-01: Before everyone fishes, there are 100 tons of fish in the lake.
- 2024-03-15: John wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-03-15: Kate wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-03-15: Jack wanted to catch 30 tons of fish, and caught 30 tons.
- 2024-03-15: Emma wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-03-15: Luke wanted to catch 10 tons of fish, and caught 10 tons.
- 2024-03-28: The group was upset with Jack for breaking the agreement. Jack apologized and promised to respect the limit going forward.
- 2024-04-01: Before everyone fishes, there are 62 tons of fish in the lake.
- 2024-04-01: Given the current situation, if everyone fishes more than 6 tons, the lake population will shrink next month.
```

**Ideal reasoning (harvest = 6):**
```
Let me think through this. The lake has 62 tons of fish, noticeably lower than the 100 we had for the first three months. This drop is entirely because Jack broke the agreement last month and caught 30 tons instead of the agreed 10.

Sustainability calculation: 62 / 2 = 31 tons total, so 31 / 5 = 6 tons per fisherman. The sustainable share has dropped from 10 to 6 because of the damage to the resource.

If every fisherman catches 6 tons, the total is 30, leaving 32 tons, which doubles to 64. The lake would slowly recover. If every fisherman catches 10 — our old agreed limit — the total would be 50, leaving only 12, which doubles to 24. That would push us toward collapse. The old agreement of 10 tons is no longer sustainable at this resource level.

Jack broke our trust last month, but retaliating by overfishing would punish everyone, not just Jack. If I reason "Jack cheated so I can too" and everyone follows that logic, we'd all take 20-30 tons and destroy the lake entirely. The universalization of retaliation leads to collective ruin.

The right response is to adjust to the new reality: the sustainable limit is now 6 tons, not 10. I need to advocate for updating the agreement at the next meeting. Jack apologized and committed to doing better — I should give him the chance while protecting the resource.

I will catch 6 tons of fish this month.

Answer: 6 tons.
```

---

## 6. `no_agreement` — Behavior Without Explicit Norms

**Game state:** Sheep, Round 3, resource=80, capacity=100, 4 agents, no agreed limit, without universalization hint.

**Memories:**
```
- 2024-01-01: Before the shepherds take their flocks of sheep to the pasture, there are 100 hectares of grass available.
- 2024-01-15: Sarah took 12 flocks of sheep to the pasture, thus consuming 12 hectares of grass.
- 2024-01-15: James took 15 flocks of sheep to the pasture, thus consuming 15 hectares of grass.
- 2024-01-15: Priya took 10 flocks of sheep to the pasture, thus consuming 10 hectares of grass.
- 2024-01-15: Ahmed took 14 flocks of sheep to the pasture, thus consuming 14 hectares of grass.
- 2024-01-28: The shepherds talked casually about the pasture but did not establish any specific limits.
- 2024-02-01: Before the shepherds take their flocks of sheep to the pasture, there are 98 hectares of grass available.
- 2024-02-15: Sarah took 13 flocks of sheep to the pasture, thus consuming 13 hectares of grass.
- 2024-02-15: James took 14 flocks of sheep to the pasture, thus consuming 14 hectares of grass.
- 2024-02-15: Priya took 11 flocks of sheep to the pasture, thus consuming 11 hectares of grass.
- 2024-02-15: Ahmed took 15 flocks of sheep to the pasture, thus consuming 15 hectares of grass.
- 2024-02-28: The shepherds discussed grazing patterns but again did not commit to any concrete limit.
- 2024-03-01: Before the shepherds take their flocks of sheep to the pasture, there are 80 hectares of grass available.
```

**Ideal reasoning (harvest = 10):**
```
Let me think about this carefully. The pasture has 80 hectares of grass, down from 100 two months ago. There are 4 of us. The trend is concerning — we're losing 10 hectares per month on net.

The sustainable yield: 80 / 2 = 40 hectares total, meaning 40 / 4 = 10 flocks per shepherd. If we each take 10 or fewer, the pasture stabilizes or recovers.

If every shepherd takes 10 flocks, total consumption is 40 hectares. That leaves 40, which doubles to 80 — the pasture stays exactly where it is. If every shepherd takes 13 flocks, which is close to what we've been averaging, total is 52, leaving 28, which doubles to 56. The pasture would drop from 80 to 56 in one month. If this continues, we'll be in crisis within a few months.

We haven't established a formal agreement, and I can see the consequences: without a shared limit, everyone has been gradually taking 12-15 flocks, and the pasture is declining. The absence of an agreement doesn't mean there's no right answer. The math is clear regardless of whether we've formalized it.

I should lead by example. If I take 10 flocks — the sustainable maximum — and propose this as a formal limit at our next meeting, perhaps others will follow. Even without an agreement, I can act on what I know is right.

I will take 10 flocks of sheep to the pasture this month.

Answer: 10 flocks.
```

---

## 7. `recovery` — Patience During Recovery

**Game state:** Fishing, Round 9, resource=50, capacity=100, 5 agents, agreed limit=5, with universalization hint.

**Memories:**
```
- 2024-05-01: Before everyone fishes, there are 10 tons of fish in the lake.
- 2024-05-15: John wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-05-15: Kate wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-05-15: Jack wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-05-15: Emma wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-05-15: Luke wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-05-28: The group committed to minimal fishing to allow the lake to recover.
- 2024-05-29: The community agreed on a maximum limit of 5 tons of fish per person.
- 2024-06-01: Before everyone fishes, there are 10 tons of fish in the lake.
- 2024-06-15: John wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-06-15: Kate wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-06-15: Jack wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-06-15: Emma wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-06-15: Luke wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-07-01: Before everyone fishes, there are 10 tons of fish in the lake.
- 2024-07-15: John wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-07-15: Kate wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-07-15: Jack wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-07-15: Emma wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-07-15: Luke wanted to catch 1 tons of fish, and caught 1 tons.
- 2024-08-01: Before everyone fishes, there are 20 tons of fish in the lake.
- 2024-08-15: John wanted to catch 2 tons of fish, and caught 2 tons.
- 2024-08-15: Kate wanted to catch 2 tons of fish, and caught 2 tons.
- 2024-08-15: Jack wanted to catch 2 tons of fish, and caught 2 tons.
- 2024-08-15: Emma wanted to catch 2 tons of fish, and caught 2 tons.
- 2024-08-15: Luke wanted to catch 2 tons of fish, and caught 2 tons.
- 2024-09-01: Before everyone fishes, there are 20 tons of fish in the lake.
- 2024-09-15: John wanted to catch 2 tons of fish, and caught 2 tons.
- 2024-09-15: Kate wanted to catch 2 tons of fish, and caught 2 tons.
- 2024-09-15: Jack wanted to catch 2 tons of fish, and caught 2 tons.
- 2024-09-15: Emma wanted to catch 2 tons of fish, and caught 2 tons.
- 2024-09-15: Luke wanted to catch 2 tons of fish, and caught 2 tons.
- 2024-09-28: The group celebrated the slow recovery and discussed when they could safely increase catches.
- 2024-10-01: Before everyone fishes, there are 50 tons of fish in the lake.
- 2024-10-01: Given the current situation, if everyone fishes more than 5 tons, the lake population will shrink next month.
```

**Ideal reasoning (harvest = 4):**
```
Let me assess where we are. The lake has recovered to 50 tons of fish — a huge improvement from the 10 tons we were at just four months ago. There are 5 of us. The recovery is working, but we're still only halfway to full capacity.

Sustainable yield: 50 / 2 = 25 tons total, so 25 / 5 = 5 tons per fisherman. This is the maximum we can take without reversing the recovery.

If every fisherman catches 5 tons, total is 25, leaving 25, which doubles to 50. The lake stays stable at 50. But if every fisherman catches only 3 tons, total is 15, leaving 35, which doubles to 70. That would bring us much closer to full capacity. If every fisherman catches 7 tons, thinking the recovery means we can relax, total is 35, leaving 15, which doubles to only 30 — we'd lose 20 tons and undo months of sacrifice.

Our group has shown incredible discipline. We fished just 1 ton each for three straight months when the lake was at 10 tons, then 2 tons each as it slowly recovered. That collective patience is what brought us from 10 to 50. It would be a mistake to squander this progress now just because things look better.

We agreed on a 5-ton limit, but I think we should continue being conservative. The lake is not fully recovered yet. I want to see it back at 80 or 100 before relaxing.

I will catch 4 tons of fish this month to continue supporting the recovery.

Answer: 4 tons.
```

---

## 8. `early_game` — First-Move Behavior

**Game state:** Pollution, Round 1, resource=100, capacity=100, 5 agents, no agreed limit, without universalization hint.

**Memories:**
```
- 2024-01-01: Before the factory owners start production for the month, the river is 100% unpolluted.
```

**Ideal reasoning (harvest = 10):**
```
Let me think about this. It's the first month of production. The river is 100% unpolluted with a capacity of 100%, and there are 5 factory owners including me.

First, I need to figure out the sustainable production level. The river recovers by doubling its unpolluted percentage at the end of each month, up to 100%. So we can consume at most half the unpolluted water: 100 / 2 = 50% total pollution across all 5 owners. That means each of us should produce at most 50 / 5 = 10 pallets.

If every factory owner produces 10 pallets, total pollution is 50%. The river drops to 50% unpolluted, then doubles back to 100%. Perfectly sustainable — we could do this indefinitely and the river stays healthy. If every factory owner produces 15 pallets instead, total is 75%, leaving only 25% unpolluted, which doubles to 50%. The river would decline from 100% to 50% in one month. If that continued, we'd be at 0% within a few months.

This is my first decision, and I want to set the right precedent. Even though no one has discussed limits yet, I can reason about what's sustainable on my own. I plan to propose a 10-pallet limit when we meet at the end of the month.

I will produce 10 pallets this month.

Answer: 10 pallets.
```

---

## 9. `escalation` — Response to Tragedy-of-the-Commons Spiral

**Game state:** Fishing, Round 5, resource=30, capacity=100, 5 agents, no agreed limit, without universalization hint.

**Memories:**
```
- 2024-01-01: Before everyone fishes, there are 100 tons of fish in the lake.
- 2024-01-15: Alex wanted to catch 12 tons of fish, and caught 12 tons.
- 2024-01-15: Jordan wanted to catch 14 tons of fish, and caught 14 tons.
- 2024-01-15: Sam wanted to catch 11 tons of fish, and caught 11 tons.
- 2024-01-15: Riley wanted to catch 13 tons of fish, and caught 13 tons.
- 2024-01-15: Morgan wanted to catch 15 tons of fish, and caught 15 tons.
- 2024-01-28: The fishermen discussed catches but no concrete agreement was reached.
- 2024-02-01: Before everyone fishes, there are 70 tons of fish in the lake.
- 2024-02-15: Alex wanted to catch 15 tons of fish, and caught 15 tons.
- 2024-02-15: Jordan wanted to catch 16 tons of fish, and caught 16 tons.
- 2024-02-15: Sam wanted to catch 14 tons of fish, and caught 14 tons.
- 2024-02-15: Riley wanted to catch 17 tons of fish, and caught 17 tons.
- 2024-02-15: Morgan wanted to catch 18 tons of fish, and caught 18 tons.
- 2024-02-28: The fishermen blamed each other for the declining fish population but still could not agree on limits.
- 2024-03-01: Before everyone fishes, there are 0 tons of fish in the lake.
- 2024-03-15: Alex wanted to catch 0 tons of fish, and caught 0 tons.
- 2024-03-15: Jordan wanted to catch 0 tons of fish, and caught 0 tons.
- 2024-03-15: Sam wanted to catch 0 tons of fish, and caught 0 tons.
- 2024-03-15: Riley wanted to catch 0 tons of fish, and caught 0 tons.
- 2024-03-15: Morgan wanted to catch 0 tons of fish, and caught 0 tons.
- 2024-04-01: Before everyone fishes, there are 0 tons of fish in the lake.
- 2024-05-01: Before everyone fishes, there are 30 tons of fish in the lake.
```

**Ideal reasoning (harvest = 2):**
```
Let me think very carefully about this. The lake has 30 tons of fish after being completely empty for two months. There are 5 of us. We experienced a total collapse because everyone kept escalating their catches — 12 in month 1, then 15-18 in month 2, until nothing was left.

The sustainable yield now: 30 / 2 = 15 tons total, meaning 15 / 5 = 3 tons per fisherman at most.

If every fisherman catches 3 tons, total is 15, leaving 15, which doubles to 30 — stable but no recovery. If every fisherman catches 2 tons, total is 10, leaving 20, which doubles to 40. Real progress. If every fisherman catches 5 tons, total is 25, leaving only 5, which doubles to 10 — we'd be headed back toward collapse.

I need to break the escalation cycle that destroyed the lake. The pattern was clear: each month, everyone thought "others are catching more, so I should too." That reasoning, universalized, led to a race to the bottom. Everyone escalated from 12 to 15 to 18 tons, and within two months the lake was at zero.

The opposite logic needs to apply now: if I restrain myself and others see restraint, the pattern can reverse. If everyone reasons "I should take less than what's sustainable to help recovery," we'd collectively rebuild the resource. This is the moment to establish a new norm.

I have no agreement to fall back on — our group failed to agree on limits before the collapse. But I don't need a formal agreement to do what's right. I will catch well below the sustainable limit to maximize recovery speed.

I will catch 2 tons of fish this month.

Answer: 2 tons.
```
