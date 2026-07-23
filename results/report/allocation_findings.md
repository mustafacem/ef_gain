# Allocation showdown: do patches earn their complexity?

Two competing explanations for why 3-bit patches close only ~47% of the
quantization gap while uniform 4-bit closes ~85%:

- **(a) allocation** — the budget is spread over the wrong layers; push it
  toward `o_proj` (where activation-space said patches reach 0.97–1.12× of
  uniform) and patches become competitive.
- **(b) nothing new** — patches are an expensive way to rediscover "attention
  needs more bits than MLP", which plain per-layer-type **mixed precision**
  achieves with no Hessian solve, no mask, no indices, no swappable artifact.

Both were tested at matched bits/weight. **Both are wrong**, in opposite
directions, and the truth is more interesting than either.

## Results (0.5B, 4 tasks, % of the 3-bit→fp16 gap recovered)

| config | bits/w | extra | code | math | know | chat | **mean** | %/extra bit |
|---|---|---|---|---|---|---|---|---|
| mixed o4+d3 | 3.406 | 0.156 | 19.3 | 29.0 | 13.6 | 18.6 | 20.1 | **129.2** |
| mixed o6+d3 | 3.717 | 0.467 | 22.5 | 31.5 | 16.4 | 23.1 | 23.4 | 50.0 |
| mixed o8+d3 | 4.028 | 0.778 | 22.7 | 34.6 | 17.7 | 22.8 | 24.4 | 31.4 |
| **patch pb3, proportional** | 4.075 | 0.825 | 23.8 | **76.2** | 47.4 | 40.4 | **46.9** | 56.9 |
| patch pb3, 50% to attn | 4.075 | 0.825 | 31.5 | 67.2 | 49.9 | 38.7 | 46.8 | 56.8 |
| patch pb3, 100% to attn | 3.833 | 0.583 | 19.2 | 34.5 | 12.7 | 14.9 | 20.3 | 34.9 |
| **uniform 4-bit** | 4.250 | 1.000 | 79.4 | 93.3 | 87.1 | 78.1 | **84.5** | 84.5 |

## (b) is refuted: patches are not just mixed precision

At **matched cost** (4.028 vs 4.075 bits/weight), patches close **46.9%** of
the gap against mixed precision's **24.4%** — nearly **2×**, at
**p = 1.000 on all four tasks**. Whatever the OBS scores are selecting, it is
substantially finer-grained than "attention tolerates less error than MLP".
The sensitivity signal contains real within-layer structure.

## (a) is also refuted: allocation was not the bug

Forcing budget toward attention does not help. Proportional (global top-k)
gives 46.9%; 50%-to-attention gives 46.8% — indistinguishable. Pushing
100% to attention **collapses to 20.3%**, and cannot even spend the full
budget (3.833 bits/w) because `o_proj` is only 15.6% of patchable weights.

The global top-k was already allocating correctly. My hypothesis that
`down_proj`'s larger activations were letting it capture undeserved budget
was wrong — that budget is earning its keep.

## But uniform 4-bit still wins, and the margin is not close

84.5% vs 46.9% for **0.175 more bits/weight**. `patch_pb3` loses to uniform at
p = 0.000 on code, knowledge and chat. The single exception remains **math**
(p = 0.138 — statistically indistinguishable) at *lower* memory.

Per marginal bit spent, the ranking is:

1. **mixed o4+d3 — 129.2 %/extra bit.** The most byte-efficient intervention
   tested, by a wide margin: one extra bit on `o_proj` alone costs 0.156
   bits/weight and recovers 20% of the gap. Cheap and nearly free to implement.
2. uniform 4-bit — 84.5
3. patch pb3 — 56.9
4. mixed o6+d3 — 50.0, mixed o8+d3 — 31.4 (mixed precision saturates fast)

## Honest reading

The patch mechanism is **real and non-trivial** — it beats both null
hypotheses at matched cost, decisively and significantly. It is also, at this
scale and budget, **not the best use of a bit**. Uniform precision recovers
nearly twice as much for ~4% more memory, and if the goal is cheap marginal
improvement rather than maximum recovery, simply giving `o_proj` one more bit
dominates everything on efficiency.

Where patches do stand out is **task-dependence**: on math they recover 76.2%
against mixed precision's 34.6% and approach uniform (93.3%) closely enough to
be statistically tied at lower memory. That is consistent with the earlier
finding that task-conditioning is real but narrow — it pays where a task's
sensitivity profile genuinely departs from the generic one, and math is the
clearest such case here.

## Reproduce

```
python scripts/allocation_showdown.py     # results/allocation_showdown.json
```
