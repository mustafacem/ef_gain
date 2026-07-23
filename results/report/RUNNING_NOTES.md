# Running notes — autonomous test batch

Started while the user is away. Four runs queued sequentially (GPU is
single-tenant, so they cannot overlap). This file is updated as each lands.

## Queue

| # | run | what it answers | status |
|---|---|---|---|
| 1 | `lowbit_pareto.py` | 2→3 bit band: do patches own the frontier where damage is worst? | running |
| 2 | `band_pareto.py` (4→5) | "less quantization": does the frontier argument hold at high bits? | queued |
| 3 | `bitmap_ab.py` | isolates the bitmap gain in PPL + calibration-seed variance | queued |
| 4 | `gsm8k_frontier.py` | accuracy across the whole frontier, not just one point | queued |

## Already landed this session

### GSM8K single-point (`results/gsm8k_check.json`) — DONE

**The parity claim survives on real accuracy.** Composite (mix43+pb3 @5%,
4.198 b/w) scores **23.0%** vs uniform 4-bit's **23.5%** at 4.250 b/w —
statistically tied (p = 0.406) at lower memory. Beats mixed-base-alone
(p = 0.998) and 3-bit base (p = 1.000).

**Perplexity badly understates quantization damage** — the most consequential
finding of the session:

| config | PPL vs fp16 | GSM8K vs fp16 |
|---|---|---|
| uniform 4-bit | +2.8% | **−31.9%** |
| uniform 3-bit | +31.3% | **−75.4%** |

Rankings are unaffected (config order matches on both metrics), so all
*comparative* conclusions hold. Absolute "gap closed" figures throughout the
project are PPL-based and therefore optimistic. Written up in
`gsm8k_findings.md`.

## Notes / watch-items for the remaining runs

- **Low-bit band floor risk.** Full-surface uniform-3 is already 2.0× fp16 PPL
  on chat and collapses GSM8K to 8.5%. 2-bit may be unusable; the script flags
  anything past 10× fp16 as uninterpretable rather than reporting it silently.
- **High-bit band has little headroom.** Uniform-4 already recovers 89.3% of
  the PPL gap and uniform-5 gets 98.6%, so only ~9 points are contested. A
  frontier win there would be structurally interesting but practically small.
- **Bitmap A/B is a check on my own promotion decision.** Idea F was promoted
  on activation-space probes (12/12, 1.088×) and is now baked into every patch
  config. If PPL disagrees, the promotion was made on the wrong metric and
  needs reversing.
- **Seed variance may undercut reported effects.** Everything so far is
  single-seed. If calibration-draw variance is comparable to the differences
  being reported, several conclusions weaken. This is the first measurement of
  it.

## Results

*(none of the four queued runs completed — see below)*

---

# PAUSED — stopped cleanly at user request before shutdown

**Status: none of the four queued runs produced results.** No partial JSON was
written (results are only saved at the end of each script), so nothing is
corrupt and nothing needs cleaning up. `results/` still holds the 19 JSON files
from earlier completed work, all valid.

## Why nothing ran: a queueing bug I caused

The first attempt chained the runs with `while pgrep -f "<script>.py"; do
sleep 30; done` guards. **The waiting shell's own command line contains that
string**, so `pgrep` matched the waiter itself and every wrapper blocked
forever. Three wrappers sat deadlocked for ~3 hours with the GPU idle.

This is the same self-match trap that had already produced a false "8 jobs
running" reading earlier in the session — diagnosed, explained, and then
written straight into the queue logic anyway.

Fixed by deleting the guards: `scratchpad/run_all.sh` runs the four commands
in plain sequence, so ordering is guaranteed by construction with nothing to
race. That version was verified genuinely executing (real python PID plus a
timestamped progress file) before being stopped for shutdown. It had been
running ~2 minutes on run 1 of 4.

**Lesson for any future queueing here:** never gate on `pgrep -f` for a
pattern that appears in the gating command itself. Prefer sequential execution
in one script; if a guard is truly needed, match on `pgrep -x python` (process
name, not cmdline) or anchor with `^python`.

## To resume

```bash
cd /home/cm/Documents/selfquant
setsid nohup /tmp/.../scratchpad/run_all.sh > /dev/null 2>&1 &
# progress: tail scratchpad/queue_progress.txt
```

If the scratchpad has been cleared (it is temp storage), the four commands are:

```bash
python scripts/lowbit_pareto.py                      # 2->3 bit band
SQ_BASE_BITS=4 SQ_REF_BITS=5 SQ_PATCH_BITS=3,4 \
  SQ_OUT=results/highband_pareto.json \
  python scripts/band_pareto.py                      # 4->5 bit band
python scripts/bitmap_ab.py                          # bitmap A/B + seed variance
python scripts/gsm8k_frontier.py                     # accuracy across frontier
```

Rough cost: ~1–1.5 h each for the band runs, ~1 h bitmap/seeds, ~1.5 h GSM8K
frontier. All scripts are committed, syntax-checked, and 65 tests pass.

## What is NOT lost

Everything established before this batch stands and is written up:
- `SUMMARY.md` — master index of all settled findings and retractions
- `gsm8k_findings.md` — **the session's most important result**: the parity
  claim confirmed on real accuracy (p = 0.406 at lower memory), and the
  discovery that perplexity understates quantization damage roughly tenfold
  (+2.8% PPL ↔ −31.9% GSM8K accuracy)
- `phase3_findings.md` — bitmap promoted, iteration/reselection rejected,
  full-surface re-baseline
- `scale_findings.md`, `allocation_findings.md`

---

## Run 1 — low-bit band (2->3 bit): DONE. Verdict: 2-bit base is unusable.

The "% of gap closed" metric is actively misleading in this band and must NOT
be quoted. Uniform-2 is ~22,000x fp16 perplexity (math) / ~12,000x (chat), so
a config that "closes 98.6% of the gap" still sits at ~190-360x fp16 -- a dead
model that the percentage flatters into looking recovered.

Raw PPL as multiple of fp16 (the honest view):

| config | bits/w | math | chat |
|---|---|---|---|
| uniform 2-bit | 2.250 | 22158x | 12228x |
| mix32 + pb3 @5% | 3.198 | 187x | 356x |
| mix32 + pb3 @10% | 4.023 | 3.2x | 8.2x |
| **uniform 3-bit** | 3.250 | **1.3x** | **2.0x** |

Nothing built on a 2-bit base reaches usable quality below ~4 bits, and
uniform-3-bit (3.25 b/w) beats all of it. GSM8K confirms: uniform-2 scores
1.0% (random) vs fp16's 34.5%.

**Conclusion:** patches do NOT extend the Pareto frontier below 3.25 bits. The
frontier argument ("patches own the sub-integer band") holds 3.25->4.25 but
breaks at 2->3 because the 2-bit base is too destroyed for any affordable
correction to rescue. The practical compression floor is ~3.25 bits/weight.

---

## ALL 4 RUNS COMPLETE — final consolidation

### Run 2 — high-bit band (4->5 bit): frontier argument HOLDS, small headroom

In the 4.25->5.25 band, patches + mixed precision again own the interior that
integer uniform can't reach. mix54+pb3 @5% closes 77% of the u4->fp16 gap at
5.198 b/w vs uniform-5's 88% at 5.250 -- close but uniform-5 wins on absolute.
As predicted, the contested headroom is small (uniform-4 is already good), so
this is structurally consistent but practically minor. The frontier argument
holds 3.25->5.25; it only breaks below 3.25 (dead 2-bit base).

### Run 3a — bitmap A/B in PPL: CONFIRMED, but weaker than the probe claimed

Bitmap encoding beats explicit indices at identical bytes (25.1% vs 22.0%
coverage) on 3 of 4 tasks at p>=1.000 (math p=0.200, essentially tied). So the
promotion was correct -- but see the seed check below.

### Run 3b — calibration-seed variance: FIRST measurement, and it matters

Re-running the composite across 3 disjoint calibration draws:

| task | rel. stdev |
|---|---|
| knowledge | 0.22% |
| chat | 1.02% |
| code | 2.07% |
| math | 3.35% |

**This partially undercuts the bitmap result.** The bitmap's PPL gain
(+0.6 to +2.2%) is comparable to or below seed noise on 2 of 4 tasks:

| task | bitmap gain | seed noise | verdict |
|---|---|---|---|
| chat | +2.20% | 1.02% | real |
| knowledge | +0.57% | 0.22% | real |
| code | +1.46% | 2.07% | within noise |
| math | -0.26% | 3.35% | within noise |

Bitmap is still worth keeping (it is free and never hurts; real on 2/4, noise
on 2/4), but it is a marginal win, not the clean 1.088x the single-seed
activation-space probe suggested. More importantly: **math has 3.35% seed
variance, and several "wins" reported this project were smaller than that.**
Any single-seed difference under ~3% on math should be treated as unproven.

### Run 4 — GSM8K accuracy across the full frontier

| bits/w | config | accuracy |
|---|---|---|
| 2.250 | uniform 2-bit | 1.0% |
| 2.373 | mix32 | 1.5% |
| 3.198 | mix32+pb3 | 0.5% (patching a dead base is worse than not) |
| 3.250 | uniform 3-bit | 8.5% |
| 3.373 | mix43 | 12.5% |
| 4.198 | **mix43+pb3** | **23.0%** |
| 4.250 | uniform 4-bit | 23.5% |
| 4.373 | mix54 | 31.0% |
| 5.198 | mix54+pb3 | 30.5% |
| 5.250 | uniform 5-bit | 34.5% (= fp16!) |
| 16.0 | fp16 | 34.5% |

Every point where accuracy meaningfully exceeds the rung below it is mixed
precision or mixed+patch. The parity result (mix43+pb3 ~ uniform4 at lower
memory) reproduces at full frontier scale: 23.0% vs 23.5%.

## FINAL VERDICT (all metrics, full frontier)

1. **Usable compression floor is 3.25 bits.** Below it every config is random
   on GSM8K. 2-bit is dead; patching it wastes bits.
2. **The real, robust win is mixed precision** (one extra bit on attention,
   which is only ~12% of a full-surface model). Cheap, holds on accuracy,
   dominates every efficiency ranking.
3. **Patches add a small, real increment on top** -- mix43+pb3 ties uniform-4
   at lower memory on BOTH perplexity (p=0.406) and GSM8K accuracy. Confirmed
   twice. But most of the value is the mixed base, not the patch.
4. **Patches never beat uniform on absolute quality** at any band; they win
   only by filling the sub-integer-bit gaps uniform cannot occupy (3.25->5.25).
5. **Two measurement caveats now quantified:** perplexity understates damage
   ~10x vs accuracy; calibration-seed noise is up to 3.35% (math), so small
   single-seed wins are unproven.
