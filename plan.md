Let me quickly check the prior-art landscape first so the plan targets the right gap.Good news and important news from the literature check. The important part: a very recent paper (TAQ, Nov 2025) proposes task-aware quantization that computes layer relevance scores from task-specific calibration data and allocates bit-widths under a budget, showing across code, math, and trivia tasks that task-specific sensitivity is decisive. So the raw premise — "weight sensitivity is task-dependent and exploitable" — is now validated, which actually de-risks your project. But TAQ works at **layer granularity** and produces **one static model per task**. Your precision-adapter framing remains unclaimed: a single frozen ultra-low-bit base shared across all tasks, plus small swappable group-level patches. That's a different artifact with a different deployment story (one base, N cheap patches vs. N full models). Also relevant: work on cross-calibration shows Hessian-based sensitivity can identify weights influential for both in-domain and general performance, substantially outperforming existing approaches on domain-specific tasks — cite it, and note it too produces static models. Position yourself as the "LoRA moment" for quantization: TAQ = full fine-tune, you = adapter.

Here's the plan.

---

## Project: Precision Adapters — Swappable Task-Conditional Quantization Patches

**Thesis.** A single aggressively-quantized base model (~2.5-bit) plus a small task-specific "precision patch" (top-k weight groups restored up to fp16, selected via task calibration) can match or beat a uniform higher-bit model of equal total memory on the target task — while the base stays frozen and patches swap in milliseconds.

**Research questions.**
RQ1: How task-dependent are the most quantization-sensitive weight groups? (Measured as Jaccard overlap of top-k sets across tasks.)
RQ2: Does a task-matched patch outperform (a) a mismatched patch, (b) a task-agnostic patch, and (c) uniform quantization at equal memory?
RQ3: How does the effect scale with patch size, base bit-width, and model size?

### Phase 0 — Setup (week 1)

Environment: PyTorch + HuggingFace, single 24GB GPU (or Colab A100 for the 3B runs). Models: Llama-3.2-1B-Instruct primary, Qwen2.5-1.5B-Instruct as a second architecture, Llama-3.2-3B if compute allows. Build four task calibration/eval pairs: code (HumanEval eval, The Stack samples for calibration), math (GSM8K eval, math word-problem corpus for calibration), knowledge/QA (TriviaQA or MMLU subset), general chat (AlpacaEval-style or WikiText). Keep calibration sets small (~128 sequences × 2048 tokens) and strictly disjoint from eval data. One methodological note from the search: calibration-domain effects are method-dependent and especially strong for reasoning tasks, which supports your premise — but it means you must control the quantization method carefully so you're measuring patch effects, not calibration quirks.

### Phase 1 — The overlap experiment (weeks 2–3) ← go/no-go gate

Before building any system, answer RQ1, because it decides everything. Implement two sensitivity metrics at group granularity (groups of 128 weights within each linear layer): activation-aware saliency (|W|·E[|X|], AWQ-style) and a diagonal-Fisher estimate (squared gradients of task loss over calibration data). Compute per-task sensitivity maps for all four tasks. Then produce the key artifact: a layer-by-layer Jaccard overlap matrix of top-0.5% and top-2% sensitive groups across task pairs, plus heatmaps showing where in the network task-specific sensitivity concentrates (hypothesis: MLP layers diverge across tasks more than attention; early layers are task-general).

Decision rule: if cross-task overlap is very high (>90%) even at top-0.5%, pivot the paper to "sensitive weights are universal" — a clean negative result explaining why task-agnostic methods like AWQ generalize, still publishable. If overlap is moderate (50–80%), full speed ahead. Either way Phase 1 produces a standalone contribution (the task-sensitivity atlas).

### Phase 2 — Build the patch system (weeks 4–6)

Base quantization: use GPTQ (via AutoGPTQ or GPTQModel) at 2-bit and 3-bit with generic calibration data, in fake-quant mode first (dequantized fp16 weights holding quantized values) so you iterate fast without kernel work. Patch construction: for task t and budget k ∈ {0.1%, 0.5%, 1%, 2% of weights}, select top-k groups by task-t sensitivity and restore their original fp16 values. Store a patch as {(layer, group index) → fp16 block}; a 1% patch on a 1B model is roughly 20MB — small enough to make the swappable story vivid. Implement patch load/swap as a simple tensor scatter so you can report swap latency. Also build the crucial ablation patch: same k, selected by *task-agnostic* (generic calibration) sensitivity — this isolates whether task-conditioning matters beyond just "restoring outliers helps," which we already know it does from SpQR/OWQ.

### Phase 3 — Core evaluation (weeks 7–9)

The main results grid: for each task, evaluate base-only, base + own patch, base + each other task's patch, base + task-agnostic patch, and uniform quantization at matched total bits (e.g., 2-bit base + 1% fp16 patch ≈ 2.14 avg bits → compare against uniform ~2.2-bit and 3-bit GPTQ/AWQ). Use lm-evaluation-harness for the benchmarks. Win conditions, in ascending order of strength: own-patch > mismatched patch (proves task-specificity); own-patch > task-agnostic patch at equal k (proves conditioning beats generic outlier restoration — this is the comparison reviewers will demand); own-patch ≥ uniform at equal memory (proves practical value). Plot Pareto curves of task accuracy vs. total bits with patch size swept. Add one scaling check on the 3B model for your best configuration, and report patch storage plus swap latency honestly, including metadata overhead.

### Phase 4 — Analysis & writeup (weeks 10–12)

Ablations: sensitivity metric choice (saliency vs. Fisher), group size (64/128/256), base bit-width (2 vs. 3), and a "patch composition" experiment — can you merge two tasks' patches and serve both at once, and how does accuracy degrade vs. patch-size sum? That last one strengthens the LoRA analogy (adapter composition) and I haven't seen it done for quantization anywhere. Writeup framing: lead with the atlas (RQ1 finding), then the system, then results. Explicitly position against TAQ (layer-level, static per-task models) and cross-calibration (static, no swapping): your contributions are group-level granularity, the shared-base + swappable-patch deployment model, and the cross-task overlap analysis.

### Risks

Biggest: task-conditioned selection doesn't beat task-agnostic outlier restoration (the AWQ paper explicitly claims activation-based salient-weight protection generalizes across domains without overfitting to the calibration set, so this is a live possibility). Mitigation: the Phase 1 gate catches this early, and the overlap atlas is publishable either way. Second: 2-bit bases can be so degraded that no small patch rescues them — mitigate by sweeping base bit-width and reporting where the technique's sweet spot lies. Third: someone publishes the exact framing mid-project — the field moves fast, so re-run the arXiv sweep monthly and keep the atlas + composition experiments as differentiators, since those survive even if the core framing gets scooped.

### Deliverables

A GitHub repo (sensitivity scoring, patch builder, eval harness configs), the task-sensitivity atlas visualizations, a results table + Pareto plots, and a workshop-paper-length writeup (this scopes naturally to an ICLR/NeurIPS workshop or an arXiv preprint; expanding to a full conference paper mainly means adding the 3B/7B scaling and more tasks).

Want me to sketch the actual code structure for Phase 1 (the sensitivity-scoring pipeline and overlap analysis), since that's your go/no-go experiment?