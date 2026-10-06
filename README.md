# BalancedMultiDimensionMemory

A **LangGraph `AgentMiddleware`** that gives LLM agents long-term memory — powered by Ebbinghaus forgetting curves,
multi-dimensional weighted scoring, LLM-driven parameter tuning, and incremental user-profile consolidation.

> Extracted as a standalone package from the production
> project [fireflymall-ai-customer-service](https://github.com/ljm-codes/fireflymall-ai-customer-service). Works
> offline
> out of the box — **zero external services required** for tests and demo; Redis / RabbitMQ / real embeddings are
> optional
> plug-ins.

---

## Why

Standard chat contexts forget. The official `SummarizationMiddleware` compresses old turns into a summary — useful, but
it loses retrievable detail and doesn't build a *user model*. This middleware implements a **three-layer memory**
designed around how human memory actually works:

```
┌────────────────────────────────────────────────────────────────────┐
│ 1. Working memory (short-term)                                     │
│    timestamped message list — the live conversation                │
└──────────────────────────┬─────────────────────────────────────────┘
                           │  maturity M(Δt) ≥ threshold  (Ebbinghaus)
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ 2. Staging store (mid-term) — vector RAG                          │
│    dialogue slices with metadata: theme / type / time /            │
│    strengthen_count. Retrieved on demand, strengthened on reuse.   │
└──────────────────────────┬─────────────────────────────────────────┘
                           │  cursor-based incremental summarization
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ 3. Long-term store — structured user profile (UserProfile)         │
│    facts / preferences / goals / constraints ... written per user  │
└────────────────────────────────────────────────────────────────────┘
```

When the context budget is approached, the middleware retrieves the most relevant slices and injects them **into the
system prompt** (core → profile → fragments, ordered for KV-cache-prefix friendliness), then truncates the raw history —
keeping context bounded while the important facts survive.

## The math

| Quantity   | Formula                                                                       | Meaning                                                    |
|------------|-------------------------------------------------------------------------------|------------------------------------------------------------|
| Maturity   | `M(Δt) = 1 − exp(−(Δt/τ_m)^c_m)`                                              | when the oldest message is "ripe" → slice it               |
| Time decay | `T(m) = exp(−(Δt/τ)^c)`                                                       | Ebbinghaus forgetting curve; τ ≈ time to 37% strength      |
| Importance | `F(m) = clamp(w₀ + w₁·u + w₂·(1−exp(−refresh·k)), 0, 1)`, `u = 1 − Π(1 − vᵢ)` | type value (**probability union**) + consolidation count   |
| Relevance  | `R(m, q) = max(0, (cos − median) / (max − median))` (within candidates)       | semantic fit to the current topic (**set-relative scale**) |
| Score      | `S(m) = α·R + β·T + γ·F + δ`                                                  | ranking for top-K injection                                |

The reasoning behind both calibrations (including "why not the alternatives") lives in the docstrings of
`type_score_union` / `relevance_scores` in `memory_middleware/formula.py`: summing type scores saturates
the clamp (measured: 5/6 candidates pinned at exactly F=1.000, consolidation invisible), while a probability
union is bounded, increasing and marginally diminishing; and an absolute relevance threshold cannot be
compared across themes/embedding models, whereas a set-relative scale gives the tuned α something to act on
(measured: events where dropping relevance changes the injected set went 1/9 → 5/9).

α/β/γ/δ and w₀/w₁/w₂ are **re-tuned by the LLM per conversation theme** (structured output, α+β+γ=1, w₀+w₁+w₂=1), so
weighting adapts to what the user is talking about.

> **Measured reading (v0.2.0 — don't judge the formula by the formula)**: the measurement first found two
> calibration defects — the LLM gave α the largest weight in 8 of 9 events while αR had the *smallest*
> weighted spread of the three dimensions (all-zero in 3/10 events), and F was pinned by the clamp
> (40/165 candidates at exactly 1.000, hiding consolidation entirely). **Both were then fixed from that
> evidence** (relevance → set-relative scale, type scores → probability union, see the table above); on the
> same data the before/after is: **F saturation 40/165 → 0/180; consolidation "eaten by the clamp" 24/32 → 0/32;
> events where dropping relevance changes the injected set 1/9 → 5/9; recall and injected volume do not
> regress.** Still true afterwards: decay is ~constant within a session (τ=12 h is a cross-day scale), and the
> `always_inject_types` floor remains the single most influential knob (6/9 events).
> Every event's tuned params and each candidate's R/T/F **plus per-dimension weighted contributions** are
> archived in `bmdm.scoring_log` (`benchmarks/analyze_scoring_log.py`; `--pair old new` prints the comparison).

## Quickstart

```bash
pip install memory-middleware           # from PyPI
# or from source:
git clone https://github.com/ljm-codes/memory-middleware.git && cd memory-middleware
pip install -e ".[dev]"
python examples/quickstart_offline.py   # fully offline, scripted models
pytest                                  # 115 offline tests, no services, <1s (3 integration cases need -m integration)
```

To use with a real model (DeepSeek):

```python
from memory_middleware import BalancedMultiDimensionMemory, MemoryConfig

memory = BalancedMultiDimensionMemory(MemoryConfig.from_vocation('customer service'))

agent = create_agent(
    model, tools,
    middleware=[memory],  # same slot as SummarizationMiddleware
)
```

Runtime requirements: `runtime.context.user_id` (or `state['user_id']`), `runtime.store` (any LangGraph store,
e.g. `InMemoryStore`), and state channels `messages` / `new_msg_idx` / `user_profile` / `system_prompt`.

**Message timestamps are self-contained**: any message missing the `time` field is
stamped automatically on first sight (treated as "now"), so the Ebbinghaus maturity /
decay machinery works with no external ingestion hook. Existing timestamps are never
overwritten — **for exact timestamps, stamp at ingestion in your own pipeline (the
upstream pattern); an external stamp always wins**. The fallback's bias is
conservative: unknown age is treated as "now", which delays (never accelerates)
slicing/decay/consolidation. A debug log records every fallback stamp. (The upstream
project also stamps at its message-entry node; both are idempotent and can coexist.)

See `examples/quickstart_real.py` for a complete harness with **token & cost tracking** (≈ ¥0.004 / 4-turn conversation
on deepseek-v4-flash).

## Dependency decoupling

Everything external is an **injectable narrow protocol with an offline default** — the test suite and demo run with zero
services:

| Concern                                | Protocol / injection point                              | Offline default                            | Production plug-in                                     |
|----------------------------------------|---------------------------------------------------------|--------------------------------------------|--------------------------------------------------------|
| Fragment id cursor                     | `KVStore` (`aget`/`aset`/`aincr`)                       | `MemoryKVStore` (in-process)               | `RedisKVStore` (atomic, multi-process)                 |
| Failure recovery                       | `ErrorRecovery` (`on_split_error` / `on_summary_error`) | `NullRecovery` (log only)                  | `RabbitMQRecovery` (durable queue + optional notifier) |
| Embeddings                             | langchain `Embeddings`                                  | `HashEmbeddings` (deterministic, offline)  | DashScope / OpenAI / Ollama                            |
| Vector store                           | `VectorStore` interface                                 | `SQLiteVecStore` (local file / `:memory:`) | Milvus / ES via the same interface                     |
| Chat / split / summary / tuning models | model instances or `model_factory`                      | scripted fakes                             | any `init_chat_model` provider (DeepSeek, OpenAI, …)   |
| Token counting                         | `TokenCounter`                                          | chars/3.3 heuristic                        | tiktoken / transformers / provider usage               |
| Base system prompt                     | `MemoryConfig.initial_prompt`                           | —                                          | app-provided prompt                                    |

Honest limitations (documented, not hidden):

- `MemoryKVStore` is single-process only — use `RedisKVStore` for multi-process deployments.
- `NullRecovery` drops failed slice/summarize payloads (the round is skipped, the agent keeps working) —
  attach `RabbitMQRecovery` if you need retries.
- `HashEmbeddings` has low semantic quality — swap in a real embedding model for production recall.

## Testing strategy

**Offline suite (default `pytest`, no API keys):**

- Layer 0 — pure math: Ebbinghaus formulas, clamp/constraints, profile merging, fragment range (start_idx/end_idx)
  parsing, prompt ordering
- Layer 1 — component tests with fake LLMs + in-memory stores: slicing flow, incremental summary (cursor never rewinds),
  retrieval injection, cross-user isolation, degradation paths, concurrent access
- The concurrency test caught a real bug (shared sqlite connection racing across users) that also existed in the
  original project

**Integration suite (`pytest -m integration`, requires `DEEPSEEK_API_KEY`):**

- End-to-end pipeline with the real model: slices land, profile is written, fragments are injected, answers mention
  early facts
- Token & cost tracking per call — using the official `usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens` fields,
  priced at current DeepSeek rates (¥0.02 / ¥1 / ¥2 per 1M tokens; configurable via `memory_middleware.cost`)

## Measured results

All numbers below are from actual runs, not estimates, produced by **v0.2.0**;
every benchmark record in `benchmarks/output/recall_results.json` carries a `_meta` stamp
(version / commit / run time / full config), so each number can be traced to its own run. Cache accounting uses
DeepSeek's
official billing fields `usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens`
(¥0.02 per 1M for cache-hit input, ¥1.0 for miss, ¥2.0 for output — configurable in
`memory_middleware/cost.py`).

**Offline suite** — 115 tests, ~1 s, zero services, zero API keys (3 integration cases skipped by default).

**End-to-end with real DeepSeek (deepseek-v4-flash), 4-turn conversation**
(integration test `tests/integration/test_cost_tracking.py`, run 2026-10-05 — after the
"slicing runs with the event" fix):

| Metric                 | Value                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------------|
| LLM calls              | 12 — **4 main-chat calls** and **8 middleware-internal calls** (slice 3 / tune 3 / induction 2) |
| Input tokens           | 7,964                                                                                           |
| Output tokens          | 1,061                                                                                           |
| Average cache hit rate | 51.4% (middleware-internal calls: 71.7–80.4%)                                                   |
| Total cost             | ¥0.004316                                                                                       |
| Cost split             | main chat **15%**; middleware internals **85%** (slice 34% / tune 34% / induction 17%)          |

Per-call accounting comes from the official `usage` fields, priced at ¥0.02 / ¥1 / ¥2
per 1M tokens (`memory_middleware/cost.py`). The main-chat and the three middleware call
types are collected in the same run through one usage handler and printed grouped by
`tag` (main / split / summary / param) — see `tests/integration/conftest.py`. A single run
carries model randomness: two consecutive runs gave 12 calls, 7,912–7,964 input tokens,
¥0.0043–0.0045; run it a few times before reading trends.

**Recall evidence** — after 5 real turns (name + hobby stated in turn 1), the final
reply of the **main-agent chat call** was:

> 张三先生，我记得您上次提到游泳爱好。考虑到您的运动场景，我为您推荐 IP68 防水手机…

### Three-way comparison (needle-in-haystack)

Harness: `benchmarks/three_way_recall.py` (deterministic facts + filler + questions,
same conversation text for all three configs, context budget = 8 messages, real
DeepSeek + DashScope embeddings). Fact recall = answer contains the fact keyword.

Reported under **two regimes** (re-measured 2026-10-05, after fixing "slicing runs
with the event"):

- **Production regime**: design gate values (`slice=0.7` / `long_term=0.9` / `τ_m=600`)
    + a virtual clock (35 s per turn) + a 50-message window budget — how the middleware
      behaves at real conversation pacing.
- **Gates-open regime**: maturity gate pinned to 0.0 and an 8-message budget, so the
  machinery fires densely inside a time-compressed harness — used to measure the
  *per-event unit cost* upper bound.

| Conversation length | No-memory baseline | SummarizationMiddleware | BMDM | Regime         |
|---------------------|--------------------|-------------------------|------|----------------|
| 8 turns             | 0/4                | 3/4                     | 3/4  | gates-open     |
| 16 turns            | 0/4                | 3/4                     | 3/4  | gates-open     |
| 24 turns            | 0/4                | 3/4                     | 3/4  | gates-open     |
| 24 turns            | 0/4                | 3/4                     | 3/4  | **production** |

![Long-term memory recall benchmark](benchmarks/output/recall_benchmark.png)

> The chart shows the **gates-open** regime (8/16/24 turns); the "production" row of the table
> above is not in it — that is a separate 24-turn run tagged `-gated`.

Cost at 24 turns (input tokens / calls):

| Regime                                     | baseline   | Summarization | BMDM        | BMDM/Summarization            |
|--------------------------------------------|------------|---------------|-------------|-------------------------------|
| gates-open                                 | 3,796 / 31 | 12,089 / 40   | 28,768 / 52 | 2.38× input · 1.30× calls     |
| production (unequal budgets¹)              | 3,880 / 31 | 12,029 / 40   | 13,223 / 33 | 1.10× input · 0.83× calls¹    |
| **production · matched budgets** (both 50) | 3,718 / 31 | 14,088 / 32   | 14,733 / 33 | **1.05× input · 1.03× calls** |

¹ That run gave the two sides unequal window budgets (BMDM 50 messages vs summarization 10), inflating
the official trigger frequency ~9×; "fewer calls" was a threshold artifact. **The matched-budget run is
the comparable one**: both sides fire exactly 1 event, BMDM spends 2 internal calls per event
(slice + tune) against the official's 1, so totals are essentially equal — with recall on par.

> **Correction of historical numbers**: earlier versions quoted "BMDM 66,751 tokens /
> 99 calls at 24 turns ≈ 3–5.6× Summarization". That figure stacked two distortions:
> ① the maturity gate was pinned to `0.0` (i.e. removed), and ② slicing used to run
> **independently of the primary trigger event**, so whenever the window was not
> reset it called the splitter model every single turn (measured: over half of all
> internal calls). With slicing folded into the event (one event = slice + tune +
> profile induction when fragments are mature), cost at production pacing is on par
> with summarization. High-salience fact recall is also summarization's home turf —
> BMDM's differentiation is in the detail-retention and cross-session sections below.

Honest reading:

- With a strict 8-message budget, the **no-memory baseline forgets everything** —
  facts stated at turn 1 are simply gone.
- Both memory mechanisms hold **3/4 at 8/16/24 turns** (BMDM on par with
  summarization here). Note single-sample recall swings by ±2 — re-running the same
  config, summarization itself went 1/4 → 3/4 — use `--samples 3` before reading trends.
- Both mechanisms lose the *same* fact (a "Wednesdays are busy" constraint): summaries
  drop such details, and the profile constraint field is not reliably filled.
- BMDM's distinguishing value is beyond this short-horizon chart: a **per-user
  structured profile + theme-retrievable fragments** (vs an opaque summary string),
  and cross-session persistence. First pass, single sample per length — larger
  samples / longer conversations would sharpen the separation.
- Data file: `benchmarks/output/recall_results.json` (regenerate the chart with
  `python benchmarks/three_way_recall.py --chart-only`).

### Cross-session recall (the dimension the within-session benchmark cannot see)

SummarizationMiddleware's summary lives in the session state — a **brand-new thread
starts with nothing**. BMDM persists fragments + profile in the store. Session 1
states facts, session 2 (new thread, warm-up turns + the same 4 questions) asks:

| Session-1 length | No-memory baseline | SummarizationMiddleware | BMDM    |
|------------------|--------------------|-------------------------|---------|
| 8 turns          | 0/4                | 0/4                     | **3/4** |
| 16 turns         | 0/4                | 0/4                     | **3/4** |
| 24 turns         | 0/4                | 0/4                     | **3/4** |

![Cross-session recall benchmark](benchmarks/output/cross_session_recall.png)

Re-measured after the 2026-10-05 fix (16/24 turns): BMDM still **3/4**, total calls down
to **11** (3 internal: slice + tune + induction; 17 before the fix); both baselines remain 0/4.

Honest reading:

- Without a cross-session mechanism, a new session recalls **nothing** — both baselines
  are 0/4 at every length, exactly as designed.
- BMDM recalls 3/4 in a brand-new session (name/occupation/hobby via the persisted
  profile; the "Wednesdays busy" constraint is captured in the profile but not
  surfaced in the answer — same constraint-detail gap seen within-session).
- Characteristic worth knowing: memory injection only engages once the window reaches
  the trigger threshold (8 messages) — the first turns of a new session run without
  injected memory, by design.
- Observed data-quality issue: the incremental profile merge dedups exact duplicates
  only, so `user_constraints` accumulated ~7–14 paraphrases of the same constraint
  across slices (semantic-dedup or list capping is a future improvement).

### Detail retention (the "summaries lose specifics" claim, measured)

8 low-salience details (pet name, order number, card tail, birthday, phone model,
free-time window, favorite food, address) stated early, then queried one by one:

| Conversation length | Regime                                                                                          | No-memory baseline | SummarizationMiddleware | BMDM                             |
|---------------------|-------------------------------------------------------------------------------------------------|--------------------|-------------------------|----------------------------------|
| 24 turns            | production · matched budgets · 3 samples (**old defaults**: prefilter k=6, no floor, `top_k=3`) | 0.33/8             | 7.33/8 (8·7·7)          | 6.67/8 (4·8·8, σ≈1.9)            |
| 24 turns            | same · **no prefilter + floor** (`retrieve_k=None` + identity/preference floor)                 | 0.33/8             | 7.67/8 (7·8·8)          | **7.67/8 (7·8·8, σ≈0.5)**        |
| 24 turns            | same · **current defaults** (no prefilter + floor, `top_k=4`)                                   | 0.33/8             | 7.67/8 (8·8·7)          | **8.0/8 (8·8·8, zero variance)** |

**Memory payload volume** (same counter for both sides — chars/3.3, body text only; the official
baseline is taken from the **same run**, never mixed across batches):

| Config                                                 | BMDM payload              | Official summary, same run | vs official |
|--------------------------------------------------------|---------------------------|----------------------------|-------------|
| **current defaults** (no prefilter + floor, `top_k=4`) | 223·251·217 → **230 tok** | 236·273·302 → **270 tok**  | **0.85×**   |
| no prefilter + floor, `top_k=3`                        | 91·104·166 → **120 tok**  | 221·265·280 → **255 tok**  | **0.47×**   |

So under the **current defaults** the injected payload is ~15% smaller than the official
summary (230 vs 270 tok) at slightly higher recall (8.0/8 vs 7.67/8); narrowing the slots to
`top_k=3` halves the payload at recall on par with the official — fragments are structured
extraction, not prose paraphrase.

Why the **old-default** selection wobbled (4·8·8 / 8·8·6): details live concentrated in a few
fragments, and *which* fragments get injected is a competition — **either gate can drop
the fragment carrying them (the theme-similarity prefilter `retrieve_k=6`, or the slot
limit `top_k=3`), and losing it loses the whole block**. Split granularity varies per run,
so whether it gets dropped varies too. Observed: a store of 8 fragments where the theme
"pet & order-refund" pushed 「personal basics」and「preferences & address」out of the
candidate set → 3/8. Fix: widen `retrieve_k` to cover the store (local vector search, no
LLM cost) + `always_inject_types=('identity','preference')` so high-value types always
hold a seat (consistent with the formula's own `TYPE_SCORE_MAP` 0.95/0.80).

![Detail retention benchmark](benchmarks/output/detail_retention.png)

> The chart shows the **current-default** regime (no prefilter + floor, `top_k=4`), 24 turns,
> mean of 3 samples: BMDM 100%, official 96%, baseline 4%.

Honest reading:

- **Under matched budgets and with no handicaps, the two are essentially tied on
  detail retention** (**old-default** regime: prefilter k=6, `top_k=3`): summarization
  7.33/8, BMDM 6.67/8 (3 samples). BMDM is the noisier one — one sample dropped to 4/8
  (theme-gated retrieval missed that batch), while summarization stayed at 7–8/8. Under
  the current defaults BMDM rises to 8.0/8 (zero variance) against the official's 7.67/8.
- Cost (old-default regime): BMDM averages 20,728 input tokens vs summarization's 29,414 → **0.70×** (the
  official's single summary ingests ~46 messages at once); calls 42 vs 41 → 1.02×.
  So the claim is "**as-good recall for ~30% less input**", *not* "same cost for
  higher recall".
- The normalized unit cost still holds: BMDM spends 2 structured calls per event
  (slice + tune; induction not yet mature) against the official's 1.
- ⚠️ **Correction**: earlier versions reported "BMDM 7–8/8 vs summarization 0–2/8 at
  ~6× the cost". That was an artifact of **two handicaps on the official**: ① its
  summarizer shared the chat model (`max_tokens=100`, so summaries were truncated —
  the longer the conversation, the worse); ② the two sides were given unequal trigger
  budgets (BMDM 50 messages vs the official's 10). With both removed, the official's
  24-turn detail recall rises from 0/8 to 7–8/8. The old numbers are void.

### Why the middleware calls carry no conversation history (measured)

A design decision that looks counter-intuitive — adding history *raises* the cache hit
rate but *raises* the bill. Measured on deepseek-v4-flash, 8 groups × 6 calls
(input-cost ratios only; both regimes come from **the same real run**, re-priced from
the raw usage fields):

| Call type   | First run (empty cache) | Immediate re-run (identical bytes) |
|-------------|-------------------------|------------------------------------|
| Math tuning | B/A = 1.19×             | 0.96×                              |
| Slice       | B/A = 1.48×             | 1.07×                              |
| Summary     | B/A = 1.98×             | **1.09×**                          |

- **First run** (the production reality: conversation content never byte-repeats — the
  middleware processes disjoint increments): every history token is paid at full price
  on first write, then at the discounted read price. B always loses, and the gap grows
  with history length (1.19 → 1.98×).
- **Immediate re-run** (the *most* favourable case for B: byte-identical requests
  re-sent within the cache TTL): **the best case is only a tie** (math 0.96×); slice and
  summary still cost 7–9% more — B's input is simply longer, and a 2% read price
  multiplied by a bigger number still loses.
- **The trap**: in the warm regime B-summary reached an **84.8% hit rate** (A-summary:
  53.4%) yet still cost **9% more**. **Hit rate is not cost.**
- **Control** (`A-math-same`): the same request re-sent byte-identically 6 times is
  **identical in both regimes** — 379 hit / **134 miss** every time, same cost to the
  cent. The trailing cache unit (128-token aligned) always bills at full price; a
  pre-warmed cache is not free.
- **Positive control** (`D-chat`, main-chat style: growing prefix, no ignore prefix):
  the warm run is **55% cheaper** than the first run (¥0.002492 → ¥0.001112, hit rate
  65.1% → 85.6%) — the prefix dividend is real, it just isn't on middleware calls whose
  input is new every time.

Conclusion: the three middleware calls stay stateless; the KV-cache-prefix thinking is
applied where it pays — the static system prompts (cached across calls and users) and
the main chat loop's growing history.

> **Provenance**: `tests/data/bench_cache_rate_cold.json` (first run) and
> `bench_cache_rate_result.json` (immediate re-run), raw logs alongside as
> `bench_cache_rate_*.log`. Reproduce with
> `python tests/bench_cache_rate.py --out bench_cache_rate_cold.json`, then run the same
> command again without `--out` for the warm regime. **The first file is only "cold" if
> the cache is empty** — wait out the TTL or change a prompt first. Earlier versions
> quoted "cold 1.82/1.89/2.30×, warm 0.59/0.71/1.90×" from a different design that
> replayed the previous request (feeding new content to A and old content to B —
> structurally biased toward B); that raw data was overwritten at the same path and is
> **no longer cited**.

## Configuration

Key fields of `MemoryConfig` (see `memory_middleware/config.py` for all):

| Field                           | Default                     | Meaning                                                                                           |
|---------------------------------|-----------------------------|---------------------------------------------------------------------------------------------------|
| `model_name` / `model_kwargs`   | `deepseek-v4-flash`         | model used by the three internal calls                                                            |
| `vocation`                      | customer service            | τ_m / τ / c / thresholds presets (collaborative creation / customer service / accompany / custom) |
| `pattern` / `trigger_threshold` | fraction / 0.8              | when retrieval injection fires (fraction of context budget, token count, or message count)        |
| `rag_db_path`                   | `memory_fragments.db`       | local sqlite file (`:memory:` allowed)                                                            |
| `retrieve_k` / `top_k`          | `None` / 4                  | candidates retrieved (`None` = no prefilter, whole store scored) → top-K injected                 |
| `always_inject_types`           | `('identity','preference')` | types that always hold a seat (matches `TYPE_SCORE_MAP` 0.95/0.80)                                |

## Upstream project

This package is a decoupled extraction of the memory middleware that powers
[fireflymall-ai-customer-service](https://github.com/ljm-codes/fireflymall-ai-customer-service)
— a production intelligent customer-service agent (LangGraph + DeepSeek). The upstream
repo contains the business-coupled version (Redis / RabbitMQ / DashScope wiring) plus
the reproducible cache-rate benchmark (`tests/bench_cache_rate.py`, raw data in
`tests/data/bench_cache_rate_*.json` with matching `.log` files) whose results are shown above.

### Standalone vs upstream — what you lose

This package is a **decoupled subset**, not a superset. Before using it in production,
know the differences:

| Concern          | This package (offline default)                   | Upstream production wiring                               |
|------------------|--------------------------------------------------|----------------------------------------------------------|
| Failure recovery | `NullRecovery` — failed rounds are skipped       | RabbitMQ durable queues + email alerts (`memory_rag.py`) |
| Fragment cursor  | `MemoryKVStore` — single process only            | Redis (multi-process safe)                               |
| Embeddings       | `HashEmbeddings` — offline, low semantic quality | DashScope `text-embedding-v4`                            |
| Token counting   | chars/3.3 heuristic                              | tiktoken / transformers                                  |

The full production implementation lives in the upstream repo at
`Tools/middleware/memory/time_memory.py` (BalancedMultiDimensionMemory),
`memory_rag.py` (spliter / incremental summary / RabbitMQ recovery) and
`Tools/middleware/compose.py` (middleware onion-chain composition). If you need
production-grade resilience, either use the upstream project directly or plug the
matching implementations into the injection points here.

## License

MIT
