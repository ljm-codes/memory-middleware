# BalancedMultiDimensionMemory

A **LangGraph `AgentMiddleware`** that gives LLM agents long-term memory — powered by Ebbinghaus forgetting curves, multi-dimensional weighted scoring, LLM-driven parameter tuning, and incremental user-profile consolidation.

> Extracted as a standalone package from the production project [fireflymall-ai-customer-service](https://github.com/lijia-ming/fireflymall-ai-customer-service). Works offline out of the box — **zero external services required** for tests and demo; Redis / RabbitMQ / real embeddings are optional plug-ins.

---

## Why

Standard chat contexts forget. The official `SummarizationMiddleware` compresses old turns into a summary — useful, but it loses retrievable detail and doesn't build a *user model*. This middleware implements a **three-layer memory** designed around how human memory actually works:

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

When the context budget is approached, the middleware retrieves the most relevant slices and injects them **into the system prompt** (core → profile → fragments, ordered for KV-cache-prefix friendliness), then truncates the raw history — keeping context bounded while the important facts survive.

## The math

| Quantity | Formula | Meaning |
|---|---|---|
| Maturity | `M(Δt) = 1 − exp(−(Δt/τ_m)^c_m)` | when the oldest message is "ripe" → slice it |
| Time decay | `T(m) = exp(−(Δt/τ)^c)` | Ebbinghaus forgetting curve; τ ≈ time to 37% strength |
| Importance | `F(m) = clamp(w₀ + w₁·Σ(vᵢ·I_type(i)) + w₂·(1−exp(−refresh·k)), 0, 1)` | inherent value by memory type + consolidation count |
| Relevance | `R(m, q) = max(0, cos(m, q) − θ_min)` | semantic fit to the current topic |
| Score | `S(m) = α·R + β·T + γ·F + δ` | ranking for top-K injection |

α/β/γ/δ and w₀/w₁/w₂ are **re-tuned by the LLM per conversation theme** (structured output, α+β+γ=1, w₀+w₁+w₂=1), so weighting adapts to what the user is talking about.

## Quickstart

```bash
pip install memory-middleware           # from PyPI
# or from source:
git clone https://github.com/lijia-ming/memory-middleware.git && cd memory-middleware
pip install -e ".[dev]"
python examples/quickstart_offline.py   # fully offline, scripted models
pytest                                  # 100 offline tests, no services, <1s
```

To use with a real model (DeepSeek):

```python
from memory_middleware import BalancedMultiDimensionMemory, MemoryConfig

memory = BalancedMultiDimensionMemory(MemoryConfig.from_vocation('customer service'))

agent = create_agent(
    model, tools,
    middleware=[memory],                 # same slot as SummarizationMiddleware
)
```

Runtime requirements: `runtime.context.user_id` (or `state['user_id']`), `runtime.store` (any LangGraph store, e.g. `InMemoryStore`), and state channels `messages` / `new_msg_idx` / `user_profile` / `system_prompt`.

**Message timestamps are self-contained**: any message missing the `time` field is
stamped automatically on first sight (treated as "now"), so the Ebbinghaus maturity /
decay machinery works with no external ingestion hook. Existing timestamps are never
overwritten — **for exact timestamps, stamp at ingestion in your own pipeline (the
upstream pattern); an external stamp always wins**. The fallback's bias is
conservative: unknown age is treated as "now", which delays (never accelerates)
slicing/decay/consolidation. A debug log records every fallback stamp. (The upstream
project also stamps at its message-entry node; both are idempotent and can coexist.)

See `examples/quickstart_real.py` for a complete harness with **token & cost tracking** (≈ ¥0.003 / 4-turn conversation on deepseek-v4-flash).

## Dependency decoupling

Everything external is an **injectable narrow protocol with an offline default** — the test suite and demo run with zero services:

| Concern | Protocol / injection point | Offline default | Production plug-in |
|---|---|---|---|
| Fragment id cursor | `KVStore` (`aget`/`aset`/`aincr`) | `MemoryKVStore` (in-process) | `RedisKVStore` (atomic, multi-process) |
| Failure recovery | `ErrorRecovery` (`on_split_error` / `on_summary_error`) | `NullRecovery` (log only) | `RabbitMQRecovery` (durable queue + optional notifier) |
| Embeddings | langchain `Embeddings` | `HashEmbeddings` (deterministic, offline) | DashScope / OpenAI / Ollama |
| Vector store | `VectorStore` interface | `SQLiteVecStore` (local file / `:memory:`) | Milvus / ES via the same interface |
| Chat / split / summary / tuning models | model instances or `model_factory` | scripted fakes | any `init_chat_model` provider (DeepSeek, OpenAI, …) |
| Token counting | `TokenCounter` | chars/3.3 heuristic | tiktoken / transformers / provider usage |
| Base system prompt | `MemoryConfig.initial_prompt` | — | app-provided prompt |

Honest limitations (documented, not hidden):
- `MemoryKVStore` is single-process only — use `RedisKVStore` for multi-process deployments.
- `NullRecovery` drops failed slice/summarize payloads (the round is skipped, the agent keeps working) — attach `RabbitMQRecovery` if you need retries.
- `HashEmbeddings` has low semantic quality — swap in a real embedding model for production recall.

## Testing strategy

**Offline suite (default `pytest`, no API keys):**
- Layer 0 — pure math: Ebbinghaus formulas, clamp/constraints, profile merging, fragment range (start_idx/end_idx) parsing, prompt ordering
- Layer 1 — component tests with fake LLMs + in-memory stores: slicing flow, incremental summary (cursor never rewinds), retrieval injection, cross-user isolation, degradation paths, concurrent access
- The concurrency test caught a real bug (shared sqlite connection racing across users) that also existed in the original project

**Integration suite (`pytest -m integration`, requires `DEEPSEEK_API_KEY`):**
- End-to-end pipeline with the real model: slices land, profile is written, fragments are injected, answers mention early facts
- Token & cost tracking per call — using the official `usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens` fields, priced at current DeepSeek rates (¥0.02 / ¥1 / ¥2 per 1M tokens; configurable via `memory_middleware.cost`)

## Measured results

All numbers below are from actual runs, not estimates. Cache accounting uses DeepSeek's
official billing fields `usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens`
(¥0.02 per 1M for cache-hit input, ¥1.0 for miss, ¥2.0 for output — configurable in
`memory_middleware/cost.py`).

**Offline suite** — 100 tests, ~1 s, zero services, zero API keys.

**End-to-end with real DeepSeek (deepseek-v4-flash), 4-turn conversation**
(integration test `tests/integration/test_cost_tracking.py`, run 2026-08-16):

| Metric | Value |
|---|---|
| LLM calls | 13 — of which **4 are main-agent chat calls** (the conversation loop) and **9 are middleware-internal calls** (theme slicing / incremental summarization / parameter tuning) |
| Input tokens | 8,276 |
| Output tokens | 1,009 |
| Average cache hit rate | 62.1% (middleware-internal slice/summary calls: 86–93%) |
| Total cost | ¥0.0033 |

Per-call accounting comes from the official `usage` fields, priced at ¥0.02 / ¥1 / ¥2
per 1M tokens (`memory_middleware/cost.py`). The main-agent and the middleware-internal
calls are mixed in the same collection run; both go through the same usage collection
handler (see `tests/integration/conftest.py`).

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

| Conversation length | No-memory baseline | SummarizationMiddleware | BMDM | Regime |
|---|---|---|---|---|
| 8 turns | 0/4 | 3/4 | 3/4 | gates-open |
| 16 turns | 0/4 | 3/4 | 3/4 | gates-open |
| 24 turns | 0/4 | 3/4 | 3/4 | gates-open |
| 24 turns | 0/4 | 3/4 | 3/4 | **production** |

![Long-term memory recall benchmark](benchmarks/output/recall_benchmark.png)

Cost at 24 turns (input tokens / calls):

| Regime | baseline | Summarization | BMDM | BMDM/Summarization |
|---|---|---|---|---|
| gates-open | 3,796 / 31 | 12,089 / 40 | 28,768 / 52 | 2.38× input · 1.30× calls |
| production (unequal budgets¹) | 3,880 / 31 | 12,029 / 40 | 13,223 / 33 | 1.10× input · 0.83× calls¹ |
| **production · matched budgets** (both 50) | 3,718 / 31 | 14,088 / 32 | 14,733 / 33 | **1.05× input · 1.03× calls** |

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

### Cross-session recall (the dimension the within-session benchmark cannot see)

SummarizationMiddleware's summary lives in the session state — a **brand-new thread
starts with nothing**. BMDM persists fragments + profile in the store. Session 1
states facts, session 2 (new thread, warm-up turns + the same 4 questions) asks:

| Session-1 length | No-memory baseline | SummarizationMiddleware | BMDM |
|---|---|---|---|
| 8 turns | 0/4 | 0/4 | **3/4** |
| 16 turns | 0/4 | 0/4 | **3/4** |
| 24 turns | 0/4 | 0/4 | **3/4** |

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

| Conversation length | No-memory baseline | SummarizationMiddleware | BMDM |
|---|---|---|---|
| 24 turns | production · matched budgets · 3 samples (**old defaults**: prefilter k=6, no floor) | 0.33/8 | 7.33/8 (8·7·7) | 6.67/8 (4·8·8, σ≈1.9) |
| 24 turns | same · **new defaults** (no prefilter + identity/preference floor, top_k=3) | 0.33/8 | 7.67/8 (7·8·8) | **7.67/8 (7·8·8, σ≈0.5)** |
| 24 turns | new defaults + `top_k=4` | 0.33/8 | 7.67/8 (8·8·7) | **8.0/8 (8·8·8, zero variance)** |

**Memory payload volume** (same counter for both sides — chars/3.3, body text only):

| Config | payload | vs official |
|---|---|---|
| Summarization summary body | 221·265·280 → **255 tok** | — |
| **new defaults** (no prefilter + floor, top_k=3) | 91·104·166 → **120 tok** | **0.47×** |
| new defaults + top_k=4 | 223·251·217 → **230 tok** | 0.90× |

So under the defaults the injected payload is **half the official summary** at equal recall —
fragments are structured extraction, not prose paraphrase.

Why the default selection wobbles (4·8·8 / 8·8·6): details live concentrated in a few
fragments, and *which* fragments get injected is a competition — **either gate can drop
the fragment carrying them (the theme-similarity prefilter `retrieve_k=6`, or the slot
limit `top_k=3`), and losing it loses the whole block**. Split granularity varies per run,
so whether it gets dropped varies too. Observed: a store of 8 fragments where the theme
"pet & order-refund" pushed 「personal basics」and「preferences & address」out of the
candidate set → 3/8. Fix: widen `retrieve_k` to cover the store (local vector search, no
LLM cost) + `always_inject_types=('identity','preference')` so high-value types always
hold a seat (consistent with the formula's own `TYPE_SCORE_MAP` 0.95/0.80).

![Detail retention benchmark](benchmarks/output/detail_retention.png)

Honest reading:
- **Under matched budgets and with no handicaps, the two are essentially tied on
  detail retention**: summarization 7.33/8, BMDM 6.67/8 (3 samples). BMDM is the
  noisier one — one sample dropped to 4/8 (theme-gated retrieval missed that batch),
  while summarization stayed at 7–8/8.
- Cost: BMDM averages 20,728 input tokens vs summarization's 29,414 → **0.70×** (the
  official's single summary ingests ~46 messages at once); calls 42 vs 41 → 1.02×.
  So the claim is "**~30% less input for the same recall**", *not* "same cost for
  higher recall".
- The normalized unit cost still holds: BMDM spends 2 structured calls per event
  (slice + tune; induction not yet mature) against the official's 1.
- ⚠️ **Correction**: earlier versions reported "BMDM 7–8/8 vs summarization 0–2/8 at
  ~6× the cost". That was an artifact of **two handicaps on the official**: ① its
  summarizer shared the chat model (`max_tokens=100`, so summaries were truncated —
  the longer the conversation, the worse); ② the two sides were given unequal trigger
  budgets (BMDM 50 messages vs the official's 10). With both removed, the official's
  24-turn detail recall rises from 0/8 to 7–8/8. The old numbers are void.

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

**Recall evidence** — after 5 real turns (name + hobby stated in turn 1), the final
reply was:

> 张三先生，我记得您上次提到游泳爱好。考虑到您的运动场景，我为您推荐 IP68 防水手机…

### Why the middleware calls carry no conversation history (measured)

A design decision that looks counter-intuitive — adding history *raises* the cache hit
rate but *raises* the bill. Measured on deepseek-v4-flash, 8 groups × 6 calls,
input-cost ratios (B = carrying an "ignore" history prefix, A = stateless baseline):

| Call type | Cold regime (first write) | Warm replay (pre-warmed cache) |
|---|---|---|
| Math tuning | B/A = 1.82× | 0.59× |
| Slice | B/A = 1.89× | 0.71× |
| Summary | B/A = 2.30× | **1.90×** |

- **Cold regime** (the production reality: conversation content never byte-repeats —
  the middleware processes disjoint increments): every history token is paid at full
  price on first write, then at the discounted read price. B always loses.
- **Warm replay** (identical requests re-sent within the cache TTL): history amortizes
  and B can win *while the history stays short* — but production history grows without
  bound, so the 2% read cost grows with it.
- **The trap**: in the warm run, B-summary reached a 93.3% hit rate vs A-summary's 0%,
  yet still cost 1.9× more. **Hit rate is not cost.**
- Measured cache mechanics: DeepSeek caches in 128-token units from the start; the
  final unit bills at full price even for byte-identical input.

Conclusion: the three middleware calls stay stateless; the KV-cache-prefix thinking is
applied where it pays — the static system prompts (cached across calls and users) and
the main chat loop's growing history.

## Configuration

Key fields of `MemoryConfig` (see `memory_middleware/config.py` for all):

| Field | Default | Meaning |
|---|---|---|
| `model_name` / `model_kwargs` | `deepseek-v4-flash` | model used by the three internal calls |
| `vocation` | customer service | τ_m / τ / c / thresholds presets (collaborative creation / customer service / accompany / custom) |
| `pattern` / `trigger_threshold` | fraction / 0.8 | when retrieval injection fires (fraction of context budget, token count, or message count) |
| `rag_db_path` | `memory_fragments.db` | local sqlite file (`:memory:` allowed) |
| `retrieve_k` / `top_k` | 6 / 3 | candidates retrieved → top-K injected |

## Upstream project

This package is a decoupled extraction of the memory middleware that powers
[fireflymall-ai-customer-service](https://github.com/lijia-ming/fireflymall-ai-customer-service)
— a production intelligent customer-service agent (LangGraph + DeepSeek). The upstream
repo contains the business-coupled version (Redis / RabbitMQ / DashScope wiring) plus
the reproducible cache-rate benchmark (`bench_cache_rate.py`) whose results are shown above.

### Standalone vs upstream — what you lose

This package is a **decoupled subset**, not a superset. Before using it in production,
know the differences:

| Concern | This package (offline default) | Upstream production wiring |
|---|---|---|
| Failure recovery | `NullRecovery` — failed rounds are skipped | RabbitMQ durable queues + email alerts (`memory_rag.py`) |
| Fragment cursor | `MemoryKVStore` — single process only | Redis (multi-process safe) |
| Embeddings | `HashEmbeddings` — offline, low semantic quality | DashScope `text-embedding-v4` |
| Token counting | chars/3.3 heuristic | tiktoken / transformers |

The full production implementation lives in the upstream repo at
`Tools/middleware/memory/time_memory.py` (BalancedMultiDimensionMemory),
`memory_rag.py` (spliter / incremental summary / RabbitMQ recovery) and
`Tools/middleware/compose.py` (middleware onion-chain composition). If you need
production-grade resilience, either use the upstream project directly or plug the
matching implementations into the injection points here.

## License

MIT
