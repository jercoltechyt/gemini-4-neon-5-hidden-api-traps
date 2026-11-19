# Gemini 4 Neon Migration Auditor & Token Economics Toolkit

A deterministic, zero-dependency offline Python linter and payload validator that catches the **5 hidden API and billing traps** when migrating production applications from Gemini 3 or Gemini 4 Carbon / Argon / Hydrogen to **`gemini-4-neon`**.

> **Note:** API pricing and Artificial Analysis v5.0.1 metrics shown in this repository reflect verified rates and benchmarks as of **November 19, 2026**, and may change over time.

---

## 1. Quick Start — Offline Audit ($0 API Billing)

Requires **Python 3.10+**. Runs 100% locally via AST and payload inspection—zero network calls and zero paid API tokens consumed.

```bash
# 1. Audit a legacy Gemini 3 / Gemini 4 payload file and print all 5 trap detections + fixes
python gemini4_migration_audit.py --check examples/broken_legacy_agent.py

# 2. Verify a migrated Gemini 4 Neon file passes with 0 errors and 0 warnings
python gemini4_migration_audit.py --check examples/fixed_neon_agent.py

# 3. Compare task cost & TTFT across Neon vs. Hydrogen / Carbon / Argon (AA v5.0.1 telemetry)
python gemini4_migration_audit.py --economics

# 4. Run the deterministic unit test suite (zero external dependencies)
python -m unittest discover -s tests -v
```

---

## 2. The 5 Hidden `gemini-4-neon` Traps (And How to Fix Them)

### Trap #1 — `thinking_budget`, `MINIMAL`, and the Endpoint Casing Mismatch (`HTTP 400`)

Passing Gemini 2.5 / Gemini 3 `thinking_budget` integers or Gemini 3 Flash `thinking_level="MINIMAL"` to `gemini-4-neon` returns an immediate **`HTTP 400 Invalid Argument`**.

Furthermore, `thinking_level` casing depends on which API surface you call, and full `"XHIGH"` enum support requires **`google-genai >= 2.34.0`** in Python (`>= 2.30.1` minimum on `GenerateContent`) or **`@google/genai >= 2.32.0`** in TypeScript/JavaScript:

| API Surface | Valid `thinking_level` Values | Casing Rule |
| :--- | :--- | :--- |
| **`GenerateContent` API** (`client.models.generate_content`) | `"LOW"`, `"MEDIUM"`, `"HIGH"`, `"XHIGH"` | **UPPERCASE** strings |
| **`Interactions API`** (`client.interactions.create`) | `"low"`, `"medium"`, `"high"`, `"xhigh"` | **lowercase** strings |

```python
# ❌ BROKEN ON GEMINI 4 NEON (Throws HTTP 400)
from google.genai import types

bad_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        thinking_budget=1024,          # HTTP 400 on Gemini 4 Neon
        thinking_level="MINIMAL",      # HTTP 400 on Gemini 4 Neon
    )
)

# ✅ FIXED FOR GEMINI 4 NEON (GenerateContent API — UPPERCASE)
good_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        thinking_level="MEDIUM",       # Explicit override; avoids default "HIGH" billing spike
    )
)
```

---

### Trap #2 — Parallel `thought_signature` Serialization Rules (`HTTP 400` & `MALFORMED_FUNCTION_CALL`)

When `gemini-4-neon` triggers multiple parallel tool calls inside a single assistant turn, **only the first `functionCall` part** in `candidates[0].content.parts` contains the opaque `thought_signature`:

1. **Omitting `thought_signature`** from the first `functionCall` part on the active turn returns **`HTTP 400`**.
2. **Duplicating/copying `thought_signature`** onto the 2nd, 3rd, or N-th sibling `functionCall` parts in the same parallel turn also returns **`HTTP 400`**.
3. **Stripping `thought_signature` from historical turns** (whether attached to `functionCall` parts or final `text` parts) degrades multi-step reasoning and can trigger **`MALFORMED_FUNCTION_CALL`**.

```python
# ❌ BROKEN CUSTOM SERIALIZER (Duplicates signature across sibling parallel tool calls -> HTTP 400)
first_sig = parts[0].thought_signature
broken_parts = [
    types.Part(function_call=p.function_call, thought_signature=first_sig)
    for p in parts
]

# ✅ FIXED SERIALIZER (Preserves exact per-part signature placement across all turns)
valid_parts = [
    types.Part(
        function_call=p.function_call,
        text=p.text,
        thought_signature=p.thought_signature,  # Present ONLY on parts[0] in parallel tool turns
    )
    for p in parts
]
```

---

### Trap #3 — Explicit Context Caching Is Unsupported + Model Prefill Ban

1. **Explicit Context Caching (`cached_content`, `cachedContents`, `client.caches.create`) is not supported** on `gemini-4-neon`.
2. **Automatic Implicit Caching (`95%` Discount):** Replaced by automatic prefix caching with **durable caching enabled by default** for shared static prefixes of **`>= 4,096` tokens**:
   - **`gemini-4-neon` Input:** `$3.00 / 1M` uncached -> **`$0.15 / 1M` cached** (`95%` discount).
   - **`gemini-4-hydrogen` Input:** `$1.50 / 1M` uncached -> **`$0.075 / 1M` cached** (`95%` discount).
   - Place all immutable system instructions, tool schemas, and reference documents at the **very beginning** of the request, and append dynamic per-request variables at the tail.
3. **No Prefilled `model` Turns:** Multi-turn conversation histories cannot end with a `model` role turn, and prefilled `model` responses (e.g., forcing `{"` in an assistant turn) are rejected. Use `response_mime_type="application/json"` and `response_schema` instead.

---

### Trap #4 — Ghost Sampling Parameters, Unified `1,048,576` Window & `65,536` Sync Cap

#### 4.1 Sampling Parameter Enforcement
| Parameter | Behavior on `gemini-4-neon` |
| :--- | :--- |
| `temperature`, `top_p`, `top_k` (in-range) | **Silently ignored** (no effect on output distribution; reasoning is controlled via `thinking_level`) |
| `temperature`, `top_p`, `top_k` (out-of-range) | **`HTTP 400 Invalid Argument`** on Gemini API |
| `frequency_penalty`, `presence_penalty` | **`HTTP 400 Invalid Argument`** immediately |
| `candidate_count > 1` | **`HTTP 400 Invalid Argument`** immediately |

#### 4.2 Unified `1,048,576`-Token Context Window & Full-Pipeline Billing
Unlike legacy models with separate input and output caps, `gemini-4-neon` uses a **single unified `1,048,576`-token window**:

```text
Total Window (1,048,576) = Input Tokens + Thinking Tokens + Function Calls + Visible Output Tokens
```

- **Every processed token is billed**, including hidden `thinking` tokens (billed at the `$15.00 / 1M` output rate) and internal grounding system instructions / tool declarations injected when grounding is enabled (Google Search grounding is additionally billed per executed search query).
- **Synchronous / Foreground Output Cap (`65,536` tokens):** Both streaming and non-streaming synchronous requests cap output at `65,536` tokens per call.
- **Long Decode (`> 65,536` up to `1,048,576` tokens):** Requires passing `max_output_tokens > 65536` and iterating with `continuation_token` (requires API version **`v1beta1`**, not `v1`, on Gemini Enterprise Agent Platform) or executing asynchronously with **`background=True`** on the Interactions API.

---

### Trap #5 — The Silent Default Jump (`MEDIUM` -> `HIGH`) & Paid-Only Tier

When migrating from `gemini-4-carbon`, `gemini-4-argon`, or `gemini-3.8-flash`, omitting `thinking_level` silently changes your reasoning tier:

- `gemini-4-carbon`, `gemini-4-argon`, and `gemini-3.8-flash` default to **`thinking_level="MEDIUM"`** in the API.
- `gemini-4-neon` is **Paid Tier only (no Free Tier)** and defaults to **`thinking_level="HIGH"`** in the API.

Leaving `thinking_level` unset on `gemini-4-neon` nearly **doubles your cost per task (`$1.90` -> `$3.74`, `+96.8%`)** and **doubles Time-To-First-Token (`20.14 s` -> `40.64 s`)**.

---

## 3. Verified Token Economics & Artificial Analysis v5.0.1 Benchmark Matrix (As of November 19, 2026)

### 3.1 API Pricing per 1M Tokens (`ai.google.dev/gemini-api/docs/pricing`)

| Model | Free Tier | Uncached Input / 1M | Implicit Cached Input / 1M (`>= 4,096` tok) | Output + Thinking / 1M | Default API `thinking_level` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemini-4-neon`** | No *(Paid Tier only)* | `$3.00` | `$0.15` *(95% off)* | `$15.00` | **`"HIGH"`** |
| **`gemini-4-hydrogen`** | Yes | `$1.50` | `$0.075` *(95% off)* | `$7.50` | `"MEDIUM"` / `"HIGH"` *(surface-dependent)* |

### 3.2 Real-World Task Economics & Hallucination Telemetry (Artificial Analysis Index v5.0.1)

| Model & Effort Level | AA Intelligence Index (v5.0.1) | Avg Cost per Task | Suite Runtime | Time to First Reasoning Token (TTFT) | Output Tokens Generated | AA-Omniscience v1.1 Hallucination Rate *(Ungrounded)* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemini-4-neon` (`xhigh`)** | **61** | `$6.44` | `21.5 min` | `56.51 s` | `283M` | `14.4%` |
| **`gemini-4-neon` (`high` — API Default)** | **60** | `$3.74` | `14.8 min` | `40.64 s` | `123M` | `13.9%` |
| **`gemini-4-neon` (`medium` — Sweet Spot)** | **55** | **`$1.90`** | **`8.3 min`** | **`20.14 s`** | **`62.9M`** | `16.2%` |
| **`gemini-4-neon` (`low`)** | **44** | `$0.75` | `3.7 min` | `13.93 s` | — | `18.8%` |
| **`gemini-4-hydrogen` (`xhigh`)** | **54** | `$3.16` | `15.2 min` | `45.22 s` | — | `8.5%` |
| **`gemini-4-hydrogen` (`high`)** | **51** | `$1.87` | `8.6 min` | `20.08 s` | — | `8.7%` |
| **`gemini-4-hydrogen` (`medium`)** | **41** | `$0.80` | — | — | — | — |
| **`gemini-4-hydrogen` (`low`)** | **27** | `$0.29` | — | — | — | — |
| **`gemini-4-carbon` (`high`)** | **60** | `$7.76` | `36.6 min` | — | — | `11.4%` |
| **`gemini-4-argon` (`high`)** | **50** | `$4.97` | `29.5 min` | — | — | **`4.3%`** |

### 3.3 Production Routing Rules Derived from the Data

1. **Pin `gemini-4-neon` to `"MEDIUM"` for General Agentic & Coding Workloads:**
   - Cuts cost by **`49.1%`** (`$3.74` -> `$1.90`) and reduces TTFT by **`2.02x`** (`40.64 s` -> `20.14 s`) compared to the `"HIGH"` API default.
   - Matches `gemini-4-hydrogen` (`"HIGH"`, `$1.87`) in cost and latency while delivering **`+4` Intelligence Index points** (`55` vs. `51`).
   - Outperforms `gemini-4-hydrogen` (`"XHIGH"`, `$3.16`, `54` pts) at **`39.7%` lower cost** and **`1.83x` faster runtime** (`8.3 min` vs. `15.2 min`).
2. **Keep `gemini-4-hydrogen` on `"LOW"` / `"MEDIUM"` for Heavy-Input, Low-Reasoning Pipelines:**
   - When your cost is dominated by massive uncached input payloads rather than reasoning complexity, Hydrogen's `2x` lower base input rate (`$1.50/1M` vs. `$3.00/1M`), Free Tier availability, and `$0.29–$0.80` per-task footprint remain significantly cheaper.
3. **Never Run `gemini-4-neon` Ungrounded on Factual Recall Workloads:**
   - AA-Omniscience v1.1 measures **ungrounded factual recall without tools or documentation**. Because `gemini-4-neon` exhibits a `13.9%` (`high`) to `16.2%` (`medium`) ungrounded hallucination rate—compared to `8.7%` on Hydrogen (`high`) and `4.3%` on Argon (`high`)—always pair Neon with RAG context prefixes (`>= 4,096` tokens for `95%` implicit caching discount) or verified tool calls.
4. **Google Antigravity IDE Quota Conservation:**
   - In Google Antigravity (`512K` context window on Google AI Pro and Ultra; Enterprise includes Neon without Orchestrator), manually setting your active model effort to **Medium** on routine editing tasks prevents unnecessary quota burn and halves response wait times, reserving **High** or **Orchestrator (Neon)** (where Neon plans/reviews and Hydrogen executes) for multi-file architectural refactors.

---

## 4. Official Sources (Verified November 19, 2026)

- [Google Cloud Docs — Migrate from Gemini 3 models (`get-started-with-gemini-4`)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/get-started-with-gemini-4)
- [Google Cloud Docs — Gemini 4 Neon Developer Guide (`guides/gemini-4-neon`)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-4-neon)
- [Google AI for Developers — What's new in Gemini 4 Neon](https://ai.google.dev/gemini-api/docs/whats-new-gemini-4-neon)
- [Google AI for Developers — Gemini Developer API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Artificial Analysis — Gemini 4 Neon Intelligence Index v5.0.1 & Token Economics](https://artificialanalysis.ai/models/gemini-4-neon)
- [Artificial Analysis — Gemini 4 Hydrogen Intelligence Index v5.0.1](https://artificialanalysis.ai/models/gemini-4-hydrogen)
- [Google Antigravity Docs — Supported Models & Context Windows](https://antigravity.google/docs/models)
