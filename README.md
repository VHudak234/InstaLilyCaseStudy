# PartSelect Agent — Instalily Case Study

A chat agent for PartSelect's e-commerce site, scoped to **refrigerator and
dishwasher parts only**. The agent helps customers find parts, check
compatibility with their appliance, troubleshoot symptoms, and get install
guidance — and politely declines anything outside that scope.

## Stack

- **Frontend:** React (Create React App) — provided template, wired to backend.
- **Backend:** Python 3.12 + FastAPI.
- **LLM:** Gemini 2.5 Flash via `google-genai` with function calling.
- **Scraping:** Playwright (headed Chromium) — needed to bypass PartSelect's
  Akamai Bot Manager. See [Challenges](#challenges) below.
- **Vector store:** ChromaDB (on-disk, persistent) with Gemini
  `gemini-embedding-001` (3072-dim, cosine). Rebuilt lazily when the
  catalogue changes.

## Project structure

```
.
├── backend/
│   ├── main.py              # FastAPI app, /chat endpoint, CORS
│   ├── agent.py             # Agent loop + retry/backoff + partial-reply salvage
│   ├── tools.py             # Eleven tools: search, details, compat, install, troubleshoot,
│   │                        #   cart (add/view/remove/update), checkout, remember_appliance
│   ├── session_state.py     # In-memory cart + appliance profile (single-user demo state)
│   ├── vector_store.py      # Chroma + Gemini embeddings, fuzzy PN matching, lazy rebuild
│   ├── prompts.py           # System prompt: scope guardrail + HITL + diagnostics + memory
│   ├── scraper.py           # Playwright scraper w/ --inspect + --build modes
│   ├── tests/               # pytest suite — tools, salvage logic, cart lifecycle (49 tests)
│   ├── requirements.txt
│   └── .env.example
├── case-study-main/         # React frontend (provided template, wired up)
│   └── src/
│       ├── PartSelect-Logo.png
│       ├── api/api.js       # fetch POST -> http://localhost:8000/chat
│       └── components/ChatWindow.js
├── data/
│   └── parts_catalogue.json # Scraped catalogue, built by scraper.py
└── README.md                # this file
```

## How to run

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium          # one-time, ~150MB
cp .env.example .env                 # then edit .env to add GOOGLE_API_KEY
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd case-study-main
npm install
npm start                            # opens http://localhost:3000
```

### Build the parts catalogue (one-time)

```bash
cd backend
source venv/bin/activate
python scraper.py --build --limit 15 # scrapes ~30 parts total
```

### Run the tests

```bash
cd backend
source venv/bin/activate
pytest tests/ -q                     # 49 tests, ~2s
```

## Architecture

### Agent loop

```
user message
    ↓
build history + system prompt + tool schemas
    ↓
Gemini function calling
    ↓
┌─── text response?  → return to frontend
│
└─── tool call?      → execute tool → feed result back to Gemini → loop
                       (capped at MAX_ITERS=8 to prevent runaway)
```

- **Error-as-result:** tools return `{"error": "..."}` dicts instead of raising.
  The LLM sees the error and can recover (retry with different args, ask the
  user for clarification, apologise) rather than crashing the request.
- **Stateless backend:** conversation history is re-sent by the frontend each
  turn. Keeps scaling trivial — no session storage to shard or evict.
- **Trace log:** each tool call is recorded in a `trace` array returned with
  every response. The frontend renders these as "reasoning pills" under each
  assistant message (*"✓ Diagnosing symptom · rack rollers falling off"*)
  — transparent agency without leaking implementation details.
- **Retry + salvage for transient upstream failures.** Every Gemini call is
  wrapped in exponential backoff (5 attempts, 1s → 8s). When the *final*
  generate call 503s but earlier tool calls have already succeeded, we
  synthesise a short reply from the trace instead of showing a blanket
  "overloaded" message — otherwise the user sees product cards contradicting
  the text bubble above them. See `agent.py::_partial_reply_from_trace`.
- **Typo-tolerant part lookups (lexical + semantic).** If `get_part_details`
  misses, it returns a `suggestions` field populated by `difflib`-based
  edit-distance matching against every known part number — so a one-character
  typo like `PS1175277` for `PS11752778` reaches the right part. If the
  fuzzy match is empty, the agent falls back to semantic search using the
  mistyped number as the query.
- **Multi-step diagnostic dialogue.** For vague symptoms ("my fridge isn't
  working"), the system prompt requires the agent to ask 1–2 clarifying
  questions *before* calling `troubleshoot`. The tool accepts an optional
  `observations: list[str]` so the user's answers narrow the embedding query
  on the next call. Result quality: `troubleshoot` also applies a
  relative-gap + absolute-distance filter on retrieved candidates, so the
  agent isn't tempted to list distantly-related parts (e.g. door bins for an
  ice-maker query).
- **Appliance memory.** The first time a user mentions their appliance, the
  agent calls `remember_appliance(type, brand, model)` and confirms back to
  them. State lives in `session_state.py`; on subsequent turns the agent
  loop appends the appliance to the system instruction
  (`_build_system_instruction`) so the model treats it as a known fact
  without re-querying. Visible in the trace as a "Saving appliance" pill.
- **List-selection awareness.** Prompt rules teach the agent to resolve
  ordinal/positional references ("the second one", "the Whirlpool one") to
  the most recent list it offered, and to acknowledge multi-fact user
  messages ("I want the second part and my brand is Whirlpool") by
  addressing each piece — never silently dropping the second.

### Scope guardrail (defence in depth)

The assignment requires the agent to only discuss **refrigerator and
dishwasher parts**, politely refusing anything else. That scope is enforced
in four places — not just the prompt — so a jailbreak attempt can't quietly
route the model into, say, microwave parts or appliance repair chat.

1. **System prompt** (`backend/prompts.py`) tells the model to politely refuse
   off-topic queries and redirect to what it can help with.
2. **Tool surface** is physically limited — there's no general web search, no
   off-topic tool. The model's action space is bounded by what we expose.
3. **Tool argument validation** in `search_parts` and `troubleshoot` hard-fails
   on any `category` / `appliance_type` other than `"refrigerator"` or
   `"dishwasher"` — the error goes back to the model as a tool result, which
   the system prompt instructs it to surface as a warm refusal.
4. **Tool schemas** in `agent.py` advertise the allowed values in their
   `description` fields, so the model rarely tries invalid ones in the first
   place.

#### Widening the scope (if you wanted to)

If PartSelect decided the agent should cover more of its catalogue, the
change is small and concentrated:

- **`backend/prompts.py`** — remove the "refrigerators and dishwashers only"
  clause from the scope rules, or replace it with the broader allowed list.
- **`backend/tools.py`** — delete the category allow-list guards in
  `search_parts` and `troubleshoot` (the `if category not in (...)` branch).
  The catalogue is category-tagged, so the semantic search and filtering keep
  working untouched.
- **`backend/agent.py`** — update the `category` and `appliance_type` argument
  descriptions in the tool schemas to reflect the new allowed values (or
  drop the enumeration entirely and let the catalogue decide).
- **`backend/scraper.py`** — the scraper's `--build` list of seed URLs is
  currently just the two categories; extend it to cover whichever product
  lines you want in the index.

Everything else — agent loop, vector store, HITL flow, frontend, trace — is
category-agnostic and doesn't need to change.

### Tools

| Tool | Purpose |
|---|---|
| `search_parts(query, category)` | Semantic search over the catalogue (Chroma + Gemini embeddings). |
| `get_part_details(part_number)` | Exact lookup by PartSelect part number; returns fuzzy `suggestions` on a miss. |
| `check_compatibility(part_number, model_number)` | Heuristic brand/prefix match — honest about uncertainty. |
| `get_installation_guide(part_number)` | Difficulty, time, and guidance for installing a part. |
| `troubleshoot(appliance_type, symptom, observations?)` | Symptom → likely parts. Accepts `observations` for narrowing after clarifying questions. |
| `remember_appliance(type, brand, model)` | Records the user's appliance so the agent doesn't re-ask. |
| `add_to_cart(part_number, quantity, confirmed)` | **Transactional** — two-step HITL confirmation. |
| `view_cart()` | Show current cart contents (items, quantities, total). |
| `remove_from_cart(part_number, confirmed)` | **Transactional** — two-step HITL confirmation. |
| `update_cart_quantity(part_number, new_quantity, confirmed)` | **Transactional** — two-step HITL; rejects ≤0. |
| `initiate_checkout(confirmed)` | **Transactional** — two-step HITL; returns checkout URL and clears local cart. |

Each tool is **single-responsibility**. The system prompt steers the model to
chain them (e.g. `troubleshoot` → `get_part_details`, `add_to_cart` →
`view_cart` → `initiate_checkout`).

### Transactional surface (cart, checkout, HITL)

Four tools mutate user state — `add_to_cart`, `remove_from_cart`,
`update_cart_quantity`, `initiate_checkout`. Every one of them follows the
same **two-step human-in-the-loop pattern**:

1. **First call** (`confirmed=False`, the default): the tool returns a
   `pending_confirmation` payload with a human-readable summary of what
   would happen. The agent relays that summary in plain language and asks
   the user to confirm.
2. **Stop and wait.** The system prompt explicitly forbids calling a
   transactional tool with `confirmed=true` on the same turn as the user's
   original request. The user must have a chance to review and approve.
3. **Second call** (`confirmed=True`): only after the user explicitly
   agrees ("yes", "go ahead", "confirm") does the agent re-call the tool
   to actually mutate state.

**Cart is real, scoped to the demo.** `backend/session_state.py` holds a
single in-memory cart keyed by part number. It increments quantity if the
same part is added twice, and is wiped on server restart. There's no
`session_id` because the demo assumes a single user; production would key
the same dict by user/session and persist it. The tool surface doesn't
change — only the storage layer.

**Checkout is a deliberate handoff, not a stub.** `initiate_checkout` ends
the cart flow at the natural boundary: a real agent shouldn't collect
shipping or payment in chat. On confirmation the tool returns a checkout
URL (currently `partselect.com/cart.aspx`) and clears the local cart. A
production integration would replace the URL with a session-bound token
returned by PartSelect's checkout API; everything around it (HITL summary,
confirmation, cart-clearing) stays the same.

#### What the agent deliberately does NOT do (and why)

- **Order tracking, shipping status, returns, account history.** None of
  these are mocked. The system prompt instructs the agent to redirect
  users to their PartSelect account (`partselect.com/user/login.aspx`) and
  pivot to what it *can* do. Reasoning: a fake `track_order` tool with a
  hardcoded order #12345 would obscure what's actually working. By
  drawing the line at the API boundary, the agent's capabilities are
  unambiguous, and the integration story is honest: with PartSelect API
  access, you'd add `track_order` and `cancel_order` tools using the same
  HITL pattern as `remove_from_cart`.

### Frontend

**Visual design.** White-dominant palette using the two PartSelect brand
colors — teal `#457577` for primary identity (user bubble, send button,
links, prices) and gold `#EBC261` as a sparing accent (trace-pill check,
part-number chip, thinking-bubble shimmer). Typography is **IBM Plex Sans**
loaded via Google Fonts. The header carries the PartSelect logo + a teal
"Assistant" title separated by a hairline divider.

**Animated thinking state.** While the agent is working, the assistant
bubble shows:
- A small gold dot pulsing on a 1.6s loop (scale + ring fade).
- The text "Thinking…" rendered with `background-clip: text` over a moving
  teal → gold → teal gradient (shimmer, 2.2s) plus an opacity breathe
  (0.72 → 1 → 0.72, 3s).
- All animations respect `prefers-reduced-motion`.

**Product cards** render below the assistant's text bubble whenever a tool
returns part records — pulled out of the `trace` by walking for
`results` / `likely_parts` / `suggestions` / `items` / single-part
responses, deduped by part number, capped at 6.

**Click-to-select.** The card body is a button: clicking it sends
*"Tell me more about PS<num>"* into the chat — the chat-native equivalent
of "select this from the list," removing the burden of retyping a part
number. A small gold-on-soft-gold `↗` arrow in the corner is a separate
external link to the PartSelect product page, so navigating away never
happens accidentally. Disabled state during loading; keyboard-focusable.

**Reasoning pills** show what the agent did (`✓ Looking up part ·
PS11752778`). Tool identifiers are mapped to human labels in
`TOOL_LABELS` so end users never see implementation names like
`get_part_details`.

**User bubble width** is content-hugging up to a `min(60ch, 85%)` cap, so
short messages stay on one line and long ones wrap naturally around the
60-character mark. User input is rendered as plain text (no markdown), so
trailing newlines from `marked()` can't force a forced second line.

**Graceful network/upstream error handling** in `api/api.js` — distinguishes
a dead backend ("make sure the server is running") from an upstream 503
that the backend already turned into a warm message.

## Challenges

### 1. PartSelect is protected by Akamai Bot Manager

**The problem.** PartSelect blocks scrapers aggressively. Our attempts failed
in this order:

| Approach | Result |
|---|---|
| `httpx` with a realistic browser User-Agent | 403 (TLS fingerprint check) |
| `curl_cffi` impersonating Chrome's TLS fingerprint | 403 (deeper checks) |
| `cloudscraper` (solves Cloudflare JS challenge) | 403 (turned out to be Akamai, not Cloudflare) |
| Playwright **headless** Chromium | "Access Denied" (318-byte stub page — Akamai detects `navigator.webdriver`) |
| Playwright **headed** Chromium with `navigator.webdriver` override | ✅ works |

**The fix.** Headed (visible) Playwright with a small init script that deletes
the `navigator.webdriver` tell. Real Chromium sailing past Akamai's checks.

**Tradeoff.** Slower (~3–5s per page vs ~300ms for raw HTTP). Totally fine for
a one-off catalogue build; we throttle at 1s between requests to be polite.

### 2. Gemini model availability

Initial `gemini-2.0-flash` returned 404 "not available to new users" — Google
has been tightening model access. Switched to `gemini-2.5-flash`, which is
current, fast, and cheap on the pay-as-you-go tier. Similarly,
`text-embedding-004` has been retired from the `v1beta` API, so the vector
store uses `gemini-embedding-001` (3072-dim).

### 2a. Gemini 503 spikes under demand

Even on the stable models, `gemini-2.5-flash` returns 503 UNAVAILABLE fairly
often during busy periods. A single user turn can make several generate
calls (one per tool-use round-trip), so each call has to survive on its own.
The fix is two-layered:

1. Per-call exponential backoff (5 attempts, 1s → 8s) inside the agent loop.
2. When retries are exhausted mid-turn but tool calls have already succeeded,
   `_partial_reply_from_trace` builds a short reply from the trace instead
   of the generic "overloaded" message — otherwise the user sees product
   cards below a bubble that claims nothing worked.

### 3. Incomplete compatibility data

PartSelect's *Model Cross Reference* section uses infinite scroll (lazy-loads
via JS). Our scraper only captures the first batch of rows, so the
`compatible_brands` field is shallow (typically 1–2 brands per part). The
richer signal is the part's own `brand` (manufacturer) field, which is
complete.

## Scalability notes

Current design targets a demo-scale catalogue (~30 parts). The shape scales
cleanly to production because the layers are well-separated:

- **Catalogue size.** ChromaDB uses HNSW internally; sub-100ms search up to
  ~10k parts on a single node. Beyond that, a managed vector DB
  (Pinecone, Weaviate) would swap in via `vector_store.py` without changing
  the tool interface.
- **Fresh data.** The scraper is write-ahead; rerunning it overwrites the
  catalogue. Hook it to a cron job (hourly/daily) for near-live data.
- **Throughput.** FastAPI is stateless, so horizontal scaling is trivial —
  put it behind a load balancer, scale replicas based on QPS.
- **Latency.** Agent turns are ~3–5s today. The obvious next step is streaming
  responses (Server-Sent Events) so the user sees tokens as they're produced.

## Known limitations

- `compatible_brands` is shallow (see Challenges §3); compatibility check
  is a brand-prefix heuristic, and the agent tells the user to verify on
  PartSelect before purchasing.
- Cart and appliance state are single-user (module-level globals in
  `session_state.py`) and reset on server restart. Production: key by
  session/user and persist.
- Order tracking, shipping status, and account-bound queries are
  deliberately not implemented — see [Transactional surface](#transactional-surface-cart-checkout-hitl).
  With PartSelect API access, the same HITL pattern would apply.
- No streaming responses yet; turns take ~3–5s perceived. Server-Sent
  Events on top of the existing FastAPI endpoint is the planned next step.
- At ~20 parts the embedding distances cluster tightly, so the relevance
  filter in `troubleshoot` mostly relies on the prompt's "lead with the
  most plausible candidate" rule. The threshold structure is in place and
  will start mattering at larger catalogue sizes.
