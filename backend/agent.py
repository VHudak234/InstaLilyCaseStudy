"""Agent loop: Gemini function calling with bounded iterations and error-as-result.

Loop shape:
    user msg -> LLM -> (tool calls? -> run -> feed back -> LLM) -> text reply
A hard MAX_ITERS prevents runaway loops; if we hit the cap without a final text
reply we return a graceful fallback rather than letting the user wait forever.
"""

import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import session_state
from prompts import SYSTEM_PROMPT
from tools import TOOLS

load_dotenv()

MODEL = "gemini-2.5-flash"
MAX_ITERS = 8

# Gemini returns 503 UNAVAILABLE fairly often under load. These are transient,
# so we retry with exponential backoff before giving up. A single user turn may
# make several generate calls (one per tool-use round-trip), so each call has
# to be robust on its own.
_RETRY_STATUS = {429, 503}
_MAX_RETRIES = 5
_BACKOFF_BASE = 1.0  # sleeps: 1s, 2s, 4s, 8s (capped at _BACKOFF_CAP)
_BACKOFF_CAP = 8.0

_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def _build_system_instruction() -> str:
    """Compose the system prompt with any per-turn dynamic context.

    Right now that's just the remembered appliance — appended so the model
    treats it as a fact about the user, without us having to retrieve it via
    a tool every turn.
    """
    appliance = session_state.get_appliance()
    if not appliance:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT
        + "\n\nUser appliance on file: "
        + f"{appliance['brand']} {appliance['appliance_type']}, model "
        + f"{appliance['model_number']}. Use this when relevant; don't ask for it again "
        + "unless the user signals they're talking about a different appliance."
    )


def _generate_with_retry(contents: list[types.Content]) -> Any:
    """Call Gemini with retry on transient 429/503 errors."""
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return _client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_build_system_instruction(),
                    tools=_TOOL_DECLARATIONS,
                ),
            )
        except genai_errors.APIError as e:
            last_err = e
            if getattr(e, "code", None) not in _RETRY_STATUS:
                raise
            if attempt == _MAX_RETRIES - 1:
                break
            time.sleep(min(_BACKOFF_BASE * (2**attempt), _BACKOFF_CAP))
    assert last_err is not None
    raise last_err


# Tool schemas exposed to Gemini. Declared as plain dicts (the SDK accepts these
# directly) so reviewers can see exactly what the agent's action space is.
_TOOL_DECLARATIONS = [
    {
        "function_declarations": [
            {
                "name": "search_parts",
                "description": (
                    "Semantic search over the PartSelect parts catalogue. "
                    "Use when the user is looking for a part but doesn't know the part number."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Natural-language description of what the user needs."},
                        "category": {"type": "STRING", "description": "Must be 'refrigerator' or 'dishwasher'."},
                    },
                    "required": ["query", "category"],
                },
            },
            {
                "name": "get_part_details",
                "description": (
                    "Return full details for a specific part (description, price, install info, image). "
                    "Use when the user gives a PartSelect part number like 'PS11752778'."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "part_number": {"type": "STRING", "description": "PartSelect part number, e.g. 'PS11752778'."},
                    },
                    "required": ["part_number"],
                },
            },
            {
                "name": "check_compatibility",
                "description": (
                    "Check whether a part is likely to fit a user's appliance model number. "
                    "Use when the user provides both a part number and a model number."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "part_number": {"type": "STRING", "description": "PartSelect part number."},
                        "model_number": {"type": "STRING", "description": "Appliance model number, e.g. 'WDT780SAEM1'."},
                    },
                    "required": ["part_number", "model_number"],
                },
            },
            {
                "name": "get_installation_guide",
                "description": (
                    "Return installation difficulty, estimated time, and guidance for a part. "
                    "Use after the user has identified a specific part and wants to know how to install it."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "part_number": {"type": "STRING", "description": "PartSelect part number."},
                    },
                    "required": ["part_number"],
                },
            },
            {
                "name": "add_to_cart",
                "description": (
                    "Add a part to the user's cart. This is a TRANSACTIONAL tool. "
                    "You must call it twice: first with confirmed=false to produce a confirmation "
                    "summary that you present to the user, then again with confirmed=true only "
                    "after the user has explicitly agreed in their next message."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "part_number": {"type": "STRING", "description": "PartSelect part number."},
                        "quantity": {"type": "INTEGER", "description": "How many to add. Defaults to 1."},
                        "confirmed": {"type": "BOOLEAN", "description": "True only after the user has explicitly confirmed."},
                    },
                    "required": ["part_number"],
                },
            },
            {
                "name": "troubleshoot",
                "description": (
                    "Given an appliance type and a symptom, suggest the parts most likely to "
                    "be the cause. Use when the user describes a problem rather than asking "
                    "for a specific part. For VAGUE symptoms ('it's not working', 'something's "
                    "wrong with my fridge'), ASK the user 1-2 clarifying questions FIRST "
                    "(e.g. 'Is the dispenser making any sound?', 'Is water reaching it?'), "
                    "then call this tool with the answers passed in `observations`. Repeat "
                    "calls are fine as the picture sharpens."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "appliance_type": {"type": "STRING", "description": "Must be 'refrigerator' or 'dishwasher'."},
                        "symptom": {"type": "STRING", "description": "What's going wrong, e.g. 'ice maker not producing ice'."},
                        "observations": {
                            "type": "ARRAY",
                            "description": "Optional: clarifying details the user shared, e.g. ['dispenser has power', 'no water'].",
                            "items": {"type": "STRING"},
                        },
                    },
                    "required": ["appliance_type", "symptom"],
                },
            },
            {
                "name": "view_cart",
                "description": (
                    "Show the user's current cart contents (items, quantities, total). "
                    "Use when the user asks 'what's in my cart?', requests a summary, or "
                    "you've just changed it and want to confirm."
                ),
                "parameters": {"type": "OBJECT", "properties": {}},
            },
            {
                "name": "remove_from_cart",
                "description": (
                    "Remove a part from the cart. Like add_to_cart this is TRANSACTIONAL: "
                    "first call with confirmed=false to produce a confirmation summary, then "
                    "call again with confirmed=true only after the user explicitly agrees."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "part_number": {"type": "STRING", "description": "PartSelect part number already in the cart."},
                        "confirmed": {"type": "BOOLEAN", "description": "True only after the user has explicitly confirmed."},
                    },
                    "required": ["part_number"],
                },
            },
            {
                "name": "update_cart_quantity",
                "description": (
                    "Set the quantity of an existing cart line (must be >= 1; use "
                    "remove_from_cart to take a line out entirely). TRANSACTIONAL: same "
                    "two-step confirmation pattern as add_to_cart."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "part_number": {"type": "STRING", "description": "PartSelect part number already in the cart."},
                        "new_quantity": {"type": "INTEGER", "description": "Desired quantity, >= 1."},
                        "confirmed": {"type": "BOOLEAN", "description": "True only after the user has explicitly confirmed."},
                    },
                    "required": ["part_number", "new_quantity"],
                },
            },
            {
                "name": "initiate_checkout",
                "description": (
                    "Hand the user off to PartSelect's checkout to complete their order. "
                    "Use when the cart has items and the user wants to buy / check out / "
                    "place the order. TRANSACTIONAL: same two-step confirmation pattern "
                    "as add_to_cart. On success the response includes a checkout_url to "
                    "share with the user."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "confirmed": {"type": "BOOLEAN", "description": "True only after the user has explicitly confirmed."},
                    },
                },
            },
            {
                "name": "remember_appliance",
                "description": (
                    "Record the user's appliance (type, brand, model number) so subsequent "
                    "queries are anchored to it. Call this once the user has shared and "
                    "confirmed those details, so they don't have to repeat themselves."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "appliance_type": {"type": "STRING", "description": "Must be 'refrigerator' or 'dishwasher'."},
                        "brand": {"type": "STRING", "description": "e.g. 'Whirlpool', 'Bosch'."},
                        "model_number": {"type": "STRING", "description": "e.g. 'WDT780SAEM1'."},
                    },
                    "required": ["appliance_type", "brand", "model_number"],
                },
            },
        ],
    },
]


def _partial_reply_from_trace(trace: list[dict[str, Any]]) -> str | None:
    """Synthesise a minimal reply when the final generate call 503s mid-turn.

    The tool calls already ran and their results are in `trace`, so instead of
    telling the user "the model is overloaded" while we silently render product
    cards below, we acknowledge what we found. Returns None if there's nothing
    useful to salvage, in which case the caller falls back to a plain apology.
    """
    if not trace:
        return None

    # Walk the trace newest-first for the first tool that produced visible content.
    for step in reversed(trace):
        result = step.get("result") or {}
        # A get_part_details miss with fuzzy suggestions is still useful — the
        # error is structural (no exact match) but the suggestions carry signal.
        if result.get("error") and not result.get("suggestions"):
            continue

        # Tools that return lists of parts.
        parts = (
            result.get("results")
            or result.get("likely_parts")
            or result.get("suggestions")
        )
        if parts:
            names = ", ".join(p.get("name", p.get("part_number", "?")) for p in parts[:3])
            return (
                f"I found some likely matches ({names}) — see the cards below. "
                "The model had trouble generating a full written summary just now; "
                "please resend or rephrase if you'd like more detail."
            )

        # get_part_details / get_installation_guide single-part response.
        if result.get("part_number") and result.get("name"):
            return (
                f"I pulled up {result['name']} ({result['part_number']}) — details "
                "are shown below. The model had trouble generating a full written "
                "summary just now; please resend if you'd like more detail."
            )

        # add_to_cart pending confirmation — don't silently accept it.
        if result.get("pending_confirmation"):
            return result.get("prompt_user") or (
                "I need you to confirm that action before I proceed — please resend."
            )

    return None


def _to_contents(messages: list[dict[str, str]]) -> list[types.Content]:
    """Convert frontend message history to Gemini Content objects.

    Frontend uses {role: 'user'|'assistant', content: str}. Gemini uses 'model'
    instead of 'assistant'.
    """
    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
    return contents


def run_agent(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Run the agent loop and return {reply: str, trace: [...]}.

    `trace` records each tool call so the frontend can show reasoning steps.
    """
    contents = _to_contents(messages)
    trace: list[dict[str, Any]] = []

    for _ in range(MAX_ITERS):
        try:
            response = _generate_with_retry(contents)
        except genai_errors.APIError as e:
            # After retries, surface a warm, specific message instead of a 500.
            # If tool calls already succeeded this turn, prefer a salvage reply
            # derived from the trace — otherwise the user sees product cards
            # contradicting an "overloaded" message.
            code = getattr(e, "code", None)
            salvage = _partial_reply_from_trace(trace)
            if salvage:
                reply = salvage
            elif code == 503:
                reply = (
                    "Gemini is overloaded right now (the model is seeing a "
                    "spike in traffic). This is usually brief — please try "
                    "sending your message again in a few seconds."
                )
            elif code == 429:
                reply = (
                    "I've hit the API rate limit for the moment. Please wait "
                    "a few seconds and try again."
                )
            else:
                reply = (
                    "I ran into a problem talking to the language model. "
                    "Please try again — if it keeps happening, let me know."
                )
            return {"reply": reply, "trace": trace, "error": {"code": code, "kind": "upstream"}}

        candidate = response.candidates[0]
        # Gemini sometimes returns a candidate with no content (e.g. safety stop,
        # or a quirky MAX_TOKENS finish). Treat that as "no further action" and
        # return any text we have.
        if candidate.content is None:
            return {"reply": response.text or "", "trace": trace}
        parts = candidate.content.parts or []
        function_calls = [p.function_call for p in parts if p.function_call]

        # No tool calls -> the model produced a final text reply.
        if not function_calls:
            return {"reply": response.text or "", "trace": trace}

        # Append the model's tool-call turn to the conversation.
        contents.append(candidate.content)

        # Execute each tool call and feed the results back as a single user turn.
        result_parts = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            tool = TOOLS.get(fc.name)
            if tool is None:
                result = {"error": f"unknown tool: {fc.name}"}
            else:
                # Tools return error dicts rather than raising; this guard is a
                # belt-and-braces catch for anything unexpected.
                try:
                    result = tool(**args)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
            trace.append({"tool": fc.name, "args": args, "result": result})
            result_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": result})
            )
        contents.append(types.Content(role="user", parts=result_parts))

    # Hit the iteration cap without a final reply.
    return {
        "reply": "Sorry — I couldn't complete that request. Could you rephrase or give me more detail?",
        "trace": trace,
    }
