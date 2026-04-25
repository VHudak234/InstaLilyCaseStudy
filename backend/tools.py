"""Tool implementations exposed to the agent.

Each tool is single-responsibility. Docstrings + type hints are what Gemini
sees as the tool description. Errors are returned as dicts (never raised) so
the agent can reason about failures and try alternatives rather than crash.
"""

from typing import Any

import session_state
import vector_store


def _summarise(part: dict[str, Any]) -> dict[str, Any]:
    """Trim a catalogue entry to the fields useful to the LLM + frontend.

    Keeps payloads small (cheaper tokens, faster responses) while preserving
    everything needed for a product card or downstream tool call.
    """
    return {
        "part_number": part["part_number"],
        "name": part["name"],
        "brand": part.get("brand"),
        "category": part["category"],
        "price": part.get("price"),
        "summary": part.get("description", "")[:280],
        "image_url": part.get("image_url"),
        "url": part.get("url"),
    }


def search_parts(query: str, category: str) -> dict[str, Any]:
    """Semantic search over the parts catalogue.

    Args:
        query: Natural-language description of what the user needs
            (e.g. "door bin for a side-by-side fridge" or "ice maker broken").
        category: Either "refrigerator" or "dishwasher".
    """
    try:
        if category not in ("refrigerator", "dishwasher"):
            return {"error": f"category must be 'refrigerator' or 'dishwasher', got '{category}'"}
        hits = vector_store.search(query, category=category, k=5)
        return {"results": [_summarise(p) for p in hits], "query": query}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def get_part_details(part_number: str) -> dict[str, Any]:
    """Return full details (description, install info, image) for one part.

    Args:
        part_number: PartSelect part number, e.g. "PS11752778".
    """
    try:
        part = vector_store.get(part_number)
        if part is None:
            # Typo recovery: a mistyped PN ("PS1175277" for "PS11752778") should
            # still reach the right part. We return "did you mean" suggestions
            # so the agent can confirm with the user before acting on them.
            suggestions = vector_store.similar_part_numbers(part_number)
            return {
                "error": f"part {part_number!r} not found in catalogue",
                "suggestions": [_summarise(p) for p in suggestions],
            }
        # Full record — includes the long description and install fields that
        # the trimmed search_parts result drops.
        return {
            "part_number": part["part_number"],
            "mpn": part.get("mpn"),
            "name": part["name"],
            "brand": part.get("brand"),
            "category": part["category"],
            "price": part.get("price"),
            "description": part.get("description"),
            "install_difficulty": part.get("install_difficulty"),
            "install_time": part.get("install_time"),
            "image_url": part.get("image_url"),
            "url": part.get("url"),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def check_compatibility(part_number: str, model_number: str) -> dict[str, Any]:
    """Check whether a part is likely to fit a given appliance model number.

    Honest limitation: our scraped catalogue captures the part's manufacturer
    brand but not the full model cross-reference (PartSelect lazy-loads that
    via JS). So we do a best-effort check: if the model number's prefix looks
    like one of the brands the part is compatible with, we say "likely".
    The LLM is told in its response to tell the user to verify on-site.

    Args:
        part_number: e.g. "PS11752778".
        model_number: Appliance model, e.g. "WDT780SAEM1".
    """
    try:
        part = vector_store.get(part_number)
        if part is None:
            return {"error": f"part {part_number!r} not found"}

        # Heuristic: Whirlpool models often start with WDT/WRF/WRS, Frigidaire
        # with FFHT/FFSS, GE with GSS/GDF, Bosch with SHE/SHX. We don't try to
        # be exhaustive — just enough that the LLM can report a sensible answer.
        brand = (part.get("brand") or "").lower()
        compat = [b.lower() for b in part.get("compatible_brands") or []]
        compat.append(brand)

        model_upper = model_number.upper()
        brand_prefixes = {
            "whirlpool": ("W", "KUD", "KDT", "MDB"),
            "frigidaire": ("F", "GLD", "PLD"),
            "ge": ("G", "GDT", "GSS", "PDT"),
            "bosch": ("SHE", "SHX", "SHP", "SGE"),
            "kenmore": ("665", "110"),
            "maytag": ("MDB", "MDC"),
            "kitchenaid": ("KDT", "KUD", "KDF"),
        }

        likely = False
        matched_brand: str | None = None
        for b in compat:
            if b in brand_prefixes and model_upper.startswith(brand_prefixes[b]):
                likely = True
                matched_brand = b
                break

        return {
            "part_number": part_number,
            "model_number": model_number,
            "likely_compatible": likely,
            "matched_brand": matched_brand,
            "part_brand": part.get("brand"),
            "note": (
                "Heuristic match on brand/model-prefix. Always verify on the "
                "PartSelect product page before purchasing."
            ),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def get_installation_guide(part_number: str) -> dict[str, Any]:
    """Return installation guidance for a part: difficulty, time, and steps.

    Steps come from the part's description, which on PartSelect typically
    includes install guidance inline ("snap the old part out and snap the new
    part into place...").

    Args:
        part_number: e.g. "PS11752778".
    """
    try:
        part = vector_store.get(part_number)
        if part is None:
            return {"error": f"part {part_number!r} not found"}
        return {
            "part_number": part_number,
            "name": part.get("name"),
            "difficulty": part.get("install_difficulty"),
            "estimated_time": part.get("install_time"),
            "guidance": part.get("description"),
            "full_page_url": part.get("url"),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def troubleshoot(
    appliance_type: str,
    symptom: str,
    observations: list[str] | None = None,
) -> dict[str, Any]:
    """Suggest likely parts for a reported symptom.

    Uses the same semantic index as search_parts — the catalogue descriptions
    already discuss symptoms ("if the rack is sagging...") so vector search on
    the symptom surfaces the right parts without a separate symptom table.

    The `observations` list is for *progressive narrowing*: after the agent
    asks the user clarifying questions ("Is the ice maker getting water?",
    "Any sound from the dispenser?"), it passes the answers in here. They get
    concatenated into the embedding query so the search is steered by what
    the user actually reported, not just the headline symptom.

    Args:
        appliance_type: "refrigerator" or "dishwasher".
        symptom: Natural language description, e.g. "ice maker not working".
        observations: Optional list of clarifying details the user has shared,
            e.g. ["dispenser has power", "no water reaching the maker"].
    """
    try:
        if appliance_type not in ("refrigerator", "dishwasher"):
            return {"error": f"appliance_type must be 'refrigerator' or 'dishwasher'"}
        query = symptom
        if observations:
            query = symptom + " — " + "; ".join(observations)
        scored = vector_store.search_scored(query, category=appliance_type, k=3)

        # Drop weak matches so the agent isn't tempted to list distantly
        # related parts (e.g. door bins for an ice-maker query).
        #   1) Absolute floor: anything beyond MAX_DIST is too unrelated to mention.
        #   2) Relative gap: drop results meaningfully further from the top hit.
        # NOTE: at our current ~20-part catalogue, distances cluster tightly
        # (often within 0.01 of each other) so this filtering is most useful
        # at larger catalogue sizes. The prompt also tells the agent to lead
        # with the single most plausible candidate — that's what does the
        # presentation-level filtering today.
        MAX_DIST = 0.65
        RELATIVE_GAP = 0.05
        filtered = [(p, d) for (p, d) in scored if d <= MAX_DIST]
        if filtered:
            top_dist = filtered[0][1]
            filtered = [(p, d) for (p, d) in filtered if d <= top_dist + RELATIVE_GAP]

        return {
            "symptom": symptom,
            "observations": observations or [],
            "appliance_type": appliance_type,
            "likely_parts": [_summarise(p) for (p, _d) in filtered],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def add_to_cart(part_number: str, quantity: int = 1, confirmed: bool = False) -> dict[str, Any]:
    """Add a part to the user's cart — stubbed, with a mandatory HITL confirmation step.

    This is a transactional action, so we enforce two calls:
      1. First call (confirmed=False): returns a "pending_confirmation" payload
         summarising what would happen. The agent surfaces this to the user.
      2. Second call (confirmed=True): the user has explicitly agreed, so we
         "add" the item (stubbed) and return a success payload.

    No real cart exists — this is a demo for the HITL pattern. In production the
    second call would hit an order service and return the cart ID.

    Args:
        part_number: PartSelect part number, e.g. "PS11752778".
        quantity: How many to add. Defaults to 1.
        confirmed: Must be True on the second call, after the user has agreed.
    """
    try:
        part = vector_store.get(part_number)
        if part is None:
            return {"error": f"part {part_number!r} not found"}
        if quantity < 1:
            return {"error": "quantity must be at least 1"}

        price = part.get("price") or 0.0
        total = round(price * quantity, 2)

        if not confirmed:
            return {
                "pending_confirmation": True,
                "part_number": part_number,
                "name": part["name"],
                "quantity": quantity,
                "unit_price": price,
                "total": total,
                "prompt_user": (
                    f"Please confirm: add {quantity} × {part['name']} "
                    f"({part_number}) at ${price:.2f} each, total ${total:.2f}?"
                ),
            }

        line = session_state.cart_add(part, quantity)
        return {
            "success": True,
            "added": {
                "part_number": part_number,
                "name": part["name"],
                "quantity": quantity,
                "line_quantity": line["quantity"],
                "line_total": round((part.get("price") or 0.0) * line["quantity"], 2),
            },
            "cart_total": session_state.cart_total(),
            "cart_size": len(session_state.cart_lines()),
            "note": "Demo cart — state lives in memory and resets on server restart.",
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def view_cart() -> dict[str, Any]:
    """Show the current contents of the user's cart (items, quantities, total).

    Use when the user asks what's in their cart, asks for a summary before
    checkout, or you want to confirm a recent change.
    """
    try:
        lines = session_state.cart_lines()
        items = [
            {
                "part_number": line["part"]["part_number"],
                "name": line["part"]["name"],
                "brand": line["part"].get("brand"),
                "category": line["part"]["category"],
                "image_url": line["part"].get("image_url"),
                "url": line["part"].get("url"),
                "unit_price": line["part"].get("price") or 0.0,
                "quantity": line["quantity"],
                "line_total": round((line["part"].get("price") or 0.0) * line["quantity"], 2),
            }
            for line in lines
        ]
        return {
            "items": items,
            "item_count": len(items),
            "total": session_state.cart_total(),
            "empty": len(items) == 0,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def remove_from_cart(part_number: str, confirmed: bool = False) -> dict[str, Any]:
    """Remove a part from the user's cart. Two-step HITL like add_to_cart.

    Args:
        part_number: PartSelect part number to remove.
        confirmed: True only after the user has explicitly confirmed.
    """
    try:
        # Find the line first so the confirmation summary is informative.
        existing = next(
            (l for l in session_state.cart_lines() if l["part"]["part_number"] == part_number),
            None,
        )
        if existing is None:
            return {"error": f"part {part_number!r} is not in the cart"}

        if not confirmed:
            return {
                "pending_confirmation": True,
                "action": "remove",
                "part_number": part_number,
                "name": existing["part"]["name"],
                "quantity": existing["quantity"],
                "prompt_user": (
                    f"Please confirm: remove {existing['quantity']} × "
                    f"{existing['part']['name']} ({part_number}) from your cart?"
                ),
            }

        session_state.cart_remove(part_number)
        return {
            "success": True,
            "removed": {"part_number": part_number, "name": existing["part"]["name"]},
            "cart_total": session_state.cart_total(),
            "cart_size": len(session_state.cart_lines()),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def update_cart_quantity(
    part_number: str,
    new_quantity: int,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Set the quantity of an existing cart line. Two-step HITL.

    To remove a line, call `remove_from_cart` instead — this tool requires
    `new_quantity >= 1`.

    Args:
        part_number: PartSelect part number already in the cart.
        new_quantity: Desired quantity (must be >= 1).
        confirmed: True only after the user has explicitly confirmed.
    """
    try:
        if new_quantity < 1:
            return {
                "error": (
                    "new_quantity must be at least 1; use remove_from_cart "
                    "to take the line out entirely"
                )
            }
        line = next(
            (l for l in session_state.cart_lines() if l["part"]["part_number"] == part_number),
            None,
        )
        if line is None:
            return {"error": f"part {part_number!r} is not in the cart"}

        old_quantity = line["quantity"]
        if old_quantity == new_quantity:
            return {
                "success": True,
                "no_change": True,
                "part_number": part_number,
                "quantity": new_quantity,
            }

        if not confirmed:
            return {
                "pending_confirmation": True,
                "action": "update_quantity",
                "part_number": part_number,
                "name": line["part"]["name"],
                "old_quantity": old_quantity,
                "new_quantity": new_quantity,
                "prompt_user": (
                    f"Please confirm: change {line['part']['name']} ({part_number}) "
                    f"from {old_quantity} to {new_quantity}?"
                ),
            }

        session_state.cart_set_quantity(part_number, new_quantity)
        return {
            "success": True,
            "part_number": part_number,
            "old_quantity": old_quantity,
            "new_quantity": new_quantity,
            "cart_total": session_state.cart_total(),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def initiate_checkout(confirmed: bool = False) -> dict[str, Any]:
    """Hand the user off to PartSelect's checkout. Two-step HITL.

    First call (`confirmed=False`) returns a summary of what's about to happen
    so the user can review their cart before being redirected. Second call
    (`confirmed=True`) returns the checkout URL and clears the demo cart.

    Honest about what's real: this is a handoff, not a real cart transfer.
    A production integration would POST the cart to PartSelect's checkout API
    and redirect the user with a session token; here we just return their
    public checkout URL. Documented in the response `note` so the model
    surfaces the limitation if asked.

    Args:
        confirmed: True only after the user has explicitly confirmed.
    """
    try:
        lines = session_state.cart_lines()
        if not lines:
            return {"error": "cart is empty — add items before checking out"}

        items = [
            {
                "part_number": line["part"]["part_number"],
                "name": line["part"]["name"],
                "quantity": line["quantity"],
            }
            for line in lines
        ]
        total = session_state.cart_total()

        if not confirmed:
            return {
                "pending_confirmation": True,
                "action": "checkout",
                "items": items,
                "item_count": len(items),
                "total": total,
                "prompt_user": (
                    f"Ready to check out? Your cart has {len(items)} item(s) "
                    f"totalling ${total:.2f}. Confirming will hand you off to "
                    "PartSelect's checkout to enter shipping and payment."
                ),
            }

        checkout_url = "https://www.partselect.com/cart.aspx"
        session_state.cart_clear()
        return {
            "success": True,
            "checkout_url": checkout_url,
            "items": items,
            "total": total,
            "note": (
                "Demo handoff: a production integration would persist the cart "
                "to PartSelect via their API and redirect with a session token. "
                "The local demo cart has been cleared."
            ),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def remember_appliance(
    appliance_type: str,
    brand: str,
    model_number: str,
) -> dict[str, Any]:
    """Record the user's appliance so subsequent queries are anchored to it.

    Call this once the user mentions their appliance — typically after asking
    them to confirm. The agent loop sees the stored appliance via the
    conversation history; this tool's main value is making the context
    explicit and visible in the trace.

    Args:
        appliance_type: "refrigerator" or "dishwasher".
        brand: e.g. "Whirlpool".
        model_number: e.g. "WDT780SAEM1".
    """
    try:
        if appliance_type not in ("refrigerator", "dishwasher"):
            return {"error": "appliance_type must be 'refrigerator' or 'dishwasher'"}
        stored = session_state.set_appliance(brand, model_number, appliance_type)
        return {"remembered": stored}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# Registry the agent loop dispatches against.
TOOLS = {
    "search_parts": search_parts,
    "get_part_details": get_part_details,
    "check_compatibility": check_compatibility,
    "get_installation_guide": get_installation_guide,
    "troubleshoot": troubleshoot,
    "add_to_cart": add_to_cart,
    "view_cart": view_cart,
    "remove_from_cart": remove_from_cart,
    "update_cart_quantity": update_cart_quantity,
    "initiate_checkout": initiate_checkout,
    "remember_appliance": remember_appliance,
}
