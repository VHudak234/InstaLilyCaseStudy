"""In-memory session state — appliance profile + shopping cart.

Demo simplification: this module holds a SINGLE global state. There's no
session_id, no user identity. A real deployment would key these by a
session/user identifier and persist to a database — the tool surface here
would not change, only the storage layer.

Kept deliberately tiny so it's obvious what's stubbed and what isn't.
"""

from __future__ import annotations

from typing import Any

# Appliance the user is asking about. Cleared explicitly via clear_appliance().
_appliance: dict[str, Any] | None = None

# part_number -> {"part": <full part dict>, "quantity": int}
_cart: dict[str, dict[str, Any]] = {}


# --- Appliance profile ----------------------------------------------------

def set_appliance(brand: str, model_number: str, appliance_type: str) -> dict[str, Any]:
    global _appliance
    _appliance = {
        "brand": brand,
        "model_number": model_number,
        "appliance_type": appliance_type,
    }
    return dict(_appliance)


def get_appliance() -> dict[str, Any] | None:
    return dict(_appliance) if _appliance else None


def clear_appliance() -> None:
    global _appliance
    _appliance = None


# --- Cart ------------------------------------------------------------------

def cart_add(part: dict[str, Any], quantity: int) -> dict[str, Any]:
    """Add (or increment) a cart line. Returns the updated line."""
    pn = part["part_number"]
    existing = _cart.get(pn)
    if existing:
        existing["quantity"] += quantity
        return existing
    _cart[pn] = {"part": part, "quantity": quantity}
    return _cart[pn]


def cart_set_quantity(part_number: str, quantity: int) -> dict[str, Any] | None:
    """Set the quantity for a cart line. Returns the line, or None if absent."""
    line = _cart.get(part_number)
    if line is None:
        return None
    line["quantity"] = quantity
    return line


def cart_remove(part_number: str) -> dict[str, Any] | None:
    """Remove a cart line. Returns the removed line, or None if absent."""
    return _cart.pop(part_number, None)


def cart_lines() -> list[dict[str, Any]]:
    """Return cart lines in insertion order."""
    return list(_cart.values())


def cart_total() -> float:
    return round(
        sum((line["part"].get("price") or 0.0) * line["quantity"] for line in _cart.values()),
        2,
    )


def cart_clear() -> None:
    _cart.clear()
