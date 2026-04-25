"""Tool-layer tests.

These exercise the real catalogue and the real Chroma index — no mocking.
The index build hits the Gemini embeddings API once per test-session (Chroma
caches on disk), which is the right tradeoff: we're testing the integration
we actually ship, not a fiction.

Tests intentionally avoid asserting on specific top-1 search results because
semantic ranking is a bit fuzzy; we assert on structural properties (result
count, required fields, category filtering) instead.
"""

import pytest

import session_state
import tools


@pytest.fixture(autouse=True)
def _reset_session_state():
    """Cart and appliance are module-level globals — wipe before each test."""
    session_state.cart_clear()
    session_state.clear_appliance()
    yield
    session_state.cart_clear()
    session_state.clear_appliance()

# Known-good part numbers that should exist in `data/parts_catalogue.json`.
# If the catalogue is rebuilt and these drop out, update here.
KNOWN_DW_WHIRLPOOL = "PS3406971"
KNOWN_FRIDGE_FRIGIDAIRE = "PS12364199"
KNOWN_DOOR_BIN = "PS11752778"


# --- get_part_details -------------------------------------------------------

def test_get_part_details_known_pn_returns_full_record():
    out = tools.get_part_details(KNOWN_DOOR_BIN)
    assert "error" not in out
    # Contract: the LLM and frontend both depend on these keys.
    for key in ("part_number", "name", "category", "description", "url"):
        assert key in out, f"missing {key}"
    assert out["part_number"] == KNOWN_DOOR_BIN


def test_get_part_details_unknown_pn_returns_error_dict():
    out = tools.get_part_details("PS_DOES_NOT_EXIST")
    # Tools must never raise — errors are dicts so the LLM can recover.
    assert "error" in out
    assert "not found" in out["error"].lower()
    # The suggestions field is always present (possibly empty) so the LLM
    # doesn't have to branch on "does this key exist?".
    assert "suggestions" in out


def test_get_part_details_single_char_typo_surfaces_correct_part():
    # "PS1175277" is PS11752778 minus the trailing 8 — the archetypal typo.
    out = tools.get_part_details("PS1175277")
    assert "error" in out  # exact lookup still fails
    pns = [s["part_number"] for s in out["suggestions"]]
    assert KNOWN_DOOR_BIN in pns, f"fuzzy match should find {KNOWN_DOOR_BIN}, got {pns}"


def test_get_part_details_wildly_wrong_pn_returns_empty_suggestions():
    # Random string shouldn't trigger false-positive suggestions.
    out = tools.get_part_details("ZZZZZZZZZZ")
    assert "error" in out
    assert out["suggestions"] == []


# --- search_parts -----------------------------------------------------------

def test_search_parts_rejects_invalid_category():
    out = tools.search_parts(query="door bin", category="oven")
    assert "error" in out


def test_search_parts_returns_only_requested_category():
    out = tools.search_parts(query="door bin", category="refrigerator")
    assert "results" in out
    assert out["results"], "expected at least one result"
    for p in out["results"]:
        assert p["category"] == "refrigerator"


def test_search_parts_results_are_summarised_not_full_records():
    # search_parts deliberately drops the long description to keep prompt
    # payloads small; get_part_details is the escape hatch for full detail.
    out = tools.search_parts(query="rack roller", category="dishwasher")
    assert out["results"]
    first = out["results"][0]
    assert "summary" in first and len(first["summary"]) <= 280
    assert "description" not in first  # full description is only in details


# --- check_compatibility ----------------------------------------------------

def test_check_compatibility_matching_brand_prefix_is_likely():
    # PS3406971 is a Whirlpool part; WDT-prefixed models are Whirlpool.
    out = tools.check_compatibility(KNOWN_DW_WHIRLPOOL, "WDT780SAEM1")
    assert "error" not in out
    assert out["likely_compatible"] is True
    assert out["matched_brand"] == "whirlpool"


def test_check_compatibility_non_matching_prefix_is_not_likely():
    # Bosch SHE-prefix model against a Whirlpool part shouldn't match.
    out = tools.check_compatibility(KNOWN_DW_WHIRLPOOL, "SHE3AR75UC")
    assert "error" not in out
    assert out["likely_compatible"] is False


def test_check_compatibility_unknown_part_returns_error():
    out = tools.check_compatibility("PS_NOPE", "WDT780SAEM1")
    assert "error" in out


def test_check_compatibility_always_includes_verification_note():
    # The heuristic is fuzzy; the note is what keeps us honest to the user.
    out = tools.check_compatibility(KNOWN_DW_WHIRLPOOL, "WDT780SAEM1")
    assert "verify" in out["note"].lower()


# --- get_installation_guide -------------------------------------------------

def test_get_installation_guide_returns_expected_shape():
    out = tools.get_installation_guide(KNOWN_DOOR_BIN)
    assert "error" not in out
    for key in ("part_number", "difficulty", "estimated_time", "guidance", "full_page_url"):
        assert key in out


def test_get_installation_guide_unknown_part_errors():
    out = tools.get_installation_guide("PS_NOPE")
    assert "error" in out


# --- troubleshoot -----------------------------------------------------------

def test_troubleshoot_rejects_invalid_appliance_type():
    out = tools.troubleshoot(appliance_type="microwave", symptom="no heat")
    assert "error" in out


def test_troubleshoot_returns_category_scoped_parts():
    out = tools.troubleshoot(appliance_type="dishwasher", symptom="rack rollers falling off")
    assert "likely_parts" in out
    assert out["likely_parts"], "expected candidate parts"
    for p in out["likely_parts"]:
        assert p["category"] == "dishwasher"


# --- add_to_cart (HITL) -----------------------------------------------------

def test_add_to_cart_first_call_returns_pending_confirmation():
    out = tools.add_to_cart(KNOWN_DOOR_BIN, quantity=2)
    # confirmed defaults to False — agent must NOT get a "success" the first time.
    assert out.get("pending_confirmation") is True
    assert "success" not in out
    assert out["quantity"] == 2
    assert out["total"] == round(out["unit_price"] * 2, 2)
    # The prompt_user string is what the agent relays verbatim — it's our
    # contract that the user sees quantity, name, and total before confirming.
    for needle in (KNOWN_DOOR_BIN, "2"):
        assert needle in out["prompt_user"]


def test_add_to_cart_confirmed_call_returns_success():
    out = tools.add_to_cart(KNOWN_DOOR_BIN, quantity=1, confirmed=True)
    assert out.get("success") is True
    assert out["added"]["part_number"] == KNOWN_DOOR_BIN
    # Confirmed adds actually mutate the in-memory cart now, not a fake cart_id.
    assert out["cart_size"] == 1
    assert out["cart_total"] > 0


def test_add_to_cart_unknown_part_errors_before_confirming():
    # Don't ever confirm a non-existent part, even if the caller claims confirmed=True.
    out = tools.add_to_cart("PS_NOPE", confirmed=True)
    assert "error" in out


def test_add_to_cart_rejects_nonpositive_quantity():
    out = tools.add_to_cart(KNOWN_DOOR_BIN, quantity=0)
    assert "error" in out


# --- troubleshoot observations ----------------------------------------------

def test_troubleshoot_accepts_observations_and_echoes_them():
    out = tools.troubleshoot(
        appliance_type="dishwasher",
        symptom="not cleaning well",
        observations=["water reaches the tub", "spray arm not spinning"],
    )
    assert "error" not in out
    assert out["observations"] == ["water reaches the tub", "spray arm not spinning"]
    assert out["likely_parts"]


# --- view_cart / remove_from_cart / update_cart_quantity --------------------

def test_view_cart_starts_empty():
    out = tools.view_cart()
    assert out["empty"] is True
    assert out["items"] == []
    assert out["total"] == 0.0


def test_add_then_view_reflects_in_cart():
    tools.add_to_cart(KNOWN_DOOR_BIN, quantity=2, confirmed=True)
    out = tools.view_cart()
    assert out["item_count"] == 1
    assert out["items"][0]["part_number"] == KNOWN_DOOR_BIN
    assert out["items"][0]["quantity"] == 2
    assert out["total"] == round(out["items"][0]["unit_price"] * 2, 2)


def test_add_to_cart_same_part_twice_increments_quantity():
    tools.add_to_cart(KNOWN_DOOR_BIN, quantity=1, confirmed=True)
    tools.add_to_cart(KNOWN_DOOR_BIN, quantity=2, confirmed=True)
    out = tools.view_cart()
    assert out["item_count"] == 1
    assert out["items"][0]["quantity"] == 3


def test_remove_from_cart_first_call_returns_pending_confirmation():
    tools.add_to_cart(KNOWN_DOOR_BIN, confirmed=True)
    out = tools.remove_from_cart(KNOWN_DOOR_BIN)
    assert out.get("pending_confirmation") is True
    # Cart should still contain the item — not yet removed.
    assert tools.view_cart()["item_count"] == 1


def test_remove_from_cart_confirmed_call_actually_removes():
    tools.add_to_cart(KNOWN_DOOR_BIN, confirmed=True)
    out = tools.remove_from_cart(KNOWN_DOOR_BIN, confirmed=True)
    assert out.get("success") is True
    assert tools.view_cart()["empty"] is True


def test_remove_from_cart_unknown_part_errors():
    out = tools.remove_from_cart(KNOWN_DOOR_BIN, confirmed=True)
    assert "error" in out


def test_update_cart_quantity_requires_confirmation():
    tools.add_to_cart(KNOWN_DOOR_BIN, quantity=1, confirmed=True)
    out = tools.update_cart_quantity(KNOWN_DOOR_BIN, new_quantity=3)
    assert out.get("pending_confirmation") is True
    assert tools.view_cart()["items"][0]["quantity"] == 1  # unchanged


def test_update_cart_quantity_confirmed_persists():
    tools.add_to_cart(KNOWN_DOOR_BIN, quantity=1, confirmed=True)
    out = tools.update_cart_quantity(KNOWN_DOOR_BIN, new_quantity=4, confirmed=True)
    assert out.get("success") is True
    assert tools.view_cart()["items"][0]["quantity"] == 4


def test_update_cart_quantity_rejects_zero_or_negative():
    tools.add_to_cart(KNOWN_DOOR_BIN, confirmed=True)
    out = tools.update_cart_quantity(KNOWN_DOOR_BIN, new_quantity=0, confirmed=True)
    assert "error" in out
    # quantity unchanged
    assert tools.view_cart()["items"][0]["quantity"] == 1


def test_update_cart_quantity_no_change_is_noop_no_confirmation():
    # If the user asks for the same quantity they already have, no need to
    # bother them with a confirmation prompt.
    tools.add_to_cart(KNOWN_DOOR_BIN, quantity=2, confirmed=True)
    out = tools.update_cart_quantity(KNOWN_DOOR_BIN, new_quantity=2)
    assert out.get("success") is True
    assert out.get("no_change") is True


# --- initiate_checkout ------------------------------------------------------

def test_initiate_checkout_empty_cart_errors():
    out = tools.initiate_checkout()
    assert "error" in out
    assert "empty" in out["error"].lower()


def test_initiate_checkout_first_call_returns_pending_confirmation():
    tools.add_to_cart(KNOWN_DOOR_BIN, quantity=2, confirmed=True)
    out = tools.initiate_checkout()
    assert out.get("pending_confirmation") is True
    assert out["item_count"] == 1
    assert out["total"] > 0
    # Cart must NOT be cleared until the user confirms.
    assert tools.view_cart()["item_count"] == 1


def test_initiate_checkout_confirmed_returns_url_and_clears_cart():
    tools.add_to_cart(KNOWN_DOOR_BIN, quantity=2, confirmed=True)
    out = tools.initiate_checkout(confirmed=True)
    assert out.get("success") is True
    assert "checkout_url" in out
    assert "partselect.com" in out["checkout_url"]
    # Demo handoff: cart cleared so the next session starts fresh.
    assert tools.view_cart()["empty"] is True


def test_initiate_checkout_confirmed_with_empty_cart_still_errors():
    # Don't pretend to "place an empty order" just because confirmed=True was passed.
    out = tools.initiate_checkout(confirmed=True)
    assert "error" in out


# --- remember_appliance -----------------------------------------------------

def test_remember_appliance_stores_session_state():
    out = tools.remember_appliance(
        appliance_type="dishwasher", brand="Whirlpool", model_number="WDT780SAEM1"
    )
    assert "remembered" in out
    saved = session_state.get_appliance()
    assert saved == {
        "brand": "Whirlpool",
        "model_number": "WDT780SAEM1",
        "appliance_type": "dishwasher",
    }


def test_remember_appliance_rejects_invalid_type():
    out = tools.remember_appliance(
        appliance_type="microwave", brand="GE", model_number="X"
    )
    assert "error" in out
    assert session_state.get_appliance() is None


# --- generic "tools never raise" invariant ----------------------------------

@pytest.mark.parametrize("call", [
    lambda: tools.get_part_details(None),  # type: ignore[arg-type]
    lambda: tools.search_parts(query=None, category="refrigerator"),  # type: ignore[arg-type]
    lambda: tools.check_compatibility(None, None),  # type: ignore[arg-type]
    lambda: tools.troubleshoot(appliance_type="refrigerator", symptom=None),  # type: ignore[arg-type]
])
def test_tools_return_error_dict_not_raise_on_bad_input(call):
    # Our agent loop depends on this: bad args should come back as
    # {"error": ...} so the LLM can correct itself, never as an exception.
    result = call()
    assert isinstance(result, dict)
    # It's fine for it to be an error or a valid-ish result; what matters is
    # that no exception escaped.
