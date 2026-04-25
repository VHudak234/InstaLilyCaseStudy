"""Tests for `_partial_reply_from_trace` — the salvage path that synthesises a
reply from the trace when Gemini 503s on the final generate call.

This function is pure (no I/O), so it's cheap and worth covering directly.
The agent loop itself isn't unit-tested because it's a thin dispatcher around
the Gemini SDK and mocking that adds more risk than it removes.
"""

from agent import _partial_reply_from_trace


def test_empty_trace_returns_none():
    assert _partial_reply_from_trace([]) is None
    assert _partial_reply_from_trace(None) is None  # type: ignore[arg-type]


def test_trace_with_only_errors_returns_none():
    # If every tool errored, there's nothing useful to salvage — caller falls
    # back to the generic "Gemini is overloaded" message, which is correct.
    trace = [{"tool": "get_part_details", "args": {"part_number": "X"}, "result": {"error": "not found"}}]
    assert _partial_reply_from_trace(trace) is None


def test_salvage_from_search_parts_results_mentions_part_names():
    trace = [{
        "tool": "search_parts",
        "args": {"query": "rack roller", "category": "dishwasher"},
        "result": {
            "query": "rack roller",
            "results": [
                {"part_number": "PS1", "name": "Lower Dishrack Wheel"},
                {"part_number": "PS2", "name": "Dishrack Roller"},
                {"part_number": "PS3", "name": "Upper Rack Adjuster"},
                {"part_number": "PS4", "name": "Silverware Basket"},
            ],
        },
    }]
    reply = _partial_reply_from_trace(trace)
    assert reply is not None
    # First three are mentioned; the fourth is truncated out (keeps the reply short).
    assert "Lower Dishrack Wheel" in reply
    assert "Dishrack Roller" in reply
    assert "Upper Rack Adjuster" in reply
    assert "Silverware Basket" not in reply


def test_salvage_from_troubleshoot_likely_parts():
    trace = [{
        "tool": "troubleshoot",
        "args": {"appliance_type": "dishwasher", "symptom": "rollers"},
        "result": {"likely_parts": [{"part_number": "PS1", "name": "Roller A"}]},
    }]
    reply = _partial_reply_from_trace(trace)
    assert reply and "Roller A" in reply


def test_salvage_from_get_part_details_mentions_pn_and_name():
    trace = [{
        "tool": "get_part_details",
        "args": {"part_number": "PS11752778"},
        "result": {"part_number": "PS11752778", "name": "Door Shelf Bin"},
    }]
    reply = _partial_reply_from_trace(trace)
    assert reply is not None
    assert "PS11752778" in reply
    assert "Door Shelf Bin" in reply


def test_salvage_from_pending_confirmation_uses_prompt_user():
    # If the LLM failed mid-turn during an add_to_cart, we must NOT silently
    # say "all good" — we relay the confirmation prompt so the user still gets
    # a chance to approve.
    prompt = "Please confirm: add 1 × Door Shelf Bin (PS11752778) at $47.40?"
    trace = [{
        "tool": "add_to_cart",
        "args": {"part_number": "PS11752778", "confirmed": False},
        "result": {"pending_confirmation": True, "prompt_user": prompt},
    }]
    reply = _partial_reply_from_trace(trace)
    assert reply == prompt


def test_salvage_walks_backwards_preferring_most_recent_success():
    # Trace: troubleshoot succeeded, then get_part_details succeeded. The
    # more specific (latest) result should drive the reply.
    trace = [
        {
            "tool": "troubleshoot",
            "args": {"appliance_type": "dishwasher", "symptom": "x"},
            "result": {"likely_parts": [{"part_number": "PS1", "name": "Earlier Part"}]},
        },
        {
            "tool": "get_part_details",
            "args": {"part_number": "PS2"},
            "result": {"part_number": "PS2", "name": "Later Part"},
        },
    ]
    reply = _partial_reply_from_trace(trace)
    assert reply is not None
    assert "Later Part" in reply
    assert "Earlier Part" not in reply


def test_salvage_uses_suggestions_from_fuzzy_miss():
    # A get_part_details miss that returned fuzzy suggestions is still
    # useful — salvage should mention the suggested parts rather than giving up.
    trace = [{
        "tool": "get_part_details",
        "args": {"part_number": "PS1175277"},
        "result": {
            "error": "not found",
            "suggestions": [{"part_number": "PS11752778", "name": "Door Shelf Bin"}],
        },
    }]
    reply = _partial_reply_from_trace(trace)
    assert reply is not None
    assert "Door Shelf Bin" in reply


def test_salvage_skips_errored_steps_and_uses_last_good_one():
    trace = [
        {"tool": "search_parts", "args": {}, "result": {"results": [{"part_number": "PS1", "name": "Good Part"}]}},
        {"tool": "get_part_details", "args": {}, "result": {"error": "not found"}},
    ]
    reply = _partial_reply_from_trace(trace)
    assert reply is not None
    assert "Good Part" in reply
