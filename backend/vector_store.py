"""Vector store over the scraped parts catalogue.

Uses Chroma (on-disk, persistent) and Gemini's text-embedding-004. Exposes
two operations the tools layer needs:

    search(query, category=None, k=5) -> list[dict]
    get(part_number)                  -> dict | None

The embedding text is shaped for *symptom-driven* user queries ("my ice maker
is broken") rather than part specs — that's how customers actually phrase
problems. We concatenate name, brand, category, and the rich description into
one doc per part.

Design notes:
- We build the index lazily at first access, and skip re-embedding if the
  catalogue hasn't changed (fingerprinted by part_number set).
- Category filtering is done client-side on top of a larger k, rather than
  via Chroma's metadata `where` clause. At ~20 parts this is faster and
  simpler; at 10k+ parts we'd switch to `where={"category": category}`.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
from pathlib import Path
from typing import Any

# Silence the "Failed to send telemetry event ..." noise — a harmless
# chromadb/posthog version mismatch. The env vars are the documented switch;
# the logger silencing is a belt-and-braces fallback because some chromadb
# versions ignore the env vars for telemetry init errors.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

_EMBED_MODEL = "gemini-embedding-001"
_COLLECTION = "partselect"
_CATALOGUE_PATH = Path(__file__).parent.parent / "data" / "parts_catalogue.json"
_CHROMA_PATH = Path(__file__).parent / ".chroma"

_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
# anonymized_telemetry=False silences the harmless but noisy
# "Failed to send telemetry event ..." logs caused by a chromadb/posthog
# version mismatch. Nothing functional depends on it.
_chroma = chromadb.PersistentClient(path=str(_CHROMA_PATH))


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with Gemini. One API call per batch."""
    resp = _client.models.embed_content(model=_EMBED_MODEL, contents=texts)
    return [e.values for e in resp.embeddings]


def _doc_for_embedding(part: dict[str, Any]) -> str:
    """The text we embed for each part.

    Biased towards natural-language problem phrasing — names, brand, category,
    and the description (which often reads like "if your X is broken, replace
    this part"). MPN stays out of the embedding to avoid lexical noise.
    """
    parts = [
        part.get("name") or "",
        f"Brand: {part.get('brand')}" if part.get("brand") else "",
        f"Category: {part.get('category')}" if part.get("category") else "",
        part.get("description") or "",
    ]
    return "\n".join(p for p in parts if p)


def _load_catalogue() -> list[dict[str, Any]]:
    if not _CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"{_CATALOGUE_PATH} not found. Run `python scraper.py --build` first."
        )
    return json.loads(_CATALOGUE_PATH.read_text())


def _ensure_index() -> chromadb.Collection:
    """Return the Chroma collection, (re)building it if the catalogue changed."""
    catalogue = _load_catalogue()
    wanted_ids = sorted(p["part_number"] for p in catalogue)

    col = _chroma.get_or_create_collection(
        name=_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids = sorted(col.get()["ids"])

    if existing_ids == wanted_ids:
        return col  # already in sync.

    # Rebuild from scratch. Cheap at this scale; simpler than diffing.
    if existing_ids:
        _chroma.delete_collection(_COLLECTION)
        col = _chroma.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    docs = [_doc_for_embedding(p) for p in catalogue]
    embeddings = _embed(docs)
    # Chroma metadata must be flat primitives; list fields are stored in the
    # catalogue JSON, which we re-read in get() for full detail.
    metadatas = [
        {
            "part_number": p["part_number"],
            "name": p["name"],
            "brand": p.get("brand") or "",
            "category": p["category"],
            "price": p.get("price") or 0.0,
        }
        for p in catalogue
    ]
    col.add(
        ids=[p["part_number"] for p in catalogue],
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return col


# Catalogue stays in memory for fast get() by part_number.
_catalogue_by_pn: dict[str, dict[str, Any]] = {
    p["part_number"]: p for p in _load_catalogue()
}
_ensure_index()


def search(query: str, category: str | None = None, k: int = 5) -> list[dict[str, Any]]:
    """Top-k semantic matches, optionally restricted to a category.

    Returns full part dicts (from the catalogue) in rank order.
    """
    col = _ensure_index()
    query_emb = _embed([query])[0]

    # Over-fetch when filtering, so post-filter still yields k results.
    n = k * 4 if category else k
    result = col.query(query_embeddings=[query_emb], n_results=n)

    ids = result["ids"][0]
    parts: list[dict[str, Any]] = []
    for pn in ids:
        part = _catalogue_by_pn.get(pn)
        if part is None:
            continue
        if category and part.get("category") != category:
            continue
        parts.append(part)
        if len(parts) >= k:
            break
    return parts


def search_scored(
    query: str, category: str | None = None, k: int = 5
) -> list[tuple[dict[str, Any], float]]:
    """Like `search`, but returns each hit alongside its cosine distance.

    Lower distance = closer match. For normalised embeddings these typically
    fall in [0, 1]. Callers that care about *how good* the matches are (e.g.
    `troubleshoot`, which should drop weak candidates rather than list them)
    use this; callers that just want top-k semantic results use `search`.
    """
    col = _ensure_index()
    query_emb = _embed([query])[0]
    n = k * 4 if category else k
    result = col.query(query_embeddings=[query_emb], n_results=n)

    ids = result["ids"][0]
    distances = result["distances"][0]
    out: list[tuple[dict[str, Any], float]] = []
    for pn, dist in zip(ids, distances):
        part = _catalogue_by_pn.get(pn)
        if part is None:
            continue
        if category and part.get("category") != category:
            continue
        out.append((part, float(dist)))
        if len(out) >= k:
            break
    return out


def get(part_number: str) -> dict[str, Any] | None:
    """Direct lookup by PartSelect part number."""
    return _catalogue_by_pn.get(part_number)


def similar_part_numbers(part_number: str, n: int = 3, cutoff: float = 0.75) -> list[dict[str, Any]]:
    """Return catalogue parts whose part_number is lexically close to the input.

    Designed for typo recovery: a user typing "PS1175277" (missing a digit)
    should still reach "PS11752778". Uses difflib's SequenceMatcher ratio,
    which handles single-character insertions/deletions/substitutions well at
    this string length (~10 chars).

    Empty input or an exact hit both return []. The cutoff is deliberately
    permissive; `get_part_details` only uses this when the exact lookup has
    already missed, so false positives just become "did you mean..." options.
    """
    if not part_number or part_number in _catalogue_by_pn:
        return []
    # Case-insensitive matching — users often type "ps11752778".
    pn_upper = part_number.upper()
    candidates = list(_catalogue_by_pn.keys())
    close = difflib.get_close_matches(pn_upper, candidates, n=n, cutoff=cutoff)
    return [_catalogue_by_pn[pn] for pn in close]
