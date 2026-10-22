"""Prefix cache counters read straight from engines, for the hit rate the
engines themselves report: vLLM's prefix cache queries and hits, or
SGLang's prompt and cached tokens."""

import asyncio
import re

import httpx2


def _family(*names: str) -> re.Pattern:
    joined = "|".join(re.escape(n) for n in names)
    return re.compile(rf"^(?:{joined})(?:\{{[^}}]*\}})?\s+(\S+)", re.M)


_COUNTERS = {
    "queries": _family("vllm:prefix_cache_queries_total", "sglang:prompt_tokens_total"),
    "hits": _family("vllm:prefix_cache_hits_total", "sglang:cached_tokens_total"),
}


def prefix_counters(page: str) -> dict[str, float]:
    return {
        name: sum(float(v) for v in pattern.findall(page)) for name, pattern in _COUNTERS.items()
    }


async def _page(http: httpx2.AsyncClient, url: str) -> dict[str, float] | None:
    try:
        reply = await http.get(f"{url}/metrics")
    except httpx2.HTTPError:
        return None
    return prefix_counters(reply.text) if reply.status_code == 200 else None


async def scrape(urls: list[str]) -> dict[str, dict[str, float] | None]:
    """Each engine's counters, or `None` for an engine that did not answer."""
    async with httpx2.AsyncClient(timeout=5) as http:
        pages = await asyncio.gather(*(_page(http, url) for url in urls))
    return dict(zip(urls, pages, strict=True))


def hit_rate(before: dict, after: dict) -> dict:
    """Prompt tokens looked up and found between two scrapes, over the
    engines that answered both with counters that did not go backwards (an
    engine that restarted starts from zero); the others are listed as
    missing."""

    def usable(url: str) -> bool:
        old, new = before.get(url), after[url]
        if old is None or new is None:
            return False
        return all(new[k] >= old[k] for k in ("queries", "hits"))

    counted = [u for u in after if usable(u)]
    queries = sum(after[u]["queries"] - before[u]["queries"] for u in counted)
    hits = sum(after[u]["hits"] - before[u]["hits"] for u in counted)
    return {
        "queries": queries,
        "hits": hits,
        "hit_rate": hits / queries if queries else 0.0,
        "missing": sorted(set(after) - set(counted)),
    }
