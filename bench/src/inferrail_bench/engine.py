"""Prefix cache counters read straight from engines, for the hit rate the
engines themselves report."""

import asyncio
import re

import httpx2

_COUNTERS = {
    "queries": re.compile(r"^vllm:prefix_cache_queries_total(?:\{[^}]*\})?\s+(\S+)", re.M),
    "hits": re.compile(r"^vllm:prefix_cache_hits_total(?:\{[^}]*\})?\s+(\S+)", re.M),
}


def prefix_counters(page: str) -> dict[str, float]:
    return {
        name: sum(float(v) for v in pattern.findall(page)) for name, pattern in _COUNTERS.items()
    }


async def scrape(urls: list[str]) -> dict[str, dict[str, float]]:
    async with httpx2.AsyncClient(timeout=5) as http:
        pages = await asyncio.gather(*(http.get(f"{url}/metrics") for url in urls))
    return {url: prefix_counters(page.text) for url, page in zip(urls, pages, strict=True)}


def hit_rate(before: dict, after: dict) -> dict:
    """Prompt tokens looked up and found between two scrapes, over all engines."""
    queries = sum(after[u]["queries"] - before[u]["queries"] for u in after)
    hits = sum(after[u]["hits"] - before[u]["hits"] for u in after)
    return {
        "queries": queries,
        "hits": hits,
        "hit_rate": hits / queries if queries else 0.0,
    }
