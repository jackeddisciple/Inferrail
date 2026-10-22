"""The first dashboard must only query metrics that exist, and must show TTFT,
load and policy per replica."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "dashboards" / "inferrail.json"
SNAPSHOT = (
    REPO_ROOT / "crates/inferrail-gateway/tests/snapshots/admin__metric_names_and_labels.snap"
)
VLLM = REPO_ROOT / "tests/fixtures/vllm/metrics.txt"
COMPOSE = REPO_ROOT / "deploy/compose/docker-compose.yaml"

METRIC = re.compile(r"\b(?:inferrail_|vllm:)[a-zA-Z0-9_:]*")


def exported() -> set[str]:
    names = set()
    for line in SNAPSHOT.read_text().splitlines():
        if line.startswith("inferrail_"):
            names.add(line.split("{")[0])
    for line in VLLM.read_text().splitlines():
        if line and not line.startswith("#"):
            names.add(re.split(r"[{ ]", line)[0])
    return names


def panels() -> list[dict]:
    dashboard = json.loads(DASHBOARD.read_text())
    return [p for p in dashboard["panels"] if p["type"] != "row"]


def test_every_query_names_a_metric_that_exists():
    known = exported()
    queried = set()
    for panel in panels():
        assert panel["datasource"]["uid"] == "prometheus", panel["title"]
        for target in panel["targets"]:
            queried.update(METRIC.findall(target["expr"]))
    assert queried, "no metrics queried"
    assert queried <= known, f"unknown metrics: {sorted(queried - known)}"


def test_ttft_load_and_policy_are_shown_per_replica():
    by_title = {p["title"]: p for p in panels()}

    def per_replica(title_part: str) -> list[dict]:
        found = [
            p
            for title, p in by_title.items()
            if title_part in title.lower()
            and any("replica" in t["expr"] + t.get("legendFormat", "") for t in p["targets"])
        ]
        assert found, f"no per replica panel about {title_part}"
        return found

    per_replica("ttft")
    per_replica("in flight")
    per_replica("kv cache")
    per_replica("policy")
    by_policy = by_title["TTFT by policy (gateway)"]["targets"]
    assert any("by (le, policy)" in t["expr"] for t in by_policy)


def test_panels_do_not_overlap():
    taken = set()
    for panel in json.loads(DASHBOARD.read_text())["panels"]:
        grid = panel["gridPos"]
        assert grid["x"] + grid["w"] <= 24, panel["title"]
        for x in range(grid["x"], grid["x"] + grid["w"]):
            for y in range(grid["y"], grid["y"] + grid["h"]):
                assert (x, y) not in taken, panel["title"]
                taken.add((x, y))


def test_prometheus_scrapes_the_gateway_admin_port_and_every_engine():
    text = (REPO_ROOT / "deploy/compose/prometheus.yml").read_text()
    assert '"gateway:9090"' in text
    gateway = (REPO_ROOT / "deploy/compose/gateway.yaml").read_text()
    for replica in re.findall(r"id: (sim-\d)", gateway):
        assert f'"{replica}:8000"' in text
        assert f"replica: {replica}" in text


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_compose_file_is_valid():
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", "--quiet"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
