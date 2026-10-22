import subprocess
from pathlib import Path

import pytest

from inferrail_bench import fleet

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL = "sim-model"
SOLO = "sim-solo"

SIM_FLAGS = [
    "--enable-prompt-tokens-details",
    "--prefill-token-us",
    "20",
    "--prefill-fixed-ms",
    "1",
    "--decode-token-ms",
    "1",
    "--max-num-seqs",
    "1024",
]


def _binary(name: str) -> Path:
    try:
        return fleet.binary(name)
    except FileNotFoundError as err:
        pytest.fail(f"{err} or `make e2e`")


def _start(args: list[str], log_path: Path, health_url: str) -> subprocess.Popen:
    try:
        return fleet.start(args, log_path, health_url)
    except RuntimeError as err:
        pytest.fail(str(err))


_stop = fleet.stop
_free_port = fleet.free_port


def _sim(logs: Path, names: list[str]) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    args = [str(_binary("inferrail-sim")), "--host", "127.0.0.1"]
    args += ["--port", str(port), "--served-model-name", *names, *SIM_FLAGS]
    return _start(args, logs / f"sim-{port}.log", f"{url}/health"), url


@pytest.fixture(scope="session")
def sim_base_url(tmp_path_factory):
    process, url = _sim(tmp_path_factory.mktemp("sim"), [MODEL, "sim-alias"])
    yield url
    _stop(process)


@pytest.fixture(scope="session")
def gateway_base_url(tmp_path_factory):
    logs = tmp_path_factory.mktemp("gateway")
    sims = [_sim(logs, [MODEL, "sim-alias", *([SOLO] if i == 0 else [])]) for i in range(3)]
    port = _free_port()
    lines = [
        "server:",
        f"  listen: 127.0.0.1:{port}",
        f"  admin_listen: 127.0.0.1:{_free_port()}",
        "models:",
        f"  - name: {MODEL}",
        "  - name: sim-alias",
        f"  - name: {SOLO}",
        "replicas:",
    ]
    for i, (_, url) in enumerate(sims):
        served = [MODEL, "sim-alias", *([SOLO] if i == 0 else [])]
        lines += [f"  - id: r{i}", f"    url: {url}", f"    models: [{', '.join(served)}]"]
    lines += ["routing:", "  default_policy: least_request"]
    lines += ["health:", "  interval: 200ms", "  timeout: 100ms"]
    config = logs / "gateway.yaml"
    config.write_text("\n".join(lines) + "\n")
    url = f"http://127.0.0.1:{port}"
    binary = _binary("inferrail-gateway")
    gateway = _start([str(binary), "--config", str(config)], logs / "gateway.log", f"{url}/health")
    yield url
    _stop(gateway)
    for process, _ in sims:
        _stop(process)


@pytest.fixture(scope="session", params=["sim", "gateway"])
def target(request):
    return request.param, request.getfixturevalue(f"{request.param}_base_url")


@pytest.fixture
def base_url(target):
    return target[1]


@pytest.fixture
def cache_model(target):
    return MODEL if target[0] == "sim" else SOLO


@pytest.fixture
def client(base_url):
    from openai import OpenAI

    with OpenAI(base_url=f"{base_url}/v1", api_key="unused", max_retries=0) as client:
        yield client
