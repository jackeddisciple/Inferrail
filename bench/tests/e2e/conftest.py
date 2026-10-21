import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _binary(variable: str, name: str) -> Path:
    configured = os.environ.get(variable)
    path = Path(configured) if configured else REPO_ROOT / "target" / "debug" / name
    if not path.exists():
        pytest.fail(f"{path} not found; run `cargo build -p {name}` or `make e2e`")
    return path


def _start(args: list[str], log_path: Path, health_url: str) -> subprocess.Popen:
    log = log_path.open("wb")
    process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 15
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=1) as reply:
                if reply.status == 200:
                    return process
        except OSError:
            if process.poll() is not None:
                log.close()
                pytest.fail(f"{args[0]} exited: {log_path.read_text()[-2000:]}")
            if time.monotonic() > deadline:
                process.kill()
                pytest.fail(f"{args[0]} did not become healthy")
            time.sleep(0.05)


def _stop(process: subprocess.Popen) -> None:
    process.terminate()
    process.wait(timeout=10)


def _sim(logs: Path, names: list[str]) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    args = [str(_binary("INFERRAIL_SIM_BIN", "inferrail-sim")), "--host", "127.0.0.1"]
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
    binary = _binary("INFERRAIL_GATEWAY_BIN", "inferrail-gateway")
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
