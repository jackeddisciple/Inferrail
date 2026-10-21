import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL = "sim-model"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _binary() -> Path:
    configured = os.environ.get("INFERRAIL_SIM_BIN")
    path = Path(configured) if configured else REPO_ROOT / "target" / "debug" / "inferrail-sim"
    if not path.exists():
        pytest.fail(f"{path} not found; run `cargo build -p inferrail-sim` or `make e2e`")
    return path


@pytest.fixture(scope="session")
def sim_base_url(tmp_path_factory):
    port = _free_port()
    log_path = tmp_path_factory.mktemp("sim") / "inferrail-sim.log"
    log = log_path.open("wb")
    process = subprocess.Popen(
        [
            str(_binary()),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            MODEL,
            "sim-alias",
            "--enable-prompt-tokens-details",
            "--prefill-token-us",
            "20",
            "--prefill-fixed-ms",
            "1",
            "--decode-token-ms",
            "1",
            "--max-num-seqs",
            "1024",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 15
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as reply:
                if reply.status == 200:
                    break
        except OSError:
            if process.poll() is not None:
                log.close()
                pytest.fail(f"simulator exited: {log_path.read_text()[-2000:]}")
            if time.monotonic() > deadline:
                process.kill()
                pytest.fail("simulator did not become healthy")
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    process.terminate()
    process.wait(timeout=10)
    log.close()


@pytest.fixture
def client(sim_base_url):
    from openai import OpenAI

    with OpenAI(base_url=f"{sim_base_url}/v1", api_key="unused", max_retries=0) as client:
        yield client
