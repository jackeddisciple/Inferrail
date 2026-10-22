"""Starts simulated engines and a gateway as real processes, for tests and
for comparisons that need a cold fleet per run."""

import os
import socket
import subprocess
import time
import urllib.request
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL = "inferrail/sim-8b"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def binary(name: str, profile: str = "debug") -> Path:
    """`INFERRAIL_SIM_BIN` or `INFERRAIL_GATEWAY_BIN` when set, otherwise the
    cargo build output for `profile`."""
    variable = {"inferrail-sim": "INFERRAIL_SIM_BIN", "inferrail-gateway": "INFERRAIL_GATEWAY_BIN"}
    configured = os.environ.get(variable[name])
    path = Path(configured) if configured else REPO_ROOT / "target" / profile / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `cargo build -p {name}`")
    return path


def start(
    args: list[str], log_path: Path, health_url: str, timeout: float = 15
) -> subprocess.Popen:
    log = log_path.open("wb")
    process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=1) as reply:
                if reply.status == 200:
                    return process
        except OSError:
            if process.poll() is not None:
                log.close()
                raise RuntimeError(f"{args[0]} exited: {log_path.read_text()[-2000:]}") from None
            if time.monotonic() > deadline:
                process.kill()
                raise RuntimeError(f"{args[0]} did not become healthy") from None
            time.sleep(0.05)


def stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def sim(logs: Path, names: list[str], flags: list[str], profile: str = "debug"):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    args = [str(binary("inferrail-sim", profile)), "--host", "127.0.0.1", "--port", str(port)]
    args += ["--served-model-name", *names, "--log", "warn", *flags]
    return start(args, logs / f"sim-{port}.log", f"{url}/health"), url


def gateway_config(port: int, admin: int, model: str, sims: list[str], policy: str) -> str:
    lines = [
        "server:",
        f"  listen: 127.0.0.1:{port}",
        f"  admin_listen: 127.0.0.1:{admin}",
        "models:",
        f"  - name: {model}",
        "replicas:",
    ]
    for i, url in enumerate(sims):
        lines += [f"  - id: r{i}", f"    url: {url}", f"    models: [{model}]"]
    lines += ["routing:", f"  default_policy: {policy}", "  seed: 7"]
    lines += ["health:", "  interval: 200ms", "  timeout: 100ms", "logging:", "  level: warn"]
    return "\n".join(lines) + "\n"


@dataclass
class Fleet:
    gateway: str
    admin: str
    sims: list[str]
    config: str


@contextmanager
def fleet(workdir: Path, sims: int, policy: str, flags: list[str], profile: str = "debug"):
    """A gateway routing `MODEL` with `policy` over `sims` fresh simulators."""
    workdir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        urls = []
        for _ in range(sims):
            process, url = sim(workdir, [MODEL], flags, profile)
            stack.callback(stop, process)
            urls.append(url)
        port, admin = free_port(), free_port()
        config = gateway_config(port, admin, MODEL, urls, policy)
        path = workdir / f"gateway-{port}.yaml"
        path.write_text(config)
        url = f"http://127.0.0.1:{port}"
        args = [str(binary("inferrail-gateway", profile)), "--config", str(path)]
        process = start(args, workdir / f"gateway-{port}.log", f"{url}/health")
        stack.callback(stop, process)
        yield Fleet(url, f"http://127.0.0.1:{admin}", urls, config)


def version(name: str, profile: str = "debug") -> str:
    out = subprocess.run([str(binary(name, profile)), "--version"], capture_output=True, text=True)
    return out.stdout.strip()
