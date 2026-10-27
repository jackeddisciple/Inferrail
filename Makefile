CARGO ?= cargo
UV ?= uv
NEXTEST := $(shell $(CARGO) nextest --version >/dev/null 2>&1 && echo "nextest run" || echo "test")
JOBS ?= 2

.PHONY: check fmt lint audit test test-rust test-python bench e2e stress goldens baselines routing sweep report stack stack-down

check: fmt lint test

fmt:
	$(CARGO) fmt --all -- --check

lint:
	$(CARGO) clippy --workspace --all-targets -- -D warnings
	cd bench && $(UV) run --frozen ruff check . && $(UV) run --frozen ruff format --check .

audit:
	$(CARGO) audit
	$(CARGO) machete

test: test-rust test-python

test-rust:
	$(CARGO) $(NEXTEST) --workspace
	$(CARGO) test --workspace --doc

test-python:
	cd bench && $(UV) run --frozen pytest -q

bench:
	$(CARGO) bench --workspace

e2e:
	$(CARGO) build -p inferrail-sim -p inferrail-gateway
	cd bench && INFERRAIL_SIM_BIN=$(CURDIR)/target/debug/inferrail-sim \
		INFERRAIL_GATEWAY_BIN=$(CURDIR)/target/debug/inferrail-gateway \
		$(UV) run --frozen pytest -q -m e2e

stress:
	$(CARGO) test --release -p inferrail-sim --test stress -- --ignored --nocapture --test-threads=1
	$(CARGO) test --release -p inferrail-tokenize --test stress -- --ignored --nocapture --test-threads=1
	$(CARGO) test --release -p inferrail-index --test stress -- --ignored --nocapture --test-threads=1
	$(CARGO) test --release -p inferrail-policy --test stress -- --ignored --nocapture --test-threads=1
	$(CARGO) test --release -p inferrail-gateway --test stress -- --ignored --nocapture --test-threads=1
	$(CARGO) test --release -p inferrail-gateway --test stress_telemetry -- --ignored --nocapture --test-threads=1
	$(CARGO) test --release -p inferrail-gateway --test stress_prediction -- --ignored --nocapture --test-threads=1
	$(CARGO) test --release -p inferrail-gateway --test stress_routing -- --ignored --nocapture --test-threads=1
	$(CARGO) build --release -p inferrail-sim -p inferrail-gateway
	cd bench && INFERRAIL_SIM_BIN=$(CURDIR)/target/release/inferrail-sim \
		INFERRAIL_GATEWAY_BIN=$(CURDIR)/target/release/inferrail-gateway \
		$(UV) run --frozen pytest -q -s -m stress

goldens:
	cd tests/fixtures/tokenizers && $(UV) run make_goldens.py

baselines:
	$(CARGO) build --release -p inferrail-sim -p inferrail-gateway
	cd bench && $(UV) run --frozen inferrail-bench compare --out results/baselines

routing:
	$(CARGO) build --release -p inferrail-sim -p inferrail-gateway
	cd bench && $(UV) run --frozen inferrail-bench compare --workloads w1,w2,w6,mix,hot \
		--policies round_robin,random,least_request,p2c,prefix_hash,kv_aware,cost_model \
		--out results/routing
	cd bench && $(UV) run --frozen inferrail-bench compare --workloads w2 \
		--policies cost_model+sessions --out results/routing

sweep:
	$(CARGO) build --release -p inferrail-sim -p inferrail-gateway
	cd bench && $(UV) run --frozen inferrail-bench sweep sweeps/sim.toml --out results/sweep --jobs $(JOBS)
	cd bench && $(UV) run --frozen inferrail-bench report results/sweep --out report/sweep

report:
	cd bench && $(UV) run --frozen inferrail-bench plot results/baselines/*/*.json.gz --out report/baselines
	cd bench && $(UV) run --frozen inferrail-bench plot results/routing/*/*.json.gz --out report/routing
	cd bench && test ! -d results/sweep || $(UV) run --frozen inferrail-bench report results/sweep --out report/sweep

COMPOSE := docker compose -f deploy/compose/docker-compose.yaml

stack:
	$(COMPOSE) up --build --detach

stack-down:
	$(COMPOSE) down
