CARGO ?= cargo
UV ?= uv
NEXTEST := $(shell $(CARGO) nextest --version >/dev/null 2>&1 && echo "nextest run" || echo "test")

.PHONY: check fmt lint audit test test-rust test-python bench e2e stress report

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
	$(CARGO) build -p inferrail-sim
	cd bench && INFERRAIL_SIM_BIN=$(CURDIR)/target/debug/inferrail-sim $(UV) run --frozen pytest -q -m e2e

stress:
	$(CARGO) test --release -p inferrail-sim --test stress -- --ignored --nocapture --test-threads=1
	$(CARGO) build --release -p inferrail-sim
	cd bench && INFERRAIL_SIM_BIN=$(CURDIR)/target/release/inferrail-sim $(UV) run --frozen pytest -q -s -m stress

report:
	@echo "report: no benchmark results yet"
