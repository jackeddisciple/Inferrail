CARGO ?= cargo
UV ?= uv
NEXTEST := $(shell $(CARGO) nextest --version >/dev/null 2>&1 && echo "nextest run" || echo "test")

.PHONY: check fmt lint test test-rust test-python bench e2e report

check: fmt lint test

fmt:
	$(CARGO) fmt --all -- --check

lint:
	$(CARGO) clippy --workspace --all-targets -- -D warnings

test: test-rust test-python

test-rust:
	$(CARGO) $(NEXTEST) --workspace
	$(CARGO) test --workspace --doc

test-python:
	cd bench && $(UV) run --frozen pytest -q

bench:
	$(CARGO) bench --workspace

e2e:
	@echo "e2e: no end to end suite yet"

report:
	@echo "report: no benchmark results yet"
