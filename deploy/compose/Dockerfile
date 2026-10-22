# Builds inferrail-gateway and inferrail-sim into one small runtime image.
FROM rust:1.98.1-slim-trixie AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p inferrail-gateway -p inferrail-sim \
    && cp target/release/inferrail-gateway target/release/inferrail-sim /usr/local/bin/

FROM debian:trixie-20261005-slim
COPY --from=build /usr/local/bin/inferrail-gateway /usr/local/bin/inferrail-sim /usr/local/bin/
USER 65534
