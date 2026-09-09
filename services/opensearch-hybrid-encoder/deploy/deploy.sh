#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: deploy.sh /absolute/path/to/encoder.env" >&2
    exit 64
fi

environment_file=$1
deployment_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

python3 "$deployment_directory/preflight.py" "$environment_file"

base_file=$deployment_directory/compose.yaml
routing_mode=$(sed -n 's/^ENCODER_ROUTING_MODE=//p' "$environment_file")
case "$routing_mode" in
    docker-network)
        routing_file=$deployment_directory/compose.docker-network.yaml
        magento_network=$(sed -n 's/^MAGENTO_DOCKER_NETWORK=//p' "$environment_file")
        docker network inspect "$magento_network" >/dev/null
        ;;
    loopback)
        routing_file=$deployment_directory/compose.loopback.yaml
        ;;
    *)
        echo "ENCODER_ROUTING_MODE must be docker-network or loopback" >&2
        exit 65
        ;;
esac

docker compose \
    --env-file "$environment_file" \
    --file "$base_file" \
    --file "$routing_file" \
    config --quiet
docker compose \
    --env-file "$environment_file" \
    --file "$base_file" \
    --file "$routing_file" \
    up --detach --wait
