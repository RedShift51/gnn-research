#!/usr/bin/env bash
# Source this from the fraud-diffusion/ project root to load non-secret config from .env and
# secrets from macOS Keychain into the current shell:
#   source infra/load_secrets.sh
#
# Secrets are never stored in this file or in .env — see infra/set_secret.sh to add one.

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "load_secrets.sh: run this from the fraud-diffusion/ project root (.env not found in cwd)" >&2
fi

_gnnfin_load_secret() {
    local name="$1"
    local value
    value=$(security find-generic-password -a "$(whoami)" -s "gnnfin/$name" -w 2>/dev/null)
    if [ -n "$value" ]; then
        export "$name=$value"
    fi
}

for secret in RUNPOD_API_KEY KAGGLE_KEY GHCR_TOKEN; do
    _gnnfin_load_secret "$secret"
done
unset -f _gnnfin_load_secret
