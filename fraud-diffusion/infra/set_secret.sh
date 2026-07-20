#!/usr/bin/env bash
# Store a secret in macOS Keychain (encrypted at rest) without it ever being typed into chat
# or a Bash tool call. Run this YOURSELF in a real Terminal window (not via the assistant):
#   ./infra/set_secret.sh GHCR_TOKEN
#   ./infra/set_secret.sh KAGGLE_KEY
#   ./infra/set_secret.sh RUNPOD_API_KEY
#
# Reads the value with input hidden (like a password prompt) — it won't echo to the screen
# or be captured in any log.

set -e
NAME="$1"
if [ -z "$NAME" ]; then
    echo "Usage: $0 SECRET_NAME"
    exit 1
fi

read -r -s -p "Value for $NAME (input hidden): " VALUE
echo
security add-generic-password -a "$(whoami)" -s "gnnfin/$NAME" -w "$VALUE" -U
echo "Stored $NAME in Keychain (service: gnnfin/$NAME)."
