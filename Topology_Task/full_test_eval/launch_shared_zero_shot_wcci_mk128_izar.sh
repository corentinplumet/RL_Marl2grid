#!/usr/bin/env bash
# Convenience entry point for the 128-actions-per-agent zero-shot evaluation.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACTION_SIZE=128
exec "$script_dir/launch_shared_zero_shot_wcci_mk64_izar.sh" "$@"
