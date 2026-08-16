#!/usr/bin/env bash
# Convenience entry point for the 1024-action top-k zero-shot evaluation.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACTION_SIZE=1024
exec "$script_dir/launch_shared_zero_shot_wcci_mk64_izar.sh" "$@"
