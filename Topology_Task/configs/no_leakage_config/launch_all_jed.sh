#!/usr/bin/env bash
# All four corrected-construction screens at seed 0: 44 runs on JED.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in screen_a_depth_width screen_b_encoder_pooling screen_c_structure screen_d_message_direction; do
  bash "$here/$s/launch_jed.sh" "$@"
done
