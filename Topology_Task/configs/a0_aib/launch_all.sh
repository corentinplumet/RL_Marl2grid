#!/usr/bin/env bash
set -euo pipefail

sbatch job_jed.sh configs/a0_aib/a0_aib_00_flat_local_t020_s0.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_00_flat_local_t020_s1.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_00_flat_local_t020_s2.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_01_flat_local_t010_s0.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_01_flat_local_t010_s1.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_01_flat_local_t010_s2.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_02_flat_local_t035_s0.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_02_flat_local_t035_s1.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_02_flat_local_t035_s2.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_03_gate_hgreedy_sep_local_t020_s0.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_03_gate_hgreedy_sep_local_t020_s1.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_03_gate_hgreedy_sep_local_t020_s2.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_04_flat_nonidle_t020_s0.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_04_flat_nonidle_t020_s1.toml
sbatch job_jed.sh configs/a0_aib/a0_aib_04_flat_nonidle_t020_s2.toml
