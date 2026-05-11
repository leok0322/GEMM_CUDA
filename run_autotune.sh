#!/usr/bin/env bash
for i in $(seq 10 10); do
  bash "$(dirname "$0")/scripts/kernel_${i}_autotuner.sh"
done
