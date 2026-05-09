#!/usr/bin/env bash
for i in $(seq 9 9); do
  bash "$(dirname "$0")/scripts/kernel_${i}_autotuner.sh"
done
