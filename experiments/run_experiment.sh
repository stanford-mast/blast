#!/bin/bash
cd "$(dirname "$0")/.." || exit
export LMNR_LOGGING_LEVEL=debug
python -m experiments.runner --config experiments/configs/my-experiment-config.yaml