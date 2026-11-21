#!/bin/sh
# Full flow. Linux / macOS / WSL2.  Prereqs: iverilog, make, python3 -m pip install numpy matplotlib cocotb==1.9.2
set -e
cd python
python3 export.py                          # train, quantize, write tb/*.hex
python3 gen_scenarios.py 300               # LLM_BACKEND=rules|param|manual python3 gen_scenarios.py 300
python3 run_experiments.py                 # Verilog simulation, 7 designs x 2 test sets
python3 analyze.py                         # graphs + table in results/
cd ..; ./yosys/run_yosys.sh || true               # optional: synthesis statistics
# Live demo (two terminals):  1) cd tb && make MODULE=test_server   2) cd python && python3 dashboard.py
