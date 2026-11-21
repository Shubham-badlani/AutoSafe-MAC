#!/bin/sh
# Synthesis statistics with Yosys (free). Run from this folder. Weight .hex files are read from ../tb
set -e
cd "$(dirname "$0")/../tb"
yosys -p "read_verilog ../rtl/autosafe_mac.v; hierarchy -top autosafe_mac; synth -flatten -top autosafe_mac; stat" | tee ../results/yosys_stats.txt
echo "Saved: results/yosys_stats.txt (use the cell count and flip-flop count in your report)"
