#!/bin/sh
# Needs Quartus Prime Lite on PATH (quartus_sh). The $readmemh files must sit next to the project.
set -e
cd "$(dirname "$0")"
cp ../tb/*.hex .
quartus_sh -t run_quartus.tcl
quartus_sh --flow compile autosafe
echo "---- fitter summary ----";  cat output_files/autosafe.fit.summary
echo "---- timing summary ----";  cat output_files/autosafe.sta.summary 2>/dev/null || echo "see output_files/autosafe.sta.rpt (Fmax summary)"
