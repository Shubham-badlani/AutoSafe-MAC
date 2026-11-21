# AutoSafe-MAC: a safety-verified INT8 AI accelerator for self-driving decisions

Four outputs only: **GO, STOP, TURN_LEFT, TURN_RIGHT**.

A tiny neural network (12 sensor features -> 16 hidden -> 4 actions) is trained in Python, quantized to INT8
and implemented in Verilog. A camera (YOLOv8) and a microphone (Whisper) feed it in a live dashboard. The
hardware is verified against a bit-exact Python golden model with a UVM-style cocotb testbench, functional
coverage and 6 injected bugs. BharatGen Param (or Codex) proposes new test scenes from the uncovered bins.

**Honest scope:** training data is synthetic (rule-based labels); camera distances are rough pinhole estimates;
the "hardware" is an RTL simulation, not a chip or FPGA. Proof of concept only, no ISO 26262 claim.

## Component status (fill in the last column yourself after you run each one)
| Component | File |
|---|---|
| Python network + INT8 quantization | python/model.py, export.py |
| Trained weights | python/weights.npz |
| INT8 hex weight files | tb/*.hex |
| Verilog 12-16-4 accelerator | rtl/autosafe_mac.v | 
| cocotb testbench + bug injection | tb/test_autosafe.py | 
| Random / guided scenarios | python/gen_scenarios.py | 
| BharatGen integration | python/bharatgen_check.py | 
| YOLOv8 camera detection | python/dashboard.py | 
| Whisper voice command | python/voice.py | 
| Live dashboard | python/dashboard.py | 
| Yosys synthesis stats | yosys/run_yosys.sh | 
| Quartus area / Fmax (optional) | quartus/run_quartus.sh | 

## Setup (Linux, macOS or Windows WSL2; the webcam/mic demo is easier on native Windows/macOS Python, see note)
```
sudo apt install iverilog make gtkwave yosys python3-pip ffmpeg
python3 -m pip install -r requirements.txt
```
Note: the verification flow needs `make` + Icarus (use WSL2 on Windows). WSL2 usually cannot see the webcam or
microphone; if so, run the Verilog server in WSL2 and the dashboard on Windows (it connects to 127.0.0.1:5555).

## Run
1. Verification experiments: `./run_all.sh` -> `results/` (accuracy.json, coverage_curve.png, results_table.md, llm_log.json)
2. BharatGen check: `cd python && python3 bharatgen_check.py` (prints PASS or FAIL)
   Then `LLM_BACKEND=param python3 gen_scenarios.py 300 && python3 run_experiments.py && python3 analyze.py`
3. Live demo, two terminals:
   - Terminal 1: `cd tb && make MODULE=test_server` (wait for "hardware server ready")
   - Terminal 2: `cd python && python3 dashboard.py` (keys: v voice, + - speed, w wet, r clear, q quit)
   Say "emergency stop" -> forced STOP. If terminal 1 is not running, the screen says "PYTHON GOLDEN".

## Demo flow
Camera sees a pedestrian (YOLOv8) -> features -> Verilog accelerator decides -> golden model cross-check shown on
screen -> voice command overrides -> cocotb/coverage/bug-injection results table.

## Bugs injected (rtl/autosafe_mac.v, parameter BUG)
1 hidden clamp at 63 instead of 127 | 2 ReLU removed | 3 layer-1 bias ignored | 4 shift off by one |
5 layer-2 bias ignored | 6 layer-1 accumulator clamped at +/-14500 (very hard to hit: report honestly if missed)

## AI-use disclosure
(Write which model did what: e.g. Claude drafted code; BharatGen Param / Codex generated test scenes; YOLOv8n and
Whisper are pretrained third-party models used under their own licenses; you ran and checked all results.)

## Results
(paste results/results_table.md, the coverage graph, a dashboard screenshot and the Yosys stats)

## Limitations
Synthetic data; single sequential MAC (not a systolic array); distance estimates are approximate; no FPGA/ASIC flow.
