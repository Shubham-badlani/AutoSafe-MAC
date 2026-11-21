"""cocotb testbench (UVM-style roles in one file):
   driver  -> run_scene(): writes 12 inputs, pulses start
   monitor -> waits for done, samples action and cycle count
   scoreboard -> compares hardware action with the bit-exact golden model (python/model.py)
Env: SCEN = scenario json path, RESULT = output json path, BUG = injected bug id (for the report only)."""
import os, sys, json
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import model as m

async def run_scene(dut, x):
    for k, v in enumerate(x):                                   # driver
        dut.in_we.value = 1; dut.in_addr.value = k; dut.in_data.value = int(v) & 0xFF
        await RisingEdge(dut.clk)
    dut.in_we.value = 0; dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.clk)
    cycles = 0
    while int(dut.done.value) != 1:                             # monitor
        await RisingEdge(dut.clk); cycles += 1
        assert cycles < 5000, "timeout: done never asserted"
    return int(dut.action.value), cycles

@cocotb.test()
async def scoreboard(dut):
    scen_path = os.environ.get("SCEN", os.path.join(HERE, "..", "results", "scen_random.json"))
    res_path = os.environ.get("RESULT", os.path.join(HERE, "..", "results", "result_tmp.json"))
    bug = int(os.environ.get("BUG", "0"))
    scenes = json.load(open(scen_path))
    Q = m.load()
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0; dut.in_we.value = 0; dut.start.value = 0
    dut.in_addr.value = 0; dut.in_data.value = 0
    for _ in range(3): await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    mismatches, first, unsafe, unsafe_gold, cyc = 0, None, 0, 0, 0
    for idx, s in enumerate(scenes):
        x = m.encode(s)
        exp = m.golden(Q, x)["action"]
        label = m.rule_label(s)
        got, cyc = await run_scene(dut, x)
        if got != exp:
            mismatches += 1
            if first is None: first = idx + 1                  # tests needed to catch the bug
        if label == 1 and got != 1: unsafe += 1                # hardware missed a STOP
        if label == 1 and exp != 1: unsafe_gold += 1           # fault-free model missed a STOP
    out = dict(bug=bug, n=len(scenes), mismatches=mismatches, first_detect=first,
               unsafe=unsafe, unsafe_golden=unsafe_gold, cycles_per_inference=cyc + 2)
    json.dump(out, open(res_path, "w"))
    dut._log.info("RESULT %s", out)
    if bug == 0:
        assert mismatches == 0, "fault-free hardware differs from golden model!"
