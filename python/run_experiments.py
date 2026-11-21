"""Run every bug x {random, guided} through the Verilog simulation and collect results.
Run from anywhere: python run_experiments.py   (needs iverilog + cocotb==1.9.2 + make)"""
import os, json, subprocess, sys
import model as m
TB = os.path.join(m.ROOT, "tb"); RES = os.path.join(m.ROOT, "results")
rows = []
for bug in range(0, 7):
    for kind in ("random", "guided"):
        out = os.path.join(RES, "result_bug%d_%s.json" % (bug, kind))
        env = dict(os.environ, BUG=str(bug), SCEN=os.path.join(RES, "scen_%s.json" % kind), RESULT=out)
        r = subprocess.run(["make", "-C", TB], env=env, capture_output=True, text=True)
        if not os.path.exists(out):
            print(r.stdout[-2000:], r.stderr[-2000:]); sys.exit("simulation failed for bug %d %s" % (bug, kind))
        d = json.load(open(out)); d["set"] = kind; rows.append(d)
        print("bug %d %-7s mismatches=%4d first_detect=%s unsafe=%d" % (bug, kind, d["mismatches"], d["first_detect"], d["unsafe"]))
json.dump(rows, open(os.path.join(RES, "all_results.json"), "w"), indent=1)
