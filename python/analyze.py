"""Make graphs and a results table from results/*.json"""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as m
RES = os.path.join(m.ROOT, "results")
cv = json.load(open(os.path.join(RES, "coverage_curves.json")))
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cv["random"]) + 1), cv["random"], label="random")
plt.plot(range(1, len(cv["guided"]) + 1), cv["guided"], label="LLM-guided (%s)" % cv["backend"])
plt.xlabel("number of test scenes"); plt.ylabel("coverage bins hit (of %d)" % cv["total_bins"])
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(RES, "coverage_curve.png"), dpi=150)

path = os.path.join(RES, "all_results.json")
if os.path.exists(path):
    rows = json.load(open(path)); acc = json.load(open(os.path.join(RES, "accuracy.json")))
    names = {0: "fault-free", 1: "clamp at 63", 2: "no ReLU", 3: "L1 bias ignored",
             4: "shift off by one", 5: "L2 bias ignored", 6: "acc clamp 14500"}
    lines = ["| Bug | Description | Random: caught (first test #) | Guided: caught (first test #) | Unsafe (random) | Unsafe (guided) |",
             "|---|---|---|---|---|---|"]
    for b in range(0, 7):
        r = {x["set"]: x for x in rows if x["bug"] == b}
        f = lambda x: ("yes (%s)" % x["first_detect"]) if x["mismatches"] else "no"
        lines.append("| %d | %s | %s | %s | %d | %d |" % (b, names[b], f(r["random"]), f(r["guided"]), r["random"]["unsafe"], r["guided"]["unsafe"]))
    cyc = rows[0]["cycles_per_inference"]
    lines += ["", "Accuracy: float %.1f%%, INT8 %.1f%%. Cycles per inference: %d (at 100 MHz = %.2f us)." %
              (100 * acc["float_acc"], 100 * acc["int8_acc"], cyc, cyc * 0.01)]
    open(os.path.join(RES, "results_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
