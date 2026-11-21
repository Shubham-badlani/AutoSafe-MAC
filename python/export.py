"""Train, quantize, report accuracy, and write the .hex files the Verilog reads."""
import os, json
import numpy as np
import model as m

def hx(v, bits):
    return format(int(v) & ((1 << bits) - 1), "0%dx" % (bits // 4))

def main():
    P = m.train()
    Q = m.quantize(P); m.save(Q)
    scenes, X, y = m.make_dataset(2000, seed=99)          # unseen test set
    f_acc = (m.float_predict(P, X) == y).mean()
    q_acc = np.mean([m.golden(Q, x)["action"] == t for x, t in zip(X, y)])
    print("float acc %.3f | INT8 acc %.3f | shift %d" % (f_acc, q_acc, Q["sh1"][0]))
    tb = os.path.join(m.ROOT, "tb")
    def w(name, vals, bits):
        open(os.path.join(tb, name), "w").write("\n".join(hx(v, bits) for v in vals) + "\n")
    w("w1.hex", Q["w1"].flatten(), 8); w("w2.hex", Q["w2"].flatten(), 8)
    w("b1.hex", Q["b1"], 32);          w("b2.hex", Q["b2"], 32)
    w("sh1.hex", Q["sh1"], 8)
    os.makedirs(os.path.join(m.ROOT, "results"), exist_ok=True)
    json.dump({"float_acc": float(f_acc), "int8_acc": float(q_acc)},
              open(os.path.join(m.ROOT, "results", "accuracy.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
