"""Link to the simulated Verilog accelerator (tb/test_server.py running under cocotb) with golden-model fallback."""
import json, socket
import model as m

class HardwareLink:
    def __init__(self, host="127.0.0.1", port=5555):
        self.Q = m.load(); self.f = None
        try:
            self.sock = socket.create_connection((host, port), timeout=2)
            self.f = self.sock.makefile("rw")
        except OSError:
            self.f = None                                     # simulator not running

    @property
    def source(self):
        return "VERILOG SIM" if self.f else "PYTHON GOLDEN (sim not running)"

    def infer(self, scene):
        """Returns (action_from_hw_or_golden, golden_action)."""
        x = m.encode(scene); gold = m.golden(self.Q, x)["action"]
        if self.f is None: return gold, gold
        self.f.write(json.dumps([int(v) for v in x]) + "\n"); self.f.flush()
        return int(self.f.readline()), gold

    def close(self):
        if self.f:
            try: self.f.write("quit\n"); self.f.flush()
            except OSError: pass
