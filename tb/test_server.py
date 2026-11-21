"""Non-blocking TCP server for the software-simulated Verilog accelerator."""

import socket
import json
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from test_autosafe import run_scene


@cocotb.test()
async def server(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst_n.value = 0
    dut.in_we.value = 0
    dut.start.value = 0
    dut.in_addr.value = 0
    dut.in_data.value = 0

    for _ in range(3):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 5555))
    srv.listen(1)
    srv.setblocking(False)

    conn = None
    buffer = b""

    dut._log.info("Verilog simulation server ready on 127.0.0.1:5555")

    try:
        while True:
            if conn is None:
                try:
                    conn, address = srv.accept()
                    conn.setblocking(False)
                    dut._log.info("Dashboard connected: %s", address)
                except BlockingIOError:
                    await Timer(10, units="ns")
                    continue

            try:
                chunk = conn.recv(4096)
            except BlockingIOError:
                chunk = None

            if chunk == b"":
                dut._log.info("Client disconnected; waiting for another client")
                conn.close()
                conn = None
                buffer = b""
                continue

            if chunk:
                buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)

                if not line.strip():
                    continue
                if line.strip() == b"quit":
                    return

                try:
                    features = json.loads(line.decode("utf-8"))
                    if not isinstance(features, list) or len(features) != 12:
                        raise ValueError("Expected a JSON array of exactly 12 features")
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                    dut._log.warning("Ignoring invalid client message %r: %s", line, exc)
                    continue
                action, cycles = await run_scene(dut, features)

                conn.sendall(("%d\n" % action).encode("utf-8"))

            await Timer(10, units="ns")

    finally:
        if conn is not None:
            conn.close()
        srv.close()
