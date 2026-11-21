"""Live demo: webcam -> YOLOv8 -> sensor features -> Verilog accelerator (simulated) -> action,
with a Whisper voice override and a golden-model cross-check on every frame.

Keys:  v = listen for a voice command (3 s)   + / - = speed   w = toggle wet road
       r = clear voice override               q = quit
Voice: "emergency stop" (latched) | "resume" / "go" | "turn left" | "turn right" (3 s)
Start the hardware side first (optional): cd tb && make MODULE=test_server   -> otherwise the golden model is used and the screen says so.
Run:  python dashboard.py [--source 0 | video.mp4]"""
import argparse, csv, os, time
import cv2
import model as m, perception as pc, voice
from hwlink import HardwareLink

COLORS = {"GO": (0, 200, 0), "STOP": (0, 0, 230), "TURN_LEFT": (230, 160, 0), "TURN_RIGHT": (230, 160, 0)}
ap = argparse.ArgumentParser(); ap.add_argument("--source", default="0"); ap.add_argument("--model", default="yolov8n.pt")
args = ap.parse_args()

from ultralytics import YOLO
yolo = YOLO(args.model)                                  # downloads yolov8n.pt on first run (needs internet)
link = HardwareLink()
cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
os.makedirs(os.path.join(m.ROOT, "results"), exist_ok=True)
log = csv.writer(open(os.path.join(m.ROOT, "results", "demo_log.csv"), "w", newline=""))
log.writerow(["t", "front_m", "left_m", "right_m", "ped", "speed", "wet", "hw", "golden", "final", "voice"])

speed, wet, override, until, heard, frame_no, t0 = 30.0, 0, None, 0.0, "", 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    res = yolo(frame, verbose=False)[0]
    dets = []
    for b in res.boxes:
        cls = res.names[int(b.cls)]; x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append((cls, x1, y1, x2, y2, float(b.conf)))
        if cls in pc.REAL_HEIGHT_M:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            d = pc.estimate_distance(cls, y2 - y1)
            cv2.putText(frame, "%s %.1fm" % (cls, d), (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    scene = pc.scene_from_detections(dets, w, speed, wet)
    hw, gold = link.infer(scene)
    final = hw
    if override == "STOP": final = 1
    elif override in ("TURN_LEFT", "TURN_RIGHT") and time.time() < until: final = m.ACTIONS.index(override)
    name = m.ACTIONS[final]

    lines = ["Source: " + link.source,
             "HW=%s  golden=%s  %s" % (m.ACTIONS[hw], m.ACTIONS[gold], "MATCH" if hw == gold else "*** MISMATCH ***"),
             "front %.1fm  left %.1fm  right %.1fm  ped=%d" % (scene["d0"], scene["d6"], scene["d2"], scene["ped"]),
             "speed %d km/h  wet=%d" % (speed, wet),
             "voice: '%s'  override=%s" % (heard, override)]
    cv2.rectangle(frame, (0, 0), (w, 22 * len(lines) + 8), (30, 30, 30), -1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(frame, (0, h - 50), (w, h), COLORS[name], -1)
    cv2.putText(frame, "DECISION: " + name, (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow("AutoSafe-MAC", frame)

    frame_no += 1
    if frame_no % 10 == 0:
        log.writerow([round(time.time() - t0, 1), round(scene["d0"], 1), round(scene["d6"], 1), round(scene["d2"], 1),
                      scene["ped"], speed, wet, m.ACTIONS[hw], m.ACTIONS[gold], name, heard])
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    elif k == ord("w"): wet ^= 1
    elif k in (ord("+"), ord("=")): speed = min(60, speed + 5)
    elif k == ord("-"): speed = max(0, speed - 5)
    elif k == ord("r"): override = None
    elif k == ord("v"):
        cv2.putText(frame, "LISTENING...", (w // 3, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        cv2.imshow("AutoSafe-MAC", frame); cv2.waitKey(1)
        heard = voice.listen(3.0); cmd = voice.parse_command(heard)
        if cmd == "RESUME": override = None
        elif cmd: override = cmd; until = time.time() + 3.0
link.close(); cap.release(); cv2.destroyAllWindows()
