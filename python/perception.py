"""Turn YOLOv8 detections into the 12 sensor features the accelerator expects.
Distance is a rough pinhole estimate: dist = real_height * focal_px / box_height_px (calibrate FOCAL_PX)."""
FOCAL_PX = 600.0
REAL_HEIGHT_M = {"person": 1.7, "bicycle": 1.1, "motorcycle": 1.2, "car": 1.5, "bus": 3.0, "truck": 3.0}
MAX_D = 30.0

def estimate_distance(cls, box_h_px):
    h = REAL_HEIGHT_M.get(cls)
    if h is None or box_h_px <= 1: return None
    return max(0.5, min(MAX_D, h * FOCAL_PX / box_h_px))

def scene_from_detections(dets, frame_w, speed=30.0, wet=0, lane=0.0):
    """dets: list of (class_name, x1, y1, x2, y2, conf). Returns a scene dict (see model.encode)."""
    front = left = right = MAX_D
    ped = 0
    for cls, x1, y1, x2, y2, conf in dets:
        d = estimate_distance(cls, y2 - y1)
        if d is None: continue
        if cls == "person": ped = 1
        fx = ((x1 + x2) / 2.0) / float(frame_w)
        if fx < 0.33: left = min(left, d)
        elif fx > 0.67: right = min(right, d)
        else: front = min(front, d)
    d = [front, min(front, right), right, right, MAX_D, left, left, min(front, left)]   # d0..d7
    s = {"d%d" % i: float(v) for i, v in enumerate(d)}
    s.update(speed=float(speed), ped=int(ped), wet=int(wet), lane=float(lane))
    return s
