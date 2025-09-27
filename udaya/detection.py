import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
from collections import deque, defaultdict

model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

INPUT_SIZE = 192  

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.int32)
    img = np.expand_dims(img, axis=0)
    return img

def run_movenet(frame):
    inp = preprocess(frame)
    outputs = movenet(tf.constant(inp, dtype=tf.int32))
    keypoints = outputs['output_0'].numpy()  # shape (1, 1, 17, 3)
    # keypoints[..., :2] are normalized (y,x); last channel is score
    kpts = keypoints[0,0,:,:]
    return kpts  # 17 x 3

# simple person tracker: naive — assumes single person or uses bbox center distance
def keypoints_to_bbox(kpts, frame_w, frame_h):
    coords = kpts[:, :2]
    # coords are normalized (y, x)
    xs = coords[:,1] * frame_w
    ys = coords[:,0] * frame_h
    valid = kpts[:,2] > 0.2
    if valid.sum() == 0:
        return None
    x1, x2 = np.min(xs[valid]), np.max(xs[valid])
    y1, y2 = np.min(ys[valid]), np.max(ys[valid])
    return int(x1), int(y1), int(x2), int(y2)

# buffer per-person features: using a simple single-person approach for starter
BUFFER_SECONDS = 2.0
FPS = 60.0
BUFFER_LEN = int(BUFFER_SECONDS * FPS)

hip_index = 11  
shoulder_left = 5
shoulder_right = 6
hip_left = 11
hip_right = 12

# --- NEW helper functions & parameters for improved detection ---
KEYPOINT_SCORE_THRESH = 0.2

# detection params (tune these on your video)
FALL_VEL_THRESHOLD = 0.7          # normalized units/sec (increase if too many false alarms)
TORSO_ANGLE_THRESH_DEG = 45      # degrees where ~90 is horizontal, 0 vertical
POST_FALL_INACTIVITY_THRESH = 0.02  # normalized motion threshold
POST_FALL_CHECK_SECONDS = 1.0

# buffer holds (ts, mid_hip_y, torso_angle_deg, activity_energy_norm)
feat_buffer = deque(maxlen=BUFFER_LEN)

def compute_midpoint(a, b):
    return (a + b) / 2.0

def compute_torso_angle_deg(kpts):
    # vector from mid-shoulder -> mid-hip. Note kpts: (y,x)
    sh = compute_midpoint(kpts[shoulder_left,:2], kpts[shoulder_right,:2])
    hp = compute_midpoint(kpts[hip_left,:2], kpts[hip_right,:2])
    dy = hp[0] - sh[0]  # positive = hip below shoulder
    dx = hp[1] - sh[1]
    # angle between vector and vertical: atan2(|dx|, |dy|)
    angle_rad = np.arctan2(abs(dx), abs(dy) + 1e-9)
    return np.degrees(angle_rad)  # 0deg = vertical, 90deg = horizontal

def mid_hip_y_normalized(kpts):
    vals = []
    if kpts[hip_left,2] > KEYPOINT_SCORE_THRESH:
        vals.append(kpts[hip_left,0])
    if kpts[hip_right,2] > KEYPOINT_SCORE_THRESH:
        vals.append(kpts[hip_right,0])
    if len(vals) == 0:
        return None
    return float(np.mean(vals))

def compute_activity_energy_norm(kpts_prev, kpts_curr, frame_h, frame_w):
    if kpts_prev is None:
        return 0.0
    disp = np.linalg.norm(((kpts_curr[:,:2] - kpts_prev[:,:2]) * np.array([frame_h, frame_w])), axis=1)
    valid = (kpts_curr[:,2] > KEYPOINT_SCORE_THRESH) & (kpts_prev[:,2] > KEYPOINT_SCORE_THRESH)
    if not valid.any():
        return 0.0
    energy = np.nansum(disp[valid])
    diag = np.sqrt(frame_h**2 + frame_w**2)
    return float(energy / (diag + 1e-9))

# state for previous frame
cap = cv2.VideoCapture(0)
prev_kpts = None
prev_ts = None
fall_state = {'detected': False, 'detected_ts': None}

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    ts = time.time()

    kpts = run_movenet(frame)  # 17x3 (y,x,score)

    # compute features
    mid_hip = mid_hip_y_normalized(kpts)
    torso_ang = compute_torso_angle_deg(kpts)
    activity = compute_activity_energy_norm(prev_kpts, kpts, h, w)

    feat_buffer.append((ts, mid_hip, torso_ang, activity))

    # compute hip velocity (normalized y / sec) using an earlier sample ~0.12-0.3s ago
    hip_vel = None
    if len(feat_buffer) >= 2 and feat_buffer[-1][1] is not None:
        newest_ts, newest_hip, _, _ = feat_buffer[-1]
        # find a previous sample at least 0.12s earlier to smooth noise
        prev_sample = None
        for t_ in reversed(list(feat_buffer)[:-1]):
            if t_[0] <= newest_ts - 0.12:
                prev_sample = t_
                break
        if prev_sample is None:
            prev_sample = feat_buffer[-2]
        prev_ts_, prev_hip, _, _ = prev_sample
        if prev_hip is not None:
            dt = newest_ts - prev_ts_
            if dt > 0:
                hip_vel = (newest_hip - prev_hip) / dt  # normalized units/sec (positive = moving down)

    # FALL detection logic: velocity spike + torso horizontal + post-fall inactivity
    state="ACTIVE"
    if hip_vel is not None:
        vel_check = hip_vel > FALL_VEL_THRESHOLD
        torso_check = torso_ang >= TORSO_ANGLE_THRESH_DEG

        post_fall_ok = False
        if vel_check and torso_check:
            # examine recent buffer for low activity after the spike
            tail = [f for f in feat_buffer if f[0] >= ts - POST_FALL_CHECK_SECONDS]
            if len(tail) > 0:
                avg_activity = np.mean([f[3] for f in tail])
                post_fall_ok = avg_activity < POST_FALL_INACTIVITY_THRESH
            else:
                post_fall_ok = False

        if vel_check or torso_check or post_fall_ok:
            state="INACTIVE"

    # Visualization & debug text
    if state == "INACTIVE":
      cv2.putText(frame, "INACTIVE (FALL)", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    else:
      cv2.putText(frame, "ACTIVE", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    # hip_disp = mid_hip if mid_hip is not None else 0.0
    # hip_vel_disp = hip_vel if hip_vel is not None else 0.0
    # debug_text = f"hip_y={hip_disp:.3f} vel={hip_vel_disp:.3f} ang={torso_ang:.1f} act={activity:.4f}"
    # cv2.putText(frame, debug_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # if detected_fall:
    #     cv2.putText(frame, "FALL DETECTED", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    #     # optionally save frame or log event
    #     # cv2.imwrite("fall_{}.jpg".format(int(ts)), frame)

    # draw keypoints (for debug)
    for i, kp in enumerate(kpts):
        y, x, s = kp
        if s > 0.2:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 3, (0,255,0), -1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_kpts = kpts.copy()
    prev_ts = ts

cap.release()
cv2.destroyAllWindows()
