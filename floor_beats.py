import numpy as np
import sounddevice as sd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import urllib.request
import time
import threading

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_PATH = "pose_landmarker_lite.task"

SR = 44100

# ─────────────────────────────────────────────
# Model Download
# ─────────────────────────────────────────────
def download_model():
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"Downloading Pose model to {POSE_MODEL_PATH}...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("Download complete.")

# ─────────────────────────────────────────────
# Sound Generation
# ─────────────────────────────────────────────
SOUND_DUR = 0.5

def pad_to(wave, n):
    if len(wave) >= n:
        return wave[:n]
    out = np.zeros(n, dtype=np.float32)
    out[:len(wave)] = wave
    return out

def gen_note(freq):
    n = int(SR * SOUND_DUR)
    t = np.linspace(0, SOUND_DUR, n)
    wave = np.sin(2*np.pi*freq*t)*0.5 + np.sin(2*np.pi*freq*2*t)*0.2
    env = np.exp(-4*t) * (1 - np.exp(-200*t))
    return pad_to((wave*env).astype(np.float32), n)

def gen_kick():
    n = int(SR * SOUND_DUR)
    t = np.linspace(0, 0.35, int(SR*0.35))
    freq = 55 + 140*np.exp(-20*t)
    wave = np.sin(np.cumsum(2*np.pi*freq/SR)) * np.exp(-10*t) * 0.8
    return pad_to(wave.astype(np.float32), n)

def gen_snare():
    n = int(SR * SOUND_DUR)
    t = np.linspace(0, 0.25, int(SR*0.25))
    tone = np.sin(2*np.pi*200*t) * np.exp(-20*t) * 0.4
    noise = np.random.uniform(-1,1,len(t)) * np.exp(-15*t) * 0.4
    return pad_to((tone+noise).astype(np.float32), n)

def gen_hihat():
    n = int(SR * SOUND_DUR)
    t = np.linspace(0, 0.08, int(SR*0.08))
    noise = np.random.uniform(-1,1,len(t))
    return pad_to((noise * np.exp(-50*t) * 0.3).astype(np.float32), n)

def gen_clap():
    n = int(SR * SOUND_DUR)
    t = np.linspace(0, 0.15, int(SR*0.15))
    noise = np.random.uniform(-1,1,len(t))
    env = np.exp(-20*t) * (1 + 0.5*np.sin(2*np.pi*100*t))
    return pad_to((noise*env*0.4).astype(np.float32), n)

def gen_bass(freq=55):
    n = int(SR * SOUND_DUR)
    t = np.linspace(0, SOUND_DUR, n)
    wave = np.tanh(np.sin(2*np.pi*freq*t)*3) * 0.45
    env = np.exp(-2.5*t)
    return pad_to((wave*env).astype(np.float32), n)

def gen_bg_loop():
    dur = 2.0
    n = int(SR * dur)
    loop = np.zeros(n, dtype=np.float32)
    for off in [0.0, 1.0]:
        idx = int(off*SR); k = gen_kick()*0.4; end = min(idx+len(k),n)
        loop[idx:end] += k[:end-idx]
    for beat in range(4):
        idx = int(beat*0.5*SR); h = gen_hihat()*0.3; end = min(idx+len(h),n)
        loop[idx:end] += h[:end-idx]
    return loop

TILE_SOUNDS = [
    gen_note(523.25),gen_note(587.33),gen_note(659.25),gen_note(783.99),
    gen_kick(),gen_note(440.00),gen_note(493.88),gen_snare(),
    gen_clap(),gen_note(349.23),gen_note(392.00),gen_bass(65),
    gen_note(261.63),gen_note(293.66),gen_hihat(),gen_bass(82),
]
TILE_NAMES = ["C5","D5","E5","G5","KICK","A4","B4","SNARE",
              "CLAP","F4","G4","BASS","C4","D4","HAT","SUB"]
TILE_COLORS = [
    (255,100,255),(200,100,255),(150,100,255),(100,100,255),
    (50,200,255),(50,255,200),(50,255,100),(100,255,50),
    (200,255,50),(255,200,50),(255,150,50),(255,100,50),
    (255,50,100),(255,50,150),(255,50,200),(255,50,255),
]

# ─────────────────────────────────────────────
# Audio Engine
# ─────────────────────────────────────────────
class AudioEngine:
    def __init__(self):
        self.active = []
        self.bg_loop = gen_bg_loop()
        self.bg_pos = 0
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(channels=1, samplerate=SR, blocksize=512, callback=self._cb)
        self.stream.start()

    def play(self, idx):
        with self.lock:
            self.active.append({"data": TILE_SOUNDS[idx].copy(), "pos": 0})

    def _cb(self, outdata, frames, time_info, status):
        buf = np.zeros(frames, dtype=np.float32)
        pos = self.bg_pos
        rem = frames; wp = 0
        while rem > 0:
            chunk = min(rem, len(self.bg_loop) - pos)
            buf[wp:wp+chunk] += self.bg_loop[pos:pos+chunk]
            pos = (pos+chunk) % len(self.bg_loop)
            wp += chunk; rem -= chunk
        self.bg_pos = pos
        with self.lock:
            alive = []
            for s in self.active:
                d = s["data"]; p = s["pos"]
                chunk = min(frames, len(d)-p)
                if chunk > 0:
                    buf[:chunk] += d[p:p+chunk]
                s["pos"] += chunk
                if s["pos"] < len(d): alive.append(s)
            self.active = alive
        outdata[:, 0] = np.clip(buf, -1.0, 1.0)

    def close(self):
        self.stream.stop(); self.stream.close()


# ─────────────────────────────────────────────
# 3D Perspective Grid
# ─────────────────────────────────────────────
def build_perspective_grid(w, h, rows, cols):
    top_y = h * 0.05
    bottom_y = h
    shrink = 0.3
    top_left = w * shrink
    top_right = w * (1 - shrink)
    bottom_left = 0
    bottom_right = w

    quads = []
    for r in range(rows):
        t0 = r / rows; t1 = (r+1) / rows
        y0 = top_y + (bottom_y - top_y)*t0
        y1 = top_y + (bottom_y - top_y)*t1
        x0l = top_left + (bottom_left - top_left)*t0
        x0r = top_right + (bottom_right - top_right)*t0
        x1l = top_left + (bottom_left - top_left)*t1
        x1r = top_right + (bottom_right - top_right)*t1
        for c in range(cols):
            s0 = c/cols; s1 = (c+1)/cols
            tl = (int(x0l + (x0r-x0l)*s0), int(y0))
            tr = (int(x0l + (x0r-x0l)*s1), int(y0))
            br = (int(x1l + (x1r-x1l)*s1), int(y1))
            bl = (int(x1l + (x1r-x1l)*s0), int(y1))
            quads.append((np.array([tl,tr,br,bl], dtype=np.int32), r, c))
    return quads


# ─────────────────────────────────────────────
# Foot landmark indices in MediaPipe Pose
# ─────────────────────────────────────────────
# Left foot:  27=ankle, 29=heel, 31=foot_index(toe)
# Right foot: 28=ankle, 30=heel, 32=foot_index(toe)
FOOT_LANDMARKS = [27, 28, 29, 30, 31, 32]
FOOT_LABELS = {27:"L_ANKLE",28:"R_ANKLE",29:"L_HEEL",30:"R_HEEL",31:"L_TOE",32:"R_TOE"}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    download_model()

    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=2,  # Track up to 2 people
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    engine = AudioEngine()

    ROWS, COLS = 4, 4
    COOLDOWN_FRAMES = 12
    tile_cooldowns = [[0]*COLS for _ in range(ROWS)]
    tile_flash = [[0.0]*COLS for _ in range(ROWS)]
    # Track which tile each foot was in last frame (to trigger on ENTRY only)
    prev_foot_tiles = {}  # landmark_idx -> tile_idx or -1

    quads = None

    print("=" * 50)
    print("  AR Floor Beat Tiles (Foot Tracking)")
    print("=" * 50)
    print("Camera should see your feet.")
    print("Step on tiles to play beats!")
    print("  Q - Quit")
    print("=" * 50)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if quads is None:
            quads = build_perspective_grid(w, h, ROWS, COLS)

        # Detect Pose
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        display = cv2.convertScaleAbs(frame, alpha=0.4, beta=5)
        overlay = np.zeros_like(display, dtype=np.uint8)

        # Tick cooldowns
        for r in range(ROWS):
            for c in range(COLS):
                if tile_cooldowns[r][c] > 0:
                    tile_cooldowns[r][c] -= 1

        # Collect all foot points from all detected poses
        foot_points = []  # list of (px, py, landmark_key)
        if results.pose_landmarks:
            for pose_idx, landmarks in enumerate(results.pose_landmarks):
                for lm_idx in FOOT_LANDMARKS:
                    lm = landmarks[lm_idx]
                    # Only use if visibility is decent
                    if lm.visibility > 0.3:
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        foot_points.append((px, py, f"pose{pose_idx}_lm{lm_idx}"))

                        # Draw foot landmark
                        cv2.circle(overlay, (px, py), 10, (0, 255, 0), -1, cv2.LINE_AA)
                        cv2.circle(overlay, (px, py), 14, (255, 255, 255), 2, cv2.LINE_AA)

        # Check each foot point against tiles
        current_foot_tiles = {}
        for px, py, key in foot_points:
            found_tile = -1
            for quad_pts, r, c in quads:
                idx = r * COLS + c
                result = cv2.pointPolygonTest(quad_pts, (float(px), float(py)), False)
                if result >= 0:
                    found_tile = idx
                    # Trigger on ENTRY: foot was not in this tile last frame
                    prev_tile = prev_foot_tiles.get(key, -1)
                    if prev_tile != idx and tile_cooldowns[r][c] <= 0:
                        engine.play(idx)
                        tile_cooldowns[r][c] = COOLDOWN_FRAMES
                        tile_flash[r][c] = 1.0
                    break
            current_foot_tiles[key] = found_tile
        prev_foot_tiles.clear()
        prev_foot_tiles.update(current_foot_tiles)

        # Draw tiles
        for quad_pts, r, c in quads:
            idx = r * COLS + c
            color = TILE_COLORS[idx]

            if tile_flash[r][c] > 0:
                flash_c = tuple(int(ch * tile_flash[r][c]) for ch in color)
                cv2.fillConvexPoly(overlay, quad_pts, flash_c)
                tile_flash[r][c] = max(0, tile_flash[r][c] - 0.07)

            bw = 3 if tile_cooldowns[r][c] > 0 else 2
            bc = (255,255,255) if tile_cooldowns[r][c] > 0 else color
            cv2.polylines(overlay, [quad_pts], True, bc, bw, cv2.LINE_AA)

            cx = int(np.mean(quad_pts[:,0]))
            cy = int(np.mean(quad_pts[:,1]))
            tile_h = quad_pts[3][1] - quad_pts[0][1]
            scale = max(0.4, min(1.0, tile_h / 180))
            cv2.putText(overlay, TILE_NAMES[idx], (cx-20, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

        # Composite
        display = cv2.addWeighted(overlay, 0.8, display, 1.0, 0)

        # HUD
        cv2.rectangle(display, (0,0), (w,40), (20,20,30), -1)
        n_feet = len(foot_points)
        cv2.putText(display, "AR FLOOR BEATS (Foot Tracking)", (20,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,255), 2, cv2.LINE_AA)
        cv2.putText(display, f"Feet detected: {n_feet} | Q: Quit", (550,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        cv2.imshow("AR Floor Beat Tiles", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    engine.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
