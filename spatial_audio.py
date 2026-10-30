# ====================================================================
# spatial_audio.py - surround sound with your hands: virtual speakers float
# in front of the webcam; pinch one to grab it and drag it around your head,
# and the music really moves - the engine fakes 3D hearing by panning and
# slightly delaying the sound reaching each ear.
# ====================================================================
import numpy as np
import sounddevice as sd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import time
import threading

MODEL_PATH = "hand_landmarker.task"

# ─────────────────────────────────────────────
# Audio Source Generation
# ─────────────────────────────────────────────
SR = 44100

# # Pre-build the looping drum track (kick/snare/hihat stamped in a buffer).
def generate_beat_loop():
    """Generate a punchy looping beat pattern (~2 seconds)."""
    duration = 2.0
    n = int(SR * duration)
    t = np.linspace(0, duration, n)
    loop = np.zeros(n, dtype=np.float32)

    # Kick at beat 1, 3
    for offset in [0.0, 1.0]:
        idx = int(offset * SR)
        length = min(int(0.3 * SR), n - idx)
        tk = np.linspace(0, 0.3, length)
        freq = 55 + 120 * np.exp(-18 * tk)
        phase = np.cumsum(2 * np.pi * freq / SR)
        kick = np.sin(phase) * np.exp(-10 * tk) * 0.7
        loop[idx:idx+length] += kick

    # Hi-hat at every 0.25s
    for beat in range(8):
        offset = beat * 0.25
        idx = int(offset * SR)
        length = min(int(0.05 * SR), n - idx)
        tk = np.linspace(0, 0.05, length)
        hh = np.random.uniform(-1, 1, length) * np.exp(-60 * tk) * 0.15
        loop[idx:idx+length] += hh

    # Snare at beat 2, 4
    for offset in [0.5, 1.5]:
        idx = int(offset * SR)
        length = min(int(0.2 * SR), n - idx)
        tk = np.linspace(0, 0.2, length)
        tone = np.sin(2 * np.pi * 200 * tk) * np.exp(-20 * tk) * 0.3
        noise = np.random.uniform(-1, 1, length) * np.exp(-15 * tk) * 0.25
        loop[idx:idx+length] += tone + noise

    return loop

# # Pre-build the looping melody line.
def generate_melody_loop():
    """Generate a simple melodic synth loop."""
    duration = 2.0
    n = int(SR * duration)
    loop = np.zeros(n, dtype=np.float32)
    notes = [523.25, 659.25, 783.99, 659.25]  # C5 E5 G5 E5

    for i, freq in enumerate(notes):
        offset = i * 0.5
        idx = int(offset * SR)
        length = min(int(0.4 * SR), n - idx)
        tk = np.linspace(0, 0.4, length)
        wave = np.sin(2 * np.pi * freq * tk) * 0.3 + np.sin(2 * np.pi * freq * 2 * tk) * 0.1
        env = np.exp(-3 * tk) * (1 - np.exp(-100 * tk))
        loop[idx:idx+length] += wave * env

    return loop

# # Pre-build the looping bass line.
def generate_bass_loop():
    """Generate a deep bass loop."""
    duration = 2.0
    n = int(SR * duration)
    loop = np.zeros(n, dtype=np.float32)
    notes = [65.41, 65.41, 82.41, 73.42]  # C2 C2 E2 D2

    for i, freq in enumerate(notes):
        offset = i * 0.5
        idx = int(offset * SR)
        length = min(int(0.45 * SR), n - idx)
        tk = np.linspace(0, 0.45, length)
        wave = np.sin(2 * np.pi * freq * tk)
        wave = np.tanh(wave * 3) * 0.4
        env = np.exp(-2 * tk)
        loop[idx:idx+length] += wave * env

    return loop


# ─────────────────────────────────────────────
# Binaural HRTF Spatial Engine
# ─────────────────────────────────────────────
# # One floating sound source: its on-screen position maps to a direction
# # around your head, and it owns one of the music loops.
class SpatialSource:
    """A virtual speaker placed in AR space."""
    def __init__(self, x, y, name, color, loop):
        self.x = x  # Normalized 0..1 (left..right)
        self.y = y  # Normalized 0..1 (top..bottom)
        self.name = name
        self.color = color
        self.loop = loop
        self.pos = 0  # Playback position in loop
        self.grabbed = False
        self.muted = False
        self.pulse = 0.0

    def get_azimuth(self):
        """Convert x position (0..1) to azimuth angle (-90 to +90 degrees)."""
        return (self.x - 0.5) * 180  # -90 (left) to +90 (right)

    def get_elevation(self):
        """Convert y position to subtle elevation."""
        return (0.5 - self.y) * 40  # -20 (bottom) to +20 (top)

    def get_distance(self):
        """Map y position to distance (closer at bottom, farther at top)."""
        return 0.5 + self.y * 1.5  # 0.5m to 2.0m

# # The 3D-sound mixer: each source's sound is panned and slightly delayed
# # differently per ear. Your brain locates sound mostly by loudness +
# # arrival-time differences between the ears - that's the whole trick.
class SpatialAudioEngine:
    """Binaural audio engine using ITD + ILD for 3D spatial audio."""
    HEAD_RADIUS = 0.0875  # ~8.75cm, average human head radius
    SPEED_OF_SOUND = 343.0

    def __init__(self, sources):
        self.sources = sources
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            channels=2,  # STEREO for binaural
            samplerate=SR,
            blocksize=1024,
            callback=self._callback
        )
        self.stream.start()

    def _apply_hrtf(self, mono_chunk, azimuth_deg, distance):
        """Apply simplified HRTF: ITD (time delay) + ILD (level difference)."""
        n = len(mono_chunk)
        azimuth_rad = math.radians(azimuth_deg)

        # 1. ILD: Interaural Level Difference
        #    Ear on the same side as source gets more volume
        #    Simple model: cos/sin based panning with head shadow
        pan = math.sin(azimuth_rad)  # -1 (full left) to +1 (full right)
        gain_left = math.sqrt(max(0, 0.5 * (1 - pan)))
        gain_right = math.sqrt(max(0, 0.5 * (1 + pan)))

        # Head shadow: frequencies above ~1500Hz are attenuated on the far side
        # Approximate with an overall extra attenuation
        shadow_factor = 0.3 * abs(pan)
        if pan > 0:  # Source on right, shadow left ear
            gain_left *= (1 - shadow_factor)
        else:
            gain_right *= (1 - shadow_factor)

        # 2. ITD: Interaural Time Difference
        #    Woodworth formula: ITD = (r/c)(sin(θ) + θ)
        itd_seconds = (self.HEAD_RADIUS / self.SPEED_OF_SOUND) * (abs(math.sin(azimuth_rad)) + abs(azimuth_rad))
        itd_samples = int(itd_seconds * SR)
        itd_samples = min(itd_samples, 30)  # Cap at ~0.7ms

        # 3. Distance attenuation (inverse square law, clamped)
        dist_gain = 1.0 / max(0.3, distance)
        dist_gain = min(1.0, dist_gain)

        # Build stereo output
        left = np.zeros(n + itd_samples, dtype=np.float32)
        right = np.zeros(n + itd_samples, dtype=np.float32)

        if pan >= 0:
            # Source is on the right: delay left ear
            right[:n] = mono_chunk * gain_right * dist_gain
            left[itd_samples:itd_samples+n] = mono_chunk * gain_left * dist_gain
        else:
            # Source is on the left: delay right ear
            left[:n] = mono_chunk * gain_left * dist_gain
            right[itd_samples:itd_samples+n] = mono_chunk * gain_right * dist_gain

        # Trim to original length
        return left[:n], right[:n]

    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        with self.lock:
            for src in self.sources:
                if src.muted:
                    continue

                # Get chunk from loop
                loop = src.loop
                loop_len = len(loop)
                mono = np.zeros(frames, dtype=np.float32)

                pos = src.pos
                remaining = frames
                write_pos = 0
                while remaining > 0:
                    chunk_size = min(remaining, loop_len - pos)
                    mono[write_pos:write_pos+chunk_size] = loop[pos:pos+chunk_size]
                    pos = (pos + chunk_size) % loop_len
                    write_pos += chunk_size
                    remaining -= chunk_size
                src.pos = pos

                # Apply HRTF spatialization
                azimuth = src.get_azimuth()
                distance = src.get_distance()
                left, right = self._apply_hrtf(mono, azimuth, distance)

                outdata[:, 0] += left
                outdata[:, 1] += right

        # Soft clip
        outdata[:] = np.tanh(outdata)

    def close(self):
        self.stream.stop()
        self.stream.close()


# ─────────────────────────────────────────────
# CLAHE Preprocessor (reuse from drums)
# ─────────────────────────────────────────────
# # Lighting fix-up (CLAHE) so hand tracking survives bad lighting.
class FramePreprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def process(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found. Run hand_magic.py first to download it.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)
    preprocessor = FramePreprocessor()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create 3 virtual speakers with different sounds
    sources = [
        SpatialSource(0.25, 0.4, "DRUMS", (50, 200, 255), generate_beat_loop()),
        SpatialSource(0.5, 0.3, "MELODY", (255, 150, 50), generate_melody_loop()),
        SpatialSource(0.75, 0.4, "BASS", (50, 255, 100), generate_bass_loop()),
    ]

    engine = SpatialAudioEngine(sources)

    print("=" * 50)
    print("  AR Spatial Audio Placer")
    print("=" * 50)
    print("Controls:")
    print("  PINCH (thumb+index) near a speaker to GRAB it")
    print("  DRAG it around to change where the sound comes from")
    print("  RELEASE pinch to DROP it in place")
    print("  Press 1/2/3 to MUTE/UNMUTE individual speakers")
    print("  Press Q to Quit")
    print("  USE HEADPHONES for best 3D audio effect!")
    print("=" * 50)

    grabbed_source = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        processed = preprocessor.process(frame)
        display = cv2.convertScaleAbs(frame, alpha=0.45, beta=5)
        overlay = np.zeros_like(display, dtype=np.uint8)

        # Detect hand
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        pinch_active = False
        pinch_x, pinch_y = 0, 0

        if results.hand_landmarks:
            lm = results.hand_landmarks[0]
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)

            dist = math.hypot(ix - tx, iy - ty)
            pinch_x = (ix + tx) // 2
            pinch_y = (iy + ty) // 2

            if dist < 45:
                pinch_active = True
                cv2.circle(overlay, (pinch_x, pinch_y), 20, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(overlay, (pinch_x, pinch_y), 5, (255, 255, 255), -1)
            else:
                cv2.circle(overlay, (ix, iy), 8, (200, 200, 200), -1)
                cv2.circle(overlay, (tx, ty), 8, (200, 200, 200), -1)

        with engine.lock:
            # Grab / drag logic
            if pinch_active:
                if grabbed_source is None:
                    # Check if pinch is near any source
                    for src in sources:
                        sx = int(src.x * w)
                        sy = int(src.y * h)
                        if math.hypot(pinch_x - sx, pinch_y - sy) < 80:
                            grabbed_source = src
                            src.grabbed = True
                            break

                if grabbed_source is not None:
                    grabbed_source.x = max(0.05, min(0.95, pinch_x / w))
                    grabbed_source.y = max(0.05, min(0.95, pinch_y / h))
            else:
                if grabbed_source is not None:
                    grabbed_source.grabbed = False
                    grabbed_source = None

            # Draw speakers
            t = time.time()
            for i, src in enumerate(sources):
                sx = int(src.x * w)
                sy = int(src.y * h)
                color = src.color

                if src.muted:
                    dim = tuple(int(c * 0.2) for c in color)
                    cv2.circle(overlay, (sx, sy), 35, dim, 2, cv2.LINE_AA)
                    cv2.putText(overlay, f"{src.name} (MUTED)", (sx - 50, sy + 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
                    continue

                # Pulsing glow rings
                pulse = 0.5 + 0.5 * math.sin(t * 4 + i * 2)
                r1 = 35 + int(pulse * 15)
                r2 = 55 + int(pulse * 20)
                r3 = 75 + int(pulse * 25)

                cv2.circle(overlay, (sx, sy), r3, tuple(int(c * 0.15) for c in color), -1, cv2.LINE_AA)
                cv2.circle(overlay, (sx, sy), r2, tuple(int(c * 0.3) for c in color), -1, cv2.LINE_AA)
                cv2.circle(overlay, (sx, sy), r1, tuple(int(c * 0.6) for c in color), -1, cv2.LINE_AA)
                cv2.circle(overlay, (sx, sy), 30, color, -1, cv2.LINE_AA)

                # Speaker icon (concentric arcs)
                cv2.circle(overlay, (sx, sy), 12, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(overlay, (sx, sy), 5, (255, 255, 255), -1)

                # Sound wave arcs emanating
                for wave_r in range(1, 4):
                    wave_radius = 30 + wave_r * 25 + int(pulse * 10)
                    arc_alpha = max(0, 1.0 - wave_r * 0.3) * pulse
                    arc_color = tuple(int(c * arc_alpha) for c in color)
                    cv2.ellipse(overlay, (sx, sy), (wave_radius, wave_radius), 0, -60, 60, arc_color, 2, cv2.LINE_AA)
                    cv2.ellipse(overlay, (sx, sy), (wave_radius, wave_radius), 0, 120, 240, arc_color, 2, cv2.LINE_AA)

                if src.grabbed:
                    cv2.circle(overlay, (sx, sy), r2 + 10, (0, 255, 255), 3, cv2.LINE_AA)

                # Label
                label = f"{src.name} [{i+1}]"
                cv2.putText(overlay, label, (sx - 40, sy + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # Direction indicator (azimuth arrow on HUD)
                az = src.get_azimuth()
                arrow_x = int(w // 2 + (az / 90) * (w // 2 - 50))
                arrow_y = h - 30
                cv2.circle(overlay, (arrow_x, arrow_y), 8, color, -1)

        # Composite
        display = cv2.addWeighted(overlay, 0.9, display, 1.0, 0)

        # HUD
        cv2.rectangle(display, (0, 0), (w, 60), (20, 20, 30), -1)
        cv2.putText(display, "AR SPATIAL AUDIO", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 50), 3, cv2.LINE_AA)
        cv2.putText(display, "Pinch & Drag speakers | 1/2/3: Mute | Headphones recommended!",
                    (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Bottom direction bar
        cv2.rectangle(display, (0, h - 50), (w, h), (20, 20, 30), -1)
        cv2.putText(display, "LEFT", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(display, "CENTER", (w // 2 - 30, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(display, "RIGHT", (w - 80, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.line(display, (50, h - 40), (w - 50, h - 40), (80, 80, 80), 2)

        cv2.imshow("AR Spatial Audio", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('1'):
            with engine.lock:
                sources[0].muted = not sources[0].muted
        elif key == ord('2'):
            with engine.lock:
                sources[1].muted = not sources[1].muted
        elif key == ord('3'):
            with engine.lock:
                sources[2].muted = not sources[2].muted

    engine.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
