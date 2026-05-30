"""
posture_overlay.py — Always-on-top floating posture indicator.

A small draggable circle sits in the corner of your screen and changes
color in real time based on your posture — green, orange, or red.
You can work on your PC as normal; the indicator stays on top of everything.

How it works
------------
  • A background thread runs the webcam + marker detection + SVM inference
    (the exact same logic as real_time_inference.py).
  • The main thread runs a tiny Tkinter window — borderless, always-on-top,
    semi-transparent — that reads the latest prediction and updates its color.
  • The two threads communicate through a thread-safe queue.

Controls
--------
  Left-click + drag   — move the indicator anywhere on screen
  Right-click         — open the context menu (show stats / quit)
  Double-click        — toggle expanded stats panel

Usage
-----
    python scripts/posture_overlay.py

Prerequisites (same as real_time_inference.py):
  1. feature_extraction.py   — collect training data
  2. data_augmentation.py    — expand dataset
  3. train_classifier.py     — train and save the model
  Then run this script.
"""

import cv2
import numpy as np
import joblib
import os
import sys
import time
import threading
import queue
from collections import deque
from datetime import datetime
import tkinter as tk
from tkinter import font as tkfont

# ── Allow imports from the project root ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_utils import detect_red_markers, extract_features
from utils.config import (
    MODEL_SVM_PATH, FEATURE_COLUMNS, LABEL_NAMES, OUTPUT_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Colors for each posture label  (Tkinter hex strings)
TK_COLORS = {
    0: "#22c55e",   # green  — Good Posture
    1: "#f97316",   # orange — Medium Risk
    2: "#ef4444",   # red    — High Risk
    None: "#6b7280" # grey   — no markers detected
}

LABEL_TEXT = {
    0: "Good",
    1: "Medium",
    2: "High Risk",
    None: "Searching…"
}

# Indicator size (compact circle)
INDICATOR_SIZE = 72          # px — diameter of the floating circle
SMOOTHING_WINDOW = 8         # frames to smooth over (majority vote)
UPDATE_INTERVAL_MS = 80      # how often the Tkinter window refreshes (ms)

# ─────────────────────────────────────────────────────────────────────────────
#  Inference Thread
# ─────────────────────────────────────────────────────────────────────────────

class InferenceThread(threading.Thread):
    """
    Runs continuously in the background.
    Captures webcam frames, detects markers, extracts features,
    and pushes the predicted label into result_queue.
    """

    def __init__(self, result_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.stop_event   = stop_event
        self.recent_labels = deque(maxlen=SMOOTHING_WINDOW)
        self.session_counts = {0: 0, 1: 0, 2: 0}
        self.start_time     = time.time()
        self.current_feats  = None   # latest extracted feature dict (for display)

    # ── smoothed majority-vote prediction ────────────────────────────────────
    def _smooth(self, label):
        self.recent_labels.append(label)
        return max(set(self.recent_labels), key=self.recent_labels.count)

    def run(self):
        # Load model
        if not os.path.exists(MODEL_SVM_PATH):
            self.result_queue.put(("error", "No model found. Run train_classifier.py first."))
            return

        model = joblib.load(MODEL_SVM_PATH)

        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.result_queue.put(("error", "Cannot open webcam."))
            return

        print("[Inference] Background thread started.")

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            points, _ = detect_red_markers(frame)
            feats      = extract_features(points) if len(points) >= 3 else None

            if feats:
                self.current_feats = feats
                feat_vec  = np.array([[feats[col] for col in FEATURE_COLUMNS]])
                raw_label = int(model.predict(feat_vec)[0])
                label     = self._smooth(raw_label)
                self.session_counts[label] += 1

                # Try to get confidence
                try:
                    proba      = model.predict_proba(feat_vec)[0]
                    confidence = float(proba[raw_label])
                except AttributeError:
                    confidence = None

                self.result_queue.put(("label", label, confidence, dict(self.session_counts)))
            else:
                # Not enough markers visible
                self.result_queue.put(("no_markers", len(points)))

        cap.release()
        print("[Inference] Background thread stopped.")

    def get_elapsed(self):
        elapsed = int(time.time() - self.start_time)
        m, s = divmod(elapsed, 60)
        return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
#  Floating Indicator (Tkinter UI)
# ─────────────────────────────────────────────────────────────────────────────

class PostureOverlay:
    """
    A small, always-on-top, borderless Tkinter window.
    Shows a colored circle + short label that reflects current posture.
    """

    def __init__(self, root: tk.Tk, inference: InferenceThread, result_queue: queue.Queue):
        self.root         = root
        self.inference    = inference
        self.result_queue = result_queue

        # Current state
        self.current_label      = None
        self.current_confidence = None
        self.session_counts     = {0: 0, 1: 0, 2: 0}
        self.markers_found      = 0
        self.expanded           = False   # toggle stats panel on double-click

        # ── Window setup ─────────────────────────────────────────────────────
        self.root.overrideredirect(True)          # no title bar / borders
        self.root.attributes("-topmost", True)    # always on top
        self.root.attributes("-alpha", 0.92)      # slight transparency
        self.root.configure(bg="#0f0f0f")         # fallback bg
        self.root.resizable(False, False)

        # Starting position — top-right corner (20px from edge)
        screen_w = self.root.winfo_screenwidth()
        self.root.geometry(f"{INDICATOR_SIZE}x{INDICATOR_SIZE}+"
                           f"{screen_w - INDICATOR_SIZE - 20}+20")

        # ── Canvas (draws the circle) ─────────────────────────────────────────
        self.canvas = tk.Canvas(
            self.root,
            width=INDICATOR_SIZE,
            height=INDICATOR_SIZE,
            bg="#0f0f0f",
            highlightthickness=0
        )
        self.canvas.pack()

        # Circle items (drawn once, recolored on update)
        pad = 5
        self.circle = self.canvas.create_oval(
            pad, pad,
            INDICATOR_SIZE - pad, INDICATOR_SIZE - pad,
            fill=TK_COLORS[None], outline="#ffffff", width=2
        )

        # Label text inside the circle
        try:
            self.label_font = tkfont.Font(family="Helvetica Neue", size=9, weight="bold")
        except Exception:
            self.label_font = tkfont.Font(family="Helvetica", size=9, weight="bold")

        self.text_item = self.canvas.create_text(
            INDICATOR_SIZE // 2, INDICATOR_SIZE // 2,
            text="…",
            fill="#ffffff",
            font=self.label_font,
            justify="center"
        )

        # ── Stats panel (hidden by default) ──────────────────────────────────
        self.stats_frame = tk.Frame(self.root, bg="#1a1a1a", padx=10, pady=8)
        # (not packed yet — toggled on double-click)

        self.stats_labels = {}
        rows = [
            ("session", "Session: --:--"),
            ("good",    "🟢 Good:     0%"),
            ("medium",  "🟡 Medium:   0%"),
            ("high",    "🔴 High:     0%"),
            ("conf",    "Confidence: --"),
            ("markers", "Markers: 0/3"),
        ]
        try:
            sf = tkfont.Font(family="Courier New", size=9)
        except Exception:
            sf = tkfont.Font(family="Courier", size=9)

        for key, default_text in rows:
            lbl = tk.Label(
                self.stats_frame,
                text=default_text,
                bg="#1a1a1a",
                fg="#d1d5db",
                font=sf,
                anchor="w"
            )
            lbl.pack(fill="x")
            self.stats_labels[key] = lbl

        # ── Drag support ──────────────────────────────────────────────────────
        self._drag_x = 0
        self._drag_y = 0
        self.canvas.bind("<ButtonPress-1>",   self._on_drag_start)
        self.canvas.bind("<B1-Motion>",       self._on_drag_motion)
        self.canvas.bind("<Double-Button-1>", self._toggle_stats)
        self.canvas.bind("<Button-3>",        self._show_context_menu)

        # ── Context menu ──────────────────────────────────────────────────────
        self.menu = tk.Menu(self.root, tearoff=0, bg="#1a1a1a", fg="#f9fafb",
                            activebackground="#374151", activeforeground="#ffffff",
                            font=("Helvetica", 10))
        self.menu.add_command(label="📊  Toggle Stats",   command=self._toggle_stats)
        self.menu.add_separator()
        self.menu.add_command(label="❌  Quit",           command=self._quit)

        # ── Start polling ─────────────────────────────────────────────────────
        self._poll()

    # ── Drag ─────────────────────────────────────────────────────────────────

    def _on_drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag_motion(self, event):
        x = self.root.winfo_x() + (event.x - self._drag_x)
        y = self.root.winfo_y() + (event.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")

    # ── Toggle stats panel ────────────────────────────────────────────────────

    def _toggle_stats(self, event=None):
        self.expanded = not self.expanded
        if self.expanded:
            self.stats_frame.pack(fill="x")
            # Resize window to fit stats
            self.root.geometry(f"{INDICATOR_SIZE + 80}x{INDICATOR_SIZE + 110}"
                               f"+{self.root.winfo_x()}+{self.root.winfo_y()}")
        else:
            self.stats_frame.pack_forget()
            self.root.geometry(f"{INDICATOR_SIZE}x{INDICATOR_SIZE}"
                               f"+{self.root.winfo_x()}+{self.root.winfo_y()}")

    # ── Context menu ──────────────────────────────────────────────────────────

    def _show_context_menu(self, event):
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    # ── Quit ──────────────────────────────────────────────────────────────────

    def _quit(self):
        stop_event.set()
        self._print_summary()
        self.root.destroy()

    # ── Main polling loop ────────────────────────────────────────────────────

    def _poll(self):
        """
        Called every UPDATE_INTERVAL_MS milliseconds.
        Drains the result queue and updates the UI.
        """
        try:
            while True:
                msg = self.result_queue.get_nowait()

                if msg[0] == "label":
                    _, label, conf, counts = msg
                    self.current_label      = label
                    self.current_confidence = conf
                    self.session_counts     = counts
                    self.markers_found      = 3

                elif msg[0] == "no_markers":
                    self.markers_found = msg[1]

                elif msg[0] == "error":
                    print(f"[ERROR] {msg[1]}")
                    self._quit()
                    return

        except queue.Empty:
            pass

        # ── Redraw circle ─────────────────────────────────────────────────────
        color = TK_COLORS.get(self.current_label, TK_COLORS[None])
        self.canvas.itemconfig(self.circle, fill=color)

        short_text = LABEL_TEXT.get(self.current_label, "…")
        # Wrap text: "Good" fits, "Medium" fits, "High Risk" → two lines
        display = short_text.replace(" ", "\n") if " " in short_text else short_text
        self.canvas.itemconfig(self.text_item, text=display)

        # ── Update stats panel (if visible) ──────────────────────────────────
        if self.expanded:
            total = sum(self.session_counts.values()) or 1

            self.stats_labels["session"].config(
                text=f"Session: {self.inference.get_elapsed()}"
            )
            self.stats_labels["good"].config(
                text=f"🟢 Good:   {100*self.session_counts.get(0,0)/total:4.0f}%"
            )
            self.stats_labels["medium"].config(
                text=f"🟡 Medium: {100*self.session_counts.get(1,0)/total:4.0f}%"
            )
            self.stats_labels["high"].config(
                text=f"🔴 High:   {100*self.session_counts.get(2,0)/total:4.0f}%"
            )
            if self.current_confidence is not None:
                self.stats_labels["conf"].config(
                    text=f"Confidence: {self.current_confidence*100:.0f}%"
                )
            self.stats_labels["markers"].config(
                text=f"Markers: {self.markers_found}/3"
            )

        # Schedule next poll
        self.root.after(UPDATE_INTERVAL_MS, self._poll)

    # ── Session summary (printed on quit) ────────────────────────────────────

    def _print_summary(self):
        total = sum(self.session_counts.values()) or 1
        print("\n=== Session Summary ===")
        print(f"Duration : {self.inference.get_elapsed()}")
        for lbl in [0, 1, 2]:
            cnt = self.session_counts.get(lbl, 0)
            pct = 100 * cnt / total
            print(f"  {LABEL_NAMES[lbl]:<16}: {cnt:>5} frames  ({pct:.1f}%)")
        print(f"  Total frames: {total}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Posture Overlay ===")
    print("A floating indicator will appear in the top-right corner.")
    print("  * Left-click + drag  -- move it anywhere")
    print("  * Double-click       -- toggle stats panel")
    print("  * Right-click        -- menu (stats / quit)\n")

    # Shared communication objects
    result_queue = queue.Queue(maxsize=30)
    stop_event   = threading.Event()

    # Start inference in the background
    inference = InferenceThread(result_queue, stop_event)
    inference.start()

    # Start Tkinter UI on the main thread
    root = tk.Tk()
    app  = PostureOverlay(root, inference, result_queue)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
