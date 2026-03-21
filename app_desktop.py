"""
Meat Freshness Classifier — Desktop Inference App (Tkinter)

Provides a simple graphical interface to classify meat images as Fresh or Spoiled
using the best trained model.

Two input modes:
  1. Upload Image — open any image file from disk (JPEG, PNG, WEBP, AVIF, BMP, etc.)
  2. Capture from Camera — grab a live frame from the default webcam (requires
     `pip install opencv-python`)

All preprocessing (resize to 224×224, white-background compositing, ImageNet
normalisation) is handled internally — the user just picks an image.

Usage:
    python app_desktop.py
"""

from __future__ import annotations

import os
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, messagebox
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

# ── Lazy-load TensorFlow and model (expensive) ───────────────────────────────
_model = None
_model_name: Optional[str] = None


def _get_model():
    global _model, _model_name
    if _model is None:
        from phase3_inference import _find_best_model_name, load_model
        _model_name = _find_best_model_name()
        if _model_name is None:
            raise RuntimeError(
                "No trained model found.\n"
                "Run 'python phase3_evaluation.py --best' first."
            )
        _model = load_model(_model_name)
    return _model, _model_name


def _classify_pil(pil_img: Image.Image):
    """Run classification on a PIL Image. Returns list of (label, probability)."""
    from phase3_inference import predict

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Composite transparent images onto white background
        if pil_img.mode in ("RGBA", "P", "PA", "LA"):
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            alpha = pil_img.convert("RGBA").split()[3]
            bg.paste(pil_img.convert("RGB"), mask=alpha)
            pil_img = bg
        else:
            pil_img = pil_img.convert("RGB")

        pil_img.save(tmp_path, "JPEG", quality=95)
        model, _ = _get_model()
        results = predict(model, [Path(tmp_path)])
        probs = results[0]["probabilities"]
        return sorted(probs.items(), key=lambda x: -x[1])   # high-confidence first
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _thumbnail(pil_img: Image.Image, max_size: int = 380) -> ImageTk.PhotoImage:
    """Resize for display, preserving aspect ratio."""
    img = pil_img.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


# ── Main GUI ──────────────────────────────────────────────────────────────────

class App(tk.Tk):
    # Colour palette
    BG             = "#1e1e2e"
    PANEL          = "#2a2a3e"
    ACCENT_FRESH   = "#4ade80"   # green
    ACCENT_SPOILED = "#f87171"   # red
    ACCENT_NEUTRAL = "#94a3b8"   # grey
    TEXT           = "#e2e8f0"

    def __init__(self):
        super().__init__()
        self.title("Meat Freshness Classifier")
        self.resizable(False, False)
        self.configure(bg=self.BG)

        self._current_pil: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None   # keep reference

        self._build_ui()
        self._status("Ready — upload an image or capture from camera.")

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        title_f = font.Font(family="Segoe UI", size=16, weight="bold")
        lbl_f   = font.Font(family="Segoe UI", size=11)
        btn_f   = font.Font(family="Segoe UI", size=11, weight="bold")
        res_f   = font.Font(family="Segoe UI", size=22, weight="bold")
        sub_f   = font.Font(family="Segoe UI", size=10)

        # Title
        tk.Label(self, text="🥩  Meat Freshness Classifier",
                 font=title_f, bg=self.BG, fg=self.TEXT).pack(pady=(18, 4))

        # Image preview panel
        self.preview_frame = tk.Frame(self, bg=self.PANEL,
                                      width=400, height=400,
                                      relief="flat", bd=1)
        self.preview_frame.pack(padx=20, pady=8)
        self.preview_frame.pack_propagate(False)

        self.canvas = tk.Label(self.preview_frame, bg=self.PANEL,
                               text="No image loaded",
                               font=lbl_f, fg=self.ACCENT_NEUTRAL)
        self.canvas.pack(expand=True)

        # Buttons
        btn_frame = tk.Frame(self, bg=self.BG)
        btn_frame.pack(pady=6)

        tk.Button(btn_frame, text="📂  Upload Image",
                  font=btn_f, bg="#3b3b55", fg=self.TEXT,
                  activebackground="#4a4a70", cursor="hand2",
                  relief="flat", padx=16, pady=8,
                  command=self._upload).pack(side="left", padx=8)

        tk.Button(btn_frame, text="📷  Capture from Camera",
                  font=btn_f, bg="#3b3b55", fg=self.TEXT,
                  activebackground="#4a4a70", cursor="hand2",
                  relief="flat", padx=16, pady=8,
                  command=self._capture).pack(side="left", padx=8)

        # Result panel
        result_frame = tk.Frame(self, bg=self.PANEL, relief="flat")
        result_frame.pack(fill="x", padx=20, pady=8)

        self.result_label = tk.Label(result_frame, text="—",
                                     font=res_f, bg=self.PANEL,
                                     fg=self.ACCENT_NEUTRAL)
        self.result_label.pack(pady=(12, 2))

        self.confidence_label = tk.Label(result_frame, text="",
                                         font=sub_f, bg=self.PANEL,
                                         fg=self.ACCENT_NEUTRAL)
        self.confidence_label.pack(pady=(0, 4))

        self.bar_frame = tk.Frame(result_frame, bg=self.PANEL)
        self.bar_frame.pack(fill="x", padx=20, pady=(4, 12))

        # Status bar
        self.status_var = tk.StringVar()
        tk.Label(self, textvariable=self.status_var,
                 font=sub_f, bg=self.BG, fg=self.ACCENT_NEUTRAL,
                 anchor="w").pack(fill="x", padx=20, pady=(2, 12))

    # ── Actions ──────────────────────────────────────────────────────────────

    def _upload(self):
        path = filedialog.askopenfilename(
            title="Select a meat image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp *.avif *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            pil_img = Image.open(path)
            self._load_image(pil_img, label=Path(path).name)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image:\n{exc}")

    def _capture(self):
        try:
            import cv2
        except ImportError:
            messagebox.showinfo(
                "OpenCV not installed",
                "Camera capture requires OpenCV.\n\n"
                "Install it with:\n  pip install opencv-python\n\n"
                "Then restart the app."
            )
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera error",
                                 "Could not open camera.\n"
                                 "Make sure a webcam is connected and accessible.")
            return

        self._status("Camera open — press SPACE to capture, ESC to cancel.")
        captured_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Camera — SPACE: capture  |  ESC: cancel", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:       # SPACE
                captured_frame = frame
                break
            elif key == 27:     # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_frame is None:
            self._status("Capture cancelled.")
            return

        # Convert BGR OpenCV frame → RGB PIL image
        rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._load_image(pil_img, label="Camera capture")

    def _load_image(self, pil_img: Image.Image, label: str = ""):
        self._current_pil = pil_img
        self._photo = _thumbnail(pil_img)
        self.canvas.configure(image=self._photo, text="")
        self._clear_results()
        self._status(f"Loaded: {label}  — classifying…")
        self.update()
        self._classify()

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self):
        if self._current_pil is None:
            return
        try:
            sorted_probs = _classify_pil(self._current_pil)
            self._show_results(sorted_probs)
        except RuntimeError as exc:
            messagebox.showerror("Model error", str(exc))
            self._status("Error — see dialog.")
        except Exception as exc:
            messagebox.showerror("Unexpected error", str(exc))
            self._status("Unexpected error — see dialog.")

    def _show_results(self, sorted_probs: list):
        top_label, top_prob = sorted_probs[0]

        if top_label == "Fresh":
            colour = self.ACCENT_FRESH
            icon = "✅"
        else:
            colour = self.ACCENT_SPOILED
            icon = "❌"

        self.result_label.configure(text=f"{icon}  {top_label}", fg=colour)
        self.confidence_label.configure(
            text=f"{top_prob * 100:.1f}% confidence", fg=colour)

        # Probability bars
        for w in self.bar_frame.winfo_children():
            w.destroy()

        bar_width = 320
        for cls, prob in sorted_probs:
            row = tk.Frame(self.bar_frame, bg=self.PANEL)
            row.pack(fill="x", pady=2)

            bar_colour = self.ACCENT_FRESH if cls == "Fresh" else self.ACCENT_SPOILED

            tk.Label(row, text=f"{cls:<10}", width=10, anchor="w",
                     bg=self.PANEL, fg=self.TEXT,
                     font=("Segoe UI", 10)).pack(side="left")

            filled = max(4, int(bar_width * prob))
            c = tk.Canvas(row, width=bar_width, height=16,
                          bg="#3b3b55", highlightthickness=0)
            c.pack(side="left", padx=(4, 8))
            c.create_rectangle(0, 0, filled, 16, fill=bar_colour, outline="")

            tk.Label(row, text=f"{prob * 100:.1f}%", width=7, anchor="e",
                     bg=self.PANEL, fg=self.TEXT,
                     font=("Segoe UI", 10)).pack(side="left")

        self._status(f"Classified as {top_label} ({top_prob * 100:.1f}%)")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _clear_results(self):
        self.result_label.configure(text="—", fg=self.ACCENT_NEUTRAL)
        self.confidence_label.configure(text="")
        for w in self.bar_frame.winfo_children():
            w.destroy()

    def _status(self, msg: str):
        self.status_var.set(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
