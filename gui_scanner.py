from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from scanner_core import ScanResult, build_pipeline_grid, export_to_pdf, save_stage_outputs, scan_document

class DocumentScannerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Smart Document Scanner - Tugas Final CVIA")
        self.root.geometry("1220x760")
        self.root.configure(bg="#1e1e2e")

        self.base_dir = Path(__file__).resolve().parent
        self.input_dir = self.base_dir / "input_dokumen"
        self.output_dir = self.base_dir / "output_tahapan"

        self.current_image: np.ndarray | None = None
        self.current_name: str = "dokumen"
        self.last_result: ScanResult | None = None
        self.view_mode = tk.StringVar(value="blurred")

        self._left_photo = None
        self._right_photo = None

        self._build_ui()

    def _build_ui(self) -> None:

        top_bar = tk.Frame(self.root, bg="#2a2a3c", pady=10, padx=14)
        top_bar.pack(fill=tk.X)

        btn_pick = tk.Button(
            top_bar,
            text="Ambil Foto Dokumen",
            command=self.pick_image,
            bg="#3b82f6",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
        )
        btn_pick.pack(side=tk.LEFT, padx=(0, 10))

        btn_save = tk.Button(
            top_bar,
            text="Simpan PDF & Semua Tahap",
            command=self.save_outputs,
            bg="#10b981",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
        )
        btn_save.pack(side=tk.LEFT, padx=(0, 16))

        self.status_label = tk.Label(
            top_bar,
            text="Pilih foto dokumen untuk memulai...",
            bg="#2a2a3c",
            fg="#e2e8f0",
            font=("Segoe UI", 10),
        )
        self.status_label.pack(side=tk.LEFT)

        mode_bar = tk.Frame(self.root, bg="#1e1e2e", pady=6, padx=14)
        mode_bar.pack(fill=tk.X)

        tk.Label(
            mode_bar,
            text="Tampilan Kanan:",
            bg="#1e1e2e",
            fg="#94a3b8",
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8))

        modes = [
            ("1. Grayscale + Blur", "blurred"),
            ("2. Canny Edge", "edged"),
            ("3. Deteksi 4 Sudut", "contour"),
            ("4. Lurus Berwarna (Warp)", "warped"),
            ("5. Pembanding (Otsu)", "otsu"),
            ("6. Hasil Akhir (Adaptive)", "adaptive"),
            ("7. Ringkasan 6 Tahap", "grid"),
        ]
        for label, val in modes:
            rb = tk.Radiobutton(
                mode_bar,
                text=label,
                variable=self.view_mode,
                value=val,
                command=self._refresh_right_panel,
                indicatoron=0,
                bg="#334155",
                fg="white",
                selectcolor="#3b82f6",
                activebackground="#475569",
                activeforeground="white",
                font=("Segoe UI", 9),
                padx=10,
                pady=4,
                relief=tk.FLAT,
                cursor="hand2",
            )
            rb.pack(side=tk.LEFT, padx=3)

        self.content_frame = tk.Frame(self.root, bg="#1e1e2e", padx=14, pady=6)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.columnconfigure(1, weight=1)
        self.content_frame.rowconfigure(1, weight=1)

        self.left_title = tk.Label(
            self.content_frame,
            text="Input Asli",
            bg="#1e1e2e",
            fg="#cbd5e1",
            font=("Segoe UI", 10, "bold"),
        )
        self.left_title.grid(row=0, column=0, pady=(0, 4))

        self.right_title = tk.Label(
            self.content_frame,
            text="Hasil Pemrosesan CVIA",
            bg="#1e1e2e",
            fg="#cbd5e1",
            font=("Segoe UI", 10, "bold"),
        )
        self.right_title.grid(row=0, column=1, pady=(0, 4))

        self.left_panel = tk.Label(
            self.content_frame,
            text="Belum ada foto.\nKlik 'Ambil Foto Dokumen' di kiri atas",
            bg="#111827",
            fg="#64748b",
            font=("Segoe UI", 11),
        )
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        self.right_panel = tk.Label(
            self.content_frame,
            text="Hasil pemrosesan akan tampil di sini",
            bg="#111827",
            fg="#64748b",
            font=("Segoe UI", 11),
        )
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

    def _load_initial_image(self) -> None:
        if not self.input_dir.exists():
            return
        preferred = [
            self.input_dir / "catatan_miring_30_derajat.jpg",
            self.input_dir / "catatan_asli.jpeg",
            self.input_dir / "catatan_asli.jpg",
        ]
        target = next((p for p in preferred if p.exists()), None)
        if target is None:
            imgs = sorted(
                list(self.input_dir.glob("*.jpg"))
                + list(self.input_dir.glob("*.jpeg"))
                + list(self.input_dir.glob("*.png"))
            )
            target = imgs[0] if imgs else None
        if target is not None:
            self.load_image(target)

    def pick_image(self) -> None:
        initial = str(self.input_dir) if self.input_dir.exists() else str(self.base_dir)
        file_path = filedialog.askopenfilename(
            title="Pilih Foto Dokumen yang Akan Di-Scan",
            initialdir=initial,
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if file_path:
            self.load_image(Path(file_path))

    def load_image(self, path: Path) -> None:
        img = cv2.imread(str(path))
        if img is None:
            messagebox.showerror("Error", f"Gagal membaca gambar: {path}")
            return
        self.current_image = img
        self.current_name = path.stem
        self.run_scan()

    def run_scan(self) -> None:
        if self.current_image is None:
            return
        t0 = time.perf_counter()
        self.last_result = scan_document(self.current_image)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        status_sudut = "Terdeteksi" if self.last_result.detected_4_corners else "Tidak Terdeteksi (Fallback)"
        h, w = self.last_result.warped_color.shape[:2]
        self.status_label.config(
            text=f"File: {self.current_name}  |  4 Sudut: {status_sudut}  |  Waktu: {elapsed_ms:.1f} ms  |  Resolusi: {w}x{h}"
        )

        self._left_photo = self._to_tk_image(self.last_result.resized, max_w=560, max_h=580)
        self.left_panel.config(image=self._left_photo, text="")
        self._refresh_right_panel()

    def _refresh_right_panel(self) -> None:
        if self.last_result is None:
            return
        mode = self.view_mode.get()
        if mode == "grid":
            self.left_title.grid_remove()
            self.left_panel.grid_remove()
            self.content_frame.columnconfigure(0, weight=0)
            self.content_frame.columnconfigure(1, weight=1)
            self.right_title.config(text="Ringkasan 6 Tahap Pemrosesan CVIA")
            self.right_title.grid(row=0, column=0, columnspan=2, pady=(0, 4))
            self.right_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0)
            img = build_pipeline_grid(self.last_result)
            self._right_photo = self._to_tk_image(img, max_w=1140, max_h=590)
            self.right_panel.config(image=self._right_photo, text="")
            return

        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.columnconfigure(1, weight=1)
        self.left_title.grid(row=0, column=0, pady=(0, 4))
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.right_title.config(text="Hasil Pemrosesan CVIA")
        self.right_title.grid(row=0, column=1, columnspan=1, pady=(0, 4))
        self.right_panel.grid(row=1, column=1, columnspan=1, sticky="nsew", padx=(8, 0))

        if mode == "adaptive":
            img = self.last_result.adaptive_binary
        elif mode == "warped":
            img = self.last_result.warped_color
        elif mode == "contour":
            img = self.last_result.contour_overlay
        elif mode == "edged":
            img = self.last_result.edged
        elif mode == "blurred":
            img = self.last_result.blurred
        elif mode == "otsu":
            img = self.last_result.otsu_binary
        else:
            img = self.last_result.adaptive_binary

        self._right_photo = self._to_tk_image(img, max_w=560, max_h=580)
        self.right_panel.config(image=self._right_photo, text="")

    @staticmethod
    def _to_tk_image(img: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(resized))

    def save_outputs(self) -> None:
        if self.last_result is None:
            messagebox.showwarning("Peringatan", "Belum ada dokumen yang diproses.")
            return
        save_stage_outputs(self.last_result, self.output_dir, prefix=self.current_name)
        pdf_path = export_to_pdf(
            self.last_result.adaptive_binary,
            self.output_dir / f"{self.current_name}_hasil_scan.pdf",
        )
        messagebox.showinfo(
            "Berhasil Disimpan",
            f"Seluruh gambar tahapan & file PDF berhasil disimpan ke:\n{pdf_path}",
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentScannerGUI(root)
    root.mainloop()
