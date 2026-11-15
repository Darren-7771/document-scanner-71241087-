from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

@dataclass
class ScanResult:
    original: np.ndarray
    resized: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    edged: np.ndarray
    contour_overlay: np.ndarray
    corners: Optional[np.ndarray]
    warped_color: np.ndarray
    warped_gray: np.ndarray
    otsu_binary: np.ndarray
    adaptive_binary: np.ndarray
    detected_4_corners: bool

def resize_keep_aspect(image: np.ndarray, target_height: int = 500) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if h <= target_height:
        return image.copy(), 1.0
    ratio = h / float(target_height)
    new_w = int(w / ratio)
    resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized, ratio

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)
    max_width = max(int(width_bottom), int(width_top))

    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_height = max(int(height_right), int(height_left))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    homography_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, homography_matrix, (max_width, max_height))
    return warped

def find_document_corners(
    edged: np.ndarray,
    blurred_gray: Optional[np.ndarray] = None,
    min_area_ratio: float = 0.10,
    approx_eps_ratios: Tuple[float, ...] = (0.02, 0.03, 0.04, 0.05)
) -> Optional[np.ndarray]:
    sources = [edged]
    if blurred_gray is not None:
        _, otsu_mask = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sources.append(otsu_mask)

    img_area = edged.shape[0] * edged.shape[1]
    best_quad: Optional[np.ndarray] = None
    best_area: float = 0.0

    for src in sources:
        contours, _ = cv2.findContours(src.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for contour in contours:
            hull = cv2.convexHull(contour)
            area = cv2.contourArea(hull)
            if area < img_area * min_area_ratio:
                continue

            peri = cv2.arcLength(hull, True)
            for eps in approx_eps_ratios:
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4 and area > best_area:
                    best_area = area
                    best_quad = approx.reshape(4, 2)
                    break

    return best_quad

def scan_document(
    image: np.ndarray,
    canny_low: int = 60,
    canny_high: int = 160,
    blur_ksize: int = 5,
    adaptive_block_size: int = 21,
    adaptive_c: int = 10
) -> ScanResult:
    original = image.copy()

    resized, ratio = resize_keep_aspect(original, target_height=500)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed_gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, close_kernel)
    blurred = cv2.GaussianBlur(closed_gray, (blur_ksize, blur_ksize), 0)

    edged = cv2.Canny(blurred, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged_closed = cv2.dilate(edged, kernel, iterations=1)

    corners_resized = find_document_corners(edged_closed, blurred_gray=blurred)
    contour_overlay = resized.copy()

    if corners_resized is not None:
        detected_4_corners = True
        ordered_resized = order_points(corners_resized.astype("float32"))
        pts_draw = ordered_resized.astype(int).reshape((-1, 1, 2))
        cv2.polylines(contour_overlay, [pts_draw], True, (0, 255, 0), 3)
        for idx, (x, y) in enumerate(ordered_resized.astype(int)):
            cv2.circle(contour_overlay, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(
                contour_overlay, f"P{idx+1}", (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

        corners_original = corners_resized.astype("float32") * ratio

        warped_color = four_point_transform(original, corners_original)
    else:
        detected_4_corners = False
        corners_original = None
        cv2.putText(
            contour_overlay, "4 Sudut Tidak Terdeteksi (Fallback Asli)",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        warped_color = original.copy()

    warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)

    _, otsu_binary = cv2.threshold(
        warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1
    adaptive_binary = cv2.adaptiveThreshold(
        warped_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block_size,
        adaptive_c,
    )

    return ScanResult(
        original=original,
        resized=resized,
        gray=gray,
        blurred=blurred,
        edged=edged_closed,
        contour_overlay=contour_overlay,
        corners=corners_original,
        warped_color=warped_color,
        warped_gray=warped_gray,
        otsu_binary=otsu_binary,
        adaptive_binary=adaptive_binary,
        detected_4_corners=detected_4_corners,
    )

def save_stage_outputs(result: ScanResult, output_dir: Path, prefix: str = "scan") -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "1_original": result.resized,
        "2_gray_blur": result.blurred,
        "3_canny_edge": result.edged,
        "4_contour_4_sudut": result.contour_overlay,
        "5_warped_color": result.warped_color,
        "6_otsu_pembanding": result.otsu_binary,
        "7_adaptive_hasil_akhir": result.adaptive_binary,
    }
    saved_paths: Dict[str, Path] = {}
    for name, img in files.items():
        path = output_dir / f"{prefix}_{name}.png"
        cv2.imwrite(str(path), img)
        saved_paths[name] = path
    return saved_paths

def export_to_pdf(binary_or_color_img: np.ndarray, pdf_path: Path) -> Path:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if len(binary_or_color_img.shape) == 2:
        pil_img = Image.fromarray(binary_or_color_img).convert("RGB")
    else:
        rgb = cv2.cvtColor(binary_or_color_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    pil_img.save(str(pdf_path), "PDF", resolution=150.0)
    return pdf_path

def make_display_tile(img: np.ndarray, label: str, tile_w: int = 400, tile_h: int = 320) -> np.ndarray:
    if len(img.shape) == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img.copy()

    h, w = bgr.shape[:2]
    scale = min(tile_w / w, (tile_h - 36) / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((tile_h, tile_w, 3), 25, dtype=np.uint8)
    x_offset = (tile_w - new_w) // 2
    y_offset = 36 + ((tile_h - 36 - new_h) // 2)
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    cv2.rectangle(canvas, (0, 0), (tile_w, 34), (45, 45, 55), -1)
    cv2.putText(canvas, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas

def build_pipeline_grid(result: ScanResult) -> np.ndarray:
    tiles = [
        make_display_tile(result.blurred, "1. Grayscale + Gaussian Blur"),
        make_display_tile(result.edged, "2. Canny Edge Detection"),
        make_display_tile(result.contour_overlay, "3. Deteksi 4 Titik Sudut"),
        make_display_tile(result.warped_color, "4. Perspective Warp (Lurus)"),
        make_display_tile(result.otsu_binary, "5. Pembanding (Otsu)"),
        make_display_tile(result.adaptive_binary, "6. Adaptive Threshold (Hasil)"),
    ]
    row1 = np.hstack(tiles[:3])
    row2 = np.hstack(tiles[3:])
    return np.vstack([row1, row2])
