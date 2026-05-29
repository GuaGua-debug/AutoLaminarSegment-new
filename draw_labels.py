#!/usr/bin/env python
"""
Interactive semantic boundary and label drawing tool.

This is a semantic variant of draw_boundaries.py.  Each completed line keeps
the label selected when it was finalized.  Gray and L5/6(White) are saved as
GM.csv and WM.csv as well, so the existing pipeline can consume them directly.
"""

import argparse
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback for minimal environments
    cKDTree = None


SEMANTICS = [
    ("Edge", "Edge", "#FF8800"),
    ("Gray", "Gray", "#44FF88"),
    ("L1", "L1", "#FFCC00"),
    ("L2/3", "L2_3", "#44AAFF"),
    ("L4", "L4", "#FF44CC"),
    ("L5/6(White)", "White", "#FFFFFF"),
]
SEMANTIC_BY_NAME = {name.lower(): (name, slug, color) for name, slug, color in SEMANTICS}
DIRECT_KEYS = {
    "0": "Edge",
    "e": "Edge",
    "g": "Gray",
    "1": "L1",
    "2": "L2/3",
    "3": "L2/3",
    "4": "L4",
    "5": "L5/6(White)",
    "6": "L5/6(White)",
    "w": "L5/6(White)",
}

MASK_COLORS_BGR = [
    (230, 80, 60),
    (60, 180, 90),
    (60, 120, 230),
    (230, 190, 60),
    (190, 80, 220),
    (60, 210, 210),
    (240, 130, 50),
    (150, 150, 240),
]

LAYER_CHOICES = {
    "1": ("L1", 1, (255, 100, 100)),
    "2": ("L2/3", 2, (100, 255, 100)),
    "3": ("L2/3", 2, (100, 255, 100)),
    "4": ("L4", 4, (100, 100, 255)),
    "5": ("L5/6", 5, (255, 100, 255)),
    "6": ("L5/6", 5, (255, 100, 255)),
}

LUT_DEFS = {
    "Fire": ([0, 85, 170, 255], [[0, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255]]),
    "Gray": ([0, 255], [[0, 0, 0], [255, 255, 255]]),
    "Ice": ([0, 100, 200, 255], [[0, 0, 0], [0, 140, 255], [180, 220, 255], [255, 255, 255]]),
    "Green": ([0, 100, 200, 255], [[0, 0, 0], [0, 120, 0], [100, 255, 100], [200, 255, 200]]),
    "Hot": ([0, 100, 200, 255], [[0, 0, 0], [200, 50, 0], [255, 200, 0], [255, 255, 255]]),
    "Red": ([0, 100, 200, 255], [[0, 0, 0], [120, 0, 0], [255, 100, 100], [255, 200, 200]]),
}


@dataclass
class LabeledBoundary:
    points: np.ndarray
    label: str
    slug: str
    color: str


def load_display_image(image_path: Path) -> np.ndarray:
    """Load an image as uint8 grayscale for interactive display."""
    try:
        import tifffile

        img = tifffile.imread(str(image_path))
    except Exception:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]

    img = img.astype(np.float32)
    lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

    if img.ndim == 3:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return cv2.convertScaleAbs(img, alpha=1.5, beta=25)


class LabelDrawer:
    def __init__(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        save_path: Path,
        image_shape: tuple[int, int],
        gray_img: np.ndarray,
        save_mask: bool = False,
        line_width: int = 2,
        draw_margin: int = 0,
    ):
        self.fig = fig
        self.ax = ax
        self.save_path = save_path
        self._image_shape = image_shape
        self._gray_img = gray_img
        self._save_mask_flag = save_mask
        self._line_width = max(int(line_width), 1)
        self._draw_margin = max(int(draw_margin), 0)
        h, w = image_shape
        self._canvas_shape = (h + self._draw_margin * 2, w + self._draw_margin * 2)

        self.boundaries: list[LabeledBoundary] = []
        self.current_pts: list[tuple[float, float]] = []
        self._semantic_idx = 0

        self._cur_line = None
        self._cur_dots = None
        self._completed_artists: list[tuple] = []
        self._mask_artists: list = []
        self._mask_mode = False
        self._selected_mask: np.ndarray | None = None
        self._selected_contour: np.ndarray | None = None
        self._fixed_regions: list[dict] = []
        self._awaiting_region_layer = False
        self._prompt_active = False
        self._layer_input_queue: queue.Queue[str] = queue.Queue()
        self._barrier_cache: np.ndarray | None = None
        self._barrier_dirty = True

        self._img_handle = None
        self._lut_idx = 0
        self._lut_names = list(LUT_DEFS.keys())
        self._bg = None

        self._apply_lut()
        fig.canvas.mpl_connect("draw_event", self._on_draw)
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._layer_input_timer = fig.canvas.new_timer(interval=100)
        self._layer_input_timer.add_callback(self._poll_layer_input_queue)
        self._layer_input_timer.start()
        self._refresh_status()

    @property
    def current_semantic(self) -> tuple[str, str, str]:
        return SEMANTICS[self._semantic_idx]

    def _apply_lut(self):
        name = self._lut_names[self._lut_idx]
        stops, colors = LUT_DEFS[name]
        colors = np.asarray(colors, dtype=np.uint8)
        lut = np.zeros((256, 3), dtype=np.uint8)
        for c in range(3):
            lut[:, c] = np.interp(np.arange(256), stops, colors[:, c]).astype(np.uint8)

        rgb = lut[self._gray_img]
        if self._img_handle is None:
            self._img_handle = self.ax.imshow(rgb, aspect="equal", interpolation="nearest")
        else:
            self._img_handle.set_data(rgb)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if self._toolbar_active():
            return
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return

        if self._mask_mode:
            if event.button == 1:
                self._select_mask_region(int(round(event.xdata)), int(round(event.ydata)))
            elif event.button == 3:
                self._request_region_layer()
            return

        if event.button == 1:
            self.current_pts.append((float(event.xdata), float(event.ydata)))
            self._redraw_current()
            self._blit()
            self._refresh_status()
        elif event.button == 3:
            self._close_boundary()

    def _on_key(self, event):
        k = (event.key or "").lower()

        if self._mask_mode and k in LAYER_CHOICES:
            self._fix_selected_region(k)
        elif self._mask_mode and k in ("enter", " "):
            self._request_region_layer()
        elif k in ("enter", " "):
            self._close_boundary()
        elif k in ("u", "backspace"):
            if self.current_pts:
                self.current_pts.pop()
                self._redraw_current()
                self._blit()
                self._refresh_status()
        elif k == "c":
            self._cancel_current()
        elif k == "d":
            self._delete_last()
        elif k == "t":
            self._cycle_semantic()
        elif k in DIRECT_KEYS:
            self._set_semantic(DIRECT_KEYS[k])
        elif k == "l":
            self._lut_idx = (self._lut_idx + 1) % len(self._lut_names)
            self._apply_lut()
            self._set_message(f"LUT: {self._lut_names[self._lut_idx]}")
        elif k == "m":
            self._toggle_mask_mode()
        elif k == "s":
            self._save()
        elif k in ("q", "escape"):
            self._save()
            plt.close(self.fig)

    def _set_semantic(self, label: str):
        for i, (name, _slug, _color) in enumerate(SEMANTICS):
            if name == label:
                self._semantic_idx = i
                self._set_message(f"Current label: {label}")
                return

    def _cycle_semantic(self):
        self._semantic_idx = (self._semantic_idx + 1) % len(SEMANTICS)
        self._set_message(f"Current label: {self.current_semantic[0]}")

    def _close_boundary(self):
        if len(self.current_pts) < 2:
            self._set_message("Need at least 2 points to finalize a line.", warn=True)
            return

        label, slug, color = self.current_semantic
        pts = np.asarray(self.current_pts, dtype=np.float64)
        boundary = LabeledBoundary(points=pts, label=label, slug=slug, color=color)
        self.boundaries.append(boundary)

        (line,) = self.ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=2, zorder=9)
        dots = self.ax.scatter(pts[:, 0], pts[:, 1], s=20, c=color, zorder=10, linewidths=0)
        text = self.ax.text(
            pts[0, 0],
            pts[0, 1],
            f" {label}",
            color=color,
            fontsize=9,
            weight="bold",
            zorder=11,
            path_effects=[],
        )
        self._completed_artists.append((line, dots, text))

        self.current_pts = []
        self._remove_artist("_cur_line")
        self._remove_artist("_cur_dots")
        self._blit()
        self._refresh_status()
        print(f"{label} line {len(self.boundaries)} finalized ({len(pts)} vertices)")
        self._invalidate_selection_barrier()

        if self._mask_mode:
            self._redraw_mask_selection()

    def _cancel_current(self):
        self.current_pts = []
        self._remove_artist("_cur_line")
        self._remove_artist("_cur_dots")
        self._blit()
        self._refresh_status()

    def _delete_last(self):
        if not self.boundaries:
            return

        removed = self.boundaries.pop()
        for artist in self._completed_artists.pop():
            try:
                artist.remove()
            except Exception:
                pass

        self._blit()
        self._refresh_status()
        print(f"Deleted last line ({removed.label}). {len(self.boundaries)} remaining.")
        self._invalidate_selection_barrier()

        if self._mask_mode:
            self._redraw_mask_selection()

    def _toggle_mask_mode(self):
        self._mask_mode = not self._mask_mode
        if self._mask_mode:
            self._cancel_current()
            self._redraw_mask_selection()
            self._set_message("Mask mode ON: left click selects a closed region; right click fixes it.")
        else:
            self._remove_mask_artists()
            self._set_message("Mask mode OFF")
        self._refresh_status()

    def _select_mask_region(self, x: int, y: int):
        h, w = self._image_shape
        m = self._draw_margin
        if not (-m <= x < w + m and -m <= y < h + m):
            self._set_message("Selection point is outside the drawing canvas.", warn=True)
            return

        selected = self._flood_select_region(x, y)
        if selected is None:
            return

        self._selected_mask, self._selected_contour = selected
        self._awaiting_region_layer = False
        area = int(np.count_nonzero(self._selected_mask))
        self._redraw_mask_selection()
        self._set_message(f"Selected region: {area} px. Right click, then press 1/2/3/4/5/6.")

    def _request_region_layer(self):
        if self._selected_mask is None or self._selected_contour is None:
            self._set_message("No selected region. Left click inside a closed region first.", warn=True)
            return
        self._awaiting_region_layer = True
        if not self._prompt_active:
            self._prompt_active = True
            threading.Thread(target=self._read_region_layer_from_console, daemon=True).start()
        print()
        print("Assign selected region layer by typing in the terminal or pressing a number in the image window:")
        print("  1=L1, 2/3=L2/3, 4=L4, 5/6=L5/6")
        self._set_message("Type or press 1, 2/3, 4, or 5/6 to assign selected region.")

    def _read_region_layer_from_console(self):
        try:
            value = input("Layer number: ").strip()
            self._layer_input_queue.put(value)
        except Exception as exc:
            self._layer_input_queue.put(f"__ERROR__:{exc}")

    def _poll_layer_input_queue(self):
        while True:
            try:
                value = self._layer_input_queue.get_nowait()
            except queue.Empty:
                break

            self._prompt_active = False
            if value.startswith("__ERROR__:"):
                self._set_message(f"Terminal input failed: {value[len('__ERROR__:'):]}", warn=True)
            else:
                self._fix_selected_region(value)
        return True

    def _fix_selected_region(self, value: str):
        if self._selected_mask is None or self._selected_contour is None:
            self._set_message("No selected region. Left click inside a closed region first.", warn=True)
            return
        if value not in LAYER_CHOICES:
            self._set_message("Valid layer keys are 1, 2, 3, 4, 5, or 6.", warn=True)
            return

        layer_name, layer_id, color = LAYER_CHOICES[value]
        region_id = len(self._fixed_regions) + 1
        filled = self._filled_contour_mask(self._selected_contour)
        self._fixed_regions.append(
            {
                "region_id": region_id,
                "layer": layer_name,
                "layer_id": layer_id,
                "color": color,
                "mask": filled,
                "contour": self._selected_contour.copy(),
            }
        )
        self._selected_mask = None
        self._selected_contour = None
        self._awaiting_region_layer = False
        self._prompt_active = False
        self._prompt_active = False
        self._redraw_mask_selection()
        self._set_message(f"Fixed region {region_id}: {layer_name}")

    def _redraw_mask_selection(self):
        self._remove_mask_artists(redraw=False)

        for region in self._fixed_regions:
            pts = self._display_contour(region["contour"])
            color = np.asarray(region["color"][::-1], dtype=float) / 255.0
            (line,) = self.ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=2, zorder=20)
            self._mask_artists.append(line)
            x, y = pts[0]
            text = self.ax.text(
                x,
                y,
                f" {region['layer']}",
                color=color,
                fontsize=9,
                weight="bold",
                zorder=21,
            )
            self._mask_artists.append(text)

        if self._selected_contour is not None:
            pts = self._display_contour(self._selected_contour)
            (line,) = self.ax.plot(pts[:, 0], pts[:, 1], "--", color="yellow", linewidth=2.5, zorder=22)
            self._mask_artists.append(line)

        self._blit()

    def _remove_mask_artists(self, redraw: bool = True):
        for artist in self._mask_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._mask_artists = []
        if redraw:
            self._blit()

    def _build_region_labels(self):
        h, w = self._image_shape
        if not self.boundaries:
            return None, []

        barrier = self._build_selection_barrier()

        # Treat drawn lines as walls.  Any background component that does not
        # touch the image border is an enclosed region, including adjacent
        # regions that share a drawn wall.
        passable = (barrier == 0).astype(np.uint8)
        num_labels, components = cv2.connectedComponents(passable, connectivity=8)
        if num_labels <= 1:
            return np.zeros((h, w), dtype=np.uint16), []

        label_map = np.zeros((h, w), dtype=np.uint16)
        contours = []
        border_labels = set(np.unique(components[0, :]))
        border_labels.update(np.unique(components[-1, :]))
        border_labels.update(np.unique(components[:, 0]))
        border_labels.update(np.unique(components[:, -1]))

        out_idx = 1
        for component_id in range(1, num_labels):
            if component_id in border_labels:
                continue
            region = components == component_id
            if int(np.count_nonzero(region)) < 20:
                continue

            label_map[region] = out_idx
            region_u8 = region.astype(np.uint8) * 255
            region_contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if region_contours:
                contours.append(max(region_contours, key=cv2.contourArea))
            out_idx += 1

        return label_map, contours

    def _build_selection_barrier(self) -> np.ndarray:
        h, w = self._canvas_shape
        barrier = np.zeros((h, w), dtype=np.uint8)
        draw_width = max(self._line_width, 2)
        offset = np.array([self._draw_margin, self._draw_margin], dtype=np.int32)

        for boundary in self.boundaries:
            pts = self._interpolate(boundary.points) + offset
            pts = self._clip_points(pts, w, h)
            if len(pts) >= 2:
                cv2.polylines(barrier, [pts.reshape(-1, 1, 2)], False, 255, thickness=draw_width)
        return barrier

    def _get_selection_barrier(self) -> np.ndarray:
        if self._barrier_dirty or self._barrier_cache is None:
            self._barrier_cache = self._build_selection_barrier()
            self._barrier_dirty = False
        return self._barrier_cache

    def _invalidate_selection_barrier(self):
        self._barrier_cache = None
        self._barrier_dirty = True
        self._selected_mask = None
        self._selected_contour = None
        self._awaiting_region_layer = False

    def _flood_select_region(self, x: int, y: int):
        if not self.boundaries:
            self._set_message("Draw boundary lines before using mask selection.", warn=True)
            return None

        h, w = self._canvas_shape
        cx = int(x + self._draw_margin)
        cy = int(y + self._draw_margin)
        if not (0 <= cx < w and 0 <= cy < h):
            self._set_message("Selection point is outside the drawing canvas.", warn=True)
            return None

        barrier = self._get_selection_barrier()
        if barrier[cy, cx] > 0:
            self._set_message("Clicked on a boundary line. Click inside a closed region.", warn=True)
            return None

        passable = (barrier == 0).astype(np.uint8)
        cv2.floodFill(passable, None, (cx, cy), 2)
        region = passable == 2
        touches_border = (
            np.any(region[0, :])
            or np.any(region[-1, :])
            or np.any(region[:, 0])
            or np.any(region[:, -1])
        )
        if touches_border:
            self._set_message(
                "Selected area leaks to the image edge. Close the boundary manually.",
                warn=True,
            )
            return None

        if int(np.count_nonzero(region)) < 20:
            self._set_message("Selected region is too small.", warn=True)
            return None

        region_u8 = region.astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._set_message("Could not trace the selected region.", warn=True)
            return None

        contour = max(contours, key=cv2.contourArea)
        filled = self._filled_contour_mask(contour)
        return filled, contour

    def _filled_contour_mask(self, contour: np.ndarray) -> np.ndarray:
        h, w = self._canvas_shape
        canvas_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(canvas_mask, [contour], -1, 255, thickness=cv2.FILLED)

        m = self._draw_margin
        ih, iw = self._image_shape
        mask = canvas_mask[m : m + ih, m : m + iw].copy()
        return mask

    def _display_contour(self, contour: np.ndarray) -> np.ndarray:
        pts = contour.reshape(-1, 2).astype(np.int32).copy()
        pts[:, 0] -= self._draw_margin
        pts[:, 1] -= self._draw_margin
        return pts

    def _save_mask(self):
        if not self._fixed_regions:
            print("  [mask] No fixed regions to save. Press M, left-click a region, then right-click and assign a layer.")
            return

        base = self.save_path.parent / self.save_path.stem
        binary_path = base.parent / "gray_mask.png"
        color_path = base.parent / "layer_mask.png"

        h, w = self._image_shape
        binary = np.zeros((h, w), dtype=np.uint8)
        color = np.zeros((h, w, 3), dtype=np.uint8)

        for region in self._fixed_regions:
            mask = region["mask"] > 0
            binary[mask] = 255
            color[mask] = region["color"]

        line_mask = self._boundary_line_mask()
        binary[line_mask > 0] = 255
        color = self._fill_color_mask_gaps(color, binary)

        cv2.imwrite(str(binary_path), binary)
        cv2.imwrite(str(color_path), color)

        print(f"  gray mask -> {binary_path.name}")
        print(f"  layer color mask ({len(self._fixed_regions)} fixed region(s)) -> {color_path.name}")

    def _boundary_line_mask(self) -> np.ndarray:
        h, w = self._image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for boundary in self.boundaries:
            pts = self._interpolate_in_image(boundary)
            if len(pts) >= 2:
                cv2.polylines(
                    mask,
                    [pts.reshape(-1, 1, 2)],
                    False,
                    255,
                    thickness=max(self._line_width, 1),
                )
        return mask

    @staticmethod
    def _fill_color_mask_gaps(color: np.ndarray, binary: np.ndarray) -> np.ndarray:
        missing = (binary > 0) & ~np.any(color > 0, axis=2)
        if not np.any(missing):
            return color

        known = np.any(color > 0, axis=2).astype(np.uint8)
        if not np.any(known):
            return color

        _, labels = cv2.distanceTransformWithLabels(
            1 - known,
            cv2.DIST_L2,
            3,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        known_coords = np.column_stack(np.where(known > 0))
        flat_labels = labels[missing] - 1
        valid = (flat_labels >= 0) & (flat_labels < len(known_coords))

        missing_coords = np.column_stack(np.where(missing))
        target = missing_coords[valid]
        source = known_coords[flat_labels[valid]]
        color[target[:, 0], target[:, 1]] = color[source[:, 0], source[:, 1]]
        return color

    def _save(self):
        if not self.boundaries:
            print("No lines to save.")
            return

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for line_idx, boundary in enumerate(self.boundaries, start=1):
            interp = self._interpolate_in_image(boundary)
            for point_idx, (x, y) in enumerate(interp, start=1):
                rows.append(
                    {
                        "line_id": line_idx,
                        "label": boundary.label,
                        "semantic": boundary.slug,
                        "point_id": point_idx,
                        "x": int(x),
                        "y": int(y),
                    }
                )
        columns = ["line_id", "label", "semantic", "point_id", "x", "y"]
        pd.DataFrame(rows, columns=columns).to_csv(self.save_path, index=False)
        print(f"Saved labeled lines -> {self.save_path}")

        self._save_lines_by_semantic()
        self._save_depth_ranges()

        if self._mask_mode or self._save_mask_flag:
            self._save_mask()

        self._set_message(f"Saved {len(self.boundaries)} labeled lines.")

    def _save_lines_by_semantic(self):
        counts: dict[str, int] = {}
        first_by_slug: dict[str, Path] = {}

        for i, boundary in enumerate(self.boundaries, start=1):
            counts[boundary.slug] = counts.get(boundary.slug, 0) + 1
            suffix = "" if counts[boundary.slug] == 1 else f"_{counts[boundary.slug]}"
            filename = f"{boundary.slug}{suffix}.csv"
            if boundary.slug == "Edge":
                filename = f"Edge_{counts[boundary.slug]}.csv"

            csv_path = self.save_path.parent / filename
            interp = self._interpolate_in_image(boundary)
            pd.DataFrame(interp, columns=["x", "y"]).to_csv(csv_path, index=False)
            first_by_slug.setdefault(boundary.slug, csv_path)
            print(f"  {boundary.label} line {i} -> {csv_path.name}")

        # Compatibility aliases for the existing pipeline.
        if "Gray" in first_by_slug:
            df = pd.read_csv(first_by_slug["Gray"])
            df.to_csv(self.save_path.parent / "GM.csv", index=False)
            print("  Gray alias -> GM.csv")
        if "White" in first_by_slug:
            df = pd.read_csv(first_by_slug["White"])
            df.to_csv(self.save_path.parent / "WM.csv", index=False)
            print("  White alias -> WM.csv")

    def _save_depth_ranges(self):
        needed = ["Gray", "White", "L1", "L2_3", "L4"]
        by_slug = self._first_boundary_by_slug()
        missing = [slug for slug in needed if slug not in by_slug]
        if missing:
            print(f"  [depth] Skipped depth ranges. Missing: {', '.join(missing)}")
            return

        gray = self._interpolate_in_image(by_slug["Gray"])
        white = self._interpolate_in_image(by_slug["White"])
        if len(gray) == 0 or len(white) == 0:
            print("  [depth] Skipped depth ranges. Gray/White has no in-image points.")
            return

        summaries = []
        boundary_depths = {}
        for slug in ["L1", "L2_3", "L4"]:
            pts = self._interpolate_in_image(by_slug[slug])
            if len(pts) == 0:
                print(f"  [depth] Skipped depth ranges. {by_slug[slug].label} has no in-image points.")
                return
            depths = self._depth_from_gray_white(pts, gray, white)
            boundary_depths[slug] = float(np.nanmedian(depths))
            summaries.append(
                {
                    "boundary": by_slug[slug].label,
                    "depth_median": float(np.nanmedian(depths)),
                    "depth_mean": float(np.nanmean(depths)),
                    "depth_p05": float(np.nanpercentile(depths, 5)),
                    "depth_p95": float(np.nanpercentile(depths, 95)),
                    "n_points": int(np.count_nonzero(np.isfinite(depths))),
                }
            )

        summary_path = self.save_path.parent / "manual_boundary_depth_summary.csv"
        pd.DataFrame(summaries).to_csv(summary_path, index=False)

        b1 = boundary_depths["L1"]
        b23 = boundary_depths["L2_3"]
        b4 = boundary_depths["L4"]
        ordered = b1 <= b23 <= b4
        if not ordered:
            print(
                "  [depth] Warning: boundary depths are not ordered "
                f"(L1={b1:.3f}, L2/3={b23:.3f}, L4={b4:.3f})."
            )

        layers = [
            {"layer": "L1", "start": 0.0, "end": b1, "mean_density": np.nan, "ordered": ordered},
            {"layer": "L2/3", "start": b1, "end": b23, "mean_density": np.nan, "ordered": ordered},
            {"layer": "L4", "start": b23, "end": b4, "mean_density": np.nan, "ordered": ordered},
            {"layer": "L5/6", "start": b4, "end": 1.0, "mean_density": np.nan, "ordered": ordered},
        ]
        layers_path = self.save_path.parent / "manual_segmented_layers.csv"
        pd.DataFrame(layers).to_csv(layers_path, index=False)

        print(f"  boundary depth summary -> {summary_path.name}")
        print(f"  manual layer ranges -> {layers_path.name}")

    def _first_boundary_by_slug(self) -> dict[str, LabeledBoundary]:
        found = {}
        for boundary in self.boundaries:
            found.setdefault(boundary.slug, boundary)
        return found

    @staticmethod
    def _depth_from_gray_white(points: np.ndarray, gray: np.ndarray, white: np.ndarray) -> np.ndarray:
        sample = np.asarray(points, dtype=float)
        gray = np.asarray(gray, dtype=float)
        white = np.asarray(white, dtype=float)

        if cKDTree is not None:
            dist_gray, _ = cKDTree(gray).query(sample, k=1)
            dist_white, _ = cKDTree(white).query(sample, k=1)
        else:
            dist_gray = LabelDrawer._nearest_distance_bruteforce(sample, gray)
            dist_white = LabelDrawer._nearest_distance_bruteforce(sample, white)

        return dist_gray / (dist_gray + dist_white + 1e-8)

    @staticmethod
    def _nearest_distance_bruteforce(sample: np.ndarray, boundary: np.ndarray, chunk_size: int = 5000) -> np.ndarray:
        out = np.empty(len(sample), dtype=float)
        for start in range(0, len(sample), chunk_size):
            chunk = sample[start : start + chunk_size]
            d2 = np.sum((chunk[:, None, :] - boundary[None, :, :]) ** 2, axis=2)
            out[start : start + chunk_size] = np.sqrt(np.min(d2, axis=1))
        return out

    def _redraw_current(self):
        self._remove_artist("_cur_line")
        self._remove_artist("_cur_dots")
        if not self.current_pts:
            return

        label, _slug, color = self.current_semantic
        xs = [p[0] for p in self.current_pts]
        ys = [p[1] for p in self.current_pts]
        (self._cur_line,) = self.ax.plot(xs, ys, "--", color=color, linewidth=1.5, zorder=12)
        self._cur_dots = self.ax.scatter(xs, ys, s=25, c=color, zorder=13, linewidths=0)

    def _on_draw(self, _event):
        try:
            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        except Exception:
            self._bg = None

    def _blit(self):
        # Full redraw is more reliable across Matplotlib backends than manual
        # blitting for this interactive editor.
        self.fig.canvas.draw_idle()

    def _refresh_status(self, message: str = ""):
        n_done = len(self.boundaries)
        n_cur = len(self.current_pts)
        label, _slug, _color = self.current_semantic
        mask_tag = " [MASK]" if self._mask_mode else ""
        title = (
            f"Completed: {n_done} | Current vertices: {n_cur} | Label: {label}{mask_tag} "
            f"| Fixed masks:{len(self._fixed_regions)} "
            f"| LUT:{self._lut_names[self._lut_idx]}"
            f"{' | ' + message if message else ''}\n"
            "Draw: LClick=point RClick/Enter=finish. Mask: M, LClick=select, RClick/Enter=fix layer. "
            "T=cycle label 0/E Edge G Gray 1 L1 2/3 L2/3 4 L4 5/6/W White "
            "U=undo C=cancel D=delete L=LUT S=save Q=quit"
        )
        self.ax.set_title(title, fontsize=8.5, color="white", loc="left", pad=6)
        self._blit()

    def _set_message(self, msg: str, warn: bool = False):
        print(("WARNING: " if warn else "") + msg)
        self._refresh_status(msg)

    def _remove_artist(self, attr: str):
        artist = getattr(self, attr, None)
        if artist is None:
            return
        try:
            artist.remove()
        except Exception:
            pass
        setattr(self, attr, None)

    def _toolbar_active(self) -> bool:
        try:
            toolbar = self.fig.canvas.toolbar
            if toolbar is None:
                return False
            mode = getattr(toolbar, "mode", "")
            if mode is None:
                return False
            mode_text = str(mode).strip().lower()
            return "pan" in mode_text or "zoom" in mode_text
        except Exception:
            return False

    @staticmethod
    def _interpolate(pts: np.ndarray) -> np.ndarray:
        segments = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            dist = np.hypot(x1 - x0, y1 - y0)
            n = max(int(np.ceil(dist)), 2)
            xs = np.linspace(x0, x1, n, endpoint=False)
            ys = np.linspace(y0, y1, n, endpoint=False)
            segments.append(np.column_stack([xs, ys]))
        segments.append(pts[-1:])
        return np.round(np.vstack(segments)).astype(np.int32)

    def _interpolate_in_image(self, boundary: LabeledBoundary) -> np.ndarray:
        h, w = self._image_shape
        return self._clip_points(self._interpolate(boundary.points), w, h)

    @staticmethod
    def _clip_points(pts: np.ndarray, w: int, h: int) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.int32)
        valid = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        return pts[valid]


def main():
    parser = argparse.ArgumentParser(
        description="Draw semantic cortical label lines and optional multi-region masks."
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--output",
        default=None,
        help="Output labeled CSV path (default: {image_stem}_labels.csv beside the image)",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Save binary, label-index, and color masks when saving.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Rasterized line width used for mask construction. Default: 2.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    save_path = Path(args.output) if args.output else image_path.parent / f"{image_path.stem}_labels.csv"

    print(f"Loading: {image_path}")
    image = load_display_image(image_path)
    h, w = image.shape[:2]
    print(f"  {w}w x {h}h px\n")

    print("=" * 72)
    print("Controls:")
    print("  Left click          add vertex")
    print("  Right click/Enter   finalize current semantic line")
    print("  T                   cycle semantic label")
    print("  0/E                 Edge (default)")
    print("  G                   Gray")
    print("  1                   L1")
    print("  2/3                 L2/3")
    print("  4                   L4")
    print("  5/6/W               L5/6(White)")
    print("  M                   mask selection mode")
    print("    In mask mode: left click selects a closed region; right click assigns layer")
    print("    Points can be drawn outside the image; saved CSV points are clipped to the image.")
    print("  S                   save")
    print("  Q/Esc               save and quit")
    print("=" * 72)
    print(f"Output: {save_path}\n")

    max_in = 14.0
    ratio = w / h
    if ratio >= 1:
        fig_w, fig_h = max_in, max_in / ratio
    else:
        fig_w, fig_h = max_in * ratio, max_in

    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.6))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    draw_margin = max(50, int(round(max(w, h) * 0.05)))
    ax.set_xlim(-draw_margin, w + draw_margin)
    ax.set_ylim(h + draw_margin, -draw_margin)
    ax.axis("off")
    plt.tight_layout(pad=0.3)

    drawer = LabelDrawer(
        fig,
        ax,
        save_path,
        image_shape=image.shape[:2],
        gray_img=image,
        save_mask=args.mask,
        line_width=args.line_width,
        draw_margin=draw_margin,
    )
    # Keep a strong reference while the Matplotlib window is open.  Callback
    # registries may weak-reference bound methods, so an unnamed instance can
    # be garbage-collected and leave clicks with no handler.
    fig._label_drawer = drawer
    plt.show()


if __name__ == "__main__":
    main()
