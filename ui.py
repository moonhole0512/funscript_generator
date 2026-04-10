import asyncio
import flet as ft
import os
import threading
import traceback
import json
from main import run_analysis, pass1_analyze, pass2_extract, UserConfig

# flet-dropzone requires `flet build windows` (VS2022 + Flutter SDK) to work.
HAS_DROPZONE = False
try:
    import flet_dropzone as ftd
except ImportError:
    ftd = None

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.mov', '.avi', '.wmv', '.webm')


class QueueItem:
    """Represents a video in the processing queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    COMPLETE = "complete"
    ERROR = "error"

    def __init__(self, video_path):
        self.video_path = video_path
        self.filename = os.path.basename(video_path)
        self.status = self.PENDING
        self.progress = 0
        self.progress_text = ""
        self.error_message = ""
        self.output_path = ""
        self.video_type = ""
        # Preview data (updated during processing)
        self.preview_b64 = None        # base64 JPEG: current frame with overlays
        self.preview_info = {}         # {'velocity', 'magnitude', 'zoom', 'frame', 'total'}
        self.result_graph_b64 = None   # base64 PNG: position signal + action points
        # Interactive preprocessing (Pass 1.5)
        self.pass1_result = None       # Pass1Result returned by pass1_analyze()
        self.user_config = None        # UserConfig set by UI before pass2 starts
        self._input_event = None       # threading.Event: set() when user clicks "Start Processing"


class ProcessingQueue:
    """Manages a queue of videos to process sequentially."""

    def __init__(self, on_update=None, on_frame=None, on_awaiting_input=None):
        self.items = []
        self.is_processing = False
        self.on_update = on_update
        self.on_frame = on_frame                   # called with (item) when preview frame arrives
        self.on_awaiting_input = on_awaiting_input  # called with (item) when Pass 1 done, waiting for user
        self._lock = threading.Lock()

    def add(self, video_path):
        with self._lock:
            for item in self.items:
                if item.video_path == video_path and item.status != QueueItem.ERROR:
                    return False
            item = QueueItem(video_path)
            self.items.append(item)
        self._notify()
        self._try_process_next()
        return True

    def remove(self, index):
        with self._lock:
            if 0 <= index < len(self.items):
                item = self.items[index]
                if item.status not in (QueueItem.PROCESSING, QueueItem.AWAITING_INPUT):
                    self.items.pop(index)
        self._notify()

    def clear_completed(self):
        with self._lock:
            self.items = [
                item for item in self.items
                if item.status in (QueueItem.PENDING, QueueItem.PROCESSING)
            ]
        self._notify()

    def _try_process_next(self):
        with self._lock:
            if self.is_processing:
                return
            next_item = None
            for item in self.items:
                if item.status == QueueItem.PENDING:
                    next_item = item
                    break
            if next_item is None:
                return
            self.is_processing = True
            next_item.status = QueueItem.PROCESSING

        self._notify()
        thread = threading.Thread(
            target=self._process_item, args=(next_item,), daemon=True
        )
        thread.start()

    def _process_item(self, item):
        def progress_callback(current, total, text):
            if total > 0:
                item.progress = current / total
            if text.startswith("classified:"):
                item.video_type = text.split(":", 1)[1]
                item.progress_text = f"Detected: {item.video_type.replace('_', ' ')}"
            else:
                item.progress_text = text
            self._notify()

        def frame_callback(b64_str, info):
            if info.get('type') == 'result':
                item.result_graph_b64 = b64_str
                item.preview_info = info
            else:
                item.preview_b64 = b64_str
                item.preview_info = info
            if self.on_frame:
                try:
                    self.on_frame(item)
                except Exception:
                    pass

        try:
            # ── Pass 1: Scene detection + per-scene person detection ──────
            r = pass1_analyze(item.video_path, progress_callback=progress_callback)
            if r is None:
                item.status = QueueItem.ERROR
                item.error_message = "Pass 1 (scene detection) failed"
                with self._lock:
                    self.is_processing = False
                self._notify()
                self._try_process_next()
                return

            # ── Pause: wait for user to configure scenes ──────────────────
            item.pass1_result = r
            item.user_config = UserConfig.auto_from_pass1(r)  # default: all enabled
            item._input_event = threading.Event()
            item.status = QueueItem.AWAITING_INPUT
            item.progress_text = "Configure scenes then click Start"
            self._notify()
            if self.on_awaiting_input:
                try:
                    self.on_awaiting_input(item)
                except Exception:
                    pass

            item._input_event.wait()  # blocks until UI calls event.set()

            # ── Pass 2: Motion extraction with user config ─────────────────
            item.status = QueueItem.PROCESSING
            item.progress = 0.0
            item.progress_text = "Pass 2: Extracting motion..."
            self._notify()

            success = pass2_extract(
                item.video_path, r, item.user_config,
                progress_callback=progress_callback,
                frame_callback=frame_callback,
            )
            if success:
                item.status = QueueItem.COMPLETE
                base = os.path.splitext(item.video_path)[0]
                item.output_path = base + ".funscript"
                item.progress = 1.0
                item.progress_text = "Complete"
            else:
                item.status = QueueItem.ERROR
                item.error_message = "Processing failed"
        except Exception as e:
            traceback.print_exc()  # Print full error to terminal
            item.status = QueueItem.ERROR
            item.error_message = str(e)

        with self._lock:
            self.is_processing = False

        self._notify()
        self._try_process_next()

    def _notify(self):
        if self.on_update:
            try:
                self.on_update()
            except Exception:
                pass


async def main(page: ft.Page):
    page.title = "Eroscript Generator AI"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 16

    try:
        page.window.width = 1200
        page.window.height = 780
        page.window.resizable = True
    except Exception:
        try:
            page.window_width = 1200
            page.window_height = 780
            page.window_resizable = True
        except Exception:
            pass

    # Capture the event loop so background threads can schedule updates safely.
    _loop = asyncio.get_running_loop()

    def _schedule_update():
        """Thread-safe page.update() — safe to call from any thread."""
        try:
            _loop.call_soon_threadsafe(page.update)
        except Exception:
            pass

    # ── Preview panel state ──────────────────────────────────────────────
    preview_image = ft.Image(
        src="",
        fit=ft.BoxFit.CONTAIN,
        expand=True,
        visible=False,
    )
    result_graph_image = ft.Image(
        src="",
        fit=ft.BoxFit.CONTAIN,
        width=512,
        height=130,
        visible=False,
    )
    preview_placeholder = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.MONITOR, size=52, color=ft.Colors.GREY_800),
                ft.Text(
                    "Processing preview will appear here",
                    size=13, color=ft.Colors.GREY_700,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Text(
                    "ROI (green box) · Velocity bar (right edge) · ZOOM badge",
                    size=11, color=ft.Colors.GREY_800,
                    text_align=ft.TextAlign.CENTER,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=6,
        ),
        expand=True,
        alignment=ft.Alignment(0, 0),
    )

    # Velocity / info strip below the frame image
    info_velocity = ft.Text("", size=12, color=ft.Colors.GREY_400, font_family="monospace")
    info_frame = ft.Text("", size=12, color=ft.Colors.GREY_500, font_family="monospace")
    info_zoom = ft.Container(
        content=ft.Text("ZOOM", size=11, color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=ft.Colors.RED_700,
        border_radius=4,
        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
        visible=False,
    )
    info_actions = ft.Text("", size=12, color=ft.Colors.GREEN_400, font_family="monospace")

    info_row = ft.Row(
        [info_frame, info_velocity, info_zoom, ft.Container(expand=True), info_actions],
        spacing=12,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # Legend strip (always shown in preview panel)
    legend = ft.Row(
        [
            ft.Container(width=12, height=12, bgcolor=ft.Colors.with_opacity(0.5, "#00FF80"),
                         border_radius=2),
            ft.Text("ROI", size=10, color=ft.Colors.GREY_500),
            ft.Container(width=12, height=12, bgcolor=ft.Colors.with_opacity(0.7, "#00E650"),
                         border_radius=2),
            ft.Text("Insert↑", size=10, color=ft.Colors.GREY_500),
            ft.Container(width=12, height=12, bgcolor=ft.Colors.with_opacity(0.7, "#3C78FF"),
                         border_radius=2),
            ft.Text("Withdraw↓", size=10, color=ft.Colors.GREY_500),
            ft.Container(width=12, height=12, bgcolor=ft.Colors.RED_700, border_radius=2),
            ft.Text("Zoom suppressed", size=10, color=ft.Colors.GREY_500),
        ],
        spacing=5,
    )

    # Graph legend
    graph_legend = ft.Row(
        [
            ft.Container(width=12, height=12,
                         bgcolor=ft.Colors.with_opacity(0.6, ft.Colors.GREEN_800),
                         border_radius=2),
            ft.Text("ACTIVE", size=10, color=ft.Colors.GREY_500),
            ft.Container(width=12, height=12,
                         bgcolor=ft.Colors.with_opacity(0.6, ft.Colors.AMBER_900),
                         border_radius=2),
            ft.Text("TRANSITION", size=10, color=ft.Colors.GREY_500),
            ft.Container(width=12, height=12,
                         bgcolor=ft.Colors.with_opacity(0.3, ft.Colors.GREY_600),
                         border_radius=2),
            ft.Text("QUIET", size=10, color=ft.Colors.GREY_500),
            ft.Container(width=10, height=10, bgcolor=ft.Colors.ORANGE_700,
                         border_radius=5),
            ft.Text("Action pts", size=10, color=ft.Colors.GREY_500),
        ],
        spacing=5,
        visible=False,
    )

    # Frame stack: placeholder vs live image
    frame_stack = ft.Stack(
        [
            preview_placeholder,
            ft.Container(content=preview_image, expand=True, clip_behavior=ft.ClipBehavior.ANTI_ALIAS),
        ],
        expand=True,
    )

    preview_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("Live Preview", size=14, weight=ft.FontWeight.W_600,
                        color=ft.Colors.GREY_300),
                ft.Divider(height=1, color=ft.Colors.GREY_800),
                frame_stack,
                info_row,
                ft.Divider(height=1, color=ft.Colors.GREY_800),
                graph_legend,
                result_graph_image,
                legend,
            ],
            spacing=6,
            expand=True,
        ),
        border=ft.Border.all(1, ft.Colors.GREY_800),
        border_radius=12,
        padding=12,
        width=560,
    )

    # ── Queue UI ─────────────────────────────────────────────────────────
    status_text = ft.Text("", size=13, color=ft.Colors.GREY_500)
    queue_list = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO, expand=True)

    def build_queue_item_row(idx, item):
        if item.status == QueueItem.PENDING:
            icon = ft.Icon(ft.Icons.HOURGLASS_EMPTY, color=ft.Colors.GREY_500, size=20)
            status_widget = ft.Text("Waiting", size=12, color=ft.Colors.GREY_500, width=90)
            progress_widget = ft.Container()
        elif item.status == QueueItem.AWAITING_INPUT:
            icon = ft.Icon(ft.Icons.TUNE, color=ft.Colors.ORANGE_ACCENT, size=20)
            status_widget = ft.TextButton(
                "Configure...", style=ft.ButtonStyle(color=ft.Colors.ORANGE_ACCENT),
                on_click=lambda e, it=item: show_preprocessing_dialog(it),
            )
            progress_widget = ft.Container()
        elif item.status == QueueItem.PROCESSING:
            icon = ft.Icon(ft.Icons.PLAY_CIRCLE_FILL, color=ft.Colors.CYAN_ACCENT, size=20)
            pct = int(item.progress * 100)
            status_widget = ft.Text(f"{pct}%", size=12, color=ft.Colors.CYAN_ACCENT, width=90)
            progress_widget = ft.ProgressBar(
                value=item.progress, width=130,
                color=ft.Colors.CYAN_ACCENT, bgcolor=ft.Colors.GREY_900,
            )
        elif item.status == QueueItem.COMPLETE:
            icon = ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN_ACCENT, size=20)
            status_widget = ft.Text("Complete", size=12, color=ft.Colors.GREEN_ACCENT, width=90)
            progress_widget = ft.Container()
        else:
            icon = ft.Icon(ft.Icons.ERROR, color=ft.Colors.RED_ACCENT, size=20)
            status_widget = ft.Text(
                item.error_message[:28] if item.error_message else "Error",
                size=12, color=ft.Colors.RED_ACCENT, width=90,
                tooltip=item.error_message,
            )
            progress_widget = ft.Container()

        remove_btn = ft.IconButton(
            ft.Icons.CLOSE, icon_size=16, icon_color=ft.Colors.GREY_600,
            tooltip="Remove",
            on_click=lambda e, i=idx: remove_item(i),
            visible=item.status not in (QueueItem.PROCESSING, QueueItem.AWAITING_INPUT),
        )

        # Progress text (below file name)
        prog_text = ft.Text(
            item.progress_text, size=11, color=ft.Colors.GREY_600,
            overflow=ft.TextOverflow.ELLIPSIS,
            visible=bool(item.progress_text and item.status in (QueueItem.PROCESSING, QueueItem.AWAITING_INPUT)),
        )

        if item.video_type == "single_scene":
            badge_text, badge_color = "Single", ft.Colors.TEAL_700
        elif item.video_type == "multi_scene":
            badge_text, badge_color = "Multi", ft.Colors.AMBER_800
        else:
            badge_text, badge_color = "", ft.Colors.TRANSPARENT

        type_badge = ft.Container(
            content=ft.Text(badge_text, size=10, color=ft.Colors.WHITE,
                            weight=ft.FontWeight.W_600),
            bgcolor=badge_color, border_radius=4,
            padding=ft.Padding.symmetric(horizontal=5, vertical=2),
            visible=bool(item.video_type),
        )

        return ft.Container(
            content=ft.Row(
                [
                    icon,
                    ft.Column(
                        [
                            ft.Text(item.filename, size=13, expand=True,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                    tooltip=item.video_path),
                            prog_text,
                        ],
                        spacing=0, expand=True,
                    ),
                    type_badge,
                    progress_widget,
                    status_widget,
                    remove_btn,
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=10, border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
        )

    def _update_preview_panel(processing_item=None):
        """Update preview panel from the currently processing item (or last completed)."""
        item = processing_item
        if item is None:
            for q_item in queue.items:
                if q_item.status == QueueItem.PROCESSING:
                    item = q_item
                    break
        if item is None:
            # Show last completed item's result graph if any
            for q_item in reversed(queue.items):
                if q_item.status == QueueItem.COMPLETE and q_item.result_graph_b64:
                    item = q_item
                    break

        if item is None or (item.preview_b64 is None and item.result_graph_b64 is None):
            preview_image.visible = False
            preview_placeholder.visible = True
            result_graph_image.visible = False
            graph_legend.visible = False
            info_velocity.value = ""
            info_frame.value = ""
            info_zoom.visible = False
            info_actions.value = ""
            return

        # Live frame
        if item.preview_b64:
            preview_image.src = f"data:image/jpeg;base64,{item.preview_b64}"
            preview_image.visible = True
            preview_placeholder.visible = False
        else:
            preview_image.visible = False
            preview_placeholder.visible = (item.result_graph_b64 is None)

        # Info strip
        info = item.preview_info or {}
        if info.get('type') == 'frame':
            frame_i = info.get('frame', 0)
            total_i = info.get('total', 0)
            vel = info.get('velocity', 0.0)
            mag = info.get('magnitude', 0.0)
            zoom = info.get('zoom', False)
            info_frame.value = f"Frame {frame_i}/{total_i}"
            info_velocity.value = f"vel={vel:+.3f}  mag={mag:.3f}"
            info_zoom.visible = zoom
            info_actions.value = ""
        elif info.get('type') == 'result':
            n_act = info.get('actions', 0)
            act_s = info.get('active_segs', 0)
            q_s = info.get('quiet_segs', 0)
            info_frame.value = "Done"
            info_velocity.value = ""
            info_zoom.visible = False
            info_actions.value = f"{n_act} actions | {act_s}A/{q_s}Q segs"
        else:
            info_frame.value = item.progress_text or ""
            info_velocity.value = ""
            info_zoom.visible = False
            info_actions.value = ""

        # Result graph
        if item.result_graph_b64:
            result_graph_image.src = f"data:image/png;base64,{item.result_graph_b64}"
            result_graph_image.visible = True
            graph_legend.visible = True
        else:
            result_graph_image.visible = False
            graph_legend.visible = False

    def rebuild_queue_ui():
        queue_list.controls.clear()
        for idx, item in enumerate(queue.items):
            queue_list.controls.append(build_queue_item_row(idx, item))

        pending = sum(1 for i in queue.items if i.status == QueueItem.PENDING)
        processing = sum(1 for i in queue.items if i.status == QueueItem.PROCESSING)
        awaiting = sum(1 for i in queue.items if i.status == QueueItem.AWAITING_INPUT)
        complete = sum(1 for i in queue.items if i.status == QueueItem.COMPLETE)

        parts = []
        if processing:
            parts.append(f"{processing} processing")
        if awaiting:
            parts.append(f"{awaiting} awaiting input")
        if pending:
            parts.append(f"{pending} waiting")
        if complete:
            parts.append(f"{complete} done")
        status_text.value = " | ".join(parts) if parts else ""

        clear_btn.visible = complete > 0 or any(
            i.status == QueueItem.ERROR for i in queue.items
        )

        _update_preview_panel()
        _schedule_update()

    def on_frame_update(item):
        """Called from background thread when a preview frame arrives."""
        _update_preview_panel(item)
        _schedule_update()

    # ── Preprocessing Dialog (F3/F1/F2 tabs) ─────────────────────────────
    _RESIZE_W = 512   # algorithms.RESIZE_WIDTH — preview frames are this wide
    _DISP_W   = 460   # dialog preview display width
    _DISP_H   = int(_DISP_W * 9 / 16)  # 16:9 display height
    _FSCALE   = _RESIZE_W / _DISP_W    # display→frame coordinate scale

    def show_preprocessing_dialog(item):
        """Build and open the 3-tab scene configuration dialog (F3 / F1 / F2)."""
        r = item.pass1_result
        cfg = item.user_config
        if r is None or cfg is None:
            return

        fps = r.fps or 30.0
        n_sc = len(r.scene_boundaries)
        all_persons = r.per_scene_persons or []
        all_previews = r.per_scene_previews or []

        # ── Helper: build thumbnail for a scene ───────────────────────────
        def _thumb(i, w=96, h=54):
            b64 = all_previews[i] if i < len(all_previews) else None
            if b64:
                return ft.Image(src=f"data:image/jpeg;base64,{b64}",
                                width=w, height=h, fit=ft.BoxFit.CONTAIN, border_radius=4)
            return ft.Container(width=w, height=h, bgcolor=ft.Colors.GREY_800, border_radius=4,
                                alignment=ft.Alignment(0, 0),
                                content=ft.Icon(ft.Icons.IMAGE_NOT_SUPPORTED,
                                                color=ft.Colors.GREY_600, size=16))

        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — F3: Scene ON/OFF
        # ══════════════════════════════════════════════════════════════════
        toggles = []
        scene_rows = []
        for i, (bounds, sc) in enumerate(zip(r.scene_boundaries, cfg.scene_configs)):
            start_s = bounds[0] / fps
            end_s   = bounds[1] / fps
            n_p = len(all_persons[i]) if i < len(all_persons) else 0

            sw = ft.Switch(value=sc.enabled, active_color=ft.Colors.CYAN_ACCENT, scale=0.8)
            toggles.append(sw)

            badge = ft.Container(
                content=ft.Text(f"{n_p}P", size=10, color=ft.Colors.WHITE,
                                weight=ft.FontWeight.BOLD),
                bgcolor=ft.Colors.TEAL_700 if n_p >= 2 else ft.Colors.GREY_700,
                border_radius=4, padding=ft.Padding.symmetric(horizontal=4, vertical=2),
            )
            scene_rows.append(ft.Container(
                content=ft.Row(
                    [sw, _thumb(i), ft.Column([
                        ft.Text(f"Scene {i + 1}", size=13, weight=ft.FontWeight.W_500),
                        ft.Text(f"{start_s:.1f}s – {end_s:.1f}s  ({end_s - start_s:.1f}s)",
                                size=11, color=ft.Colors.GREY_500),
                        badge,
                    ], spacing=2)],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10,
                ),
                padding=ft.Padding.symmetric(horizontal=8, vertical=6),
                border_radius=6, bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
            ))

        def _f3_all_on(e):
            for sw in toggles:
                sw.value = True
            _schedule_update()

        def _f3_yolo_only(e):
            for i2, sw in enumerate(toggles):
                n = len(all_persons[i2]) if i2 < len(all_persons) else 0
                sw.value = (n >= 1)
            _schedule_update()

        f3_content = ft.Column([
            ft.Text("씬 ON/OFF로 funscript 생성 구간을 선택하세요.",
                    size=11, color=ft.Colors.GREY_400),
            ft.Row([
                ft.TextButton("All ON", on_click=_f3_all_on,
                              style=ft.ButtonStyle(color=ft.Colors.CYAN_ACCENT)),
                ft.TextButton("YOLO scenes only", on_click=_f3_yolo_only,
                              style=ft.ButtonStyle(color=ft.Colors.TEAL_ACCENT)),
            ], spacing=4),
            ft.Divider(height=1, color=ft.Colors.GREY_800),
            ft.Column(scene_rows, spacing=4, scroll=ft.ScrollMode.AUTO, height=360),
        ], spacing=8)

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Per-scene person selection (P1/P2 dropdown)
        # ══════════════════════════════════════════════════════════════════
        # Find first enabled scene for initial tab view
        initial_enabled_idx = 0
        for idx2, sc2 in enumerate(cfg.scene_configs):
            if sc2.enabled:
                initial_enabled_idx = idx2
                break
        f1_idx = [initial_enabled_idx]
        # Use transparent 1x1 pixel gif as dummy src to prevent 'A valid src value must be specified'
        dummy_src = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        f1_img      = ft.Image(src=dummy_src, width=_DISP_W, height=_DISP_H, fit=ft.BoxFit.CONTAIN,
                               visible=False)
        f1_no_prev  = ft.Container(width=_DISP_W, height=_DISP_H, bgcolor=ft.Colors.GREY_900,
                                   border_radius=8, alignment=ft.Alignment(0, 0),
                                   content=ft.Text("No preview", color=ft.Colors.GREY_700))
        f1_lbl      = ft.Text("", size=13, weight=ft.FontWeight.W_500)
        f1_cnt      = ft.Text("", size=11, color=ft.Colors.GREY_500)
        f1_p1_dd    = ft.Dropdown(label="P1 (Primary)", width=200, dense=True)
        f1_p2_dd    = ft.Dropdown(label="P2 (Secondary)", width=200, dense=True)

        def _f1_refresh():
            i = f1_idx[0]
            persons = all_persons[i] if i < len(all_persons) else []
            b64     = all_previews[i] if i < len(all_previews) else None
            sc      = cfg.scene_configs[i]

            f1_lbl.value = f"Scene {i + 1} / {n_sc}"
            f1_cnt.value = f"{len(persons)} person(s) detected"

            if b64:
                f1_img.src     = f"data:image/jpeg;base64,{b64}"
                f1_img.visible = True
                f1_no_prev.visible = False
            else:
                f1_img.visible = False
                f1_no_prev.visible = True

            p_opts = [ft.dropdown.Option(str(j), f"Person {j + 1}")
                      for j in range(len(persons))]
            if not p_opts:
                p_opts = [ft.dropdown.Option("0", "(none detected)")]
            f1_p1_dd.options = p_opts
            f1_p1_dd.value   = str(sc.p1_person_idx) if sc.p1_person_idx < len(persons) else "0"

            p2_opts = [ft.dropdown.Option("-1", "None (single person)")] + [
                ft.dropdown.Option(str(j), f"Person {j + 1}") for j in range(len(persons))
            ]
            f1_p2_dd.options = p2_opts
            f1_p2_dd.value   = (str(sc.p2_person_idx)
                                 if 0 <= sc.p2_person_idx < len(persons) else "-1")
            page.update()

        def _f1_prev(e):
            curr = f1_idx[0]
            while curr > 0:
                curr -= 1
                if toggles[curr].value:
                    f1_idx[0] = curr
                    _f1_refresh()
                    return

        def _f1_next(e):
            curr = f1_idx[0]
            while curr < n_sc - 1:
                curr += 1
                if toggles[curr].value:
                    f1_idx[0] = curr
                    _f1_refresh()
                    return

        def _f1_p1_changed(e):
            try:
                cfg.scene_configs[f1_idx[0]].p1_person_idx = int(e.control.value)
            except (ValueError, IndexError):
                pass

        def _f1_p2_changed(e):
            try:
                cfg.scene_configs[f1_idx[0]].p2_person_idx = int(e.control.value)
            except (ValueError, IndexError):
                pass

        def _f1_apply_all(e):
            src = cfg.scene_configs[f1_idx[0]]
            for sc2 in cfg.scene_configs:
                sc2.p1_person_idx = src.p1_person_idx
                sc2.p2_person_idx = src.p2_person_idx

        f1_p1_dd.on_change = _f1_p1_changed
        f1_p2_dd.on_change = _f1_p2_changed

        f1_content = ft.Column([
            ft.Text("씬별 P1(기준 인물)/P2(상대) 인물을 선택하세요. 엑스트라를 제외할 수 있습니다.",
                    size=11, color=ft.Colors.GREY_400),
            ft.Row([
                ft.IconButton(ft.Icons.CHEVRON_LEFT, icon_color=ft.Colors.GREY_400,
                              on_click=_f1_prev, tooltip="Previous scene"),
                f1_lbl,
                ft.IconButton(ft.Icons.CHEVRON_RIGHT, icon_color=ft.Colors.GREY_400,
                              on_click=_f1_next, tooltip="Next scene"),
                ft.Container(expand=True),
                f1_cnt,
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Stack([f1_no_prev, f1_img]),
            ft.Row([f1_p1_dd, f1_p2_dd], spacing=12),
            ft.TextButton("Apply this P1/P2 to ALL scenes",
                          on_click=_f1_apply_all,
                          style=ft.ButtonStyle(color=ft.Colors.AMBER_ACCENT)),
        ], spacing=8)

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — Manual hip pixel click input
        # ══════════════════════════════════════════════════════════════════
        f2_idx  = [initial_enabled_idx]
        f2_mode = ['p1']  # 'p1' | 'p2' | 'clear'

        f2_img      = ft.Image(src=dummy_src, width=_DISP_W, height=_DISP_H, fit=ft.BoxFit.CONTAIN)
        f2_no_prev  = ft.Container(width=_DISP_W, height=_DISP_H, bgcolor=ft.Colors.GREY_900,
                                   border_radius=8, alignment=ft.Alignment(0, 0),
                                   content=ft.Text("No preview", color=ft.Colors.GREY_700))
        f2_p1_dot   = ft.Container(width=14, height=14, bgcolor=ft.Colors.RED_ACCENT,
                                   border_radius=7, visible=False, left=0, top=0)
        f2_p2_dot   = ft.Container(width=14, height=14, bgcolor=ft.Colors.BLUE_ACCENT,
                                   border_radius=7, visible=False, left=0, top=0)
        f2_lbl      = ft.Text("", size=13, weight=ft.FontWeight.W_500)
        f2_note     = ft.Text("", size=11, color=ft.Colors.GREY_500)
        f2_coords   = ft.Text("", size=11, color=ft.Colors.GREY_400, font_family="monospace")
        f2_p1_btn   = ft.Button("P1 Hip",  bgcolor=ft.Colors.RED_900,
                                color=ft.Colors.WHITE, height=30)
        f2_p2_btn   = ft.Button("P2 Hip",  bgcolor=ft.Colors.GREY_800,
                                color=ft.Colors.WHITE, height=30)
        f2_clr_btn  = ft.Button("Clear",   bgcolor=ft.Colors.GREY_800,
                                color=ft.Colors.WHITE, height=30)

        def _f2_update_btns():
            m = f2_mode[0]
            f2_p1_btn.bgcolor  = ft.Colors.RED_900  if m == 'p1'    else ft.Colors.GREY_800
            f2_p2_btn.bgcolor  = ft.Colors.BLUE_900 if m == 'p2'    else ft.Colors.GREY_800
            f2_clr_btn.bgcolor = ft.Colors.GREY_700 if m == 'clear' else ft.Colors.GREY_800

        def _f2_set_p1(e):
            f2_mode[0] = 'p1';    _f2_update_btns(); page.update()

        def _f2_set_p2(e):
            f2_mode[0] = 'p2';    _f2_update_btns(); page.update()

        def _f2_set_clr(e):
            f2_mode[0] = 'clear'; _f2_update_btns(); page.update()

        f2_p1_btn.on_click  = _f2_set_p1
        f2_p2_btn.on_click  = _f2_set_p2
        f2_clr_btn.on_click = _f2_set_clr

        def _f2_refresh():
            i  = f2_idx[0]
            sc = cfg.scene_configs[i]
            persons = all_persons[i] if i < len(all_persons) else []
            b64     = all_previews[i] if i < len(all_previews) else None

            f2_lbl.value  = f"Scene {i + 1} / {n_sc}"
            f2_note.value = ("(no YOLO detection — manual input recommended)"
                             if not persons else f"({len(persons)} person(s) detected)")
            if b64:
                f2_img.src     = f"data:image/jpeg;base64,{b64}"
                f2_img.visible = True
                f2_no_prev.visible = False
            else:
                f2_img.src     = dummy_src
                f2_img.visible = False
                f2_no_prev.visible = True

            def _dot_pos(hip_px):
                """frame-pixel → display-pixel (top-left of 14px dot)."""
                return hip_px[0] / _FSCALE - 7, hip_px[1] / _FSCALE - 7

            if sc.p1_hip_px:
                dx, dy = _dot_pos(sc.p1_hip_px)
                f2_p1_dot.left, f2_p1_dot.top, f2_p1_dot.visible = dx, dy, True
            else:
                f2_p1_dot.visible = False

            if sc.p2_hip_px:
                dx, dy = _dot_pos(sc.p2_hip_px)
                f2_p2_dot.left, f2_p2_dot.top, f2_p2_dot.visible = dx, dy, True
            else:
                f2_p2_dot.visible = False

            p1_t = (f"P1: ({sc.p1_hip_px[0]}, {sc.p1_hip_px[1]})"
                    if sc.p1_hip_px else "P1: (unset)")
            p2_t = (f"P2: ({sc.p2_hip_px[0]}, {sc.p2_hip_px[1]})"
                    if sc.p2_hip_px else "P2: (unset)")
            f2_coords.value = f"{p1_t}   {p2_t}"
            page.update()

        def _f2_prev(e):
            curr = f2_idx[0]
            while curr > 0:
                curr -= 1
                if toggles[curr].value:
                    f2_idx[0] = curr
                    _f2_refresh()
                    return

        def _f2_next(e):
            curr = f2_idx[0]
            while curr < n_sc - 1:
                curr += 1
                if toggles[curr].value:
                    f2_idx[0] = curr
                    _f2_refresh()
                    return

        def _f2_tap(e):
            i  = f2_idx[0]
            sc = cfg.scene_configs[i]
            lp = e.local_position
            if lp is None:
                return
            px = int(lp.x * _FSCALE)
            py = int(lp.y * _FSCALE)
            m  = f2_mode[0]
            if m == 'p1':
                sc.p1_hip_px = (px, py)
                sc.mode = 'manual'
            elif m == 'p2':
                sc.p2_hip_px = (px, py)
                sc.mode = 'manual'
            elif m == 'clear':
                sc.p1_hip_px = sc.p2_hip_px = None
                sc.mode = 'auto'
            _f2_refresh()

        f2_gesture = ft.GestureDetector(
            content=ft.Stack([
                ft.Container(
                    content=ft.Stack([f2_no_prev, f2_img]),
                    width=_DISP_W, height=_DISP_H,
                    bgcolor=ft.Colors.GREY_900,
                    border_radius=8,
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                ),
                f2_p1_dot,
                f2_p2_dot,
            ], width=_DISP_W, height=_DISP_H),
            on_tap_down=_f2_tap,
        )

        f2_content = ft.Column([
            ft.Text(
                "YOLO 탐지 실패 씬: 클릭으로 골반 위치를 지정하세요. 수동 앵커로 사용됩니다.",
                size=11, color=ft.Colors.GREY_400,
            ),
            ft.Row([
                ft.IconButton(ft.Icons.CHEVRON_LEFT, icon_color=ft.Colors.GREY_400,
                              on_click=_f2_prev, tooltip="Previous scene"),
                f2_lbl, f2_note,
                ft.IconButton(ft.Icons.CHEVRON_RIGHT, icon_color=ft.Colors.GREY_400,
                              on_click=_f2_next, tooltip="Next scene"),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
            f2_gesture,
            ft.Row([f2_p1_btn, f2_p2_btn, f2_clr_btn], spacing=8),
            f2_coords,
        ], spacing=8)

        # ── F4: Physics / Post-processing Options ───────────────────────────
        bounce_slider = ft.Slider(
            min=0, max=100, divisions=20,
            value=cfg.impact_bounce_intensity,
            label="{value}%",
            active_color=ft.Colors.CYAN_ACCENT,
        )
        auto_floor_switch = ft.Switch(
            label="Auto Floor Alignment",
            value=cfg.auto_floor_align,
            active_color=ft.Colors.CYAN_ACCENT,
            scale=0.9
        )
        physics_content = ft.Column([
            ft.Text("Physics & Post-processing", size=14, weight=ft.FontWeight.BOLD),
            ft.Text("Impact Rebound Intensity (Bottom Bounce)", size=12, color=ft.Colors.GREY_400),
            ft.Row([
                ft.Icon(ft.Icons.VIBRATION, size=20, color=ft.Colors.CYAN_400),
                bounce_slider,
                ft.Text("Strong", size=11, color=ft.Colors.GREY_500),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text("Adds '0 > 20 > 0' style rebound pulses when hitting the bottom. "
                    "Higher value increases the secondary bounce height.", 
                    size=11, color=ft.Colors.GREY_500, italic=True),
            ft.Divider(height=10, color=ft.Colors.with_opacity(0.1, ft.Colors.WHITE)),
            ft.Text("Signal Correction", size=12, color=ft.Colors.GREY_400),
            auto_floor_switch,
            ft.Text("Automatically shifts the floor to 0 if the strokes are shallow. "
                    "Ensures bounces are triggered properly.",
                    size=11, color=ft.Colors.GREY_500, italic=True),
        ], spacing=15)

        # ══════════════════════════════════════════════════════════════════
        # Assemble dialog with 3 tabs + Global Options
        # ══════════════════════════════════════════════════════════════════
        report_switch = ft.Switch(
            label="Generate Analysis Report (HTML/JSON)",
            value=cfg.generate_report,
            active_color=ft.Colors.CYAN_ACCENT,
            scale=0.9
        )
        _started = [False]

        def on_start(e):
            if _started[0]:
                return
            _started[0] = True
            # Flush settings into cfg
            for sw, sc2 in zip(toggles, cfg.scene_configs):
                sc2.enabled = sw.value
            cfg.generate_report = report_switch.value
            cfg.impact_bounce_intensity = bounce_slider.value
            cfg.auto_floor_align = auto_floor_switch.value
            item.user_config = cfg
            dialog.open = False
            page.update()
            if item._input_event:
                item._input_event.set()

        def _on_tab_change(e):
            # If switching to Persons (1) or Manual Hip (2) tab,
            # ensure the current scene index is pointing to an enabled scene.
            if e.control.selected_index == 1: # Persons
                if not toggles[f1_idx[0]].value:
                    for i2, sw2 in enumerate(toggles):
                        if sw2.value:
                            f1_idx[0] = i2
                            _f1_refresh()
                            break
            elif e.control.selected_index == 2: # Manual Hip
                if not toggles[f2_idx[0]].value:
                    for i2, sw2 in enumerate(toggles):
                        if sw2.value:
                            f2_idx[0] = i2
                            _f2_refresh()
                            break

        _tab_pad = ft.Padding.symmetric(horizontal=8, vertical=10)
        tabs = ft.Tabs(
            content=ft.Column([
                ft.TabBar(tabs=[
                    ft.Tab(label="Scenes"),
                    ft.Tab(label="Persons"),
                    ft.Tab(label="Manual Hip"),
                    ft.Tab(label="Physics"),
                ]),
                ft.TabBarView(controls=[
                    ft.Container(content=f3_content, padding=_tab_pad),
                    ft.Container(content=f1_content, padding=_tab_pad),
                    ft.Container(content=f2_content, padding=_tab_pad),
                    ft.Container(content=physics_content, padding=_tab_pad),
                ], expand=True),
            ]),
            length=4,
            selected_index=0,
            on_change=_on_tab_change,
            expand=True,
        )

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Configure — {item.filename}",
                          size=14, weight=ft.FontWeight.W_600),
            content=ft.Container(content=tabs, width=520, height=510),
            actions=[
                ft.Row([
                    report_switch,
                    ft.Container(expand=True),
                    ft.Button(
                        "Start Processing",
                        icon=ft.Icons.PLAY_ARROW,
                        on_click=on_start,
                        bgcolor=ft.Colors.CYAN_700,
                        color=ft.Colors.WHITE,
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, spacing=10, 
                   vertical_alignment=ft.CrossAxisAlignment.CENTER)
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=on_start,
        )

        # Initialize dynamic tab views
        _f1_refresh()
        _f2_refresh()

        if not hasattr(page, 'overlay') or page.overlay is None:
            page.dialog = dialog
            dialog.open = True
        else:
            page.overlay.append(dialog)
            dialog.open = True
        page.update()

    def on_awaiting_input(item):
        """Called from background thread when Pass 1 completes — auto-open dialog."""
        _loop.call_soon_threadsafe(_schedule_show_dialog, item)

    def _schedule_show_dialog(item):
        show_preprocessing_dialog(item)

    queue = ProcessingQueue(
        on_update=rebuild_queue_ui,
        on_frame=on_frame_update,
        on_awaiting_input=on_awaiting_input,
    )

    def remove_item(idx):
        queue.remove(idx)

    def clear_completed(e):
        queue.clear_completed()

    clear_btn = ft.TextButton(
        "Clear Completed", icon=ft.Icons.CLEANING_SERVICES,
        on_click=clear_completed, visible=False,
    )

    # ── File Picker ──────────────────────────────────────────────────────
    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    async def open_file_picker(e):
        results = await file_picker.pick_files(
            allow_multiple=True,
            file_type=ft.FilePickerFileType.VIDEO,
        )
        if results:
            added = 0
            for f in results:
                if f.path and f.path.lower().endswith(VIDEO_EXTENSIONS):
                    if queue.add(f.path):
                        added += 1
            if added == 0:
                status_text.value = "No valid video files selected or already in queue"
                _schedule_update()

    def add_video_files(file_paths):
        added = 0
        for path in file_paths:
            p = path.strip().strip('"').strip("'")
            if p and p.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(p):
                if queue.add(p):
                    added += 1
        if added == 0 and file_paths:
            status_text.value = "No valid video files in selection"
            _schedule_update()

    # ── Drop Zone ────────────────────────────────────────────────────────
    drop_zone_default_bg = ft.Colors.with_opacity(0.05, ft.Colors.CYAN_600)
    drop_zone_hover_bg = ft.Colors.with_opacity(0.12, ft.Colors.CYAN_ACCENT)
    drop_zone_drag_bg = ft.Colors.with_opacity(0.20, ft.Colors.CYAN_ACCENT)
    drop_label = "Drop video files here or click to browse" if HAS_DROPZONE else "Click to add video files"

    drop_content = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.VIDEO_LIBRARY, size=36, color=ft.Colors.CYAN_ACCENT),
                ft.Text(drop_label, size=15, weight=ft.FontWeight.W_500,
                        text_align=ft.TextAlign.CENTER),
                ft.Text("Supports .mp4 .mkv .mov .avi .wmv .webm",
                        size=11, color=ft.Colors.GREY_600,
                        text_align=ft.TextAlign.CENTER),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=6,
        ),
        border=ft.Border.all(2, ft.Colors.CYAN_700),
        border_radius=12,
        padding=ft.Padding.symmetric(horizontal=20, vertical=18),
        alignment=ft.Alignment(0, 0),
        bgcolor=drop_zone_default_bg,
        on_click=open_file_picker,
        on_hover=lambda e: (
            setattr(e.control, "bgcolor",
                    drop_zone_hover_bg if e.data == "true" else drop_zone_default_bg),
            _schedule_update(),
        ),
    )

    if HAS_DROPZONE:
        def on_files_dropped(e):
            if hasattr(e, 'files') and e.files:
                paths = e.files if isinstance(e.files, list) else [e.files]
                add_video_files(paths)

        def on_drag_entered(e):
            drop_content.bgcolor = drop_zone_drag_bg
            drop_content.border = ft.Border.all(3, ft.Colors.CYAN_ACCENT)
            _schedule_update()

        def on_drag_exited(e):
            drop_content.bgcolor = drop_zone_default_bg
            drop_content.border = ft.Border.all(2, ft.Colors.CYAN_700)
            _schedule_update()

        drop_zone = ftd.Dropzone(
            content=drop_content,
            on_dropped=on_files_dropped,
            on_entered=on_drag_entered,
            on_exited=on_drag_exited,
        )
    else:
        drop_zone = drop_content

    # ── Queue Section ────────────────────────────────────────────────────
    queue_header = ft.Row(
        [
            ft.Text("Queue", size=15, weight=ft.FontWeight.W_600),
            ft.Container(expand=True),
            status_text,
            clear_btn,
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    queue_container = ft.Container(
        content=ft.Column(
            [queue_header, ft.Divider(height=1, color=ft.Colors.GREY_800), queue_list],
            spacing=8, expand=True,
        ),
        border=ft.Border.all(1, ft.Colors.GREY_800),
        border_radius=12,
        padding=14,
        expand=True,
    )

    # ── Page Layout ──────────────────────────────────────────────────────
    title = ft.Text(
        "Eroscript Generator AI",
        size=30, weight=ft.FontWeight.BOLD,
        color=ft.Colors.PRIMARY, text_align=ft.TextAlign.CENTER,
    )
    subtitle = ft.Text(
        "Transform video motion into precision haptic scripts",
        size=13, color=ft.Colors.GREY_400, text_align=ft.TextAlign.CENTER,
    )
    
    # ── Tracker Settings ─────────────────────────────────────────────────
    from algorithms import config as global_config
    
    current_model = global_config.get('tracker', 'model_type', 'yolo')
    current_reid = global_config.get('tracking', 'use_reid', True)
    
    def on_tracker_change(e):
        new_val = "humanart" if tracker_toggle.value else "yolo"
        global_config.set('tracker', 'model_type', new_val)
        status_text.value = f"Tracker changed to: {new_val.upper()}"
        _schedule_update()

    def on_reid_change(e):
        new_val = reid_switch.value
        global_config.set('tracking', 'use_reid', new_val)
        status_text.value = f"OSNet ReID: {'ON' if new_val else 'OFF'}"
        _schedule_update()

    tracker_toggle = ft.Switch(
        label="HumanArt (Animation Engine)",
        value=(current_model == "humanart"),
        active_color=ft.Colors.CYAN_ACCENT,
        on_change=on_tracker_change,
    )
    
    reid_switch = ft.Switch(
        label="OSNet ReID (ID Fix)",
        value=current_reid,
        active_color=ft.Colors.PINK_ACCENT,
        on_change=on_reid_change,
    )
    
    settings_row = ft.Row(
        [
            ft.Text("Engine:", size=13, color=ft.Colors.GREY_400),
            tracker_toggle,
            ft.VerticalDivider(width=1, color=ft.Colors.GREY_800),
            reid_switch,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=20,
    )

    page.add(
        ft.Column(
            [
                ft.Container(height=6),
                title,
                subtitle,
                ft.Container(height=6),
                settings_row,
                ft.Container(height=10),
                drop_zone,
                ft.Container(height=10),
                ft.Row(
                    [queue_container, preview_panel],
                    spacing=12,
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )
    )

    page.update()


if __name__ == "__main__":
    ft.run(main)
