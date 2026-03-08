import asyncio
import flet as ft
import os
import threading
from main import run_analysis

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


class ProcessingQueue:
    """Manages a queue of videos to process sequentially."""

    def __init__(self, on_update=None, on_frame=None):
        self.items = []
        self.is_processing = False
        self.on_update = on_update
        self.on_frame = on_frame   # called with (item) when preview frame arrives
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
                if self.items[index].status != QueueItem.PROCESSING:
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
            success = run_analysis(
                item.video_path,
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
            visible=item.status != QueueItem.PROCESSING,
        )

        # Progress text (below file name)
        prog_text = ft.Text(
            item.progress_text, size=11, color=ft.Colors.GREY_600,
            overflow=ft.TextOverflow.ELLIPSIS,
            visible=bool(item.progress_text and item.status == QueueItem.PROCESSING),
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
        complete = sum(1 for i in queue.items if i.status == QueueItem.COMPLETE)

        parts = []
        if processing:
            parts.append(f"{processing} processing")
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

    queue = ProcessingQueue(on_update=rebuild_queue_ui, on_frame=on_frame_update)

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

    page.add(
        ft.Column(
            [
                ft.Container(height=6),
                title,
                subtitle,
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
