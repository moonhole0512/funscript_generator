import flet as ft
import os
import threading
from main import run_analysis

# flet-dropzone requires `flet build windows` (VS2022 + Flutter SDK) to work.
# Disabled by default. Set to True only after running `flet build windows`.
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
        self.video_type = ""  # "single_scene" or "multi_scene"


class ProcessingQueue:
    """Manages a queue of videos to process sequentially."""

    def __init__(self, on_update=None):
        self.items = []
        self.is_processing = False
        self.on_update = on_update
        self._lock = threading.Lock()

    def add(self, video_path):
        """Add a video to the queue. Returns True if added, False if duplicate."""
        with self._lock:
            # Check for duplicate
            for item in self.items:
                if item.video_path == video_path and item.status != QueueItem.ERROR:
                    return False
            item = QueueItem(video_path)
            self.items.append(item)
        self._notify()
        self._try_process_next()
        return True

    def remove(self, index):
        """Remove item at index if it's not currently processing."""
        with self._lock:
            if 0 <= index < len(self.items):
                if self.items[index].status != QueueItem.PROCESSING:
                    self.items.pop(index)
        self._notify()

    def clear_completed(self):
        """Remove all completed and errored items."""
        with self._lock:
            self.items = [
                item for item in self.items
                if item.status in (QueueItem.PENDING, QueueItem.PROCESSING)
            ]
        self._notify()

    def _try_process_next(self):
        """Start processing next pending item if not already processing."""
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
        """Process a single video item."""
        def progress_callback(current, total, text):
            if total > 0:
                item.progress = current / total
            if text.startswith("classified:"):
                item.video_type = text.split(":", 1)[1]
                item.progress_text = f"Detected: {item.video_type.replace('_', ' ')}"
            else:
                item.progress_text = text
            self._notify()

        try:
            success = run_analysis(item.video_path, progress_callback=progress_callback)
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


def main(page: ft.Page):
    # --- Page Configuration ---
    page.title = "Eroscript Generator AI"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20

    # Window properties (Flet 0.80+ uses page.window object)
    try:
        page.window.width = 900
        page.window.height = 750
        page.window.resizable = True
    except Exception:
        # Fallback for older Flet versions
        try:
            page.window_width = 900
            page.window_height = 750
            page.window_resizable = True
        except Exception:
            pass

    # --- Queue ---
    queue = ProcessingQueue(on_update=lambda: rebuild_queue_ui())

    # --- UI Components ---
    title = ft.Text(
        "Eroscript Generator AI",
        size=36,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.PRIMARY,
        text_align=ft.TextAlign.CENTER,
    )

    subtitle = ft.Text(
        "Transform video motion into precision haptic scripts",
        size=14,
        color=ft.Colors.GREY_400,
        text_align=ft.TextAlign.CENTER,
    )

    status_text = ft.Text("", size=13, color=ft.Colors.GREY_500)

    queue_list = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO, expand=True)

    def build_queue_item_row(idx, item):
        """Build a UI row for a queue item."""
        if item.status == QueueItem.PENDING:
            icon = ft.Icon(ft.Icons.HOURGLASS_EMPTY, color=ft.Colors.GREY_500, size=20)
            status_widget = ft.Text("Waiting", size=12, color=ft.Colors.GREY_500, width=100)
            progress_widget = ft.Container()
        elif item.status == QueueItem.PROCESSING:
            icon = ft.Icon(ft.Icons.PLAY_CIRCLE_FILL, color=ft.Colors.CYAN_ACCENT, size=20)
            pct = int(item.progress * 100)
            status_widget = ft.Text(
                f"{pct}%", size=12, color=ft.Colors.CYAN_ACCENT, width=100
            )
            progress_widget = ft.ProgressBar(
                value=item.progress,
                width=150,
                color=ft.Colors.CYAN_ACCENT,
                bgcolor=ft.Colors.GREY_900,
            )
        elif item.status == QueueItem.COMPLETE:
            icon = ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN_ACCENT, size=20)
            status_widget = ft.Text("Complete", size=12, color=ft.Colors.GREEN_ACCENT, width=100)
            progress_widget = ft.Container()
        else:  # ERROR
            icon = ft.Icon(ft.Icons.ERROR, color=ft.Colors.RED_ACCENT, size=20)
            status_widget = ft.Text(
                item.error_message[:30] if item.error_message else "Error",
                size=12, color=ft.Colors.RED_ACCENT, width=100,
                tooltip=item.error_message,
            )
            progress_widget = ft.Container()

        # Remove button (only for non-processing items)
        remove_btn = ft.IconButton(
            ft.Icons.CLOSE,
            icon_size=16,
            icon_color=ft.Colors.GREY_600,
            tooltip="Remove",
            on_click=lambda e, i=idx: remove_item(i),
            visible=item.status != QueueItem.PROCESSING,
        )

        # Video type classification badge
        if item.video_type == "single_scene":
            badge_text = "Single Scene"
            badge_color = ft.Colors.TEAL_700
        elif item.video_type == "multi_scene":
            badge_text = "Multi Scene"
            badge_color = ft.Colors.AMBER_800
        else:
            badge_text = ""
            badge_color = ft.Colors.TRANSPARENT

        type_badge = ft.Container(
            content=ft.Text(
                badge_text,
                size=10,
                color=ft.Colors.WHITE,
                weight=ft.FontWeight.W_600,
            ),
            bgcolor=badge_color,
            border_radius=4,
            padding=ft.padding.symmetric(horizontal=6, vertical=2),
            visible=bool(item.video_type),
        )

        return ft.Container(
            content=ft.Row(
                [
                    icon,
                    ft.Text(
                        item.filename,
                        size=13,
                        expand=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        tooltip=item.video_path,
                    ),
                    type_badge,
                    progress_widget,
                    status_widget,
                    remove_btn,
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=10,
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
        )

    def rebuild_queue_ui():
        """Rebuild the queue list UI from current state."""
        queue_list.controls.clear()
        for idx, item in enumerate(queue.items):
            queue_list.controls.append(build_queue_item_row(idx, item))

        # Update status
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

        # Show/hide clear button
        clear_btn.visible = complete > 0 or any(
            i.status == QueueItem.ERROR for i in queue.items
        )

        try:
            page.update()
        except Exception:
            pass

    def remove_item(idx):
        queue.remove(idx)

    def clear_completed(e):
        queue.clear_completed()

    clear_btn = ft.TextButton(
        "Clear Completed",
        icon=ft.Icons.CLEANING_SERVICES,
        on_click=clear_completed,
        visible=False,
    )

    # --- File Picker ---
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
                page.update()

    # --- Helper: add files from paths ---
    def add_video_files(file_paths):
        """Add video files from a list of file paths."""
        added = 0
        for path in file_paths:
            p = path.strip().strip('"').strip("'")
            if p and p.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(p):
                if queue.add(p):
                    added += 1
        if added == 0 and file_paths:
            status_text.value = "No valid video files in selection"
            try:
                page.update()
            except Exception:
                pass

    # --- Drop Zone ---
    drop_zone_default_bg = ft.Colors.with_opacity(0.05, ft.Colors.CYAN_600)
    drop_zone_hover_bg = ft.Colors.with_opacity(0.12, ft.Colors.CYAN_ACCENT)
    drop_zone_drag_bg = ft.Colors.with_opacity(0.20, ft.Colors.CYAN_ACCENT)

    drop_label = "Drop video files here or click to browse" if HAS_DROPZONE else "Click to add video files"

    drop_content = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.VIDEO_LIBRARY, size=48, color=ft.Colors.CYAN_ACCENT),
                ft.Text(
                    drop_label,
                    size=18,
                    weight=ft.FontWeight.W_500,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Text(
                    "Supports .mp4, .mkv, .mov, .avi, .wmv, .webm | Multiple files supported",
                    size=12,
                    color=ft.Colors.GREY_600,
                    text_align=ft.TextAlign.CENTER,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=8,
        ),
        border=ft.Border.all(2, ft.Colors.CYAN_700),
        border_radius=16,
        padding=30,
        alignment=ft.Alignment.CENTER,
        bgcolor=drop_zone_default_bg,
        on_click=open_file_picker,
        on_hover=lambda e: (
            setattr(
                e.control,
                "bgcolor",
                drop_zone_hover_bg if e.data == "true" else drop_zone_default_bg,
            ),
            page.update(),
        ),
    )

    # Wrap with flet-dropzone if available
    if HAS_DROPZONE:
        def on_files_dropped(e):
            """Handle files dropped via OS drag-and-drop."""
            if hasattr(e, 'files') and e.files:
                paths = e.files if isinstance(e.files, list) else [e.files]
                add_video_files(paths)

        def on_drag_entered(e):
            drop_content.bgcolor = drop_zone_drag_bg
            drop_content.border = ft.Border.all(3, ft.Colors.CYAN_ACCENT)
            try:
                page.update()
            except Exception:
                pass

        def on_drag_exited(e):
            drop_content.bgcolor = drop_zone_default_bg
            drop_content.border = ft.Border.all(2, ft.Colors.CYAN_700)
            try:
                page.update()
            except Exception:
                pass

        drop_zone = ftd.Dropzone(
            content=drop_content,
            on_dropped=on_files_dropped,
            on_entered=on_drag_entered,
            on_exited=on_drag_exited,
        )
    else:
        drop_zone = drop_content

    # --- Queue Section ---
    queue_header = ft.Row(
        [
            ft.Text("Queue", size=16, weight=ft.FontWeight.W_600),
            ft.Container(expand=True),
            status_text,
            clear_btn,
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    queue_container = ft.Container(
        content=ft.Column(
            [
                queue_header,
                ft.Divider(height=1, color=ft.Colors.GREY_800),
                queue_list,
            ],
            spacing=8,
            expand=True,
        ),
        border=ft.Border.all(1, ft.Colors.GREY_800),
        border_radius=12,
        padding=16,
        expand=True,
    )

    # --- Page Layout ---
    page.add(
        ft.Column(
            [
                ft.Container(height=10),
                title,
                subtitle,
                ft.Container(height=16),
                drop_zone,
                ft.Container(height=16),
                queue_container,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )
    )

    page.update()


if __name__ == "__main__":
    ft.run(main)
