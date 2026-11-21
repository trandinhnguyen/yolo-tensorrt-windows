"""Qt desktop UI."""

import subprocess
from functools import partial
from pathlib import Path
from typing import Protocol

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import config
from .task import Job, Status

WINDOW_TITLE = "YOLO TensorRT for Windows"
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
VIDEO_FILTER = (
    "Videos (" + " ".join(f"*{s}" for s in VIDEO_SUFFIXES) + ");;All files (*)"
)
COLUMNS = ("Video", "Pipeline", "Status", "Throughput (FPS)")
FAILED_COLOR = QColor(200, 40, 40)


class JobSource(Protocol):
    """What the window needs from :class:`~.jobs.WorkerPool`."""

    pipelines: dict
    jobs: dict[int, Job]

    def submit(self, path: str, pipeline: str) -> Job: ...

    def poll(self) -> list[Job]: ...


class VideoPlayer(QWidget):
    """Minimal player for the annotated output (video only, decoded by OpenCV)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.source: Path | None = None
        self._cap: cv2.VideoCapture | None = None
        self._image: QImage | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        self.screen = QLabel(
            "No output to show", alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.screen.setMinimumSize(320, 180)
        self.screen.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.screen.setStyleSheet("background: black; color: gray;")

        self.play_button = QPushButton()
        self.play_button.clicked.connect(self.toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.seek)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.screen, 1)
        layout.addLayout(controls)
        self.clear()

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def load(self, path: str | Path) -> None:
        self.clear()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.screen.setText(f"Cannot open {path}")
            return
        self.source = Path(path)
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self._timer.setInterval(max(1, round(1000 / fps)))
        self.slider.setRange(0, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
        self.play_button.setEnabled(True)
        self.slider.setEnabled(True)
        self._next_frame()  # show the first frame

    def clear(self) -> None:
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._image = None
        self.source = None
        self.screen.clear()
        self.screen.setText("No output to show")
        self.play_button.setEnabled(False)
        self.slider.setEnabled(False)
        self._update_icon()

    def toggle(self) -> None:
        if self._cap is None:
            return
        if self.playing:
            self._timer.stop()
        else:
            if self.slider.value() >= self.slider.maximum():
                self.seek(0)  # replay from the start
            self._timer.start()
        self._update_icon()

    def seek(self, frame_index: int) -> None:
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self._next_frame()

    def _next_frame(self) -> None:
        ok, frame = self._cap.read() if self._cap is not None else (False, None)
        if not ok:
            self._timer.stop()
            self._update_icon()
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self._image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._render()

        position = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)

    def _render(self) -> None:
        if self._image is not None:
            pixmap = QPixmap.fromImage(self._image).scaled(
                self.screen.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.screen.setPixmap(pixmap)

    def _update_icon(self) -> None:
        icon = (
            QStyle.StandardPixmap.SP_MediaPause
            if self.playing
            else QStyle.StandardPixmap.SP_MediaPlay
        )
        self.play_button.setIcon(self.style().standardIcon(icon))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()


class MainWindow(QMainWindow):
    def __init__(self, jobs: JobSource, poll_ms: int = config.UI_POLL_MS):
        super().__init__()
        self.jobs = jobs
        self._rows: dict[int, int] = {}  # job id -> table row
        self.setWindowTitle(WINDOW_TITLE)
        self.setAcceptDrops(True)
        self.resize(1200, 760)

        # Input video
        self.path_edit = QLineEdit(readOnly=True)
        self.path_edit.setPlaceholderText(
            "Browse for a video, or drop one on the window"
        )
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self.browse)
        input_row = QHBoxLayout()
        input_row.addWidget(self.path_edit, 1)
        input_row.addWidget(browse_button)
        input_box = QGroupBox("1. Video")
        input_box.setLayout(input_row)

        # Pipeline buttons
        self.pipeline_buttons: dict[str, QPushButton] = {}
        grid = QGridLayout()
        for i, name in enumerate(jobs.pipelines):
            button = QPushButton(name)
            button.setEnabled(False)
            button.setMinimumHeight(36)
            button.clicked.connect(partial(self.submit, name))
            grid.addWidget(button, i // 2, i % 2)
            self.pipeline_buttons[name] = button
        pipelines_box = QGroupBox("2. Run it on a pipeline")
        pipelines_box.setLayout(grid)

        # Jobs table
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.show_selected)

        # Selected job: details, output player
        self.details = QPlainTextEdit(readOnly=True)
        self.details.setPlaceholderText("Select a job to see its details.")
        self.open_folder_button = QPushButton("Show output in Explorer")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self.show_output_in_explorer)
        details_column = QVBoxLayout()
        details_column.addWidget(self.details, 1)
        details_column.addWidget(self.open_folder_button)
        details_widget = QWidget()
        details_widget.setLayout(details_column)
        self.player = VideoPlayer()

        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(details_widget)
        bottom.addWidget(self.player)
        bottom.setStretchFactor(1, 2)

        left = QVBoxLayout()
        left.addWidget(input_box)
        left.addWidget(pipelines_box)
        left.addStretch(1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(360)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.table)
        right.addWidget(bottom)
        right.setStretchFactor(1, 2)

        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage(f"{len(jobs.pipelines)} pipelines ready")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.poll)
        self._timer.start(poll_ms)

    # --- Input -----------------------------------------------------------------

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video", "", VIDEO_FILTER)
        if path:
            self.set_video(path)

    def set_video(self, path: str) -> None:
        self.path_edit.setText(str(Path(path)))
        for button in self.pipeline_buttons.values():
            button.setEnabled(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._dropped_video(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        path = self._dropped_video(event)
        if path is not None:
            self.set_video(path)
            event.acceptProposedAction()

    @staticmethod
    def _dropped_video(event) -> str | None:
        for url in event.mimeData().urls():
            if url.isLocalFile() and url.toLocalFile().lower().endswith(VIDEO_SUFFIXES):
                return url.toLocalFile()
        return None

    # --- Jobs ------------------------------------------------------------------

    def submit(self, pipeline: str) -> None:
        path = self.path_edit.text()
        try:
            job = self.jobs.submit(path, pipeline)
        except OSError as e:
            self.statusBar().showMessage(str(e))
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._rows[job.id] = row
        self._fill_row(job)
        self.table.selectRow(row)
        self.statusBar().showMessage(f"Queued {job.name} on {pipeline}")

    def poll(self) -> None:
        selected = self.selected_job()
        for job in self.jobs.poll():
            self._fill_row(job)
            if selected is not None and job.id == selected.id:
                self.show_selected()

    def _fill_row(self, job: Job) -> None:
        throughput = f"{job.throughput:.1f}" if job.throughput else ""
        for column, text in enumerate(
            (job.name, job.pipeline, job.status.value, throughput)
        ):
            item = QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, job.id)
            if column == 3:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            if job.status == Status.FAILED:
                item.setForeground(FAILED_COLOR)
            self.table.setItem(self._rows[job.id], column, item)

    def selected_job(self) -> Job | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        job_id = self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        return self.jobs.jobs.get(job_id)

    def show_selected(self) -> None:
        job = self.selected_job()
        if job is None:
            self.details.clear()
            self.player.clear()
            self.open_folder_button.setEnabled(False)
            return
        self.details.setPlainText(job.describe())
        done = job.status == Status.DONE
        self.open_folder_button.setEnabled(done)
        if not done:
            self.player.clear()
        elif self.player.source != job.output_path:
            self.player.load(job.output_path)

    def show_output_in_explorer(self) -> None:
        job = self.selected_job()
        if job is not None and job.output_path.exists():
            subprocess.Popen(["explorer", f"/select,{job.output_path.resolve()}"])

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self.player.clear()
        super().closeEvent(event)
