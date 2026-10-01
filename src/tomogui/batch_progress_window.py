from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QProgressBar, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor

class ProgressWindow(QDialog):
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Batch Progress")
        self.setGeometry(200, 200, 400, 200)

        # Progress section inside the new window
        progress_group = QGroupBox("Batch Progress")
        progress_layout = QVBoxLayout()

        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setValue(0)  # Initialize to 0
        progress_layout.addWidget(self.batch_progress_bar)

        self.batch_status_label = QLabel("Ready")
        progress_layout.addWidget(self.batch_status_label)

        self.batch_queue_label = QLabel("Queue: 0 jobs waiting")
        progress_layout.addWidget(self.batch_queue_label)

        self.batch_stop_btn = QPushButton("Stop Batch")
        self.batch_stop_btn.setEnabled(False)
        progress_layout.addWidget(self.batch_stop_btn)

        # Set the layout of the progress window
        progress_group.setLayout(progress_layout)
        dialog_layout = QVBoxLayout(self)
        dialog_layout.addWidget(progress_group)
        self.setLayout(dialog_layout)

        # Connect stop button
        self.batch_stop_btn.clicked.connect(self._on_stop_clicked)  # emit stop_requested

        # Timer to simulate progress updates (unused unless you call start_progress)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)

        # Keep track of progress
        self.current_value = 0
        self.is_running = False

    def start_progress(self):
        """Start the timer to simulate the progress bar's progress (optional)."""
        self.is_running = True
        self.timer.start(100)  # Update progress every 100ms
        self.batch_stop_btn.setEnabled(True)  # Enable stop button

    def stop_batch(self):
        """Stop the simulated progress timer (optional)."""
        self.is_running = False
        self.timer.stop()  # Stop progress updates
        self.batch_stop_btn.setEnabled(False)  # Disable stop button
        self.batch_status_label.setText("Batch stopped.")

    def update_progress(self):
        """Simulated progress update (optional)."""
        if self.is_running:
            self.current_value += 1
            if self.current_value > 100:
                self.current_value = 100
                self.timer.stop()
                self.batch_status_label.setText("Batch completed")

            self.batch_progress_bar.setValue(self.current_value)
            self.batch_status_label.setText(f"Progress: {self.current_value}%")

    def update_queue_label(self, queue_size):
        """Update the queue label to show remaining jobs (optional)."""
        self.batch_queue_label.setText(f"Queue: {queue_size} jobs waiting")

    # better stop button UX + emit signal =====
    def _on_stop_clicked(self):
        """UI handler for the Stop button."""
        # Disable immediately to prevent double-clicks; GUI will re-enable if needed
        self.batch_stop_btn.setEnabled(False)
        self.batch_status_label.setText("Stopping…")
        self.stop_requested.emit()

    # external control API used by gui.py =====
    def set_running(self, running: bool):
        """Enable/disable the stop button depending on batch state."""
        self.batch_stop_btn.setEnabled(bool(running))

    def set_progress(self, value: int):
        try:
            self.batch_progress_bar.setValue(int(value))
        except Exception:
            pass

    def set_status(self, text: str):
        self.batch_status_label.setText(str(text))

    def set_queue(self, queue_size: int):
        self.batch_queue_label.setText(f"Queue: {int(queue_size)} jobs waiting")


class JobQueueWindow(QDialog):
    """A live table of every batch job Tomogui knows about right now:

      - What's queued but not yet dispatched.
      - What's running (and on which machine/GPU).
      - Recon type and one-word status.

    The window pulls its data from the host TomoGUI's ``batch_job_queue``
    and ``batch_running_jobs`` structures on a short timer — no callbacks
    are wired into the dispatcher, so the queue window is completely
    optional and its absence never affects the batch pipeline.
    """

    COLS = ("#", "Dataset", "Machine", "GPU", "Type", "Status")
    STATUS_COLORS = {
        "Running":  QColor("#1a8cff"),
        "Queued":   QColor("#888888"),
    }

    def __init__(self, gui, parent=None):
        # ``gui`` is the TomoGUI instance we read state from.
        super().__init__(parent)
        self.gui = gui
        self.setWindowTitle("Batch Job Queue")
        # Non-modal so the user can keep working in the main window while
        # this stays open on a second monitor.
        self.setModal(False)
        self.resize(760, 460)

        outer = QVBoxLayout(self)

        # Summary row — running / queued / completed counters. Also useful
        # to spot when the queue is idle (all three at 0).
        summary_row = QHBoxLayout()
        self.summary_label = QLabel("—")
        self.summary_label.setStyleSheet("font-weight: bold;")
        summary_row.addWidget(self.summary_label)
        summary_row.addStretch(1)
        refresh_btn = QPushButton("Refresh now")
        refresh_btn.clicked.connect(self.refresh)
        summary_row.addWidget(refresh_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        summary_row.addWidget(close_btn)
        outer.addLayout(summary_row)

        # Table
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # #
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)           # Dataset
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Machine
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # GPU
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Type
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Status
        outer.addWidget(self.table, 1)

        # Auto-refresh while the window is visible. 1 s is fast enough to
        # feel live and slow enough that it costs nothing. The timer is
        # started on show() and stopped on hide()/close so a hidden window
        # doesn't touch the queue at all.
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)

    # -- Show/hide hooks: run timer only while visible -------------------
    def showEvent(self, event):  # noqa: N802 (Qt override)
        super().showEvent(event)
        self.refresh()
        self._timer.start()

    def hideEvent(self, event):  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    # -- The one place we translate GUI state into table rows ------------
    def refresh(self):
        try:
            running = getattr(self.gui, "batch_running_jobs", {}) or {}
            queued = getattr(self.gui, "batch_job_queue", []) or []
            default_machine = getattr(self.gui, "batch_current_machine",
                                      "Local") or "Local"
            batch_running = bool(getattr(self.gui, "batch_running", False))
            completed = int(getattr(self.gui, "batch_completed_jobs", 0) or 0)
            total = int(getattr(self.gui, "batch_total_jobs", 0) or 0)
        except Exception:
            # If the host GUI is being torn down, quietly clear the table.
            self.table.setRowCount(0)
            self.summary_label.setText("(no batch state available)")
            return

        rows = []
        # Running first — they're the interesting ones. Sort by GPU id so
        # the same GPU sits on the same visual line as the batch progresses.
        for gpu_id in sorted(running.keys()):
            entry = running.get(gpu_id)
            if not entry:
                continue
            # Entry is (process, file_info, recon_type). Be defensive about
            # the shape in case an experimental path stores a longer tuple.
            try:
                _proc, fi, rtype = entry[0], entry[1], entry[2]
            except Exception:
                continue
            name = self._file_label(fi)
            machine = default_machine
            rows.append({
                "dataset": name,
                "machine": machine,
                "gpu": str(gpu_id),
                "type": str(rtype or "?"),
                "status": "Running",
            })
        # Queued next, in the order they'll be dispatched (list order).
        for entry in queued:
            try:
                fi = entry[0]
                rtype = entry[1] if len(entry) > 1 else "?"
                machine = entry[2] if len(entry) > 2 else default_machine
            except Exception:
                continue
            rows.append({
                "dataset": self._file_label(fi),
                "machine": str(machine or default_machine),
                "gpu": "—",         # not yet assigned
                "type": str(rtype),
                "status": "Queued",
            })

        # Repopulate the table. Full rebuild each tick is fine — a batch
        # rarely has more than a few dozen live rows and setRowCount is
        # cheap on that scale.
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = (str(r + 1), row["dataset"], row["machine"],
                      row["gpu"], row["type"], row["status"])
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if c in (0, 3, 4, 5):
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
            colour = self.STATUS_COLORS.get(row["status"])
            if colour is not None:
                self.table.item(r, 5).setForeground(QBrush(colour))

        state = "IDLE" if not batch_running else "RUNNING"
        n_run = sum(1 for row in rows if row["status"] == "Running")
        n_queue = sum(1 for row in rows if row["status"] == "Queued")
        progress = (f"{completed}/{total}"
                    if total else f"{completed}")
        self.summary_label.setText(
            f"Batch state: {state}   |   Running: {n_run}   "
            f"Queued: {n_queue}   Completed: {progress}"
        )

    @staticmethod
    def _file_label(file_info):
        """Best-effort pretty name for a file_info dict."""
        if not isinstance(file_info, dict):
            return "?"
        for k in ("filename", "file", "path"):
            v = file_info.get(k)
            if v:
                # If it's a path, only show the leaf so the column is legible.
                s = str(v)
                if "/" in s:
                    s = s.rsplit("/", 1)[-1]
                return s
        return "?"
