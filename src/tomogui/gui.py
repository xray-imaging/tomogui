import os, glob, json

# HDF5 file locking on NFS is broken (returns EAGAIN even when nothing holds
# the file). Disable it globally in this process AND its subprocesses BEFORE
# any h5py import can pick up the default. Every subsequent QProcess inherits
# this via os.environ.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np

# Disable vsync for better remote performance
if 'vblank_mode' not in os.environ:
    os.environ['vblank_mode'] = '0'

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QLineEdit, QLabel, QProgressBar,
    QComboBox, QSlider, QGroupBox, QSizePolicy, QMessageBox,
    QTabWidget, QFormLayout, QCheckBox, QSpinBox, QDoubleSpinBox,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,QFrame,
    QDialog
)
from PyQt5.QtCore import Qt, QEvent, QProcess, QEventLoop, QSize, QProcessEnvironment, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from pathlib import Path

from PIL import Image
import h5py, json
from datetime import datetime

# VisPy for fast GPU-accelerated rendering
try:
    from vispy import scene, app
    from vispy.scene import visuals
    from vispy.color import get_colormaps
    # Use Qt5 backend which is most stable
    try:
        app.use_app('pyqt5')
    except:
        pass
    VISPY_AVAILABLE = True
except ImportError:
    print("Warning: VisPy not available. Install with: pip install vispy")
    print("Falling back to slower rendering...")
    VISPY_AVAILABLE = False

# Auto-detect SSH X11 forwarding: VisPy/OpenGL context creation fails without VirtualGL.
# VirtualGL sets VGL_DISPLAY; absent that, fall back to the pyqtgraph software renderer.
if VISPY_AVAILABLE:
    _is_ssh = bool(os.environ.get('SSH_CONNECTION') or
                    os.environ.get('SSH_CLIENT') or
                    os.environ.get('SSH_TTY'))
    _has_vgl = bool(os.environ.get('VGL_DISPLAY') or os.environ.get('VGL_ISACTIVE'))
    if _is_ssh and not _has_vgl:
        print("SSH X11 forwarding detected: switching to pyqtgraph renderer (VisPy/OpenGL unavailable)")
        VISPY_AVAILABLE = False

# PyQtGraph: pure-software renderer that works over SSH X11 forwarding
try:
    import pyqtgraph as pg
    import matplotlib as _mpl
    pg.setConfigOptions(useOpenGL=False, imageAxisOrder='row-major')
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

from .theme_manager import ThemeManager
from .chatbot import ChatBotDialog
from .hdf5_viewer import HDF5ImageDividerDialog
from .batch_progress_window import ProgressWindow


class SyncWatcher(QThread):
    """Background thread that monitors a folder for new, fully-written HDF5 files.

    Completeness is determined by opening the HDF5 file and checking that
    /exchange/data has the same number of frames as /exchange/theta.
    This is robust against pauses during acquisition that would fool a
    simple file-size stability check.
    """
    new_file_ready = pyqtSignal(str)   # emits absolute path of the complete file
    file_progress  = pyqtSignal(str, int, int)  # path, n_done, n_total

    def __init__(self, folder, known_files, check_interval=10):
        super().__init__()
        self.folder = folder
        self.known_files = set(known_files)
        self.check_interval = check_interval   # seconds between polls
        self._stop = False

    def stop(self):
        self._stop = True

    @staticmethod
    def _check_complete(filepath):
        """Return (n_projections_written, n_angles_expected) or (0, 0) on error."""
        try:
            with h5py.File(filepath, 'r') as f:
                data = f.get('/exchange/data')
                theta = f.get('/exchange/theta')
                if data is None or theta is None:
                    return 0, 0
                return int(data.shape[0]), int(theta.shape[0])
        except Exception:
            return 0, 0

    def _sleep_interruptible(self, seconds):
        """Sleep in 100 ms chunks so stop() is responsive."""
        for _ in range(int(seconds * 10)):
            if self._stop:
                return
            self.msleep(100)

    def run(self):
        pending = set()   # files seen but not yet complete
        while not self._stop:
            try:
                current = set(glob.glob(os.path.join(self.folder, "*.h5")))
                new_files = (current - self.known_files) | pending
                pending.clear()
                for f in new_files:
                    if self._stop:
                        return
                    n_done, n_total = self._check_complete(f)
                    if n_total > 0 and n_done >= n_total:
                        self.known_files.add(f)
                        self.new_file_ready.emit(f)
                    else:
                        pending.add(f)
                        if n_total > 0:
                            self.file_progress.emit(f, n_done, n_total)
            except Exception:
                pass
            self._sleep_interruptible(self.check_interval)


class MachineSettingsDialog(QDialog):
    """Dialog for configuring remote machine settings"""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle("Remote Machine Settings")
        self.setMinimumWidth(500)
        self.config = config or {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Info label
        info = QLabel(
            "Configure remote machines for batch reconstruction.\n"
            "Leave username empty to use current system username.\n"
          "Conda environment defaults to 'tomocupy' if not specified."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Form for each machine
        form = QFormLayout()
        self.machine_inputs = {}

        machines = ["tomo1", "tomo2", "tomo3", "tomo4", "tomo5"]
        for machine in machines:
            # Get existing config or use defaults
            machine_config = self.config.get(machine, {})
            username = machine_config.get("username", "")
            hostname = machine_config.get("hostname", machine)
            conda_env = machine_config.get("conda_env", "tomocupy")

            # Create row widget
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            user_input = QLineEdit(username)
            user_input.setPlaceholderText("username")
            user_input.setFixedWidth(100)

            host_input = QLineEdit(hostname)
            host_input.setPlaceholderText("hostname or IP")
            host_input.setFixedWidth(150)

            conda_input = QLineEdit(conda_env)
            conda_input.setPlaceholderText("tomocupy")
            conda_input.setFixedWidth(100)

            row_layout.addWidget(QLabel("User:"))
            row_layout.addWidget(user_input)
            row_layout.addWidget(QLabel("Host:"))
            row_layout.addWidget(host_input)
            row_layout.addWidget(QLabel("Env:"))
            row_layout.addWidget(conda_input)
            row_layout.addStretch()

            self.machine_inputs[machine] = {
                "username": user_input,
                "hostname": host_input,
                "conda_env": conda_input
            }

            form.addRow(f"{machine}:", row)

        layout.addLayout(form)

        # Buttons
        button_box = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_box.addStretch()
        button_box.addWidget(save_btn)
        button_box.addWidget(cancel_btn)
        layout.addLayout(button_box)

    def get_config(self):
        """Return the configuration dictionary"""
        config = {}
        for machine, inputs in self.machine_inputs.items():
            username = inputs["username"].text().strip()
            hostname = inputs["hostname"].text().strip()
            conda_env = inputs["conda_env"].text().strip()
            if hostname:  # Only include if hostname is provided
                config[machine] = {
                    "username": username or os.getenv("USER", ""),
                    "hostname": hostname,
                    "conda_env": conda_env or "tomocupy"
                }
        return config


class TomoGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TomoGUI")
        self.resize(2048,1080)

        # Initialize theme manager (will apply theme after UI is built)
        self.theme_manager = ThemeManager()
        self.theme_manager.register_callback(self._on_theme_changed)

        self._chatbot_dialog = None

        #initialize progress bar for batch process
        self.progress_window = ProgressWindow(self)
        #stop button in progress window stops the same batch queue
        self.progress_window.stop_requested.connect(self._batch_stop_queue)

        # Load machine configuration
        self.machine_config = self._load_machine_config()

        # State
        self.default_cmap = "gray"
        self.current_cmap = self.default_cmap
        self.vmin = None
        self.vmax = None
        self.preview_files = []
        # self.full_files holds either a list of TIFF paths (legacy tiff
        # output) or a list of integer slice indices into an H5 volume.
        # self.full_h5 is the open h5py handle when we're viewing an H5
        # reconstruction; None for TIFF mode.
        self.full_files = []
        self.full_h5 = None
        self.full_h5_path = None
        self.process = []
        self._current_img = None
        self._current_img_path = None
        self.cor_data = {}
        self.batch_running = False
        self.batch_file_list = []
        self.highlight_scan = None
        self.highlight_row = None
        self._current_source_file = None
        self._running_full_file = None  # track which file is under local full recon
        self.cor_path = None
        self._recon_params_data = None  # per-dataset params cache; None = needs reload
        self._batch_active = False      # while True, per-scan param load/save is suppressed
        self._sync_watcher = None       # SyncWatcher thread
        self._sync_queue = []
        self._sync_processing = False
        self._sync_current_file = None
        self.batch_file_main_list = []

        # Batch selection state for shift-click
        self.batch_last_clicked_row = None

        main_layout = QHBoxLayout()

        # ==== LEFT PANEL ====
        left_layout = QVBoxLayout()

        # Data folder
        folder_layout = QHBoxLayout()
        df_label = QLabel("Data Folder:")
        df_label.setStyleSheet("QLabel { font-size: 10.5pt; }")
        folder_layout.addWidget(df_label)
        self.data_path = QLineEdit()
        self.data_path.setFixedWidth(580)
        self.data_path.setStyleSheet("QLineEdit { font-size: 10.5pt; }")
        folder_layout.addWidget(self.data_path)
        browse_btn = QPushButton(" Browse Data Folder ")
        browse_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        browse_btn.setToolTip(f"Find folder, update from files status and rot_cen.json")
        browse_btn.clicked.connect(self.browse_data_folder)
        folder_layout.addWidget(browse_btn)
        refresh_btn2 = QPushButton("     Refresh     ")
        refresh_btn2.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        refresh_btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        refresh_btn2.clicked.connect(self.refresh_main_table)      
        refresh_btn2.setToolTip(f"Update from files status and rot_cen.json")
        folder_layout.addWidget(refresh_btn2)  
        left_layout.addLayout(folder_layout)

        # ==== TABS (Configs + Params) ====
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs)        
        
        first_tab = QWidget()
        self.tabs.addTab(first_tab, "Main")

        #==========main tab may===================
        main_tab = QVBoxLayout(first_tab)
        main_tab.setSpacing(6)
        #Row 1 - Try part single file operation
        single_ops = QHBoxLayout()
        single_ops.setSpacing(10)
        label_try = QLabel("Try method")
        label_try.setStyleSheet("QLabel { font-size: 10.5pt; }")
        label_try.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        single_ops.addWidget(label_try)
        self.recon_way_box = QComboBox()
        self.recon_way_box.setFixedWidth(100)
        self.recon_way_box.setStyleSheet("QComboBox { font-size: 10.5pt; }")
        self.recon_way_box.addItems(["recon","recon_steps"])
        self.recon_way_box.setCurrentIndex(0) # make recon as default
        single_ops.addWidget(self.recon_way_box)
        cor_method_label = QLabel("COR method")
        cor_method_label.setStyleSheet("QLabel { font-size: 10.5pt; }")
        single_ops.addWidget(cor_method_label)
        self.cor_method_box = QComboBox()
        self.cor_method_box.setFixedWidth(85)
        self.cor_method_box.setStyleSheet("QComboBox { font-size: 10.5pt; }")
        self.cor_method_box.addItems(["auto","manual"])
        self.cor_method_box.setCurrentIndex(1) # make manual as default
        single_ops.addWidget(self.cor_method_box)
        cor_label = QLabel("COR")
        cor_label.setStyleSheet("QLabel { font-size: 10.5pt; }")
        single_ops.addWidget(cor_label)
        self.cor_input = QLineEdit()
        self.cor_input.setFixedWidth(55)
        self.cor_input.setStyleSheet("QLineEdit { font-size: 10.5pt; }")
        single_ops.addWidget(self.cor_input)
        cuda_label = QLabel("cuda")
        cuda_label.setStyleSheet("QLabel { font-size: 10.5pt; }")
        single_ops.addWidget(cuda_label)
        self.cuda_box_try = QSpinBox()
        self.cuda_box_try.setMinimum(0)
        self.cuda_box_try.setMaximum(1)
        self.cuda_box_try.setValue(0)
        self.cuda_box_try.setFixedWidth(52)
        self.cuda_box_try.setStyleSheet("QSpinBox { font-size: 10.5pt; }")
        single_ops.addWidget(self.cuda_box_try)
        try_btn = QPushButton("  Try  ")
        try_btn.setStyleSheet("QPushButton { font-size: 11pt; font-weight:bold; }")
        try_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        try_btn.clicked.connect(self.try_reconstruction)
        single_ops.addWidget(try_btn)
        view_try_btn = QPushButton("  View Try  ")
        view_try_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        view_try_btn.clicked.connect(self.view_try_reconstruction) 
        single_ops.addWidget(view_try_btn)
        single_ops.setStretch(0,0)
        separator = QLabel(" | ")
        separator.setStyleSheet("QLabel { font-size: 11pt; }")
        single_ops.addWidget(separator)
        clear_log_btn = QPushButton(" clear Log ")
        clear_log_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        clear_log_btn.setEnabled(True) #enable
        clear_log_btn.clicked.connect(self.clear_log)
        single_ops.addWidget(clear_log_btn)
        main_tab.addLayout(single_ops)

        # Row 1b - Try AI (tomocor inference)
        ai_ops = QHBoxLayout()
        ai_ops.setSpacing(6)
        ai_model_label = QLabel("AI Model:")
        ai_model_label.setStyleSheet("QLabel { font-size: 10.5pt; }")
        ai_ops.addWidget(ai_model_label)
        _default_ai_model = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "AImodels", "datav2_518_full_finetune", "epoch_10.pth",
        )
        self.ai_model_path = QLineEdit(_default_ai_model)
        self.ai_model_path.setPlaceholderText("Path to model weights (.pth/.pt)")
        self.ai_model_path.setStyleSheet("QLineEdit { font-size: 10pt; }")
        ai_ops.addWidget(self.ai_model_path, 1)
        def _browse_ai_model():
            fn, _ = QFileDialog.getOpenFileName(self, "Select model weights", "", "Model files (*.pth *.pt);;All files (*)")
            if fn:
                self.ai_model_path.setText(fn)
        ai_browse_btn = QPushButton("Browse")
        ai_browse_btn.setStyleSheet("QPushButton { font-size: 10pt; }")
        ai_browse_btn.setFixedWidth(65)
        ai_browse_btn.clicked.connect(_browse_ai_model)
        ai_ops.addWidget(ai_browse_btn)
        try_ai_btn = QPushButton("  AI Reco  ")
        try_ai_btn.setStyleSheet("QPushButton { font-size: 11pt; font-weight:bold; color: #1a8cff; }")
        try_ai_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        try_ai_btn.setToolTip("Run Try reconstruction, find best COR via AI, then run Full reconstruction")
        try_ai_btn.clicked.connect(self.try_ai_reconstruction)
        ai_ops.addWidget(try_ai_btn)
        main_tab.addLayout(ai_ops)

        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)  # Vertical line
        h_line.setFrameShadow(QFrame.Sunken)
        main_tab.addWidget(h_line)
        
        # Row 2 - Full part single file operation
        single_full_ops = QHBoxLayout()
        single_full_ops.setSpacing(10)
        label_full = QLabel("Full method")
        label_full.setStyleSheet("QLabel { font-size: 10.5pt; }")
        label_full.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        single_full_ops.addWidget(label_full)
        self.recon_way_box_full = QComboBox()
        self.recon_way_box_full.setFixedWidth(99)
        self.recon_way_box_full.setStyleSheet("QComboBox { font-size: 10.5pt; }")
        self.recon_way_box_full.addItems(["recon","recon_steps"])
        self.recon_way_box_full.setCurrentIndex(0) # make recon as default
        single_full_ops.addWidget(self.recon_way_box_full)
        cor_method_full_label = QLabel("COR method")
        cor_method_full_label.setStyleSheet("QLabel { font-size: 10.5pt; }")
        single_full_ops.addWidget(cor_method_full_label)
        self.cor_full_method = QComboBox()
        self.cor_full_method.setFixedWidth(85)
        self.cor_full_method.setStyleSheet("QComboBox { font-size: 10.5pt; }")
        self.cor_full_method.addItems(["auto","manual"])
        self.cor_full_method.setCurrentIndex(1) # make manual as default
        single_full_ops.addWidget(self.cor_full_method)
        rec_cor_btn = QPushButton("  Add COR  ")
        rec_cor_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        rec_cor_btn.clicked.connect(self.record_cor_main_tb)
        single_full_ops.addWidget(rec_cor_btn)
        cuda_full_label = QLabel("cuda")
        cuda_full_label.setStyleSheet("QLabel { font-size: 10.5pt; }")
        single_full_ops.addWidget(cuda_full_label)
        self.cuda_full_box = QSpinBox()
        self.cuda_full_box.setMinimum(0)
        self.cuda_full_box.setMaximum(1)
        self.cuda_full_box.setValue(0)
        self.cuda_full_box.setFixedWidth(52)
        self.cuda_full_box.setStyleSheet("QSpinBox { font-size: 10.5pt; }")
        single_full_ops.addWidget(self.cuda_full_box) 
        self.full_btn = QPushButton("    Full    ")
        self.full_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.full_btn.setStyleSheet("QPushButton { font-size: 11pt; font-weight:bold; }")
        self.full_btn.clicked.connect(self.full_reconstruction)
        single_full_ops.addWidget(self.full_btn)
        self.view_btn = QPushButton("  View Full  ")
        self.view_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        self.view_btn.setEnabled(True)
        self.view_btn.clicked.connect(self.view_full_reconstruction)
        single_full_ops.addWidget(self.view_btn)
        separator_full = QLabel(" | ")
        separator_full.setStyleSheet("QLabel { font-size: 11pt; }")
        single_full_ops.addWidget(separator_full)
        save_log_btn = QPushButton(" save Log ")
        save_log_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        save_log_btn.setEnabled(True) #enable
        save_log_btn.clicked.connect(self.save_log)
        single_full_ops.addWidget(save_log_btn)
        main_tab.addLayout(single_full_ops)


        h_line2 = QFrame()
        h_line2.setFrameShape(QFrame.HLine)  # Vertical line
        h_line2.setFrameShadow(QFrame.Sunken)
        main_tab.addWidget(h_line2)

        # Row 3: some helpful functions
        others_ops = QHBoxLayout()
        others_ops.setSpacing(5)
        camrot_btn = QPushButton("CamRot")
        camrot_btn.setStyleSheet("QPushButton { font-size: 10.5pt; color: #26a69a; font-weight: bold; }")
        camrot_btn.setToolTip(
            "Estimate the camera rotation angle of the currently highlighted "
            "file.\n"
            "Runs Try+AI-infer at nsino=0.1 (top) and nsino=0.9 (bottom), "
            "then computes the angle from the COR difference and image "
            "height."
        )
        camrot_btn.clicked.connect(self._cam_rot_estimate)
        others_ops.addWidget(camrot_btn)
        save_param_btn = QPushButton("Save params")
        save_param_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        save_param_btn.setEnabled(True) #enable
        save_param_btn.clicked.connect(self.save_params_to_file)
        others_ops.addWidget(save_param_btn)
        load_param_btn = QPushButton("Load params")
        load_param_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        load_param_btn.setEnabled(True) #enable
        load_param_btn.clicked.connect(self.load_params_from_file)    
        others_ops.addWidget(load_param_btn)
        abort_btn =QPushButton("Abort")
        abort_btn.setStyleSheet("QPushButton { font-size: 10.5pt; color: red; }")
        abort_btn.clicked.connect(self.abort_process)
        others_ops.addWidget(abort_btn)
        help_tomo_btn = QPushButton("help")
        help_tomo_btn.setStyleSheet("QPushButton { font-size: 10.5pt; color: green; }")
        help_tomo_btn.clicked.connect(self.help_tomo)
        others_ops.addWidget(help_tomo_btn)
        ask_claude_btn = QPushButton("Ask recon")
        ask_claude_btn.setStyleSheet("QPushButton { font-size: 10.5pt; color: #FF8C00; font-weight: bold; }")
        ask_claude_btn.setToolTip(
            "Open a chat dialog to ask Claude about tomocupy parameters and tomogui workflow.\n"
            "Requires the 'anthropic' package (pip install anthropic) and an ANTHROPIC_API_KEY."
        )
        ask_claude_btn.clicked.connect(self._open_chatbot)
        others_ops.addWidget(ask_claude_btn)
        main_tab.addLayout(others_ops)

        # Row 4 - batch process table
        self.batch_file_main_table = QTableWidget()
        self.batch_file_main_table.cellClicked.connect(self.on_table_row_clicked)
        self.batch_file_main_table.setStyleSheet("""
                                            QTableWidget {
                                                font-size: 10.5pt; /* Set font size for the table cells */
                                            }
                                            QHeaderView::section {
                                                font-size: 10.5pt; /* Set font size for the header */
                                                font-weight: bold; /* Make header text bold */
                                                }
                                                """)
        self.batch_file_main_table.setColumnCount(7)
        self.batch_file_main_table.setHorizontalHeaderLabels(["Select","File Name", "COR",
                                                        "Status", "Size", "Pixel", "View Data"])
        self.batch_file_main_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.batch_file_main_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)  # Allow user to resize columns
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Select checkbox
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Filename - user can resize
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # COR
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Status
        header.setSectionResizeMode(4, QHeaderView.Stretch)  # Size
        header.setSectionResizeMode(5, QHeaderView.Stretch)  # Actions
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # View Data
        self.batch_file_main_table.setColumnWidth(0,50)
        self.batch_file_main_table.setColumnWidth(1, 350) # Set initial width for filename column to be wider (can be resized by user)    
        main_tab.addWidget(self.batch_file_main_table)
        #Row 5: batch process operations
        batch_ops = QHBoxLayout()
        batch_ops.setSpacing(5)
        batch_label = QLabel("Batch process ")
        batch_label.setStyleSheet("QLabel { font-size: 10.5pt; font-weight:bold; }")
        batch_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        batch_label.setFixedWidth(97)
        batch_ops.addWidget(batch_label)
        batch_mach_label = QLabel(" Machine ")
        batch_mach_label.setStyleSheet("QLabel { font-size: 10.5pt; }")
        batch_mach_label.setFixedWidth(59)
        batch_ops.addWidget(batch_mach_label)
        self.batch_machine_box = QComboBox()
        self.batch_machine_box.setStyleSheet("QComboBox { font-size: 10.5pt; }")
        self.batch_machine_box.setToolTip("Select machine to run batch reconstructions")
        self.batch_machine_box.setFixedWidth(62)
        self._populate_machine_list()  # Populate dynamically from config
        batch_ops.addWidget(self.batch_machine_box)
        batch_gpus = QLabel("GPUs")
        batch_gpus.setStyleSheet("QLabel { font-size: 10.5pt; }")
        batch_gpus.setFixedWidth(39)
        batch_ops.addWidget(batch_gpus)
        self.batch_gpus_per_machine = QSpinBox()
        self.batch_gpus_per_machine.setMinimum(1)
        self.batch_gpus_per_machine.setMaximum(8)
        self.batch_gpus_per_machine.setValue(1)
        self.batch_gpus_per_machine.setToolTip("Number of GPUs to use on the target machine (1 job per GPU)")
        self.batch_gpus_per_machine.setFixedWidth(38)
        self.batch_gpus_per_machine.setStyleSheet("QSpinBox { font-size: 10.5pt; }")
        batch_ops.addWidget(self.batch_gpus_per_machine)

        # Checkbox for opening remote jobs in terminal
        self.batch_use_terminal = QCheckBox("Terminal")
        self.batch_use_terminal.setToolTip("Open remote jobs in separate terminal windows")
        self.batch_use_terminal.setChecked(False)
        self.batch_use_terminal.setStyleSheet("QCheckBox { font-size: 10.5pt; }")
        batch_ops.addWidget(self.batch_use_terminal)

        #TODO: monitor folder and auto recon
        #monitor_btn = QPushButton("Monitor")
        #monitor_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        #monitor_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        #monitor_btn.setToolTip(f"Auto try or full recon")
        #monitor_btn.setVisible(False)  # Unavailable
        #monitor_btn.clicked.connect(self.monitor)#TODO: to implement
        #batch_ops.addWidget(monitor_btn)
        select_all_btn = QPushButton("Select all")
        select_all_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        select_all_btn.clicked.connect(self._batch_select_all)
        select_all_btn.setFixedWidth(120)
        batch_ops.addWidget(select_all_btn)
        deselect_all_btn = QPushButton("Unselect all")
        deselect_all_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        deselect_all_btn.clicked.connect(self._batch_deselect_all)
        deselect_all_btn.setFixedWidth(120)
        batch_ops.addWidget(deselect_all_btn)
        # view_selected_btn = QPushButton("View Selected Data")
        # view_selected_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        # view_selected_btn.setToolTip("Open HDF5 viewer for first selected file")
        # view_selected_btn.clicked.connect(self._batch_view_selected_data)
        # view_selected_btn.setFixedWidth(135)
        select_donetry_btn = QPushButton("Select done try")
        select_donetry_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        select_donetry_btn.setToolTip("Select files that has done try reconstruction")
        select_donetry_btn.clicked.connect(self._select_done_try)
        select_donetry_btn.setFixedWidth(135)
        batch_ops.addWidget(select_donetry_btn)
        separator_batch = QLabel("  |  ")
        separator_batch.setStyleSheet("QLabel { font-size: 11pt; }")
        separator_batch.setFixedWidth(23)
        batch_ops.addWidget(separator_batch)
        batch_recon_btn =QPushButton("Batch Try")
        batch_recon_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        batch_recon_btn.setToolTip("Run batch try reconstruction on selected files,the params are from GUI and COR guess from value put on single operation above")
        batch_recon_btn.clicked.connect(self._batch_run_try_selected)
        #batch_recon_btn.setFixedWidth(100)
        batch_ops.addWidget(batch_recon_btn)
        batch_full_btn =QPushButton("Batch Full")
        batch_full_btn.setStyleSheet("QPushButton { font-size: 10.5pt; }")
        batch_full_btn.setToolTip("Run batch full reconstruction on selected files,the params are from GUI and COR from Table above")
        batch_full_btn.clicked.connect(self._batch_run_full_selected) #TODO: needs to modify to work with table
        #batch_full_btn.setFixedWidth(100)
        batch_ops.addWidget(batch_full_btn)
        batch_ai_btn = QPushButton("Batch AI Reco")
        batch_ai_btn.setStyleSheet("QPushButton { font-size: 10.5pt; font-weight:bold; color: #1a8cff; }")
        batch_ai_btn.setToolTip(
            "Run selected phases of AI Reco on every checked file. "
            "Tick the phase checkboxes to the right to pick which run."
        )
        batch_ai_btn.clicked.connect(self._batch_run_ai_selected)
        batch_ops.addWidget(batch_ai_btn)

        def _mk_phase(label, default, tip):
            cb = QCheckBox(label)
            cb.setStyleSheet("QCheckBox { font-size: 10.5pt; color: #1a8cff; }")
            cb.setChecked(default)
            cb.setToolTip(tip)
            batch_ops.addWidget(cb)
            return cb

        self.batch_ai_phase_try = _mk_phase(
            "Try", True,
            "Phase A — run Try reconstruction for every checked file."
        )
        self.batch_ai_phase_infer = _mk_phase(
            "Infer", True,
          "Phase B — run AI inference on each file's try_center TIFFs and "
            "fill the COR cell."
        )
        self.batch_ai_phase_full = _mk_phase(
            "Full", True,
            "Phase C — run Full reconstruction for every file that now has a COR."
        )
        self.batch_ai_phase_tomolog = _mk_phase(
            "TomoLog", False,
            "Phase D — upload each successfully reconstructed file to "
            "TomoLog using the current TomoLog panel settings."
        )
        # Back-compat alias (still read by older code paths / docs).
        self.batch_ai_upload_tomolog = self.batch_ai_phase_tomolog
        fix_cor_btn = QPushButton("Fix COR Outliers")
        fix_cor_btn.setStyleSheet("QPushButton { font-size: 10.5pt; color: #8e44ad; }")
        fix_cor_btn.setToolTip("Detect outlier COR values among selected files and replace "
                              "each outlier with the average of its two neighbours")
        fix_cor_btn.clicked.connect(lambda: self._fix_cor_outliers())
        batch_ops.addWidget(fix_cor_btn)
        clear_cor_btn = QPushButton("Clear CORs")
        clear_cor_btn.setStyleSheet("QPushButton { font-size: 10.5pt; color: #b26a00; }")
        clear_cor_btn.setToolTip("Clear the COR value for every checked file "
                                "(table and rot_cen.json). Useful before "
                                "re-running AI Reco from scratch.")
        clear_cor_btn.clicked.connect(self._clear_selected_cors)
        batch_ops.addWidget(clear_cor_btn)
        batch_ops.addWidget(QLabel("max Δ:"))
        self.cor_outlier_max = QDoubleSpinBox()
        self.cor_outlier_max.setRange(1.0, 1000.0)
        self.cor_outlier_max.setDecimals(1)
        self.cor_outlier_max.setSingleStep(5.0)
        self.cor_outlier_max.setValue(50.0)
        self.cor_outlier_max.setFixedWidth(70)
        self.cor_outlier_max.setSuffix(" px")
        self.cor_outlier_max.setToolTip("Maximum allowed |COR − series median| (pixels). "
                                        "Any deviation greater than this is flagged as an outlier. "
                                        "Tight-cluster series may use a smaller effective threshold "
                                        "(max(abs, 5·MAD), capped at this value).")
        batch_ops.addWidget(self.cor_outlier_max)
        # Per-series minimum size (% of series median). Files smaller than this
        # are auto-unchecked and marked "Skipped (small)" — typically aborted
        # scans with tiny file size compared to their series peers.
        batch_ops.addWidget(QLabel("min size:"))
        self.size_min_pct = QDoubleSpinBox()
        self.size_min_pct.setRange(0.0, 100.0)
        self.size_min_pct.setDecimals(0)
        self.size_min_pct.setSingleStep(5.0)
        self.size_min_pct.setValue(50.0)
        self.size_min_pct.setFixedWidth(60)
        self.size_min_pct.setSuffix(" %")
        self.size_min_pct.setToolTip(
            "Minimum file size as a % of the series median. "
            "Files smaller than this are auto-unchecked and marked "
            "'Skipped (small)' — typically aborted scans. "
            "Set to 0 to disable."
        )
        batch_ops.addWidget(self.size_min_pct)
        delete_sel_btn = QPushButton("Delete Selected")
        delete_sel_btn.setStyleSheet("QPushButton { font-size: 10.5pt; color: #c62828; }")
        delete_sel_btn.setToolTip("Delete the selected HDF5 files from disk (with confirmation)")
        delete_sel_btn.clicked.connect(self._delete_selected_files)
        batch_ops.addWidget(delete_sel_btn)
        main_tab.addLayout(batch_ops)
        #Row 6: log
        log_box = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("QTextEdit { font-size: 11pt; }")
        self.log_output.append("Start tomoGUI")     
        self.log_output.setFixedHeight(200)  # Set a fixed height
        log_box.addWidget(self.log_output)
        main_tab.addLayout(log_box)

        # Sync Acquisition button (bottom of left panel)
        sync_row = QHBoxLayout()
        self.sync_btn = QPushButton(" Sync Acquisition")
        self.sync_btn.setStyleSheet(
            "QPushButton { font-size: 11pt; font-weight: bold; color: white; "
            "background-color: #2e7d32; padding: 6px 18px; border-radius: 4px; }"
            "QPushButton:checked { background-color: #b71c1c; }"
        )
        self.sync_btn.setCheckable(True)
        self.sync_btn.clicked.connect(self._toggle_sync)
        sync_row.addWidget(self.sync_btn)
        main_tab.addLayout(sync_row)

        # Tab 2: Params (all CLI flags + extra args)
        self._build_params_tab()
        self._build_rings_tab()
        self._build_bhard_tab()
        self._build_phase_tab()
        self._build_Geometry_tab()
        self._build_Data_tab()        
        self._build_Performance_tab()
        self._build_ai_tab()
        self._build_advanced_config_tab()
        # Progress bar
        # self.progress = QProgressBar()
        # self.progress.setRange(0, 0)
        # self.progress.setVisible(False)
        # left_layout.addWidget(self.progress)

        # Log + COR JSON
        #log_json_layout = QHBoxLayout()
        #log_box_layout = QVBoxLayout()
        #log_box_layout.addWidget(QLabel("Log Output:"))
        #self.log_output = QTextEdit()
        #self.log_output.setReadOnly(True)
        #self.log_output.setStyleSheet("QTextEdit { font-size: 12.5pt; }")
        #self.log_output.append("Start tomoGUI")
        #log_box_layout.addWidget(self.log_output)
        #og_json_layout.addLayout(log_box_layout)
        #left_layout.addLayout(log_json_layout)
        
        main_layout.addLayout(left_layout, 4)
        
        # ==== RIGHT PANEL ====
        right_layout = QVBoxLayout()
        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(8)

        # Neither VisPy nor PyQtGraph available — show a hard error
        if not VISPY_AVAILABLE and not PG_AVAILABLE:
            error_label = QLabel(
                "ERROR: No renderer available!\n\n"
                "Install VisPy for GPU rendering:\n  pip install vispy PyOpenGL\n\n"
                "Or install PyQtGraph for SSH/software rendering:\n  pip install pyqtgraph matplotlib"
            )
            error_label.setStyleSheet("color: red; font-size: 14pt; font-weight: bold; padding: 20px;")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setWordWrap(True)
            toolbar_row.addWidget(error_label)
            right_layout.addLayout(toolbar_row)
            main_layout.addLayout(right_layout, 8)
            self.setLayout(main_layout)
            return

        # Shared canvas state
        self._last_camera_rect = None
        self._last_image_shape = None
        self.roi_extent = None
        self._drawing_roi = False
        self._roi_visual = None
        self._roi_start = None

        if VISPY_AVAILABLE:
            # --- VisPy canvas (GPU-accelerated, requires OpenGL) ---
            try:
                self.canvas = scene.SceneCanvas(keys='interactive', show=False)
            except Exception as e:
                print(f"VisPy canvas creation failed: {e}. Falling back to pyqtgraph.")
                globals()['VISPY_AVAILABLE'] = False

        if VISPY_AVAILABLE:
            self.view = self.canvas.central_widget.add_view()
            self.view.camera = scene.PanZoomCamera(aspect=1)
            self.view.camera.flip = (False, True, False)  # row-0 at top
            self.image_visual = visuals.Image(cmap='grays', parent=self.view.scene)
            self.canvas_widget = self.canvas.native
            self.canvas.events.mouse_move.connect(self._on_vispy_mouse_move)
            self.canvas.events.mouse_press.connect(self._on_vispy_mouse_click)
            self.canvas.events.mouse_release.connect(self._on_vispy_mouse_release)
        else:
            # --- PyQtGraph canvas (software renderer, works over SSH X11) ---
            self._pg_image_item = pg.ImageItem()
            self._pg_view_box = pg.ViewBox()
            self._pg_view_box.setAspectLocked(True)
            self._pg_view_box.invertY(True)
            self._pg_layout = pg.GraphicsLayoutWidget()
            self._pg_layout.addItem(self._pg_view_box)
            self._pg_view_box.addItem(self._pg_image_item)
            self._pg_roi_item = None
            self.canvas_widget = self._pg_layout
            # Mouse coordinate tracking via SignalProxy (rate-limited)
            self._pg_proxy = pg.SignalProxy(
                self._pg_image_item.scene().sigMouseMoved,
                rateLimit=30, slot=self._pg_mouse_moved
            )

        # Coordinate label
        coord_label = QLabel("(x,y):val ")
        coord_label.setFixedWidth(80)
        coord_label.setStyleSheet("font-size: 11pt;")
        toolbar_row.addWidget(coord_label)
        self.coord_label = QLabel("")
        self.coord_label.setFixedWidth(150)
        self.coord_label.setStyleSheet("font-size: 11pt;")
        toolbar_row.addWidget(self.coord_label)

        # Colormap dropdown
        cmap_label = QLabel(" Cmap ")
        cmap_label.setStyleSheet("font-size: 11pt;")
        cmap_label.setFixedWidth(45)
        toolbar_row.addWidget(cmap_label)
        self.cmap_box = QComboBox()
        self.cmap_box.setFixedWidth(72)
        self.cmap_box.setStyleSheet("font-size: 11pt;")
        self.cmap_box.addItems(["gray", "viridis", "plasma", "inferno", "magma", "cividis"])
        self.cmap_box.setCurrentText(self.default_cmap)
        self.cmap_box.currentIndexChanged.connect(self.update_cmap)
        toolbar_row.addWidget(self.cmap_box)

        # Image control buttons
        draw_box_btn = QPushButton("Draw")
        draw_box_btn.setStyleSheet("font-size: 11pt;")
        draw_box_btn.clicked.connect(self.draw_box)
        draw_box_btn.setFixedWidth(57)
        auto_scale_btn = QPushButton("Auto")
        auto_scale_btn.setStyleSheet("font-size: 11pt;")
        auto_scale_btn.setFixedWidth(57)
        auto_scale_btn.clicked.connect(self.auto_img_contrast)
        reset_scale_btn = QPushButton("Reset")
        reset_scale_btn.setStyleSheet("font-size: 11pt;")
        reset_scale_btn.setFixedWidth(65)
        reset_scale_btn.clicked.connect(self.reset_img_contrast)
        toolbar_row.addWidget(draw_box_btn)
        toolbar_row.addWidget(auto_scale_btn)
        toolbar_row.addWidget(reset_scale_btn)

        # Min/Max inputs
        min_label = QLabel(" Min")
        min_label.setFixedWidth(35)
        min_label.setStyleSheet("font-size: 11pt;")
        toolbar_row.addWidget(min_label)
        self.min_input = QLineEdit()
        self.min_input.setFixedWidth(65)
        self.min_input.setStyleSheet("font-size: 11pt;")
        self.min_input.editingFinished.connect(self.update_vmin_vmax)
        toolbar_row.addWidget(self.min_input)

        max_label = QLabel(" Max")
        max_label.setFixedWidth(35)
        max_label.setStyleSheet("font-size: 11pt;")
        toolbar_row.addWidget(max_label)
        self.max_input = QLineEdit()
        self.max_input.setStyleSheet("font-size: 11pt;")
        self.max_input.setFixedWidth(65)
        self.max_input.editingFinished.connect(self.update_vmin_vmax)
        toolbar_row.addWidget(self.max_input)

        # Theme toggle button
        self.theme_toggle_btn = QPushButton("" if self.theme_manager.get_current_theme() == 'bright' else "")
        self.theme_toggle_btn.setFixedWidth(35)
        self.theme_toggle_btn.setToolTip("Toggle bright/dark theme")
        self.theme_toggle_btn.clicked.connect(self._toggle_theme)
        toolbar_row.addWidget(self.theme_toggle_btn)

        # Settings button
        settings_btn = QPushButton("⚙")
        settings_btn.setFixedWidth(35)
        settings_btn.setToolTip("Configure remote machine settings")
        settings_btn.clicked.connect(self._open_machine_settings)
        toolbar_row.addWidget(settings_btn)

        toolbar_row.addStretch(1)
        right_layout.addLayout(toolbar_row)
        self.canvas_widget.installEventFilter(self)

        canvas_slider_frame = QVBoxLayout()
        if not VISPY_AVAILABLE:
            # Histogram LUT widget for manual level adjustment (pyqtgraph only)
            self._pg_hist = pg.HistogramLUTWidget(orientation='vertical')
            self._pg_hist.setImageItem(self._pg_image_item)
            self._pg_hist.setMinimumWidth(100)
            self._pg_hist.setMaximumWidth(130)
            self._pg_hist.item.sigLevelsChanged.connect(self._pg_hist_levels_changed)
            canvas_row = QHBoxLayout()
            canvas_row.addWidget(self.canvas_widget, 1)
            canvas_row.addWidget(self._pg_hist)
            canvas_slider_frame.addLayout(canvas_row)
        else:
            canvas_slider_frame.addWidget(self.canvas_widget)
        slider_layout = QHBoxLayout()
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 20px; background: #ccc; border-radius: 5px; }
            QSlider::handle:horizontal { background: #4CAF50; border: 1px solid #5c5c5c; width: 20px; height: 20px;
                                            margin: -5px 0; border-radius: 10px; }
        """)
        slider_layout.addWidget(QLabel("Image Index:"))
        slider_layout.addWidget(self.slice_slider)
        self.filename_label = QLabel("")
        self.filename_label.setStyleSheet("font-size: 10pt; color: #aaa; padding-left: 6px;")
        slider_layout.addWidget(self.filename_label)
        canvas_slider_frame.addLayout(slider_layout)
        right_layout.addLayout(canvas_slider_frame, 8)

        # Tomolog section
        tomolog_group = QGroupBox("Tomolog")
        tomolog_layout = QVBoxLayout()

        # Row 1: Beamline + Scan + Cloud
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Beamline"))
        self.beamline_box = QComboBox()
        self.beamline_box.addItems(["2-bm", "7-bm", "32-id"])
        row1.addWidget(self.beamline_box)

        row1.addWidget(QLabel("Scan"))
        self.scan_input = QLineEdit()
        self.scan_input.setPlaceholderText("e.g. 1 or 2-5 or 6,8-15,18")
        row1.addWidget(self.scan_input)

        row1.addWidget(QLabel("Cloud"))
        self.cloud_box = QComboBox()
        self.cloud_box.addItems(["imgur", "globus", "aps"])
        self.cloud_box.setCurrentText("imgur")
        row1.addWidget(self.cloud_box)
        tomolog_layout.addLayout(row1)

        # Row 2: URL
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("URL"))
        self.url_input = QLineEdit()
        row2.addWidget(self.url_input)
        tomolog_layout.addLayout(row2)

        # Row 3: X, Y, Z
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("X"))
        self.x_input = QLineEdit("-1")
        row3.addWidget(self.x_input)
        row3.addWidget(QLabel("Y"))
        self.y_input = QLineEdit("-1")
        row3.addWidget(self.y_input)
        row3.addWidget(QLabel("Z"))
        self.z_input = QLineEdit("-1")
        row3.addWidget(self.z_input)
        tomolog_layout.addLayout(row3)

        # Row 4: Note and Extra params
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Note"))
        self.note_input = QLineEdit()
        row4.addWidget(self.note_input)

        row4.addWidget(QLabel("Extra Params"))
        self.extra_params_input = QLineEdit()
        self.extra_params_input.setPlaceholderText("--public True")
        row4.addWidget(self.extra_params_input)
        tomolog_layout.addLayout(row4)

        # Row 5: Apply button
        row5 = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.run_tomolog)
        help_tomolog_btn = QPushButton("help-log")
        help_tomolog_btn.clicked.connect(self.help_tomolog)
        row5.addWidget(apply_btn)
        row5.addWidget(help_tomolog_btn)
        row5.setStretch(0, 3)
        row5.setStretch(1, 1)
        tomolog_layout.addLayout(row5)

        tomolog_group.setLayout(tomolog_layout)
        right_layout.addWidget(tomolog_group, 2)

        main_layout.addLayout(right_layout, 5)
        self.setLayout(main_layout)

        # Apply initial theme after UI is fully built
        self.theme_manager.apply_theme(self.theme_manager.get_current_theme())

    # ===== PARAMS TAB =====
    def _build_params_tab(self):
        params_tab = QWidget()
        outer = QVBoxLayout(params_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        scroll.setWidget(form_host)
        outer.addWidget(scroll)

        self.param_widgets = {}
        #add check box
        def _add_row(flag, kind, w, default=None, label_text=None, include=True):
            #include: show the checkbox, not--> always include in params
            label_text = label_text or flag

            # label cell = [ include_cb | "flag" ]
            label_widget = QWidget()
            h = QHBoxLayout(label_widget)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lbl = QLabel(label_text)

            include_cb = None

            if include:
                include_cb = QCheckBox()
                include_cb.setChecked(False) #default not enable
                h.addWidget(include_cb)
                lbl.setEnabled(False) #defalut not check
                w.setEnabled(False)

                def on_toggle(checked):
                    lbl.setEnabled(checked)
                    w.setEnabled(checked)
                    if not checked:
                        # reset to a sensible "off" state
                        if kind in ("spin", "dspin") and default is not None:
                            w.blockSignals(True)
                            w.setValue(default)
                            w.blockSignals(False)
                        elif kind == "combo":
                            if default is not None:
                                w.setCurrentText(str(default))
                            else:
                                w.setCurrentIndex(0)
                        elif kind == "line":
                            w.clear()
                        elif kind == "check":
                            w.setChecked(False)

                include_cb.toggled.connect(on_toggle)
            else:
                lbl.setEnabled(True)
                w.setEnabled(True)
            h.addWidget(lbl)
            h.addStretch(1)
            form.addRow(label_widget, w)

            self.param_widgets[flag] = (kind, w, include_cb, default)

        def add_line(flag, placeholder="", tip="", width=240, include=True):
            w = QLineEdit()
            if placeholder:
                w.setPlaceholderText(placeholder)
            if tip:
                w.setToolTip(tip)
            w.setFixedWidth(width)
            if flag == "--nsino":
                w.setText("0.5")
            _add_row(flag, "line", w, default="",include=include)

        def add_combo(flag, items, default=None, tip="", include=True):
            w = QComboBox()
            w.addItems(items)
            if default in items:
                w.setCurrentText(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "combo", w, default=default, include=include)

        def add_check(flag, tip="", include=True):
            w = QCheckBox()
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "check", w, default=False, include=include)

        def add_spin(flag, minv, maxv, step=1, default=None, tip="", include=True):
            w = QSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "spin", w, default=default, include=include)

        def add_dspin(flag, minv, maxv, step=0.1, default=None, tip="", include=True):
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "dspin", w, default=default, include=include)

        add_combo("--binning", ["0","1","2","3"], default="0", include=False) #always include


        add_combo("--file-type", ["standard","double_fov"], default="standard", include=False) #always include


        add_dspin("--bright-ratio", 0.0, 1e9, step=0.1, default=1.0, include=False) #always include
        add_dspin("--center-search-step", 0.0, 1e6, step=0.05, default=0.5, include=False) #always include
        add_dspin("--center-search-width", 0.0, 1e6, step=0.5, default=50.0, include=False) #always include
        add_spin("--dezinger", 0, 10000, step=1, default=5, include=False) #always include
        add_spin("--dezinger-threshold", 0, 1000000, step=100, default=5000) 

        add_combo("--fbp-filter", ["none","ramp","shepp","hann","hamming","parzen","cosine","cosine2"], default="parzen", include=False) #always include
        add_spin("--find-center-end-row", -1, 10_000_000, step=1, default=-1, include=False) #always include
        add_spin("--find-center-start-row", 0, 10_000_000, step=1, default=0, include=False) #always include
        add_combo("--flat-linear", ["False","True"], default="False", include=False) #always include

        add_combo("--minus-log", ["True","False"], default="True", include=False) #always include

        add_line("--nsino", "", include=False) #always include
        #add_line("--nsino", "0.5 or [0,0.9]", include=False) #always include
#       add_dspin("--rotation-axis", -1e9, 1e9, step=0.01, default=-1.0)
#       add_combo("--rotation-axis-auto", ["manual","auto"], default="manual")
        add_combo("--rotation-axis-method", ["sift","vo"], default="sift", include=False) #always include
        add_line("--rotation-axis-pairs", "[0,1499] or [0,1499,749,2249]")
        add_dspin("--rotation-axis-sift-threshold", 0.0, 1.0, step=0.01, default=0.5)


        # Misc / algorithm
        add_combo("--pre-processing", ["True","False"], default="True", include=False) #always include
        add_combo("--reconstruction-algorithm", ["fourierrec","linerec"], default="fourierrec", include=False) #always include

        self.tabs.addTab(params_tab, "Reconstruction")

    def _strip_flag(self, cmd, flag):
        """Remove every ``flag`` (and its value if the flag takes one) from
        ``cmd``. Returns a new list. Used to drop stale flag values that
        would otherwise shadow the AI-COR overrides we append later — argparse
        keeps the *last* occurrence, so any conflicting later value silently
        wins."""
        out = []
        skip = False
        for a in cmd:
            if skip:
                skip = False
                continue
            if a == flag:
                skip = True
                continue
            out.append(a)
        return out

    def _apply_ai_cor(self, cmd, ai_search_method=None):
        """Return ``cmd`` with the AI-COR flags applied *last*. Any stale
        ``--rotation-axis-*`` / ``--ai-search-method`` / infer flags the
        caller may have already appended are stripped first so argparse
        sees only what we mean.

        The search method (``fine`` or ``full``) defaults to whatever the
        user picked on the AI COR tab. Extra ``--infer-*`` / ``--bin-infer-*``
        parameters the user enabled on that tab are appended too.
        """
        if ai_search_method is None:
            ai_search_method = self._current_ai_search_method()
        ai_flags = self._ai_cor_args(ai_search_method)
        if not ai_flags:
            return cmd

        # Every flag the AI tab (or _ai_cor_args) can emit must be stripped
        # from earlier positions — otherwise an opt-in on the tab could be
        # shadowed by whatever a --config file or another gatherer produced
        # first (argparse keeps the LAST occurrence).
        strip_flags = {"--rotation-axis-method", "--ai-search-method",
                      "--infer-model-path", "--bin-infer-model-path"}
        tab_flags = self._gather_ai_args()
        strip_flags.update(f for f in tab_flags if f.startswith("--"))
        for flag in strip_flags:
            cmd = self._strip_flag(cmd, flag)
        return cmd + ai_flags + tab_flags

    def _ai_cor_args(self, ai_search_method="fine"):
        """Return tomocupy CLI flags that turn on its built-in AI COR finder
        for a `try` reconstruction. Returns [] when the AI model path is not
        set — callers then fall back to whatever COR method they were using.

        Tomocupy runs the try recon and the AI center search in a single
        subprocess and writes ``center_of_rotation.txt`` inside the try
        output directory (``{data}_rec/try_center/{proj}/``).

        Both ``--infer-model-path`` and ``--bin-infer-model-path`` are set to
        the same file: ``full`` mode runs the two-stage bin refinement
        (reads ``bin_infer_model_path``) and THEN a final ``run_rec`` that
        internally calls ``_find_center_ai`` (reads ``infer_model_path``).
        Passing only one flag makes ``full`` crash on the second call.
        """
        model_path = self.ai_model_path.text().strip()
        if not model_path or not os.path.exists(model_path):
            return []
        return [
            "--rotation-axis-method", "ai",
            "--ai-search-method", ai_search_method,
            "--infer-model-path", model_path,
            "--bin-infer-model-path", model_path,
        ]

    def _read_ai_cor_from_try_dir(self, proj_file):
        """Read the last value written to center_of_rotation.txt inside the
        try output directory for ``proj_file``. Returns the COR as a float
        on success, or ``None`` if the file is missing/empty/unparseable."""
        data_folder = self.data_path.text().strip()
        if not data_folder:
            return None
        proj_name = os.path.splitext(os.path.basename(proj_file))[0]
        cor_txt = os.path.join(f"{data_folder}_rec", "try_center",
                              proj_name, "center_of_rotation.txt")
        if not os.path.exists(cor_txt):
            return None
        try:
            with open(cor_txt) as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            if not lines:
                return None
            return float(lines[-1].split()[-1])
        except (OSError, ValueError):
            return None

    def _gather_params_args(self):
        args = []
        for flag, (kind, w, include_cb, _default) in self.param_widgets.items():
            if include_cb is not None and not include_cb.isChecked():
                continue
            if kind == "line":
                val = w.text().strip()
                if val != "":
                    args += [flag, val]
            elif kind == "combo":
                args += [flag, w.currentText().strip()]
            elif kind == "check":
                if w.isChecked():
                    args += [flag]
            elif kind == "spin":
                args += [flag, str(w.value())]
            elif kind == "dspin":
                args += [flag, str(w.value())]
        return args


# ===== Beam Hardening TAB =====
    def _build_bhard_tab(self):
        bhard_tab = QWidget()
        outer = QVBoxLayout(bhard_tab)

        #check all button
        ca_layout = QHBoxLayout()
        ca_button = QPushButton("Check all")
        ca_layout.addWidget(ca_button)
        outer.addLayout(ca_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        scroll.setWidget(form_host)
        outer.addWidget(scroll)

        self.bhard_widgets = {}
        #add check box before each line
        def _add_row(flag, kind, w, default=None, label_text=None, include=True):
            #include: show the checkbox, not--> always include in params
            label_text = label_text or flag

            # label cell = [ include_cb | "flag" ]
            label_widget = QWidget()
            h = QHBoxLayout(label_widget)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lbl = QLabel(label_text)

            include_cb = None

            if include:
                include_cb = QCheckBox()
                include_cb.setChecked(False) #default not enable
                h.addWidget(include_cb)
                lbl.setEnabled(False) #defalut not check
                w.setEnabled(False)

                def on_toggle(checked):
                    lbl.setEnabled(checked)
                    w.setEnabled(checked)
                    if not checked:
                        # reset to a sensible "off" state
                        if kind in ("spin", "dspin") and default is not None:
                            w.blockSignals(True)
                            w.setValue(default)
                            w.blockSignals(False)
                        elif kind == "combo":
                            if default is not None:
                                w.setCurrentText(str(default))
                            else:
                                w.setCurrentIndex(0)
                        elif kind == "line":
                            w.clear()
                        elif kind == "check":
                            w.setChecked(False)

                include_cb.toggled.connect(on_toggle)
            else:
                lbl.setEnabled(True)
                w.setEnabled(True)
            h.addWidget(lbl)
            h.addStretch(1)
            form.addRow(label_widget, w)

            self.bhard_widgets[flag] = (kind, w, include_cb, default)         

        def add_line(flag, placeholder="", tip="", width=240, include=True):
            w = QLineEdit()
            if placeholder:
                w.setPlaceholderText(placeholder)
            if tip:
                w.setToolTip(tip)
            w.setFixedWidth(width)
            _add_row(flag, "line", w, default="",include=include)

        def add_combo(flag, items, default=None, tip="",include=True):
            w = QComboBox()
            w.addItems(items)
            if default in items:
                w.setCurrentText(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "combo", w, default=default, include=include)

        def add_check(flag, tip="", include=True):
            w = QCheckBox()
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "check", w, default=False, include=include)

        def add_spin(flag, minv, maxv, step=1, default=None, tip="", include=True):
            w = QSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "spin", w, default=default, include=include)

        def add_dspin(flag, minv, maxv, step=0.1, default=None, tip="", include=True):
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "dspin", w, default=default, include=include)

        def check_all():
            for flag, (kind, w, include_cb, default) in self.bhard_widgets.items():
                if include_cb is not None:
                    include_cb.setChecked(True)
        
        ca_button.clicked.connect(check_all)

        # Beam hardening / source & scintillator
        add_combo("--beam-hardening-method", ["none","standard"], default="none")
        add_combo("--calculate-source", ["none","standard"], default="none")
        add_dspin("--b-storage-ring", 0.0, 10.0, step=0.001, default=0.653)
        add_dspin("--e-storage-ring", 0.0, 50.0, step=0.1, default=6.0)
        add_combo("--filter-1-auto", ["False","True"], default="False")
        add_dspin("--filter-1-density", 0.0, 100.0, step=0.01, default=1.0)
        add_line("--filter-1-material", "none/Al/Cu/...")
        add_dspin("--filter-1-thickness", 0.0, 1e6, step=0.1, default=0.0)
        add_combo("--filter-2-auto", ["False","True"], default="False")
        add_dspin("--filter-2-density", 0.0, 100.0, step=0.01, default=1.0)
        add_line("--filter-2-material", "none/Al/Cu/...")
        add_dspin("--filter-2-thickness", 0.0, 1e6, step=0.1, default=0.0)
        add_combo("--filter-3-auto", ["False","True"], default="False")
        add_dspin("--filter-3-density", 0.0, 100.0, step=0.01, default=1.0)
        add_line("--filter-3-material", "none")
        add_dspin("--filter-3-thickness", 0.0, 1e6, step=0.1, default=0.0)
        add_dspin("--maximum-E", 0.0, 1e9, step=1.0, default=200000.0)
        add_dspin("--maximum-psi-urad", 0.0, 1e9, step=1.0, default=40.0)
        add_dspin("--minimum-E", 0.0, 1e9, step=1.0, default=1000.0)
        add_check("--read-pixel-size")
        add_check("--read-scintillator")
        add_dspin("--sample-density", 0.0, 100.0, step=0.01, default=1.0)
        add_line("--sample-material", "Fe")
        add_dspin("--scintillator-density", 0.0, 100.0, step=0.01, default=6.0)
        add_line("--scintillator-material", "LuAG_Ce")
        add_dspin("--scintillator-thickness", 0.0, 1e6, step=0.1, default=100.0)
        add_dspin("--source-distance", 0.0, 1e9, step=0.1, default=36.0)
        add_dspin("--step-E", 0.0, 1e9, step=1.0, default=500.0)

        self.tabs.addTab(bhard_tab, "Hardening")

    def _gather_bhard_args(self):
        args = []
        for flag, (kind, w, include_cb, _default) in self.bhard_widgets.items():
            if include_cb is not None and not include_cb.isChecked():
                continue #skip grayed lines                
            if kind == "line":
                val = w.text().strip()
                if val != "":
                    args += [flag, val]
            elif kind == "combo":
                args += [flag, w.currentText().strip()]
            elif kind == "check":
                if w.isChecked():
                    args += [flag]
            elif kind == "spin":
                args += [flag, str(w.value())]
            elif kind == "dspin":
                args += [flag, str(w.value())]
        return args
        
        
# ===== Phase TAB =====
    def _build_phase_tab(self):
        phase_tab = QWidget()
        outer = QVBoxLayout(phase_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        scroll.setWidget(form_host)
        outer.addWidget(scroll)

        self.phase_widgets = {}

        def _add_row(flag, kind, w, default=None, label_text=None, include=True):
            #include: show the checkbox, not--> always include in params
            label_text = label_text or flag

            # label cell = [ include_cb | "flag" ]
            label_widget = QWidget()
            h = QHBoxLayout(label_widget)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lbl = QLabel(label_text)

            include_cb = None

            if include:
                include_cb = QCheckBox()
                include_cb.setChecked(False) #default not enable
                h.addWidget(include_cb)
                lbl.setEnabled(False) #defalut not check
                w.setEnabled(False)

                def on_toggle(checked):
                    lbl.setEnabled(checked)
                    w.setEnabled(checked)
                    if not checked:
                        # reset to a sensible "off" state
                        if kind in ("spin", "dspin") and default is not None:
                            w.blockSignals(True)
                            w.setValue(default)
                            w.blockSignals(False)
                        elif kind == "combo":
                            if default is not None:
                                w.setCurrentText(str(default))
                            else:
                                w.setCurrentIndex(0)
                        elif kind == "line":
                            w.clear()
                        elif kind == "check":
                            w.setChecked(False)

                include_cb.toggled.connect(on_toggle)
            else:
                lbl.setEnabled(True)
                w.setEnabled(True)
            h.addWidget(lbl)
            h.addStretch(1)
            form.addRow(label_widget, w)

            self.phase_widgets[flag] = (kind, w, include_cb, default)   

        def add_line(flag, placeholder="", tip="", width=240, include=True):
            w = QLineEdit()
            if placeholder:
                w.setPlaceholderText(placeholder)
            if tip:
                w.setToolTip(tip)
            w.setFixedWidth(width)
            _add_row(flag, "line", w, default="",include=include)

        def add_combo(flag, items, default=None, tip="", include=True):
            w = QComboBox()
            w.addItems(items)
            if default in items:
                w.setCurrentText(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "combo", w, default=default, include=include)

        def add_check(flag, tip="", include=True):
            w = QCheckBox()
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "check", w, default=False, include=include)

        def add_spin(flag, minv, maxv, step=1, default=None, tip="", include=True):
            w = QSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "spin", w, default=default, include=include)

        def add_dspin(flag, minv, maxv, step=0.1, default=None, tip="", include=True):
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "dspin", w, default=default, include=include)

        # Phase retrieval   
        add_combo("--retrieve-phase-method", ["none","paganin","Gpaganin"], default="none")
        add_dspin("--pixel-size", 0.0, 1e9, step=0.01, default=0.0)        
        add_dspin("--energy", 0.0, 1e6, step=0.1, default=0.0)
        add_dspin("--propagation-distance", 0.0, 1e6, step=0.1, default=0.0)        
        add_dspin("--retrieve-phase-W", 0.0, 1.0, step=0.0001,default=0.0002)
        add_dspin("--retrieve-phase-alpha", 0.0, 1e6, step=0.0001,default=0.0)
        add_dspin("--retrieve-phase-delta-beta", 0.0, 1e9, step=0.1,default=1500.0)
        add_spin("--retrieve-phase-pad", 0, 1024, step=1,default=1)
  
        self.tabs.addTab(phase_tab, "Phase")

    def _gather_phase_args(self):
        args = []
        for flag, (kind, w, include_cb, _default) in self.phase_widgets.items():
            if include_cb is not None and not include_cb.isChecked():
                continue #skip grayed lines
            if kind == "line":
                val = w.text().strip()
                if val != "":
                    args += [flag, val]
            elif kind == "combo":
                args += [flag, w.currentText().strip()]
            elif kind == "check":
                if w.isChecked():
                    args += [flag]
            elif kind == "spin":
                args += [flag, str(w.value())]
            elif kind == "dspin":
                if bool(w.value()) != "None":
                    args += [flag, str(w.value())]
                else:
                    pass

        return args        

    # ===== RINGS TAB =====
    def _build_rings_tab(self):
        rings_tab = QWidget()
        outer = QVBoxLayout(rings_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        scroll.setWidget(form_host)
        outer.addWidget(scroll)

        self.rings_widgets = {}

        def _add_row(flag, kind, w, default=None, label_text=None, include=True):
            #include: show the checkbox, not--> always include in params
            label_text = label_text or flag

            # label cell = [ include_cb | "flag" ]
            label_widget = QWidget()
            h = QHBoxLayout(label_widget)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lbl = QLabel(label_text)

            include_cb = None

            if include:
                include_cb = QCheckBox()
                include_cb.setChecked(False) #default not enable
                h.addWidget(include_cb)
                lbl.setEnabled(False) #defalut not check
                w.setEnabled(False)

                def on_toggle(checked):
                    lbl.setEnabled(checked)
                    w.setEnabled(checked)
                    if not checked:
                        # reset to a sensible "off" state
                        if kind in ("spin", "dspin") and default is not None:
                            w.blockSignals(True)
                            w.setValue(default)
                            w.blockSignals(False)
                        elif kind == "combo":
                            if default is not None:
                                w.setCurrentText(str(default))
                            else:
                                w.setCurrentIndex(0)
                        elif kind == "line":
                            w.clear()
                        elif kind == "check":
                            w.setChecked(False)

                include_cb.toggled.connect(on_toggle)
            else:
                lbl.setEnabled(True)
                w.setEnabled(True)
            h.addWidget(lbl)
            h.addStretch(1)
            form.addRow(label_widget, w)

            self.rings_widgets[flag] = (kind, w, include_cb, default)   

        def add_line(flag, placeholder="", tip="", width=240, include=True):
            w = QLineEdit()
            if placeholder:
                w.setPlaceholderText(placeholder)
            if tip:
                w.setToolTip(tip)
            w.setFixedWidth(width)
            _add_row(flag, "line", w, default="",include=include)

        def add_combo(flag, items, default=None, tip="", include=True):
            w = QComboBox()
            w.addItems(items)
            if default in items:
                w.setCurrentText(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "combo", w, default=default, include=include)

        def add_check(flag, tip="", include=True):
            w = QCheckBox()
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "check", w, default=False, include=include)

        def add_spin(flag, minv, maxv, step=1, default=None, tip="", include=True):
            w = QSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "spin", w, default=default, include=include)

        def add_dspin(flag, minv, maxv, step=0.1, default=None, tip="", include=True):
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "dspin", w, default=default, include=include)


        # Stripe/ring filters — default to 'fw' (Fourier-wavelet) so the most
        # commonly-used ring removal is active out of the box.
        add_combo("--remove-stripe-method", ["none","fw","ti","vo-all"], default="fw", include=False) #always include
        add_combo("--fw-filter", ["haar","db5","sym5","sym16"], default="sym16")
        add_spin("--fw-level", 0, 64, step=1, default=7)
        add_check("--fw-pad")
        add_dspin("--fw-sigma", 0.0, 100.0, step=0.1, default=1.0)
        add_dspin("--ti-beta", 0.0, 1.0, step=0.001, default=0.022)
        add_dspin("--ti-mask", 0.0, 1.0, step=0.01, default=1.0)
        add_spin("--vo-all-dim", 1, 3, step=1, default=1)
        add_spin("--vo-all-la-size", 1, 4096, step=2, default=61)
        add_spin("--vo-all-sm-size", 1, 4096, step=2, default=21)
        add_dspin("--vo-all-snr", 0.0, 100.0, step=0.1, default=3.0)

        self.tabs.addTab(rings_tab, "Rings")

    def _gather_rings_args(self):
        args = []
        for flag, (kind, w, include_cb, _default) in self.rings_widgets.items():
            if include_cb is not None and not include_cb.isChecked():
                continue #skip grayed lines

            if kind == "line":
                val = w.text().strip()
                if val != "":
                    args += [flag, val]
            elif kind == "combo":
                args += [flag, w.currentText().strip()]
            elif kind == "check":
                if w.isChecked():
                    args += [flag]
            elif kind == "spin":
                args += [flag, str(w.value())]
            elif kind == "dspin":
                args += [flag, str(w.value())]

        return args

# ===== Geometry TAB =====
    def _build_Geometry_tab(self):
        Geometry_tab = QWidget()
        outer = QVBoxLayout(Geometry_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        scroll.setWidget(form_host)
        outer.addWidget(scroll)

        self.Geometry_widgets = {}

        def _add_row(flag, kind, w, default=None, label_text=None, include=True):
            #include: show the checkbox, not--> always include in params
            label_text = label_text or flag

            # label cell = [ include_cb | "flag" ]
            label_widget = QWidget()
            h = QHBoxLayout(label_widget)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lbl = QLabel(label_text)

            include_cb = None

            if include:
                include_cb = QCheckBox()
                include_cb.setChecked(False) #default not enable
                h.addWidget(include_cb)
                lbl.setEnabled(False) #defalut not check
                w.setEnabled(False)

                def on_toggle(checked):
                    lbl.setEnabled(checked)
                    w.setEnabled(checked)
                    if not checked:
                        # reset to a sensible "off" state
                        if kind in ("spin", "dspin") and default is not None:
                            w.blockSignals(True)
                            w.setValue(default)
                            w.blockSignals(False)
                        elif kind == "combo":
                            if default is not None:
                                w.setCurrentText(str(default))
                            else:
                                w.setCurrentIndex(0)
                        elif kind == "line":
                            w.clear()
                        elif kind == "check":
                            w.setChecked(False)

                include_cb.toggled.connect(on_toggle)
            else:
                lbl.setEnabled(True)
                w.setEnabled(True)
            h.addWidget(lbl)
            h.addStretch(1)
            form.addRow(label_widget, w)

            self.Geometry_widgets[flag] = (kind, w, include_cb, default) 

        def add_line(flag, placeholder="", tip="", width=240, include=True):
            w = QLineEdit()
            if placeholder:
                w.setPlaceholderText(placeholder)
            if tip:
                w.setToolTip(tip)
            w.setFixedWidth(width)
            _add_row(flag, "line", w, default="",include=include)

        def add_combo(flag, items, default=None, tip="", include=True):
            w = QComboBox()
            w.addItems(items)
            if default in items:
                w.setCurrentText(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "combo", w, default=default, include=include)

        def add_check(flag, tip="", include=True):
            w = QCheckBox()
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "check", w, default=False, include=include)

        def add_spin(flag, minv, maxv, step=1, default=None, tip="", include=True):
            w = QSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "spin", w, default=default, include=include)

        def add_dspin(flag, minv, maxv, step=0.1, default=None, tip="", include=True):
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "dspin", w, default=default, include=include)

        # Geometry & lamino
        add_line("--blocked-views", "[[0,1.2],[3,3.14]]")        
        add_dspin("--rotate-proj-angle", -360.0, 360.0, step=0.1, default=0.0)
        add_spin("--rotate-proj-order", 0, 5, step=1, default=1)        
        add_dspin("--lamino-angle", -90, 90.0, step=0.01, default=0.0)
        add_spin("--lamino-end-row", -1, 10_000_000, step=1, default=-1)
        add_dspin("--lamino-search-step", 0.0, 1e6, step=0.01, default=0.25)
        add_dspin("--lamino-search-width", 0.0, 1e6, step=0.1, default=5.0)
        add_spin("--lamino-start-row", 0, 10_000_000, step=1, default=0)
  
        self.tabs.addTab(Geometry_tab, "Geometry")

    def _gather_Geometry_args(self):
        args = []
        for flag, (kind, w, include_cb, _default) in self.Geometry_widgets.items():
            if include_cb is not None and not include_cb.isChecked():
                continue #skip grayed lines

            if kind == "line":
                val = w.text().strip()
                if val != "":
                    args += [flag, val]
            elif kind == "combo":
                args += [flag, w.currentText().strip()]
            elif kind == "check":
                if w.isChecked():
                    args += [flag]
            elif kind == "spin":
                args += [flag, str(w.value())]
            elif kind == "dspin":
                args += [flag, str(w.value())]

        return args        

# ===== Data TAB =====
    def _build_Data_tab(self):
        Data_tab = QWidget()
        outer = QVBoxLayout(Data_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        scroll.setWidget(form_host)
        outer.addWidget(scroll)

        self.data_widgets = {}

        def _add_row(flag, kind, w, default=None, label_text=None, include=True):
            #include: show the checkbox, not--> always include in params
            label_text = label_text or flag

            # label cell = [ include_cb | "flag" ]
            label_widget = QWidget()
            h = QHBoxLayout(label_widget)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lbl = QLabel(label_text)

            include_cb = None

            if include:
                include_cb = QCheckBox()
                include_cb.setChecked(False) #default not enable
                h.addWidget(include_cb)
                lbl.setEnabled(False) #defalut not check
                w.setEnabled(False)

                def on_toggle(checked):
                    lbl.setEnabled(checked)
                    w.setEnabled(checked)
                    if not checked:
                        # reset to a sensible "off" state
                        if kind in ("spin", "dspin") and default is not None:
                            w.blockSignals(True)
                            w.setValue(default)
                            w.blockSignals(False)
                        elif kind == "combo":
                            if default is not None:
                                w.setCurrentText(str(default))
                            else:
                                w.setCurrentIndex(0)
                        elif kind == "line":
                            w.clear()
                        elif kind == "check":
                            w.setChecked(False)

                include_cb.toggled.connect(on_toggle)
            else:
                lbl.setEnabled(True)
                w.setEnabled(True)
            h.addWidget(lbl)
            h.addStretch(1)
            form.addRow(label_widget, w)

            self.data_widgets[flag] = (kind, w, include_cb, default) 

        def add_line(flag, placeholder="", tip="", width=240, include=True):
            w = QLineEdit()
            if placeholder:
                w.setPlaceholderText(placeholder)
            if tip:
                w.setToolTip(tip)
            w.setFixedWidth(width)
            _add_row(flag, "line", w, default="",include=include)

        def add_combo(flag, items, default=None, tip="", include=True):
            w = QComboBox()
            w.addItems(items)
            if default in items:
                w.setCurrentText(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "combo", w, default=default, include=include)

        def add_check(flag, tip="", include=True):
            w = QCheckBox()
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "check", w, default=False, include=include)

        def add_spin(flag, minv, maxv, step=1, default=None, tip="", include=True):
            w = QSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "spin", w, default=default, include=include)

        def add_dspin(flag, minv, maxv, step=0.1, default=None, tip="", include=True):
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "dspin", w, default=default, include=include)

        add_line("--dark-file-name", "/path/dark.h5")
        add_line("--flat-file-name", "/path/flat.h5")
        add_line("--out-path-name", "/path/out")
        add_combo("--save-format", ["h5nolinks","h5","h5sino","tiff"], default="h5nolinks", include=False) #always include; matches tomocupy default
        add_check("--config-update")
        add_line("--logs-home", "/home/user/logs", include=False) #always include
        add_check("--verbose", include=False) #always include
        
        self.tabs.addTab(Data_tab, "Data")

    def _gather_Data_args(self):
        args = []
        for flag, (kind, w, include_cb, _default) in self.data_widgets.items():
            if include_cb is not None and not include_cb.isChecked():
                continue #skip grayed lines

            if kind == "line":
                val = w.text().strip()
                if val != "":
                    args += [flag, val]
            elif kind == "combo":
                args += [flag, w.currentText().strip()]
            elif kind == "check":
                if w.isChecked():
                    args += [flag]
            elif kind == "spin":
                args += [flag, str(w.value())]
            elif kind == "dspin":
                args += [flag, str(w.value())]

        return args        


# ===== Performance TAB =====
    def _build_Performance_tab(self):
        Performance_tab = QWidget()
        outer = QVBoxLayout(Performance_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        scroll.setWidget(form_host)
        outer.addWidget(scroll)

        self.perf_widgets = {}

        def _add_row(flag, kind, w, default=None, label_text=None, include=True):
            #include: show the checkbox, not--> always include in params
            label_text = label_text or flag

            # label cell = [ include_cb | "flag" ]
            label_widget = QWidget()
            h = QHBoxLayout(label_widget)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lbl = QLabel(label_text)

            include_cb = None

            if include:
                include_cb = QCheckBox()
                include_cb.setChecked(False) #default not enable
                h.addWidget(include_cb)
                lbl.setEnabled(False) #defalut not check
                w.setEnabled(False)

                def on_toggle(checked):
                    lbl.setEnabled(checked)
                    w.setEnabled(checked)
                    if not checked:
                        # reset to a sensible "off" state
                        if kind in ("spin", "dspin") and default is not None:
                            w.blockSignals(True)
                            w.setValue(default)
                            w.blockSignals(False)
                        elif kind == "combo":
                            if default is not None:
                                w.setCurrentText(str(default))
                            else:
                                w.setCurrentIndex(0)
                        elif kind == "line":
                            w.clear()
                        elif kind == "check":
                            w.setChecked(False)

                include_cb.toggled.connect(on_toggle)
            else:
                lbl.setEnabled(True)
                w.setEnabled(True)
            h.addWidget(lbl)
            h.addStretch(1)
            form.addRow(label_widget, w)

            self.perf_widgets[flag] = (kind, w, include_cb, default) 

        def add_line(flag, placeholder="", tip="", width=240, include=True):
            w = QLineEdit()
            if placeholder:
                w.setPlaceholderText(placeholder)
            if tip:
                w.setToolTip(tip)
            w.setFixedWidth(width)
            _add_row(flag, "line", w, default="",include=include)

        def add_combo(flag, items, default=None, tip="", include=True):
            w = QComboBox()
            w.addItems(items)
            if default in items:
                w.setCurrentText(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "combo", w, default=default, include=include)

        def add_check(flag, tip="", include=True):
            w = QCheckBox()
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "check", w, default=False, include=include)

        def add_spin(flag, minv, maxv, step=1, default=None, tip="", include=True):
            w = QSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "spin", w, default=default, include=include)

        def add_dspin(flag, minv, maxv, step=0.1, default=None, tip="", include=True):
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "dspin", w, default=default, include=include)

        # Perfomance related settings
        # Clear output folder before each run — always included, defaults True so
        # stale partial reconstructions don't linger; user can flip to False.
        add_combo("--clear-folder", ["False","True"], default="True", include=False)
        add_combo("--dtype", ["float32","float16"], default="float32", include=False) #always include
        add_spin("--start-column", 0, 10_000_000, step=1, default=0, include=False) #always include
        add_spin("--end-column", -1, 10_000_000, step=1, default=-1, include=False) #always include
        add_spin("--start-proj", 0, 10_000_000, step=1, default=0, include=False) #always include
        add_spin("--end-proj", -1, 10_000_000, step=1, default=-1, include=False) #always include
        add_spin("--start-row", 0, 10_000_000, step=1, default=0, include=False) #always include
        add_spin("--end-row", -1, 10_000_000, step=1, default=-1, include=False) #always include
        add_spin("--nproj-per-chunk", 1, 65535, step=1, default=8, include=False) #always include      
        add_spin("--nsino-per-chunk", 1, 65535, step=1, default=8, include=False) #always include      
        add_spin("--max-read-threads", 1, 1024, step=1, default=4, include=False) #always include
        add_spin("--max-write-threads", 1, 1024, step=1, default=8, include=False) #always include      

  
        self.tabs.addTab(Performance_tab, "Performance")

    def _gather_Performance_args(self):
        args = []
        for flag, (kind, w, include_cb, _default) in self.perf_widgets.items():
            if include_cb is not None and not include_cb.isChecked():
                continue #skip grayed lines

            if kind == "line":
                val = w.text().strip()
                if val != "":
                    args += [flag, val]
            elif kind == "combo":
                args += [flag, w.currentText().strip()]
            elif kind == "check":
                if w.isChecked():
                    args += [flag]
            elif kind == "spin":
                args += [flag, str(w.value())]
            elif kind == "dspin":
                args += [flag, str(w.value())]

        return args

# ===== AI COR TAB =====
    def _build_ai_tab(self):
        """AI COR settings tab. Exposes tomocupy's --infer-* (fine mode) and
        --bin-infer-* (full/bin mode) inference parameters. The tab also owns
        the --ai-search-method selector, which _apply_ai_cor reads to decide
        which pipeline flag family to pass through."""
        ai_tab = QWidget()
        outer = QVBoxLayout(ai_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        scroll.setWidget(form_host)
        outer.addWidget(scroll)

        self.ai_widgets = {}

        def _add_row(flag, kind, w, default=None, label_text=None, include=True):
            label_text = label_text or flag
            label_widget = QWidget()
            h = QHBoxLayout(label_widget)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lbl = QLabel(label_text)
            include_cb = None
            if include:
                include_cb = QCheckBox()
                include_cb.setChecked(False)
                h.addWidget(include_cb)
                lbl.setEnabled(False)
                w.setEnabled(False)

                def on_toggle(checked):
                    lbl.setEnabled(checked)
                    w.setEnabled(checked)
                    if not checked:
                        if kind in ("spin", "dspin") and default is not None:
                            w.blockSignals(True)
                            w.setValue(default)
                            w.blockSignals(False)
                        elif kind == "combo":
                            if default is not None:
                                w.setCurrentText(str(default))
                            else:
                                w.setCurrentIndex(0)
                        elif kind == "line":
                            w.clear()
                        elif kind == "check":
                            w.setChecked(False)

                include_cb.toggled.connect(on_toggle)
            else:
                lbl.setEnabled(True)
                w.setEnabled(True)
            h.addWidget(lbl)
            h.addStretch(1)
            form.addRow(label_widget, w)
            self.ai_widgets[flag] = (kind, w, include_cb, default)

        def add_line(flag, placeholder="", tip="", width=240, include=True,
                    default_text=""):
            w = QLineEdit()
            if placeholder:
                w.setPlaceholderText(placeholder)
            if tip:
                w.setToolTip(tip)
            w.setFixedWidth(width)
            if default_text:
                w.setText(default_text)
            _add_row(flag, "line", w, default="", include=include)

        def add_combo(flag, items, default=None, tip="", include=True):
            w = QComboBox()
            w.addItems(items)
            if default in items:
                w.setCurrentText(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "combo", w, default=default, include=include)

        def add_check(flag, tip="", include=True, default_checked=False):
            w = QCheckBox()
            if default_checked:
                w.setChecked(True)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "check", w, default=False, include=include)

        def add_spin(flag, minv, maxv, step=1, default=None, tip="", include=True):
            w = QSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            if default is not None:
                w.setValue(default)
            if tip:
                w.setToolTip(tip)
            _add_row(flag, "spin", w, default=default, include=include)

        # Search-method selector (always included; controls which pipeline
        # tomocupy runs and thus which flag family below is relevant).
        add_combo("--ai-search-method", ["fine", "full"], default="fine",
                    tip="fine: single-stage inference_pipeline. "
                        "full: two-stage bin_inference_pipeline.",
                  include=False)

        # ---- fine-mode --infer-* params ---------------------------------
        add_check("--infer-use-8bits", default_checked=True,
                    tip="Requantize pixels to 8 bits before inference.")
        add_line("--infer-downsample-factor",
                placeholder="[1] or [1,2,4]",
                tip="List of downsample factors applied to try slices.")
        add_line("--infer-num-windows",
                placeholder="[3]",
                tip="Number of aggregation windows per slice.")
        add_line("--infer-window-size",
                placeholder="[518]",
                tip="Square window size (DINOv2 native = 518).")
        add_spin("--infer-seed-number", 0, 1_000_000, step=1, default=10,
                tip="RNG seed for reproducibility.")
        add_combo("--infer-input-data-type", ["raw", "try"], default="raw",
                    tip="Which cache the AI inference reads from.")
        add_check("--infer-save-intermediate-data",
                    tip="Save per-slice model predictions to predicts_all.npz.")
        add_line("--infer-input-dir",
                placeholder="/path/to/tiff/dir or blank",
                tip="Direct TIFF input dir (bypass try cache).")
        add_line("--infer-batch-list",
                placeholder="/path/to/list.txt or blank",
                tip="Batch txt file listing input directories.")
        add_line("--infer-out-dir-name",
                placeholder="/path/for/output or blank",
                tip="Output batches directory.")

        # ---- full/bin-mode --bin-infer-* params -------------------------
        add_line("--bin-infer-bin-sizes",
                placeholder="[24,12]",
                tip="Pixel step per bin per refinement stage.")
        add_line("--bin-infer-bin-counts",
                placeholder="[4,2]",
                tip="Bins per stage (must be even).")
        add_spin("--bin-infer-num-frames", 1, 1024, step=1, default=2,
                tip="Frames aggregated per bin.")
        add_line("--bin-infer-num-windows",
                placeholder="[20]",
                tip="Aggregation windows for bin inference.")
        add_line("--bin-infer-window-size",
                placeholder="[518]",
                tip="Square window size for bin inference.")
        add_spin("--bin-infer-aggregator-depth", 1, 64, step=1, default=5,
                tip="Attention layers in the feature aggregator.")
        add_spin("--bin-infer-aggregator-num-heads", 1, 64, step=1, default=12,
                tip="Attention heads per layer.")
        add_line("--bin-infer-downsample-factor",
                placeholder="[1]",
                tip="Downsample factor applied to try slices (bin mode).")
        add_combo("--bin-infer-input-data-type", ["raw", "try"], default="raw",
                    tip="Which cache the bin AI reads from.")
        add_spin("--bin-infer-seed-number", 0, 1_000_000, step=1, default=10,
                tip="RNG seed for reproducibility (bin mode).")
        add_check("--bin-infer-use-8bits", default_checked=True,
                    tip="Requantize pixels to 8 bits (bin mode).")
        add_check("--bin-infer-save-intermediate-data",
                    tip="Save per-slice bin predictions to range_predicts_all.npz.")
        add_line("--bin-infer-input-dir",
                placeholder="/path/to/tiff/dir or blank")
        add_line("--bin-infer-batch-list",
                placeholder="/path/to/list.txt or blank")
        add_line("--bin-infer-out-dir-name",
                placeholder="/path/for/output or blank")

        self.tabs.addTab(ai_tab, "AI COR")

    def _gather_ai_args(self):
        """Return the AI-COR CLI flags currently enabled on the AI tab.
        Skipped when the model path is not set (nothing else makes sense
        without a model)."""
        model_path = self.ai_model_path.text().strip()
        if not model_path or not os.path.exists(model_path):
            return []
        args = []
        for flag, (kind, w, include_cb, _default) in self.ai_widgets.items():
            if include_cb is not None and not include_cb.isChecked():
                continue
            if kind == "line":
                val = w.text().strip()
                if val != "":
                    args += [flag, val]
            elif kind == "combo":
                args += [flag, w.currentText().strip()]
            elif kind == "check":
                if w.isChecked():
                    args += [flag]
            elif kind in ("spin", "dspin"):
                args += [flag, str(w.value())]
        return args

    def _current_ai_search_method(self):
        """Read the AI search method (fine|full) from the tab widget."""
        entry = getattr(self, 'ai_widgets', {}).get("--ai-search-method")
        if entry is None:
            return "fine"
        _, w, _, _ = entry
        try:
            return w.currentText().strip() or "fine"
        except (AttributeError, RuntimeError):
            return "fine"

        # ===== advanced config tab====
    def _build_advanced_config_tab(self):
        config_tab = QWidget()
        self.tabs.addTab(config_tab, "Advanced Config")
        config_main = QVBoxLayout(config_tab) #main layout for the tab
        config_txt_main = QHBoxLayout() #layout for left/right
        config_rows = QHBoxLayout()
        config_rows.setSpacing(5)
        # some common functions for both try/full
        func_box = QHBoxLayout()
        self.use_conf_box = QCheckBox("Enable config")
        self.use_conf_box.setChecked(False)
        func_box.addWidget(self.use_conf_box)
        load_config_btn = QPushButton("Load Config")
        save_config_btn = QPushButton("Save Config")
        save_params_btn = QPushButton("Save Params")
        save_params_btn.setToolTip("Save all current GUI reconstruction parameters for the selected dataset")
        load_config_btn.clicked.connect(self.load_config)
        save_config_btn.clicked.connect(self.save_config)
        save_params_btn.clicked.connect(self._save_current_scan_params)
        func_box.addWidget(load_config_btn)
        func_box.addWidget(save_config_btn)
        func_box.addWidget(save_params_btn)
        config_main.addLayout(func_box)
        #left frame for Try
        left_try_box = QGroupBox("Try Recon Config")
        #left - row 1: config txt box
        left_try_layout = QVBoxLayout()
        self.config_editor_try = QTextEdit()
        self.config_editor_try.setFixedHeight(300)
        self.config_editor_try.setStyleSheet("QTextEdit { border: 1px solid gray; font-size: 12.5pt; }")
        self.config_editor_try.focusInEvent = lambda event: self.highlight_editor(self.config_editor_try, event)
        self.config_editor_try.focusOutEvent = lambda event: self.unhighlight_editor(self.config_editor_try, event)
        left_try_layout.addWidget(self.config_editor_try)
        left_try_box.setLayout(left_try_layout)
        #right - row 1: conf full txt box
        right_full_box = QGroupBox("Full Recon Config")
        right_full_layout = QVBoxLayout()
        self.config_editor_full = QTextEdit()
        self.config_editor_full.setFixedHeight(300)
        self.config_editor_full.setStyleSheet("QTextEdit { border: 1px solid gray; font-size: 12.5pt; }")
        self.config_editor_full.focusInEvent = lambda event: self.highlight_editor(self.config_editor_full, event)
        self.config_editor_full.focusOutEvent = lambda event: self.unhighlight_editor(self.config_editor_full, event)
        right_full_layout.addWidget(self.config_editor_full)
        right_full_box.setLayout(right_full_layout)

        self.active_editor = self.config_editor_try
        config_txt_main.addWidget(left_try_box,1)
        config_txt_main.addWidget(right_full_box,1)
        config_main.addLayout(config_txt_main)

    #     # ==== BATCH PROCESSING TAB ====
    #     batch_tab = QWidget()
    #    self.tabs.addTab(batch_tab, "Batch Processing")
    #     self._build_batch_tab(batch_tab)

    # def _build_batch_tab(self, batch_tab):
    #    """Build the batch processing tab for managing multiple datasets"""
    #     main_layout = QVBoxLayout(batch_tab)

    #     # Top controls
    #     controls_layout = QHBoxLayout()

    #    refresh_list_btn = QPushButton("Refresh File List")
    #     refresh_list_btn.clicked.connect(self._refresh_batch_file_list)
    #     controls_layout.addWidget(refresh_list_btn)

    #    save_cor_btn = QPushButton("Save COR to CSV")
    #     save_cor_btn.clicked.connect(self._batch_save_cor_csv)
    #    save_cor_btn.setToolTip("Save COR values to batch_cor_values.csv in data folder")
    #     controls_layout.addWidget(save_cor_btn)

    #    load_cor_btn = QPushButton("Load COR from CSV")
    #     load_cor_btn.clicked.connect(self._batch_load_cor_csv)
    #    load_cor_btn.setToolTip("Load COR values from batch_cor_values.csv in data folder")
    #     controls_layout.addWidget(load_cor_btn)

    #     controls_layout.addStretch()

    #    select_all_btn = QPushButton("Select All")
    #     select_all_btn.clicked.connect(self._batch_select_all)
    #     controls_layout.addWidget(select_all_btn)

    #    deselect_all_btn = QPushButton("Deselect All")
    #     deselect_all_btn.clicked.connect(self._batch_deselect_all)
    #     controls_layout.addWidget(deselect_all_btn)

    #     main_layout.addLayout(controls_layout)

    #     # File list table
    #     self.batch_file_table = QTableWidget()
    #     self.batch_file_table.setColumnCount(9)
    #     self.batch_file_table.setHorizontalHeaderLabels([
    #        "Select", "Filename", "Size", "COR", "Status", "View Data", "View Try", "View Full", "Actions"
    #     ])

    #     # Configure table
    #     self.batch_file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    #     self.batch_file_table.setSortingEnabled(True)  # Enable column sorting
    #     header = self.batch_file_table.horizontalHeader()
    #     header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Select checkbox
    #     header.setSectionResizeMode(1, QHeaderView.Interactive)  # Filename - user can resize
    #     header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Size
    #     header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # COR
    #     header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Status
    #     header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # View Data
    #     header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # View Try
    #     header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # View Full
    #     header.setSectionResizeMode(8, QHeaderView.ResizeToContents)  # Actions
    #     header.setSectionsClickable(True)  # Make headers clickable for sorting

    #     # Set initial width for filename column to be wider (can be resized by user)
    #     self.batch_file_table.setColumnWidth(1, 400)

    #     main_layout.addWidget(self.batch_file_table)

    #     # Machine and GPU configuration
    #     machine_layout = QHBoxLayout()
    #    machine_layout.addWidget(QLabel("Target Machine:"))

    #     self.batch_machine_box = QComboBox()
    #    self.batch_machine_box.addItems(["Local", "tomo1", "tomo2", "tomo3", "tomo4", "tomo5"])
    #    self.batch_machine_box.setCurrentText("Local")
    #    self.batch_machine_box.setToolTip("Select machine to run batch reconstructions")
    #     machine_layout.addWidget(self.batch_machine_box)

    #     machine_layout.addSpacing(20)
    #    machine_layout.addWidget(QLabel("GPUs per machine:"))

    #     self.batch_gpus_per_machine_batch = QSpinBox()
    #     self.batch_gpus_per_machine_batch.setMinimum(1)
    #     self.batch_gpus_per_machine_batch.setMaximum(8)
    #     self.batch_gpus_per_machine_batch.setValue(1)
    #    self.batch_gpus_per_machine_batch.setToolTip("Number of GPUs available on the target machine (1 job per GPU)")
    #     machine_layout.addWidget(self.batch_gpus_per_machine_batch)

    #     machine_layout.addSpacing(20)
    #    self.batch_queue_label = QLabel("Queue: 0 jobs waiting")
    #     machine_layout.addWidget(self.batch_queue_label)

    #     machine_layout.addStretch()
    #     main_layout.addLayout(machine_layout)

    #     # Batch operations
    #     batch_ops_layout = QHBoxLayout()

    #    batch_ops_layout.addWidget(QLabel("Batch Operations (on selected):"))

    #    batch_try_btn = QPushButton("Run Try on Selected")
    #     batch_try_btn.clicked.connect(self._batch_run_try_selected)
    #     batch_ops_layout.addWidget(batch_try_btn)

    #    batch_full_btn = QPushButton("Run Full on Selected")
    #     batch_full_btn.clicked.connect(self._batch_run_full_selected)
    #     batch_ops_layout.addWidget(batch_full_btn)

    #    self.batch_stop_btn = QPushButton("Stop Queue")
    #     self.batch_stop_btn.clicked.connect(self._batch_stop_queue)
    #     self.batch_stop_btn.setEnabled(False)
    #     batch_ops_layout.addWidget(self.batch_stop_btn)

    #     batch_ops_layout.addStretch()

    #    remove_selected_btn = QPushButton("Remove Selected from List")
    #     remove_selected_btn.clicked.connect(self._batch_remove_selected)
    #     batch_ops_layout.addWidget(remove_selected_btn)

    #     main_layout.addLayout(batch_ops_layout)

    #     # Progress section
    #    progress_group = QGroupBox("Batch Progress")
    #     progress_layout = QVBoxLayout()

    #     self.batch_progress_bar = QProgressBar()
    #     self.batch_progress_bar.setValue(0)
    #     progress_layout.addWidget(self.batch_progress_bar)

    #    self.batch_status_label = QLabel("Ready")
    #     progress_layout.addWidget(self.batch_status_label)

    #     progress_group.setLayout(progress_layout)
    #     main_layout.addWidget(progress_group)

    #     # Initialize batch state
    #     self.batch_file_list = []
    #     self.batch_current_index = 0
    #     self.batch_running = False
    #     self.batch_job_queue = []  # Queue of pending jobs: [(file_info, recon_type, machine), ...]
    #     self.batch_running_jobs = {}  # Dict of currently running jobs: {gpu_id: (process, file_info, recon_type)}
    #     self.batch_available_gpus = []  # List of available GPU IDs
    #     self.batch_total_jobs = 0  # Total number of jobs in current batch
    #     self.batch_completed_jobs = 0  # Number of completed jobs
    #    self.batch_current_machine = "Local"  # Current machine for batch
    #     self.batch_current_num_gpus = 1  # Current number of GPUs


    # ===== HELP METHODS =====
    def help_tomo(self):
        """Run the CLI `tomocupy recon (or recon_steps) -h` and show output in the GUI log."""
        name = "tomocupy-help"
        recon_way = self.recon_way_box.currentText()
        self.log_output.append(f"\U0001f4d6[{name}] tomocupy {recon_way} -h")

        p = QProcess(self)
        p.setProcessChannelMode(QProcess.SeparateChannels)

        p.readyReadStandardOutput.connect(
            lambda: self.log_output.append(
                bytes(p.readAllStandardOutput()).decode(errors="ignore")
            )
        )
        p.readyReadStandardError.connect(
            lambda: self.log_output.append(
                bytes(p.readAllStandardError()).decode(errors="ignore")
            )
        )

        def _done(code, _status):
            try:
                self.process[:] = [(pp, nn) for (pp, nn) in self.process if pp is not p]
            except Exception:
                pass
            self.log_output.append(f"[{name}] done." if code == 0
                                else f"[{name}] failed with code {code}.")

        p.finished.connect(_done)
        p.errorOccurred.connect(
            lambda _err: self.log_output.append(f"[{name}] {p.errorString()}")
        )

        if not isinstance(self.process, list):
            self.process = []
        self.process.append((p, name))

        p.start("tomocupy", [str(recon_way), "-h"])

    def update_cmap(self): #link to cmap dropdown
        self.current_cmap = self.cmap_box.currentText()
        if VISPY_AVAILABLE and self._current_img is not None:
            self.image_visual.cmap = self.current_cmap
            self.canvas.update()
        elif not VISPY_AVAILABLE and self._current_img is not None:
            try:
                lut = (_mpl.colormaps[self.current_cmap](np.linspace(0, 1, 256)) * 255).astype(np.uint8)
                self._pg_image_item.setLookupTable(lut[:, :3])
                self._pg_image_item.update()
                self.canvas_widget.update()
            except Exception as e:
                self.log_output.append(f'cmap error: {type(e).__name__}: {e}')
                pass

    def update_vmin_vmax(self): #link to min/max input
        try:
            vmin = float(self.min_input.text()) if self.min_input.text() else None
        except ValueError:
            vmin = None
        try:
            vmax = float(self.max_input.text()) if self.max_input.text() else None
        except ValueError:
            vmax = None
        self.vmin = vmin
        self.vmax = vmax

        if VISPY_AVAILABLE and self._current_img is not None and self.vmin is not None and self.vmax is not None:
            self.image_visual.clim = (self.vmin, self.vmax)
            self.canvas.update()
        elif not VISPY_AVAILABLE and self._current_img is not None and self.vmin is not None and self.vmax is not None:
            self._pg_apply_levels(self.vmin, self.vmax)

    def refresh_current_image(self):
        idx = self.slice_slider.value()
        if self.full_files and 0 <= idx < len(self.full_files):
            entry = self.full_files[idx]
            if self.full_h5 is not None:
                self.show_image(int(entry), flag="full_h5")
            else:
                self.show_image(entry, flag=None)
        elif self.preview_files and 0 <= idx < len(self.preview_files):
            self.show_image(self.preview_files[idx], flag=None)
        elif 0 <= idx < self.raw_files_num:
            self.show_image(img_path=idx, flag="raw")

    def highlight_editor(self, editor, event):
        editor.setStyleSheet("QTextEdit { border: 2px solid green; font-size: 12.5pt; }")
        self.active_editor = editor
        QTextEdit.focusInEvent(editor, event)

    def unhighlight_editor(self, editor, event):
        editor.setStyleSheet("QTextEdit { border: 1px solid gray; font-size: 12.5pt; }")
        QTextEdit.focusOutEvent(editor, event)

    def eventFilter(self, obj, event):
        if obj == self.canvas_widget and event.type() == QEvent.Wheel:
            step = 1 if event.angleDelta().y() > 0 else -1
            new_val = self.slice_slider.value() + step
            new_val = max(0, min(self.slice_slider.maximum(), new_val))
            self.slice_slider.setValue(new_val)
            return True
        return super().eventFilter(obj, event)

    def browse_data_folder(self):
        start_dir = self.data_path.text().strip() or os.path.expanduser("/")
        dialog = QFileDialog(self, "Select data folder")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setDirectory(start_dir)
        if dialog.exec():
            self.data_path.setText(dialog.selectedFiles()[0])
            self.refresh_main_table()
            # Auto-refresh batch file list when folder is selected
            #self._refresh_batch_file_list() #TODO:Need decide if remove the batch process tab

    def _load_cor_data(self, data_folder, h5_files):
        """
        Load COR data from CSV or JSON file.
        CSV format (batch_cor_values.csv): Filename,COR
        JSON format (rot_cen.json): {full_path: cor_value}

        CSV takes priority if both exist.

        Returns:
            tuple: (cor_data_dict, list_of_keys)
                    cor_data_dict uses full file paths as keys
        """
        import csv

        csv_path = os.path.join(data_folder, "batch_cor_values.csv")
        json_path = os.path.join(data_folder, "rot_cen.json")

        cor_data = {}

        # Try CSV first (legacy format, takes priority)
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    # Build a mapping from filename to full path
                    filename_to_path = {os.path.basename(f): f for f in h5_files}

                    for row in reader:
                        filename = row.get('Filename', '').strip()
                        cor_value = row.get('COR', '').strip()

                        if filename and cor_value:
                            # Convert filename to full path for consistency
                            full_path = filename_to_path.get(filename)
                            if full_path:
                                cor_data[full_path] = cor_value

                                self.log_output.append(f'<span style="color:green;">Loaded {len(cor_data)} COR values from batch_cor_values.csv</span>')
                return cor_data, list(cor_data.keys())

            except Exception as e:
              self.log_output.append(f'<span style="color:red;">Error loading CSV: {e}</span>')

        # Try JSON if CSV doesn't exist or failed
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    raw = json.load(f)
                    # Normalise: values may be lists [cor] or bare numbers/strings
                    cor_data = {
                        k: str(v[0]) if isinstance(v, list) and v else str(v)
                        for k, v in raw.items()
                    }
                    self.log_output.append(f'<span style="color:green;">Loaded {len(cor_data)} COR values from rot_cen.json</span>')
                    return cor_data, list(cor_data.keys())
            except json.JSONDecodeError as e:
              self.log_output.append(f'<span style="color:red;">Error loading rot_cen.json: {e}</span>')
              return {}, []

        # No COR file found
              self.log_output.append('<span style="color:orange;"> No COR file found (checked batch_cor_values.csv and rot_cen.json)</span>')
        return {}, []

    def refresh_main_table(self):
        table_folder = self.data_path.text()
        if not table_folder or not os.path.isdir(table_folder):
            QMessageBox.warning(self, "Warning", "Please select a valid data folder first.")
            return
        if self.batch_running:
            reply = QMessageBox.question(
                self, 'Queue Running',
                f'A batch queue is currently running ({len(self.batch_running_jobs)} jobs active, {len(self.batch_job_queue)} queued).\n\n'
                f'Refreshing will delete the table widgets but jobs will continue running in the background.\n\n'
                f'Continue with refresh?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
                self.log_output.append(f'<span style="color:orange;"> Refreshed file list while queue was running - status updates may be lost</span>')
        h5_files = sorted(glob.glob(os.path.join(table_folder, "*.h5")), key=os.path.getmtime, reverse=True)
        self.batch_file_main_table.setSortingEnabled(False)
        self.batch_file_main_table.setRowCount(0)
        self.batch_file_main_list = []
        self.batch_last_clicked_row = None
        self._recon_params_data = None  # force reload for new folder

        # Load COR data from JSON or CSV (CSV takes priority if both exist)
        self.cor_data, fns = self._load_cor_data(table_folder, h5_files)
        #populate table
        for f in h5_files:
            filename = os.path.basename(f)
            row = self.batch_file_main_table.rowCount()
            self.batch_file_main_table.insertRow(row)
            #check recon status
            proj_name = os.path.splitext(filename)[0]
            try_dir = os.path.join(f"{table_folder}_rec", "try_center", proj_name)
            has_try = os.path.isdir(try_dir) and len(glob.glob(os.path.join(try_dir, "*.tiff"))) > 0
            full_info = self._resolve_full_recon(table_folder, proj_name)
            has_full = full_info['kind'] is not None
            # Determine row color based on reconstruction status
            if has_full:
                row_color = "green"  # Full reconstruction exists
                if full_info['range'] is not None:
                    num_1, num_2 = full_info['range']
                    status_item = QTableWidgetItem(f"Full {num_1}-{num_2}")
                else:
                  status_item = QTableWidgetItem(f"Full ({full_info['kind']})")
            elif has_try:
                row_color = "orange"  # Only try reconstruction exists
                status_item = QTableWidgetItem("Done try")
            else:
                row_color = "red"  # No reconstruction
                status_item = QTableWidgetItem("Ready")
            # Store file info
            file_info = {
                'path': f,
                'filename': filename,
                'status': status_item,
                'row': row,
                'recon_status': row_color
            }
            self.batch_file_main_list.append(file_info)   
            # Checkbox for selection with shift-click support
            checkbox = QCheckBox()
            checkbox.clicked.connect(lambda checked, r=row: self._batch_checkbox_clicked(r, checked))
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.batch_file_main_table.setCellWidget(row, 0, checkbox_widget)
            file_info['checkbox'] = checkbox

            # Filename - show full name and set tooltip with full path
            filename_item = QTableWidgetItem(filename)
            filename_item.setToolTip(f"{filename}\n\nFull path:\n{f}")
            self.batch_file_main_table.setItem(row, 1, filename_item)       

            # COR value (editable)
            if f in fns:
                cor_val = self.cor_data[f]
                if isinstance(cor_val, list):
                    cor_val = str(cor_val[0]) if cor_val else ""
                else:
                    cor_val = str(cor_val)
                cor_input = QLineEdit(cor_val)
            else:
                cor_input = QLineEdit()
            cor_input.setPlaceholderText("COR value")
            cor_input.setAlignment(Qt.AlignCenter)
            cor_input.setFixedWidth(80)

            self.batch_file_main_table.setCellWidget(row, 2, cor_input)
            file_info['cor_input'] = cor_input
            #allow modify table directly but also update rot_cen.json
            try:
                cor_input.editingFinished.connect(
                    lambda fp=f, r=row: self._on_main_cor_edited(fp,r)
                )
            except Exception:
                pass

            # Status
            self.batch_file_main_table.setItem(row, 3, status_item)
            file_info['status'] = status_item

            # File size
            try:
                file_size = os.path.getsize(f)
                size_str = self._format_file_size(file_size)
                size_item = QTableWidgetItem(size_str)
                size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                # Store numeric value for proper sorting
                size_item.setData(Qt.UserRole, file_size)
                self.batch_file_main_table.setItem(row, 4, size_item)
            except Exception as e:
                self.batch_file_main_table.setItem(row, 4, QTableWidgetItem("N/A"))
            # Actions button (placeholder for future actions)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_layout.setSpacing(2)
            self.batch_file_main_table.setCellWidget(row, 5, actions_widget)

            # View Data button
            view_data_btn = QPushButton("View Data")
            view_data_btn.setFixedWidth(80)
            view_data_btn.clicked.connect(lambda checked, fp=f: self._batch_view_data(fp))
            self.batch_file_main_table.setCellWidget(row, 6, view_data_btn)

            # Apply colored left border indicator based on reconstruction status
            # Create a colored indicator in the checkbox column
            checkbox_widget.setStyleSheet(f"QWidget {{ border-left: 6px solid {row_color}; }}")
        # Visual grouping by dataset series (adjacent rows with same filename
        # prefix get the same subtle background tint on the filename cell).
        self._apply_series_tint()
        # Auto-uncheck any file whose size is a small fraction of its series
        # median — aborted scans with no useful data.
        self._auto_skip_small_size_in_series()
        # Re-enable sorting after populating the table
        #self.batch_file_main_table.setSortingEnabled(True)
        # Highlight the first row
        if self.batch_file_main_table.rowCount() > 0:
            self.batch_file_main_table.setCurrentCell(0, 0)  # Select the first cell in the first row
            self.highlight_scan = h5_files[0] #always the latest coming in scan
            self.highlight_row = 0
            self.log_output.append(f'Clicked on {self.highlight_scan}')
            self._load_scan_params(self.highlight_scan)

    def _save_cor_data(self, data_folder, cor_data_dict):
        """
        Save COR data to both CSV and JSON formats.
        Saves to the format that already exists, or JSON if neither exists.

        Args:
            data_folder: Path to data folder
            cor_data_dict: Dictionary with full file paths as keys and COR values
        """
        import csv

        csv_path = os.path.join(data_folder, "batch_cor_values.csv")
        json_path = os.path.join(data_folder, "rot_cen.json")

        csv_exists = os.path.exists(csv_path)
        json_exists = os.path.exists(json_path)

        # Save to CSV if it exists or if both don't exist (backward compatibility)
        if csv_exists:
            try:
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Filename', 'COR'])
                    for full_path, cor_value in sorted(cor_data_dict.items()):
                        filename = os.path.basename(full_path)
                        writer.writerow([filename, cor_value])
                        self.log_output.append(f'<span style="color:green;">COR values saved to CSV</span>')
            except Exception as e:
              self.log_output.append(f'<span style="color:red;">Failed to write CSV: {e}</span>')

        # Always save to JSON (current format)
        try:
            with open(json_path, "w") as f:
                json.dump(cor_data_dict, f, indent=2)
            if not csv_exists:
              self.log_output.append(f'<span style="color:green;">COR values saved to JSON</span>')
        except Exception as e:
          self.log_output.append(f'<span style="color:red;">Failed to write JSON: {e}</span>')

    # ===== PER-DATASET RECONSTRUCTION PARAMS =====

    @staticmethod
    def _get_widget_value(kind, w):
        if kind in ("spin", "dspin"):
            return w.value()
        elif kind == "combo":
            return w.currentText()
        elif kind == "line":
            return w.text()
        elif kind == "check":
            return w.isChecked()
        return None

    @staticmethod
    def _set_widget_value(kind, w, val):
        try:
            if kind == "spin":
                w.setValue(int(val))
            elif kind == "dspin":
                w.setValue(float(val))
            elif kind == "combo":
                idx = w.findText(str(val))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif kind == "line":
                w.setText(str(val))
            elif kind == "check":
                w.setChecked(bool(val))
        except Exception:
            pass

    def _gather_all_gui_params(self):
        """Collect all current GUI reconstruction parameters into a dict."""
        params = {
            "recon_way":       self.recon_way_box.currentText(),
            "recon_way_full":  self.recon_way_box_full.currentText(),
            "cor_method":      self.cor_method_box.currentText(),
            "cor_method_full": self.cor_full_method.currentText(),
            "cuda_try":        self.cuda_box_try.value(),
            "cuda_full":       self.cuda_full_box.value(),
            "use_conf":        self.use_conf_box.isChecked(),
            "config_try":      self.config_editor_try.toPlainText(),
            "config_full":     self.config_editor_full.toPlainText(),
        }
        for tab_key, widget_dict in [
            ("params",      self.param_widgets),
            ("bhard",       self.bhard_widgets),
            ("phase",       self.phase_widgets),
            ("rings",       self.rings_widgets),
            ("geometry",    self.Geometry_widgets),
            ("data",        self.data_widgets),
            ("performance", self.perf_widgets),
        ]:
            tab_data = {}
            for flag, (kind, w, include_cb, _default) in widget_dict.items():
                val = self._get_widget_value(kind, w)
                inc = include_cb.isChecked() if include_cb is not None else None
                tab_data[flag] = {"value": val, "include": inc}
            params[tab_key] = tab_data
        return params

    def _apply_params_to_gui(self, params):
        """Apply a saved params dict to all GUI reconstruction widgets."""
        try:
            self.recon_way_box.setCurrentText(params.get("recon_way", "recon"))
            self.recon_way_box_full.setCurrentText(params.get("recon_way_full", "recon"))
            self.cor_method_box.setCurrentText(params.get("cor_method", "manual"))
            self.cor_full_method.setCurrentText(params.get("cor_method_full", "manual"))
            self.cuda_box_try.setValue(int(params.get("cuda_try", 0)))
            self.cuda_full_box.setValue(int(params.get("cuda_full", 0)))
            self.use_conf_box.setChecked(bool(params.get("use_conf", False)))
            self.config_editor_try.setPlainText(params.get("config_try", ""))
            self.config_editor_full.setPlainText(params.get("config_full", ""))
            for tab_key, widget_dict in [
                ("params",      self.param_widgets),
                ("bhard",       self.bhard_widgets),
                ("phase",       self.phase_widgets),
                ("rings",       self.rings_widgets),
                ("geometry",    self.Geometry_widgets),
                ("data",        self.data_widgets),
                ("performance", self.perf_widgets),
            ]:
                tab_data = params.get(tab_key, {})
                for flag, (kind, w, include_cb, _default) in widget_dict.items():
                    if flag in tab_data:
                        entry = tab_data[flag]
                        self._set_widget_value(kind, w, entry.get("value"))
                        if include_cb is not None and entry.get("include") is not None:
                            include_cb.setChecked(bool(entry["include"]))
        except Exception as e:
          self.log_output.append(f'<span style="color:orange;">Error applying params: {e}</span>')

    def _load_recon_params_file(self, data_folder):
        """Load recon_params.json from data folder. Returns dict keyed by full file path."""
        path = os.path.join(data_folder, "recon_params.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
              self.log_output.append(f'<span style="color:orange;">Could not load recon_params.json: {e}</span>')
        return {}

    def _save_recon_params_file(self, data_folder, params_dict):
        """Save recon_params.json to data folder."""
        path = os.path.join(data_folder, "recon_params.json")
        try:
            with open(path, "w") as f:
                json.dump(params_dict, f, indent=2)
        except Exception as e:
          self.log_output.append(f'<span style="color:red;">Could not save recon_params.json: {e}</span>')

    def _save_current_scan_params(self):
        """Save current GUI params for the highlighted scan to recon_params.json."""
        if not self.highlight_scan:
            return
        self._persist_params_for_files([self.highlight_scan])

    def _persist_params_for_files(self, files):
        """Snapshot the current GUI parameters and persist them under each
        of the given file paths in recon_params.json. The write happens on
        a background thread so the GUI doesn't block on slow filesystems
        (NFS, networked home, etc.)."""
        if not files:
            return
        data_folder = self.data_path.text().strip()
        if not data_folder or not os.path.isdir(data_folder):
            return
        if self._recon_params_data is None:
            self._recon_params_data = self._load_recon_params_file(data_folder)
        snapshot = self._gather_all_gui_params()
        # Update the in-memory cache immediately so any subsequent load
        # in this session reads the new values.
        targets = [p for p in files if p]
        for path in targets:
            self._recon_params_data[path] = snapshot
        # Take a copy of what to write so the worker thread doesn't race
        # with future updates of self._recon_params_data.
        to_write = dict(self._recon_params_data)
        target_path = os.path.join(data_folder, "recon_params.json")

        def _writer():
            import sys
            import tempfile
            tmp = None
            try:
                # Unique tmp filename per thread avoids races when several
                # writer threads are in flight against the same target.
                fd, tmp = tempfile.mkstemp(prefix="recon_params.",
                                          suffix=".tmp",
                                            dir=data_folder)
                with os.fdopen(fd, "w") as fh:
                    json.dump(to_write, fh, indent=2)
                os.replace(tmp, target_path)
                tmp = None
            except Exception as exc:
                # Background thread — can't touch GUI directly. Print to
                # stderr so it shows up in the launching shell / log file.
                print(f"[tomogui] background param save failed: {exc}",
                      file=sys.stderr)
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        import threading
        t = threading.Thread(target=_writer, daemon=True,
                            name="recon-params-saver")
        t.start()

    def _load_scan_params(self, proj_file):
        """Load and apply saved GUI params for proj_file, if they exist."""
        data_folder = self.data_path.text().strip()
        if not data_folder or not os.path.isdir(data_folder):
            return
        if self._recon_params_data is None:
            self._recon_params_data = self._load_recon_params_file(data_folder)
        if proj_file in self._recon_params_data:
            self._apply_params_to_gui(self._recon_params_data[proj_file])
            self.log_output.append(f'Loaded params for {os.path.basename(proj_file)}')

    def _apply_series_tint(self):
        """Shade the filename cell (column 1) and the COR cell (column 2) with
        a rotating colour per dataset series so adjacent rows in the same
        series share a tint and the boundary between series is visually obvious.
        Series is derived from the filename: everything before the final numeric index.
        """
        import re
        idx_re = re.compile(r'^(.*?)[._-]*(\d+)$')

        def series_key(name):
            base = os.path.splitext(name)[0]
            m = idx_re.match(base)
            return m.group(1) if m else base

        # Visible tints that read clearly on both dark and light themes.
        palette = [
            QColor(42,  72, 108),   # blue
            QColor(42, 108,  60),   # green
            QColor(120, 72,  30),   # amber
            QColor( 90, 42, 120),   # purple
            QColor(130, 50,  80),   # magenta
            QColor( 30,  90, 108),  # teal
        ]

        table = self.batch_file_main_table
        prev_series = None
        colour_idx = -1
        n_series = 0
        n_rows = 0
        for row in range(table.rowCount()):
            item = table.item(row, 1)
            if item is None:
                continue
            s = series_key(item.text())
            if s != prev_series:
                colour_idx = (colour_idx + 1) % len(palette)
                prev_series = s
                n_series += 1
            # Tint the filename cell
            item.setBackground(palette[colour_idx])
            # Tint the COR cell too (via its QLineEdit stylesheet — the COR
            # column uses a cell widget, so setBackground on an item doesn't apply).
            cor_w = table.cellWidget(row, 2)
            if cor_w is not None:
                c = palette[colour_idx]
                cor_w.setStyleSheet(
                    f"QLineEdit {{ background: rgb({c.red()},{c.green()},{c.blue()}); "
                    f"color: #ffffff; }}"
                )
            n_rows += 1
        # One summary log line so it's visible if something went wrong
        try:
            self.log_output.append(
              f'<span style="color:#888;">series tint applied — '
                f'{n_series} series across {n_rows} rows</span>'
            )
        except Exception:
            pass

    def _resolve_full_recon(self, data_folder, proj_name):
        """Locate the full reconstruction for one dataset.

        Tomocupy's default output is a single ``.h5`` file
        (``{data}_rec/{proj}_rec.h5``, with the volume at ``/exchange/data``).
        Legacy TIFF output lives at ``{data}_rec/{proj}_rec/*.tiff``.

        Returns a dict::

            {'kind': 'h5' | 'tiff' | None,
            'h5_path': str | None,
            'tiff_files': list[str],
            'n_slices': int,
            'range': (first, last) | None}

        The dict is always populated; missing pieces are None / [] / 0. If
        both TIFF and H5 exist we prefer H5 (that's the new default).
        """
        result = {'kind': None, 'h5_path': None, 'tiff_files': [],
                    'n_slices': 0, 'range': None}
        if not data_folder or not proj_name:
            return result
        base = os.path.join(f"{data_folder}_rec", f"{proj_name}_rec")
        h5_path = f"{base}.h5"
        if os.path.isfile(h5_path):
            try:
                with h5py.File(h5_path, 'r') as fh:
                    n = int(fh['/exchange/data'].shape[0])
                result.update(kind='h5', h5_path=h5_path,
                              n_slices=n, range=(0, max(0, n - 1)))
                return result
            except (OSError, KeyError):
                pass
        if os.path.isdir(base):
            tiffs = sorted(glob.glob(os.path.join(base, "*.tiff")))
            if tiffs:
                try:
                    n1 = int(Path(tiffs[0]).stem.split("_")[-1])
                    n2 = int(Path(tiffs[-1]).stem.split("_")[-1])
                except (ValueError, IndexError):
                    n1, n2 = 0, len(tiffs) - 1
                result.update(kind='tiff', tiff_files=tiffs,
                              n_slices=len(tiffs), range=(n1, n2))
        return result

    def _auto_contrast_for_file(self, proj_file, lo_pct=5, hi_pct=95):
        """Compute (vmin, vmax) as formatted strings from the 5–95 % percentile
        of a representative slice of the reconstructed volume. Returns
        (None, None) if no reconstruction is available.

        Prefers the full reconstruction (H5 first, then legacy TIFF dir) —
        that's what tomolog uploads. Falls back to the try-center TIFFs
        if full is not available.
        """
        data_folder = self.data_path.text().strip()
        if not data_folder:
            return (None, None)
        proj_name = os.path.splitext(os.path.basename(proj_file))[0]
        try:
            info = self._resolve_full_recon(data_folder, proj_name)
            if info['kind'] == 'h5':
                with h5py.File(info['h5_path'], 'r') as fh:
                    dset = fh['/exchange/data']
                    mid = dset.shape[0] // 2
                    arr = np.asarray(dset[mid], dtype=np.float32)
            elif info['kind'] == 'tiff':
                mid_path = info['tiff_files'][len(info['tiff_files']) // 2]
                arr = np.array(Image.open(mid_path)).astype(np.float32)
            else:
                try_dir = os.path.join(f"{data_folder}_rec", "try_center", proj_name)
                tiffs = sorted(glob.glob(os.path.join(try_dir, "*.tiff")))
                if not tiffs:
                    return (None, None)
                arr = np.array(Image.open(tiffs[len(tiffs) // 2])).astype(np.float32)
            lo = float(np.percentile(arr, lo_pct))
            hi = float(np.percentile(arr, hi_pct))
            if hi <= lo:
                return (None, None)
            return (f"{lo:.6g}", f"{hi:.6g}")
        except Exception:
            return (None, None)

    def _auto_skip_small_size_in_series(self):
        """For each series group in the table, compute the median file size and
        auto-uncheck + mark 'Skipped (small)' any file whose size is below
        `self.size_min_pct`% of that median. Aborted/partial scans typically
        produce files a tiny fraction of the size of their series peers.
        Set the spinner to 0 to disable."""
        import re
        idx_re = re.compile(r'^(.*?)[._-]*(\d+)$')

        def series_key(name):
            base = os.path.splitext(name)[0]
            m = idx_re.match(base)
            return m.group(1) if m else base

        try:
            pct = float(self.size_min_pct.value())
        except (AttributeError, ValueError, TypeError):
            pct = 50.0
        if pct <= 0.0:
            return   # feature disabled

        # Group files by series and gather (file_info, size_bytes)
        groups = {}
        for fi in self.batch_file_main_list:
            try:
                sz = os.path.getsize(fi['path'])
            except OSError:
                sz = 0
            fi['_size_bytes'] = sz
            groups.setdefault(series_key(fi['filename']), []).append(fi)

        n_skipped = 0
        for sk, entries in groups.items():
            sizes = sorted(e['_size_bytes'] for e in entries if e['_size_bytes'] > 0)
            if len(sizes) < 2:
                continue        # lonely file, no basis for comparison
            median = sizes[len(sizes) // 2] if len(sizes) % 2 else 0.5 * (
                sizes[len(sizes) // 2 - 1] + sizes[len(sizes) // 2])
            cutoff = median * (pct / 100.0)
            for fi in entries:
                too_small = fi['_size_bytes'] < cutoff
                fi['skipped_small'] = too_small
                if too_small:
                    n_skipped += 1
                    row = fi.get('row')
                    # Uncheck
                    try:
                        fi['checkbox'].setChecked(False)
                    except Exception:
                        pass
                    # Mark status
                    if row is not None:
                        item = QTableWidgetItem("Skipped (small)")
                        item.setForeground(QColor("#999"))
                        self.batch_file_main_table.setItem(row, 3, item)
                        fi['status'] = item
                        fi['recon_status'] = 'gray'
                        # Dim the left-border indicator
                        cb_w = self.batch_file_main_table.cellWidget(row, 0)
                        if cb_w is not None:
                            cb_w.setStyleSheet("QWidget { border-left: 6px solid #666; }")

        if n_skipped:
            self.log_output.append(
              f'<span style="color:#888;">auto-skipped {n_skipped} file(s) with '
                f'size &lt; {pct:g}% of their series median.</span>'
            )

    def _on_main_cor_edited(self, file_path:str, row:int):
        """
        Called when the COR QLineEdit in the MAIN table is edited.
        Writes/updates COR data to both CSV and JSON if they exist.
        """
        data_folder = self.data_path.text().strip()
        if not data_folder:
            return

        # Get the widget (QLineEdit) from the table
        w = self.batch_file_main_table.cellWidget(row, 2)
        if w is None:
            return
        txt = w.text().strip()
        if txt == "":
            return  # user cleared it

        # Validate numeric
        try:
            float(txt)
        except ValueError:
          self.log_output.append(f'<span style="color:red;">Invalid COR "{txt}" for {os.path.basename(file_path)}</span>')
          return

        # Update in-memory data
        self.cor_data[file_path] = txt

        # Save to file(s)
        self._save_cor_data(data_folder, self.cor_data)

        # Keep your list consistent: store the widget, not a string
        try:
            self.batch_file_main_list[row]["cor_input"] = w
        except Exception:
            pass    

    def on_table_row_clicked(self, row, column):
        # Save current GUI params for the previously selected dataset before switching.
        # During a batch run we intentionally keep whatever the user set in the GUI
        # and apply it to every file, so skip the per-scan save/load in that case.
        if not self._batch_active and self.highlight_scan:
            self._save_current_scan_params()
        # Get the filename from the clicked row
        filename_item = self.batch_file_main_table.item(row, 1)  # Column 1 contains the filename
        filename = filename_item.text()
        # Update self.highlight_scan with the full path of the selected file
        for file_info in self.batch_file_main_list:
            if file_info['filename'] == filename:
                self.highlight_scan = file_info['path']
                self.highlight_row = row #gives index of the self.batch_file_table_list
        self._update_full_btn_state()  # grey out only if this file is running locally
        self.log_output.append(f'Click on {self.highlight_scan} now for other operations')
        # Load saved params for newly selected dataset only when NOT batch-processing
        if not self._batch_active:
            self._load_scan_params(self.highlight_scan)

    def load_config(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        if dialog.exec():
            fn = dialog.selectedFiles()[0]
            with open(fn, "r") as f:
                target = self.active_editor or self.config_editor_try
                target.setPlainText(f.read())

    def save_config(self):
        dialog = QFileDialog(self)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        if dialog.exec():
            fn = dialog.selectedFiles()[0]
            text = (self.active_editor or self.config_editor_try).toPlainText()
            with open(fn, "w") as f:
                f.write(text)
    
    def save_params_to_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"{self.data_path.text().strip()}/tomocupy_reconparams_{timestamp}.json"
        params = {} #gather all enabled params
        for widgets in [self.param_widgets, self.phase_widgets, self.Geometry_widgets,
                        self.bhard_widgets, self.rings_widgets, self.perf_widgets, self.data_widgets]:    
            for flag, (kind, w, include_cb, _default) in widgets.items():
                if include_cb is not None and not include_cb.isChecked():
                    continue #skip grayed lines
                if kind == "line":
                    val = w.text().strip()
                    if val != "":
                        params[flag] = val
                elif kind == "combo":
                    params[flag] = w.currentText().strip()
                elif kind == "check":
                    if w.isChecked():
                        params[flag] = "checked"
                elif kind == "spin":
                    params[flag] = str(w.value())
                elif kind == "dspin":
                    params[flag] = str(w.value())
        if params:
            try:
                with open(fn, "a") as f:
                    json.dump(params, f, indent=2)
            except Exception as e:
                self.log_output.append(f'Failed to save params to {fn}: {e}')
        self.log_output.append(f'Saved enabled params to {fn}')

    def load_params_from_file(self):
        start_dir = self.data_path.text().strip()
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("/")
        dialog = QFileDialog(self, "Select params folder")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilters(["JSON files (*.json)", "All files (*)"])
        dialog.selectNameFilter("JSON files (*.json)")
        dialog.setDirectory(start_dir)
        load_fn = None
        if dialog.exec():
            load_fn = dialog.selectedFiles()[0]
        if not load_fn or not os.path.isfile(load_fn):
          self.log_output.append(f'<span style="color:red;">Invalid file: {load_fn}</span>')
          return
        with open(load_fn, "r") as f:
            try:
                params = json.load(f)
            except Exception as e:
              self.log_output.append(f'<span style="color:red;">Failed to load params from {load_fn}: {e}</span>')
              return
        for key, v in params.items():
            for widgets in [self.param_widgets, self.phase_widgets, self.Geometry_widgets,
                            self.bhard_widgets, self.rings_widgets, self.perf_widgets, self.data_widgets]:   
                if key in widgets.keys():
                    kind, w, include_cb, _default = widgets[key]
                    if include_cb is not None and not include_cb.isChecked():
                        include_cb.setChecked(True)
                    if kind == "line":
                        w.setText(v)
                    elif kind == "combo":
                        if v in [w.itemText(i) for i in range(w.count())]:
                            w.setCurrentText(v)
                    elif kind == "check":
                        w.setChecked(True)
                    elif kind == "spin":
                        try:
                            iv = int(v)
                            if w.minimum() <= iv <= w.maximum():
                                w.setValue(iv)
                        except Exception:
                            pass
                    elif kind == "dspin":
                        try:
                            fv = float(v)
                            if w.minimum() <= fv <= w.maximum():
                                w.setValue(fv)
                        except Exception:
                            pass
        self.log_output.append(f'Loaded params from {load_fn}')

    def clear_log(self):
        self.log_output.clear()

    def save_log(self):
        start_dir = self.data_path.text().strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_fn = f"{start_dir}/Process_log_{timestamp}.txt"
        if os.path.exists(log_fn):
            return
        # QTextEdit isn't iterable; pull its plain text contents.
        text = self.log_output.toPlainText()
        with open(log_fn, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        self.log_output.append(f'Saved log to {log_fn}')


    def preset_beamhardening(self):
        enable_flags = ["--beam-hardening-method", "--calculate-source",
                        "--b-storage-ring","--e-storage-ring", 
                        "--filter-1-auto", "--filter-1-density",
                        "--filter-1-material", "--filter-1-thickness", 
                        "--filter-2-auto","--filter-2-density",
                        "--filter-2-material","--filter-2-thickness",
                        "--filter-3-auto", "--filter-3-density", 
                        "--filter-3-material","--filter-3-thickness",
                        "--maximum-E","--maximum-psi-urad",
                        "--minimum-E", "--read-pixel-size", 
                        "--read-scintillator","--sample-density", 
                        "--sample-material", "--scintillator-density", 
                        "--scintillator-material", "--scintillator-thickness", 
                        "--source-distance", "--step-E"]
        for flag in enable_flags:
            if flag in self.bhard_widgets:
                kind, w, include_cb, _default = self.bhard_widgets[flag]
                if include_cb is not None and not include_cb.isChecked():
                    include_cb.setChecked(True)
        self.log_output.append("Enable beamhardening params")

    def preset_phase(self):
        enable_flags = ["--retrieve-phase-method", 
                        "--pixel-size", 
                        "--propagation-distance", 
                        "--energy", 
                        "--retrieve-phase-alpha"]
        for flag in enable_flags:
            if flag in self.phase_widgets:
                kind, w, include_cb, _default = self.phase_widgets[flag]
                if include_cb is not None and not include_cb.isChecked():
                    include_cb.setChecked(True)
        self.recon_way_box_full.setCurrentText("recon_steps")
        self.log_output.append("Enable phase params, set recon way to recon_steps for full recon")

    def preset_laminography(self):
        enable_flags = ["--lamino-angle", 
                        "--lamino-end-row", 
                        "--lamino-search-step", 
                        "--lamino-search-width", 
                        "--lamino-start-row"]
        for flag in enable_flags:
            if flag in self.Geometry_widgets:
                kind, w, include_cb, _default = self.Geometry_widgets[flag]
                if include_cb is not None and not include_cb.isChecked():
                    include_cb.setChecked(True)
        #self.recon_way_box.setCurrentText("recon_steps")
        self.recon_way_box_full.setCurrentText("recon_steps")
        self.log_output.append("Enable laminography params, set recon way to recon_steps for full recon")
        #place holder: any params need to disable?

    def abort_process(self):
        if not self.process:
          self.log_output.append('<span style="color:red;">No running process.</span>')
          return

        for p, name in list(self.process):
            if p.state() != QProcess.NotRunning:
                p.terminate()

        for p, name in list(self.process):
            if p.state() != QProcess.NotRunning:
                if not p.waitForFinished(2000):
                    p.kill()
                    p.waitForFinished(2000)
                    self.log_output.append(f'<span style="color:red;">\u26d4 [{name}] aborted.</span>')

        self.process.clear()
        self.batch_running = False

    def run_command_live(self, cmd, proj_file=None, job_label=None, *, wait=False, cuda_devices=None):
        """
        cmd: list of command and args
        proj_file: projections path
        job_label: label for the job
        wait: whether to wait for the process to finish, if False, return QProcess object immediately, if True, return exit code when finished
        cuda_devices: str, e.g. "0", "1" for GPU tomocupy use
        """
        scan_id = None
        if proj_file:
            try:
                base = os.path.splitext(os.path.basename(proj_file))[0]
                scan_id = base[-4:]
            except Exception:
                scan_id = None

        if job_label is None:
            job_label = "job"
        name = f"{job_label}-{scan_id}" if scan_id else job_label

        # Wrap with `env HDF5_USE_FILE_LOCKING=FALSE` so it's set at the shell
        # level too — QProcessEnvironment.systemEnvironment().insert has been
        # observed to be swallowed on some Linux/PyQt combos. Belt-and-braces.
        if cmd and cmd[0] != "env":
            cmd = ["env", "HDF5_USE_FILE_LOCKING=FALSE"] + list(cmd)

        cli_str = " ".join(map(str, cmd))
        self.log_output.append(f'\U0001f680 [{name}] start: {cli_str}')
        QApplication.processEvents()

        p = QProcess(self)
        p.setProcessChannelMode(QProcess.ForwardedChannels)

        #define env with CUDA_VISIBLE_DEVICES
        env = QProcessEnvironment.systemEnvironment()
        if cuda_devices is not None:
            env.insert("CUDA_VISIBLE_DEVICES", str(cuda_devices))
        # NFS-hosted data (e.g. /data2, /data3) rejects HDF5's default BSD
        # file locks with EAGAIN, so tomocupy fails to create the output
        # H5 with "Resource temporarily unavailable". Disable HDF5 file
        # locking in the child; single-writer semantics are enforced by
        # tomogui not launching concurrent recons on the same file.
        env.insert("HDF5_USE_FILE_LOCKING", "FALSE")
        p.setProcessEnvironment(env)
        
        loop = QEventLoop() if wait else None
        result = {"code": None}

        def on_finished(code, status):
            try:
                self.process[:] = [(pp, nn) for (pp, nn) in self.process if pp is not p]
            except Exception:
                pass
            if code != 0:
                self.log_output.append(f'<span style="color:red;">[{name}] failed, check terminal</span>')
            result["code"] = code
            if loop is not None:
                loop.quit()

        def on_error(_err):
            if result["code"] is None:
                result["code"] = -1
            if loop is not None:
                loop.quit()
            # Guard: QProcess may already be destroyed if the event loop
            # reentered and cleaned up before this handler ran.
            try:
                msg = p.errorString()
            except RuntimeError:
                msg = str(_err)
            self.log_output.append(
              f'<span style="color:red;">[{name}] {msg}</span>'
            )

        p.finished.connect(on_finished)
        p.errorOccurred.connect(on_error)

        if not isinstance(self.process, list):
            self.process = []
        self.process.append((p, name))

        p.start(str(cmd[0]), [str(a) for a in cmd[1:]])

        if wait:
            loop.exec()
            # result["code"] is None if the process never started cleanly
            # (binary missing, permission denied). Return a sentinel int
            # rather than crashing on int(None).
            code = result.get("code")
            return int(code) if code is not None else -1

        return p

    # ===== RECONSTRUCTION METHODS =====
    def _update_row(self,row,color,status):
        row = self.highlight_row
        if row is None:
          self.log_output.append(f'<span style="color:red;">No row highlighted</span>')
          return
        self.batch_file_main_list[row]['recon_status'] = color
        checkbox_widget = self.batch_file_main_table.cellWidget(row, 0)
        if checkbox_widget:
            checkbox_widget.setStyleSheet(f"QWidget {{ border-left: 6px solid {color}; }}")
        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignCenter)
        self.batch_file_main_table.setItem(row, 3, status_item)
        self.batch_file_main_list[row]['status'] = status
        
    def _find_row_by_filepath(self, filepath):
        """Row lookup by full path. Scans `batch_file_main_list` (authoritative)
        and falls back to table text/tooltip matching if needed."""
        # 1) Authoritative: the Python-side list
        for fi in self.batch_file_main_list:
            if fi.get('path') == filepath:
                return fi.get('row')
        # 2) Fallback via table contents (filename is in column 1, not 0)
        base = os.path.basename(filepath)
        for row in range(self.batch_file_main_table.rowCount()):
            item = self.batch_file_main_table.item(row, 1)
            if item is None:
                continue
            if item.text() == base:
                return row
            if filepath and filepath in (item.toolTip() or ""):
                return row
        return None

    def _find_row_by_filename(self, filename, filename_col=None):
        """Row lookup by full path or by bare basename."""
        base = os.path.basename(filename)
        # 1) Authoritative Python list first
        for fi in self.batch_file_main_list:
            if fi.get('path') == filename:
                return fi.get('row')
            if fi.get('filename') == base:
                return fi.get('row')
        # 2) Fallback via the filename column (1, not 0 — 0 is the checkbox)
        for row in range(self.batch_file_main_table.rowCount()):
            item = self.batch_file_main_table.item(row, 1)
            if item is None:
                continue
            if item.text() == base:
                return row
            if filename and filename in (item.toolTip() or ""):
                return row
        return None

    def try_reconstruction(self):
        proj_file = self.highlight_scan
        if not proj_file:
            self.log_output.append(f"No file")
            return
        # Snapshot current GUI params for this file before running.
        self._persist_params_for_files([proj_file])
        recon_way = self.recon_way_box.currentText()
        cor_method = self.cor_method_box.currentText()
        cor_val = self.cor_input.text().strip()
        if cor_method == "auto":
            if cor_val:
              self.log_output.append(f'<span style="color:orange;">use auto method, ignore input cor</span>')
              pass
        else:
            try:
                cor = float(cor_val)
            except ValueError:
              self.log_output.append(f'<span style="color:red;">wrong rotation axis input</span>')
              return
        # cuda for tomocupy try
        gpu = str(self.cuda_box_try.value())
        #add check box for config, seperate from selecting parameters from GUI
        if self.use_conf_box.isChecked():
            self.log_output.append("You are using config file, only recon type, filename, rot axis from GUI")
            config_text = self.config_editor_try.toPlainText()
            if not config_text.strip():
              self.log_output.append(f'<span style="color:red;">no text in conf, stop</span>')
              return
            temp_try = os.path.join(self.data_path.text(), "temp_try.conf")
            with open(temp_try, "w") as f:
                f.write(config_text)
            # Base command
            cmd = ["tomocupy", str(recon_way), 
                "--reconstruction-type", "try", 
                "--config", temp_try, 
                "--file-name", proj_file]
            if cor_method == "auto":
                cmd += ["--rotation-axis-auto", "auto"]
            else:
                cmd += ["--rotation-axis-auto", "manual",
                        "--rotation-axis", str(cor)]
        else:
            self.log_output.append('You are using params from GUI')
            # Base command
            cmd = ["tomocupy", str(recon_way), 
                "--reconstruction-type", "try", 
                "--file-name", proj_file]
            if cor_method == "auto":
                cmd += ["--rotation-axis-auto", "auto"]
            else:
                cmd += ["--rotation-axis-auto", "manual",
                        "--rotation-axis", str(cor)]

            # Append tabs selections
            cmd += self._gather_params_args()
            cmd += self._gather_rings_args()
            cmd += self._gather_bhard_args()
            cmd += self._gather_phase_args()
            cmd += self._gather_Geometry_args()
            cmd += self._gather_Data_args()                        
            cmd += self._gather_Performance_args()
                                
        code = self.run_command_live(cmd, proj_file=proj_file, job_label="Try recon", wait=True, cuda_devices=gpu)
        try:
            if code == 0:
                self._update_row(row=self.highlight_row,color='orange',status='Done try') #change table content and self.batch_file_list
                self.log_output.append(f'<span style="color:green;">Done try recon {proj_file}</span>')
                return True
            else:
              self.log_output.append(f'<span style="color:red;">Try recon {proj_file} failed</span>')
              return False
        finally:
            if self.use_conf_box.isChecked():
                try:
                    if os.path.exists(temp_try):
                        os.remove(temp_try)
                        self.log_output.append(f"\U0001f9f9 Removed {temp_try}")
                except Exception as e:
                  self.log_output.append(f'<span style="color:red;">Could not remove {temp_try}: {e}</span>')

    def try_ai_reconstruction(self):
        """Run Try + AI-COR (in a single tomocupy call) then Full recon.

        The AI center-of-rotation search now lives inside tomocupy itself
        (``--rotation-axis-method ai``), so we no longer run a separate
        inference subprocess against the try TIFFs. Tomocupy performs the
        try reconstruction, caches the slices internally, runs the AI
        inference, and writes ``center_of_rotation.txt`` in the try output
        directory. We then read that file and launch the full recon.

        Starting-COR policy: prefer the currently-highlighted row's COR if it
        is a valid number; otherwise fall back to the top-bar Try COR input.
        """
        proj_file = self.highlight_scan
        if not proj_file:
            self.log_output.append("No file")
            return
        # Blank the viewer FIRST so nothing in the GUI keeps an NFS handle
        # on the H5 tomocupy is about to overwrite in the Full stage.
        self._blank_viewer()
        QApplication.processEvents()
        self._release_full_h5_for_write(proj_file)
        QApplication.processEvents()
        self._persist_params_for_files([proj_file])

        model_path = self.ai_model_path.text().strip()
        if not model_path or not os.path.exists(model_path):
            self.log_output.append('<span style="color:red;">Invalid AI model path</span>')
            return

        # Resolve the starting COR seed: row first, then top-bar.
        row_cor = ""
        if self.highlight_row is not None:
            w = self.batch_file_main_table.cellWidget(self.highlight_row, 2)
            if w is not None:
                row_cor = w.text().strip()
        bar_cor = self.cor_input.text().strip()
        seed = row_cor or bar_cor
        if seed:
            try:
                float(seed)
            except ValueError:
                seed = ""
        if seed:
            self.log_output.append(
                f'AI Reco seed COR = <b>{seed}</b> '
              f'(source: {"row" if row_cor else "top-bar"})'
            )

        # Wipe any stale center_of_rotation.txt so we only read this run's
        # result. Tomocupy appends, so leftover values would confuse the
        # last-line parse below.
        data_folder = self.data_path.text().strip()
        proj_name = os.path.splitext(os.path.basename(proj_file))[0]
        try_dir = os.path.join(f"{data_folder}_rec", "try_center", proj_name)
        stale = os.path.join(try_dir, "center_of_rotation.txt")
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:
                pass

        recon_way = self.recon_way_box.currentText()
        gpu = str(self.cuda_box_try.value())
        if self.use_conf_box.isChecked():
            self.log_output.append("You are using config file, only recon type, filename, rot axis from GUI")
            config_text = self.config_editor_try.toPlainText()
            if not config_text.strip():
                self.log_output.append('<span style="color:red;">no text in conf, stop</span>')
                return
            temp_try = os.path.join(self.data_path.text(), "temp_try.conf")
            with open(temp_try, "w") as f:
                f.write(config_text)
            cmd = ["tomocupy", str(recon_way),
                  "--reconstruction-type", "try",
                  "--config", temp_try,
                  "--file-name", proj_file,
                  "--rotation-axis-auto", "auto"]
            if seed:
                cmd += ["--rotation-axis", seed]
        else:
            cmd = ["tomocupy", str(recon_way),
                  "--reconstruction-type", "try",
                  "--file-name", proj_file,
                  "--rotation-axis-auto", "auto"]
            if seed:
                cmd += ["--rotation-axis", seed]
            cmd += self._gather_params_args()
            cmd += self._gather_rings_args()
            cmd += self._gather_bhard_args()
            cmd += self._gather_phase_args()
            cmd += self._gather_Geometry_args()
            cmd += self._gather_Data_args()
            cmd += self._gather_Performance_args()
            temp_try = None

        # Apply AI-COR flags LAST so they win the argparse last-wins race
        # against any --rotation-axis-method the Reconstruction tab emitted.
        cmd = self._apply_ai_cor(cmd)

        self.log_output.append('Try + AI COR search (single tomocupy call)…')
        QApplication.processEvents()
        try:
            code = self.run_command_live(
                cmd, proj_file=proj_file, job_label="Try+AI recon",
                wait=True, cuda_devices=gpu,
            )
        finally:
            if temp_try and os.path.exists(temp_try):
                try:
                    os.remove(temp_try)
                except OSError:
                    pass

        if code != 0:
            self.log_output.append(
              f'<span style="color:red;">Try+AI failed for {os.path.basename(proj_file)}</span>'
            )
            return

        ai_cor = self._read_ai_cor_from_try_dir(proj_file)
        if ai_cor is None:
            self.log_output.append(
              f'<span style="color:orange;">tomocupy did not produce '
                f'center_of_rotation.txt in {try_dir}</span>'
            )
            return

        ai_cor_str = f"{ai_cor:.2f}"
        self.cor_data[proj_file] = ai_cor_str
        if data_folder:
            self._save_cor_data(data_folder, self.cor_data)
        if self.highlight_row is not None:
            cor_widget = self.batch_file_main_table.cellWidget(self.highlight_row, 2)
            if cor_widget:
                cor_widget.setText(ai_cor_str)
        self.log_output.append(
          f'<span style="color:green;">AI COR: {ai_cor_str} — saved for '
            f'{os.path.basename(proj_file)}</span>'
        )

        self.log_output.append('Starting full reconstruction with AI COR…')
        QApplication.processEvents()
        return self.full_reconstruction()

    # ------------------------------------------------------------------ #
    #  Sync Acquisition                                                    #
    # ------------------------------------------------------------------ #

    def _toggle_sync(self, checked):
        if checked:
            self._start_sync()
        else:
            self._stop_sync()

    def _start_sync(self):
        data_folder = self.data_path.text().strip()
        if not data_folder or not os.path.isdir(data_folder):
          self.log_output.append('<span style="color:red;">Set a valid data folder before starting Sync</span>')
          self.sync_btn.setChecked(False)
          return
        known = set(glob.glob(os.path.join(data_folder, "*.h5")))
        self._sync_queue = []
        self._sync_processing = False
        self._sync_current_file = None
        
        self._sync_watcher = SyncWatcher(data_folder, known)
        self._sync_watcher.new_file_ready.connect(self._on_new_sync_file)
        self._sync_watcher.file_progress.connect(
            lambda f, n, t: self.log_output.append(
                f'⏳ {os.path.basename(f)}: {n}/{t} projections written'
            )
        )
        self._sync_watcher.start()
        
        self.sync_btn.setText("⏹  Stop Sync")
        self.batch_file_main_table.setEnabled(False)
        self.log_output.append(f'<span style="color:green;">🔄 Sync Acquisition started — watching {data_folder}</span>')

    def closeEvent(self, event):
        """Ensure background threads stop cleanly before the window closes."""
        try:
            if self._sync_watcher:
                self._sync_watcher.stop()
                if not self._sync_watcher.wait(5000):
                    self._sync_watcher.terminate()
                    self._sync_watcher.wait(1000)
                self._sync_watcher = None
        except Exception:
            pass
        super().closeEvent(event)

    def _stop_sync(self):
        if self._sync_watcher:
            self._sync_watcher.stop()
            # give the thread up to 15 s to exit its poll cycle
            if not self._sync_watcher.wait(15000):
              self.log_output.append('<span style="color:orange;">Sync thread did not exit, terminating</span>')
              self._sync_watcher.terminate()
              self._sync_watcher.wait(2000)
            self._sync_watcher = None
        
        self._sync_queue = []
        self._sync_processing = False
        self._sync_current_file = None
        
        self.sync_btn.setText(" Sync Acquisition")
        self.sync_btn.setChecked(False)
        self.batch_file_main_table.setEnabled(True)
        self.log_output.append('🔄 Sync Acquisition stopped.')

    def _on_new_sync_file(self, filepath):
        """Called on the main thread when a new stable HDF5 file is detected."""
        self.log_output.append(f'🆕 New file detected: <b>{os.path.basename(filepath)}</b>')
        QApplication.processEvents()

        # queue, not process data
        if filepath != self._sync_current_file and filepath not in self._sync_queue:
            self._sync_queue.append(filepath)
            self.log_output.append(f'📥 Added to sync queue: {os.path.basename(filepath)} '
            f'(queue={len(self._sync_queue)})')
        if not self._sync_processing:
            self._process_next_sync_file()

        # Select the row so highlight_scan is set correctly
        #for row in range(self.batch_file_main_table.rowCount()):
        #    item = self.batch_file_main_table.item(row, 0)
        #    if item and item.toolTip() == filepath:
        #        self.batch_file_main_table.selectRow(row)
        #        self.on_table_row_clicked(row, 0)
        #        break

        #QApplication.processEvents()

        # Run AI Reco (try + inference + full)
        #self.try_ai_reconstruction()

        # Run tomolog upload
        #self._run_tomolog_for_file(filepath)
        #return 

    def _add_file_to_table(self, filepath):
        """Insert a single file row into the table if it is not already there."""
        # Check if already present
        for row in range(self.batch_file_main_table.rowCount()):
            item = self.batch_file_main_table.item(row, 1)
            if item.text() == os.path.basename(filepath):
                self.log_output.append('file exists in table, leave')
                return
        data_folder = self.data_path.text().strip()
        filename = os.path.basename(filepath)
        proj_name = os.path.splitext(filename)[0]
        try_dir = os.path.join(f"{data_folder}_rec", "try_center", proj_name)
        has_try = os.path.isdir(try_dir) and len(glob.glob(os.path.join(try_dir, "*.tiff"))) > 0
        full_info = self._resolve_full_recon(data_folder, proj_name)
        has_full = full_info['kind'] is not None

        row = 0  # insert at top (newest first)
        self.batch_file_main_table.insertRow(row)

        if has_full:
            row_color, status_text = "green", "Full"
            if full_info['range'] is not None:
                num_1, num_2 = full_info['range']
                status_item = QTableWidgetItem(f"{status_text} {num_1}-{num_2}")
            else:
              status_item = QTableWidgetItem(f"{status_text} ({full_info['kind']})")
        elif has_try:
            row_color, status_text = "orange", "Done try"
            status_item = QTableWidgetItem(status_text)
        else:
            row_color, status_text = "red", "Ready"
            status_item = QTableWidgetItem(status_text)
        
        # checkbox
        checkbox = QCheckBox()
        checkbox.clicked.connect(lambda checked, r=row: self._batch_checkbox_clicked(r, checked))
        checkbox_widget = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.setAlignment(Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_widget.setStyleSheet(f"QWidget {{ border-left: 6px solid {row_color}; }}")
        self.batch_file_main_table.setCellWidget(row, 0, checkbox_widget)
        
        # filename
        filename_item = QTableWidgetItem(filename)
        filename_item.setToolTip(f"{filename}\n\nFull path:\n{filepath}")
        self.batch_file_main_table.setItem(row, 1, filename_item)
        
        #cor default
        cor_input = QLineEdit("")
        cor_input.setPlaceholderText("COR value")
        cor_input.setAlignment(Qt.AlignCenter)
        cor_input.setFixedWidth(80)
        self.batch_file_main_table.setCellWidget(row, 2, cor_input)
        
        #status
        self.batch_file_main_table.setItem(row, 3, status_item)
        
        #file size
        file_size = os.path.getsize(filepath)
        size_str = self._format_file_size(file_size)
        size_item = QTableWidgetItem(size_str)
        size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        size_item.setData(Qt.UserRole, file_size)
        self.batch_file_main_table.setItem(row, 4, size_item)
        
        #placehold
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(2, 2, 2, 2)
        actions_layout.setSpacing(2)
        self.batch_file_main_table.setCellWidget(row, 5, actions_widget)
        
        #view data
        view_data_btn = QPushButton("View Data")
        view_data_btn.setFixedWidth(80)
        view_data_btn.clicked.connect(lambda checked, fp=filepath: self._batch_view_data(fp))
        self.batch_file_main_table.setCellWidget(row, 6, view_data_btn)
        
        file_info = {
        'path': filepath,
        'filename': filename,
        'recon_status': row_color,
        'checkbox': checkbox,
        'checkbox_widget': checkbox_widget,
        'cor_input': cor_input,
        'status': status_item,
        'view_btn': view_data_btn,
        }

        self.batch_file_main_list.insert(row, file_info)
        self.log_output.append(f'this is file info {file_info}')
        '''
        from PyQt5.QtWidgets import QTableWidgetItem as _TWI
        name_item = _TWI(filename)
        name_item.setToolTip(filepath)
        name_item.setForeground(QColor(row_color))
        self.batch_file_main_table.setItem(row, 1, name_item)

        status_item = _TWI(status_text)
        status_item.setForeground(QColor(row_color))
        self.batch_file_main_table.setItem(row, 3, status_item)

        cor_val = self.cor_data.get(filepath, "")
        cor_widget = QLineEdit(str(cor_val))
        cor_widget.setAlignment(Qt.AlignCenter)
        self.batch_file_main_table.setCellWidget(row, 2, cor_widget)
        
        file_size = os.path.getsize(f)
        size_str = self._format_file_size(file_size)
        size_item = QTableWidgetItem(size_str)
        size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # Store numeric value for proper sorting
        size_item.setData(Qt.UserRole, file_size)
        self.batch_file_main_table.setItem(row, 4, size_item)
        '''
        #self.batch_file_main_list.insert(0, {'file': filepath, 'cor_input': cor_widget})
        
    def _process_next_sync_file(self):
        if not self.sync_btn.isChecked():
            return
        if self._sync_processing:
            return
        if not self._sync_queue:
            self._sync_current_file = None
            return
        filepath = self._sync_queue.pop(0)
        self._sync_current_file = filepath
        self._sync_processing = True
        self._add_file_to_table(filepath) #always the first one 
        self.log_output.append(f'Start sync processing: <b>{os.path.basename(filepath)}</b>')
        # Select the row so highlight_scan is set correctly
        for row in reversed(range(self.batch_file_main_table.rowCount())):
            item = self.batch_file_main_table.item(row, 1)
            if item.text() == os.path.basename(filepath):
                self.batch_file_main_table.selectRow(row)
                self.on_table_row_clicked(row, 0)
                break
                
        QApplication.processEvents()
        ok = self.try_ai_reconstruction()
        
        if ok:
            self._run_tomolog_for_file(filepath)
        else:
            self.log_output.append(
              f'<span style="color:red;">Sync reconstruction failed, skipping tomolog: {os.path.basename(filepath)}</span>'
                )

        self._sync_processing = False
        self._sync_current_file = None

        #continue with next file
        self._process_next_sync_file()
        
    def _run_tomolog_for_file(self, filepath):
        """Run tomolog upload for a specific file using current GUI settings.

        If the min/max fields are blank, a per-dataset 5-95 % percentile
        contrast is computed from the reconstruction — that lets mass /
        sync uploads adapt contrast automatically. Any value typed by
        the user overrides the auto-contrast for that bound."""
        beamline = self.beamline_box.currentText()
        cloud = self.cloud_box.currentText()
        url = self.url_input.text().strip()
        x = self.x_input.text().strip()
        y = self.y_input.text().strip()
        z = self.z_input.text().strip()
        vmin = self.min_input.text().strip()
        vmax = self.max_input.text().strip()
        note_value = self.get_note_value()

        if not vmin or not vmax:
            avmin, avmax = self._auto_contrast_for_file(filepath)
            if not vmin and avmin is not None:
                vmin = avmin
            if not vmax and avmax is not None:
                vmax = avmax
            if avmin is not None and avmax is not None:
                self.log_output.append(
                  f'<span style="color:#888;"> auto contrast '
                    f'{os.path.basename(filepath)}: min={vmin}, max={vmax}</span>'
                )
            else:
                self.log_output.append(
                  f'<span style="color:orange;">no reconstruction TIFFs for '
                    f'{os.path.basename(filepath)} — tomolog will use its own default contrast.</span>'
                )

        cmd = [
            "tomolog", "run",
            "--beamline", beamline,
            "--file-name", filepath,
            "--cloud", cloud,
            "--presentation-url", url,
            "--idx", x,
            "--idy", y,
            "--idz", z,
            "--note", note_value,
        ]
        if vmin:
            cmd.extend(["--min", vmin])
        if vmax:
            cmd.extend(["--max", vmax])
        extra_params = self.extra_params_input.text().strip()
        if extra_params:
            cmd.extend(extra_params.split())

        self.log_output.append(f'Uploading to tomolog: {os.path.basename(filepath)}')
        QApplication.processEvents()
        code = self.run_command_live(cmd, proj_file=filepath, job_label="tomolog-sync", wait=True, cuda_devices=None)
        if code == 0:
          self.log_output.append(f'<span style="color:green;">Tomolog upload done: {os.path.basename(filepath)}</span>')
        else:
          self.log_output.append(f'<span style="color:red;">Tomolog upload failed: {os.path.basename(filepath)}</span>')

    def _update_full_btn_state(self):
        """Grey out Full button only while the currently selected file is running locally."""
        if self._running_full_file and self.highlight_scan == self._running_full_file:
            self.full_btn.setEnabled(False)
        else:
            self.full_btn.setEnabled(True)

    def full_reconstruction(self):
        # STEP 1 (must be first, before anything else): blank the viewer
        # and flush the paint so the on-screen image is *visibly* gone
        # before we touch the H5 or launch tomocupy. Otherwise the display
        # keeps an NFS handle alive on the file we're about to overwrite.
        proj_file = self.highlight_scan
        self._blank_viewer()
        QApplication.processEvents()
        self._release_full_h5_for_write(proj_file)
        QApplication.processEvents()

        self._update_full_btn_state()
        if proj_file:
            self._persist_params_for_files([proj_file])
        # Mark this file as running locally and grey out button only for it
        self._running_full_file = proj_file
        self._update_full_btn_state()
        try:
            pn = os.path.splitext(os.path.basename(proj_file))[0]
            recon_way = self.recon_way_box_full.currentText()
            highlight_row = self.highlight_row
            cor_method = self.cor_full_method.currentText()
            gpu = str(self.cuda_full_box.value())
            if cor_method == "manual":
                try:
                    cor_value = float(self.batch_file_main_list[self.highlight_row]['cor_input'].text().strip())
                except ValueError:
                  self.log_output.append('<span style="color:red;">[ERROR] Invalid Full COR value</span>')
                  return
            if self.use_conf_box.isChecked():
                self.log_output.append("You are using config file, only recon type, filename, rot axis from GUI")
                config_text = self.config_editor_full.toPlainText()
                if not config_text.strip():
                  self.log_output.append('<span style="color:red;">No text in conf, stop</span>')
                  return
                temp_full = os.path.join(self.data_path.text(), "temp_full.conf")
                with open(temp_full, "w") as f:
                    f.write(config_text)
                cmd = ["tomocupy", str(recon_way),
                      "--reconstruction-type", "full",
                      "--config", temp_full,
                      "--file-name", proj_file,
                      "--rotation-axis", str(cor_value)]
            else:
                self.log_output.append('You are using params from GUI')
                if cor_method == "auto":
                    cmd = ["tomocupy", str(recon_way),
                          "--reconstruction-type", "full",
                          "--file-name", proj_file,
                          "--rotation-axis-auto", cor_method]
                elif cor_method == "manual":
                    cmd = ["tomocupy", str(recon_way),
                          "--reconstruction-type", "full",
                          "--file-name", proj_file,
                          "--rotation-axis-auto", cor_method,
                          "--rotation-axis", str(cor_value)]
                cmd += self._gather_params_args()
                cmd += self._gather_rings_args()
                cmd += self._gather_bhard_args()
                cmd += self._gather_phase_args()
                cmd += self._gather_Geometry_args()
                cmd += self._gather_Data_args()
                cmd += self._gather_Performance_args()

            code = self.run_command_live(cmd, proj_file=proj_file, job_label="Full recon", wait=True, cuda_devices=gpu)
            try:
                if code == 0:
                    info = self._resolve_full_recon(self.data_path.text(), pn)
                    if info['kind'] and info['range'] is not None:
                        n1, n2 = info['range']
                        status = f"Full {n1}-{n2}"
                    elif info['kind']:
                      status = f"Full ({info['kind']})"
                    else:
                        status = "Done full"
                    self._update_row(row=self.highlight_row, color='green', status=status)
                    self.log_output.append(f'<span style="color:green;">Done full recon {proj_file}</span>')
                    return True
                else:
                  self.log_output.append(f'<span style="color:red;">Full recon {proj_file} failed</span>')
                  return False
            finally:
                if self.use_conf_box.isChecked():
                    try:
                        if os.path.exists(temp_full):
                            os.remove(temp_full)
                            self.log_output.append(f"\U0001f9f9 Removed {temp_full}")
                    except Exception as e:
                      self.log_output.append(f'<span style="color:red;">Could not remove {temp_full}: {e}</span>')
        finally:
            if self.use_conf_box.isChecked():
                try:
                    if os.path.exists(temp_full):
                        os.remove(temp_full)
                        self.log_output.append(f"\U0001f9f9 Removed {temp_full}")
                except Exception as e:
                  self.log_output.append(f'<span style="color:red;">Could not remove {temp_full}: {e}</span>')
            self._running_full_file = None
            self._update_full_btn_state()
		    
    #=============Batch OPERATIONS==================
    def _batch_select_all(self):
        """Select all files in the batch list, except files flagged as
        auto-skipped (small size — typically aborted scans)."""
        skipped_small = 0
        for file_info in self.batch_file_main_list:
            if file_info.get('skipped_small'):
                skipped_small += 1
                continue
            file_info['checkbox'].setChecked(True)
            msg = '<span style="color:green;">Select all files in table</span>'
        if skipped_small:
          msg += (f' <span style="color:#888;">'
                    f'({skipped_small} small-file row(s) kept unchecked)</span>')
        self.log_output.append(msg)

    def _batch_deselect_all(self):
        """Deselect all files in the batch list"""
        for file_info in self.batch_file_main_list:
            file_info['checkbox'].setChecked(False)
            self.log_output.append(f'<span style="color:green;">Unselect all files in table</span>')

    def _get_batch_machine_command(self, cmd, machine):
        """
        Wrap command for remote execution via SSH if needed

        Args:
            cmd: List of command arguments (e.g., ["tomocupy", "recon", ...])
            machine: Machine name ("Local", "tomo1", etc.)

        Returns:
            List of command arguments, potentially wrapped in SSH
        """
        if machine == "Local":
            return cmd

        # Get machine configuration
        machine_config = self.machine_config.get(machine, {})
        username = machine_config.get("username", os.getenv("USER", ""))
        hostname = machine_config.get("hostname", machine)
        conda_env = machine_config.get("conda_env", "tomocupy")

        # Build SSH target
        if username:
            ssh_target = f"{username}@{hostname}"
        else:
            ssh_target = hostname

        # Build command with conda activation
        # Properly quote arguments for shell execution
            remote_cmd = " ".join([f'"{str(arg)}"' if " " in str(arg) else str(arg) for arg in cmd])

        # Wrap command with conda activation
            full_cmd = f"bash -l -c 'source ~/.bashrc && conda activate {conda_env} && {remote_cmd}'"

        # Use SSH with terminal (-t) to execute the command on the remote machine
        # -t forces pseudo-terminal allocation for better output handling
        ssh_cmd = ["ssh", "-t", ssh_target, full_cmd]

        self.log_output.append(f'<span style="color:gray;">SSH: {ssh_target} (env: {conda_env})</span>')

        return ssh_cmd

    # ===== COR MANAGEMENT =====
    def record_cor_main_tb(self):
        '''
        link to Add COR function: get cor from current viewing image, save to json file and update main table
        '''
        data_folder = self.data_path.text().strip()
        proj_file = self.highlight_scan
        row = self.highlight_row
        idx = self.slice_slider.value()
        cor_file = self.preview_files[idx]
        if not os.path.exists(proj_file) or not os.path.exists(cor_file):
          self.log_output.append(f'<span style="color:red;">Missing try data folder or projection file</span>')
          return
        cor_nm = os.path.basename(cor_file)
        try:
            cor_value = cor_nm.split("center")[1].split(".tiff")[0]
        except IndexError:
          self.log_output('<span style="color:red;">[ERROR] Value not found in expected format, cannot add COR</span>')
          return
        json_path = os.path.join(data_folder, "rot_cen.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                try:
                    self.cor_data = json.load(f)
                    fns = list(self.cor_data.keys())
                except json.JSONDecodeError:
                    self.cor_data = {}
        else:
            self.cor_data = {}
            fns = []
        
        if proj_file in fns:
            overfn_msg_box = QMessageBox(self)
            overfn_msg_box.setIcon(QMessageBox.Warning)
            overfn_msg_box.setWindowTitle("Overwrite Existing files in log?")
            overfn_msg_box.setText(f"The scan:\n{os.path.basename(proj_file)}\nalready exists in rot_cen.json.\n\nDo you want to overwrite it?")
            overfn_msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            overfn_msg_box.setDefaultButton(QMessageBox.No)
            result = overfn_msg_box.exec()
            if result != QMessageBox.Yes:
                self.log_output.append("Not take COR")
                return
        self.cor_data[proj_file] = cor_value

        # Save to file(s) using the helper method
        self._save_cor_data(data_folder, self.cor_data)

        # Update the table widget
        w = self.batch_file_main_table.cellWidget(row, 2)
        if w is None:
            w = QLineEdit()
            w.setAlignment(Qt.AlignCenter)
            w.setFixedWidth(80)
            self.batch_file_main_table.setCellWidget(row, 2, w)
        w.setText(str(cor_value))
        # keep list storing the widget
        self.batch_file_main_list[row]['cor_input'] = w
        self.log_output.append(f"[INFO] COR saved for: {os.path.basename(proj_file)}")

    # ===== IMAGE VIEWING =====
    def _preflight_view(self, kind):
        """Common validation for View Try / View Full. Returns
        ``(data_folder, proj_file, proj_name)`` on success, or ``None`` on
        failure \u2014 in which case the viewer is blanked and a red message is
        printed to the log panel. No modal dialogs."""
        data_folder = self.data_path.text().strip()
        proj_file = self.highlight_scan
        if not data_folder or not os.path.isdir(data_folder):
            self._blank_viewer()
            self.log_output.append(
              f'<span style="color:red;">View {kind}: no data folder '
                f'set \u2014 pick a folder in the Main tab first.</span>'
            )
            return None
        if not proj_file:
            self._blank_viewer()
            self.log_output.append(
              f'<span style="color:red;">View {kind}: no file selected '
                f'\u2014 click a row in the table first.</span>'
            )
            return None
        if not os.path.isfile(proj_file):
            self._blank_viewer()
            self.log_output.append(
              f'<span style="color:red;">View {kind}: source file no '
                f'longer exists: {proj_file}</span>'
            )
            return None
        proj_name = os.path.splitext(os.path.basename(proj_file))[0]
        return data_folder, proj_file, proj_name

    def view_try_reconstruction(self):
        pf = self._preflight_view("Try")
        if pf is None:
            return
        data_folder, proj_file, proj_name = pf

        try_dir = os.path.join(f"{data_folder}_rec", "try_center", proj_name)
        preview_files = sorted(glob.glob(os.path.join(try_dir, "*.tiff")))
        if not preview_files:
            self._blank_viewer()
            base = os.path.basename(proj_file)
            if not os.path.isdir(try_dir):
                self.log_output.append(
                  f'<span style="color:red;">View Try: no try_center '
                    f'dir for {base} \u2014 expected {try_dir}. Run a Try '
                    f'reconstruction on this file first.</span>'
                )
            else:
                self.log_output.append(
                  f'<span style="color:red;">View Try: {try_dir} '
                    f'exists but has no .tiff slices for {base}.</span>'
                )
            return

        self.preview_files = preview_files
        self.log_output.append(
            f"Try recon: {os.path.basename(proj_file)} \u2014 {len(preview_files)} "
            f"slice(s), first {os.path.basename(preview_files[0])}, "
            f"last {os.path.basename(preview_files[-1])}"
        )
        self._clear_roi()
        self._reset_view_state()
        try:
            self.slice_slider.valueChanged.disconnect()
        except TypeError:
            pass
        self.slice_slider.setMaximum(len(self.preview_files) - 1)
        self.slice_slider.valueChanged.connect(self.update_try_slice)
        self._try_proj_name = proj_name
        self.update_try_slice()

    def view_full_reconstruction(self):
        pf = self._preflight_view("Full")
        if pf is None:
            return
        data_folder, proj_file, proj_name = pf

        info = self._resolve_full_recon(data_folder, proj_name)

        # Close any previously-open H5 handle before switching source.
        self._close_full_h5()

        if info['kind'] == 'h5':
            try:
                # locking=False avoids grabbing an OS lock the writer
                # (tomocupy) would otherwise get EAGAIN on when the user
                # relaunches Full recon on the same file over NFS.
                try:
                    self.full_h5 = h5py.File(info['h5_path'], 'r',
                                                locking=False)
                except (TypeError, ValueError):
                    self.full_h5 = h5py.File(info['h5_path'], 'r')
            except OSError as e:
                self._blank_viewer()
                self.log_output.append(
                  f'<span style="color:red;">View Full: could not open '
                  f'{info["h5_path"]}: {e}</span>'
                )
                return
            self.full_h5_path = info['h5_path']
            n = info['n_slices']
            if n <= 0:
                self._close_full_h5()
                self._blank_viewer()
                self.log_output.append(
                  f'<span style="color:red;">View Full: H5 exists but '
                  f'/exchange/data is empty: {info["h5_path"]}</span>'
                )
                return
            self.full_files = list(range(n))
            self.log_output.append(
              f'\ud83d\udce6 Full recon: {os.path.basename(info["h5_path"])} '
                f'({n} slices)'
            )
        elif info['kind'] == 'tiff':
            self.full_files = info['tiff_files']
            self.log_output.append(
                f"Full recon: {len(self.full_files)} tiff(s), first "
                f"{os.path.basename(self.full_files[0])}, last "
                f"{os.path.basename(self.full_files[-1])}"
            )
        else:
            self._blank_viewer()
            base = os.path.join(f"{data_folder}_rec", f"{proj_name}_rec")
            self.log_output.append(
              f'<span style="color:red;">View Full: no reconstruction '
                f'for {os.path.basename(proj_file)} \u2014 checked {base}.h5 and '
                f'{base}/*.tiff. Run a Full reconstruction first.</span>'
            )
            return

        self._clear_roi()
        self._reset_view_state()
        try:
            self.slice_slider.valueChanged.disconnect()
        except TypeError:
            pass
        self.slice_slider.setMaximum(max(0, len(self.full_files) - 1))
        self.slice_slider.valueChanged.connect(self.update_full_slice)
        self._current_source_file = os.path.basename(proj_file)
        self._current_view_mode = "full"
        self.update_full_slice()

    def _close_full_h5(self):
        """Close any open full-recon H5 handle. Safe to call repeatedly."""
        if self.full_h5 is not None:
            try:
                self.full_h5.close()
            except OSError:
                pass
        self.full_h5 = None
        self.full_h5_path = None

    def _blank_viewer(self):
        """Clear the on-screen image and drop every reference to the file
        it was loaded from. Called before a Full recon so the display can't
        keep an NFS handle alive on the file tomocupy is about to overwrite.
        """
        # Drop file-list state so slider handlers can't try to reopen the
        # old paths after we've torn everything down.
        self.full_files = []
        self.preview_files = []
        self._current_img = None
        self._current_img_path = None
        try:
            self.slice_slider.valueChanged.disconnect()
        except TypeError:
            pass
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        try:
            self.filename_label.setText("")
        except (AttributeError, RuntimeError):
            pass
        self._clear_roi()
        self._reset_view_state()

        # Replace the pixels with an empty 1×1 array so the widget shows
        # blank instead of the last frame from the file we're about to
        # overwrite. Guarded because either renderer may be unavailable.
        blank = np.zeros((1, 1), dtype=np.float32)
        if VISPY_AVAILABLE and hasattr(self, 'image_visual'):
            try:
                self.image_visual.set_data(blank)
                self.canvas.update()
            except (ValueError, RuntimeError, AttributeError):
                pass
        elif hasattr(self, '_pg_image_item'):
            try:
                self._pg_image_item.setImage(blank, autoLevels=False)
                self._pg_image_item.update()
            except (ValueError, TypeError, RuntimeError, AttributeError):
                pass

    def _middle_of_width(self, proj_file):
        """Return image_width / 2 read from ``proj_file`` (/exchange/data
        shape is (nproj, nz, nx)). Used as the AI-COR fallback seed when
        neither the row nor the top-bar carries a valid number. Returns
      ``None`` if the file can't be opened or the dataset is missing."""
        if not proj_file or not os.path.isfile(proj_file):
            return None
        try:
            with h5py.File(proj_file, 'r') as fh:
                shape = fh['/exchange/data'].shape
        except (OSError, KeyError):
            return None
        if len(shape) < 3:
            return None
        return float(shape[2]) / 2.0

    def _release_full_h5_for_write(self, proj_file):
        """Blank the viewer and drop any H5 handle it holds so tomocupy can
        overwrite the file without colliding with the GUI's reader.

        Tomocupy's own ``--clear-folder`` step handles removing/truncating
        the output on disk — we do NOT touch files here (an earlier version
        deleted them and destroyed valid reconstructions when the new run
        crashed)."""
        was_holding = self.full_h5 is not None or bool(self.full_files) or bool(self.preview_files)
        self._blank_viewer()
        if was_holding:
            self.log_output.append(
                '<span style="color:#888;">Cleared viewer before '
                'Full recon.</span>'
            )

    def set_image_scale(self, img_path, flag=None):
        if flag == "raw":
            img = img_path
        else:
            img = np.array(Image.open(img_path))
        self.vmin, self.vmax = round(np.nanmin(img), 5), round(np.nanmax(img), 5)
        self.min_input.setText(str(self.vmin))
        self.max_input.setText(str(self.vmax))

    # ===== ROI AND CONTRAST =====
    def draw_box(self):
        """Enable interactive ROI drawing."""
        if self._current_img is None:
            self.log_output.append("No image loaded to draw box.")
            return

        if not VISPY_AVAILABLE:
            # PyQtGraph: place a resizable RectROI on the image
            h, w = self._current_img.shape[:2]
            if self._pg_roi_item is not None:
                self._pg_view_box.removeItem(self._pg_roi_item)
            self._pg_roi_item = pg.RectROI([w // 4, h // 4], [w // 2, h // 2],
                                          pen=pg.mkPen('r', width=2))
            self._pg_roi_item.sigRegionChanged.connect(self._pg_roi_changed)
            self._pg_view_box.addItem(self._pg_roi_item)
            self._pg_roi_changed(self._pg_roi_item)
            self.log_output.append("Drag the red ROI handles to resize/move it.")
            return

        self._drawing_roi = True
        self.roi_extent = None
        self._roi_start = None
        self.view.camera.interactive = False
        self.log_output.append("Click and drag to draw ROI. Release to set. Click again to clear.")

    def _on_vispy_mouse_click(self, event):
        """Handle mouse press for ROI drawing"""
        if not self._drawing_roi:
            if self.roi_extent is not None:
                # Click clears existing ROI
                self._clear_roi()
                self.log_output.append('<span style="color:green;">ROI cleared</span>')
            return

        tr = self.image_visual.get_transform(map_from='canvas', map_to='visual')
        pos = tr.map(event.pos)[:2]
        self._roi_start = pos

    def _on_vispy_mouse_release(self, event):
        """Handle mouse release for ROI drawing"""
        if not self._drawing_roi or self._roi_start is None:
            return

        tr = self.image_visual.get_transform(map_from='canvas', map_to='visual')
        pos = tr.map(event.pos)[:2]

        x0, y0 = self._roi_start
        x1, y1 = pos
        self.roi_extent = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
        self._drawing_roi = False

        # Draw ROI rectangle with vispy
        self._draw_roi_visual()
        self.view.camera.interactive = True

        self.log_output.append(
            f"ROI set: x[{int(self.roi_extent[0])}:{int(self.roi_extent[1])}], "
            f"y[{int(self.roi_extent[2])}:{int(self.roi_extent[3])}]"
        )

    # ---- PyQtGraph-specific helpers ----

    def _pg_mouse_moved(self, event):
        """Show pixel coordinates under the cursor (pyqtgraph path)."""
        pos = event[0]  # SignalProxy wraps pos in a tuple
        if not self._pg_image_item.sceneBoundingRect().contains(pos):
            if hasattr(self, 'coord_label'):
                self.coord_label.setText("")
            return
        mouse_pt = self._pg_view_box.mapSceneToView(pos)
        x, y = int(mouse_pt.x()), int(mouse_pt.y())
        if self._current_img is not None:
            h, w = self._current_img.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                val = self._current_img[y, x]
                if hasattr(self, 'coord_label'):
                    self.coord_label.setText(f"({x},{y}): {float(val):.5f}")
                return
        if hasattr(self, 'coord_label'):
            self.coord_label.setText("")

    def _pg_roi_changed(self, roi):
        """Update roi_extent when the pyqtgraph RectROI is moved/resized."""
        pos = roi.pos()
        size = roi.size()
        x0, y0 = pos.x(), pos.y()
        x1, y1 = x0 + size.x(), y0 + size.y()
        self.roi_extent = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    def _pg_apply_levels(self, vmin, vmax):
        """Update display levels on the pyqtgraph ImageItem without reloading the image."""
        self._pg_image_item.setLevels([vmin, vmax])
        # Sync the histogram widget so its level lines match
        if hasattr(self, '_pg_hist'):
            self._pg_hist.item.setLevels(vmin, vmax)
        self.min_input.setText(str(round(vmin, 5)))
        self.max_input.setText(str(round(vmax, 5)))

    def _pg_hist_levels_changed(self):
        """Sync vmin/vmax when the user drags the histogram level lines."""
        vmin, vmax = self._pg_hist.item.getLevels()
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.min_input.setText(str(round(self.vmin, 5)))
        self.max_input.setText(str(round(self.vmax, 5)))

    # ---- VisPy-only helpers ----

    def _draw_roi_visual(self):
        """Draw ROI rectangle using vispy Line visual (VisPy path only)."""
        if not VISPY_AVAILABLE or self.roi_extent is None:
            return

        x0, x1, y0, y1 = self.roi_extent
        w, h = (x1 - x0), (y1 - y0)
        cx, cy = (x0 + w/2, y0 + h/2)
        if self._roi_visual is None:
            self._roi_visual = scene.visuals.Rectangle(
                                center=(cx,cy),
                                width=w,
                                height=h,
                                color=(0, 0, 0, 0),     # transparent fill
                                border_color='red',
                                border_width=2,
                                parent=self.image_visual
                            )
            self._roi_visual.set_gl_state(depth_test=False, blend=True) #force it on top of image
        else:
            if self._roi_visual.parent is None:
                self._roi_visual.parent = self.image_visual
            self._roi_visual.center = (cx,cy)
            self._roi_visual.width = w
            self._roi_visual.height = h
        self._roi_visual.visible = True
        self.canvas.update()

    def _clear_roi(self):
        """Hide/remove any active ROI."""
        if not VISPY_AVAILABLE:
            if self._pg_roi_item is not None:
                self._pg_view_box.removeItem(self._pg_roi_item)
                self._pg_roi_item = None
            self.roi_extent = None
            self._drawing_roi = False
            return
        if self._roi_visual is not None:
            self._roi_visual.parent = None
            self._roi_visual = None
        self.roi_extent = None
        self._drawing_roi = False
        self.canvas.update()

    def _on_vispy_mouse_move(self, event):
        """Show coordinates under the mouse in the coord label."""
        if self._current_img is None:
            if hasattr(self, "coord_label"):
                self.coord_label.setText("")
            return

        tr = self.image_visual.get_transform(map_from='canvas', map_to='visual')
        pos = tr.map(event.pos)[:2]
        x, y = int(pos[0]), int(pos[1])

        h, w = self._current_img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            val = self._current_img[y, x]
            msg = f"({x},{y}): {float(val):.5f}"
        else:
            msg = ""
        if hasattr(self, "coord_label"):
            self.coord_label.setText(msg)

    def auto_img_contrast(self, saturation=10):
        """Fiji-like Auto: trims tails within current window; uses ROI if present; never edits pixels."""
        if self._current_img is None:
            self.log_output.append("No image loaded to auto contrast.")
            return

        img = self._current_img

        if self.roi_extent is not None:
            x0, x1, y0, y1 = self.roi_extent
            h, w = img.shape[:2]
            x0 = max(0, min(w, int(np.floor(x0))))
            x1 = max(0, min(w, int(np.ceil(x1))))
            y0 = max(0, min(h, int(np.floor(y0))))
            y1 = max(0, min(h, int(np.ceil(y1))))
            if x1 <= x0 or y1 <= y0:
                self.log_output.append("ROI too small.")
                return
            data = img[y0:y1, x0:x1]
        else:
            data = img

        a = np.asarray(data, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            self.log_output.append("No finite pixels for Auto.")
            return

        vmin = self.vmin if self.vmin is not None else float(np.nanmin(a))
        vmax = self.vmax if self.vmax is not None else float(np.nanmax(a))
        vis = a[(a >= vmin) & (a <= vmax)]
        if vis.size < 64:
            vis = a

        lo, hi = np.nanpercentile(vis, [1.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.nanmin(vis)), float(np.nanmax(vis))
            if lo >= hi:
                hi = lo + 1.0

        new_vmin, new_vmax = float(round(lo, 5)), float(round(hi, 5))
        if (new_vmin, new_vmax) == (self.vmin, self.vmax):
            self.log_output.append("Auto B&C optimal")
            return

        self.vmin, self.vmax = new_vmin, new_vmax

        self.min_input.setText(str(self.vmin))
        self.max_input.setText(str(self.vmax))

        if VISPY_AVAILABLE and self._current_img is not None:
            self.image_visual.clim = (self.vmin, self.vmax)
            self.canvas.update()
        elif not VISPY_AVAILABLE and self._current_img is not None:
            self._pg_apply_levels(self.vmin, self.vmax)

    def reset_img_contrast(self): #link to Reset button
        if self._current_img is None:
            self.log_output.append("No image loaded to reset contrast.")
            return
        else:
            self.vmin, self.vmax = round(self._current_img.min(), 5), round(self._current_img.max(), 5)
            self.min_input.setText(str(self.vmin))
            self.max_input.setText(str(self.vmax))
            if VISPY_AVAILABLE and self._current_img is not None:
                self.image_visual.clim = (self.vmin, self.vmax)
                h, w = self._current_img.shape
                self.view.camera.rect = (0, 0, w, h)   # reset pan+zoom
                self._last_camera_rect = None
                self._last_image_shape = None
                self.canvas.update()
            elif not VISPY_AVAILABLE and self._current_img is not None:
                self._last_image_shape = None
                self._pg_apply_levels(self.vmin, self.vmax)
                self._pg_view_box.autoRange()

    def update_raw_slice(self):
        idx = self.slice_slider.value()
        self._remember_view()
        if 0 <= idx < self.raw_files_num:
            self.show_image(img_path=idx, flag="raw")
        self.filename_label.setText("")
        

    def update_try_slice(self):
        idx = self.slice_slider.value()
        self._remember_view()
        if 0 <= idx < len(self.preview_files):
            path = self.preview_files[idx]
            self.show_image(path, flag=None)
            self.filename_label.setText(os.path.basename(path))

    def update_full_slice(self):
        idx = self.slice_slider.value()
        self._remember_view()
        if not (0 <= idx < len(self.full_files)):
            return
        entry = self.full_files[idx]
        if self.full_h5 is not None:
            # H5 mode: entry is a slice index.
            self.show_image(int(entry), flag="full_h5")
            self.filename_label.setText(
                f"{os.path.basename(self.full_h5_path)} [slice {entry}]"
            )
        else:
            self.show_image(entry, flag=None)
            self.filename_label.setText(os.path.basename(entry))
        

    def _safe_open_image(self, path, flag=None, retries=3):
        # flag values: None → TIFF path, "raw" → projection index into raw H5,
        # "full_h5" → slice index into the open full-recon H5.
        # Returns the array on success, or None if every attempt failed
        # (e.g. H5 volume being actively written by tomocupy, missing part
        # files, corrupt header). Callers must handle None.
        last_exc = None
        for _ in range(retries):
            try:
                if flag == "raw":
                    return self._raw_h5['/exchange/data'][path, :, :]
                if flag == "full_h5":
                    return self.full_h5['/exchange/data'][int(path), :, :]
                with Image.open(path) as im:
                    return np.array(im)
            except (OSError, KeyError, ValueError) as exc:
                last_exc = exc
                QApplication.processEvents()
        self.log_output.append(
            f'<span style="color:orange;">Could not read image '
            f'({flag or "tiff"}, {path!r}): {last_exc}. '
            f'File may still be under reconstruction.</span>'
        )
        return None

    def show_image(self, img_path, flag=None):
        # flag: None → TIFF path; "raw" → raw projection index (dark/flat
        # corrected inline); "full_h5" → slice index into self.full_h5.
        if flag == "raw":
            try:
                img = self._raw_h5['/exchange/data'][img_path, :, :]
                img = (img - self.dark) / (self.flat - self.dark)
            except (OSError, KeyError, ValueError) as exc:
                self._blank_viewer()
                self.log_output.append(
                    f'<span style="color:orange;">Could not read raw '
                    f'projection {img_path}: {exc}</span>'
                )
                return
        elif flag == "full_h5":
            img = self._safe_open_image(img_path, flag="full_h5")
        else:
            img = self._safe_open_image(img_path)
            if img is not None and img.ndim == 3:
                img = img[..., 0]
        if img is None:
            # _safe_open_image already logged; just make sure the viewer
            # isn't left showing a stale frame from another file.
            self._blank_viewer()
            return
        h, w = img.shape
        self._current_img = img
        self._current_img_path = img_path
        self._clear_roi()

        if self.vmin is not None:
            vmin = self.vmin
        else:
            vmin = float(round(np.percentile(img, 1), 5))
            self.vmin = vmin
            self.min_input.setText(str(vmin))
        if self.vmax is not None:
            vmax = self.vmax
        else:
            vmax = float(round(np.percentile(img, 99), 5))
            self.vmax = vmax
            self.max_input.setText(str(vmax))
        if not VISPY_AVAILABLE:
            # --- PyQtGraph path ---
            self._pg_image_item.setImage(img, autoLevels=False)
            self._pg_image_item.setLevels([vmin, vmax])
            try:
                lut = (_mpl.colormaps[self.current_cmap](np.linspace(0, 1, 256)) * 255).astype(np.uint8)
                self._pg_image_item.setLookupTable(lut[:, :3])
            except Exception:
                pass
            self._last_image_shape = (h, w)
            return

        # --- VisPy path ---
        self.image_visual.set_data(img)
        self.image_visual.clim = (vmin, vmax)
        self.image_visual.cmap = self.current_cmap

        # Adapt canvas background to current theme
        current_theme = self.theme_manager.get_current_theme()
        bg_color = 'black' if current_theme == 'dark' else 'white'
        self.canvas.bgcolor = bg_color

        # Handle zoom with camera rect
        if (self._last_image_shape == (h, w) and
            self._last_camera_rect is not None):
            self.view.camera.rect = self._last_camera_rect
        else:
            self.view.camera.rect = (0, 0, w, h)

        self._last_camera_rect = self.view.camera.rect
        self._last_image_shape = (h, w)

        self.canvas.update()



    def _remember_view(self):
        """Record current view so the next image keeps the same zoom/pan."""
        try:
            if hasattr(self, 'view'):
                self._last_camera_rect = self.view.camera.rect
                if self._current_img is not None:
                    self._last_image_shape = self._current_img.shape
        except Exception:
            pass

    def _reset_view_state(self):
        """Forget any prior zoom/pan and contrast so the next image shows fresh."""
        self._last_camera_rect = None
        self._last_image_shape = None
        self.vmin = None
        self.vmax = None
        self.min_input.clear()
        self.max_input.clear()
        if not VISPY_AVAILABLE and hasattr(self, '_pg_view_box'):
            self._pg_view_box.enableAutoRange()

    # ===== TOMOLOG METHODS =====
    def get_note_value(self):
        note = self.note_input.text().strip()
        return f'"{note}"' if note else None

    def help_tomolog(self):
        """Run the CLI `tomolog run -h` and show output in the GUI log."""
        name = "tomolog-help"
        self.log_output.append(f"\U0001f4d6[{name}] tomolog run -h")

        p = QProcess(self)
        p.setProcessChannelMode(QProcess.SeparateChannels)

        p.readyReadStandardOutput.connect(
            lambda: self.log_output.append(
                bytes(p.readAllStandardOutput()).decode(errors="ignore")
            )
        )
        p.readyReadStandardError.connect(
            lambda: self.log_output.append(
                bytes(p.readAllStandardError()).decode(errors="ignore")
            )
        )

        def _done(code, _status):
            try:
                self.process[:] = [(pp, nn) for (pp, nn) in self.process if pp is not p]
            except Exception:
                pass
            self.log_output.append(f"[{name}] done." if code == 0
                                else f"[{name}] failed with code {code}.")

        p.finished.connect(_done)
        p.errorOccurred.connect(
            lambda _err: self.log_output.append(f"[{name}] {p.errorString()}")
        )

        if not isinstance(self.process, list):
            self.process = []
        self.process.append((p, name))

        p.start("tomolog", ["run", "-h"])

    def run_tomolog(self):
        beamline = self.beamline_box.currentText()
        cloud = self.cloud_box.currentText()
        url = self.url_input.text().strip()
        x = self.x_input.text().strip()
        y = self.y_input.text().strip()
        z = self.z_input.text().strip()
        scan_number = self.scan_input.text().strip()
        data_folder = self.data_path.text().strip()
        vmin = self.min_input.text().strip()
        vmax = self.max_input.text().strip()

        if not data_folder:
          self.log_output.append(f'<span style="color:red;">[ERROR] Data folder not set</span>')
          return

        flist = []
        if not scan_number:
            # Use selected (checked) files from the table; fall back to highlighted row
            for file_info in self.batch_file_main_list:
                if file_info['checkbox'].isChecked():
                    fp = (file_info.get('path')
                            or file_info.get('file')
                            or (os.path.join(data_folder, file_info['filename'])
                                if file_info.get('filename') else None))
                    if fp:
                        flist.append(fp)
            if not flist and self.highlight_scan:
                flist.append(self.highlight_scan)
            if not flist:
              self.log_output.append('<span style="color:red;">[ERROR] No files selected in table</span>')
              return
        else:
            numbers = set()
            sns = scan_number.split(",")
            for sn in sns:
                if "-" in sn:
                    try:
                        start, end = map(int, sn.split("-"))
                        numbers.update(range(start, end+1))
                    except ValueError:
                      self.log_output.append(f'<span style="color:red;">[ERROR] Invalid range: {sn}</span>')
                else:
                    try:
                        numbers.add(int(sn))
                    except ValueError:
                      self.log_output.append(f'<span style="color:red;">[ERROR] Invalid scan number: {sn}</span>')
            for n in numbers:
                fn = os.path.join(data_folder, f"*{n:04d}.h5")
                try:
                    filename = glob.glob(fn)[0]
                except IndexError:
                  self.log_output.append(f'<span style="color:red;">Scan {n:04d} not exist, stop</span>')
                  break
                flist.append(filename)
        
        note_value = self.get_note_value()
        auto_contrast = (vmin == "" and vmax == "")
        if auto_contrast:
            self.log_output.append(
              '<span style="color:#888;">min/max blank — per-file 5–95% '
                'percentile contrast will be computed from each reconstruction.</span>'
            )
        for input_fn in flist:
            cmd = [
                "tomolog", "run",
                "--beamline", beamline,
                "--file-name", input_fn,
                "--cloud", cloud,
                "--presentation-url", url,
                "--idx", x,
                "--idy", y,
                "--idz", z,
                "--note", note_value
            ]
            if auto_contrast:
                avmin, avmax = self._auto_contrast_for_file(input_fn)
                if avmin is not None and avmax is not None:
                    cmd.extend(["--min", avmin, "--max", avmax])
                    self.log_output.append(
                      f'<span style="color:#888;"> auto contrast {os.path.basename(input_fn)}: '
                        f'min={avmin}, max={avmax}</span>'
                    )
                else:
                    self.log_output.append(
                      f'<span style="color:orange;">no reconstruction TIFFs for '
                        f'{os.path.basename(input_fn)} — tomolog will use its own default contrast.</span>'
                    )
            else:
                if vmin != "":
                    cmd.extend(["--min", vmin])
                if vmax != "":
                    cmd.extend(["--max", vmax])
            extra_params = self.extra_params_input.text().strip()
            if extra_params:
                cmd.extend(extra_params.split())
            
            QApplication.processEvents()
            code = self.run_command_live(cmd, proj_file=input_fn, job_label="tomolog", wait=True, cuda_devices=None)
            if code == 0:
              self.log_output.append(f'<span style="color:green;">Done tomolog {input_fn}</span>')
            else:
              self.log_output.append(f'<span style="color:red;">Tomolog {input_fn} failed</span>')

    # ===== BATCH PROCESSING METHODS =====
    # def _run_reconstruction_on_machine(self, file_path, recon_type='try'):
    #    """
    #     Run reconstruction on selected machine (local or remote)

    #     Args:
    #         file_path: Path to the .h5 file
    #        recon_type: 'try' or 'full'

    #     Returns:
    #         Exit code (0 for success)
    #    """
    #     machine = self.batch_machine_box.currentText()

    #     # Get reconstruction parameters from Main tab
    #     recon_way = self.recon_way_box.currentText()

    #     # Get COR value EXCLUSIVELY from the batch table for this file
    #     filename = os.path.basename(file_path)
    #     cor_val = None
    #     for file_info in self.batch_file_list:
    #        if file_info['filename'] == filename:
    #            cor_val = file_info['cor_input'].text().strip()
    #             break

    #     # Batch tab ALWAYS uses manual COR with the value from the batch table
    #     # Validate COR input - batch tab requires COR to be set in table
    #     if not cor_val:
    #       self.log_output.append(f'<span style="color:red;">No COR value in batch table for {filename}</span>')
    #         return -1

    #     try:
    #         cor = float(cor_val)
    #        self.log_output.append(f'Using COR value from batch table: {cor_val} for {filename}')
    #     except ValueError:
    #       self.log_output.append(f'<span style="color:red;">Invalid COR value "{cor_val}" for {filename}</span>')
    #         return -1

    #    gpu = self.cuda_box_try.currentText().strip() if recon_type == 'try' else self.cuda_box_full.currentText().strip()

    #     # Build command
    #     # Batch tab ALWAYS uses manual COR with the value from the batch table
    #     if self.use_conf_box.isChecked():
    #        config_editor = self.config_editor_try if recon_type == 'try' else self.config_editor_full
    #         config_text = config_editor.toPlainText()
    #         if not config_text.strip():
    #           self.log_output.append(f'<span style="color:red;">No config text</span>')
    #             return -1

    #        temp_conf = os.path.join(self.data_path.text(), f"temp_{recon_type}.conf")
    #        with open(temp_conf, "w") as f:
    #             f.write(config_text)

    #        cmd = ["tomocupy", str(recon_way),
    #               "--reconstruction-type", recon_type,
    #               "--config", temp_conf,
    #               "--file-name", file_path,
    #               "--rotation-axis-auto", "manual",
    #               "--rotation-axis", str(cor)]
    #     else:
    #        cmd = ["tomocupy", str(recon_way),
    #               "--reconstruction-type", recon_type,
    #               "--file-name", file_path,
    #               "--rotation-axis-auto", "manual",
    #               "--rotation-axis", str(cor)]

    #     # Wrap command for remote execution if needed
    #     cmd = self._get_batch_machine_command(cmd, machine)

    #     # Log the machine being used
    #    if machine != "Local":
    #        self.log_output.append(f'🖥️ Running on {machine}: {os.path.basename(file_path)}')

    #     # Execute command
    #     code = self.run_command_live(cmd, proj_file=file_path,
    #                                 job_label=f"{recon_type}-{machine}",
    #                                 wait=True, cuda_devices=gpu if machine == "Local" else None)

    #     return code

    def _format_file_size(self, size_bytes):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def _batch_checkbox_clicked(self, row, checked):
        """
        Handle checkbox clicks with shift-click support for range selection
        If shift is held, select all rows between last click and current click
        """
        from PyQt5.QtWidgets import QApplication

        modifiers = QApplication.keyboardModifiers()
        from PyQt5.QtCore import Qt

        if modifiers == Qt.ShiftModifier and self.batch_last_clicked_row is not None:
            # Shift-click: check every row between the previously-clicked row
            # and this one, EXCEPT files flagged as auto-skipped (small size).
            # Those stay unchecked so a range select doesn't silently re-enable
            # aborted scans.
            start_row = min(self.batch_last_clicked_row, row)
            end_row = max(self.batch_last_clicked_row, row)

            n_skipped_small = 0
            for r in range(start_row, end_row + 1):
                for file_info in self.batch_file_main_list:
                    if file_info['row'] != r:
                        continue
                    if file_info.get('skipped_small'):
                        n_skipped_small += 1
                        break
                    file_info['checkbox'].setChecked(True)
                    break

            n = end_row - start_row + 1
            self.log_output.append(
              f'<span style="color:green;">Selected rows {start_row}–{end_row} '
                f'({n} files)</span>'
            )
            if n_skipped_small:
                self.log_output.append(
                  f'<span style="color:#888;"> ({n_skipped_small} small-file '
                    f'row(s) kept unchecked)</span>'
                )

        # Update last clicked row
        self.batch_last_clicked_row = row

    def _update_row_color(self, file_info):
        """Update the row color based on current reconstruction status"""
        try:
            # Check reconstruction status
            data_folder = self.data_path.text().strip()
            filename = file_info['filename']
            proj_name = os.path.splitext(filename)[0]
            try_dir = os.path.join(f"{data_folder}_rec", "try_center", proj_name)

            has_try = os.path.isdir(try_dir) and len(glob.glob(os.path.join(try_dir, "*.tiff"))) > 0
            has_full = self._resolve_full_recon(data_folder, proj_name)['kind'] is not None

            # Determine new color
            if has_full:
                row_color = "green"
            elif has_try:
                row_color = "orange"
            else:
                row_color = "red"

            # Update the checkbox widget border color
            if 'checkbox' in file_info:
                checkbox = file_info['checkbox']
                checkbox_widget = checkbox.parentWidget()
                if checkbox_widget:
                    checkbox_widget.setStyleSheet(f"QWidget {{ border-left: 6px solid {row_color}; }}")

            # Update stored status
            file_info['recon_status'] = row_color

        except Exception as e:
            # Silently ignore errors (widget might be deleted)
            pass

    # def _refresh_batch_file_list(self):
    #    """Refresh the file list in the batch processing tab"""
    #     folder = self.data_path.text()
    #     if not folder or not os.path.isdir(folder):
    #        QMessageBox.warning(self, "Warning", "Please select a valid data folder first.")
    #         return

    #     # Warn if queue is running
    #     if self.batch_running:
    #         reply = QMessageBox.question(
    #            self, 'Queue Running',
    #            f'A batch queue is currently running ({len(self.batch_running_jobs)} jobs active, {len(self.batch_job_queue)} queued).\n\n'
    #            f'Refreshing will delete the table widgets but jobs will continue running in the background.\n\n'
    #            f'Continue with refresh?',
    #             QMessageBox.Yes | QMessageBox.No, QMessageBox.No
    #         )
    #         if reply == QMessageBox.No:
    #             return
    #       self.log_output.append(f'<span style="color:orange;"> Refreshed file list while queue was running - status updates may be lost</span>')

    #     # Get all .h5 files
    #    h5_files = sorted(glob.glob(os.path.join(folder, "*.h5")), key=os.path.getmtime, reverse=True)

    #     # Save current COR values before clearing (to preserve user input)
    #     cor_values = {}
    #     for file_info in self.batch_file_list:
    #         try:
    #            filename = file_info['filename']
    #            cor_val = file_info['cor_input'].text().strip()
    #             if cor_val:
    #                 cor_values[filename] = cor_val
    #         except (KeyError, RuntimeError):
    #             # Widget may have been deleted
    #             pass

    #     # Clear existing table - disable sorting first to avoid issues
    #     self.batch_file_table.setSortingEnabled(False)
    #     self.batch_file_table.setRowCount(0)
    #     self.batch_file_list = []
    #     # Reset last clicked row to avoid stale row references
    #     self.batch_last_clicked_row = None

    #     # Populate table
    #     data_folder = self.data_path.text().strip()
    #     for file_path in h5_files:
    #         filename = os.path.basename(file_path)
    #         row = self.batch_file_table.rowCount()
    #         self.batch_file_table.insertRow(row)

    #         # Check reconstruction status
    #         proj_name = os.path.splitext(filename)[0]
    #        try_dir = os.path.join(f"{data_folder}_rec", "try_center", proj_name)
    #        full_dir = os.path.join(f"{data_folder}_rec", f"{proj_name}_rec")

    #        has_try = os.path.isdir(try_dir) and len(glob.glob(os.path.join(try_dir, "*.tiff"))) > 0
    #        has_full = os.path.isdir(full_dir) and len(glob.glob(os.path.join(full_dir, "*.tiff"))) > 0

    #         # Determine row color based on reconstruction status
    #         if has_full:
    #            row_color = "green"  # Full reconstruction exists
    #         elif has_try:
    #            row_color = "orange"  # Only try reconstruction exists
    #         else:
    #            row_color = "red"  # No reconstruction

    #         # Store file info
    #         file_info = {
    #            'path': file_path,
    #            'filename': filename,
    #            'status': 'Ready',
    #            'row': row,
    #            'recon_status': row_color
    #         }
    #         self.batch_file_list.append(file_info)

    #         # Checkbox for selection with shift-click support
    #         checkbox = QCheckBox()
    #         checkbox.clicked.connect(lambda checked, r=row: self._batch_checkbox_clicked(r, checked))
    #         checkbox_widget = QWidget()
    #         checkbox_layout = QHBoxLayout(checkbox_widget)
    #         checkbox_layout.addWidget(checkbox)
    #         checkbox_layout.setAlignment(Qt.AlignCenter)
    #         checkbox_layout.setContentsMargins(0, 0, 0, 0)
    #         self.batch_file_table.setCellWidget(row, 0, checkbox_widget)
    #        file_info['checkbox'] = checkbox

    #         # Filename - show full name and set tooltip with full path
    #         filename_item = QTableWidgetItem(filename)
    #        filename_item.setToolTip(f"{filename}\n\nFull path:\n{file_path}")
    #         self.batch_file_table.setItem(row, 1, filename_item)

    #         # File size
    #         try:
    #             file_size = os.path.getsize(file_path)
    #             size_str = self._format_file_size(file_size)
    #             size_item = QTableWidgetItem(size_str)
    #             size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    #             # Store numeric value for proper sorting
    #             size_item.setData(Qt.UserRole, file_size)
    #             self.batch_file_table.setItem(row, 2, size_item)
    #         except Exception as e:
    #            self.batch_file_table.setItem(row, 2, QTableWidgetItem("N/A"))

    #         # COR value (editable)
    #         cor_input = QLineEdit()
    #        cor_input.setPlaceholderText("COR value")
    #         cor_input.setAlignment(Qt.AlignCenter)
    #         cor_input.setFixedWidth(80)
    #         # Restore previous COR value if it exists
    #         if filename in cor_values:
    #             cor_input.setText(cor_values[filename])
    #         self.batch_file_table.setCellWidget(row, 3, cor_input)
    #        file_info['cor_input'] = cor_input

    #         # Status
    #        status_item = QTableWidgetItem('Ready')
    #         self.batch_file_table.setItem(row, 4, status_item)
    #        file_info['status_item'] = status_item

    #         # View Data button (original HDF5 data)
    #        view_data_btn = QPushButton("View Data")
    #         view_data_btn.clicked.connect(lambda checked, fp=file_path: self._batch_view_data(fp))
    #         self.batch_file_table.setCellWidget(row, 5, view_data_btn)

    #         # View Try button
    #        view_try_btn = QPushButton("View Try")
    #         view_try_btn.clicked.connect(lambda checked, fp=file_path: self._batch_view_try(fp))
    #         self.batch_file_table.setCellWidget(row, 6, view_try_btn)

    #         # View Full button
    #        view_full_btn = QPushButton("View Full")
    #         view_full_btn.clicked.connect(lambda checked, fp=file_path: self._batch_view_full(fp))
    #         self.batch_file_table.setCellWidget(row, 7, view_full_btn)

    #         # Actions button
    #         actions_widget = QWidget()
    #         actions_layout = QHBoxLayout(actions_widget)
    #         actions_layout.setContentsMargins(2, 2, 2, 2)
    #         actions_layout.setSpacing(2)

    #        try_btn = QPushButton("Try")
    #         try_btn.setFixedWidth(50)
    #         try_btn.clicked.connect(lambda checked, fp=file_path: self._batch_run_try_single(fp))
    #         actions_layout.addWidget(try_btn)

    #        full_btn = QPushButton("Full")
    #         full_btn.setFixedWidth(50)
    #         full_btn.clicked.connect(lambda checked, fp=file_path: self._batch_run_full_single(fp))
    #         actions_layout.addWidget(full_btn)

    #         self.batch_file_table.setCellWidget(row, 8, actions_widget)

    #         # Apply colored left border indicator based on reconstruction status
    #         # Create a colored indicator in the checkbox column
    #        checkbox_widget.setStyleSheet(f"QWidget {{ border-left: 6px solid {row_color}; }}")

    #     # Re-enable sorting after populating the table
    #     self.batch_file_table.setSortingEnabled(True)

    #    self.batch_status_label.setText(f"Loaded {len(h5_files)} files")

    #     # Try to auto-load COR values from CSV if no values were preserved from previous refresh
    #    # Only auto-load if we don't already have COR values
    #     if not cor_values:
    #         self._batch_load_cor_csv(silent=True)
    #     else:
    #         # Count how many COR values were restored
    #         restored_count = len(cor_values)
    #        self.batch_status_label.setText(f"Loaded {len(h5_files)} files ({restored_count} with COR values)")

    # def _batch_save_cor_csv(self):
    #    """Save COR values to CSV file in the data directory"""
    #     folder = self.data_path.text()
    #     if not folder or not os.path.isdir(folder):
    #        QMessageBox.warning(self, "Warning", "Please select a valid data folder first.")
    #         return

    #     if not self.batch_file_list:
    #        QMessageBox.warning(self, "Warning", "No files in the batch list.")
    #         return

    #    csv_path = os.path.join(folder, "batch_cor_values.csv")

    #     try:
    #         import csv
    #         saved_count = 0
    #         skipped_count = 0

    #         with open(csv_path, 'w', newline='') as csvfile:
    #             writer = csv.writer(csvfile)
    #            writer.writerow(['Filename', 'COR'])

    #             for file_info in self.batch_file_list:
    #                 try:
    #                    filename = file_info['filename']
    #                    cor_value = file_info['cor_input'].text().strip()
    #                     writer.writerow([filename, cor_value])
    #                     saved_count += 1
    #                 except (RuntimeError, KeyError):
    #                     # Widget was deleted (e.g., file was removed)
    #                     skipped_count += 1
    #                     continue

    #         if skipped_count > 0:
    #           self.log_output.append(f'<span style="color:orange;"> Saved {saved_count} COR values to {csv_path} ({skipped_count} skipped - widgets deleted)</span>')
    #            self.batch_status_label.setText(f"COR values saved ({skipped_count} files skipped)")
    #            QMessageBox.information(self, "Success", f"COR values saved to:\n{csv_path}\n\n{saved_count} saved, {skipped_count} skipped (deleted files)")
    #         else:
    #           self.log_output.append(f'<span style="color:green;">Saved {saved_count} COR values to {csv_path}</span>')
    #            self.batch_status_label.setText(f"COR values saved to batch_cor_values.csv")
    #            QMessageBox.information(self, "Success", f"COR values saved to:\n{csv_path}")

    #     except Exception as e:
    #       self.log_output.append(f'<span style="color:red;">Failed to save COR CSV: {e}</span>')
    #        QMessageBox.critical(self, "Error", f"Failed to save COR values:\n{e}")

    # def _batch_load_cor_csv(self, silent=False):
    #    """Load COR values from CSV file in the data directory"""
    #     folder = self.data_path.text()
    #     if not folder or not os.path.isdir(folder):
    #         if not silent:
    #            QMessageBox.warning(self, "Warning", "Please select a valid data folder first.")
    #         return

    #     if not self.batch_file_list:
    #         if not silent:
    #            QMessageBox.warning(self, "Warning", "No files in the batch list. Refresh the file list first.")
    #         return

    #    csv_path = os.path.join(folder, "batch_cor_values.csv")

    #     if not os.path.exists(csv_path):
    #         if not silent:
    #            QMessageBox.warning(self, "Warning", f"COR CSV file not found:\n{csv_path}")
    #         return

    #     try:
    #         import csv
    #         cor_dict = {}
    #        with open(csv_path, 'r') as csvfile:
    #             reader = csv.DictReader(csvfile)
    #             for row in reader:
    #                filename = row.get('Filename', '').strip()
    #                cor_value = row.get('COR', '').strip()
    #                 if filename:
    #                     cor_dict[filename] = cor_value

    #         # Apply COR values to the table
    #         loaded_count = 0
    #         skipped_count = 0
    #         for file_info in self.batch_file_list:
    #             try:
    #                filename = file_info['filename']
    #                 if filename in cor_dict:
    #                    file_info['cor_input'].setText(cor_dict[filename])
    #                     loaded_count += 1
    #             except (RuntimeError, KeyError):
    #                 # Widget was deleted (e.g., file was removed)
    #                 skipped_count += 1
    #                 continue

    #         if not silent:
    #             if skipped_count > 0:
    #               self.log_output.append(f'<span style="color:orange;"> Loaded {loaded_count} COR values from {csv_path} ({skipped_count} skipped - widgets deleted)</span>')
    #                self.batch_status_label.setText(f"Loaded {loaded_count} COR values ({skipped_count} skipped)")
    #                QMessageBox.information(self, "Success", f"Loaded {loaded_count} COR values from:\n{csv_path}\n\n{skipped_count} files skipped (deleted widgets)")
    #             else:
    #               self.log_output.append(f'<span style="color:green;">Loaded COR values from {csv_path}</span>')
    #                self.batch_status_label.setText(f"Loaded {loaded_count} COR values from CSV")
    #                QMessageBox.information(self, "Success", f"Loaded {loaded_count} COR values from:\n{csv_path}")
    #         else:
    #            self.batch_status_label.setText(f"Loaded {len(self.batch_file_list)} files ({loaded_count} with COR values)")

    #     except Exception as e:
    #         if not silent:
    #           self.log_output.append(f'<span style="color:red;">Failed to load COR CSV: {e}</span>')
    #            QMessageBox.critical(self, "Error", f"Failed to load COR values:\n{e}")


    # def _batch_remove_selected(self):
    #    """Physically delete selected files from the filesystem"""
    #    files_to_remove = [f for f in self.batch_file_list if f['checkbox'].isChecked()]

    #     if not files_to_remove:
    #        QMessageBox.warning(self, "Warning", "No files selected.")
    #         return

    #     # Confirm deletion
    #     reply = QMessageBox.question(
    #        self, 'Confirm File Deletion',
    #        f'Are you sure you want to PERMANENTLY DELETE {len(files_to_remove)} file(s) from disk?\n\nThis action cannot be undone!',
    #         QMessageBox.Yes | QMessageBox.No, QMessageBox.No
    #     )

    #     if reply == QMessageBox.No:
    #         return

    #     # Delete files from disk
    #     deleted_count = 0
    #     failed_files = []
    #     rows_to_remove = []

    #     for file_info in files_to_remove:
    #         try:
    #            os.remove(file_info['path'])
    #            rows_to_remove.append(file_info['row'])
    #             deleted_count += 1
    #           self.log_output.append(f'<span style="color:green;">Deleted: {file_info["filename"]}</span>')
    #         except Exception as e:
    #            failed_files.append(file_info['filename'])
    #           self.log_output.append(f'<span style="color:red;">Failed to delete {file_info["filename"]}: {e}</span>')

    #     # Remove rows from table
    #     for row in sorted(rows_to_remove, reverse=True):
    #         self.batch_file_table.removeRow(row)

    #     # Update file list and row indices
    #    self.batch_file_list = [f for f in self.batch_file_list if f['row'] not in rows_to_remove]
    #     for i, file_info in enumerate(self.batch_file_list):
    #        file_info['row'] = i

    #     # Update status
    #     if failed_files:
    #        self.batch_status_label.setText(f"Deleted {deleted_count} files, {len(failed_files)} failed")
    #     else:
    #        self.batch_status_label.setText(f"Successfully deleted {deleted_count} files")

    #     # Refresh the main file dropdown
    #     self.refresh_h5_files()

    def _batch_view_data(self, file_path):
        """Open HDF5 viewer to view original data"""
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "File Not Found", f"File does not exist:\n{file_path}")
            return

        try:
            # Create and show the HDF5 viewer dialog
            viewer = HDF5ImageDividerDialog(file_path=file_path, parent=self)
            viewer.show()
            self.log_output.append(f'<span style="color:green;">Opened HDF5 viewer for: {os.path.basename(file_path)}</span>')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open HDF5 viewer:\n{str(e)}")
            self.log_output.append(f'<span style="color:red;">Failed to open HDF5 viewer: {str(e)}</span>')

    def _batch_view_selected_data(self):
        """Open HDF5 viewer for the first selected file"""
        selected_files = []
        for file_info in self.batch_file_main_list:
            if file_info['checkbox'].isChecked():
                selected_files.append(file_info['path'])

        if not selected_files:
            QMessageBox.warning(self, "No Selection", "Please select at least one file to view.")
            self.log_output.append('<span style="color:orange;"> No files selected for viewing</span>')
            return

        # Open viewer for the first selected file
        first_file = selected_files[0]
        self._batch_view_data(first_file)

        if len(selected_files) > 1:
            self.log_output.append(f'<span style="color:blue;"> {len(selected_files)} files selected, opened first: {os.path.basename(first_file)}</span>')

    def _delete_selected_files(self):
        """Delete selected HDF5 files from disk after user confirmation."""
        selected = []
        for file_info in self.batch_file_main_list:
            if file_info['checkbox'].isChecked():
                selected.append(file_info['path'])

        if not selected:
            self.log_output.append('<span style="color:orange;">No files selected for deletion</span>')
            return

        file_list = "\n".join(os.path.basename(f) for f in selected)
        reply = QMessageBox.warning(
            self, "Confirm Deletion",
            f"Are you sure you want to permanently delete {len(selected)} file(s)?\n\n{file_list}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            self.log_output.append('Deletion cancelled.')
            return

        deleted = 0
        for f in selected:
            try:
                os.remove(f)
                deleted += 1
                self.log_output.append(f'️ Deleted {os.path.basename(f)}')
            except OSError as e:
              self.log_output.append(f'<span style="color:red;">Cannot delete {os.path.basename(f)}: {e}</span>')

        if deleted:
          self.log_output.append(f'<span style="color:green;">Deleted {deleted}/{len(selected)} files. Refreshing table...</span>')
          self.refresh_main_table()

    def _select_done_try(self):
        found = False
        for file_info in self.batch_file_main_list:
            status = file_info['status'].text().strip()
            if status == "Done try":
                file_info['checkbox'].setChecked(True)
                found = True
        if not found:
            self.log_output.append(f"No file is in Done try state")
        else:
            self.log_output.append(f"Select all files done try")

    # def _batch_view_try(self, file_path):
    #    """View try reconstruction for a specific file"""
    #     # Set the file in the main dropdown
    #     index = self.proj_file_box.findData(file_path)
    #     if index >= 0:
    #         self.proj_file_box.setCurrentIndex(index)
    #     else:
    #         # File not in dropdown, refresh and try again
    #         self.refresh_h5_files()
    #         index = self.proj_file_box.findData(file_path)
    #         if index >= 0:
    #             self.proj_file_box.setCurrentIndex(index)

    #     # Call the existing view try method
    #     self.view_try_reconstruction()

    # def _batch_view_full(self, file_path):
    #    """View full reconstruction for a specific file"""
    #     # Set the file in the main dropdown
    #     index = self.proj_file_box.findData(file_path)
    #     if index >= 0:
    #         self.proj_file_box.setCurrentIndex(index)
    #     else:
    #         # File not in dropdown, refresh and try again
    #         self.refresh_h5_files()
    #         index = self.proj_file_box.findData(file_path)
    #         if index >= 0:
    #             self.proj_file_box.setCurrentIndex(index)

    #     # Call the existing view full method
    #     self.view_full_reconstruction()

    def _batch_run_try_single(self, file_path):
        """Run try reconstruction on a single file using the queue system"""
        # Find the file info in batch list
        file_info = None
        for f in self.batch_file_list:
            if f['path'] == file_path:
                file_info = f
                break

        if not file_info:
          self.log_output.append(f'<span style="color:red;">File not found in batch list</span>')
          return

        # Get COR value from batch table
        batch_cor = file_info['cor_input'].text().strip()
        if not batch_cor:
          self.log_output.append(f'<span style="color:red;">No COR value in batch table for {os.path.basename(file_path)}</span>')
          QMessageBox.warning(self, "Missing COR", f"Please enter a COR value in the batch table for:\n{os.path.basename(file_path)}")
          return

        # Use the queue system with 1 GPU (respects the GPU settings)
        machine = self.batch_machine_box.currentText()
        num_gpus = self.batch_gpus_per_machine.value()

        # Run through the queue system to prevent memory overflow
        self._run_batch_with_queue([file_info], recon_type='try', num_gpus=num_gpus, machine=machine)

    def _batch_run_full_single(self, file_path):
        """Run full reconstruction on a single file using the queue system"""
        # Find the file info in batch list
        file_info = None
        for f in self.batch_file_list:
            if f['path'] == file_path:
                file_info = f
                break

        if not file_info:
          self.log_output.append(f'<span style="color:red;">File not found in batch list</span>')
          return

        # Get COR value from batch table
        batch_cor = file_info['cor_input'].text().strip()
        if not batch_cor:
          self.log_output.append(f'<span style="color:red;">No COR value in batch table for {os.path.basename(file_path)}</span>')
          QMessageBox.warning(self, "Missing COR", f"Please enter a COR value in the batch table for:\n{os.path.basename(file_path)}")
          return

        # Use the queue system with configured GPUs (respects the GPU settings)
        machine = self.batch_machine_box.currentText()
        num_gpus = self.batch_gpus_per_machine.value()

        # Run through the queue system to prevent memory overflow
        self._run_batch_with_queue([file_info], recon_type='full', num_gpus=num_gpus, machine=machine)

    def _batch_run_try_selected(self):
        """Run try reconstruction on all selected files with GPU queue management"""
        selected_files = [f for f in self.batch_file_main_list
                            if f['checkbox'].isChecked() and not f.get('skipped_small')]
        self._persist_params_for_files([f.get('path') for f in selected_files])
        machine = self.batch_machine_box.currentText()

        if not selected_files:
            QMessageBox.warning(self, "Warning", "No files selected.")
            return

        num_gpus = self.batch_gpus_per_machine.value()
        print(f'this is num gpus {num_gpus} before start')
        machine_text = f" on {machine}" if machine != "Local" else ""

        reply = QMessageBox.question(
            self, 'Confirm Batch Try',
            f'Run try reconstruction on {len(selected_files)} selected files{machine_text}?\n'
            f'Using {num_gpus} GPU(s) in parallel.',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        self._run_batch_with_queue(selected_files, recon_type='try', num_gpus=num_gpus, machine=machine)

    def _clear_selected_cors(self):
        """Clear the COR cell of every checked row, drop the value from
        self.cor_data, and persist the change to rot_cen.json."""
        selected = [f for f in self.batch_file_main_list if f['checkbox'].isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "No files selected.")
            return
        reply = QMessageBox.question(
            self, 'Clear CORs',
            f'Clear the COR value for {len(selected)} selected file(s)?\n'
            f'This updates the table and rot_cen.json.',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        cleared = 0
        for fi in selected:
            # Prefer the live cellWidget in column 2 so we survive stale
            # file_info['cor_input'] references.
            row_found = -1
            base = os.path.basename(fi.get('filename', '') or '')
            for r in range(self.batch_file_main_table.rowCount()):
                item = self.batch_file_main_table.item(r, 1)
                if item and item.text() == base:
                    row_found = r
                    break
            w = (self.batch_file_main_table.cellWidget(row_found, 2)
                    if row_found >= 0 else None)
            w = w or fi.get('cor_input')
            if w is None:
                continue
            try:
                w.setText("")
                fi['cor_input'] = w
                cleared += 1
            except RuntimeError:
                continue
            # Drop from the authoritative COR map too.
            path = fi.get('path')
            if path and path in self.cor_data:
                del self.cor_data[path]

        data_folder = self.data_path.text().strip()
        if data_folder:
            self._save_cor_data(data_folder, self.cor_data)
        self.log_output.append(
          f'<span style="color:#b26a00;">🧽 Cleared COR on {cleared} '
            f'selected row(s) and persisted to rot_cen.json.</span>'
        )

    def _cam_rot_estimate(self):
        """Estimate camera rotation angle of the highlighted file.
        Runs Try + AI-infer at nsino=0.1 and nsino=0.9 and computes

            angle_deg = np.degrees(atan((COR1 - COR2) / 2) / verticalImageSize)

        where verticalImageSize is the number of rows in /exchange/data.
        """
        import math
        import shutil
        import numpy as np
        import h5py

        proj_file = self.highlight_scan
        if not proj_file:
            QMessageBox.warning(self, "CamRot",
                                "Select a file first (click a row).")
            return
        self._persist_params_for_files([proj_file])

        model_path = self.ai_model_path.text().strip()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "CamRot",
                                "AI model path is not valid.")
            return

        # Read vertical image size from /exchange/data
        try:
            with h5py.File(proj_file, 'r') as fh:
                vertical = int(fh['/exchange/data'].shape[1])
        except Exception as e:
            QMessageBox.warning(self, "CamRot",
                                f"Cannot read /exchange/data shape: {e}")
            return

        data_folder = self.data_path.text().strip()
        proj_name = os.path.splitext(os.path.basename(proj_file))[0]
        try_dir = os.path.join(f"{data_folder}_rec", "try_center", proj_name)

        # Seed COR: per-row (if the highlighted row has one) else top-bar.
        seed = ""
        for fi in self.batch_file_main_list:
            if fi.get('path') == proj_file:
                seed = (fi['cor_input'].text().strip()
                      if fi.get('cor_input') else "")
                break
        seed = seed or self.cor_input.text().strip()
        try:
            float(seed)
        except (ValueError, TypeError):
            QMessageBox.warning(self, "CamRot",
                                "Need a valid starting COR in the row or "
                                "the top-bar for the Try runs.")
            return

        self.log_output.append(
          f'<span style="color:#00796b;">CamRot on '
            f'{os.path.basename(proj_file)} — image height = '
            f'{vertical} px, seed COR = {seed}</span>'
        )
        QApplication.processEvents()

        # Build base tomocupy-try+AI cmd; we'll swap --nsino per call.
        # Use tomocupy's built-in AI center finder so this is one subprocess
        # per nsino instead of "run try, then run inference on the TIFFs".
        base_cmd = [
            "tomocupy", self.recon_way_box.currentText(),
            "--reconstruction-type", "try",
            "--file-name", proj_file,
            "--rotation-axis-auto", "auto",
            "--rotation-axis", str(seed),
        ]
        base_cmd += self._gather_params_args()
        base_cmd += self._gather_rings_args()
        base_cmd += self._gather_bhard_args()
        base_cmd += self._gather_phase_args()
        base_cmd += self._gather_Geometry_args()
        base_cmd += self._gather_Data_args()
        base_cmd += self._gather_Performance_args()
        # AI-COR flags win argparse's last-wins tie-break against any
        # --rotation-axis-method emitted by the Reconstruction tab.
        base_cmd = self._apply_ai_cor(base_cmd)

        def _strip_nsino(cmd):
            out = []
            skip = False
            for a in cmd:
                if skip:
                    skip = False
                    continue
                if a == "--nsino":
                    skip = True
                    continue
                out.append(a)
            return out

        def _try_and_infer(nsino):
            # Clean slate for this nsino
            if os.path.isdir(try_dir):
                shutil.rmtree(try_dir, ignore_errors=True)
            cmd = _strip_nsino(base_cmd) + ["--nsino", str(nsino)]
            self.log_output.append(
                f'<span style="color:#00796b;"> → Try + AI COR at nsino={nsino} …</span>'
            )
            QApplication.processEvents()
            code = self.run_command_live(
                cmd, proj_file=proj_file,
                job_label=f"camrot-try-n{nsino}",
                wait=True, cuda_devices=None
            )
            if code != 0:
                raise RuntimeError(f"tomocupy try+AI at nsino={nsino} failed (exit {code})")

            ai_cor = self._read_ai_cor_from_try_dir(proj_file)
            if ai_cor is None:
                raise RuntimeError(
                    f"tomocupy at nsino={nsino} did not produce "
                    f"center_of_rotation.txt")
            return ai_cor

        try:
            cor1 = _try_and_infer(0.1)
            cor2 = _try_and_infer(0.9)
        except Exception as e:
            self.log_output.append(
                f'<span style="color:red;">CamRot failed: {e}</span>'
            )
            return

        # Camera tilt from COR shift over the image height:
        #   tan(angle) = (COR_top - COR_bottom) / verticalImageSize
        angle_deg = float(np.degrees(math.atan((cor1 - cor2) / vertical)))

        msg_short = (
            f"COR @ nsino=0.1 (top):    {cor1:.2f}\n"
            f"COR @ nsino=0.9 (bottom): {cor2:.2f}\n"
            f"Vertical image size:      {vertical} px\n"
            f"Estimated camera rotation: {angle_deg:.6f}°"
        )
        self.log_output.append(
          f'<span style="color:#00796b;">CamRot result: COR(top)={cor1:.2f}, '
            f'COR(bottom)={cor2:.2f}</span>'
        )
        self.log_output.append(
          f'<span style="color:#00796b;font-weight:bold;">Estimated camera '
            f'rotation: {angle_deg:.6f}°</span>'
        )
        QMessageBox.information(self, "CamRot", msg_short)

    def _fix_cor_outliers(self, abs_thresh=10.0, mad_k=5.0, max_thresh=None):
        # Read the cap from the GUI spinner if the caller didn't override it.
        if max_thresh is None:
            try:
                max_thresh = float(self.cor_outlier_max.value())
            except (AttributeError, ValueError, TypeError):
                max_thresh = 50.0
        """For the currently-checked set of files, detect outlier COR values
        within each DATASET SERIES and replace them with the average of the
        two closest non-outlier neighbours in the same series.

        A "series" is derived from the filename: everything before the final
        numeric index is treated as the series key. Examples:
            UPC15_NMC811_SC_b1_Ni_edge_1124.h5   → series 'UPC15_NMC811_SC_b1_Ni_edge'
            UPC15_NMC811_SC_b1_Mn_Elemental_1045 → series 'UPC15_NMC811_SC_b1_Mn_Elemental'

        Files within a series are sorted by their numeric index so 'neighbour'
        always means 'adjacent scan in the same series', regardless of how the
        table happens to be sorted.

        Outlier rule inside a series:
          - thr = min(max_thresh, max(abs_thresh, mad_k * MAD))
              • Small MAD (tight cluster)  → thr stays at abs_thresh (10 px).
              • Growing MAD                → thr grows with 5 × MAD.
              • MAD very large             → thr capped at max_thresh (100 px).
          - Flag v if |v − median| > thr.
          - Missing (empty) CORs also get marked for fill.

        Replacement:
          - Avg of the 2 nearest non-flagged neighbours by index in the series.
          - For MISSING values only fill when those neighbours are close to
            each other (|L − R| ≤ abs_thresh); otherwise skip — ambiguous
            cluster boundary inside the series.

        Defaults: abs_thresh=10 px, mad_k=5, max_thresh=100 px
        (anything > 100 pixels from the series median is always an outlier).
        """
        import re
        _IDX_RE = re.compile(r'^(.*?)[._-]*(\d+)$')

        def _series_key(filename):
            base = os.path.splitext(filename)[0]
            m = _IDX_RE.match(base)
            if m:
                return m.group(1), int(m.group(2))
            return base, 0
        # 1) Collect selected rows in table order
        selected = []         # list of [file_info, cor_value_or_None]
        for file_info in self.batch_file_main_list:
            if not file_info['checkbox'].isChecked():
                continue
            txt = file_info['cor_input'].text().strip()
            try:
                cor = float(txt)
            except (ValueError, TypeError):
                cor = None
            selected.append([file_info, cor])
        if not selected:
            self.log_output.append(
              '<span style="color:orange;">No files selected.</span>'
            )
            return

        def _median(arr):
            s = sorted(arr)
            m = len(s)
            return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])

        # 2) Group files by their filename series key
        series_groups = {}     # prefix -> list of (idx_num, file_info, cor)
        for file_info, cor in selected:
            prefix, idx_num = _series_key(file_info['filename'])
            series_groups.setdefault(prefix, []).append((idx_num, file_info, cor))

        changes = []   # (file_info, old_text, new_value, series, reason)
        skipped = []   # (file_info, series, reason)

        for prefix, entries in series_groups.items():
            # Sort entries by the trailing numeric index so "neighbour" is
            # the adjacent scan in the same series, not the row above/below
            # the current table sort.
            entries.sort(key=lambda t: t[0])
            n = len(entries)
            if n < 2:
                continue                              # lonely file, nothing to compare

            values = [e[2] for e in entries if e[2] is not None]
            if len(values) < 2:
                # Not enough numeric CORs in this series to compute anything
                for _, fi, c in entries:
                    if c is None:
                        skipped.append((fi, prefix, "series has <2 numeric CORs"))
                continue

            med = _median(values)
            mad = _median([abs(x - med) for x in values])
            thr = min(max_thresh, max(abs_thresh, mad_k * mad))

            # Flag: empty cells + numeric outliers against this series' median
            flagged = set()
            for i, (_, fi, c) in enumerate(entries):
                if c is None:
                    flagged.add(i)
                elif abs(c - med) > thr:
                    flagged.add(i)

            if not flagged:
                continue

            # Replace each flagged entry with avg of its 2 nearest non-flagged
            # neighbours within the SAME series.
            for i in sorted(flagged):
                was_missing = entries[i][2] is None
                left = None
                for j in range(i - 1, -1, -1):
                    if j in flagged or entries[j][2] is None:
                        continue
                    left = entries[j][2]
                    break
                right = None
                for j in range(i + 1, n):
                    if j in flagged or entries[j][2] is None:
                        continue
                    right = entries[j][2]
                    break
                fi = entries[i][1]
                if left is not None and right is not None:
                    if was_missing and abs(left - right) > abs_thresh:
                        skipped.append((fi, prefix,
                                        f"neighbours {left:.2f} / {right:.2f} differ by "
                                        f"{abs(left - right):.2f}  (> {abs_thresh})"))
                        continue
                    replacement = 0.5 * (left + right)
                    reason = f"avg({left:.2f}, {right:.2f})"
                elif left is not None:
                    if was_missing:
                        skipped.append((fi, prefix,
                                        f"only left neighbour known ({left:.2f})"))
                        continue
                    replacement = left
                    reason = f"left only = {left:.2f}"
                elif right is not None:
                    if was_missing:
                        skipped.append((fi, prefix,
                                        f"only right neighbour known ({right:.2f})"))
                        continue
                    replacement = right
                    reason = f"right only = {right:.2f}"
                else:
                    continue
                old_txt = fi['cor_input'].text().strip()
                new_txt = f"{replacement:.2f}"
                fi['cor_input'].setText(new_txt)
                changes.append((fi, old_txt, replacement, prefix, reason))

        # 3) Missing-COR fill. Independent of the outlier loop above.
        # For every selected row still without a valid COR, fill it with the
        # MEAN of CORs in the same series across the WHOLE table (donors can
        # be checked or unchecked, anywhere in the list).
        filled_from_series = []   # (fi, series, value, n_donors)

        # Build series_key → list of numeric CORs across the whole table.
        table_series_cors = {}
        for fi_all in self.batch_file_main_list:
            txt_all = (fi_all['cor_input'].text().strip()
                        if fi_all.get('cor_input') else "")
            try:
                v = float(txt_all)
            except (ValueError, TypeError):
                continue
            key_all = _series_key(fi_all['filename'])[0]
            table_series_cors.setdefault(key_all, []).append(v)

      # Drop any skipped-entry whose reason was "no COR" — it'll be replaced
        # by a fresh verdict from this pass (either filled or truly no donor).
        _missing_reasons = ("series has <2 numeric CORs",
                            "only left neighbour known",
                            "only right neighbour known",
                            "neighbours")  # prefix-match for 'neighbours ... differ'
        skipped = [s for s in skipped
                    if not any(s[2].startswith(pfx) for pfx in _missing_reasons)]

        for file_info, _ in selected:
            current_txt = (file_info['cor_input'].text().strip()
                            if file_info.get('cor_input') else "")
            if current_txt:
                continue   # already has a COR (original or freshly filled)
            series = _series_key(file_info['filename'])[0]
            donors = table_series_cors.get(series, [])
            if not donors:
                skipped.append((file_info, series,
                                "no COR donor in series (whole table)"))
                continue
            mean_val = sum(donors) / len(donors)
            file_info['cor_input'].setText(f"{mean_val:.2f}")
            filled_from_series.append((file_info, series, mean_val, len(donors)))
            changes.append(
                (file_info, "", mean_val, series,
                f"series-mean of {len(donors)} donor(s)")
            )

        if not changes and not skipped:
            self.log_output.append(
              '<span style="color:green;">No COR outliers detected and '
                'no missing CORs to fill.</span>'
            )
            return

        # 4) Persist to rot_cen.json
        data_folder = self.data_path.text().strip()
        for fi, _, newv, _, _ in changes:
          self.cor_data[fi['path']] = f"{newv:.2f}"
        if data_folder:
            self._save_cor_data(data_folder, self.cor_data)

        # 5) Log
        n_fill = len(filled_from_series)
        n_outlier = len(changes) - n_fill
        self.log_output.append(
          f'<span style="color:#8e44ad;">Fix COR Outliers (max Δ = {max_thresh:g} px): '
            f'{n_outlier} outlier(s) replaced, {n_fill} missing filled from '
            f'series mean, {len(skipped)} left unchanged across '
            f'{len(series_groups)} series.</span>'
        )
        for fi, old, newv, series, reason in changes:
            self.log_output.append(
                f'  <b>{fi["filename"]}</b> <span style="color:#888;">[{series}]</span>'
              f' : {old or "(empty)"} → {newv:.2f}   [{reason}]'
            )
        for fi, series, reason in skipped:
            self.log_output.append(
              f'<span style="color:#888;"> <b>{fi["filename"]}</b> [{series}] '
                f'skipped — {reason}</span>'
            )

    def _batch_run_ai_selected(self):
        """Run AI Reco (Try → inference → Full) sequentially on all selected files.
        Unlike Try/Full batch, this one does NOT parallelise on GPUs — each file
        goes through the full AI pipeline one after the other so the torch
      inference step doesn't fight the GPU with concurrent tomocupy jobs."""
        selected_files = [f for f in self.batch_file_main_list if f['checkbox'].isChecked()]
        self._persist_params_for_files([f.get('path') for f in selected_files])
        # Drop auto-skipped small files even if they somehow ended up checked
        # (e.g. user re-checked manually). They're marked as aborted scans and
        # have no COR — including them would wrongly block the run.
        dropped_small = [f for f in selected_files if f.get('skipped_small')]
        if dropped_small:
            selected_files = [f for f in selected_files if not f.get('skipped_small')]
            self.log_output.append(
              f'<span style="color:#888;">ignoring {len(dropped_small)} '
                f'small-file row(s) flagged as aborted scans.</span>'
            )
            for fi in dropped_small:
                try:
                    fi['checkbox'].setChecked(False)
                except Exception:
                    pass
        if not selected_files:
            QMessageBox.warning(self, "Warning", "No files selected.")
            return

        model_path = self.ai_model_path.text().strip()
        if not model_path or not os.path.exists(model_path):
          self.log_output.append('<span style="color:red;">Invalid AI model path</span>')
          return

        # AI Reco seed policy (per file): row COR if set, else top-bar COR.
        # Validate up front that every selected file will have SOMETHING to
        # seed from — either its own row COR, or the top-bar fallback.
        top_bar_txt = self.cor_input.text().strip()
        top_bar_ok = False
        try:
            float(top_bar_txt)
            top_bar_ok = True
        except (ValueError, TypeError):
            top_bar_ok = False

        # Read phase selection up front so we can validate seeds only when
        # the seed-consuming phase (Try) is actually going to run.
        run_try = self.batch_ai_phase_try.isChecked()
        run_infer = self.batch_ai_phase_infer.isChecked()
        run_full = self.batch_ai_phase_full.isChecked()
        run_tomolog = self.batch_ai_phase_tomolog.isChecked()
        if not any((run_try, run_infer, run_full, run_tomolog)):
            QMessageBox.warning(
                self, "No phase selected",
                "Tick at least one of Try / Infer / Full / TomoLog next to "
                "the Batch AI Reco button."
            )
            return

        # Series grouping helper (same rule as Fix COR Outliers).
        import re as _re
        _IDX_RE = _re.compile(r'^(.*?)[._-]*(\d+)$')

        def _sk(name):
            m = _IDX_RE.match(os.path.splitext(os.path.basename(name))[0])
            return m.group(1) if m else os.path.splitext(os.path.basename(name))[0]

        # Build donor map (whole table) up front — PREVIEW only: we need to
        # know which rows would get auto-filled to validate the run, but we
        # won't actually mutate the table until after the user confirms.
        _series_cors = {}
        for fi_all in self.batch_file_main_list:
            try:
                v = float((fi_all['cor_input'].text().strip()
                            if fi_all.get('cor_input') else ""))
            except (ValueError, TypeError):
                continue
            _series_cors.setdefault(_sk(fi_all['filename']), []).append(v)

        will_auto_fill = []    # (file_info, mean_val)
        missing_seed = []
        for fi in selected_files:
            cur = (fi['cor_input'].text().strip()
                    if fi.get('cor_input') else "")
            try:
                float(cur)
                continue  # already has a COR — nothing to do
            except (ValueError, TypeError):
                pass
            donors = _series_cors.get(_sk(fi['filename']), [])
            if donors:
                will_auto_fill.append((fi, sum(donors) / len(donors)))
            elif not top_bar_ok and self.cor_method_box.currentText() != "auto":
                missing_seed.append(fi['filename'])

        # No hard failure on missing_seed anymore — files without row / top-bar
        # / series donor will be seeded with image_width / 2 further down.
        if missing_seed:
            self.log_output.append(
              f'<span style="color:#888;">{len(missing_seed)} file(s) have '
                f'no COR anywhere; they will be seeded with mid-width at '
                f'dispatch time.</span>'
            )

        # Count how many will use the row COR vs the top-bar fallback
        row_cor_count = sum(1 for fi in selected_files
                            if fi.get('cor_input') and fi['cor_input'].text().strip()
                            and fi['cor_input'].text().strip()
                                .replace('.', '', 1).replace('-', '', 1).isdigit())
        seed_summary = (
            f"{row_cor_count} file(s) will use their table COR; "
            f"{len(selected_files) - row_cor_count} will fall back to the top-bar "
          f"({top_bar_txt or 'auto'})."
        )
        fill_summary = (
            f"\n{len(will_auto_fill)} file(s) will be auto-filled from series mean."
            if will_auto_fill else ""
        )

        phases_str = " + ".join(
            p for p, on in [("Try", run_try), ("Infer", run_infer),
                            ("Full", run_full), ("TomoLog", run_tomolog)] if on
        )
        reply = QMessageBox.question(
            self, 'Confirm Batch AI Reco',
            f'Run phases: <b>{phases_str}</b> on '
            f'{len(selected_files)} selected files?\n'
            f'Seed policy per file: row COR if present, else top-bar.\n'
            f'{seed_summary}{fill_summary}',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Confirmed → now actually apply the auto-fills so the table reflects
        # what the run will use.
        if will_auto_fill:
            for fi, mean_val in will_auto_fill:
                try:
                  fi['cor_input'].setText(f"{mean_val:.2f}")
                except (RuntimeError, AttributeError):
                    pass
            self.log_output.append(
              f'<span style="color:#8e44ad;">Auto-filled {len(will_auto_fill)} '
                f'missing COR(s) from series mean before AI Reco.</span>'
            )

        num_gpus = self.batch_gpus_per_machine.value()
        machine = self.batch_machine_box.currentText()
        self.log_output.append(
          f'<span style="color:#1a8cff;font-weight:bold;">Batch AI Reco on '
            f'{len(selected_files)} files using {num_gpus} GPU(s) — current GUI '
            f'tab settings apply to every file.</span>'
        )

        # Every file needs a seed COR that tomocupy's AI can refine around.
        # Precedence: row COR > top-bar COR > centre of the image width
        # (read from /exchange/data.shape[2] of the source H5).
        for fi in selected_files:
            row_txt = fi['cor_input'].text().strip() if fi.get('cor_input') else ""
            try:
                float(row_txt)
                continue    # already a valid seed
            except (ValueError, TypeError):
                pass
            # No row value — try top-bar, then middle-of-width.
            if top_bar_ok:
                fi['cor_input'].setText(top_bar_txt)
                continue
            mid = self._middle_of_width(fi.get('path'))
            if mid is not None:
              fi['cor_input'].setText(f"{mid:.1f}")
              self.log_output.append(
                  f'<span style="color:#888;">{os.path.basename(fi["path"])}: '
                    f'no COR — seeding AI with mid-width = {mid:.1f}</span>'
                )

        self._batch_active = True
        data_folder = self.data_path.text().strip()
        failed_inf = []      # set by Phase B (or left empty if skipped)
        # Phase B (AI COR) now runs try + AI in a single tomocupy call, so a
        # separate Phase A would just redo the try. Skip Phase A automatically
        # whenever Phase B is on to avoid burning GPU time twice.
        skip_try_for_ai = run_try and run_infer
        try:
            # ── Phase A: multi-GPU try reconstructions ───────────────
            if run_try and not skip_try_for_ai:
                self.log_output.append(
                  '<span style="color:#1a8cff;">── Phase A: TRY reconstructions '
                    '(parallel across GPUs)…</span>'
                )
                QApplication.processEvents()
                self._run_batch_with_queue(selected_files, recon_type='try',
                                            num_gpus=num_gpus, machine=machine)
            elif skip_try_for_ai:
                self.log_output.append(
                  '<span style="color:#888;">── Phase A (Try) skipped — '
                    'Phase B (AI COR) will run try + inference in one call.</span>'
                )
            else:
                self.log_output.append(
                  '<span style="color:#888;">── Phase A (Try) skipped — '
                    'using existing try_center TIFFs.</span>'
                )

            # ── Phase B: tomocupy try + AI COR (one file per GPU slot) ─
            if run_infer:
                self.log_output.append(
                  f'<span style="color:#1a8cff;">── Phase B: Try + AI COR — '
                    f'{len(selected_files)} file(s), {num_gpus} GPU slot(s). '
                    f'Row COR is used as the AI seed; the AI refines around it.</span>'
                )
                QApplication.processEvents()
                self._run_batch_with_queue(selected_files, recon_type='infer',
                                            num_gpus=num_gpus, machine=machine)

                # ─── The ONE place that writes AI CORs back to the table ───
                # Scan column 1 (filename) to find the row, then setText on
                # cellWidget(row, 2). No stored widget references, no lambda
                # captures, no cross-thread state.
                inferred = 0
                for fi in selected_files:
                    proj_file = fi.get('path') or fi.get('file')
                    if not proj_file:
                        continue
                    basename = os.path.basename(proj_file)
                    proj_name = os.path.splitext(basename)[0]
                    cor_txt = os.path.join(
                        f"{data_folder}_rec", "try_center",
                        proj_name, 'center_of_rotation.txt')

                    # Read the AI's answer
                    if not os.path.exists(cor_txt):
                        failed_inf.append(basename)
                        self.log_output.append(
                            f'<span style="color:red;"> {basename}: '
                            f'no center_of_rotation.txt</span>'
                        )
                        continue
                    try:
                        with open(cor_txt) as f:
                            raw = [ln.strip() for ln in f if ln.strip()]
                        if not raw:
                            raise ValueError("empty file")
                        ai_cor = float(raw[-1].split()[-1])
                    except Exception as e:
                        failed_inf.append(basename)
                        self.log_output.append(
                          f'<span style="color:red;"> {basename}: '
                            f'could not parse {cor_txt} ({e})</span>'
                        )
                        continue

                    txt = f"{ai_cor:.2f}"

                    # Find the row by scanning column 1 directly. This is the
                    # authoritative lookup — ignores fi['row'] entirely.
                    row_found = -1
                    for r in range(self.batch_file_main_table.rowCount()):
                        item = self.batch_file_main_table.item(r, 1)
                        if item and item.text() == basename:
                            row_found = r
                            break
                    if row_found < 0:
                        self.log_output.append(
                          f'<span style="color:red;"> {basename}: '
                            f'row not found in table</span>'
                        )
                        continue

                    cell_w = self.batch_file_main_table.cellWidget(row_found, 2)
                    if cell_w is None:
                        self.log_output.append(
                          f'<span style="color:red;"> {basename}: '
                            f'no widget at row {row_found} col 2</span>'
                        )
                        continue
                    old_txt = cell_w.text().strip()
                    cell_w.setText(txt)
                    cell_w.setModified(True)
                    # Force the widget AND the table to repaint right now —
                    # over SSH X11 / pyqtgraph fallback, scheduled paints
                    # often don't fire until the queue loop yields enough.
                    cell_w.repaint()
                    self.batch_file_main_table.viewport().update()
                    QApplication.sendPostedEvents()
                    QApplication.processEvents()
                    # Keep the file_info dict and the global cor_data in sync
                    fi['cor_input'] = cell_w   # refresh stale ref, just in case
                    self.cor_data[proj_file] = txt
                    inferred += 1
                    if old_txt == txt:
                        self.log_output.append(
                          f'<span style="color:#888;"> {basename}: '
                            f'{txt} (unchanged)</span>'
                        )
                    else:
                        self.log_output.append(
                          f'<span style="color:#1a8cff;"> {basename}: '
                          f'{old_txt or "(empty)"} → {txt}</span>'
                        )

                # Final blanket repaint so any deferred paint events flush
                # before the user's eyes see the table.
                self.batch_file_main_table.viewport().update()
                self.batch_file_main_table.repaint()
                QApplication.sendPostedEvents()
                QApplication.processEvents()

                if data_folder:
                    self._save_cor_data(data_folder, self.cor_data)
                self.log_output.append(
                    f'<span style="color:#1a8cff;"> Phase B done: {inferred} succeeded, '
                    f'{len(failed_inf)} failed.</span>'
                )
            else:
                self.log_output.append(
                  '<span style="color:#888;">── Phase B (Infer) skipped — '
                    'Full will use CORs already in the table.</span>'
                )

            # ── Phase C: multi-GPU full reconstructions ──────────────
            if run_full:
                self.log_output.append(
                  '<span style="color:#1a8cff;">── Phase C: FULL reconstructions '
                    '(parallel across GPUs)…</span>'
                )
                QApplication.processEvents()
                # Only run full on files where inference actually produced a COR
                # (or all files if Phase B was skipped — trust the existing CORs).
                good_for_full = [fi for fi in selected_files
                                if os.path.basename(fi.get('path') or '')
                                    not in set(failed_inf)]
                self._run_batch_with_queue(good_for_full, recon_type='full',
                                            num_gpus=num_gpus, machine=machine)
            else:
                self.log_output.append(
                  '<span style="color:#888;">── Phase C (Full) skipped.</span>'
                )

            # ── Phase D (optional): upload reconstructions to TomoLog ──
            if run_tomolog:
                self.log_output.append(
                  '<span style="color:#1a8cff;">── Phase D: TomoLog upload '
                    '(sequential, one file at a time)…</span>'
                )
                QApplication.processEvents()
                uploaded = 0
                upload_failed = []
                for fi in good_for_full:
                    fpath = fi.get('path') or fi.get('file')
                    if not fpath:
                        continue
                    # Only upload files whose Full completed successfully. Check
                    # the table status text we set after Phase C.
                    base = os.path.basename(fpath)
                    try:
                        r = self._find_row_by_filename(base)
                        st = self.batch_file_main_table.item(r, 3) if r is not None else None
                        status_txt = (st.text() if st else "").lower()
                    except Exception:
                        status_txt = ""
                    if "fail" in status_txt or "skip" in status_txt:
                        upload_failed.append(base + " (no full recon)")
                        continue
                    try:
                        self._set_status_by_filename(
                            base, "Uploading…",
                            status_col=3, filename_col=1, color="#8e44ad"
                        )
                    except RuntimeError:
                        pass
                    QApplication.processEvents()
                    try:
                        self._run_tomolog_for_file(fpath)
                        uploaded += 1
                        try:
                            self._set_status_by_filename(
                                base, "Uploaded",
                                status_col=3, filename_col=1, color="#27ae60"
                            )
                        except RuntimeError:
                            pass
                    except Exception as e:
                        upload_failed.append(f"{base} ({e})")
                        try:
                            self._set_status_by_filename(
                                base, "Upload failed",
                                status_col=3, filename_col=1, color="#c0392b"
                            )
                        except RuntimeError:
                            pass
                    QApplication.processEvents()
                self.log_output.append(
                  f'<span style="color:#1a8cff;"> Phase D done: {uploaded} uploaded, '
                    f'{len(upload_failed)} failed.</span>'
                )
        finally:
            self._batch_active = False

        self.log_output.append(
          '<span style="color:green;font-weight:bold;">Batch AI Reco finished.</span>'
        )

    def _batch_run_full_selected(self):
        """Run full reconstruction on all selected files with GPU queue management"""
        selected_files = [f for f in self.batch_file_main_list
                            if f['checkbox'].isChecked() and not f.get('skipped_small')]
        self._persist_params_for_files([f.get('path') for f in selected_files])
        machine = self.batch_machine_box.currentText()

        if not selected_files:
            QMessageBox.warning(self, "Warning", "No files selected.")
            return

        num_gpus = self.batch_gpus_per_machine.value()
        machine_text = f" on {machine}" if machine != "Local" else ""

        reply = QMessageBox.question(
            self, 'Confirm Batch Full Reconstruction',
            f'Run full reconstruction on {len(selected_files)} selected files{machine_text}?\n'
            f'Using {num_gpus} GPU(s) in parallel.\nThis may take a long time.',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        self._run_batch_with_queue(selected_files, recon_type='full', num_gpus=num_gpus, machine=machine)

  #=========helper to update table based on filename========================
    def _get_full_recon_status(self, file_path):
        """Return (status_text, status_color) for the full reconstruction of
        ``file_path``. Handles the tomocupy default H5 output as well as the
        legacy TIFF directory.
        """
        try:
            table_folder = self.data_path.text()
            if not table_folder:
                return "Done full", "green"
            filename = os.path.basename(file_path)
            proj_name = os.path.splitext(filename)[0]
            info = self._resolve_full_recon(table_folder, proj_name)
            if info['kind'] and info['range'] is not None:
                n1, n2 = info['range']
                return f"Full {n1}-{n2}", "green"
            if info['kind']:
              return f"Full ({info['kind']})", "green"
            return "Done full", "green"
        except Exception:
            return "Done full", "green"

    def _set_status_by_filename(self, filename, text, status_col=3, filename_col=1, color=None):
        table = self.batch_file_main_table
        row = self._find_row_by_filename(filename, filename_col=filename_col)
        if row is None:
            return False  # not found (maybe list refreshed)

        item = table.item(row, status_col)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # optional
            table.setItem(row, status_col, item)
        if color is not None:
            self.batch_file_main_list[row]['recon_status'] = color
            checkbox_widget = self.batch_file_main_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox_widget.setStyleSheet(f"QWidget {{ border-left: 6px solid {color}; }}")
            status_item = QTableWidgetItem(text)
            self.batch_file_main_table.setItem(row, 3, status_item)
            self.batch_file_main_list[row]['status'] = text
        return True

    def _run_batch_with_queue(self, selected_files, recon_type, num_gpus, machine):
        """
        Run batch reconstructions with GPU queue management.
        Sets _batch_active so row-click events during the queue do NOT
        swap in each row's saved params — the current GUI tab settings
        (ring correction, phase, geometry, performance, ...) are used
        uniformly for every file in this batch.
        """
        self._batch_active = True
        self.log_output.append(
            f'<span style="color:magenta;">DEBUG: Starting {recon_type} batch, '
            f'batch_running={self.batch_running}</span>'
        )
        self.log_output.append(
          '<span style="color:#888;">⚙️ Using current GUI tab settings for every file in this batch.</span>'
        )

        jobs_to_add = [(f, recon_type, machine) for f in selected_files]

        # Mark all jobs as queued
        for f, _, _ in jobs_to_add:
            try:
                self._set_status_by_filename(
                    os.path.basename(f["filename"]), "Queued", status_col=3, filename_col=1, color="blue"
                )
            except RuntimeError:
                pass

        # If queue is already running, just add to it
        if self.batch_running:
            self.batch_job_queue.extend(jobs_to_add)
            self.batch_total_jobs += len(selected_files)
            self.log_output.append(
              f'<span style="color:blue;">➕ Added {len(selected_files)} job(s) to running queue</span>'
            )
            return

        # Start new queue
        self.batch_running = True
        self.batch_job_queue = jobs_to_add
        self.batch_running_jobs = {}
        self.batch_available_gpus = list(range(num_gpus))
        self.batch_current_machine = machine
        self.batch_current_num_gpus = num_gpus

        self.batch_total_jobs = len(selected_files)
        self.batch_completed_jobs = 0

        self.log_output.append(
          f'<span style="color:blue;">Starting batch queue: {len(self.batch_job_queue)} jobs, {num_gpus} GPU(s)</span>'
        )

        QApplication.processEvents()

        progress_window_opened = False  #gate progress window

        # Keep processing until queue is empty and all jobs are done
        while self.batch_job_queue or self.batch_running_jobs:
            self.log_output.append(
              f'<span style="color:gray;">🔄 Queue loop: {len(self.batch_job_queue)} queued, {len(self.batch_running_jobs)} running, {len(self.batch_available_gpus)} GPUs available</span>'
            )
            QApplication.processEvents()

            # Start new jobs if GPUs are available and jobs are queued
            while self.batch_available_gpus and self.batch_job_queue:
                gpu_id = self.batch_available_gpus.pop(0)
                file_info, job_recon_type, job_machine = self.batch_job_queue.pop(0)

                try:
                    self._set_status_by_filename(
                        os.path.basename(file_info["filename"]),
                        f"Running on GPU {gpu_id}",
                        status_col=3,
                        filename_col=1,
                        color="yellow"
                    )
                except RuntimeError:
                    pass

                QApplication.processEvents()

                # capture return value (process) and validate it
                try:
                    process = self._start_batch_job_async(file_info, job_recon_type, gpu_id, job_machine)  
                except Exception as e:
                    self.log_output.append(
                      f'<span style="color:red;">Failed to start job for {file_info.get("filename","?")}: {e}</span>'
                    )
                    # Put GPU back and mark failed
                    self.batch_available_gpus.append(gpu_id)
                    self.batch_available_gpus.sort()
                    try:
                        self._set_status_by_filename(
                            os.path.basename(file_info["filename"]), "Ready", 3, 1, color="red"
                        )
                    except RuntimeError:
                        pass
                    continue  # <<< CHANGED: continue queue

                if process is None or not isinstance(process, QProcess):
                    # Process is None when job is skipped (e.g., missing COR)
                    # The specific reason was already logged in _start_batch_job_async
                    self.batch_available_gpus.append(gpu_id)
                    self.batch_available_gpus.sort()
                    try:
                        self._set_status_by_filename(os.path.basename(file_info["filename"]), "Skipped", 3, 1, color="gray")
                    except RuntimeError:
                        pass
                    # Count as completed to keep progress accurate
                    self.batch_completed_jobs += 1
                    continue

                # open progress window ONLY after first process starts successfully
                if not progress_window_opened:
                    self.progress_window.batch_progress_bar.setValue(0)
                    self.progress_window.batch_queue_label.setText(f"Queue: {len(self.batch_job_queue)} jobs waiting")
                    self.progress_window.batch_status_label.setText("Running batch jobs…")
                    self.progress_window.show()
                    progress_window_opened = True
                    #enable Stop button in the progress window
                    try:
                        self.progress_window.set_running(True)
                    except Exception:
                        pass

                self.batch_running_jobs[gpu_id] = (process, file_info, job_recon_type)

                self.log_output.append(
                  f'<span style="color:blue;">GPU {gpu_id}: Started {job_recon_type} - {file_info["filename"]} '
                    f'(Running: {len(self.batch_running_jobs)}, Queued: {len(self.batch_job_queue)})</span>'
                )

            # Check for completed jobs
            completed_gpus = []
            for gpu_id, (process, file_info, job_recon_type) in list(self.batch_running_jobs.items()):
                if process.state() == QProcess.NotRunning:
                    exit_code = process.exitCode()
                    self.batch_completed_jobs += 1

                    try:
                        if exit_code == 0:
                            # Set status based on reconstruction type
                            if job_recon_type == 'try':
                                status_text = "Done try"
                                status_color = "orange"
                            elif job_recon_type == 'infer':
                                # Only set status here. The actual COR cell
                                # write is done by a single clean pass at the
                                # end of Phase B (see _batch_run_ai_selected)
                                # so there's exactly one update path and no
                                # stale-widget races.
                                status_text = "Inferred"
                                status_color = "#27ae60"
                            else:  # full
                                # Check output directory for actual slice numbers
                                status_text, status_color = self._get_full_recon_status(file_info["filename"])

                            self._set_status_by_filename(
                                os.path.basename(file_info["filename"]),
                                status_text,
                                status_col=3,
                                filename_col=1,
                                color=status_color
                            )
                            self.log_output.append(f'<span style="color:green;">GPU {gpu_id} finished {job_recon_type}: {file_info["filename"]}</span>')
                        else:
                            self._set_status_by_filename(
                                os.path.basename(file_info["filename"]),
                                f"{job_recon_type.capitalize()} Failed",
                                status_col=3,
                                filename_col=1,
                                color="red"
                            )
                            self.log_output.append(f'<span style="color:red;">GPU {gpu_id} failed {job_recon_type}: {file_info["filename"]}</span>')
                    except RuntimeError:
                        self.log_output.append(
                          f'<span style="color:gray;">GPU {gpu_id} finished: {file_info["filename"]} (widget deleted)</span>'
                        )

                    completed_gpus.append(gpu_id)

            # Free up completed GPUs
            for gpu_id in completed_gpus:
                del self.batch_running_jobs[gpu_id]
                self.batch_available_gpus.append(gpu_id)
                self.batch_available_gpus.sort()

            # Update progress
            progress = int((self.batch_completed_jobs / self.batch_total_jobs) * 100) if self.batch_total_jobs else 0

            if progress_window_opened:
                self.progress_window.batch_progress_bar.setValue(progress)
                active_gpus = sorted(self.batch_running_jobs.keys())
                gpu_status = f"GPUs: {active_gpus}" if active_gpus else "GPUs: idle"
                self.progress_window.batch_status_label.setText(
                    f"Completed {self.batch_completed_jobs}/{self.batch_total_jobs} | {gpu_status} | Queue: {len(self.batch_job_queue)}"
                )
                self.progress_window.batch_queue_label.setText(f"Queue: {len(self.batch_job_queue)} jobs waiting")

            QApplication.processEvents()

            if self.batch_running_jobs:
                import time
                time.sleep(0.2)

        # Finalize
        if progress_window_opened:
            self.progress_window.batch_progress_bar.setValue(100)
            self.progress_window.batch_status_label.setText("Batch completed.")
            self.progress_window.batch_queue_label.setText("Queue: 0 jobs waiting")

            self.log_output.append(f'<span style="color:green;">Batch queue finished: {self.batch_completed_jobs} files completed</span>')

        # Reset batch running flag so new batches can start
        self.batch_running = False
        self._batch_active = False   # re-enable per-scan param load/save on clicks
        self.log_output.append('<span style="color:blue;">batch_running set to False, ready for new batch</span>')


    def _batch_stop_queue(self):
        """Immediately stop the batch queue and kill all running jobs."""

        # Nothing to stop
        if not getattr(self, "batch_running", False):
            return

        # ===== CRITICAL: stop scheduling FIRST =====
        self.batch_running = False
        self._batch_active = False   # re-enable per-scan param load/save on clicks

        # ===== Kill all running processes =====
        for gpu_id, (process, file_info, job_recon_type) in list(self.batch_running_jobs.items()):
            try:
                # Best-effort terminate → kill
                try:
                    process.terminate()
                    if not process.waitForFinished(1500):
                        process.kill()
                except Exception:
                    process.kill()
            except Exception:
                pass

            # Update table status
            try:
              self._set_status_by_filename(file_info['filename'],text="Cancelled batch",color='red')
            except Exception:
                pass

            self.log_output.append(
              f'<span style="color:orange;">Cancelled job on GPU {gpu_id}: '
              f'{file_info.get("filename", "")}</span>'
            )

        # ===== Cancel queued (not yet started) jobs =====
        for file_info, job_recon_type, job_machine in self.batch_job_queue:
            try:
              self._set_status_by_filename(file_info['filename'],text="Cancelled batch",color='red')
            except Exception:
                pass

        # ===== Clear internal state =====
        self.batch_job_queue.clear()
        self.batch_running_jobs.clear()

        # ===== Reset MAIN GUI =====
        #self.batch_stop_btn.setEnabled(False)
        #self.batch_progress_bar.setValue(0)
        #self.batch_status_label.setText("Batch stopped")
        #self.batch_queue_label.setText("Queue: 0 jobs waiting")

        # ===== Mirror state to PROGRESS WINDOW =====
        try:
            if hasattr(self, "progress_window") and self.progress_window is not None:
                self.progress_window.set_running(False)
                self.progress_window.set_progress(0)
                self.progress_window.set_status("Batch stopped")
                self.progress_window.set_queue(0)
        except Exception:
            pass

        self.log_output.append(
          '<span style="color:orange;">Batch queue stopped by user</span>'
        )
            
    def _start_batch_job_async(self, file_info, recon_type, gpu_id, machine):
        """
        Start a reconstruction job asynchronously
        Returns: QProcess object
        """
        file_path = file_info['path']
        filename = os.path.basename(file_path)

        # AI COR: launch tomocupy in try mode with the built-in AI center
        # finder. Tomocupy runs the try recon, executes the AI inference on
        # the cached slices, and writes center_of_rotation.txt inside the
        # try output dir. The batch collector reads that file to fill in the
        # table + feed Full.
        if recon_type == 'infer':
            model_path = self.ai_model_path.text().strip()
            if not model_path or not os.path.exists(model_path):
                self.log_output.append(
                  f'<span style="color:red;">AI model path invalid for {filename}</span>'
                )
                return None
            data_folder = self.data_path.text().strip()
            proj_name = os.path.splitext(filename)[0]
            try_dir = os.path.join(f"{data_folder}_rec", "try_center", proj_name)
            stale = os.path.join(try_dir, "center_of_rotation.txt")
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass

            # Per-file COR seed: row first, top-bar fallback (same policy
            # as the try phase).
            row_cor = ""
            if file_info.get('cor_input') is not None:
                try:
                    row_cor = file_info['cor_input'].text().strip()
                except Exception:
                    row_cor = ""
            seed = row_cor or self.cor_input.text().strip()
            try:
                float(seed)
            except (ValueError, TypeError):
                seed = ""

            recon_way = self.recon_way_box.currentText()
            if self.use_conf_box.isChecked():
                config_text = self.config_editor_try.toPlainText()
                temp_conf = os.path.join(self.data_path.text(),
                                        f"temp_infer_gpu{gpu_id}.conf")
                with open(temp_conf, "w") as f:
                    f.write(config_text)
                cmd = ["tomocupy", str(recon_way),
                      "--reconstruction-type", "try",
                      "--config", temp_conf,
                      "--file-name", file_path,
                      "--rotation-axis-auto", "auto"]
                if seed:
                    cmd += ["--rotation-axis", seed]
            else:
                cmd = ["tomocupy", str(recon_way),
                      "--reconstruction-type", "try",
                      "--file-name", file_path,
                      "--rotation-axis-auto", "auto"]
                if seed:
                    cmd += ["--rotation-axis", seed]
                cmd += self._gather_params_args()
                cmd += self._gather_rings_args()
                cmd += self._gather_bhard_args()
                cmd += self._gather_phase_args()
                cmd += self._gather_Geometry_args()
                cmd += self._gather_Data_args()
                cmd += self._gather_Performance_args()

            # AI flags must be applied AFTER params so they win argparse's
            # last-value-wins tie-break (otherwise --rotation-axis-method
            # from the Reconstruction tab would silently override "ai").
            cmd = self._apply_ai_cor(cmd)

            cmd = self._get_batch_machine_command(cmd, machine)
            p = QProcess(self)
            p.setProcessChannelMode(QProcess.SeparateChannels)
            p.readyReadStandardOutput.connect(
                lambda proc=p, fn=filename: self._on_process_output(proc, fn, is_error=False)
            )
            p.readyReadStandardError.connect(
                lambda proc=p, fn=filename: self._on_process_output(proc, fn, is_error=True)
            )
            if machine == "Local":
                env = QProcessEnvironment.systemEnvironment()
                env.insert("CUDA_VISIBLE_DEVICES", str(gpu_id))
                env.insert("HDF5_USE_FILE_LOCKING", "FALSE")
                p.setProcessEnvironment(env)
            if cmd and cmd[0] != "env":
                cmd = ["env", "HDF5_USE_FILE_LOCKING=FALSE"] + list(cmd)
            p.start(str(cmd[0]), [str(a) for a in cmd[1:]])
            if not p.waitForStarted(5000):
                self.log_output.append(
                    f'<span style="color:red;">AI COR job failed to start for {filename}</span>'
                )
                return None
            self.log_output.append(
              f'<span style="color:blue;">GPU {gpu_id} Try+AI-COR start: {filename} '
                f'(PID {p.processId()})</span>'
            )
            return p

        # properly read method text from combo box
        if recon_type == 'try':
            recon_way = self.recon_way_box.currentText()
            rec_method = self.cor_method_box.currentText()
            # Per-file seed: row COR first, top-bar fallback. This matters for
            # Batch AI Reco where each file gets its own starting guess.
            row_cor = ""
            if file_info.get('cor_input') is not None:
                try:
                    row_cor = file_info['cor_input'].text().strip()
                except Exception:
                    row_cor = ""
            cor_val = row_cor or self.cor_input.text().strip()
            if rec_method == 'manual':
                try:
                    cor = float(cor_val)
                except ValueError:
                    self.log_output.append(
                      f'<span style="color:red;">Invalid COR "{cor_val}" for {filename}, skipping</span>'
                    )
                    return None  #return None to indicate failure

        elif recon_type == 'full':
            # Same file about to be rewritten by tomocupy — release the
            # viewer's H5 handle if it points at this dataset.
            self._release_full_h5_for_write(file_path)
            recon_way = self.recon_way_box_full.currentText()
            cor_val = file_info['cor_input'].text().strip()
            rec_method = self.cor_full_method.currentText()

            if not cor_val:
                # Fall back to the top-bar Try COR input so batches don't
                # skip files that never had a per-row COR filled in.
                fallback = self.cor_input.text().strip()
                if fallback:
                    cor_val = fallback
                    file_info['cor_input'].setText(cor_val)   # reflect in the table
                    self.log_output.append(
                      f'<span style="color:orange;">No row COR for {filename}, '
                        f'using Try-bar COR = {cor_val}</span>'
                    )
                else:
                    self.log_output.append(
                      f'<span style="color:orange;">No COR value in batch table '
                        f'for {filename} and Try-bar is empty, skipping</span>'
                    )
                    return None

            try:
                cor = float(cor_val)
            except ValueError:
                self.log_output.append(
                  f'<span style="color:red;">Invalid COR value "{cor_val}" for {filename}, skipping</span>'
                )
                # Return None to skip this job - queue will handle it
                return None
        else:
            self.log_output.append(
              f'<span style="color:red;">Unknown recon_type "{recon_type}"</span>'
            )
            return None  

        # Build command
        if self.use_conf_box.isChecked():
            config_editor = self.config_editor_try if recon_type == 'try' else self.config_editor_full
            config_text = config_editor.toPlainText()

            temp_conf = os.path.join(self.data_path.text(), f"temp_{recon_type}_gpu{gpu_id}.conf")
            with open(temp_conf, "w") as f:
                f.write(config_text)

            #always build cmd deterministically
            if rec_method == "manual":
                cmd = [
                    "tomocupy", str(recon_way),
                    "--reconstruction-type", recon_type,
                    "--config", temp_conf,
                    "--file-name", file_path,
                    "--rotation-axis-auto", "manual",
                    "--rotation-axis", str(cor)
                ]
            else:
                cmd = [
                    "tomocupy", str(recon_way),
                    "--reconstruction-type", recon_type,
                    "--config", temp_conf,
                    "--file-name", file_path,
                    "--rotation-axis-auto", "auto"
                ]
        else:
            if rec_method == "manual":
                cmd = [
                    "tomocupy", str(recon_way),
                    "--reconstruction-type", recon_type,
                    "--file-name", file_path,
                    "--rotation-axis-auto", rec_method,
                    "--rotation-axis", str(cor)
                ]
            else:
                cmd = [
                    "tomocupy", str(recon_way),
                    "--reconstruction-type", recon_type,
                    "--file-name", file_path,
                    "--rotation-axis-auto", "auto"
                ]
            # Append tabs selections
            cmd += self._gather_params_args()
            cmd += self._gather_rings_args()
            cmd += self._gather_bhard_args()
            cmd += self._gather_phase_args()
            cmd += self._gather_Geometry_args()        
            cmd += self._gather_Data_args()                
            cmd += self._gather_Performance_args()           
        self.log_output.append(f'{cmd}')    

        # <<< FIX: assign wrapped cmd (previously return value was ignored)
        cmd = self._get_batch_machine_command(cmd, machine)  # <<< FIX

        # Check if user wants terminal window for remote jobs
        use_terminal = self.batch_use_terminal.isChecked() and machine != "Local"

        if use_terminal:
            # Open in separate terminal window (xterm, gnome-terminal, etc.)
            # Use xterm with -hold to keep window open after completion
            terminal_cmd = ["xterm", "-hold", "-title", f"Reconstruction: {filename}", "-e"] + cmd
            cmd = terminal_cmd

        # Create and configure process
        p = QProcess(self)

        if use_terminal:
            # Terminal handles output display, just need basic channel mode
            p.setProcessChannelMode(QProcess.ForwardedChannels)
        else:
            # Capture output to display in log
            p.setProcessChannelMode(QProcess.SeparateChannels)
            # Connect output signals to display in log
            p.readyReadStandardOutput.connect(
                lambda proc=p, fn=filename: self._on_process_output(proc, fn, is_error=False)
            )
            p.readyReadStandardError.connect(
                lambda proc=p, fn=filename: self._on_process_output(proc, fn, is_error=True)
            )

        # Set CUDA_VISIBLE_DEVICES for GPU assignment (local only)
        if machine == "Local":
            env = QProcessEnvironment.systemEnvironment()
            env.insert("CUDA_VISIBLE_DEVICES", str(gpu_id))
            env.insert("HDF5_USE_FILE_LOCKING", "FALSE")
            p.setProcessEnvironment(env)

        # Also wrap with `env HDF5_USE_FILE_LOCKING=FALSE` at the shell level
        # in case the QProcess environment insert gets dropped somewhere.
        if cmd and cmd[0] != "env":
            cmd = ["env", "HDF5_USE_FILE_LOCKING=FALSE"] + list(cmd)

        # Start process
        p.start(str(cmd[0]), [str(a) for a in cmd[1:]])

        # Wait a moment for process to actually start
        if not p.waitForStarted(5000):  # Wait up to 5 seconds
            self.log_output.append(
              f'<span style="color:red;">Process failed to start for {filename}</span>'
            )
            return None

        self.log_output.append(
          f'<span style="color:blue;">Process started successfully for {filename} (PID: {p.processId()})</span>'
        )
        return p

    def _on_process_output(self, process, filename, is_error=False):
        """Handle stdout/stderr from batch reconstruction processes.

        Mirrors the child process's output to BOTH the GUI log panel and
        the parent terminal (sys.stdout / sys.stderr). Users running tomogui
        from a shell expect to see tomocupy progress in that shell, not just
        in the GUI's log widget.
        """
        import sys
        if is_error:
            data = bytes(process.readAllStandardError()).decode(errors="ignore")
            color = "orange"
            sink = sys.stderr
        else:
            data = bytes(process.readAllStandardOutput()).decode(errors="ignore")
            color = "gray"
            sink = sys.stdout

        if not data.strip():
            return

        basename = os.path.basename(filename)
        try:
            sink.write(data if data.endswith("\n") else data + "\n")
            sink.flush()
        except (OSError, ValueError):
            pass

        for line in data.strip().split('\n'):
            if line.strip():
                self.log_output.append(
                    f'<span style="color:{color};">[{basename}] {line}</span>'
                )

    def _set_cor_cell(self, file_info, cor_val):
        """Robustly update a batch-table row's COR cell. Returns True iff
        at least one widget was actually updated. Always logs what happened
        so the user can see whether the AI value reached the cell or not."""
        base = os.path.basename(file_info.get('filename', '') or '')
        try:
            txt = f"{float(cor_val):.2f}"
        except (ValueError, TypeError):
            self.log_output.append(
                f'<span style="color:red;"> COR {base}: invalid value '
                f'{cor_val!r} from AI — cell NOT updated</span>'
            )
            return False

        old_txt = ""
        widgets_seen = 0
        widgets_written = 0
        w = file_info.get('cor_input')
        if w is not None:
            widgets_seen += 1
            try:
                old_txt = w.text().strip()
                w.setText(txt)
                widgets_written += 1
            except RuntimeError as e:
                self.log_output.append(
                  f'<span style="color:red;"> COR {base}: stored '
                    f'widget is dead ({e}) — falling through to live '
                    f'cellWidget lookup</span>'
                )
        # Always also try the live cellWidget by row lookup.
        try:
            r = self._find_row_by_filename(base)
        except Exception:
            r = None
        if r is not None:
            live_w = self.batch_file_main_table.cellWidget(r, 2)
            if live_w is not None and live_w is not w:
                widgets_seen += 1
                try:
                    if not old_txt:
                        old_txt = live_w.text().strip()
                    live_w.setText(txt)
                    widgets_written += 1
                except RuntimeError as e:
                    self.log_output.append(
                      f'<span style="color:red;"> COR {base}: live '
                        f'cellWidget setText failed ({e})</span>'
                    )

        if widgets_written == 0:
            self.log_output.append(
              f'<span style="color:red;"> COR {base}: no writable '
                f'widget found (file_info[\'cor_input\']={w!r}, row={r!r}) '
                f'— UI cell NOT updated. AI value was {txt}.</span>'
            )
            return False

        path = file_info.get('path')
        if path:
            self.cor_data[path] = txt

        try:
            unchanged = (old_txt and
                            abs(float(old_txt) - float(txt)) < 1e-6)
        except (ValueError, TypeError):
            unchanged = False
        if unchanged:
            self.log_output.append(
              f'<span style="color:#888;"> COR {base}: AI returned '
                f'the same value ({txt}) as was already in the cell '
                f'[{widgets_written}/{widgets_seen} widget(s) updated]</span>'
            )
        elif old_txt:
            self.log_output.append(
              f'<span style="color:#1a8cff;"> COR {base}: '
                f'{old_txt} → {txt} '
                f'[{widgets_written}/{widgets_seen} widget(s) updated]</span>'
            )
        else:
            self.log_output.append(
              f'<span style="color:#1a8cff;"> COR {base}: '
                f'(empty) → {txt} '
                f'[{widgets_written}/{widgets_seen} widget(s) updated]</span>'
            )
        QApplication.processEvents()
        return True

    # ===== THEME METHODS =====

    def _toggle_theme(self):
        """Toggle between bright and dark themes"""
        self.theme_manager.toggle_theme()

    def _on_theme_changed(self, theme_name):
        """Callback when theme changes - update UI elements"""
        # Update theme toggle button icon
        if theme_name == 'bright':
            self.theme_toggle_btn.setText("")
        else:
            self.theme_toggle_btn.setText("")

        # Update canvas background
        if VISPY_AVAILABLE and hasattr(self, 'canvas'):
            bg_color = 'black' if theme_name == 'dark' else 'white'
            self.canvas.bgcolor = bg_color
            self.canvas.update()
        elif not VISPY_AVAILABLE and hasattr(self, '_pg_layout'):
            self._pg_layout.setBackground('k' if theme_name == 'dark' else 'w')

        # Refresh current image if available
        if self._current_img is not None:
            self.refresh_current_image()

    # ===== MACHINE CONFIGURATION METHODS =====

    def _get_config_path(self):
        """Get path to machine configuration file"""
        config_dir = os.path.expanduser("~/.tomogui")
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "machine_config.json")

    def _load_machine_config(self):
        """Load machine configuration from file"""
        config_path = self._get_config_path()
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load machine config: {e}")
        return {}

    def _save_machine_config(self, config):
        """Save machine configuration to file"""
        config_path = self._get_config_path()
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save machine config: {e}")

    def _populate_machine_list(self):
        """Populate the machine dropdown with Local + configured machines"""
        current = self.batch_machine_box.currentText()
        self.batch_machine_box.clear()

        # Always include Local
        machines = ["Local"]

        # Add configured machines
        if self.machine_config:
            machines.extend(sorted(self.machine_config.keys()))

        self.batch_machine_box.addItems(machines)

        # Restore previous selection if it still exists
        if current and current in machines:
            self.batch_machine_box.setCurrentText(current)
        else:
            self.batch_machine_box.setCurrentText("Local")

    def _open_machine_settings(self):
        """Open the machine settings dialog"""
        dialog = MachineSettingsDialog(self, self.machine_config)
        if dialog.exec_() == QDialog.Accepted:
            self.machine_config = dialog.get_config()
            self._save_machine_config(self.machine_config)
            self._populate_machine_list()  # Refresh the dropdown
            self.log_output.append('<span style="color:green;">Machine settings saved</span>')

    def _open_chatbot(self):
        """Open the Claude-powered chatbot dialog (or raise it if already open)."""
        if self._chatbot_dialog is None:
            self._chatbot_dialog = ChatBotDialog(self, self.theme_manager)
        self._chatbot_dialog.show()
        self._chatbot_dialog.raise_()
        self._chatbot_dialog.activateWindow()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = TomoGUI()
    w.show()
    sys.exit(app.exec_())

