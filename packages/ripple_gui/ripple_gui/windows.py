# gui/windows.py
import os
import numpy as np
import scipy.io
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QFrame, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QGridLayout, QScrollArea, QListWidget, QTabWidget,
    QGroupBox, QFormLayout, QSpinBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
    QMessageBox, QInputDialog, QLineEdit, QCheckBox, QToolBar, QFileDialog
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from math import ceil
from functools import partial
import matplotlib.pyplot as plt
plt.ioff()

class MainWindow(QMainWindow):
# Step 1: Build UI elements
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ripple Analysis GUI")
        self.resize(1200, 800)

        self._setup_ui()
        self._setup_connections()
        self._init_state()
    def _setup_ui(self):
        # Central container and layout
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT LAYOUT: A Wrapper for the purpose of creating a toggle + a left column layout
        self.left_wrapper  = QFrame(self)
        self.left_wrapper.setObjectName("LeftWrapper")
        wrapper_layout = QHBoxLayout(self.left_wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)

        self.sidebar_btn = QPushButton("«", parent = self.left_wrapper)
        self.sidebar_btn.setFixedWidth(25)
        self.sidebar_btn.setToolTip("Hide sidebar")
        wrapper_layout.addWidget(self.sidebar_btn, alignment=Qt.AlignRight)
        self.left  = self._build_left_panel()
        wrapper_layout.addWidget(self.left)
        main_layout.addWidget(self.left_wrapper)

        # RIGHT LAYOUT: QTabWidget
        self.right_tabs = QTabWidget(self)
        self.right_tabs.setMinimumWidth(1000)
        main_layout.addWidget(self.right_tabs, 1)
        main_layout.setStretchFactor(self.right_tabs, 1)
        self._build_right_tabs()
    
    # 1.1 Build Right Tabs
    def _build_right_tabs(self) -> QTabWidget:
        # Tab 0 – Processing Notebook
        self.fig_note   = Figure(figsize=(8, 6))
        self.can_note   = FigureCanvas(self.fig_note)
        self.right_tabs.addTab(self.can_note, "Processing Notebook")
                
        # Tab 1 – Grand Average        
        self.fig_grand_auto   = Figure(figsize=(4, 3))
        self.fig_grand_manual = Figure(figsize=(4, 3))
        self.can_grand_auto   = FigureCanvas(self.fig_grand_auto)
        self.can_grand_manual = FigureCanvas(self.fig_grand_manual)

        # wrap both canvas into one container widget and give the container widget a layout, then throw the container widget into the tab
        grand_page = QWidget()
        vbox = QVBoxLayout(grand_page)
        vbox.addWidget(self.can_grand_auto)
        vbox.addWidget(self.can_grand_manual)
        self.right_tabs.addTab(grand_page, "Grand Average")

        # Tab 2 - All Events Layout
        '''
        QTabWidget (self.right_tabs)
        └─ QWidget  (self.event_widget)  ← the tab page
            └─ QVBoxLayout (eventbox)
                ├─ QToolBar (bar)
                └─ QScrollArea (scroll)
                    └─ QWidget (grid_container)
                        └─ QGridLayout (self.grid_layout)
                        ├─ [row0] QCheckBox, QLabel, Thumbnail
                        ├─ [row1] ...
                        └─ ...
        '''

        self.event_widget = QWidget()
        eventbox       = QVBoxLayout(self.event_widget)

        # 2) scroll area + grid
        scroll = QScrollArea(self.event_widget)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea { background: white; border: none; }
            QScrollArea > QWidget { background: white; }          /* viewport */
        """)

        self.grid_container = QWidget()
        self.grid_container.setStyleSheet("background: white;")  
        self.grid_layout    = QGridLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignTop)
        self.grid_layout.setSpacing(6)

        scroll.setWidget(self.grid_container)   # grid lives *inside* scroll
        eventbox.addWidget(scroll, 1)           # stretch factor = 1
        self.right_tabs.addTab(self.event_widget, "All Events Analysis")

    # 1.2 Build Left Wrapper for the toggle
    def _build_left_panel(self) -> QFrame:
        left = QFrame(self)
        left.setObjectName("LeftColumn")
        left.setMinimumWidth(400)
        left_layout = QVBoxLayout(left)
        
        left_layout.addWidget(self._build_signal_selection(left))
        left_layout.addWidget(self._build_detection_section(left))
        left_layout.addWidget(self._build_tfd_section(left))
        left_layout.addWidget(self._build_rejection_section(left))
        left_layout.addWidget(self._build_visualization_section(left))
        left_layout.addWidget(self._build_save_result_section(left),1)
        return left
    def _build_toggle_sidebar(self):
        visible = self.left.isVisible()
        self.left.setVisible(not visible)
        # keep wrapper narrow when hidden
        if visible:
            self.left_wrapper.setMaximumWidth(self.sidebar_btn.width()+4)
            self.sidebar_btn.setText("»")
            self.sidebar_btn.setToolTip("Show sidebar")
        else:
            self.left_wrapper.setMaximumWidth(16777215)  # Qt’s “no limit”
            self.sidebar_btn.setText("«")
            self.sidebar_btn.setToolTip("Hide sidebar")
    def _build_signal_selection(self, parent) -> QGroupBox:        
        # Section 1: LFP Channel Selection Section
        signal_box = QGroupBox("Stage 1: LFP Signal Selection", parent)
        signal_layout = QFormLayout(signal_box)

        # Input 0: Fs
        self.fs_input = QSpinBox()
        self.fs_input.setRange(800, 1500)
        self.fs_input.setValue(1000)
        signal_layout.addRow("Fs:", self.fs_input)

        # Input 1: Session
        self.session_input = QSpinBox()
        self.session_input.setRange(1, 151)
        signal_layout.addRow("Session:", self.session_input)

        # Input 2: Trial
        self.trial_input = QSpinBox()
        self.trial_input.setRange(1, 10)
        signal_layout.addRow("Trial:", self.trial_input)
        
        # Input 3: Channel
        # 1) Select a single channel or multiple channels
        # 2) Select a brain region or brain regions
        # 3) Select all channels

        # 0. Prepare for Method 2)
        regions = [
            "(r) anterior amygdalar area",
            "(r) anterior cingulate gyrus",
            "(r) caudate nucleus",
            "(r) central amygdalar nucleus",
            "(r) cerebral white matter",
            "(r) frontal white matter",
            "(r) fronto-orbital gyrus",
            "(r) genu of the corpus callosum",
            "(r) internal capsule",
            "(r) lateral globus pallidus",
            "(r) lateral orbital gyrus",
            "(r) medial orbital gyrus",
            "(r) middle frontal gyrus",
            "(r) nucleus accumbens",
            "(r) optic tract",
            "(r) postcentral gyrus",
            "(l) postcentral gyrus",
            "(r) posterior cingulate gyrus",
            "(r) precentral gyrus",
            "(r) precuneus",
            "(r) presubiculum",
            "(r) putamen",
            "(r) superior frontal gyrus",
            "(r) superior parietal lobule",
            "(r) supramarginal gyrus",
            "(r) thalamus"
        ]       
        
        # 1. Create a mode selector
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Select Channel(s)",
            "Select Region(s)",
            "All Channels"
        ])
        signal_layout.addRow("Channel Load Mode:", self.mode_combo)

        # 2. Channel selector (hidden by default)
        self.channel_input = QSpinBox()
        self.channel_input.setRange(1, 220)
        self.channel_input.setVisible(True)
        signal_layout.addRow("Channel:", self.channel_input)

        # 3. Region selector (hidden by default)
        self.region_combo = QComboBox()
        self.region_combo.addItems(regions)
        self.region_combo.setVisible(False)
        signal_layout.addRow("Region:", self.region_combo)
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)

        # Load button
        self.load_btn = QPushButton("Load LFP", parent=signal_box)
        self.load_btn.setProperty("class", "FunctionBtn")
        signal_layout.addRow(self.load_btn)
        return signal_box
    def _on_mode_change(self, text):
        """Show/hide widgets based on load mode."""
        if text == "Select Channel(s)":
            self.channel_input.setVisible(True)
            self.region_combo.setVisible(False)
        elif text == "Select Region(s)":
            self.channel_input.setVisible(False)
            self.region_combo.setVisible(True)
        else:  # "All Channels"
            self.channel_input.setVisible(False)
            self.region_combo.setVisible(False)   
    
    # 1.3 Build Left Control Panel
    def _build_detection_section(self, parent) -> QGroupBox:
        detect_box = QGroupBox("Stage 2: Detection", parent)
        detect_layout = QFormLayout(detect_box)

        # Filter taps
        self.bp_taps_spin = QSpinBox()
        self.bp_taps_spin.setRange(1, 1000)
        self.bp_taps_spin.setValue(550)
        detect_layout.addRow("Filter Taps:", self.bp_taps_spin)

        # BP frequency range (Hz)
        rippleband_hbox = QHBoxLayout()
        self.bp_low_spin = QSpinBox()
        self.bp_low_spin.setRange(1, 1000)
        self.bp_low_spin.setValue(100)
        self.bp_high_spin = QSpinBox()
        self.bp_high_spin.setRange(1, 1000)
        self.bp_high_spin.setValue(140)
        rippleband_hbox.addWidget(self.bp_low_spin)
        rippleband_hbox.addWidget(QLabel("–"))
        rippleband_hbox.addWidget(self.bp_high_spin)
        detect_layout.addRow("BP Range (Hz):", rippleband_hbox)

        # RMS window length (ms)
        self.rms_spin = QDoubleSpinBox()
        self.rms_spin.setRange(1.0, 1000.0)
        self.rms_spin.setValue(20.0)
        detect_layout.addRow("RMS Length (ms):", self.rms_spin)

        # Rejection Threshold setup
        # Criteria 1: Outlier
        self.outlier = QDoubleSpinBox()
        self.outlier.setRange(1.0, 20.0)
        self.outlier.setValue(9.0)
        self.outlier.setDecimals(1)
        self.outlier.setSingleStep(0.5)
        detect_layout.addRow("RMS Remove Outlier:", self.outlier)

        # Criteria 2: RMS bound
        self.lower_bound = QDoubleSpinBox()
        self.lower_bound.setRange(1, 10)
        self.lower_bound.setValue(2.5)
        self.lower_bound.setDecimals(1)
        self.lower_bound.setSingleStep(0.5)
        detect_layout.addRow("RMS Threshold (mu + z * sd):", self.lower_bound)


        # Criteria 3: Minimum duration
        self.min_dur_spin = QDoubleSpinBox()
        self.min_dur_spin.setRange(1, 100)
        self.min_dur_spin.setValue(30)
        self.min_dur_spin.setDecimals(0)
        self.min_dur_spin.setSingleStep(1)
        detect_layout.addRow("Minimum Duration (ms):", self.min_dur_spin)
 

        # Criteria 4: Merge duration
        self.merge_dur_spin = QDoubleSpinBox()
        self.merge_dur_spin.setRange(1, 100)
        self.merge_dur_spin.setValue(10)
        self.merge_dur_spin.setDecimals(0)
        self.merge_dur_spin.setSingleStep(1)
        detect_layout.addRow("Merge Duration (ms):", self.merge_dur_spin)

        # Visualization Window Size
        self.visual_size = QSpinBox()
        self.visual_size.setRange(70, 1000)
        self.visual_size.setValue(200)
        self.visual_size.setSingleStep(1)
        self.visual_size.setSuffix(" ms")       
        detect_layout.addRow("Ripple Analysis Window (± input ms, Ripple Peak Centered):", self.visual_size)

        # Detection button
        self.detect_btn = QPushButton("Detect Ripples", parent=detect_box)
        self.detect_btn.setProperty("class", "FunctionBtn")
        detect_layout.addRow(self.detect_btn)
        return detect_box
    def _build_tfd_section(self, parent) -> QGroupBox:
        tfd_box = QGroupBox("Stage 3: Time-Frequency Decomposition", parent)
        tfd_layout = QFormLayout(tfd_box)
        # Frequency Analysis Range - lower
        self.freq_lower_spin = QSpinBox()
        self.freq_lower_spin.setRange(0, 150)
        self.freq_lower_spin.setValue(2)
        self.freq_lower_spin.setSingleStep(1)
        self.freq_lower_spin.setSuffix(" Hz")

        # Step size / hop (seconds)
        self.freq_upper_spin = QSpinBox()
        self.freq_upper_spin.setRange(151, 500)
        self.freq_upper_spin.setValue(200)
        self.freq_upper_spin.setSingleStep(1)
        self.freq_upper_spin.setSuffix(" Hz")
        self.freq_hbox = QHBoxLayout()
        self.freq_hbox.addWidget(self.freq_lower_spin)
        self.freq_hbox.addWidget(self.freq_upper_spin)
        tfd_layout.addRow("Frequency Analysis Range:", self.freq_hbox)

        # window parameters [Window Size, Step Size]
        # Window size (seconds)
        self.tfd_window_time_spin = QDoubleSpinBox()
        self.tfd_window_time_spin.setRange(0.001, 1.000)
        self.tfd_window_time_spin.setValue(0.060)
        self.tfd_window_time_spin.setDecimals(3)
        self.tfd_window_time_spin.setSingleStep(0.010)
        self.tfd_window_time_spin.setSuffix(" s")

        # Step size / hop (seconds)
        self.tfd_step_time_spin = QDoubleSpinBox()
        self.tfd_step_time_spin.setRange(0.001, 1.000)
        self.tfd_step_time_spin.setValue(0.001)
        self.tfd_step_time_spin.setDecimals(3)
        self.tfd_step_time_spin.setSingleStep(0.001)
        self.tfd_step_time_spin.setSuffix(" s")
        self.tfd_window_hbox = QHBoxLayout()
        self.tfd_window_hbox.addWidget(self.tfd_window_time_spin)
        self.tfd_window_hbox.addWidget(self.tfd_step_time_spin)
        tfd_layout.addRow("Window Parameters [N, Dn] - [Window Time, Step Size]:", self.tfd_window_hbox)

        # time bandwidth product
        self.timebandwidth_spin = QDoubleSpinBox()
        self.timebandwidth_spin.setRange(0, 7)
        self.timebandwidth_spin.setValue(1.2)
        self.timebandwidth_spin.setDecimals(1)
        self.timebandwidth_spin.setSingleStep(0.1)
        tfd_layout.addRow("Time-Bandwidth Product (NW = Window Time ⋅ Half-bandwidth):", self.timebandwidth_spin)
        
         # Number of tapers
        self.taper_spin = QSpinBox()
        self.taper_spin.setRange(0, 7)
        self.taper_spin.setValue(1)
        self.taper_spin.setSingleStep(1)
        tfd_layout.addRow("Number of Tapers (K = 2NW - 1):", self.taper_spin)

        # signal padding to fit the tfspec window
        self.pad_spin = QSpinBox()
        self.pad_spin.setRange(0, 200)
        self.pad_spin.setValue(20)
        self.pad_spin.setSingleStep(1)
        tfd_layout.addRow("Pad Signal:", self.pad_spin)

        # Normalize button
        self.normalize_btn = QPushButton("Normalize Ripples", parent=tfd_box)
        self.normalize_btn.setProperty("class", "FunctionBtn")
        tfd_layout.addRow(self.normalize_btn)
        return tfd_box
    def _build_rejection_section(self, parent) -> QGroupBox:
        rej_box = QGroupBox("Stage 4: Rejection", parent)
        rej_layout = QFormLayout(rej_box)
        self.rej_thresh_spin = QDoubleSpinBox(); self.rej_thresh_spin.setRange(0.0, 10.0); self.rej_thresh_spin.setValue(3.0)
        rej_layout.addRow("Reject thresh (z):", self.rej_thresh_spin)

        # Reject button
        self.reject_btn = QPushButton("Reject Ripples", parent=rej_box)
        self.reject_btn.setProperty("class", "FunctionBtn")
        rej_layout.addRow(self.reject_btn)
        return rej_box
    def _build_visualization_section(self, parent) -> QGroupBox:
        # add things to allow users to select what graphs they want
        vis_box = QGroupBox("Stage 5: Visualization", parent)
        vis_layout = QFormLayout(vis_box)

        self.col_raw    = QCheckBox("Raw LFP")
        self.col_bp     = QCheckBox("BP LFP")
        self.col_spec   = QCheckBox("Spectrogram (Analysis Window)")
        self.col_winsp  = QCheckBox("Spectrum (Analysis Window)")
        self.col_actsp  = QCheckBox("Spectrum (Actual Event Duration)")
        for cb in (self.col_raw, self.col_bp, self.col_spec, self.col_winsp, self.col_actsp):
            cb.setChecked(True)
            vis_layout.addRow(cb)

        # visualize button
        self.vis_btn = QPushButton("Visualize Ripples", parent=vis_box)
        self.vis_btn.setProperty("class", "FunctionBtn")
        vis_layout.addRow(self.vis_btn)
        return vis_box
    def _build_save_result_section(self, parent) -> QGroupBox:        
        save_box = QGroupBox("Stage 6: Save User Selected Results", parent)
        save_layout = QFormLayout(save_box)

        self.status_lbl = QLabel()
        save_layout.addWidget(self.status_lbl)
        self.status_lbl.hide()


        # save button
        self.save_btn = QPushButton("Save Ripples", parent=save_box)
        self.save_btn.setProperty("class", "FunctionBtn")
        save_layout.addRow(self.save_btn)
        return save_box
    
    # 1.4 Connect Buttons with the Analysis Functions and Initialize Variables to save the computation results
    def _setup_connections(self):
        self.sidebar_btn.clicked.connect(self._build_toggle_sidebar)
        self.load_btn.clicked.connect(self.on_load)
        self.detect_btn.clicked.connect(self.on_detect)
        self.normalize_btn.clicked.connect(self.on_normalize)
        self.reject_btn.clicked.connect(self.on_reject)
        # Connect to the control buttons on the left column
        for cb in (self.col_raw, self.col_bp, self.col_spec, self.col_winsp, self.col_actsp):
            cb.stateChanged.connect(lambda _=None: self._populate_event_grid)
        self.vis_btn.clicked.connect(self.on_visualize)
        self.save_btn.clicked.connect(self.on_save)
    def _init_state(self):
        self.lfp = None
        self.det_res = None
        self.norm_res = None
        self.rej_res  = None
        self.avg_res = None


# Step 2: Collect User Input from the UI and call load and analyze functions   
    def on_load(self):
        """
        Load data based on the current mode:
        - "Select Channel(s)" → load one channel via load_electrodes()
        - "Select Region(s)"  → load all region channels via load_region_data()
        - "All Channels"      → load every channel in the session/trial

        Widgets required on self:
        self.mode_combo      (QComboBox)
        self.session_input   (QSpinBox)    # session index
        self.trial_input     (QSpinBox)    # trial number
        self.channel_input   (QSpinBox)    # single-channel index
        self.region_combo    (QComboBox)   # region names
        Side-effects:
        sets self.lfp to an (n_chans×n_samples) np.ndarray
        updates statusBar with a summary message
        """
        from ripple_core.load import load_movie_database, load_electrodes, load_region_data, load_all_channels
        from ripple_core.visualize import plot_processing_notebook
        # make sure to clear all tabs before loading a new event
        # 1) Processing Notebook
        if hasattr(self, "fig_note"):
            self.fig_note.clear()
            self.can_note.draw()

        # 2) Grand Average (both canvases)
        if hasattr(self, "fig_grand_auto"):
            self.fig_grand_auto.clear()
            self.can_grand_auto.draw()
        if hasattr(self, "fig_grand_manual"):
            self.fig_grand_manual.clear()
            self.can_grand_manual.draw()

        # 3) All Events Analysis grid
        if hasattr(self, "grid_layout"):
            # remove every widget in the grid
            while self.grid_layout.count():
                item = self.grid_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        # 1) Read common inputs
        mode      = self.mode_combo.currentText()
        pick_sess = self.session_input.value()
        ch = self.channel_input.value()

        # 2) Fetch session info once
        sessions  = load_movie_database()
        sess      = sessions[pick_sess]
        sess_day  = sess['date']

        if mode == "Select Channel(s)":
            ch = self.channel_input.value()

            while True:
                rec = self.trial_input.value()  # current trial
                try:
                    ch_data = load_electrodes(pick_sess, rec, ch)  # 1D (n_samples,)
                except Exception as e:
                    # trial likely invalid; ask user for another trial and retry
                    trial_new, ok = QInputDialog.getInt(
                        self,
                        "Load error",
                        f"Could not load trial {rec}.\n\n{e}\n\nEnter a different trial:",
                        value=rec
                    )
                    if not ok:
                        return  # user cancelled
                    self.trial_input.setValue(trial_new)
                    continue  # loop and try again
            
                # length check (don’t proceed on weird durations)
                n_samps = ch_data.size
                if n_samps > 390000 or n_samps < 280000:
                    QMessageBox.information(
                        self, "Recording length out of range",
                        f"Loaded LFP has {n_samps:,} samples.\n"
                        "Most recordings in this dataset are ~5 minutes (~300k samples).\n"
                        "Please select a different session or trial."
                    )
                    return  # stop here; user must choose another session/trial

                # success
                self.lfp = ch_data
                self.ch  = ch
                self.statusBar().showMessage(
                    f"Loaded session {sess_day}, trial {str(rec).zfill(3)}, channel {self.ch} "
                    f"({n_samps:,} samples)"
                )
                return  # done

        elif mode == "Select Region(s)":
            # Load all channels in the selected region
            region = self.region_combo.currentText()
            ch_data, _ = load_region_data(pick_sess, rec, region)
            # shape (n_region_chans, n_samples)
            self.lfp = ch_data
            self.statusBar().showMessage(
                f"Loaded {ch_data.shape[0]} channel(s) from region '{region}' "
                f"on {sess_day}, trial {str(rec).zfill(3)}"
            )

        elif mode == "All Channels":
            # Use the dedicated loader for all channels
            data = load_all_channels(pick_sess, rec)
            # load_all_channels returns (n_samples, nCh), so transpose
            self.lfp = data
            msg = (f"Loaded ALL {self.lfp.shape[0]} channels "
                f"on {sess_day}, trial {str(rec).zfill(3)}")

        else:
            raise RuntimeError(f"Unknown load mode: {mode}")  
    
        plot_processing_notebook(
            self.fig_note,
            self.lfp,
            fs=1000,
            # no bp/env/events yet
        )
        self.can_note.draw()

    def on_detect(self):
        from ripple_core.analyze import detect_ripples
        from ripple_core.visualize import plot_processing_notebook

        # from analysis.visualize import plot_processing_notebook
        if self.lfp is None:
            QMessageBox.warning(self, "Error", "Load some data first!")
            return
        # Read parameter controls
        rp_low    = self.bp_low_spin.value()
        rp_high   = self.bp_high_spin.value()
        order     = self.bp_taps_spin.value()
        window_ms = self.rms_spin.value()
        z_low     = self.lower_bound.value()
        z_outlier    = self.outlier.value()
        min_dur   = self.min_dur_spin.value()
        merge_dur = self.merge_dur_spin.value()
        epoch_dur = self.visual_size.value()

        # Call detection
        self.det_res = detect_ripples(
            self.lfp,
            fs=1000,
            rp_band=(rp_low, rp_high),
            order=order,
            window_ms=window_ms,
            z_low=z_low,
            z_outlier=z_outlier,
            min_dur_ms=min_dur,
            merge_dur_ms=merge_dur,
            epoch_ms = epoch_dur
        )

        n_events = len(self.det_res.peak_idx)
        self.statusBar().showMessage(f"Detected {n_events} ripples on channel {self.ch}.")

        plot_processing_notebook(
            self.fig_note,
            self.lfp,
            fs=1000,
            bp_lfp=self.det_res.bp_lfp,
            env_rip=self.det_res.env_rip,
            event_bounds=self.det_res.real_duration,   # (N,2) samples
            fmax=float(self.freq_upper_spin.value()),
        )
        self.can_note.draw()

    def on_normalize(self):
        from ripple_core.analyze import normalize_ripples
        if self.det_res is None:
            QMessageBox.warning(self, "Error", "Run detection first!")
            return
        
         # Read parameter controls
        fmin_read = int(self.freq_lower_spin.value())
        fmax_read = int(self.freq_upper_spin.value())
        if fmax_read <= fmin_read:
            QMessageBox.warning(self, "Bad frequency range", "Upper freq must be > lower freq.")
            raise ValueError("fmax must be > fmin")

        # --- Window + hop (seconds)
        window_size = self.tfd_window_time_spin.value()
        hop = float(self.tfd_step_time_spin.value())
        if window_size <= 0 or hop <= 0:
            QMessageBox.warning(self, "Bad window", "Window and step must be > 0.")
            raise ValueError("nonpositive T/hop")

        # --- Time–bandwidth (NW) and tapers (K)
        NW = float(self.timebandwidth_spin.value())
        tapers_val = int(self.taper_spin.value())
        pad_val = self.pad_spin.value()
        fs = 1000


        # 2) Check against the shortest actual event duration (optional but robust)
        #    (in case you compute spectra on actual ripple duration)
        dur_samps = self.det_res.real_duration[:,1] - self.det_res.real_duration[:,0] + 1
        min_event = int(dur_samps.min())
        if window_size > min_event:
            QMessageBox.warning(
                self, "Window too long for actual events",
                (f"Your time–frequency window ({window_size} samples, {1e3*window_size:.0f} ms) "
                f"exceeds the shortest event ({min_event} samples, {1e3*min_event/fs:.0f} ms).\n\n"
                "Please increase pad your signal or reduce the time-frequency decomposition window length.")
            )
            return

        # 3) Call with the SAME window value you just validated
        try:
            self.norm_res = normalize_ripples(
                self.lfp,
                fs=fs,
                raw_windowed_lfp=self.det_res.raw_windowed_lfp,
                real_duration=self.det_res.real_duration,
                fmin=fmin_read,
                fmax=fmax_read,
                win_length=window_size,
                step=hop,
                nw=NW,
                tapers=tapers_val,
                tfspec_pad=pad_val
            )
        except ValueError as e:
            # 4) No crash; show a friendly error
            QMessageBox.critical(self, "Normalization failed", str(e))
            return

        self.statusBar().showMessage("Normalization complete")
    
    def on_reject(self):
        from ripple_core.analyze import reject_ripples
        if self.norm_res is None:
            QMessageBox.warning(self, "Error", "Normalize first!")
            return
        user_str_thres = self.rej_thresh_spin.value()
        self.rej_res = reject_ripples(
            freq_spec_actual = self.norm_res.freq_spec_actual,
            spec_f = self.norm_res.spec_f,
            mu =  self.det_res.mu,
            sd = self.det_res.sd,
            strict_threshold = user_str_thres,
            env_rip =  self.det_res.env_rip,
            peak_idx = self.det_res.peak_idx,
        )
        # use rej_res.markers to mask out rejected events in plots:
        passed = len(self.rej_res.pass_idx)
        total  = self.norm_res.freq_spec_actual.shape[0]
        self.statusBar().showMessage(f"{passed}/{total} ripples passed rejection")

# Step 3: Visualization on Tabs
    def _toggle_event(self, data_idx, state):
        # Update your model
        accepted = (state == Qt.Checked)
        self.rej_res.markers[data_idx] = (not accepted)

        # Update just this row’s label color
        lbl = self._event_label_by_index.get(data_idx)
        if lbl is not None:
            lbl.setStyleSheet(f"color:{'green' if accepted else 'red'}")

        # Refresh counts
        self.status_lbl.setText(
            f"Events: {(~self.rej_res.markers).sum()} accepted | {self.rej_res.markers.sum()} rejected"
        )
        
    def _populate_event_grid(self, idx_range=None):
        """
        Populate the event grid with checkboxes, IDs, and the selected columns
        (raw/bp LFP, spectrogram, spectra...). Fixes index mismatch by binding
        the true data index (data_idx) when connecting signals.
        """
        from ripple_core.visualize import make_lfp_pix, make_spectrum_pix, make_spec_pix

        if self.det_res is None or self.rej_res is None or self.norm_res is None:
            return

        specs = self.norm_res.normalized_ripple_windowed  # shape: (F, T, N)
        if specs is None:
            return

        # Which rows to show
        if idx_range is None:
            idx_range = range(specs.shape[2])  # raw indices of events

        # ---- clear old cells -----------------------------------------------------
        while self.grid_layout.count():
            it = self.grid_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        # ---- figure out which dynamic columns are visible ------------------------
        epoch_dur     = self.visual_size.value()
        tf_decomp_min = self.freq_lower_spin.value()
        tf_decomp_max = self.freq_upper_spin.value()

        all_cols = [
            ("Raw LFP",                        self.col_raw.isChecked(),
            lambda k: make_lfp_pix(self.det_res.raw_windowed_lfp[k], "#777", epoch_ms=epoch_dur)),
            ("BP LFP",                         self.col_bp.isChecked(),
            lambda k: make_lfp_pix(self.det_res.bp_windowed_lfp[k],  "#d62728", epoch_ms=epoch_dur)),
            ("Spectrogram (Analysis Window)",  self.col_spec.isChecked(),
            lambda k: make_spec_pix(specs[:, :, k], epoch_ms=epoch_dur, fmin=tf_decomp_min, fmax=tf_decomp_max)),
            ("Spectrum (Analysis Window)",     self.col_winsp.isChecked(),
            lambda k: make_spectrum_pix(self.norm_res.freq_spec_windowed[k], "#2ca02c", fmin=tf_decomp_min, fmax=tf_decomp_max)),
            ("Spectrum (Actual Event Duration)", self.col_actsp.isChecked(),
            lambda k: make_spectrum_pix(self.norm_res.freq_spec_actual[k], "#ff7f0e", fmin=tf_decomp_min, fmax=tf_decomp_max)),
        ]
        visible_cols = [(title, fn) for (title, checked, fn) in all_cols if checked]

        # ---- header row ----------------------------------------------------------
        # fixed cols: [0]=checkbox, [1]=ID
        chk_hdr = QLabel("✓")
        chk_hdr.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(chk_hdr, 0, 0)

        id_hdr = QLabel("Event Id")
        id_hdr.setStyleSheet("font-weight:bold;")
        id_hdr.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(id_hdr, 0, 1)

        # allow ID column to stay narrow; dynamic columns stretch
        self.grid_layout.setColumnMinimumWidth(1, 60)
        self.grid_layout.setColumnStretch(1, 0)
        for c in range(2, 2 + len(visible_cols)):
            self.grid_layout.setColumnStretch(c, 1)

        # dynamic headers
        for col, (title, _) in enumerate(visible_cols, start=2):
            lbl = QLabel(title)
            lbl.setStyleSheet("font-weight:bold;")
            lbl.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(lbl, 0, col)

        # ---- rows ----------------------------------------------------------------
        # map: data_idx -> the ID label widget (so we can recolor on toggle)
        self._event_label_by_index = {}

        for row, data_idx in enumerate(idx_range, start=1):
            accepted = not self.rej_res.markers[data_idx]

            # checkbox (bind the true data index, not the loop var by reference)
            cb = QCheckBox()
            cb.setChecked(accepted)
            cb.stateChanged.connect(partial(self._toggle_event, data_idx))
            self.grid_layout.addWidget(cb, row, 0)

            # ID label (store handle so we can recolor just this one later)
            id_lbl = QLabel(f"Ripple {data_idx+1:03d}")
            id_lbl.setStyleSheet(f"color:{'green' if accepted else 'red'}")
            id_lbl.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(id_lbl, row, 1)
            self._event_label_by_index[data_idx] = id_lbl

            # dynamic content cells
            for c, (_, fn) in enumerate(visible_cols, start=2):
                try:
                    pix = fn(data_idx)
                except Exception:
                    # If a single cell fails, keep the grid usable
                    err = QLabel("N/A")
                    err.setAlignment(Qt.AlignCenter)
                    self.grid_layout.addWidget(err, row, c)
                    continue

                pl = QLabel()
                pl.setAlignment(Qt.AlignCenter)
                pl.setPixmap(pix)
                self.grid_layout.addWidget(pl, row, c)

        # ---- status --------------------------------------------------------------
        self.status_lbl.setText(
            f"Events: {(~self.rej_res.markers).sum()} accepted | {self.rej_res.markers.sum()} rejected"
        )
        self.status_lbl.show()

    
    def on_visualize(self):
        from ripple_core.analyze import find_avg
        from ripple_core.visualize import plot_grand_average_grid
        if self.rej_res is None:
            QMessageBox.warning(self, "Error", "Reject first!")
            return
        self.avg_res = find_avg(
            self.rej_res.pass_idx,
            self.rej_res.reject_idx,
            self.det_res.raw_windowed_lfp,
            self.det_res.bp_windowed_lfp,
            self.norm_res.normalized_ripple_windowed,
            self.norm_res.freq_spec_windowed,
            self.norm_res.freq_spec_actual
        )
        self.statusBar().showMessage(f"Averages are saved down for visualization.")

        tf_decomp_min = self.freq_lower_spin.value()
        tf_decomp_max = self.freq_upper_spin.value()

        plot_grand_average_grid(
            grand=self.avg_res,
            spec_f=self.norm_res.spec_f,
            fig=self.fig_grand_auto,
            canvas=self.can_grand_auto,
            epoch_ms=self.visual_size.value(),
            session=self.session_input.value(),
            trial=self.trial_input.value(),
            channel=self.channel_input.value(),
            mode = 'Auto-Detection',
            fmin = tf_decomp_min,
            fmax = tf_decomp_max
        ) 

        self._populate_event_grid()
        self.status_lbl.show()

    def on_save(self):
        from ripple_core.load import UserInput
        from ripple_core.analyze import find_avg
        from ripple_core.visualize import plot_grand_average_grid
        if self.rej_res is None:
            QMessageBox.warning(self, "Error", "Reject and visualize first!")
            return
        # build new windowed arrays based on rej_mask
        mask = ~self.rej_res.markers
        self.user_selected = {
            "user_input": UserInput,
            "raw_windowed": self.det_res.raw_windowed_lfp[mask],
            "bp_windowed":  self.det_res.bp_windowed_lfp[mask],
            "spec_windowed": self.norm_res.normalized_ripple_windowed[:, :, mask],
        }
        #np.savez("user_selected.npz", **self.user_selected)
        path, _ = QFileDialog.getSaveFileName(self, "Save Accepted Ripples", "", "NPZ (*.npz)")
        if path:
            if not path.endswith(".npz"): path += ".npz"
            np.savez(path, **self.user_selected)


        # recompute grand average on-the-fly
        self.grand_res_user = find_avg(
            pass_idx=np.where(mask)[0],
            reject_idx=np.where(~mask)[0],
            raw_windowed_lfp = self.det_res.raw_windowed_lfp,
            bp_windowed_lfp  = self.det_res.bp_windowed_lfp,
            normalized_ripple_windowed = self.norm_res.normalized_ripple_windowed,
            freq_spec_windowed  = self.norm_res.freq_spec_windowed,
            freq_spec_actual    = self.norm_res.freq_spec_actual,
        )

        tf_decomp_min = self.freq_lower_spin.value()
        tf_decomp_max = self.freq_upper_spin.value()

        # plot top (auto) + bottom (manual)
        plot_grand_average_grid(
            grand=self.avg_res,          # automatic
            spec_f=self.norm_res.spec_f,
            fig=self.fig_grand_auto,
            canvas=self.can_grand_auto,
            epoch_ms=self.visual_size.value(),
            session=self.session_input.value(),
            trial=self.trial_input.value(),
            channel=self.channel_input.value(),
            mode='Auto-Detection',
            fmin=tf_decomp_min,
            fmax=tf_decomp_max
        )

        # add new figure row below
        plot_grand_average_grid(
            grand=self.grand_res_user,     # manual
            spec_f=self.norm_res.spec_f,
            fig=self.fig_grand_manual,
            canvas=self.can_grand_manual,
            epoch_ms=self.visual_size.value(),
            session=self.session_input.value(),
            trial=self.trial_input.value(),
            channel=self.channel_input.value(),
            mode='User Manual Selection',
            fmin=tf_decomp_min,
            fmax=tf_decomp_max
        )
        QMessageBox.information(self, "Saved", "Manual selections saved and averages updated.")

