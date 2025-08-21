#!/usr/bin/env python3
"""
HackRF Sweep GUI – a tiny PyQt5 + pyqtgraph wrapper around `hackrf_sweep`.

Features
- Start/Stop `hackrf_sweep` with GUI controls
- Set start/stop frequency, bin width, samples per bin, LNA/VGA gains
- Live spectrum trace (last completed sweep)
- Live waterfall (scrolling)
- Optional CSV logging of raw `hackrf_sweep` lines

Requirements
    pip install pyqt5 pyqtgraph numpy
    # and HackRF tools installed so `hackrf_sweep` is on PATH

Tested on Linux/macOS/Windows (requires Python 3.8+).

Notes
- This stitches each sweep from multiple printed segments. When the tool
  detects a new cycle (segment start frequency decreased), it finalizes the
  previous sweep row for plotting.
- dBFS levels are displayed as-is; adjust the color scale if your noise
  floor differs from ~-100 dBFS.
"""
import os
import sys
import shlex
import time
import threading
import subprocess
from collections import deque

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


# ----------------------------- Worker Thread ----------------------------- #
class SweepReader(QtCore.QThread):
    segment = QtCore.pyqtSignal(dict)   # {start_hz, end_hz, bin_hz, nsamp, powers(list[float])}
    started_ok = QtCore.pyqtSignal(str) # full command line actually used
    process_error = QtCore.pyqtSignal(str)

    def __init__(self, start_hz, end_hz, cmd_list, log_path=None, parent=None):
        super().__init__(parent)
        self._cmd_list = cmd_list
        self._proc = None
        self._stop = threading.Event()
        self._log_path = log_path
        self._log_fp = None

        self.start_hz = start_hz
        self.stop_hz = end_hz

    def run(self):
        try:
            # Open log file early if requested
            if self._log_path:
                self._log_fp = open(self._log_path, 'a', buffering=1)

            # Launch hackrf_sweep
            # Use text mode for line-by-line reading; bufsize=1 asks for line buffering
            self._proc = subprocess.Popen(
                self._cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
            )
            self.started_ok.emit(' '.join(shlex.quote(p) for p in self._cmd_list))

            # Read stdout continuously

            full_output = []
            full_samples = 0
            samps_rxd = 0

            for raw in self._proc.stdout:
                if self._stop.is_set():
                    break
                line = raw.strip()
                if not line:
                    continue
                if self._log_fp:
                    self._log_fp.write(line + "\n")

                ret_data = self._parse_line(line)
                start = ret_data[0]
                end = ret_data[1]
                bin_hz = ret_data[2]
                nsamp = ret_data[3]
                powers = ret_data[4]
                if start == self.start_hz:
                    # This is the first segment of a new sweep, reset full_output
                    full_output = []
                    for i in powers:
                        full_output.append(i)
                    full_samples += nsamp
                    #print(len(powers), "powers in first segment")
                elif end == self.stop_hz:
                    # This is the last segment of a sweep, finalize it
                    if full_output:
                        # Combine all segments into a single dict
                        # print(full_output)
                        for i in powers:
                            full_output.append(i)
                        #print(len(powers), "powers in last segment")
                        full_samples += nsamp
                        combined = {
                            'start_hz': self.start_hz,
                            'end_hz': self.stop_hz,
                            'bin_hz': bin_hz,
                            'nsamp': full_samples,
                            'powers': full_output, # np.concatenate(full_output).tolist(),
                        }
                        print("Emitting segment: start = {} end = {} nsamp = {} {}".format(
                            combined['start_hz'], combined['end_hz'], combined['nsamp'], len(combined['powers'])))
                        self.segment.emit(combined)
                        full_output = []
                        full_samples = 0
                else:
                    for i in powers:
                        # Append to the full output for this sweep
                        full_output.append(i)
                    full_samples += nsamp
                    #print(len(powers), "powers in mid segment")
                    #full_output.append( X for X in parsed.get('powers', []) )

            # Drain remaining output briefly (optional)
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

            # Capture any error text
            if self._proc:
                err = self._proc.stderr.read() if self._proc.stderr else ''
                rc = self._proc.wait(timeout=1) if self._proc else None
                if rc not in (0, None) and err:
                    self.process_error.emit(err.strip())

        except FileNotFoundError:
            self.process_error.emit("Failed to launch hackrf_sweep. Is it installed and on PATH?")
        except Exception as e:
            # get the line number of the errror
            import traceback
            tb = traceback.format_exc()
            print(f"Exception in SweepReader: {tb}")
        
        
            self.process_error.emit(f"Reader crashed: {e}")
        finally:
            if self._log_fp:
                try:
                    self._log_fp.flush(); self._log_fp.close()
                except Exception:
                    pass

    def stop(self):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    @staticmethod
    def _parse_line(line):
        # Expected CSV fields:
        # date, time, start_hz, end_hz, bin_hz, nsamp, p0, p1, ...
        try:
            parts = line.split(',')
            if len(parts) < 7:
                return None
            # Some builds include an ISO timestamp with millis; we just skip first 2 fields
            start_hz = int(parts[2])
            end_hz   = int(parts[3])
            bin_hz   = float(parts[4])
            nsamp    = float(parts[5])
            # Remaining are powers in dBFS
            powers = [float(x) for x in parts[6:] if x]
            nsamp = len(powers)  # Update nsamp to actual count of powers
            if len(powers) == 0:
                return None
            return start_hz,end_hz,bin_hz,nsamp,powers
        except Exception as e:
            print("Exception parsing line:", line, "Error:", e)
            return None


# ------------------------------- Main Window ----------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HackRF Sweep – Realtime GUI")
        self.resize(1200, 800)

        self.reader = None
        self.last_seg_start = None

        # Will be set on first full sweep
        self.col_bins = None
        self.freq_axis = None

        # Waterfall: keep last N rows (sweeps)
        self.max_rows = 500
        self.rows = deque(maxlen=self.max_rows)  # list of np arrays (powers per bin)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        layout.addLayout(self._build_controls())
        layout.addWidget(self._build_plots(), stretch=1)
        layout.addWidget(self._build_status())

        self._update_command_preview()

        # Sweep timing
        self.sweeps_count = 0
        self.t0 = time.time()

    # --------------------------- UI Builders --------------------------- #
    def _build_controls(self):
        g = QtWidgets.QGridLayout()

        # Frequencies
        self.start_freq = QtWidgets.QDoubleSpinBox(); self.start_freq.setSuffix(" MHz"); self.start_freq.setRange(1, 7250); self.start_freq.setValue(100)
        self.stop_freq  = QtWidgets.QDoubleSpinBox();  self.stop_freq.setSuffix(" MHz");  self.stop_freq.setRange(1, 7250);  self.stop_freq.setValue(6000)

        # Resolution
        self.bin_width = QtWidgets.QSpinBox(); self.bin_width.setSuffix(" Hz"); self.bin_width.setRange(1000, 5_000_000); self.bin_width.setSingleStep(10_000); self.bin_width.setValue(1_000_000)
        self.nsamp = QtWidgets.QSpinBox(); self.nsamp.setRange(1, 1000); self.nsamp.setValue(25)

        # Gains
        self.lna_gain = QtWidgets.QSpinBox(); self.lna_gain.setRange(0, 40); self.lna_gain.setSingleStep(8); self.lna_gain.setValue(16)
        self.vga_gain = QtWidgets.QSpinBox(); self.vga_gain.setRange(0, 62); self.vga_gain.setSingleStep(2); self.vga_gain.setValue(20)

        # Device and extras
        self.dev_index = QtWidgets.QSpinBox(); self.dev_index.setRange(0, 7); self.dev_index.setValue(0)
        self.invert_fft = QtWidgets.QCheckBox("Invert FFT order (-I)")
        self.bias_t = QtWidgets.QCheckBox("Enable bias tee (-b)"); self.bias_t.setChecked(True)  # Default enabled
        self.log_checkbox = QtWidgets.QCheckBox("Log CSV to file…")
        self.log_path = QtWidgets.QLineEdit(); self.log_path.setPlaceholderText("Optional: /path/to/log.csv"); self.log_path.setEnabled(False)
        self.log_checkbox.toggled.connect(self.log_path.setEnabled)

        # Buttons
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_sweep)
        self.btn_stop.clicked.connect(self.stop_sweep)

        # Layout
        r = 0
        g.addWidget(QtWidgets.QLabel("Start freq"), r, 0); g.addWidget(self.start_freq, r, 1)
        g.addWidget(QtWidgets.QLabel("Stop freq"),  r, 2); g.addWidget(self.stop_freq,  r, 3)
        r += 1
        g.addWidget(QtWidgets.QLabel("Bin width (-w)"), r, 0); g.addWidget(self.bin_width, r, 1)
        g.addWidget(QtWidgets.QLabel("Samples/bin (-n)"), r, 2); g.addWidget(self.nsamp, r, 3)
        r += 1
        g.addWidget(QtWidgets.QLabel("LNA gain (-l)"), r, 0); g.addWidget(self.lna_gain, r, 1)
        g.addWidget(QtWidgets.QLabel("VGA gain (-g)"), r, 2); g.addWidget(self.vga_gain, r, 3)
        r += 1
        g.addWidget(QtWidgets.QLabel("Device index (-d)"), r, 0); g.addWidget(self.dev_index, r, 1)
        g.addWidget(self.invert_fft, r, 2, 1, 2)
        g.addWidget(self.bias_t, r, 3)
        r += 1
        g.addWidget(self.log_checkbox, r, 0, 1, 1); g.addWidget(self.log_path, r, 1, 1, 3)
        r += 1
        g.addWidget(self._build_cmd_preview(), r, 0, 1, 4)
        r += 1
        g.addWidget(self.btn_start, r, 2); g.addWidget(self.btn_stop, r, 3)

        # Change handlers to keep command preview fresh
        for w in (self.start_freq, self.stop_freq, self.bin_width, self.nsamp,
                  self.lna_gain, self.vga_gain, self.dev_index, self.invert_fft,self.bias_t,
                  self.log_checkbox, self.log_path):
            if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                w.valueChanged.connect(self._update_command_preview)
            elif isinstance(w, QtWidgets.QLineEdit):
                w.textChanged.connect(self._update_command_preview)
            elif isinstance(w, QtWidgets.QCheckBox):
                w.toggled.connect(self._update_command_preview)

        return g

    def _build_cmd_preview(self):
        self.cmd_preview = QtWidgets.QLabel()
        self.cmd_preview.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.cmd_preview.setStyleSheet("QLabel{font-family:monospace}")
        return self.cmd_preview


    def _on_marker_moved(self):
        """While dragging: update the readout to the nearest FFT bin."""
        if getattr(self, "last_freqs", None) is None or getattr(self, "last_powers", None) is None:
            return
        x = float(self.marker.value())  # current x of vertical InfiniteLine
        # Find nearest bin
        idx = int(np.clip(np.searchsorted(self.last_freqs, x), 1, self.last_freqs.size-1))
        # Choose closer of idx-1 vs idx
        if abs(self.last_freqs[idx] - x) >= abs(self.last_freqs[idx-1] - x):
            idx = idx - 1
        fx = float(self.last_freqs[idx])
        py = float(self.last_powers[idx])
        # Update label at the marker
        self.marker_label.setText(f"{fx/1e6:.3f} MHz\n{py:.1f} dBFS")
        self.marker_label.setPos(fx, py)

    def _on_marker_released(self):
        """When drag ends: snap the marker to the exact nearest bin center."""
        if getattr(self, "last_freqs", None) is None:
            return
        x = float(self.marker.value())
        idx = int(np.clip(np.searchsorted(self.last_freqs, x), 1, self.last_freqs.size-1))
        if abs(self.last_freqs[idx] - x) >= abs(self.last_freqs[idx-1] - x):
            idx = idx - 1
        self.marker.setPos(float(self.last_freqs[idx]))
        # Also refresh the readout to ensure it matches the snapped bin
        self._on_marker_moved()


    def _build_plots(self):
        w = pg.GraphicsLayoutWidget()
        # Set background color (e.g., dark gray)
        pg.setConfigOption('background', "#063E77")  # or any valid color string

        # Spectrum (top)
        self.plot_spectrum = w.addPlot(row=0, col=0)
        self.plot_spectrum.showGrid(x=True, y=True)
        self.plot_spectrum.setYRange(-100, 0)  # dBFS range
        self.plot_spectrum.setLabel('left', 'Power', units='dBFS')
        self.plot_spectrum.setLabel('bottom', 'Frequency', units='Hz')
        self.curve = self.plot_spectrum.plot(pen=pg.mkPen(width=1))
        # Set a color for the spectrum curve (e.g., blue)
        self.curve.setPen(pg.mkPen(color='g', width=1))

        # --- Draggable marker + readout ---
        self.marker = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('y', width=2))
        self.plot_spectrum.addItem(self.marker)

        self.marker_label = pg.TextItem(color='y', anchor=(0, 1))
        self.plot_spectrum.addItem(self.marker_label)

        # Move/update handlers
        self.marker.sigPositionChanged.connect(self._on_marker_moved)
        self.marker.sigPositionChangeFinished.connect(self._on_marker_released)

        

        # Waterfall (bottom)
        self.img = pg.ImageItem()
        self.view_waterfall = w.addViewBox(row=1, col=0)
        self.view_waterfall.addItem(self.img)
        self.view_waterfall.setMouseEnabled(x=True, y=False)
        self.view_waterfall.setAspectLocked(False)

        # Color LUT (grayscale by default). User can tweak levels in UI if desired.
        lut = np.repeat(np.arange(256, dtype=np.ubyte)[:, None], 3, axis=1)
        self.img.setLookupTable(lut)
        self.img.setLevels((-100, 0))

        return w

    def _build_status(self):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("Idle.")
        h.addWidget(self.status)
        h.addStretch(1)
        w.setLayout(h)
        return w

    # --------------------------- Command Build -------------------------- #
    def _build_cmd(self):
        f_start = int(self.start_freq.value())
        f_stop  = int(self.stop_freq.value())
        bw      = int(self.bin_width.value())
        nsamp   = int(self.nsamp.value())
        lna     = int(self.lna_gain.value())
        vga     = int(self.vga_gain.value())
        dev     = int(self.dev_index.value())
        biast   = self.bias_t.isChecked()

        cmd = ["hackrf_sweep",
               "-f", f"{f_start}:{f_stop}",
               "-w", str(bw),
               "-n", str(nsamp),
               "-l", str(lna),
               "-g", str(vga),]
        if biast:
            cmd.append("-p 1")
               #"-d", str(dev)]
        if self.invert_fft.isChecked():
            cmd.append("-I")
        # Print timestamps to help detect cycles – default output already has date/time
        return cmd

    def _update_command_preview(self):
        cmd = self._build_cmd()
        preview = ' '.join(shlex.quote(p) for p in cmd)
        if self.log_checkbox.isChecked() and self.log_path.text().strip():
            preview += f"  # logging to {self.log_path.text().strip()}"
        self.cmd_preview.setText(preview)

    # --------------------------- Start/Stop ----------------------------- #
    def start_sweep(self):
        if self.reader is not None:
            return
        # Reset buffers
        self.rows.clear()
        self.col_bins = None
        self.freq_axis = None
        self.last_seg_start = None
        self.curve.setData([], [])
        self.img.clear()
        self.sweeps_count = 0
        self.t0 = time.time()

        cmd = self._build_cmd()
        log_path = self.log_path.text().strip() if self.log_checkbox.isChecked() and self.log_path.text().strip() else None

        self.reader = SweepReader(int(int(self.start_freq.value()) * 1e6),
                                  int(int(self.stop_freq.value()) * 1e6),
                                  cmd,
                                  log_path=log_path)
        self.reader.segment.connect(self.on_segment)
        self.reader.started_ok.connect(self.on_started_ok)
        self.reader.process_error.connect(self.on_process_error)
        self.reader.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("Starting…")

    def stop_sweep(self):
        if self.reader:
            self.reader.segment.disconnect(self.on_segment)
            self.reader.started_ok.disconnect(self.on_started_ok)
            self.reader.process_error.disconnect(self.on_process_error)
            self.reader.stop()
            self.reader.wait(2000)
            self.reader = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("Stopped.")

    # --------------------------- Data Handling -------------------------- #
    def on_started_ok(self, cmdline):
        self.status.setText(f"Running: {cmdline}")

    def on_process_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "hackrf_sweep error", msg)
        self.stop_sweep()

    def on_segment(self, seg: dict):
        start_hz = seg['start_hz']
        bin_hz = seg['bin_hz']
        end_hz = seg['end_hz']  # Not used, but could be useful for debugging

        print(f"Received segment: start={start_hz} Hz, end={end_hz} Hz, bin={bin_hz} Hz, nsamp={seg['nsamp']}")
        powers = np.array(seg['powers'], dtype=float)
        freqs = start_hz + bin_hz * np.arange(powers.size)
        self.row_powers = powers
        self.row_freqs = freqs
        self._finalize_row()
        # Detect new sweep when the segment start decreases
        # if self.last_seg_start is not None and start_hz < self.last_seg_start:
        #     # Finalize previous sweep row
        #     self._finalize_row()
        # self.last_seg_start = start_hz

        # # Append to an in-progress row cache
        # if not hasattr(self, 'row_freqs'):
        #     self.row_freqs = []
        #     self.row_powers = []
        # # Compute per-bin frequencies for this segment
        # freqs = start_hz + bin_hz * np.arange(powers.size)
        # self.row_freqs.append(freqs)
        # self.row_powers.append(powers)

    def _finalize_row(self):
        if not hasattr(self, 'row_freqs') or len(self.row_freqs) == 0:
            return
        # freqs = np.concatenate(self.row_freqs)
        # powers = np.concatenate(self.row_powers)
        # # Sort by frequency just in case segments arrived out of order
        # order = np.argsort(freqs)
        # freqs = freqs[order]
        # powers = powers[order]
        powers = self.row_powers
        freqs = self.row_freqs

        # Initialize axes/columns on first row
        if self.col_bins is None:
            self.col_bins = powers.size
            self.freq_axis = freqs
        else:
            # If resolution changed mid-run, reset waterfall
            if powers.size != self.col_bins:
                self.rows.clear()
                self.col_bins = powers.size
                self.freq_axis = freqs

        # Update spectrum trace (last complete sweep)
        self.curve.setData(self.freq_axis, powers)

        # Push into waterfall buffer
        self.rows.append(powers)
        img_arr = np.vstack(self.rows)  # shape: (rows, cols)
        # Display newest at bottom; pg.ImageItem expects row-major top-to-bottom
        # We'll simply show it as-is and stretch X to frequency span
        self.img.setImage(img_arr, autoLevels=False)
        # Map X axis to frequency range, Y to number of rows
        if self.freq_axis is not None and self.freq_axis.size > 1:
            f0 = float(self.freq_axis[0]); f1 = float(self.freq_axis[-1])
            rows = img_arr.shape[0]
            self.img.resetTransform()
            self.img.setRect(QtCore.QRectF(f0, 0, f1 - f0, rows))
            self.view_waterfall.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
            self.view_waterfall.setYRange(0, rows)

        # Stats
        self.sweeps_count += 1
        dt = time.time() - self.t0
        rate = self.sweeps_count / dt if dt > 0 else 0.0
        self.status.setText(f"Sweeps: {self.sweeps_count}  |  Rate: {rate:.2f} sweeps/s  |  Bins: {self.col_bins}")

        # Keep a copy of the latest sweep for marker readout
        self.last_freqs = self.freq_axis.copy() if self.freq_axis is not None else None
        self.last_powers = powers.copy()

        # Initialize/refresh marker bounds & default position
        if self.last_freqs is not None and self.last_freqs.size > 1:
            f0 = float(self.last_freqs[0]); f1 = float(self.last_freqs[-1])
            try:
                self.marker.setBounds([f0, f1])
            except Exception:
                pass
            # If the marker hasn't been placed yet, drop it in the middle
            if not hasattr(self, "_marker_initialized") or not self._marker_initialized:
                self.marker.setPos(0.5 * (f0 + f1))
                self._marker_initialized = True

        # Reset row cache
        self.row_freqs = []
        self.row_powers = []

    # --------------------------- Qt Events ------------------------------ #
    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self.stop_sweep()
        finally:
            super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

