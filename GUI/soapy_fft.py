#!/usr/bin/env python3
"""
Basic PyQt/pyqtgraph GUI that streams IQ from a SoapySDR device and plots an FFT in real time.

Requirements (install via pip):
  pip install PyQt5 pyqtgraph numpy SoapySDR

Example device args:
  RTL-SDR:    driver=rtlsdr
  LimeSDR:    driver=lime
  UHD/USRP:   driver=uhd
  PlutoSDR:   driver=plutosdr

Run:
  python soapy_fft_viewer.py
"""

import sys
import time
import numpy as np

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
except Exception as e:
    raise SystemExit("SoapySDR Python bindings not found. Install with 'pip install SoapySDR'.") from e


# ----------------------------- SDR Worker Thread -----------------------------
class SDRWorker(QtCore.QThread):
    fft_ready = QtCore.pyqtSignal(np.ndarray, float, float)  # (dB spectrum, f_start_Hz, f_stop_Hz)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, dev_args: str, center_freq: float, sample_rate: float, gain: float,
                 fft_size: int = 2048, avg_alpha: float = 0.5, chan: int = 0, parent=None):
        super().__init__(parent)
        self.dev_args = dev_args
        self.center_freq = float(center_freq)
        self.sample_rate = float(sample_rate)
        self.gain = float(gain)
        self.fft_size = int(fft_size)
        self.avg_alpha = float(np.clip(avg_alpha, 0.0, 1.0))
        self.chan = int(chan)
        self._running = False
        self._device = None
        self._stream = None

    def stop(self):
        self._running = False

    # Helper for dB scaling with a tiny epsilon
    @staticmethod
    def _pow2db(x):
        return 10.0 * np.log10(np.maximum(x, 1e-16))

    def run(self):
        try:
            self._running = True
            self.status.emit("Creating device: {}".format(self.dev_args or "(auto)"))
            self._device = SoapySDR.Device(self.dev_args)

            # Configure RX
            self._device.setSampleRate(SOAPY_SDR_RX, self.chan, self.sample_rate)
            self._device.setFrequency(SOAPY_SDR_RX, self.chan, self.center_freq)

            # Some drivers expose a single overall gain; others want stage names.
            try:
                self._device.setGain(SOAPY_SDR_RX, self.chan, self.gain)
            except Exception:
                # If overall setGain fails, try setting the first available stage.
                try:
                    stages = self._device.listGains(SOAPY_SDR_RX, self.chan)
                    if stages:
                        self._device.setGain(SOAPY_SDR_RX, self.chan, stages[0], self.gain)
                except Exception:
                    pass

            # Stream setup
            self._stream = self._device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [self.chan])
            self._device.activateStream(self._stream)

            # Choose a block size >= fft_size for decent FFT updates
            block_size = max(self.fft_size, 4096)
            buff = np.empty(block_size, dtype=np.complex64)

            # Window for FFT
            window = np.hanning(self.fft_size).astype(np.float32)

            # Exponential moving average for smoother display
            avg_spec = None

            # Frequency span for plotting (baseband -> absolute freqs)
            f_start = self.center_freq - self.sample_rate / 2.0
            f_stop = self.center_freq + self.sample_rate / 2.0

            self.status.emit("Streaming… sr={:.0f} Hz, fc={:.0f} Hz, gain={:.1f} dB".format(
                self.sample_rate, self.center_freq, self.gain
            ))

            # Main loop
            while self._running:
                # Read samples (blocking with small timeout)
                sr = self._device.readStream(self._stream, [buff], len(buff), timeoutUs=200000)
                if sr.ret > 0:
                    # Use the most recent fft_size samples
                    x = buff[sr.ret - self.fft_size:sr.ret] if sr.ret >= self.fft_size else buff[:sr.ret]
                    if x.size < self.fft_size:
                        # Not enough data yet to form a full FFT
                        continue

                    # Apply window and compute FFT
                    xw = (x[:self.fft_size] * window)
                    X = np.fft.fftshift(np.fft.fft(xw, n=self.fft_size))
                    psd = np.abs(X) ** 2 / np.sum(window**2)

                    # Average
                    if avg_spec is None:
                        avg_spec = psd
                    else:
                        avg_spec = self.avg_alpha * psd + (1.0 - self.avg_alpha) * avg_spec

                    db = self._pow2db(avg_spec)
                    self.fft_ready.emit(db.astype(np.float32), f_start, f_stop)
                else:
                    # Small nap on underflow/timeout to avoid tight loop
                    time.sleep(0.005)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                if self._device and self._stream:
                    self._device.deactivateStream(self._stream)
                    self._device.closeStream(self._stream)
            except Exception:
                pass
            self._stream = None
            self._device = None
            self.status.emit("Stopped")


# ----------------------------- Main Window -----------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SoapySDR FFT Viewer")
        self.resize(1000, 600)

        # Central widget
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        # Controls
        ctl_layout = QtWidgets.QGridLayout()
        row = 0

        self.dev_edit = QtWidgets.QLineEdit("driver=rtlsdr")
        ctl_layout.addWidget(QtWidgets.QLabel("Device args"), row, 0)
        ctl_layout.addWidget(self.dev_edit, row, 1, 1, 3)
        row += 1

        self.freq_edit = QtWidgets.QLineEdit("100e6")
        self.sr_edit = QtWidgets.QLineEdit("2.048e6")
        self.gain_edit = QtWidgets.QLineEdit("30")
        self.fft_edit = QtWidgets.QLineEdit("2048")
        self.avg_edit = QtWidgets.QLineEdit("0.5")

        ctl_layout.addWidget(QtWidgets.QLabel("Center Freq (Hz)"), row, 0)
        ctl_layout.addWidget(self.freq_edit, row, 1)
        ctl_layout.addWidget(QtWidgets.QLabel("Sample Rate (Hz)"), row, 2)
        ctl_layout.addWidget(self.sr_edit, row, 3)
        row += 1

        ctl_layout.addWidget(QtWidgets.QLabel("Gain (dB)"), row, 0)
        ctl_layout.addWidget(self.gain_edit, row, 1)
        ctl_layout.addWidget(QtWidgets.QLabel("FFT Size"), row, 2)
        ctl_layout.addWidget(self.fft_edit, row, 3)
        row += 1

        ctl_layout.addWidget(QtWidgets.QLabel("Averaging (0-1)"), row, 0)
        ctl_layout.addWidget(self.avg_edit, row, 1)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        ctl_layout.addWidget(self.start_btn, row, 2)
        ctl_layout.addWidget(self.stop_btn, row, 3)
        row += 1

        layout.addLayout(ctl_layout)

        # Plot
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'Power', units='dB')
        self.plot.setLabel('bottom', 'Frequency', units='Hz')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=pg.mkPen(width=1))
        layout.addWidget(self.plot, 1)

        # Status bar
        self.sb = self.statusBar()

        # Thread placeholder
        self.worker = None

        # Signals
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)

    def start_stream(self):
        if self.worker is not None:
            return
        try:
            dev_args = self.dev_edit.text().strip()
            fc = float(eval(self.freq_edit.text().strip(), {"__builtins__": {}}, {}))
            sr = float(eval(self.sr_edit.text().strip(), {"__builtins__": {}}, {}))
            gain = float(eval(self.gain_edit.text().strip(), {"__builtins__": {}}, {}))
            fft_size = int(eval(self.fft_edit.text().strip(), {"__builtins__": {}}, {}))
            avg = float(eval(self.avg_edit.text().strip(), {"__builtins__": {}}, {}))
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Invalid input", "Check numeric fields (you can use 100e6 style literals).")
            return

        self.worker = SDRWorker(dev_args, fc, sr, gain, fft_size=fft_size, avg_alpha=avg)
        self.worker.fft_ready.connect(self.update_fft)
        self.worker.status.connect(self.sb.showMessage)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    @QtCore.pyqtSlot()
    def stop_stream(self):
        if self.worker:
            self.worker.stop()

    @QtCore.pyqtSlot()
    def on_finished(self):
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @QtCore.pyqtSlot(str)
    def on_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "SDR Error", msg)

    @QtCore.pyqtSlot(np.ndarray, float, float)
    def update_fft(self, db: np.ndarray, f_start: float, f_stop: float):
        # Build the frequency axis corresponding to the FFT bins
        n = db.size
        freqs = np.linspace(f_start, f_stop, n, dtype=np.float64)
        self.curve.setData(freqs, db)
        self.plot.setXRange(f_start, f_stop, padding=0)


# ----------------------------- Main entry ------------------------------------
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

