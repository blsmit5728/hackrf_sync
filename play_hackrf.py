import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# === CONFIGURATION ===
filename = "capture.bin"       # Path to your HackRF data file
sample_rate = 20e6             # Hz (match this to your hackrf_transfer -s value)
center_freq = 915e6            # Hz (match this to your hackrf_transfer -f value)

# === LOAD DATA ===
# Read interleaved int8 I/Q samples from file
raw_data = np.fromfile(filename, dtype=np.int8)

# Convert to complex baseband I/Q
iq_data = raw_data[0::2] + 1j * raw_data[1::2]

# Optional: use a smaller chunk if needed
iq_data = iq_data[:2_000_000]  # 200 ms at 10 MS/s

# === COMPUTE SPECTROGRAM ===
# Parameters:
nperseg = 1024         # FFT window length (in samples)
noverlap = 768         # Overlap between windows
nfft = 1024            # FFT size

f, t, Sxx = spectrogram(
    iq_data,
    fs=sample_rate,
    window='hann',
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=nfft,
    detrend=False,
    return_onesided=False,
    scaling='density',
    mode='magnitude'
)

# Convert frequency axis to absolute frequency in MHz
f = np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx, axes=0)
freq_axis_mhz = (center_freq + f) / 1e6

# === PLOT ===
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, freq_axis_mhz, 20 * np.log10(Sxx + 1e-12), shading='auto', cmap='viridis')
plt.ylabel("Frequency (MHz)")
plt.xlabel("Time (s)")
plt.title("Spectrogram of HackRF IQ Capture")
plt.colorbar(label='Power (dB)')
plt.tight_layout()
plt.show()
