import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate


def load_iq(filename, dtype=np.int8):
    raw = np.fromfile(filename, dtype=dtype)
    iq = raw[::2] + 1j * raw[1::2]
    return iq

iq1 = load_iq("hackrf1_output.bin")
iq2 = load_iq("hackrf1_output.bin")


# Optionally normalize
# iq1 = iq1 / np.abs(iq1).max()
# iq2 = iq2 / np.abs(iq2).max()

correlation = correlate(iq1[:10000], iq2[:10000], mode='full')
lags = np.arange(-len(iq1[:10000])+1, len(iq2[:10000]))

# Find the lag with max correlation
max_lag = lags[np.argmax(np.abs(correlation))]
print(f"Max correlation lag: {max_lag}")

plt.plot(lags, np.abs(correlation))
plt.title("Cross-correlation of IQ Samples")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.grid()
plt.show()