from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

from vibra.config import MAX_DURATION_S, MAX_FS, MIN_FS, PLOT_POINTS
from vibra.loaders import InputError

Z24_CHANNEL_SETS = {1: [22], 5: [22, 23, 24, 25, 26], 27: list(range(27))}


def channel_labels(n_rows):
    if n_rows not in Z24_CHANNEL_SETS:
        raise InputError(
            f"Found {n_rows} channels. Provide 1 channel (the reference accelerometer, Z24 channel 22), "
            "5 channels (Z24 reference channels 22-26) or all 27 Z24 channels.",
            "bad_channels",
        )
    return Z24_CHANNEL_SETS[n_rows]


def resample(signal, fs, target):
    if abs(fs - target) < 1e-6:
        return signal
    frac = Fraction(target / fs).limit_denominator(1000)
    return resample_poly(signal, frac.numerator, frac.denominator, axis=-1)


def validate(signal, fs, cfg):
    if not (MIN_FS <= fs <= MAX_FS):
        raise InputError(f"Sampling rate {fs:g} Hz is outside the supported range {MIN_FS:g}-{MAX_FS:g} Hz.", "bad_fs")
    duration = signal.shape[-1] / fs
    min_s = cfg["window"] / cfg["fs"]
    if duration < min_s:
        raise InputError(f"The recording is {duration:.1f} s long; at least {min_s:.0f} s ({cfg['window']} samples at {cfg['fs']:g} Hz) is needed.", "too_short")
    if duration > MAX_DURATION_S:
        raise InputError(f"The recording is {duration:.0f} s long; the limit is {MAX_DURATION_S:.0f} s.", "too_long")
    if np.any(signal.std(axis=-1) < 1e-12):
        raise InputError("At least one channel is constant (zero variance). Check the sensor data.", "flat_signal")


def bandpass(signal, cfg):
    sos = butter(cfg["filter_order"], cfg["band"], btype="bandpass", fs=cfg["fs"], output="sos")
    centred = signal - signal.mean(axis=-1, keepdims=True)
    return sosfiltfilt(sos, centred, axis=-1)


def windows(filtered, cfg):
    length = filtered.shape[-1]
    starts = np.arange(0, length - cfg["window"] + 1, cfg["stride"])
    w = np.stack([filtered[:, s : s + cfg["window"]] for s in starts])
    mu = w.mean(axis=-1, keepdims=True)
    sd = w.std(axis=-1, keepdims=True)
    return ((w - mu) / (sd + 1e-8)).astype(np.float32), starts


def prepare(signal, fs, cfg, already_filtered=False):
    validate(signal, fs, cfg)
    x = resample(signal, fs, cfg["fs"])
    filtered = x if already_filtered else bandpass(x, cfg)
    w, starts = windows(filtered, cfg)
    return filtered, w, starts


def plot_series(trace, fs):
    step = max(1, int(np.ceil(len(trace) / PLOT_POINTS)))
    y = trace[::step]
    t = np.arange(len(trace))[::step] / fs
    return {"t": np.round(t, 3).tolist(), "y": np.round(y, 5).tolist(), "fs": fs}


def spectrum(trace, fs, fmax=30.0):
    nper = min(len(trace), 2048)
    f, p = welch(trace, fs=fs, nperseg=nper)
    keep = (f >= 0.5) & (f <= fmax)
    f, p = f[keep], p[keep]
    band = (f >= 3.0) & (f <= 4.6)
    peak = float(f[band][np.argmax(p[band])]) if band.any() else None
    logp = 10 * np.log10(p / p.max() + 1e-12)
    return {"f": np.round(f, 3).tolist(), "db": np.round(logp, 2).tolist(), "first_mode_hz": peak}
