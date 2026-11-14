import csv
import io

import numpy as np
from scipy.io import loadmat

from vibra.config import MAX_CHANNELS, MAX_UPLOAD_BYTES

SUPPORTED = (".npy", ".csv", ".txt", ".mat")
PREFERRED_MAT_KEYS = ("data", "signal", "signals", "x", "acc", "accel", "acceleration", "inputs")


class InputError(ValueError):
    def __init__(self, message, code="invalid_input"):
        super().__init__(message)
        self.code = code


def extension(filename):
    name = (filename or "").lower()
    for ext in SUPPORTED:
        if name.endswith(ext):
            return ext
    raise InputError(
        f"Unsupported file type for '{filename}'. Upload a .npy, .csv or .mat file.", "unsupported_type"
    )


def load_npy(raw):
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise InputError(f"Could not read the .npy file: {exc}", "unreadable") from exc
    return arr


def sniff_delimiter(sample):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t ").delimiter
    except csv.Error:
        return ","


def load_csv(raw):
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError("The CSV file is not valid UTF-8 text.", "unreadable") from exc
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise InputError("The CSV file is empty.", "empty")
    delim = sniff_delimiter("\n".join(lines[:20]))
    rows = list(csv.reader(lines, delimiter=delim))
    if delim == " ":
        rows = [[c for c in r if c != ""] for r in rows]
    header = None
    try:
        [float(c) for c in rows[0]]
    except ValueError:
        header, rows = rows[0], rows[1:]
    if not rows:
        raise InputError("The CSV file has a header but no data rows.", "empty")
    width = len(rows[0])
    try:
        arr = np.array([[float(c) for c in r] for r in rows if len(r) == width], dtype=np.float64)
    except ValueError as exc:
        raise InputError(f"The CSV file contains non-numeric values: {exc}", "non_numeric") from exc
    if arr.shape[0] != len(rows):
        raise InputError("Every CSV row must have the same number of columns.", "ragged")
    return arr, header


def load_mat(raw):
    try:
        content = loadmat(io.BytesIO(raw))
    except NotImplementedError as exc:
        raise InputError("MATLAB v7.3 files are not supported. Save with save(..., '-v7') and try again.", "unsupported_type") from exc
    except Exception as exc:
        raise InputError(f"Could not read the .mat file: {exc}", "unreadable") from exc
    arrays = {k: v for k, v in content.items() if not k.startswith("__") and isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number) and v.size > 1}
    if not arrays:
        raise InputError("The .mat file contains no numeric arrays.", "empty")
    for key in PREFERRED_MAT_KEYS:
        if key in arrays:
            return arrays[key], key
    key = max(arrays, key=lambda k: arrays[k].size)
    return arrays[key], key


def time_column(arr, header):
    if arr.shape[1] < 2 or arr.shape[0] < 3:
        return None
    names = [h.strip().lower() for h in header] if header else []
    first = arr[:, 0]
    diffs = np.diff(first)
    named = bool(names) and names[0] in ("t", "time", "time_s", "time (s)", "seconds", "timestamp")
    regular = np.all(diffs > 0) and np.allclose(diffs, diffs[0], rtol=1e-3, atol=1e-9)
    if named or regular:
        return float(1.0 / np.median(diffs)) if np.all(diffs > 0) else None
    return None


def orient(arr):
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise InputError(f"Expected a 1-D signal or a 2-D (channels x samples) array, got shape {tuple(arr.shape)}.", "bad_shape")
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    if arr.shape[0] > MAX_CHANNELS:
        raise InputError(f"Found {arr.shape[0]} channels; at most {MAX_CHANNELS} are supported.", "bad_shape")
    return arr


def parse_upload(filename, raw):
    if not raw:
        raise InputError("The uploaded file is empty.", "empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InputError(f"The file is {len(raw) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB.", "too_large")
    ext = extension(filename)
    fs_detected, source = None, ext
    if ext == ".npy":
        arr = load_npy(raw)
    elif ext == ".mat":
        arr, key = load_mat(raw)
        source = f".mat variable '{key}'"
    else:
        arr, header = load_csv(raw)
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            fs_detected = time_column(arr, header)
            if fs_detected is not None:
                arr = arr[:, 1:]
    if not np.issubdtype(np.asarray(arr).dtype, np.number):
        raise InputError("The file does not contain numeric data.", "non_numeric")
    signal = orient(arr).astype(np.float64)
    if not np.all(np.isfinite(signal)):
        raise InputError("The signal contains NaN or infinite values. Remove or interpolate them first.", "non_finite")
    return signal, {"source": source, "fs_detected": fs_detected}
