import json

import numpy as np
from scipy.signal import find_peaks, welch

from src.preprocessing.pipeline import N_CHANNELS, N_SEGMENTS, PROCESSED_DIR, load_raw, sample_index

MODE1_BAND = (3.7, 4.1)
MODE2_BAND = (4.9, 5.3)


def boundary_jumps(inputs, n=180):
    x = np.asarray(inputs[:n], dtype=np.float64)
    step = np.abs(np.diff(x, axis=2)).mean(axis=(0, 2))
    jumps = np.median(np.abs(x[1:, :, 0] - x[:-1, :, -1]) / step, axis=1)
    within = [jumps[k] for k in range(n - 1) if (k + 1) % N_SEGMENTS != 0]
    across = [jumps[k] for k in range(n - 1) if (k + 1) % N_SEGMENTS == 0]
    return float(np.median(within)), float(np.median(across))


def channel_spectra(inputs, channel, fs=100.0, nperseg=1024):
    x = np.asarray(inputs[:, channel, :], dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    return x, welch(x, fs=fs, nperseg=nperseg, axis=-1)


def neighbour_clustering(log_psd, scenario, setup):
    v = log_psd - log_psd.mean(axis=1, keepdims=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    sim = v @ v.T
    same_record = (scenario[:, None] == scenario[None, :]) & (setup[:, None] == setup[None, :])
    sim[same_record] = -np.inf
    nn = sim.argmax(axis=1)
    return float((scenario[nn] == scenario).mean()), float((setup[nn] == setup).mean())


def analyse():
    inputs, _ = load_raw()
    meta = sample_index()
    scenario, setup = meta["scenario"], meta["setup"]
    within, across = boundary_jumps(inputs)

    ref = np.asarray(inputs[scenario == 0], dtype=np.float64)
    fp, pp = welch(ref - ref.mean(axis=-1, keepdims=True), fs=100.0, nperseg=2048, axis=-1)
    b = (fp > 1) & (fp < 30)
    idx, _ = find_peaks(np.log(pp.mean(axis=(0, 1))[b]), prominence=0.7)
    peaks_hz = np.round(fp[b][idx], 2).tolist()

    rows = []
    for c in range(N_CHANNELS):
        x, (f, p) = channel_spectra(inputs, c)
        band = (f > 0.5) & (f < 30)
        same_scenario, same_setup = neighbour_clustering(np.log(p[:, band]), scenario, setup)
        m1 = p[:, (f >= MODE1_BAND[0]) & (f <= MODE1_BAND[1])].sum(axis=1)
        m2 = p[:, (f >= MODE2_BAND[0]) & (f <= MODE2_BAND[1])].sum(axis=1)
        repeated = (np.abs(np.diff(x, axis=1)) < 1e-12).mean(axis=1)
        rows.append(
            {
                "channel": c,
                "nn_same_setup": round(same_setup, 3),
                "nn_same_scenario": round(same_scenario, 3),
                "mode1_minus_mode2_logpower": round(float(np.log(m1 / m2).mean()), 2),
                "max_repeated_sample_fraction": round(float(repeated.max()), 3),
            }
        )

    rows.sort(key=lambda r: r["nn_same_setup"])
    return {
        "median_jump_within_setup": within,
        "median_jump_across_setup": across,
        "spectral_peaks_hz_scenario0_all_channels": peaks_hz,
        "chance_nn_same_setup": round(16 / 152, 3),
        "chance_nn_same_scenario": round(8 / 152, 3),
        "channels": rows,
    }


def main():
    report = analyse()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "channel_analysis.json").write_text(json.dumps(report, indent=2))
    print(f"median boundary jump within setup: {report['median_jump_within_setup']:.2f}")
    print(f"median boundary jump across setup: {report['median_jump_across_setup']:.2f}")
    print(f"spectral peaks (Hz): {report['spectral_peaks_hz_scenario0_all_channels']}")
    print(f"chance: same setup {report['chance_nn_same_setup']}, same scenario {report['chance_nn_same_scenario']}")
    print("| channel | NN same setup | NN same scenario | mode1 - mode2 log-power | max repeated-sample fraction |")
    print("|---|---|---|---|---|")
    for r in report["channels"]:
        print(
            f"| {r['channel']} | {r['nn_same_setup']} | {r['nn_same_scenario']} | "
            f"{r['mode1_minus_mode2_logpower']} | {r['max_repeated_sample_fraction']} |"
        )


if __name__ == "__main__":
    main()
