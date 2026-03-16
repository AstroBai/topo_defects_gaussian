import glob
import os
import numpy as np


DATA_DIR = "/cephfs/jbai/DefectFisherGaussian"


def to_diff_last_axis(arr):
    if arr.shape[-1] < 2:
        return np.empty((*arr.shape[:-1], 0), dtype=arr.dtype)
    return np.diff(arr, axis=-1)


def convert_dpdf_file(path):
    with np.load(path, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}

        # In GenData.py, TrainDPDFs/hist is generated from cumulative counts
        if "hist" in data.files:
            out["hist_diff"] = to_diff_last_axis(data["hist"])

        for key in data.files:
            if key.endswith("_cum"):
                out[key + "_diff"] = to_diff_last_axis(data[key])

    out_path = path.replace(".npz", "_diff.npz")
    np.savez(out_path, **out)
    print(f"Converted DPDF: {path} -> {out_path}")


def convert_nns_file(path):
    with np.load(path, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}

        # In GenData.py, nn_combined = 16 cumulative blocks concatenated
        if "nn_combined" in data.files:
            nn = data["nn_combined"]
            if nn.ndim == 1 and nn.shape[0] % 16 == 0:
                block = nn.shape[0] // 16
                chunks = [nn[i * block:(i + 1) * block] for i in range(16)]
                diff_chunks = [to_diff_last_axis(chunk) for chunk in chunks]
                out["nn_combined_diff"] = np.hstack(diff_chunks)
            else:
                out["nn_combined_diff"] = to_diff_last_axis(nn)

        for key in data.files:
            if key.endswith("_cum"):
                out[key + "_diff"] = to_diff_last_axis(data[key])

    out_path = path.replace(".npz", "_diff.npz")
    np.savez(out_path, **out)
    print(f"Converted NNs: {path} -> {out_path}")


def main():
    dpdf_files = sorted(glob.glob(f"{DATA_DIR}/TrainDPDFs/*.npz"))
    nns_files = sorted(glob.glob(f"{DATA_DIR}/TrainNNs/*_NNs_combined.npz"))

    print(f"Found {len(dpdf_files)} TrainDPDFs files")
    print(f"Found {len(nns_files)} TrainNNs files")

    for path in dpdf_files:
        if path.endswith("_diff.npz"):
            continue
        convert_dpdf_file(path)

    for path in nns_files:
        if path.endswith("_diff.npz"):
            continue
        convert_nns_file(path)


if __name__ == "__main__":
    main()
