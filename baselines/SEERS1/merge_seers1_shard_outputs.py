import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from util.EvalHelper import evl
from util.FileHelper import DumpJson, LoadJson


def _load_json_if_exists(file_path):
    if not os.path.isfile(file_path):
        return {}
    return LoadJson(file_path, encoding="utf-8")


def _merge_dict_files(shard_dirs, file_name):
    merged = {}
    for shard_dir in shard_dirs:
        file_path = os.path.join(shard_dir, file_name)
        shard_data = _load_json_if_exists(file_path)
        for key, value in shard_data.items():
            merged[key] = value
    return merged


def _build_predict_metrics(predict_fields):
    ordered_items = sorted(predict_fields.items(), key=lambda item: item[0])
    labels = [int(item[1]["label"]) for item in ordered_items]
    predictions = [int(item[1]["prediction"]) for item in ordered_items]
    result = evl(labels, predictions, return_raw=True)
    return {
        "tn": int(result[0]),
        "fp": int(result[1]),
        "fn": int(result[2]),
        "tp": int(result[3]),
        "acc": float(result[4]),
        "prc": float(result[5]),
        "rc": float(result[6]),
        "f1": float(result[7]),
        "auc": float(result[8]),
        "num_patches": int(len(labels))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=8)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive.")

    shard_dirs = []
    for shard_index in range(args.num_shards):
        shard_dir = os.path.join(args.output_root, "shard_{}".format(shard_index))
        if not os.path.isdir(shard_dir):
            raise FileNotFoundError("Missing shard output directory: {}".format(shard_dir))
        shard_dirs.append(shard_dir)

    file_names = [
        "predict_fields.json",
        "predict_process.json",
        "shared_symptoms_fields.json",
        "shared_symptoms_process.json"
    ]

    merged_results = {}
    for file_name in file_names:
        merged = _merge_dict_files(shard_dirs, file_name)
        merged_results[file_name] = merged
        DumpJson(
            merged,
            os.path.join(args.output_root, file_name),
            encoding="utf-8",
            all_ascii=False,
            indent=2
        )
        print("Merged {} entries into {}".format(len(merged), os.path.join(args.output_root, file_name)))

    predict_fields = merged_results.get("predict_fields.json", {})
    if predict_fields:
        metrics = _build_predict_metrics(predict_fields)
        metrics_path = os.path.join(args.output_root, "predict_metrics.json")
        DumpJson(metrics, metrics_path, encoding="utf-8", all_ascii=False, indent=2)
        print("Saved merged metrics to {}".format(metrics_path))


if __name__ == "__main__":
    main()
