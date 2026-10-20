import argparse
import os
import pickle
import hashlib
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from util.EvalHelper import calculate_metrics
from util.FileHelper import XlsxWriterSave


def InferBackboneTag(model_path):
    if model_path in [None, "", "default"]:
        return "Default"
    normalized = str(model_path).replace("\\", "/").lower()
    if "graphcodebert" in normalized:
        return "GraphCodeBERT"
    if "codebert" in normalized:
        return "CodeBERT"
    if "bert" in normalized:
        return "BERT"
    tail = normalized.rstrip("/").split("/")[-1]
    tail = tail.replace("models--", "").replace("--", "_").replace("-", "_")
    tail = "".join(ch for ch in tail if ch.isalnum() or ch == "_").strip("_")
    return tail or "CustomModel"


def BuildDivisionFolder(args):
    if args.baseline == "SEERS2":
        return "{}_{}".format(args.division_mode, InferBackboneTag(args.model_path))
    return args.division_mode


def GetBaselineMethod(baseline_name):
    if baseline_name == "SEERS1":
        from SEERS1.SEERS1Model import SEERS1Model
        return SEERS1Model
    if baseline_name == "SEERS2":
        from SEERS2.SEERS2Model import SEERS2Model
        return SEERS2Model
    return None


def BuildDetailedResultRows(args, method, result, patch_path):
    third_column = result[-1]
    if args.save_probability and args.baseline == "SEERS2":
        scores = getattr(method, "eval_scores", None)
        if scores is None:
            scores = getattr(method, "last_eval_scores", None)
        if scores is None:
            if args.working_mode == "test":
                _, rerun_patch_path = method.Evaluate(working_mode="test")
            else:
                method.Load()
                _, rerun_patch_path = method.Evaluate(working_mode="train")
            scores = getattr(method, "last_eval_scores", None)
            if scores is not None:
                patch_path = rerun_patch_path
                third_column = scores
        if scores is None:
            raise ValueError("Probability scores are not available for {}".format(args.baseline))
        third_column = scores
    return zip(patch_path, result[-2], third_column)


def GetDetailedResultFilename(args, base_name):
    if args.save_probability and args.baseline == "SEERS2":
        stem, ext = os.path.splitext(base_name)
        return "{}_prob{}".format(stem, ext)
    return base_name


def LoadCrossProjectHeldOutProject(dataset_root, split_index):
    meta_path = os.path.join(dataset_root, "split_meta_{}.pkl".format(split_index))
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "rb") as handle:
        meta_info = pickle.load(handle)
    return meta_info.get("held_out_project")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, choices=["SEERS1", "SEERS2"])
    parser.add_argument("--save_target_root", type=str, default=".")
    parser.add_argument("--working_mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--division_mode", type=str, default="group", choices=["fold", "group", "cross_project"])
    parser.add_argument("--dataset", type=str, default="Lin2022TOSEM")
    parser.add_argument("--save_detailed_result", default=False, action="store_true")
    parser.add_argument("--save_probability", default=False, action="store_true")

    parser.add_argument("--model_path", type=str, default="microsoft/codebert-base")
    parser.add_argument("--semantic_fields_path", type=str, default="default")
    parser.add_argument("--lambda_code", type=float, default=0.15)
    parser.add_argument("--lambda_semantic", type=float, default=0.15)
    parser.add_argument("--gate_hidden_size", type=int, default=16)
    parser.add_argument("--run_name", type=str, default="main")

    parser.add_argument("--model", type=str, default="gpt5.2")
    parser.add_argument("--key_file", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--bug_report_path", type=str, default="../resources/clean_d4j12_bug_report.json")
    parser.add_argument("--enhance_info_path", type=str, default="../resources/llm4patchcorrect_d4j12_enhance_info.pkl")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    parser.add_argument("--test_model_index", type=int, default=-1)
    parser.add_argument("--test_dataset_path", type=str, default="default")
    parser.add_argument("--test_dataset", type=str, default="default")
    args = parser.parse_args()

    if args.test_dataset == "default":
        args.test_dataset = args.dataset
    supported_datasets = {"Lin2022TOSEM", "ChatPatch"}
    if args.dataset not in supported_datasets or args.test_dataset not in supported_datasets:
        raise ValueError("Supported datasets are Lin2022TOSEM and ChatPatch.")
    if args.baseline == "SEERS2" and args.dataset == "ChatPatch" and args.division_mode != "fold":
        raise ValueError("ChatPatch only includes 10-fold splits.")
    if args.gate_hidden_size <= 0 or args.lambda_code < 0 or args.lambda_semantic < 0:
        raise ValueError("Invalid gate hidden size or auxiliary loss weights.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("run_name must contain only letters, digits, underscores or hyphens.")
    if args.baseline == "SEERS2":
        if args.semantic_fields_path == "default":
            args.semantic_fields_path = str(ROOT_DIR / "resources" / "seers1_reference_evidence" /
                                            "predict_fields_{}.json".format(args.test_dataset.lower()))
        if not os.path.isfile(args.semantic_fields_path):
            raise FileNotFoundError(args.semantic_fields_path)

    Method = GetBaselineMethod(args.baseline)
    assert Method is not None

    division_folder = BuildDivisionFolder(args)
    save_base = os.path.join(
        args.save_target_root,
        args.baseline,
        "output",
        "train_on_{}".format(args.dataset),
        division_folder
    )

    if args.baseline == "SEERS1":
        signature = "{}_t{}_{}".format(args.model, args.temperature, args.run_name)
        signature = re.sub(r"[^A-Za-z0-9_.-]", "_", signature)
    else:
        evidence_digest = hashlib.sha256(Path(args.semantic_fields_path).read_bytes()).hexdigest()[:12]
        model_digest = hashlib.sha256(str(args.model_path).encode("utf-8")).hexdigest()[:8]
        signature = "lc_{}__ls_{}__hs_{}__model_{}__evidence_{}__{}".format(
            args.lambda_code, args.lambda_semantic, args.gate_hidden_size, model_digest, evidence_digest, args.run_name)
    save_base = os.path.join(save_base, signature)
    print("Output directory:", save_base)

    if args.baseline == "SEERS1":
        if args.num_shards <= 0:
            raise ValueError("num_shards must be positive.")
        if not (0 <= args.shard_index < args.num_shards):
            raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards.")
        save_root = os.path.join(save_base, "shard_{}".format(args.shard_index))
        os.makedirs(save_root, exist_ok=True)
        method = Method(
            save_root,
            model_name=args.model,
            key_file=args.key_file,
            temperature=args.temperature,
            bug_report_path=args.bug_report_path,
            enhance_info_path=args.enhance_info_path,
            dataset_name=args.test_dataset,
            shard_index=args.shard_index,
            num_shards=args.num_shards
        )
        print("-------------- Running {} ---------------".format(args.baseline))
        result, patch_path = method.RunPredictPipeline()
        if args.save_detailed_result:
            XlsxWriterSave(
                os.path.join(save_root, GetDetailedResultFilename(args, "eval_result.xlsx")),
                BuildDetailedResultRows(args, method, result, patch_path)
            )
        return

    if args.division_mode == "cross_project":
        dataset_root = os.path.join(ROOT_DIR, "datasets", "cross_project", args.dataset)
        split_projects_path = os.path.join(dataset_root, "split_projects.pkl")
        assert os.path.isfile(split_projects_path)
        with open(split_projects_path, "rb") as handle:
            split_projects = pickle.load(handle)
        range_num = len(split_projects)
        train_dataset_path_tpl = os.path.join(dataset_root, "patch_path_train_cross_project_{}.pkl")
        valid_dataset_path_tpl = test_dataset_path_tpl = os.path.join(dataset_root, "patch_path_test_cross_project_{}.pkl")
    else:
        dataset_root = os.path.join(ROOT_DIR, "datasets", "10_{}_division".format(args.division_mode), args.dataset)
        assert os.path.exists(dataset_root)
        train_dataset_path_tpl = os.path.join(dataset_root, "patch_path_train_" + args.division_mode + "_{}.pkl")
        valid_dataset_path_tpl = test_dataset_path_tpl = os.path.join(dataset_root, "patch_path_test_" + args.division_mode + "_{}.pkl")
        range_num = 10

    if args.working_mode == "train":
        for index in range(range_num):
            target = Path(save_base) / str(index)
            if target.exists():
                raise FileExistsError("Training output already exists: {}. Use another --run_name or --save_target_root.".format(target))

    results = []
    detailed_outputs = []

    if args.working_mode == "test":
        if args.test_model_index >= 0:
            range_num = 1
        cross_dataset_test_file = None
        if args.test_dataset_path == "default" and args.test_dataset != args.dataset:
            cross_dataset_test_file = os.path.join(ROOT_DIR, "datasets", "{}_all_patches.pkl".format(args.test_dataset))
            assert os.path.isfile(cross_dataset_test_file)
        g = 0
        while g < range_num:
            if range_num == 1:
                g = args.test_model_index
            held_out_project = LoadCrossProjectHeldOutProject(dataset_root, g) if args.division_mode == "cross_project" else None
            if held_out_project is None:
                print("-------------- Running {} {} ---------------".format(args.division_mode, str(g)))
            else:
                print("-------------- Running {} {} (held-out: {}) ---------------".format(args.division_mode, str(g), held_out_project))

            save_root = os.path.join(save_base, str(g))
            train_dataset_path = train_dataset_path_tpl.format(str(g))
            if os.path.isfile(args.test_dataset_path):
                test_data_file = args.test_dataset_path
            elif cross_dataset_test_file is not None:
                test_data_file = cross_dataset_test_file
            else:
                test_data_file = test_dataset_path_tpl.format(str(g))
            print("Test data file:", test_data_file)

            if args.baseline == "SEERS2":
                method = Method(
                    save_root,
                    args.model_path,
                    semantic_fields_path=args.semantic_fields_path,
                    train_data_file=train_dataset_path,
                    test_data_file=test_data_file,
                    dataset_name=args.test_dataset,
                    lambda_code=args.lambda_code, lambda_semantic=args.lambda_semantic,
                    gate_hidden_size=args.gate_hidden_size
                )
            else:
                method = Method(
                    save_root,
                    model_name=args.model,
                    key_file=args.key_file,
                    temperature=args.temperature,
                    bug_report_path=args.bug_report_path,
                    enhance_info_path=args.enhance_info_path,
                    train_data_file=train_dataset_path,
                    test_data_file=test_data_file,
                    dataset_name=args.test_dataset
                )
            result, patch_path = method.RunTestPipeline()
            results.append(list(result))
            if args.save_detailed_result:
                detailed_rows = list(BuildDetailedResultRows(args, method, result, patch_path))
                detailed_outputs.append(detailed_rows)
                result_save_target = os.path.join(save_root, "eval_result", GetDetailedResultFilename(args, "eval_result.xlsx"))
                os.makedirs(os.path.dirname(result_save_target), exist_ok=True)
                XlsxWriterSave(result_save_target, detailed_rows)
            del method
            g += 1
    else:
        for g in range(range_num):
            held_out_project = LoadCrossProjectHeldOutProject(dataset_root, g) if args.division_mode == "cross_project" else None
            if held_out_project is None:
                print("-------------- Running {} {} ---------------".format(args.division_mode, str(g)))
            else:
                print("-------------- Running {} {} (held-out: {}) ---------------".format(args.division_mode, str(g), held_out_project))

            save_root = os.path.join(save_base, str(g))
            os.makedirs(save_root, exist_ok=True)
            train_dataset_path = train_dataset_path_tpl.format(str(g))
            valid_dataset_path = valid_dataset_path_tpl.format(str(g))
            if args.baseline == "SEERS2":
                method = Method(
                    save_root,
                    args.model_path,
                    semantic_fields_path=args.semantic_fields_path,
                    train_data_file=train_dataset_path,
                    valid_data_file=valid_dataset_path,
                    dataset_name=args.dataset,
                    lambda_code=args.lambda_code, lambda_semantic=args.lambda_semantic,
                    gate_hidden_size=args.gate_hidden_size
                )
            else:
                method = Method(
                    save_root,
                    model_name=args.model,
                    key_file=args.key_file,
                    temperature=args.temperature,
                    bug_report_path=args.bug_report_path,
                    enhance_info_path=args.enhance_info_path,
                    train_data_file=train_dataset_path,
                    valid_data_file=valid_dataset_path,
                    dataset_name=args.dataset
                )
            result, patch_path = method.RunTrainPipeline()
            results.append(list(result))
            if args.save_detailed_result:
                detailed_rows = list(BuildDetailedResultRows(args, method, result, patch_path))
                detailed_outputs.append(detailed_rows)
                result_save_target = os.path.join(save_root, "eval_result", GetDetailedResultFilename(args, "eval_result.xlsx"))
                os.makedirs(os.path.dirname(result_save_target), exist_ok=True)
                XlsxWriterSave(result_save_target, detailed_rows)
            del method

    tn = sum([item[0] for item in results])
    fp = sum([item[1] for item in results])
    fn = sum([item[2] for item in results])
    tp = sum([item[3] for item in results])
    auc = sum([item[8] for item in results]) / range_num
    acc, prc, rc, f1 = calculate_metrics(tp, tn, fp, fn)
    print("-------------- Total performance of {} {}(s) ---------------".format(str(range_num), args.division_mode))
    print("TP:", tp)
    print("FP:", fp)
    print("TN:", tn)
    print("FN:", fn)
    print("Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f" % (acc, prc, rc, f1, auc))
    if args.save_detailed_result:
        result_all_save_target = os.path.join(save_base, GetDetailedResultFilename(args, "eval_result_all.xlsx"))
        if os.path.exists(result_all_save_target):
            os.remove(result_all_save_target)
        merged_rows = [row for rows in detailed_outputs for row in rows]
        XlsxWriterSave(result_all_save_target, merged_rows)


if __name__ == "__main__":
    main()
