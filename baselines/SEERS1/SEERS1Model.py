import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from util.EvalHelper import evl, GetPredictCorrectness
from util.FileHelper import LoadPickleFile, LoadJson, DumpJson
from util.ModelEngine import ModelEngine
from util.PatchHelper import ExtractPatchInfo


def _resolve_path(path_value, fallback_relative=None):
    candidates = []
    if path_value not in [None, "", "default"]:
        raw_path = Path(path_value)
        candidates.append(raw_path)
        if not raw_path.is_absolute():
            candidates.append(ROOT_DIR / raw_path)
            candidates.append(ROOT_DIR / "baselines" / raw_path)
    if fallback_relative:
        fallback_path = Path(fallback_relative)
        candidates.append(ROOT_DIR / fallback_path)
        candidates.append(ROOT_DIR / "baselines" / fallback_path)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if os.path.isfile(candidate_str):
            return candidate_str
    return str(candidates[-1].resolve()) if candidates else None


def _truncate_text(text, max_chars):
    if text is None:
        return "NONE"
    text = str(text).strip()
    if text == "":
        return "NONE"
    if len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head - len("\n...\n")
    return text[:keep_head] + "\n...\n" + text[-keep_tail:]


def _normalize_none(text):
    if text is None:
        return "NONE"
    text = str(text).strip()
    return text if text else "NONE"


def _parse_kv_fields(raw_text, field_names):
    result = {field: "NONE" for field in field_names}
    if raw_text is None:
        return result
    lines = [line.rstrip() for line in str(raw_text).splitlines()]
    for line in lines:
        for field in field_names:
            prefix = field + ":"
            if line.lower().startswith(prefix.lower()):
                result[field] = _normalize_none(line[len(prefix):].strip())
                break
    return result


def _build_patch_diff(patch_text):
    return _truncate_text(patch_text, 12000)


def _format_bug_report(bug_report):
    if bug_report is None:
        return "Title: NONE\nDescription: NONE"
    if isinstance(bug_report, list):
        title = bug_report[0] if len(bug_report) > 0 else "NONE"
        desc = bug_report[1] if len(bug_report) > 1 else "NONE"
    else:
        title = str(bug_report)
        desc = "NONE"
    return "Title: {}\nDescription: {}".format(_normalize_none(title), _normalize_none(desc))


def _format_test_info(test_infos, max_tests=2, max_trace_lines=20):
    if not test_infos:
        return "NONE"
    chunks = []
    for idx, item in enumerate(test_infos[:max_tests], start=1):
        failing_test = _truncate_text(item[0], 2500)
        failure_message = _truncate_text(item[1], 1000)
        trace_lines = str(item[2]).splitlines()[:max_trace_lines]
        trace_text = _truncate_text("\n".join(trace_lines), 2500)
        chunks.append(
            "[Failing Test {}]\nTest Code:\n{}\nFailure Message:\n{}\nStack Trace:\n{}".format(
                idx, failing_test, failure_message, trace_text
            )
        )
    return "\n\n".join(chunks)


def build_symptom_prompt(bug_report_text, test_info_text):
    return """You are a bug symptom summarizer for automated patch correctness assessment.

Your task is to summarize the bug symptom from incomplete debugging artifacts.
Do not discuss patch correctness. Do not mention whether any patch is correct or incorrect.

Output exactly these fields:
failure_trigger:
observed_behavior:
expected_behavior:
relevant_location:
symptom_summary:

Rules:
1. One sentence per field.
2. Use concise technical language.
3. Output NONE if unavailable.

[Bug Report]
{bug_report}

[Failing Test and Stack Trace]
{test_info}
""".format(
        bug_report=bug_report_text,
        test_info=test_info_text
    )


def build_evidence_prompt(symptom_text, patch_diff):
    return """You are an evidence extractor for automated patch correctness assessment.

Extract structured semantic evidence from the patch.
Do not judge whether the patch is correct or incorrect.

Output exactly these fields:

core_edit_1:
core_edit_2:
core_edit_3:
expected_effect:
symptom_alignment:

Field definitions:
- core_edit_1/2/3: concise semantic abstractions of the main modification actions, focusing on what the patch changes in behavior-relevant logic rather than quoting the exact changed code; order them by importance; if fewer than three exist, output NONE for the unused fields.
- expected_effect: the expected logical or runtime effect of the main modification(s).
- symptom_alignment: how the main modification(s) relate to the given failing symptom.

Strict rules:
1. Each field must contain at most one sentence.
2. Use concise technical language.
3. If a field is not applicable, output NONE.
4. Do not output JSON, markdown, or extra explanation.
5. Do not quote the exact changed code unless necessary.
6. Focus on local modification actions and their behavioral roles.
7. Do not use verdict-like words such as correct, incorrect, overfitting, likely, probably, confidence, or verdict.
8. symptom_alignment must describe the relation between the modification and the symptom, not whether the patch fully fixes the bug.

Example:

[Bug Symptom]
The program mishandles the boundary input 0 and enters the wrong branch, causing an invalid result.

[Patch Diff]
--- Example.java
+++ Example.java
@@ -10,5 +10,5 @@
-if (x > 0) {{
+if (x >= 0) {{
    return processPositive(x);
}} else {{
    return handleNegative(x);

Output:
core_edit_1: Expands the branch predicate to include the previously excluded boundary input.
core_edit_2: NONE
core_edit_3: NONE
expected_effect: The boundary value 0 is redirected to the non-negative processing path instead of the negative-handling path.
symptom_alignment: The modification directly targets the incorrect branch selection reported for the boundary input.

Now process the following input.

[Bug Symptom]
{symptom}

[Patch Diff]
{patch_diff}
""".format(symptom=symptom_text, patch_diff=patch_diff)


def build_correctness_prompt(symptom_fields, evidence_fields, patch_diff):
    symptom_block = "symptom_summary: {}".format(symptom_fields.get("symptom_summary", "NONE"))
    evidence_block = "\n".join([
        "core_edit_1: {}".format(evidence_fields.get("core_edit_1", "NONE")),
        "core_edit_2: {}".format(evidence_fields.get("core_edit_2", "NONE")),
        "core_edit_3: {}".format(evidence_fields.get("core_edit_3", "NONE")),
        "expected_effect: {}".format(evidence_fields.get("expected_effect", "NONE")),
        "symptom_alignment: {}".format(evidence_fields.get("symptom_alignment", "NONE"))
    ])
    return """You are an expert reviewer for automated patch correctness assessment.

Decide whether the patch truly fixes the defect according to developer intent rather than only passing limited tests.

You will be given:
1. a concise bug symptom summary,
2. structured semantic evidence,
3. the patch diff.

Output exactly:
reasoning:
answer:

Rules:
1. reasoning must be at most two concise sentences.
2. answer must be exactly one of:
   It was correct
   It was wrong
3. Do not output anything else.

[Bug Symptom]
{symptom_block}

[Semantic Evidence]
{evidence_block}

[Patch Diff]
{patch_diff}
""".format(
        symptom_block=symptom_block,
        evidence_block=evidence_block,
        patch_diff=patch_diff
    )


class SEERS1Model(object):
    def __init__(self, save_root, model_name="gpt5.2", key_file="default", temperature=0.2,
                 bug_report_path="default", enhance_info_path="default",
                 train_data_file="default", valid_data_file="default", test_data_file="default",
                 dataset_name="Lin2022TOSEM", overwrite_cache=False,
                 shard_index=0, num_shards=1):
        self.save_root = save_root
        self.model_name = model_name
        self.key_file = key_file
        self.temperature = temperature
        self.dataset_name = dataset_name
        self.overwrite_cache = overwrite_cache
        self.shard_index = shard_index
        self.num_shards = num_shards

        self.train_data_file = _resolve_path(train_data_file, "datasets/patch_path_train_group_0.pkl")
        self.valid_data_file = _resolve_path(valid_data_file, "datasets/patch_path_test_group_0.pkl")
        self.test_data_file = _resolve_path(test_data_file, "datasets/patch_path_test_group_0.pkl")
        self.bug_report_path = _resolve_path(bug_report_path, "resources/clean_d4j12_bug_report.json")
        self.enhance_info_path = _resolve_path(
            enhance_info_path,
            "resources/llm4patchcorrect_d4j12_enhance_info.pkl"
        )
        self.key_file = _resolve_path(key_file) if model_name != "starcoder-7b" else key_file

        if model_name != "starcoder-7b" and not os.path.isfile(self.key_file):
            raise FileNotFoundError("A valid key_file is required for close-source chat models.")

        args = SimpleNamespace(model=model_name, key_file=self.key_file, temperature=temperature)
        self.engine = ModelEngine(args)

        self.bug_reports = LoadJson(self.bug_report_path, encoding="utf-8")
        self.enhance_info = LoadPickleFile(self.enhance_info_path)

        self.train_patches = None
        self.train_labels = None
        self.valid_patches = None
        self.valid_labels = None
        self.test_patches = None
        self.test_labels = None
        self.predict_patches = None
        self.predict_labels = None

        os.makedirs(self.save_root, exist_ok=True)

    def _slice_for_shard(self, patches, labels):
        assert self.num_shards >= 1
        assert 0 <= self.shard_index < self.num_shards
        total = len(patches)
        base = total // self.num_shards
        rem = total % self.num_shards
        start = self.shard_index * base + min(self.shard_index, rem)
        end = start + base + (1 if self.shard_index < rem else 0)
        return patches[start:end], labels[start:end], start, end

    def _load_split(self, file_path):
        patches, labels = LoadPickleFile(file_path)
        return patches, labels

    def PrepareData(self, train=True, test=True):
        if train:
            self.train_patches, self.train_labels = self._load_split(self.train_data_file)
            self.valid_patches, self.valid_labels = self._load_split(self.valid_data_file)
            print("Train Set:", len(self.train_labels))
            print("Valid Set:", len(self.valid_labels))
        if test:
            self.test_patches, self.test_labels = self._load_split(self.test_data_file)
            print("Test Set:", len(self.test_labels))

    def PreparePredictData(self):
        predict_file = os.path.join(ROOT_DIR, "datasets", "{}_all_patches.pkl".format(self.dataset_name))
        if not os.path.isfile(predict_file):
            raise FileNotFoundError("Cannot find dataset file: {}".format(predict_file))
        all_patches, all_labels = self._load_split(predict_file)
        self.predict_patches, self.predict_labels, start, end = self._slice_for_shard(all_patches, all_labels)
        print("Predict Shard: {}/{} [{}:{})".format(self.shard_index, self.num_shards, start, end))
        print("Predict Set:", len(self.predict_labels))

    def _get_patch_metadata(self, patch_path):
        correctness, proj, bug_id, tool = ExtractPatchInfo(patch_path, self.dataset_name)
        defect_key = "{}-{}".format(proj, bug_id)
        bug_key = defect_key.lower()
        return {
            "correctness": correctness,
            "project": proj,
            "bug_id": bug_id,
            "tool": tool,
            "defect_key": defect_key,
            "bug_key": bug_key
        }

    def _build_artifact_bundle(self, patch_path, patch_text):
        meta = self._get_patch_metadata(patch_path)
        bug_report = self.bug_reports.get(meta["bug_key"], ["NONE", "NONE"])
        enhance = self.enhance_info.get(meta["defect_key"], [[], "NONE"])
        test_infos = enhance[0] if isinstance(enhance, list) and len(enhance) > 0 else []
        coverage_info = enhance[1] if isinstance(enhance, list) and len(enhance) > 1 else "NONE"
        return {
            "meta": meta,
            "bug_report_text": _format_bug_report(bug_report),
            "test_info_text": _format_test_info(test_infos),
            "patch_diff": _build_patch_diff(patch_text)
        }

    def _predict(self, prompt):
        return self.engine.predict(prompt)

    def _predict_with_tokens(self, prompt):
        if self.model_name == "starcoder-7b":
            response = self.engine.predict(prompt)
            return response, {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
        response, consumed = self.engine.predict(prompt, return_consumed_tokens=True)
        return response, {
            "prompt_tokens": consumed[0],
            "completion_tokens": consumed[1],
            "total_tokens": consumed[2]
        }

    def _load_cache(self, cache_path):
        if os.path.isfile(cache_path) and not self.overwrite_cache:
            return LoadJson(cache_path, encoding="utf-8")
        return {}

    def _save_cache(self, cache_path, cache_obj):
        DumpJson(cache_obj, cache_path, encoding="utf-8", all_ascii=False, indent=2)

    def _load_split_caches(self, split_name):
        fields_path = os.path.join(self.save_root, "{}_fields.json".format(split_name))
        process_path = os.path.join(self.save_root, "{}_process.json".format(split_name))
        return fields_path, process_path, self._load_cache(fields_path), self._load_cache(process_path)

    def _load_symptom_caches(self):
        fields_path = os.path.join(self.save_root, "shared_symptoms_fields.json")
        process_path = os.path.join(self.save_root, "shared_symptoms_process.json")
        return fields_path, process_path, self._load_cache(fields_path), self._load_cache(process_path)

    def _get_or_create_symptom(self, artifact, symptom_cache_path):
        defect_key = artifact["meta"]["defect_key"]
        fields_path, process_path, fields_cache, process_cache = self._load_symptom_caches()
        if defect_key in fields_cache and not self.overwrite_cache:
            return {
                "fields": fields_cache[defect_key]["fields"],
                "raw_response": process_cache.get(defect_key, {}).get("raw_response", ""),
                "token_usage": process_cache.get(defect_key, {}).get("token_usage", {
                    "prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1
                }),
                "prompt": process_cache.get(defect_key, {}).get("prompt", "")
            }

        prompt = build_symptom_prompt(
            artifact["bug_report_text"],
            artifact["test_info_text"]
        )
        raw_response, token_usage = self._predict_with_tokens(prompt)
        fields = _parse_kv_fields(raw_response, [
            "failure_trigger", "observed_behavior", "expected_behavior",
            "relevant_location", "symptom_summary"
        ])
        fields_cache[defect_key] = {
            "defect_key": defect_key,
            "project": artifact["meta"]["project"],
            "bug_id": artifact["meta"]["bug_id"],
            "fields": fields
        }
        process_cache[defect_key] = {
            "defect_key": defect_key,
            "project": artifact["meta"]["project"],
            "bug_id": artifact["meta"]["bug_id"],
            "prompt": prompt,
            "raw_response": raw_response,
            "token_usage": token_usage,
            "fields": fields
        }
        self._save_cache(fields_path, fields_cache)
        self._save_cache(process_path, process_cache)
        return {
            "fields": fields,
            "raw_response": raw_response,
            "token_usage": token_usage,
            "prompt": prompt
        }

    def _build_field_record(self, patch, label, symptom_info, evidence_info, correctness_raw, pred):
        return {
            "patch_path": patch[0],
            "label": int(label),
            "prediction": int(pred),
            "prediction_text": "correct" if pred == 1 else "wrong",
            "symptom_fields": symptom_info["fields"],
            "evidence_fields": evidence_info["fields"],
            "correctness_output": _normalize_none(correctness_raw)
        }

    def _build_process_record(self, patch, label, symptom_prompt, symptom_info,
                              evidence_prompt, evidence_info,
                              correctness_prompt, correctness_raw, correctness_token_usage, pred):
        return {
            "patch_path": patch[0],
            "label": int(label),
            "prediction": int(pred),
            "prediction_text": "correct" if pred == 1 else "wrong",
            "symptom_prompt": symptom_prompt,
            "symptom_raw_response": symptom_info["raw_response"],
            "symptom_token_usage": symptom_info.get("token_usage", {
                "prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1
            }),
            "symptom_fields": symptom_info["fields"],
            "evidence_prompt": evidence_prompt,
            "evidence_raw_response": evidence_info["raw_response"],
            "evidence_token_usage": evidence_info.get("token_usage", {
                "prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1
            }),
            "evidence_fields": evidence_info["fields"],
            "correctness_prompt": correctness_prompt,
            "correctness_raw_response": correctness_raw,
            "correctness_token_usage": correctness_token_usage,
            "correctness_reasoning": self._parse_correctness_reasoning(correctness_raw),
            "token_usage_total": {
                "prompt_tokens": symptom_info.get("token_usage", {}).get("prompt_tokens", -1)
                                 + evidence_info.get("token_usage", {}).get("prompt_tokens", -1)
                                 + correctness_token_usage.get("prompt_tokens", -1),
                "completion_tokens": symptom_info.get("token_usage", {}).get("completion_tokens", -1)
                                     + evidence_info.get("token_usage", {}).get("completion_tokens", -1)
                                     + correctness_token_usage.get("completion_tokens", -1),
                "total_tokens": symptom_info.get("token_usage", {}).get("total_tokens", -1)
                                + evidence_info.get("token_usage", {}).get("total_tokens", -1)
                                + correctness_token_usage.get("total_tokens", -1)
            }
        }

    @staticmethod
    def _parse_correctness_reasoning(raw_text):
        if raw_text is None:
            return ""
        lines = [line.rstrip() for line in str(raw_text).splitlines()]
        for line in lines:
            if line.lower().startswith("reasoning:"):
                return line[len("reasoning:"):].strip()
        return ""

    def _run_on_split(self, patches, labels, split_name):
        fields_path, process_path, field_cache, process_cache = self._load_split_caches(split_name)
        symptom_cache_path = os.path.join(self.save_root, "shared_symptoms_fields.json")
        predictions = []
        eval_patch_paths = []

        for patch, label in tqdm(list(zip(patches, labels)), desc="Running {}".format(split_name)):
            patch_path, patch_text = patch
            eval_patch_paths.append(patch_path)
            if patch_path in field_cache and not self.overwrite_cache:
                predictions.append(int(field_cache[patch_path]["prediction"]))
                continue

            artifact = self._build_artifact_bundle(patch_path, patch_text)
            symptom_info = self._get_or_create_symptom(artifact, symptom_cache_path)
            symptom_prompt = symptom_info.get("prompt", build_symptom_prompt(
                artifact["bug_report_text"],
                artifact["test_info_text"]
            ))
            evidence_prompt = build_evidence_prompt(
                symptom_info["fields"].get("symptom_summary", "NONE"),
                artifact["patch_diff"]
            )
            evidence_raw, evidence_token_usage = self._predict_with_tokens(evidence_prompt)
            evidence_fields = _parse_kv_fields(evidence_raw, [
                "core_edit_1", "core_edit_2", "core_edit_3", "expected_effect",
                "symptom_alignment"
            ])
            evidence_info = {
                "raw_response": evidence_raw,
                "token_usage": evidence_token_usage,
                "fields": evidence_fields
            }

            correctness_prompt = build_correctness_prompt(
                symptom_info["fields"],
                evidence_fields,
                artifact["patch_diff"]
            )
            correctness_raw, correctness_token_usage = self._predict_with_tokens(correctness_prompt)
            pred = GetPredictCorrectness(correctness_raw)
            if pred == -1:
                # conservative fallback: ask for a strict answer only
                fallback_prompt = correctness_prompt + "\nRemember: answer must be exactly 'It was correct' or 'It was wrong'."
                correctness_raw, correctness_token_usage = self._predict_with_tokens(fallback_prompt)
                pred = GetPredictCorrectness(correctness_raw)
            if pred == -1:
                pred = 0

            predictions.append(pred)
            field_cache[patch_path] = self._build_field_record(
                patch, label, symptom_info, evidence_info, correctness_raw, pred
            )
            process_cache[patch_path] = self._build_process_record(
                patch, label, symptom_prompt, symptom_info,
                evidence_prompt, evidence_info,
                correctness_prompt, correctness_raw, correctness_token_usage, pred
            )
            self._save_cache(fields_path, field_cache)
            self._save_cache(process_path, process_cache)

        return evl(labels, predictions, return_raw=True), eval_patch_paths

    def RunTrainPipeline(self):
        self.PrepareData(test=False)
        train_result, _ = self._run_on_split(self.train_patches, self.train_labels, "train")
        valid_result, valid_patch_paths = self._run_on_split(self.valid_patches, self.valid_labels, "valid")
        DumpJson({
            "train_metrics": {
                "tn": train_result[0], "fp": train_result[1], "fn": train_result[2], "tp": train_result[3],
                "acc": train_result[4], "prc": train_result[5], "rc": train_result[6], "f1": train_result[7], "auc": train_result[8]
            },
            "valid_metrics": {
                "tn": valid_result[0], "fp": valid_result[1], "fn": valid_result[2], "tp": valid_result[3],
                "acc": valid_result[4], "prc": valid_result[5], "rc": valid_result[6], "f1": valid_result[7], "auc": valid_result[8]
            }
        }, os.path.join(self.save_root, "train_valid_metrics.json"), encoding="utf-8", all_ascii=False, indent=2)
        return valid_result, valid_patch_paths

    def RunTestPipeline(self):
        self.PrepareData(train=False)
        return self._run_on_split(self.test_patches, self.test_labels, "test")

    def RunPredictPipeline(self):
        self.PreparePredictData()
        return self._run_on_split(self.predict_patches, self.predict_labels, "predict")
