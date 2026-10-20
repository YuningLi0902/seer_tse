import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from util.EvalHelper import evl
from util.FileHelper import DumpPickleFile, LoadJson, LoadPickleFile


FIELD_NAMES = [
    "symptom_summary",
    "core_edit_1",
    "core_edit_2",
    "core_edit_3",
    "expected_effect",
    "symptom_alignment"
]


def _resolve_file_path(path_value, fallback_relative=None):
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
        resolved = candidate.resolve()
        resolved_str = str(resolved)
        if resolved_str in seen:
            continue
        seen.add(resolved_str)
        if os.path.isfile(resolved_str):
            return resolved_str
    return str(candidates[-1].resolve()) if candidates else None


def _normalize_patch_key(patch_path):
    return patch_path.replace("\\", "/")


def _build_patch_key_candidates(patch_path, dataset_name):
    normalized = _normalize_patch_key(patch_path)
    candidates = [normalized]
    return candidates


def _normalize_text(text):
    if text is None:
        return "NONE"
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if text == "":
        return "NONE"
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _build_semantic_fields(record):
    symptom_fields = record.get("symptom_fields", {})
    evidence_fields = record.get("evidence_fields", {})
    return {
        "symptom_summary": _normalize_text(symptom_fields.get("symptom_summary", "NONE")),
        "core_edit_1": _normalize_text(evidence_fields.get("core_edit_1", "NONE")),
        "core_edit_2": _normalize_text(evidence_fields.get("core_edit_2", "NONE")),
        "core_edit_3": _normalize_text(evidence_fields.get("core_edit_3", "NONE")),
        "expected_effect": _normalize_text(evidence_fields.get("expected_effect", "NONE")),
        "symptom_alignment": _normalize_text(evidence_fields.get("symptom_alignment", "NONE"))
    }


def _build_empty_semantic_fields():
    return {field_name: "NONE" for field_name in FIELD_NAMES}


class SemanticFieldPatchDataset(Dataset):
    def __init__(self, patch_paths, code_texts, semantic_fields, labels):
        self.patch_paths = patch_paths
        self.code_texts = code_texts
        self.semantic_fields = semantic_fields
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {"patch_path": self.patch_paths[idx], "code_text": self.code_texts[idx], "label": int(self.labels[idx])}
        item.update(self.semantic_fields[idx])
        return item


class FieldAwareStrongFusionCodeBERT(nn.Module):
    """Shared encoder, patch-conditioned evidence aggregation and three heads."""
    def __init__(self, model_path, num_fields, dropout=0.2, gate_hidden_size=16):
        super().__init__()
        if gate_hidden_size <= 0:
            raise ValueError("gate_hidden_size must be positive")
        self.encoder = AutoModel.from_pretrained(model_path)
        hidden_size = self.encoder.config.hidden_size
        self.num_fields = num_fields
        # Preserve the initialization scale and random stream of the training implementation.
        initial_gate = nn.Linear(hidden_size * 2, 1)
        self.code_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        self.semantic_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        self.fused_head = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        with torch.random.fork_rng(devices=[]):
            self.field_gate = nn.Linear(hidden_size, 1, bias=False)
            self.patch_projection = nn.Linear(hidden_size, gate_hidden_size, bias=False)
            self.field_projection = nn.Linear(hidden_size, gate_hidden_size)
            self.residual_output = nn.Linear(gate_hidden_size, 1, bias=False)
            with torch.no_grad():
                self.field_gate.weight.copy_(initial_gate.weight[:, hidden_size:])
                self.residual_output.weight.zero_()
    def _encode_cls(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
    def forward(self, code_input_ids, code_attention_mask, field_input_ids_list, field_attention_mask_list):
        h_code = self._encode_cls(code_input_ids, code_attention_mask)
        fields = [self._encode_cls(ids, mask) for ids, mask in zip(field_input_ids_list, field_attention_mask_list)]
        if len(fields) != self.num_fields:
            raise ValueError("Unexpected number of evidence slots")
        stacked_fields = torch.stack(fields, dim=1)
        linear_scores = self.field_gate(stacked_fields).squeeze(-1)
        hidden = torch.tanh(self.patch_projection(h_code).unsqueeze(1) + self.field_projection(stacked_fields))
        residual_scores = self.residual_output(hidden).squeeze(-1)
        field_weights = torch.softmax(linear_scores + residual_scores, dim=1)
        h_sem = torch.sum(field_weights.unsqueeze(-1) * stacked_fields, dim=1)
        h_fuse = torch.cat([h_code, h_sem, torch.abs(h_code - h_sem), h_code * h_sem], dim=1)
        return {"code_logit": self.code_head(h_code).squeeze(-1),
                "semantic_logit": self.semantic_head(h_sem).squeeze(-1),
                "fused_logit": self.fused_head(h_fuse).squeeze(-1), "field_weights": field_weights}


class SEERS2Model(object):
    def __init__(self, model_save_target, model_path="default", semantic_fields_path="default",
                 train_data_file="default", valid_data_file="default", test_data_file="default",
                 dataset_name="Lin2022TOSEM", lambda_code=0.15, lambda_semantic=0.15,
                 gate_hidden_size=16):
        self.model_save_target = model_save_target
        self.dataset_name = dataset_name
        self.model_path = self._resolve_model_path(model_path)
        self.semantic_fields_path = _resolve_file_path(
            semantic_fields_path,
            "resources/seers1_reference_evidence/predict_fields_{}.json".format(dataset_name.lower())
        )
        self.train_data_file = _resolve_file_path(train_data_file, "datasets/patch_path_train_group_0.pkl")
        self.valid_data_file = _resolve_file_path(valid_data_file, "datasets/patch_path_test_group_0.pkl")
        self.test_data_file = _resolve_file_path(test_data_file, "datasets/patch_path_test_group_0.pkl")

        self.code_max_length = 512
        self.field_max_length = 128
        self.train_batch_size = 8
        self.eval_batch_size = 16
        self.grad_accum_steps = 2
        self.max_epoch = 20
        self.patience = 4
        self.encoder_lr = 2e-5
        self.head_lr = 1e-4
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.dropout = 0.2
        self.max_grad_norm = 1.0
        self.seed = 42
        self.lambda_code = float(lambda_code)
        self.lambda_semantic = float(lambda_semantic)
        self.gate_hidden_size = int(gate_hidden_size)
        if self.lambda_code < 0 or self.lambda_semantic < 0:
            raise ValueError("Auxiliary loss weights must be nonnegative")

        self._set_seed(self.seed)
        os.makedirs(self.model_save_target, exist_ok=True)

        self.semantic_record_dict = self._load_semantic_records()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FieldAwareStrongFusionCodeBERT(
            self.model_path,
            len(FIELD_NAMES),
            dropout=self.dropout,
            gate_hidden_size=self.gate_hidden_size
        ).to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = None
        self.scheduler = None
        self.loss_function = None

        self.train_patches = None
        self.train_labels = None
        self.valid_patches = None
        self.valid_labels = None
        self.test_patches = None
        self.test_labels = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.best_epoch = -1
        self.best_metric = -1.0
        self.no_improve_epoch = 0
        self.max_tn = 0
        self.max_fp = 0
        self.max_fn = 0
        self.max_tp = 0
        self.max_acc = 0
        self.max_prc = 0
        self.max_rc = 0
        self.max_f1 = 0
        self.max_auc_ = 0
        self.eval_labels = None
        self.eval_preds = None
        self.eval_paths = None
        self.eval_scores = None
        self.last_eval_scores = None

    @staticmethod
    def _resolve_model_path(model_path):
        return "microsoft/codebert-base" if model_path in [None, "", "default"] else model_path

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_semantic_records(self):
        records = LoadJson(self.semantic_fields_path, encoding="utf-8")
        if records is None:
            raise FileNotFoundError("Cannot find semantic fields json: {}".format(self.semantic_fields_path))
        normalized = {}
        for key, value in records.items():
            normalized[_normalize_patch_key(key)] = value
            patch_path = value.get("patch_path")
            if patch_path:
                normalized[_normalize_patch_key(patch_path)] = value
        return normalized

    def _load_split(self, file_path):
        patches, labels = LoadPickleFile(file_path)
        return patches, labels

    def _build_sample_texts(self, patches):
        patch_paths, code_texts, semantic_fields_all = [], [], []
        for patch_path, patch_text in patches:
            patch_paths.append(patch_path)
            code_texts.append(_normalize_text(patch_text))
            semantic_record = next((self.semantic_record_dict[k] for k in
                                    _build_patch_key_candidates(patch_path, self.dataset_name)
                                    if k in self.semantic_record_dict), None)
            semantic_fields_all.append(_build_empty_semantic_fields() if semantic_record is None
                                       else _build_semantic_fields(semantic_record))
        return patch_paths, code_texts, semantic_fields_all

    def _build_loader(self, patches, labels, shuffle, split_tag):
        paths, texts, fields = self._build_sample_texts(patches)
        dataset = SemanticFieldPatchDataset(paths, texts, fields, labels)
        return DataLoader(dataset, batch_size=self.train_batch_size if shuffle else self.eval_batch_size,
                          shuffle=shuffle, collate_fn=self._collate_batch)

    def _collate_batch(self, batch):
        collated = {"labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
                    "patch_paths": [item["patch_path"] for item in batch]}
        for name in ["code"] + FIELD_NAMES:
            key = "code_text" if name == "code" else name
            encoding = self.tokenizer([item[key] for item in batch], truncation=True, padding="max_length",
                                      max_length=self.code_max_length if name == "code" else self.field_max_length,
                                      return_tensors="pt")
            collated[name + "_input_ids"] = encoding["input_ids"]
            collated[name + "_attention_mask"] = encoding["attention_mask"]
        return collated

    def _init_optimizer(self):
        model_core = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        no_decay = ["bias", "LayerNorm.weight"]
        encoder_params = []
        head_params = []
        for name, param in model_core.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append((name, param))
            else:
                head_params.append((name, param))
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
                "lr": self.encoder_lr,
                "weight_decay": self.weight_decay
            },
            {
                "params": [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
                "lr": self.encoder_lr,
                "weight_decay": 0.0
            },
            {
                "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
                "lr": self.head_lr,
                "weight_decay": self.weight_decay
            },
            {
                "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
                "lr": self.head_lr,
                "weight_decay": 0.0
            }
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        steps_per_epoch = max(1, int(np.ceil(len(self.train_loader) / self.grad_accum_steps)))
        total_steps = steps_per_epoch * self.max_epoch
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        positive_num = max(1, int(sum(self.train_labels)))
        negative_num = max(1, int(len(self.train_labels) - positive_num))
        pos_weight = torch.tensor([negative_num / positive_num], dtype=torch.float32, device=self.device)
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _forward_batch(self, batch):
        return self.model(batch["code_input_ids"].to(self.device), batch["code_attention_mask"].to(self.device),
                          [batch[name + "_input_ids"].to(self.device) for name in FIELD_NAMES],
                          [batch[name + "_attention_mask"].to(self.device) for name in FIELD_NAMES])

    def _compute_loss(self, outputs, labels):
        labels = labels.to(self.device)
        fused_loss = self.loss_function(outputs["fused_logit"], labels)
        code_loss = self.loss_function(outputs["code_logit"], labels)
        semantic_loss = self.loss_function(outputs["semantic_logit"], labels)
        total_loss = fused_loss + self.lambda_code * code_loss + self.lambda_semantic * semantic_loss
        return total_loss, {"fused": fused_loss.item(), "code": code_loss.item(), "semantic": semantic_loss.item()}

    def Save(self):
        for checkpoint in os.listdir(self.model_save_target):
            if checkpoint.startswith("checkpoint"):
                os.remove(os.path.join(self.model_save_target, checkpoint))
        model_core = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        torch.save({
            "epoch": self.best_epoch,
            "model_state_dict": model_core.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None
        }, os.path.join(self.model_save_target, "checkpoint{}".format(self.best_epoch)))
        print("-------------- The model has been saved! --------------")

    def Load(self):
        model_file = None
        for item in os.listdir(self.model_save_target):
            if item.startswith("checkpoint"):
                model_file = item
                break
        if model_file is None:
            raise FileNotFoundError("Cannot find checkpoint under {}".format(self.model_save_target))
        checkpoint = torch.load(os.path.join(self.model_save_target, model_file), map_location=self.device)
        model_core = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model_core.load_state_dict(checkpoint["model_state_dict"])
        print("-------------- Load target model parameter successfully! --------------")

    def PrepareData(self, train=True, test=True):
        if train:
            self.train_patches, self.train_labels = self._load_split(self.train_data_file)
            self.valid_patches, self.valid_labels = self._load_split(self.valid_data_file)
            print("Train Set:", len(self.train_labels))
            print("Valid Set:", len(self.valid_labels))
            self.train_loader = self._build_loader(self.train_patches, self.train_labels, shuffle=True, split_tag="train")
            self.valid_loader = self._build_loader(self.valid_patches, self.valid_labels, shuffle=False, split_tag="valid")
            self._init_optimizer()
        if test:
            self.test_patches, self.test_labels = self._load_split(self.test_data_file)
            print("Test Set:", len(self.test_labels))
            self.test_loader = self._build_loader(self.test_patches, self.test_labels, shuffle=False, split_tag="test")

    def Evaluate(self, working_mode="train"):
        loader = self.test_loader if working_mode == "test" else self.valid_loader
        self.model.eval()
        y_true = []
        y_score = []
        patch_paths = []
        with torch.no_grad():
            for batch in loader:
                outputs = self._forward_batch(batch)
                probs = torch.sigmoid(outputs["fused_logit"])
                labels = batch["labels"]
                y_true.extend(labels.numpy().tolist())
                y_score.extend(probs.cpu().numpy().tolist())
                patch_paths.extend(batch["patch_paths"])
        self.last_eval_scores = y_score
        return evl(y_true, y_score, return_raw=True), patch_paths

    def ReturnBestPerformance(self):
        return (
            self.max_tn, self.max_fp, self.max_fn, self.max_tp, self.max_acc, self.max_prc, self.max_rc,
            self.max_f1, self.max_auc_, self.eval_labels, self.eval_preds
        ), self.eval_paths

    def Train(self):
        print("-------------- start training ---------------")
        epoch_iterator = tqdm(range(self.max_epoch), desc="Training Epochs", leave=True)
        for epoch in epoch_iterator:
            self.model.train()
            losses = []
            self.optimizer.zero_grad()
            batch_iterator = tqdm(self.train_loader, desc="Epoch {}".format(epoch), leave=False)
            for step, batch in enumerate(batch_iterator, start=1):
                labels = batch["labels"]
                outputs = self._forward_batch(batch)
                loss, loss_items = self._compute_loss(outputs, labels)
                loss = loss / self.grad_accum_steps
                loss.backward()
                losses.append(loss.item() * self.grad_accum_steps)
                batch_iterator.set_postfix({
                    "loss": "{:.4f}".format(float(np.mean(losses))),
                    "fused": "{:.4f}".format(loss_items["fused"]),
                    "lr": "{:.2e}".format(self.optimizer.param_groups[0]["lr"])
                })
                if step % self.grad_accum_steps == 0 or step == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            epoch_loss = float(np.mean(losses))
            result, paths = self.Evaluate()
            current_f1 = result[7]
            epoch_iterator.set_postfix({
                "loss": "{:.4f}".format(epoch_loss),
                "f1": "{:.4f}".format(current_f1),
                "best_f1": "{:.4f}".format(max(self.best_metric, current_f1))
            })
            print("========= epoch:", epoch, "loss:", epoch_loss, "f1:", current_f1, "==============")
            if current_f1 > self.best_metric:
                self.best_metric = current_f1
                self.no_improve_epoch = 0
                self.max_tn = result[0]
                self.max_fp = result[1]
                self.max_fn = result[2]
                self.max_tp = result[3]
                self.max_acc = result[4]
                self.max_prc = result[5]
                self.max_rc = result[6]
                self.max_f1 = result[7]
                self.max_auc_ = result[8]
                self.eval_labels = result[9]
                self.eval_preds = result[10]
                self.eval_scores = list(self.last_eval_scores) if self.last_eval_scores is not None else None
                self.eval_paths = paths
                self.best_epoch = epoch
                self.Save()
            else:
                self.no_improve_epoch += 1
                if self.no_improve_epoch >= self.patience:
                    print("-------------- Early stopping at epoch {} --------------".format(epoch))
                    break

    def RunTrainPipeline(self):
        self.PrepareData(test=False)
        self.Train()
        metrics = {
            "tn": self.max_tn,
            "fp": self.max_fp,
            "fn": self.max_fn,
            "tp": self.max_tp,
            "acc": self.max_acc,
            "prc": self.max_prc,
            "rc": self.max_rc,
            "f1": self.max_f1,
            "auc": self.max_auc_
        }
        DumpPickleFile(os.path.join(self.model_save_target, "best_metrics.pkl"), metrics)
        return self.ReturnBestPerformance()

    def RunTestPipeline(self):
        self.PrepareData(train=False)
        self.Load()
        return self.Evaluate(working_mode="test")
