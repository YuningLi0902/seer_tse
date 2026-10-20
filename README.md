# SEER

This repository contains the standalone implementation of SEER: semantic evidence extraction followed by a supervised predictor that jointly learns from patch diffs and the extracted evidence.

- `SEERS1`: extracts semantic evidence and also runs EviReasoning by default.
- `SEERS2`: trains or evaluates the SEER predictor.

The release includes Lin2022TOSEM (1,183 patches: 535 correct and 648 overfitting) with 10-fold, 10-group and cross-project splits, and ChatPatch (323 patches: 159 correct and 164 overfitting) with 10-fold splits. Ablation experiment switches are not included.

## Structure

- `baselines/RunBaseline.py`: command-line entry point.
- `baselines/SEERS1/SEERS1Model.py`: evidence extraction prompts and EviReasoning.
- `baselines/SEERS1/merge_seers1_shard_outputs.py`: shard merging utility.
- `baselines/SEERS2/SEERS2Model.py`: prediction model and training pipeline.
- `datasets/`: patch texts, labels and predefined splits stored in PKL files.
- `resources/`: bug reports, failing-test metadata and cached semantic evidence.
- `util/`: shared helpers and API model configuration.

The dataset PKL files contain patch texts; the patch paths stored in them identify samples and do not require separate patch files for these pipelines.

## Dependencies

Install `torch`, `transformers`, `numpy`, `scikit-learn`, `tqdm`, `openai`, `openpyxl` and `xlsxwriter` in your Python environment. GPU training is recommended. No pretrained encoder weights are bundled.

## Configuration before running

Run the following commands from `baselines/`.

### Stage 1 API configuration

**Before making API calls, update the model configuration and provide your own API key.**

1. Edit `util/Model.py`: in `OpenAILM.model_dict`, replace the placeholder endpoint URL and provider model ID for the alias you intend to use. Endpoint URLs and API keys are separate settings.
2. The provided aliases are `gpt5.2`, `claude4.6sonnet` and `deepseekv4flash`. If you add or rename an alias, also update `CLOSE_SOURCE_MODEL_LIST` in `util/ModelEngine.py`.
3. Store your API key in a local text file and pass its path with `--key_file`. Do not commit the key file.
4. Ensure the selected provider accepts the OpenAI-compatible chat-completions request used by the wrapper.

The placeholders intentionally prevent API calls until configured. Cached evidence can be used for Stage 2 without an API key or API calls.

**Stage 1 behavior:** every patch receives semantic evidence extraction followed by an additional correctness-reasoning prompt, producing the EviReasoning comparison as well as SEER's evidence. Stage 2 reads only the six evidence slots; the cached correctness judgments and labels are not encoded as evidence. Stage 1 token totals therefore include EviReasoning calls and should not be interpreted as evidence-extraction-only cost.

### Stage 2 encoder and evidence

The encoder defaults to `microsoft/codebert-base`, which Transformers can download from Hugging Face. For a local model, pass `--model_path /path/to/local/codebert`; this directory must contain the model configuration, weights and tokenizer files.

The formal defaults are:

- `--lambda_code 0.15`
- `--lambda_semantic 0.15`
- `--gate_hidden_size 16`

The predictor uses a linear evidence score plus a patch-conditioned tanh correction. The correction output projection is initialized to zero; the other gate projections and prediction heads are randomly initialized. Initialization scale and random-stream handling follow the experiment implementation.

By default, `--semantic_fields_path default` selects the cached evidence for the selected dataset:

- `resources/seers1_reference_evidence/predict_fields_lin2022tosem.json`
- `resources/seers1_reference_evidence/predict_fields_chatpatch.json`

Use `--semantic_fields_path /path/to/predict_fields.json` to train on newly generated evidence.

## Reproduction

### Generate evidence and EviReasoning results

Example shard run:

```shell
python RunBaseline.py \
  --baseline SEERS1 \
  --dataset Lin2022TOSEM \
  --working_mode test \
  --division_mode fold \
  --model gpt5.2 \
  --key_file /path/to/api_key.txt \
  --temperature 0.0 \
  --shard_index 0 \
  --num_shards 8 \
  --run_name main
```

Run each shard index from 0 to 7 with the same configuration, then merge:

```shell
python SEERS1/merge_seers1_shard_outputs.py \
  --output_root ./SEERS1/output/train_on_Lin2022TOSEM/fold/gpt5.2_t0.0_main \
  --num_shards 8
```

For ChatPatch, change `--dataset` to `ChatPatch` and use the printed ChatPatch output root when merging. Stage 1 processes the entire selected dataset; `--division_mode` names its output folder rather than partitioning API calls by training fold. Shards partition the samples.

Stage 1 resumes from existing cached outputs. After changing endpoint/model IDs, prompts or artifact inputs, use a new `--run_name` to avoid reusing an earlier cache under the same model alias.

### Train SEER on Lin2022TOSEM

10-fold:

```shell
python RunBaseline.py \
  --baseline SEERS2 \
  --dataset Lin2022TOSEM \
  --working_mode train \
  --division_mode fold \
  --model_path microsoft/codebert-base \
  --save_detailed_result \
  --save_probability
```

10-group and cross-project use the same command with `--division_mode group` and `--division_mode cross_project`, respectively.

### Train SEER on ChatPatch

```shell
python RunBaseline.py \
  --baseline SEERS2 \
  --dataset ChatPatch \
  --working_mode train \
  --division_mode fold \
  --model_path microsoft/codebert-base \
  --save_detailed_result \
  --save_probability
```

### Evaluate saved checkpoints

Reuse the training command with `--working_mode test`. Keep the encoder path, evidence file, loss weights, hidden size and run name consistent so the same output directory is selected. Add `--test_model_index 0` to evaluate only fold 0; the default evaluates all splits.

## Output locations

The entry point prints the output directory before running. Results are organized under:

```text
<save_target_root>/<SEERS1-or-SEERS2>/output/train_on_<dataset>/<division-folder>/<configuration>/
```

Stage 1 configuration names include the model alias, temperature and run name. Stage 2 configuration names include auxiliary loss weights, gate hidden size, an encoder-path hash, an evidence-file content hash and the run name. Split checkpoints are stored in numbered subdirectories.

Training refuses to reuse an existing split output directory. Use a new `--run_name` or `--save_target_root` for another run. No training command silently deletes an earlier checkpoint directory.

The release provides the SEER method and its necessary inputs; implementations of other compared APCA methods are not bundled.
