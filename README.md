# ShellForge-135M

ShellForge translates a single English request into a Bash command. The default pipeline fine-tunes SmolLM2-135M with command-only targets and uses greedy decoding. Training enforces a limit of fewer than 150 million parameters, including any auxiliary heads. The CLI prints the generated command for review.

## Quick start

Run from the repository root with Python, Bash, GNU coreutils and a PyTorch build suitable for your device. The first baseline run needs the Hugging Face model and dataset, either cached locally or available to download. Install ShellCheck to include lint results in evaluation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python smoke_test.py
./run_all.sh config.yaml
python infer.py --query "Sort the lines in values.txt numerically."
```

The runner prepares the data, trains the model, then evaluates the generated challenge and the validation split. Prepared files go to `data/` and the supervised checkpoint goes to `outputs/sft_ckpt/`. To compare runs, give each checkpoint its own `sft.save_dir`. Use a separate `project.data_dir` when changing data preparation. Pass the same `--config` to training, inference and evaluation. [Default configuration](config.yaml) · [Pipeline](run_all.sh)

## Training options

| Experiment | How to enable it |
| --- | --- |
| Schema retrieval | Set `schema_rag.enabled: true` before preparation to build the SQLite index and include tool descriptions in prompts. |
| Paired examples | Set `data.paired_training: true` to add fixture training rows to supervised training. By default the pipeline generates those rows for experiments. |
| Token preference loss | On the same prepared paired data, compare `sft.contrast_weight: 0.0` with `0.2`. Use command-only targets. The loss compares a correct command with a fixture-checked error at their first differing token. |
| Verified alternatives | Run `python eval_benchmark.py --k 4 --collect-train verified.jsonl`. Set `data.verified_candidates` to that file and prepare the next run. Preparation rechecks each alternative. |
| Fixture-based GRPO | After supervised training, run `python train_rgrpo.py --config config.yaml`. Select the resulting checkpoint with `--ckpt outputs/rgrpo_ckpt`. |
| Architecture experiments | `model.soft_mtp.enabled` enables auxiliary multi-token heads. `model.mode: recshell_120m` selects the recursive model trained from scratch. |

## Command memory and the offline pilot

Command memory uses verified training examples to reuse an exact request template or transfer a demonstrated GNU sort flag edit to a closely matching request. It handles one explicitly quoted filename. Unsupported wording and conflicting examples fall back to the checkpoint. Enable it with `--command-memory` and greedy decoding (`--k 1`).

```bash
python fixture_pilot.py --seed 42
python infer.py --command-memory --query "Sort 'readings.txt' numerically (descending) and keep duplicate lines. Print the first 2 line(s)."
```

The first command runs on CPU using local fixtures. The second uses the prepared data and trained checkpoint. The default fixture generator produces 120 training and 32 development prompts before overlap filtering. The pilot uses the compositional subset with 108 training and 24 held-out prompts. [Pilot](fixture_pilot.py) · [Fixture definitions](intent_pairs.py)

| Measured on the 24 pilot prompts | Functional passes |
| --- | ---: |
| Command memory, with numeric descending order held out of training | 24/24 |
| Randomly initialized 45,601-parameter model, 160 training steps | 0/24 at each of three seeds, with and without token preference loss |

These results measure transfer within the generator’s wording. The full SmolLM2-135M checkpoint has no reported official functional score in this repository.

## Evaluation

For a development comparison, run both routes on the same challenge. The evaluator reports selected-command passes, coverage, complete pairs answered correctly, and latency. Prediction files record the command and generation source for each prompt. Choose a new output filename for each run.

```bash
python eval_benchmark.py --test-file data/challenge.jsonl --predictions neural-dev.jsonl
python eval_benchmark.py --command-memory --test-file data/challenge.jsonl --predictions hybrid-dev.jsonl
```

After selecting a configuration on development data, omit `--test-file` to evaluate the prepared official split. NL2SH-ALFA has 300 test prompts with two references per prompt. Use the upstream InterCode-ALFA environment with Docker and the `icalfa` package for execution scoring. [Dataset](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA) · [Benchmark setup](https://github.com/westenfelder/InterCode-ALFA)

```bash
python eval_benchmark.py --predictions official.jsonl
python score_official.py official.jsonl --judge tfidf --report scores.json
```

For a paired comparison, add `--baseline baseline.jsonl` to the scoring command and supply complete predictions from the other system in the same official order. Use the same judge mode and model across comparisons. [Scorer](score_official.py)
