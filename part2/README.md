# DS-GA 1011: Homework 4 Part 2

> ### Please start early.

## Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment. Note that Part 2 pins different package versions than Part 1, so use a separate environment from Part 1.

Example of virtual environment creation using conda:
```bash
conda create -n dsga1011-hw4-part2 python=3.11
conda activate dsga1011-hw4-part2
python -m pip install -r requirements.txt
```

You can refer to the [HPC Tutorials](https://github.com/Athul-R/NYU-HPC-Tutorials) for more information on how to use the NYU HPC.

## Hyperparameter grid search

### Continue training a saved model

The selected run-9 checkpoint can be fine-tuned further with a new optimizer and
scheduler. Keep a separate experiment name to preserve the three-epoch baseline:

```bash
python train_t5.py --finetune \
  --checkpoint_path checkpoints/ft_experiments/short3_lr0.0003_bs8_wd0/best \
  --experiment_name run9_continue_lr1e-4_bs8_wd0 \
  --completed_epochs 3 --max_n_epochs 27 --eval_every 3 --patience_epochs 3 \
  --learning_rate 1e-4 --batch_size 8 --weight_decay 0 \
  --num_warmup_epochs 0 --test_batch_size 8 --num_beams 4 --length_penalty 0.6
```

This allows up to 30 total epochs. Full-development loss is computed each epoch;
full-development SQL generation and Record F1 are computed every three added
epochs and at the final epoch. Early stopping waits for three F1 evaluations
without improvement. The starting checkpoint is evaluated and saved as the
initial best, so a weaker continued checkpoint cannot replace it. Selection is
recorded in `checkpoint_selection.json`; epoch histories include `total_epochs`.
This restarts optimizer/scheduler state from saved weights, rather than restoring
the exact original optimizer state. No training time limit is imposed.

The short 12-model search uses three epochs per model, full-development loss
for checkpoint selection, and one fixed 32-example development generation check
per model. It imposes no training time limits:

```bash
python grid_search.py --prefix short3 --max_n_epochs 3 --patience_epochs 2 \
  --num_warmup_epochs 0 --loss_selection --dev_samples 32 --num_beams 1
```

Ranked scores from this command are development-subset scores. Evaluate the
selected checkpoint on the full development set before reporting final results.

For a short timing experiment, select checkpoints using full-development loss
and generate SQL only once on a fixed development subset:

```bash
python train_t5.py --finetune --skip_test --loss_selection \
  --experiment_name timed_lr3e-4_bs16_wd001 \
  --learning_rate 3e-4 --batch_size 16 --weight_decay 0.01 \
  --max_n_epochs 3 --patience_epochs 2 --train_minutes 18 \
  --num_warmup_epochs 0 --test_batch_size 8 --dev_samples 32
```

`--train_minutes` ends the training loop after the current batch, then validates
and saves it. It does not cap final evaluation or total process time. The Colab
timing runner additionally uses an external 30-minute watchdog. Subset metrics
include the number of development examples and use separate filenames; they
must not be reported as full-development results. The subset's saved indices,
SQL, and ground-truth records remain aligned. Early stopping in this mode uses
development loss, with SQL generation deferred until the best checkpoint loads.

From `part2/`, inside your Part 2 environment on an allocated HPC GPU, run:

```bash
python grid_search.py --dry_run
python grid_search.py
```

The default grid trains 12 T5-small models: learning rates `5e-5`, `1e-4`, `3e-4`,
batch sizes `8`, `16`, and weight decay `0`, `0.01`. Each model starts from the
pretrained checkpoint with seed 42, uses linear scheduling with one warmup epoch,
and stops after 30 epochs or five epochs without improved development Record F1.

Checkpoints are selected using greedy development decoding. The best checkpoint
from each training run is then evaluated with beams `2`, `4` and length penalties
`0.6`, `1.0`, `1.2`, plus the greedy baseline: seven decoding configurations per
training run, 84 configurations altogether. Output length stays at 512 new tokens.
Generation comparisons reuse the checkpoint without retraining. Test inference
is skipped throughout the search.

Outputs are kept separately:

- `checkpoints/ft_experiments/grid_lr..._bs..._wd.../`: best/latest models,
  `training_args.json`, and per-epoch `history.jsonl`.
- `logs/grid/`: a log per training/decoding run, ranked `summary.csv`, and
  `summary_best.json`.
- `results/` and `records/`: development predictions, database records, and metrics,
  with separate names for each decoding configuration.

For separate HPC jobs, use `python grid_search.py --run_index 0` through
`--run_index 11` (one GPU per job). Each writes its own summary. After they finish,
combine the available results with `python grid_search.py --summarize`.
Completed evaluations are skipped when restarting. Use a new `--prefix` when
changing training settings or code, so previous experiments stay preserved.

After selecting the best development configuration, generate test outputs using
its experiment name, beam count, and length penalty:

```bash
python train_t5.py --finetune --eval_only \
  --experiment_name SELECTED_EXPERIMENT --num_beams SELECTED_BEAMS \
  --length_penalty SELECTED_PENALTY
```

This also evaluates development data and saves predictions with the decoding
suffix. For submission, copy the selected test outputs to the filenames listed
below. Do not select settings using test results.

## Evaluation commands

If you have saved predicted SQL queries and associated database records, you can compute F1 scores using:
```bash
python evaluate.py
  --predicted_sql results/t5_ft_experiment_dev.sql
  --predicted_records records/t5_ft_experiment_dev.pkl
  --development_sql data/dev.sql
  --development_records records/ground_truth_dev.pkl
```

## Submission

You need to submit your test SQL queries and their associated SQL records. Please only submit your final files corresponding to the test set.

For SQL queries, ensure that the name of the submission files (in the `results/` subfolder) are:
- `t5_ft_experiment_test.sql` (for extra credit `t5_ft_experiment_ec_test.sql`)

For database records, ensure that the name of the submission files (in the `records/` subfolder) are:
- `t5_ft_experiment_test.pkl` (for extra credit `t5_ft_experiment_ec_test.pkl`)

⚠️ Note that the predictions in each line of the .sql file or in each index of the list within the .pkl file must match each natural language query in `data/test.nl` in the order they appear.
