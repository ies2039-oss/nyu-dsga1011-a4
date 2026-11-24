# DS-GA 1011 Assignment 4: Fine-tuning Language Models

Starter code for Assignment 4. See the assignment writeup (`main.tex`/`main.pdf` in the course
handout) for the full problem statement.

Part 1 (BERT fine-tuning for sentiment classification) and Part 2 (T5 fine-tuning for text-to-SQL)
pin different, incompatible package versions, so set up a **separate environment for each part** —
see `part1/README.md` and `part2/README.md` for setup instructions specific to each part.

You will require GPUs for this homework. You can either use the HPC server (see documentation
[here](https://github.com/Athul-R/NYU-HPC-Tutorials)) or use an API service like
[lightning.ai](https://lightning.ai/) or other cloud providers.

## Layout

- `part1/`: starter code for Part 1 (`main.py`, `utils.py`). You will modify `main.py` and `utils.py`.
- `part2/`: starter code for Part 2 (`train_t5.py`, `t5_utils.py`, `utils.py`, `load_data.py`,
  `evaluate.py`), the text-to-SQL `data/`, the ground-truth dev records in `records/`, and a
  `results/` directory where you should save your model's output SQL/records for submission.

## Submission files

- `part1/out_original.txt`
- `part1/out_transformed.txt`
- `part1/out_augmented_original.txt`
- `part1/out_augmented_transformed.txt`
- `part2/results/t5_ft_experiment_test.sql`
- `part2/records/t5_ft_experiment_test.pkl`

Writeups are in `part1/q2a.ipynb` and `part2/writeup_part2.ipynb`.
Model checkpoints and local backup archives are excluded from Git.
The selected T5 checkpoint must be shared separately through Google Drive for submission.

Based on the course starter repository: https://github.com/gregdurrett/nyu-dsga1011-a4.
