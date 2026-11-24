import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from itertools import product
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description='T5 training and decoding grid search')
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[5e-5, 1e-4, 3e-4])
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--weight_decays', nargs='+', type=float, default=[0, 0.01])
    parser.add_argument('--num_beams', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--length_penalties', nargs='+', type=float, default=[0.6, 1.0, 1.2])
    parser.add_argument('--max_n_epochs', type=int, default=30)
    parser.add_argument('--patience_epochs', type=int, default=5)
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--loss_selection', action='store_true')
    parser.add_argument('--dev_samples', type=int, default=0)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefix', default='grid')
    parser.add_argument('--run_index', type=int, help='Run one training configuration (0 to 11 for the default grid)')
    parser.add_argument('--dry_run', action='store_true', help='Print the plan without running models')
    parser.add_argument('--summarize', action='store_true', help='Rank existing metrics without running models')
    return parser.parse_args()


def run_command(command, log_path, metrics_path, checkpoint_path):
    # Completed evaluations can be reused when restarting an interrupted search.
    if metrics_path.exists() and checkpoint_path.exists():
        print(f'Already completed: {metrics_path.name}', flush=True)
        return
    print(shlex.join(command), flush=True)
    with open(log_path, 'w') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)


def save_summary(rows, log_dir, name):
    if not rows:
        return
    rows = sorted(rows, key=lambda row: row['record_f1'], reverse=True)
    with open(log_dir / f'{name}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(log_dir / f'{name}_best.json', 'w') as f:
        json.dump(rows[0], f, indent=2)
    print(f'Best development Record F1 so far: {rows[0]["record_f1"]:.6f}', flush=True)


def main():
    args = get_args()
    part2_dir = Path(__file__).resolve().parent
    # Paths in the starter training and database code are relative to part2.
    os.chdir(part2_dir)
    log_dir = Path('logs') / args.prefix

    training_grid = list(product(args.learning_rates, args.batch_sizes, args.weight_decays))
    decoding_grid = [(1, 1.0)]
    for num_beams, length_penalty in product(args.num_beams, args.length_penalties):
        # Length penalties only affect beam search; do not repeat greedy decoding.
        if num_beams > 1 and (num_beams, length_penalty) not in decoding_grid:
            decoding_grid.append((num_beams, length_penalty))
    if 1 not in args.num_beams:
        # Greedy decoding is still used for checkpoint selection during training.
        print('Including the greedy decoding baseline.')
    if args.run_index is not None and not 0 <= args.run_index < len(training_grid):
        raise ValueError(f'run_index must be between 0 and {len(training_grid) - 1}')
    print(f'{len(training_grid)} training runs; {len(decoding_grid)} decoding configurations per checkpoint.')
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary_name = 'summary' if args.run_index is None else f'summary_{args.run_index:02d}'

    for index, (learning_rate, batch_size, weight_decay) in enumerate(training_grid):
        if args.run_index is not None and index != args.run_index:
            continue
        experiment_name = f'{args.prefix}_lr{learning_rate:g}_bs{batch_size}_wd{weight_decay:g}'
        checkpoint_path = Path('checkpoints/ft_experiments') / experiment_name / 'best/config.json'
        command = [sys.executable, '-u', 'train_t5.py', '--finetune', '--skip_test',
                   '--experiment_name', experiment_name,
                   '--learning_rate', str(learning_rate), '--batch_size', str(batch_size),
                   '--weight_decay', str(weight_decay), '--max_n_epochs', str(args.max_n_epochs),
                   '--patience_epochs', str(args.patience_epochs), '--scheduler_type', 'linear',
                   '--num_warmup_epochs', str(args.num_warmup_epochs), '--test_batch_size', str(args.test_batch_size),
                   '--seed', str(args.seed)]
        if args.loss_selection:
            command.append('--loss_selection')
        if args.dev_samples > 0:
            command += ['--dev_samples', str(args.dev_samples)]
        print(f'[{index}] {experiment_name}', flush=True)
        for num_beams, length_penalty in decoding_grid:
            name = experiment_name
            eval_command = command
            if num_beams > 1:
                name += f'_beams{num_beams}_lp{length_penalty:g}'
                eval_command = command + ['--eval_only', '--num_beams', str(num_beams),
                                          '--length_penalty', str(length_penalty)]
            if args.dev_samples > 0:
                name += f'_dev{args.dev_samples}'
            metrics_path = Path('results') / f't5_ft_{name}_metrics.json'
            if args.dry_run:
                print('  ' + shlex.join(eval_command))
                continue
            if not args.summarize:
                run_command(eval_command, log_dir / f'{name}.log', metrics_path, checkpoint_path)
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                metrics.update({'learning_rate': learning_rate, 'batch_size': batch_size,
                                'weight_decay': weight_decay, 'seed': args.seed,
                                'checkpoint': str(checkpoint_path.parent)})
                rows.append(metrics)
                save_summary(rows, log_dir, summary_name)

    if not args.dry_run and not rows:
        print('No completed experiment metrics found.')


if __name__ == '__main__':
    main()
