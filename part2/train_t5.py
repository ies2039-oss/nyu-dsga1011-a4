import os
import argparse
import json
import pickle
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Continue from saved model weights with a fresh optimizer and scheduler')
    parser.add_argument('--completed_epochs', type=int, default=0, help='Epochs already trained by the starting checkpoint')
    parser.add_argument('--eval_every', type=int, default=1, help='Generate full dev SQL every N additional epochs; compute loss on other epochs')
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=30,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=5,
                        help="Stop after this many checkpoint-selection evaluations without improvement")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--eval_only', action='store_true', help='Evaluate the saved best checkpoint without training')
    parser.add_argument('--skip_test', action='store_true', help='Evaluate only development data during grid search')
    parser.add_argument('--loss_selection', action='store_true', help='Select checkpoints using dev loss; generate SQL only after training')
    parser.add_argument('--train_minutes', type=float, default=0, help='Training time budget; 0 means no time limit')
    parser.add_argument('--dev_samples', type=int, default=0, help='Fixed random dev subset for final generation; 0 means the full dev set')

    args = parser.parse_args()
    if args.eval_every < 1:
        parser.error('--eval_every must be at least 1')
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    if args.loss_selection:
        best_f1 = -float('inf')
    deadline = None
    if args.train_minutes > 0:
        deadline = time.monotonic() + args.train_minutes * 60
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    history_path = os.path.join(checkpoint_dir, 'history.jsonl')
    with open(history_path, 'w'):
        pass
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    if args.checkpoint_path:
        if args.loss_selection:
            eval_loss = eval_loss_only(model, dev_loader)
            best_f1 = -eval_loss
        else:
            eval_loss, best_f1, record_em, sql_em, error_rate = eval_epoch(
                args, model, dev_loader, gt_sql_path,
                model_sql_path.replace('_dev.sql', '_initial_dev.sql'), gt_record_path,
                model_record_path.replace('_dev.pkl', '_initial_dev.pkl'))
        save_model(checkpoint_dir, model, best=True)
        with open(os.path.join(checkpoint_dir, 'checkpoint_selection.json'), 'w') as f:
            json.dump({'total_epochs': args.completed_epochs,
                       'metric': 'negative_dev_loss' if args.loss_selection else 'record_f1',
                       'score': float(best_f1)}, f, indent=2)
        print(f'Starting checkpoint selection score: {best_f1}')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler, deadline)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        evaluate_f1 = not args.loss_selection and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.max_n_epochs)
        if not evaluate_f1:
            eval_loss = eval_loss_only(model, dev_loader)
            record_f1, record_em, sql_em, error_rate = None, None, None, None
            score = -eval_loss if args.loss_selection else None
            print(f"Epoch {epoch}: Dev loss: {eval_loss} (SQL generation deferred)")
        else:
            eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
            score = record_f1
            print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
        with open(history_path, 'a') as f:
            json.dump({'epoch': epoch, 'total_epochs': args.completed_epochs + epoch + 1,
                       'train_loss': tr_loss, 'dev_loss': eval_loss,
                       'record_f1': float(record_f1) if record_f1 is not None else None, 'record_em': record_em,
                       'sql_em': sql_em, 'error_rate': error_rate}, f)
            f.write('\n')

        if args.use_wandb:
            import wandb

            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if score is not None:
            if score > best_f1:
                best_f1 = score
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if score is not None and epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)
            with open(os.path.join(checkpoint_dir, 'checkpoint_selection.json'), 'w') as f:
                json.dump({'total_epochs': args.completed_epochs + epoch + 1,
                           'metric': 'negative_dev_loss' if args.loss_selection else 'record_f1',
                           'score': float(best_f1)}, f, indent=2)

        if score is not None and epochs_since_improvement >= args.patience_epochs:
            break
        if deadline is not None and time.monotonic() >= deadline:
            print('Training time budget reached; saved checkpoints before final evaluation.')
            break

def train_epoch(args, model, train_loader, optimizer, scheduler, deadline=None):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        if deadline is not None and time.monotonic() >= deadline:
            break

    return total_loss / total_tokens

def eval_loss_only(model, dev_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in dev_loader:
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            logits = model(input_ids=encoder_input, attention_mask=encoder_mask,
                           decoder_input_ids=decoder_input)['logits']
            non_pad = decoder_targets != PAD_IDX
            total_loss += criterion(logits[non_pad], decoder_targets[non_pad]).item()
            total_tokens += non_pad.sum().item()
    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    generated_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=512,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
            )

            for seq in generated_ids:
                pred_sql = tokenizer.decode(seq, skip_special_tokens=True)
                generated_queries.append(pred_sql)

    avg_loss = total_loss / total_tokens

    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )

    num_errors = 0
    for msg in model_error_msgs:
        if msg != "":
            num_errors += 1
    error_rate = num_errors / len(model_error_msgs)

    return avg_loss, record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    generated_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=512,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
            )

            for seq in generated_ids:
                pred_sql = tokenizer.decode(seq, skip_special_tokens=True)
                generated_queries.append(pred_sql)

    # Save the generated SQL strings and execute them to save the database records
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
def main():
    # Get key arguments
    args = get_args()
    set_random_seeds(args.seed)
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    if not args.eval_only:
        model = initialize_model(args)
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        # Release training parameters and optimizer state before loading the best model.
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    # Keep decoding experiments separate while loading the same training checkpoint.
    if args.eval_only:
        experiment_name += f'_beams{args.num_beams}_lp{args.length_penalty:g}'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    if args.dev_samples > 0 and args.dev_samples < len(dev_loader.dataset):
        indices = sorted(random.Random(args.seed).sample(range(len(dev_loader.dataset)), args.dev_samples))
        dev_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dev_loader.dataset, indices), batch_size=args.test_batch_size,
            shuffle=False, collate_fn=dev_loader.collate_fn)
        experiment_name += f'_dev{args.dev_samples}'
        with open(gt_sql_path) as f:
            queries = f.readlines()
        with open(gt_record_path, 'rb') as f:
            records, errors = pickle.load(f)
        gt_sql_path = f'results/t5_{model_type}_{experiment_name}_ground_truth.sql'
        gt_record_path = f'records/t5_{model_type}_{experiment_name}_ground_truth.pkl'
        with open(gt_sql_path, 'w') as f:
            f.writelines(queries[i] for i in indices)
        with open(gt_record_path, 'wb') as f:
            pickle.dump(([records[i] for i in indices], [errors[i] for i in indices]), f)
        with open(f'results/t5_{model_type}_{experiment_name}_indices.json', 'w') as f:
            json.dump(indices, f)
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")
    metrics = {'experiment_name': args.experiment_name, 'num_beams': args.num_beams,
               'dev_examples': len(dev_loader.dataset), 'loss_selection': args.loss_selection,
               'length_penalty': args.length_penalty, 'dev_loss': dev_loss,
               'record_f1': float(dev_record_f1), 'record_em': dev_record_em,
               'sql_em': dev_sql_em, 'error_rate': dev_error_rate}
    with open(f'results/t5_{model_type}_{experiment_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    if args.skip_test:
        return

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
