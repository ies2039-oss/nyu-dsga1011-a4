# Parts of this code were adapted from my own work for CSCI-GA 3033-091 (Fall 2026),
# Introduction to LLM Based Generative AI Systems,
# Homework 4 — Problem 1: Full Fine-Tuning and LoRA.

import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    import wandb

    wandb.init(project='dsga1011-hw4-part2', name=args.experiment_name,
               config=vars(args))

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''

    model_name = 'google-t5/t5-small'
    checkpoint_path = getattr(args, 'checkpoint_path', None)
    if checkpoint_path:
        if not args.finetune:
            raise ValueError('--checkpoint_path requires --finetune')
        model = T5ForConditionalGeneration.from_pretrained(checkpoint_path, local_files_only=True)
    elif args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)

    # Fine-tune the entire model; the dataset handles tokenization.
    return model.to(DEVICE)
    

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    sub_dir = "best" if best else "latest"
    save_path = os.path.join(checkpoint_dir, sub_dir)
    os.makedirs(save_path, exist_ok=True)

    # Unwrap model if using DataParallel or DistributedDataParallel
    model_to_save = model.module if hasattr(model, "module") else model

    # Save weights and config in Hugging Face format
    model_to_save.save_pretrained(save_path)

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    sub_dir = "best" if best else "latest"
    save_path = os.path.join(checkpoint_dir, sub_dir)
    if not os.path.isfile(os.path.join(save_path, 'config.json')):
        raise FileNotFoundError(f'No saved model found at {save_path}. Train and save the model first.')
    model = T5ForConditionalGeneration.from_pretrained(save_path, local_files_only=True)
    return model.to(DEVICE).eval()

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        raise ValueError(f'Unsupported optimizer: {args.optimizer_type}')

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

