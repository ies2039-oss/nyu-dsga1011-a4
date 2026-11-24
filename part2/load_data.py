import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle
import json
import sqlite3
import warnings
from pathlib import Path
from contextlib import closing

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * T5 uses its padding token as the decoder start token.
            * Class behavior should be different on the test set.
        '''
        self.data = []
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.data_folder = data_folder
        self.input, self.target = self.process_data(self.data_folder, self.split, self.tokenizer)

        encoder = self.tokenizer(self.input, padding=False, truncation=False)
        if self.target is not None:
            decoder = self.tokenizer(self.target, padding=False, truncation=False)

        for idx in range(len(self.input)):
            item = {
                "encoder_ids": encoder["input_ids"][idx],
                "encoder_mask": encoder["attention_mask"][idx],
                "initial_decoder_inputs": [self.tokenizer.pad_token_id],
            }
            if self.target is not None:
                item["decoder_inputs"] = [self.tokenizer.pad_token_id] + decoder["input_ids"][idx][:-1]
                item["decoder_targets"] = decoder["input_ids"][idx]
            self.data.append(item)

    def process_data(self, data_folder, split, tokenizer):
        """Clean line endings and verify SQL against the supplied SQLite database.

        Invalid SQL is reported without dropping or rewriting aligned examples.
        The test split has no SQL targets. DDL is a separate artifact, not a prompt.
        """
        inputs = [clean_text(line) for line in load_lines(os.path.join(data_folder, f"{split}.nl"), strip=False)]
        self.sql_validation = None
        if split == "test":
            return inputs, None
        targets = [clean_text(line) for line in load_lines(os.path.join(data_folder, f"{split}.sql"), strip=False)]
        if len(inputs) != len(targets):
            raise ValueError(f"{split}: input and SQL files have different numbers of lines")
        self.sql_validation = validate_sql_queries(targets, os.path.join(data_folder, "flight_database.db"))
        if self.sql_validation["errors"]:
            warnings.warn(f"{split}: {len(self.sql_validation['errors'])} SQL queries failed SQLite validation; examples retained. Inspect dataset.sql_validation['errors'].", stacklevel=2)
        return inputs, targets

    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    #padding the encoder and decoder inputs to the max length in the batch
    encoder_ids = pad_sequence([torch.tensor(item["encoder_ids"], dtype=torch.long) for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([torch.tensor(item["encoder_mask"], dtype=torch.long) for item in batch], batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence([torch.tensor(item["decoder_inputs"], dtype=torch.long) for item in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([torch.tensor(item["decoder_targets"], dtype=torch.long) for item in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.tensor([item["initial_decoder_inputs"] for item in batch], dtype=torch.long)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([torch.tensor(item["encoder_ids"], dtype=torch.long) for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([torch.tensor(item["encoder_mask"], dtype=torch.long) for item in batch], batch_first=True, padding_value=0)
    initial_decoder_inputs = torch.tensor([item["initial_decoder_inputs"] for item in batch], dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path, strip=True):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() if strip else line.rstrip('\r\n') for line in f]

def clean_text(text):
    """Remove surrounding whitespace and replace actual internal line breaks with spaces."""
    return text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').strip()

def validate_sql_queries(queries, database_path):
    """Compile each query with SQLite EXPLAIN, without executing the query itself.

    This checks SQLite syntax and table/column references, not whether the SQL
    correctly answers its English question. Error lines are one-based.
    """
    database_path = Path(database_path).resolve(strict=True)
    errors = []
    with closing(sqlite3.connect(database_path.as_uri() + '?mode=ro', uri=True)) as conn:
        conn.execute('PRAGMA query_only=ON')
        for idx, query in enumerate(queries):
            try:
                if not query.strip():
                    raise ValueError("Empty SQL query")
                with closing(conn.execute('EXPLAIN ' + query)) as cursor:
                    pass
            except (sqlite3.Error, ValueError) as error:
                errors.append({"line": idx + 1, "query": query, "error": str(error)})
    return {"total": len(queries), "valid": len(queries) - len(errors), "errors": errors}

def format_schema_ddl(database_path):
    """Export CREATE TABLE statements with the types and constraints actually declared.

    SQLite's stored DDL preserves primary/foreign keys, defaults and other table
    constraints. Relationship hints or indexed columns are not inferred keys.
    """
    database_path = Path(database_path).resolve(strict=True)
    with closing(sqlite3.connect(database_path.as_uri() + '?mode=ro', uri=True)) as conn:
        tables = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    return '\n\n'.join(sql.strip().rstrip(';') + ';' for (sql,) in tables) + '\n'

def preprocess_data_files(data_folder, output_folder):
    """Save cleaned splits, SQLite validation reports and a separate schema DDL file."""
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    if data_folder.resolve() == output_folder.resolve():
        raise ValueError("Use a separate output folder to preserve the original data.")
    output_folder.mkdir(parents=True, exist_ok=True)
    database_path = data_folder / 'flight_database.db'
    report = {"validation": "SQLite EXPLAIN; queries are compiled, not executed",
              "invalid_sql_policy": "Retain all paired examples and report errors; no automatic repair",
              "schema_appended_to_inputs": False, "splits": {}}
    for split in ('train', 'dev', 'test'):
        inputs = [clean_text(line) for line in load_lines(data_folder / f'{split}.nl', strip=False)]
        split_report = {"examples": len(inputs)}
        if split != 'test':
            targets = [clean_text(line) for line in load_lines(data_folder / f'{split}.sql', strip=False)]
            if len(inputs) != len(targets):
                raise ValueError(f"{split}: input and SQL files have different numbers of lines")
            split_report["sql_validation"] = validate_sql_queries(targets, database_path)
            (output_folder / f'{split}.sql').write_text('\n'.join(targets) + '\n', encoding='utf-8')
        (output_folder / f'{split}.nl').write_text('\n'.join(inputs) + '\n', encoding='utf-8')
        report["splits"][split] = split_report

    ddl = format_schema_ddl(database_path)
    (output_folder / 'flight_database_ddl.sql').write_text(ddl, encoding='utf-8')
    with closing(sqlite3.connect(database_path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        tables = [name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        primary_keys, foreign_keys = {}, {}
        for name in tables:
            quoted_name = '"' + name.replace('"', '""') + '"'
            columns = conn.execute(f'PRAGMA table_info({quoted_name})').fetchall()
            key = [column[1] for column in sorted(columns, key=lambda column: column[5]) if column[5]]
            if key:
                primary_keys[name] = key
            keys = conn.execute(f'PRAGMA foreign_key_list({quoted_name})').fetchall()
            if keys:
                foreign_keys[name] = keys
    report["schema"] = {"source": "flight_database.db declared CREATE TABLE definitions",
                        "tables": len(tables), "primary_keys": primary_keys, "foreign_keys": foreign_keys,
                        "note": "No foreign-key constraints are declared in this database; relationship hints are not added as constraints." if not foreign_keys else "Declared foreign-key constraints are preserved."}
    schema_path = data_folder / 'flight_database.schema'
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding='utf-8'))
        hints = {"note": "Relationship hints from the supplied schema; not declared foreign-key constraints.",
                 "links": schema.get('links', {})}
        (output_folder / 'schema_relationship_hints.json').write_text(json.dumps(hints, indent=2) + '\n', encoding='utf-8')
    (output_folder / 'preprocessing_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report

def data_statistics(inputs, targets, tokenizer):
    """Measure paired texts in T5 tokens without padding, truncation or added EOS.

    Vocabulary sizes count distinct observed token IDs separately for NL and SQL.
    """
    if not inputs or len(inputs) != len(targets):
        raise ValueError("Input and SQL files must be nonempty and have matching line counts.")

    input_ids = tokenizer(inputs, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    target_ids = tokenizer(targets, add_special_tokens=False, padding=False, truncation=False)["input_ids"]

    return [
        len(inputs),
        sum(len(ids) for ids in input_ids) / len(inputs),
        sum(len(ids) for ids in target_ids) / len(targets),
        len({token for ids in input_ids for token in ids}),
        len({token for ids in target_ids for token in ids}),
    ]

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(os.path.join(data_folder, "train_x.txt"))
    train_y = load_lines(os.path.join(data_folder, "train_y.txt"))
    dev_x = load_lines(os.path.join(data_folder, "dev_x.txt"))
    dev_y = load_lines(os.path.join(data_folder, "dev_y.txt"))
    test_x = load_lines(os.path.join(data_folder, "test_x.txt"))
    test_y = load_lines(os.path.join(data_folder, "test_y.txt"))

    return train_x, train_y, dev_x, dev_y, test_x, test_y
