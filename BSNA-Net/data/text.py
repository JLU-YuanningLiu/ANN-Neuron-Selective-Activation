import csv
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, random_split


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def clean_text(text):
    return " ".join(text.lower().strip().split())


def tokenize(text):
    return TOKEN_PATTERN.findall(clean_text(text))


class Vocabulary:
    def __init__(self, max_size=50000, min_freq=1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        for token, freq in counter.most_common(self.max_size - 2):
            if freq < self.min_freq:
                break
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def encode(self, text, max_length=256):
        ids = [self.stoi.get(t, 1) for t in tokenize(text)][:max_length]
        mask = [1] * len(ids)
        pad = max_length - len(ids)
        ids.extend([0] * pad)
        mask.extend([0] * pad)
        return ids, mask

    def __len__(self):
        return len(self.itos)


class SimpleTextDataset(Dataset):
    def __init__(self, rows, vocab, max_length=256):
        self.rows = rows
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        text, label = self.rows[index]
        ids, mask = self.vocab.encode(text, self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class TransformerTextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        text, label = self.rows[index]
        encoded = self.tokenizer(clean_text(text), max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def read_csv_rows(path, text_column="text", label_column="label"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row[text_column], int(row[label_column])))
    return rows


def build_simple_text_loaders(path, batch_size, val_ratio=0.1, max_length=256, vocab_size=50000, workers=4, seed=0):
    rows = read_csv_rows(path)
    vocab = Vocabulary(vocab_size)
    vocab.fit([x[0] for x in rows])
    dataset = SimpleTextDataset(rows, vocab, max_length)
    n_val = int(round(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, vocab


def build_transformer_text_loaders(path, tokenizer, batch_size=32, val_ratio=0.1, max_length=256, workers=4, seed=0):
    rows = read_csv_rows(path)
    dataset = TransformerTextDataset(rows, tokenizer, max_length)
    n_val = int(round(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader
