from typing import List, Tuple
from pathlib import Path
import sys

import torch
import typer
from torch.utils.data import TensorDataset
from rich.progress import track

from module import Tokenizer
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}",
)

app = typer.Typer(help="Couplet dataset preprocessing utilities.")


class CoupletExample(object):
    def __init__(self, seq: List[str], tag: List[str]):
        assert len(seq) == len(tag)
        self.seq = seq
        self.tag = tag


class CoupletFeatures(object):
    def __init__(self, input_ids: List[int], target_ids: List[int]):
        self.input_ids = input_ids
        self.target_ids = target_ids


def read_examples(fdir: Path):
    seqs = []
    tags = []
    with open(fdir / "in.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            seqs.append(line.split())
    with open(fdir / "out.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            tags.append(line.split())
    examples = [CoupletExample(seq, tag) for seq, tag in zip(seqs, tags)]
    return examples


def convert_examples_to_features(examples: List[CoupletExample], tokenizer: Tokenizer):
    features = []
    for example in track(examples, description="creating features"):
        seq_ids = tokenizer.convert_tokens_to_ids(example.seq)
        tag_ids = tokenizer.convert_tokens_to_ids(example.tag)
        features.append(CoupletFeatures(seq_ids, tag_ids))
    return features


def convert_features_to_tensors(
    features: List[CoupletFeatures], tokenizer: Tokenizer, max_seq_len: int
):
    total = len(features)
    input_ids = torch.full((total, max_seq_len), tokenizer.pad_id, dtype=torch.long)
    target_ids = torch.full((total, max_seq_len), tokenizer.pad_id, dtype=torch.long)
    masks = torch.ones(total, max_seq_len, dtype=torch.bool)
    lens = torch.zeros(total, dtype=torch.long)
    for i, f in enumerate(track(features, description="creating tensors")):
        real_len = min(len(f.input_ids), max_seq_len)
        input_ids[i, :real_len] = torch.tensor(f.input_ids[:real_len])
        target_ids[i, :real_len] = torch.tensor(f.target_ids[:real_len])
        masks[i, :real_len] = 0
        lens[i] = real_len
    return input_ids, masks, lens, target_ids


def create_dataset(fdir: Path, tokenizer: Tokenizer, max_seq_len: int):
    examples = read_examples(fdir)
    features = convert_examples_to_features(examples, tokenizer)
    tensors = convert_features_to_tensors(features, tokenizer, max_seq_len)
    dataset = TensorDataset(*tensors)
    return dataset


@app.command()
def main(
    input: Path = typer.Option(
        Path("couplet"),
        "--input",
        "-i",
        help="Directory containing input data (with in.txt/out.txt).",
    ),
    output: Path = typer.Option(
        Path("dataset"),
        "--output",
        "-o",
        help="Directory to write processed dataset artifacts.",
    ),
    max_seq_len: int = typer.Option(
        32, "--max-seq-len", "-m", help="Maximum sequence length."
    ),
):
    """
    Build tokenizer and dataset tensors from raw couplet data.
    """
    input_dir = Path(input)
    output_dir = Path(output)
    output_dir.parent.mkdir(exist_ok=True, parents=True)
    vocab_file = input_dir / "vocabs"

    logger.info("creating tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.build(vocab_file)

    logger.info("creating dataset...")
    train_dataset = create_dataset(input_dir / "train", tokenizer, max_seq_len)
    test_dataset = create_dataset(input_dir / "test", tokenizer, max_seq_len)

    logger.info("saving dataset...")
    tokenizer.save_pretrained(output_dir / "vocab.pkl")
    torch.save(train_dataset, output_dir / "train.pkl")
    torch.save(test_dataset, output_dir / "test.pkl")


if __name__ == "__main__":
    app()
