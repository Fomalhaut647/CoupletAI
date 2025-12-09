from typing import List, Tuple
from pathlib import Path
import sys

import torch
import typer
from torch.utils.data import TensorDataset
from rich.progress import track
from rich import print

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
        # input_ids: 上联 token ids
        # target_ids: 下联 token ids
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
    """
    序列标注模式（原始）：返回 (input_ids, masks, lens, target_ids)
    """
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


def convert_features_to_tensors_concat(
    features: List[CoupletFeatures], tokenizer: Tokenizer, max_seq_len: int
):
    """
    单流自回归模式：上联 + [SEP] + 下联，预测下一个 token。
    返回 (input_ids, attn_mask, loss_mask, target_ids)，长度为 2*max_seq_len+1 的序列（去掉最后一位做 next-token）。
    """
    sep_id = tokenizer.token_to_ix["[SEP]"]

    total = len(features)
    max_total_len = max_seq_len * 2 + 1  # 上联 max + SEP + 下联 max
    seq_len = max_total_len - 1  # 输入/目标长度（next-token 预测去掉最后一位）

    input_ids = torch.full((total, seq_len), tokenizer.pad_id, dtype=torch.long)
    target_ids = torch.full((total, seq_len), tokenizer.pad_id, dtype=torch.long)
    attn_mask = torch.ones((total, seq_len), dtype=torch.bool)
    loss_mask = torch.zeros((total, seq_len), dtype=torch.float)

    for i, f in enumerate(track(features, description="creating tensors (concat)")):
        src = f.input_ids[:max_seq_len]
        tgt = f.target_ids[:max_seq_len]
        concat = src + [sep_id] + tgt  # 长度最多 max_total_len

        # next-token 预测：输入为 concat[:-1]，目标为 concat[1:]
        inp = concat[:-1]
        tar = concat[1:]
        real_len = len(inp)

        input_ids[i, :real_len] = torch.tensor(inp[:real_len])
        target_ids[i, :real_len] = torch.tensor(tar[:real_len])
        attn_mask[i, :real_len] = 0  # 0 表示有效

        # loss_mask：只在预测下联部分计入损失。target 索引 j 对应 concat[j+1]
        for j in range(len(tgt)):
            loss_mask[i, j + len(src)] = 1.0

    return input_ids, attn_mask, loss_mask, target_ids


def create_dataset(fdir: Path, tokenizer: Tokenizer, max_seq_len: int, mode: str):
    examples = read_examples(fdir)
    features = convert_examples_to_features(examples, tokenizer)
    if mode == "tagger":
        tensors = convert_features_to_tensors(features, tokenizer, max_seq_len)
    elif mode == "concat":
        tensors = convert_features_to_tensors_concat(features, tokenizer, max_seq_len)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
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
        32, "--max-seq-len", "-l", help="Maximum sequence length."
    ),
    mode: str = typer.Option(
        "concat",
        "--mode",
        "-m",
        help="tagger: 序列标注；concat: 上联+[SEP]+下联 的单流自回归",
    ),
):
    """
    Build tokenizer and dataset tensors from raw couplet data.
    """
    input_dir = Path(input)
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True, parents=True)
    vocab_file = input_dir / "vocabs"

    logger.info("creating tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.build(vocab_file)

    logger.info("creating dataset...")
    train_dataset = create_dataset(input_dir / "train", tokenizer, max_seq_len, mode)
    test_dataset = create_dataset(input_dir / "test", tokenizer, max_seq_len, mode)

    logger.info("saving dataset...")
    tokenizer.save_pretrained(output_dir / "vocab.pkl")
    torch.save(train_dataset, output_dir / "train.pkl")
    torch.save(test_dataset, output_dir / "test.pkl")


if __name__ == "__main__":
    app()
