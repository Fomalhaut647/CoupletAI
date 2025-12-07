from pathlib import Path
import sys
import time
from types import SimpleNamespace

import typer
from rich.traceback import install
from rich import print
from rich.progress import track
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from nltk.translate.bleu_score import sentence_bleu

from module.model import (
    BiLSTM,
    Transformer,
    CNN,
    BiLSTMAttn,
    BiLSTMCNN,
    BiLSTMConvAttRes,
)
from module import Tokenizer, init_model_by_key
from module.metric import calc_bleu, calc_rouge_l

install(show_locals=True)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
)

torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])

app = typer.Typer(help="Training entrypoint for the couplet generation models.")


def auto_evaluate(model, testloader, tokenizer):
    bleus = []
    rls = []
    device = next(model.parameters()).device
    model.eval()

    for step, batch in enumerate(testloader):
        input_ids, masks, lens = tuple(t.to(device) for t in batch[:-1])
        target_ids = batch[-1]
        with torch.no_grad():
            logits = model(input_ids, masks)
            # preds.shape=(batch_size, max_seq_len)
            _, preds = torch.max(logits, dim=-1)

        for seq, tag in zip(preds.tolist(), target_ids.tolist()):
            seq = list(filter(lambda x: x != tokenizer.pad_id, seq))
            tag = list(filter(lambda x: x != tokenizer.pad_id, tag))
            bleu = calc_bleu(seq, tag)
            rl = calc_rouge_l(seq, tag)
            bleus.append(bleu)
            rls.append(rl)

    return sum(bleus) / len(bleus), sum(rls) / len(rls)


def concat_generate(model, tokenizer: Tokenizer, upper_ids, device):
    """
    给定上联 token id 列表，自回归生成等长下联。
    """
    ctx = list(upper_ids) + [tokenizer.sep_id]
    max_len = len(upper_ids)
    for _ in range(max_len):
        inp = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        mask = torch.zeros((1, inp.size(1)), dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = model(inp, mask)
        next_id = logits[0, len(ctx) - 1].argmax().item()
        ctx.append(next_id)
    lower_ids = ctx[len(upper_ids) + 1 :]
    lower_ids = [i for i in lower_ids if i != tokenizer.pad_id]
    return lower_ids


def auto_evaluate_concat(model, testloader, tokenizer):
    """
    concat 模式的自动评测：从 batch 中恢复上联，生成下联，与目标下联计算 BLEU/ROUGE-L。
    """
    bleus, rls = [], []
    device = next(model.parameters()).device
    model.eval()
    for _, batch in enumerate(testloader):
        input_ids, masks, _, target_ids = batch
        input_ids = input_ids.to(device)
        masks = masks.to(device)
        target_ids = target_ids.to(device)

        bsz, seqlen = input_ids.size()
        for i in range(bsz):
            valid_len = int((masks[i] == 0).sum().item())
            if valid_len == 0:
                continue
            seq = input_ids[i, :valid_len].tolist()
            tar = target_ids[i, :valid_len].tolist()
            if tokenizer.sep_id not in seq:
                continue
            sep_idx = seq.index(tokenizer.sep_id)
            upper_ids = seq[:sep_idx]
            gold_lower = tar[sep_idx:valid_len]
            gold_lower = [t for t in gold_lower if t != tokenizer.pad_id]

            pred_lower = concat_generate(model, tokenizer, upper_ids, device)
            if len(gold_lower) == 0 or len(pred_lower) == 0:
                continue
            bleus.append(calc_bleu(pred_lower, gold_lower))
            rls.append(calc_rouge_l(pred_lower, gold_lower))

    if len(bleus) == 0:
        return 0.0, 0.0
    return sum(bleus) / len(bleus), sum(rls) / len(rls)


def predict_demos_concat(model, tokenizer: Tokenizer):
    demos = [
        "马齿草焉无马齿",
        "天古天今，地中地外，古今中外存天地",
        "笑取琴书温旧梦",
        "日里千人拱手划船，齐歌狂吼川江号子",
        "我有诗情堪纵酒",
        "我以真诚溶冷血",
        "三世业岐黄，妙手回春人共赞",
    ]
    model.eval()
    device = next(model.parameters()).device
    for sent in demos:
        upper_ids = tokenizer.convert_tokens_to_ids(list(sent))
        pred_ids = concat_generate(model, tokenizer, upper_ids, device)
        pred = tokenizer.decode(pred_ids)
        logger.info(f"上联：{sent}。 预测的下联：{pred}")


def predict_demos(model, tokenizer: Tokenizer):
    demos = [
        "马齿草焉无马齿",
        "天古天今，地中地外，古今中外存天地",
        "笑取琴书温旧梦",
        "日里千人拱手划船，齐歌狂吼川江号子",
        "我有诗情堪纵酒",
        "我以真诚溶冷血",
        "三世业岐黄，妙手回春人共赞",
    ]
    sents = [torch.tensor(tokenizer.encode(sent)).unsqueeze(0) for sent in demos]
    model.eval()
    device = next(model.parameters()).device

    for i, sent in enumerate(sents):
        sent = sent.to(device)
        with torch.no_grad():
            logits = model(sent).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        pred = tokenizer.decode(pred)
        logger.info(f"上联：{demos[i]}。 预测的下联：{pred}")


def compute_concat_loss(logits, targets, loss_mask, pad_id):
    """
    适配“上联+[SEP]+下联”单流自回归数据的损失计算。
    只对 loss_mask==1 的位置计入损失，其余位置（上联和分隔符）忽略。
    """
    vocab = logits.size(-1)
    ce = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_id)
    per_token = ce(logits.view(-1, vocab), targets.view(-1)).view_as(targets)
    masked = per_token * loss_mask
    denom = loss_mask.sum().clamp(min=1)
    return masked.sum() / denom


def save_model(filename, model):
    # Only persist the model weights to simplify checkpoint loading.
    torch.save(model.state_dict(), filename)


def run(args):
    fdir = Path(args.dir)
    tb = SummaryWriter(args.logdir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(args)

    logger.info(f"loading vocab...")
    tokenizer = Tokenizer.from_pretrained(fdir / "vocab.pkl")

    logger.info(f"loading dataset...")
    train_dataset = torch.load(fdir / "train.pkl")
    test_dataset = torch.load(fdir / "test.pkl")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(len(train_loader))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    logger.info(f"initializing model...")
    if args.mode == "concat":
        # concat 序列长度约为 2*max_seq_len，调整模型的 pos_embedding 上限以避免越界
        args.max_seq_len = args.max_seq_len * 2
        logger.info(
            f"[concat] reset model max_seq_len to {args.max_seq_len} for pos embedding"
        )
    model = init_model_by_key(args, tokenizer)
    model.to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    # logger.info(f"num gpu: {torch.cuda.device_count()}")
    global_step = 0

    for epoch in range(args.epochs):
        logger.info(f"***** Epoch {epoch} *****")
        model.train()
        t1 = time.time()
        accu_loss = 0.0

        for step, batch in enumerate(
            track(train_loader, description=f"Epoch {epoch + 1}/{args.epochs}")
        ):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            if args.mode == "tagger":
                input_ids, masks, lens, target_ids = batch
                logits = model(input_ids, masks)
                loss = loss_function(
                    logits.view(-1, tokenizer.vocab_size), target_ids.view(-1)
                )
            else:
                # 单流自回归训练路径，batch 形如 (input_ids, attn_mask, loss_mask, targets)
                input_ids, masks, loss_mask, target_ids = batch
                logits = model(input_ids, masks)
                loss = compute_concat_loss(
                    logits, target_ids, loss_mask, tokenizer.pad_id
                )

            accu_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            global_step += 1

        scheduler.step(accu_loss)

        t2 = time.time()
        logger.info(f"epoch time: {t2-t1:.5}, accumulation loss: {accu_loss:.6}")

        if (epoch + 1) % args.test_epoch == 0:
            if args.mode == "tagger":
                predict_demos(model, tokenizer)
                bleu, rl = auto_evaluate(model, test_loader, tokenizer)
                logger.info(f"BLEU: {round(bleu, 9)}, Rouge-L: {round(rl, 8)}")
            else:
                predict_demos_concat(model, tokenizer)
                bleu, rl = auto_evaluate_concat(model, test_loader, tokenizer)
                logger.info(f"[concat] BLEU: {round(bleu, 9)}, Rouge-L: {round(rl, 8)}")

        if (epoch + 1) % args.save_epoch == 0:
            filename = f"{model.__class__.__name__.lower()}_{epoch + 1}.bin"
            filename = output_dir / filename
            save_model(filename, model)

    save_model(output_dir / f"{model.__class__.__name__.lower()}.bin", model)


@app.command()
def main(
    epochs: int = typer.Option(20, "--epochs", "-e", help="Number of training epochs."),
    batch_size: int = typer.Option(1024, "--batch-size", help="Training batch size."),
    max_seq_len: int = typer.Option(
        32, "--max-seq-len", help="Maximum sequence length."
    ),
    lr: float = typer.Option(0.005, "--lr", help="Learning rate."),
    no_cuda: bool = typer.Option(
        False, "--no-cuda", help="Force CPU even if CUDA is available."
    ),
    model: str = typer.Option(
        "transformer", "-m", "--model", help="Model architecture key."
    ),
    max_grad_norm: float = typer.Option(
        3.0, "--max-grad-norm", help="Gradient clipping norm."
    ),
    dir: Path = typer.Option(
        Path("dataset"), "--dir", help="Directory containing dataset tensors."
    ),
    output: Path = typer.Option(
        Path("output"), "--output", help="Directory to store model checkpoints."
    ),
    logdir: Path = typer.Option(
        Path("runs"), "--logdir", help="TensorBoard log directory."
    ),
    embed_dim: int = typer.Option(128, "--embed-dim"),
    n_layer: int = typer.Option(1, "--n-layer"),
    hidden_dim: int = typer.Option(256, "--hidden-dim"),
    ff_dim: int = typer.Option(512, "--ff-dim"),
    n_head: int = typer.Option(8, "--n-head"),
    test_epoch: int = typer.Option(1, "--test-epoch"),
    save_epoch: int = typer.Option(10, "--save-epoch"),
    embed_drop: float = typer.Option(0.2, "--embed-drop"),
    hidden_drop: float = typer.Option(0.1, "--hidden-drop"),
    mode: str = typer.Option(
        "tagger",
        "--mode",
        "-m",
        help="tagger: 旧的序列标注；concat: 上联+[SEP]+下联 的单流自回归训练",
    ),
):
    args = SimpleNamespace(
        epochs=epochs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        lr=lr,
        no_cuda=no_cuda,
        model=model,
        max_grad_norm=max_grad_norm,
        dir=dir,
        output=output,
        logdir=logdir,
        embed_dim=embed_dim,
        n_layer=n_layer,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        n_head=n_head,
        test_epoch=test_epoch,
        save_epoch=save_epoch,
        embed_drop=embed_drop,
        hidden_drop=hidden_drop,
        mode=mode,
    )
    run(args)


if __name__ == "__main__":
    app()
