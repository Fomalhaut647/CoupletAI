from pathlib import Path
import sys
import time
from types import SimpleNamespace

import typer
from rich.traceback import install
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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    logger.info(f"initializing model...")
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
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, lens, target_ids = batch
            logits = model(input_ids, masks)
            loss = loss_function(
                logits.view(-1, tokenizer.vocab_size), target_ids.view(-1)
            )
            # if torch.cuda.device_count() > 1:
            #     loss = loss.mean()
            accu_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            if step % 100 == 0:
                tb.add_scalar("loss", loss.item(), global_step)
                logger.info(f"[epoch]: {epoch}, [batch]: {step}, [loss]: {loss.item()}")
            global_step += 1
        scheduler.step(accu_loss)
        t2 = time.time()
        logger.info(f"epoch time: {t2-t1:.5}, accumulation loss: {accu_loss:.6}")
        if (epoch + 1) % args.test_epoch == 0:
            predict_demos(model, tokenizer)
            bleu, rl = auto_evaluate(model, test_loader, tokenizer)
            logger.info(f"BLEU: {round(bleu, 9)}, Rouge-L: {round(rl, 8)}")
        if (epoch + 1) % args.save_epoch == 0:
            filename = f"{model.__class__.__name__.lower()}_{epoch + 1}.bin"
            filename = output_dir / filename
            save_model(filename, model)

    save_model(output_dir / f"{model.__class__.__name__.lower()}.bin", model)


@app.command()
def main(
    epochs: int = typer.Option(20, "--epochs", "-e", help="Number of training epochs."),
    batch_size: int = typer.Option(768, "--batch-size", help="Training batch size."),
    max_seq_len: int = typer.Option(
        32, "--max-seq-len", help="Maximum sequence length."
    ),
    lr: float = typer.Option(0.001, "--lr", help="Learning rate."),
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
    )
    run(args)


if __name__ == "__main__":
    app()
