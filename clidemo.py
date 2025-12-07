from pathlib import Path
from types import SimpleNamespace

import torch
import typer

from module import Tokenizer, init_model_by_key

app = typer.Typer(help="Command-line demo for couplet generation.")


@app.command()
def main(
    path: Path = typer.Option(
        Path("output"),
        "-p",
        "--path",
        help="Directory containing saved model checkpoints.",
    ),
    vocab_path: Path = typer.Option(
        Path("dataset/vocab.pkl"), "--vocab-path", help="Path to the vocab file."
    ),
    model: str = typer.Option(
        "transformer", "-m", "--model", help="Model key to load."
    ),
    epoch: int = typer.Option(0, "-e", "--epoch", help="Epoch checkpoint to load."),
    max_seq_len: int = typer.Option(32, "--max-seq-len"),
    embed_dim: int = typer.Option(128, "--embed-dim"),
    n_layer: int = typer.Option(1, "--n-layer"),
    hidden_dim: int = typer.Option(256, "--hidden-dim"),
    ff_dim: int = typer.Option(512, "--ff-dim"),
    n_head: int = typer.Option(8, "--n-head"),
    embed_drop: float = typer.Option(0.2, "--embed-drop"),
    hidden_drop: float = typer.Option(0.1, "--hidden-drop"),
    stop_flag: str = typer.Option("q", "-s", "--stop-flag", help="Exit token."),
    cuda: bool = typer.Option(False, "-c", "--cuda", help="Enable CUDA if available."),
    mode: str = typer.Option(
        "tagger",
        "--mode",
        "-m",
        help="tagger: 序列标注；concat: 上联+[SEP]+下联 的单流自回归",
    ),
):
    args = SimpleNamespace(
        path=path,
        vocab_path=vocab_path,
        model=model,
        epoch=epoch,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        n_layer=n_layer,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        n_head=n_head,
        embed_drop=embed_drop,
        hidden_drop=hidden_drop,
        stop_flag=stop_flag,
        cuda=cuda,
        mode=mode,
    )
    print("loading tokenizer...")
    tokenizer = Tokenizer.from_pretrained(Path(args.vocab_path))
    print("loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model = init_model_by_key(args, tokenizer)

    if args.epoch == 0:
        state_dict = torch.load(
            Path(args.path) / f"{args.model}.bin", map_location=device
        )
    else:
        state_dict = torch.load(
            Path(args.path) / f"{args.model}_{str(args.epoch)}.bin", map_location=device
        )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    def generate_concat(upper: str):
        """
        concat 模式：上联 + [SEP] 作为上下文，自回归生成等长下联。
        """
        upper_ids = tokenizer.convert_tokens_to_ids(list(upper))
        ctx = upper_ids + [tokenizer.sep_id]
        max_len = len(upper_ids)  # 生成下联长度与上联相同
        for _ in range(max_len):
            inp = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(inp)
            next_id = logits[0, len(ctx) - 1].argmax().item()
            ctx.append(next_id)
        # 下联部分在 [SEP] 之后
        lower_ids = ctx[len(upper_ids) + 1 :]
        # 去掉可能的 PAD
        lower_ids = [i for i in lower_ids if i != tokenizer.pad_id]
        return tokenizer.decode(lower_ids)

    def generate_tagger(upper: str):
        input_ids = torch.tensor(tokenizer.encode(upper)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_ids).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        return tokenizer.decode(pred)

    while True:
        question = input("上联：")
        if question == args.stop_flag.lower():
            print("Thank you!")
            break
        if args.mode == "concat":
            pred = generate_concat(question)
        else:
            pred = generate_tagger(question)
        print(f"下联：{pred}")


if __name__ == "__main__":
    app()
