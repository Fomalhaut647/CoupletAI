from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import json
import random
import numpy as np

import torch
import typer
from flask import Flask, request, render_template

from module import Tokenizer, init_model_by_key

cli = typer.Typer(help="Flask-based web demo for couplet generation.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path):
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"failed to load config {config_path}: {e}")
    return {}


def resolve_param(name, cli_value, config, default):
    return cli_value if cli_value is not None else config.get(name, default)


def checkpoint_stem(args):
    return f"{args.model}_ed{args.embed_dim}_hd{args.hidden_dim}_nl{args.n_layer}_ms{args.max_seq_len}"


def checkpoint_filename(args, epoch: int | None = None):
    stem = checkpoint_stem(args)
    if epoch is None:
        return f"{stem}.bin"
    return f"{stem}_ep{epoch}.bin"


def load_checkpoint(device, args):
    candidates = []
    if args.epoch == 0:
        candidates.append(checkpoint_filename(args))
        candidates.append(f"{args.model}.bin")  # legacy
    else:
        candidates.append(checkpoint_filename(args, args.epoch))
        candidates.append(f"{args.model}_{args.epoch}.bin")  # legacy

    for name in candidates:
        path = Path(args.path) / name
        if path.exists():
            return torch.load(path, map_location=device)

    raise FileNotFoundError(
        f"no checkpoint found under {args.path}, tried: {', '.join(candidates)}"
    )


class Context(object):
    def __init__(self, args):
        print(f"loading pretrained model from {args.path}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.tokenizer = Tokenizer.from_pretrained(Path(args.vocab_path))
        self.mode = args.mode
        self.model = init_model_by_key(args, self.tokenizer)
        state_dict = load_checkpoint(self.device, args)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, s):
        if self.mode == "concat":
            return self._predict_concat(s)
        return self._predict_tagger(s)

    def _predict_tagger(self, s):
        input_ids = torch.tensor(self.tokenizer.encode(s)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        pred = self.tokenizer.decode(pred)
        return pred

    def _predict_concat(self, s):
        upper_ids = self.tokenizer.convert_tokens_to_ids(list(s))
        ctx_ids = upper_ids + [self.tokenizer.sep_id]
        max_len = len(upper_ids)
        for _ in range(max_len):
            inp = torch.tensor(ctx_ids, dtype=torch.long, device=self.device).unsqueeze(
                0
            )
            with torch.no_grad():
                logits = self.model(inp)
            next_id = logits[0, len(ctx_ids) - 1].argmax().item()
            ctx_ids.append(next_id)
        lower_ids = ctx_ids[len(upper_ids) + 1 :]
        lower_ids = [i for i in lower_ids if i != self.tokenizer.pad_id]
        return self.tokenizer.decode(lower_ids)


app = Flask(__name__)
ctx = None


@app.route("/<coupletup>")
def api(coupletup):
    return ctx.predict(coupletup)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    coupletup = request.form.get("coupletup")
    coupletdown = ctx.predict(coupletup)
    return render_template("index.html", coupletdown=coupletdown)


@cli.command()
def main(
    path: Path = typer.Option(
        Path("output"),
        "-p",
        "--path",
        help="Directory containing saved model checkpoints.",
    ),
    vocab_path: Optional[Path] = typer.Option(
        None, "--vocab-path", help="Path to the vocab file."
    ),
    model: Optional[str] = typer.Option(
        None, "-m", "--model", help="Model key to load."
    ),
    epoch: int = typer.Option(0, "-e", "--epoch", help="Epoch checkpoint to load."),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(5000, "--port"),
    cuda: bool = typer.Option(False, "--cuda", help="Enable CUDA if available."),
    max_seq_len: Optional[int] = typer.Option(None, "--max-seq-len"),
    embed_dim: Optional[int] = typer.Option(None, "--embed-dim"),
    hidden_dim: Optional[int] = typer.Option(None, "--hidden-dim"),
    ff_dim: Optional[int] = typer.Option(None, "--ff-dim"),
    n_layer: Optional[int] = typer.Option(None, "--n-layer"),
    n_head: Optional[int] = typer.Option(None, "--n-head"),
    embed_drop: Optional[float] = typer.Option(None, "--embed-drop"),
    hidden_drop: Optional[float] = typer.Option(None, "--hidden-drop"),
    seed: Optional[int] = typer.Option(None, "--seed"),
):
    """
    Launch the Flask web interface for generating couplets.
    """
    global ctx

    config = load_config(Path(path) / "config.json")

    model = resolve_param("model", model, config, "transformer")
    mode = resolve_param("mode", None, config, "concat" if model == "gru" else "tagger")
    seed = resolve_param("seed", seed, config, 42)

    vocab_path = resolve_param(
        "vocab_path", vocab_path, config, Path("dataset/vocab.pkl")
    )
    vocab_path = Path(vocab_path)

    embed_dim = resolve_param("embed_dim", embed_dim, config, 128)
    hidden_dim = resolve_param("hidden_dim", hidden_dim, config, 256)
    ff_dim = resolve_param("ff_dim", ff_dim, config, 512)
    n_layer = resolve_param("n_layer", n_layer, config, 1)
    n_head = resolve_param("n_head", n_head, config, 8)
    embed_drop = resolve_param("embed_drop", embed_drop, config, 0.2)
    hidden_drop = resolve_param("hidden_drop", hidden_drop, config, 0.1)

    cli_max_seq_len = max_seq_len
    max_seq_len = resolve_param("max_seq_len", max_seq_len, config, 32)
    if (not config) or (cli_max_seq_len is not None):
        if mode == "concat":
            max_seq_len = max_seq_len * 2

    set_seed(seed)

    args = SimpleNamespace(
        path=path,
        vocab_path=vocab_path,
        model=model,
        epoch=epoch,
        host=host,
        port=port,
        cuda=cuda,
        mode=mode,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        n_layer=n_layer,
        n_head=n_head,
        embed_drop=embed_drop,
        hidden_drop=hidden_drop,
    )
    ctx = Context(args)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    cli()
