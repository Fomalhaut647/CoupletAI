from pathlib import Path
from types import SimpleNamespace

import torch
import typer
from flask import Flask, request, render_template

from module import Tokenizer, init_model_by_key

cli = typer.Typer(help="Flask-based web demo for couplet generation.")


class Context(object):
    def __init__(self, args):
        print(f"loading pretrained model from {args.path}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.tokenizer = Tokenizer.from_pretrained(Path(args.vocab_path))
        self.model = init_model_by_key(args, self.tokenizer)
        if args.epoch == 0:
            state_dict = torch.load(
                Path(args.path) / f"{args.model}.bin", map_location=self.device
            )
        else:
            state_dict = torch.load(
                Path(args.path) / f"{args.model}_{str(args.epoch)}.bin",
                map_location=self.device,
            )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, s):
        input_ids = torch.tensor(self.tokenizer.encode(s)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        pred = self.tokenizer.decode(pred)
        return pred


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
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(5000, "--port"),
    cuda: bool = typer.Option(False, "--cuda", help="Enable CUDA if available."),
):
    """
    Launch the Flask web interface for generating couplets.
    """
    global ctx
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
        host=host,
        port=port,
        cuda=cuda,
    )
    ctx = Context(args)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    cli()
