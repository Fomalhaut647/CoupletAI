import argparse
from pathlib import Path

import torch
from flask import Flask, request, render_template

from module import Tokenizer, init_model_by_key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default="output", type=str)
    parser.add_argument("--vocab-path", default="dataset/vocab.pkl", type=str)
    parser.add_argument("-m", "--model", default="transformer", type=str)
    parser.add_argument("-e", "--epoch", default=0, type=int)
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--n_layer", default=1, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--ff_dim", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--embed_drop", default=0.2, type=float)
    parser.add_argument("--hidden_drop", default=0.1, type=float)
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


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


if __name__ == "__main__":
    args = parse_args()
    ctx = Context(args)
    app.run(host=args.host, port=args.port)
