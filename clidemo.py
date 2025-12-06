import argparse
from pathlib import Path

import torch
from module import Tokenizer, init_model_by_key


def run():
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
    parser.add_argument("-s", "--stop_flag", default="q", type=str)
    parser.add_argument("-c", "--cuda", action="store_true")
    args = parser.parse_args()
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
    while True:
        question = input("上联：")
        if question == args.stop_flag.lower():
            print("Thank you!")
            break
        input_ids = torch.tensor(tokenizer.encode(question)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_ids).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        pred = tokenizer.decode(pred)
        print(f"下联：{pred}")


if __name__ == "__main__":
    run()
