from typing import List

import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sf = SmoothingFunction()

try:
    from torchmetrics.text.bleu import BLEUScore  # type: ignore
    from torchmetrics.text.rouge import ROUGEScore  # type: ignore

    _has_torchmetrics = True
except ImportError:
    BLEUScore = None
    ROUGEScore = None
    _has_torchmetrics = False


def calc_bleu(cand: List[int | str], ref: List[int | str]):
    return sentence_bleu([ref], cand, smoothing_function=sf.method1)


def calc_bleu_batch_gpu(
    cands: List[List[int | str]], refs: List[List[int | str]], device: torch.device
):
    """
    Corpus BLEU on GPU using torchmetrics (if available). Falls back to None otherwise.
    """
    if not _has_torchmetrics or device.type != "cuda":
        return None

    preds = [" ".join(map(str, seq)) for seq in cands]
    targets = [[" ".join(map(str, ref))] for ref in refs]
    metric = BLEUScore(n_gram=4, smooth=True).to(device)
    with torch.no_grad():
        score = metric(preds, targets).item()
    return score


def calc_rouge_l(cand: List[int | str], ref: List[int | str], beta: float = 1.2):
    len_cand = len(cand)
    len_ref = len(ref)
    lengths = [[0 for j in range(len_ref + 1)] for i in range(len_cand + 1)]
    for i in range(len_cand):
        for j in range(len_ref):
            if cand[i] == ref[j]:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            elif lengths[i + 1][j] > lengths[i][j + 1]:
                lengths[i + 1][j + 1] = lengths[i + 1][j]
            else:
                lengths[i + 1][j + 1] = lengths[i][j + 1]
    lcs = lengths[-1][-1]
    eps = 1e-10
    r = lcs * 1.0 / (eps + len_ref)
    p = lcs * 1.0 / (eps + len_cand)
    f = ((1 + beta**2) * r * p) / (eps + r + beta**2 * p)
    return f


def calc_rouge_l_batch_gpu(
    cands: List[List[int | str]], refs: List[List[int | str]], device: torch.device
):
    """
    Corpus Rouge-L on GPU using torchmetrics (if available). Falls back to None otherwise.
    """
    if not _has_torchmetrics or device.type != "cuda":
        return None

    preds = [" ".join(map(str, seq)) for seq in cands]
    targets = [" ".join(map(str, ref)) for ref in refs]
    metric = ROUGEScore(rouge_keys=("rougeL",)).to(device)
    with torch.no_grad():
        score = metric(preds, targets)["rougeL_fmeasure"].item()
    return score
