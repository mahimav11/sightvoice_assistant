"""SightVoice - shared model code.

Both train.py and app.py import from this file. In the original project the model
was defined twice (notebook + app.py) and the two copies behaved differently,
which is a very common reason for "works on training images, bad on new ones".
Keep ONE definition here and the two can never drift apart again.

Two attention modes are supported by the same set of weights:
  * static_context=True  -> exactly what the ORIGINAL notebook trained
                            (attention computed once from a zero hidden state).
                            Use this to run your existing encoder.pth/decoder.pth.
  * static_context=False -> proper Show-Attend-Tell attention, recomputed at every
                            word. Use this for models trained with the new train.py.
"""
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"


# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: PAD, 1: SOS, 2: EOS, 3: UNK}
        self.stoi = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = len(self.itos)
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        return [self.stoi.get(tok, self.stoi[UNK]) for tok in self.tokenizer(text)]

    def to_list(self):
        return [self.itos[i] for i in range(len(self.itos))]

    @classmethod
    def from_list(cls, itos_list):
        v = cls()
        v.itos = {i: w for i, w in enumerate(itos_list)}
        v.stoi = {w: i for i, w in enumerate(itos_list)}
        return v


class _VocabUnpickler(pickle.Unpickler):
    """Your old vocab.pkl was pickled from a notebook, so it points at
    `__main__.Vocabulary`. This maps it to the class above so it loads anywhere."""

    def find_class(self, module, name):
        if name == "Vocabulary":
            return Vocabulary
        return super().find_class(module, name)


def load_vocab_pickle(path):
    with open(path, "rb") as f:
        return _VocabUnpickler(f).load()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class EncoderCNN(nn.Module):
    """Frozen ResNet-50 -> 49 spatial features -> linear projection."""

    def __init__(self, embed_size, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)
        for p in resnet.parameters():
            p.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.embed = nn.Linear(2048, embed_size)

    def train(self, mode=True):
        # The ResNet is frozen, so its BatchNorm layers must stay in eval mode.
        # (The original notebook called encoder.train(), which silently kept
        # updating BatchNorm statistics with Flickr8k batches.)
        super().train(mode)
        self.resnet.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            f = self.resnet(images)                      # (B, 2048, 7, 7)
        f = f.permute(0, 2, 3, 1)                        # (B, 7, 7, 2048)
        f = f.reshape(f.size(0), -1, f.size(3))          # (B, 49, 2048)
        return self.embed(f)                             # (B, 49, E)


class BahdanauAttention(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(embed_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, features, hidden):
        score = torch.tanh(self.W1(features) + self.W2(hidden.unsqueeze(1)))
        weights = torch.softmax(self.V(score), dim=1)    # (B, 49, 1)
        context = (weights * features).sum(dim=1)        # (B, E)
        return context, weights


class DecoderRNN(nn.Module):
    """Same layer names/shapes as the original decoder, so old weights still load."""

    def __init__(self, embed_size, hidden_size, vocab_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(embed_size, hidden_size)
        self.gru = nn.GRU(embed_size + embed_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def step(self, features, hidden, word, context=None):
        """One decoding step. Input order is [context, word_embedding] - the same
        order used in training. (The old app.py used [word, context], which feeds
        the GRU the wrong inputs.)"""
        emb = self.embed(word)
        alpha = None
        if context is None:
            context, alpha = self.attention(features, hidden)
        gru_in = torch.cat((context, emb), dim=1).unsqueeze(1)
        out, h = self.gru(gru_in, hidden.unsqueeze(0).contiguous())
        logits = self.fc(self.dropout(out.squeeze(1)))
        return logits, h.squeeze(0), alpha

    def forward(self, features, captions_in, static_context=False):
        """Teacher-forced training pass. Returns (logits (B,T,V), alphas or None)."""
        B, T = captions_in.shape
        hidden = features.new_zeros(B, self.hidden_size)
        if static_context:  # legacy behaviour of the original notebook
            ctx, _ = self.attention(features, hidden)
            emb = self.embed(captions_in)
            gru_in = torch.cat((ctx.unsqueeze(1).expand(-1, T, -1), emb), dim=2)
            out, _ = self.gru(gru_in, hidden.unsqueeze(0))
            return self.fc(self.dropout(out)), None
        logits, alphas = [], []
        for t in range(T):
            lg, hidden, a = self.step(features, hidden, captions_in[:, t])
            logits.append(lg)
            alphas.append(a)
        return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)


# ----------------------------------------------------------------------------
# Beam search
# ----------------------------------------------------------------------------
@torch.no_grad()
def beam_search(decoder, features, vocab, beam_size=3, max_len=20,
                static_context=False, length_alpha=0.7):
    """features: (1, 49, E). beam_size=1 is greedy decoding.

    Returns a list of hypotheses (best first), each a dict:
      text, words, avg_logprob, confidence (geometric-mean token probability),
      attention (np.ndarray (n_words, 49) or None in static mode).
    """
    device = features.device
    sos, eos, pad = vocab.stoi[SOS], vocab.stoi[EOS], vocab.stoi[PAD]
    V = decoder.fc.out_features

    live = [dict(seq=[sos], score=0.0, alphas=[])]
    hidden = torch.zeros(1, decoder.hidden_size, device=device)
    ctx1 = None
    if static_context:
        ctx1, _ = decoder.attention(features, hidden)

    finished = []
    for _ in range(max_len):
        n = len(live)
        words = torch.tensor([h["seq"][-1] for h in live], device=device)
        scores = torch.tensor([h["score"] for h in live], device=device)
        ctx = ctx1.expand(n, -1) if ctx1 is not None else None
        logits, new_hidden, alpha = decoder.step(features.expand(n, -1, -1), hidden, words, context=ctx)

        logp = F.log_softmax(logits.float(), dim=-1)
        logp[:, pad] = float("-inf")
        logp[:, sos] = float("-inf")
        cand = (scores.unsqueeze(1) + logp).reshape(-1)
        top_scores, top_idx = cand.topk(min(beam_size, cand.numel()))

        next_live, keep_rows = [], []
        for sc, idx in zip(top_scores.tolist(), top_idx.tolist()):
            b, tok = divmod(idx, V)
            al = live[b]["alphas"] + ([alpha[b].squeeze(-1)] if alpha is not None else [])
            hyp = dict(seq=live[b]["seq"] + [tok], score=sc, alphas=al)
            if tok == eos:
                finished.append(hyp)
            else:
                next_live.append(hyp)
                keep_rows.append(b)
        if len(finished) >= beam_size or not next_live:
            live = next_live
            break
        live = next_live
        hidden = new_hidden[keep_rows]

    if len(finished) < beam_size:   # hit max_len: also offer the unfinished beams
        finished = finished + live

    out = []
    for h in finished:
        ended = h["seq"][-1] == eos
        tokens = h["seq"][1:-1] if ended else h["seq"][1:]
        n_scored = len(tokens) + (1 if ended else 0)
        avg_lp = h["score"] / max(n_scored, 1)
        att = None
        if h["alphas"]:
            att = torch.stack(h["alphas"][: len(tokens)]).cpu().numpy() if tokens else None
        out.append(dict(
            words=[vocab.itos[t] for t in tokens],
            text=" ".join(vocab.itos[t] for t in tokens),
            avg_logprob=avg_lp,
            confidence=float(np.exp(avg_lp)),
            attention=att,
            _rank=h["score"] / (max(n_scored, 1) ** length_alpha),
        ))
    out.sort(key=lambda d: d["_rank"], reverse=True)
    for d in out:
        d.pop("_rank")
    return out[:beam_size]
