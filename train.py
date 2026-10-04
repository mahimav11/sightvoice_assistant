"""SightVoice v2 - training script (run on Kaggle / Colab GPU).

What is different from the original notebook
  * Step-by-step attention (the decoder re-attends at every word) - matches inference.
  * ResNet BatchNorm stays frozen in eval mode.
  * Train / validation / test split BY IMAGE (no caption leakage), early stopping
    on validation loss, best checkpoint saved, BLEU-1..4 reported on the test split.
  * Image-level batches: each image goes through ResNet once and is shared by all
    of its ~5 captions (about 5x less encoder work per epoch).
  * Data augmentation, label smoothing, dropout, gradient clipping, mixed precision.
  * Can train on several datasets at once (Flickr8k + Flickr30k + COCO captions).
  * Saves ONE small file (sightvoice_v2.pt, vocab inside) - no 100 MB encoder needed.

Kaggle examples
    !python train.py                                   # Flickr8k, defaults
    !python train.py --dataset f8=/kaggle/input/flickr8k/Images:/kaggle/input/flickr8k/captions.txt \
                     --dataset f30=/kaggle/input/flickr30k/images:/kaggle/input/flickr30k/results.csv
In a notebook cell you can also do:  from train import main; main([...])
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import EOS, PAD, SOS, DecoderRNN, EncoderCNN, Vocabulary, beam_search

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DEFAULT_F8K = ("flickr8k=/kaggle/input/datasets/mahimavahadne/flickr8k/Images:"
               "/kaggle/input/datasets/mahimavahadne/flickr8k/captions.txt")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_captions(images_dir, captions_file):
    """Auto-detects Flickr8k (captions.txt), Flickr30k (results.csv with '|'),
    and COCO (captions_*.json). Returns {image_path: [caption, ...]}."""
    images_dir, captions_file = Path(images_dir), Path(captions_file)
    data = {}
    if captions_file.suffix == ".json":  # COCO
        js = json.load(open(captions_file, encoding="utf-8"))
        id2file = {im["id"]: im["file_name"] for im in js["images"]}
        for a in js["annotations"]:
            data.setdefault(images_dir / id2file[a["image_id"]], []).append(a["caption"])
        return data
    with open(captions_file, encoding="utf-8") as f:
        first = f.readline()
    if "|" in first:  # Flickr30k
        df = pd.read_csv(captions_file, sep="|", skipinitialspace=True, on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]
        img_col, cap_col = df.columns[0], df.columns[2]
    else:             # Flickr8k
        df = pd.read_csv(captions_file)
        img_col, cap_col = "image", "caption"
    df = df.dropna(subset=[img_col, cap_col])
    for img, cap in zip(df[img_col].astype(str).str.strip(), df[cap_col].astype(str).str.strip()):
        if not img.lower().endswith((".jpg", ".jpeg", ".png")):
            img += ".jpg"
        data.setdefault(images_dir / img, []).append(cap)
    return data


class ImageCaptionDataset(Dataset):
    """One item = one image + ALL its captions (already converted to token ids)."""

    def __init__(self, items, vocab, transform, max_tokens=30):
        self.paths = [p for p, _ in items]
        self.caps = []
        sos, eos = vocab.stoi[SOS], vocab.stoi[EOS]
        for _, caps in items:
            ids = [torch.tensor([sos] + vocab.numericalize(c)[:max_tokens] + [eos]) for c in caps]
            self.caps.append(ids)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.caps[i]


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    caps, counts = [], []
    for _, cs in batch:
        caps.extend(cs)
        counts.append(len(cs))
    return imgs, pad_sequence(caps, batch_first=True, padding_value=0), torch.tensor(counts)


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
def run_epoch(loader, encoder, decoder, criterion, optimizer, scaler, device, train, use_amp, params):
    encoder.train(train)
    decoder.train(train)
    V = decoder.fc.out_features
    total, batches = 0.0, 0
    with torch.set_grad_enabled(train):
        for imgs, caps, counts in tqdm(loader, leave=False, desc="train" if train else "val"):
            imgs, caps, counts = imgs.to(device), caps.to(device), counts.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                feats = encoder(imgs).repeat_interleave(counts, dim=0)
                logits, _ = decoder(feats, caps[:, :-1])
                loss = criterion(logits.float().reshape(-1, V), caps[:, 1:].reshape(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(optimizer)
                scaler.update()
            total += loss.item()
            batches += 1
    return total / max(batches, 1)


def make_checkpoint(encoder, decoder, vocab, cfg, history):
    return {
        "format": "sightvoice_v2",
        "embed": {k: v.cpu() for k, v in encoder.embed.state_dict().items()},
        "decoder": {k: v.cpu() for k, v in decoder.state_dict().items()},
        "vocab_itos": vocab.to_list(),
        "config": cfg,
        "history": history,
    }


@torch.no_grad()
def evaluate_bleu(encoder, decoder, vocab, items, transform, device, beam_size, limit):
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError:
        print("nltk not installed - skipping BLEU (pip install nltk)")
        return None
    encoder.eval()
    decoder.eval()
    refs, hyps = [], []
    for path, caps in tqdm(items[:limit], desc="BLEU"):
        img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        best = beam_search(decoder, encoder(img), vocab, beam_size=beam_size)[0]
        hyps.append(best["words"])
        refs.append([vocab.tokenizer(c) for c in caps])
    sm = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        w = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f"BLEU-{n}"] = round(corpus_bleu(refs, hyps, weights=w, smoothing_function=sm), 4)
    return scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append",
                    help="NAME=IMAGES_DIR:CAPTIONS_FILE (repeatable). Default: your Kaggle Flickr8k.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32, help="images per batch (x ~5 captions each)")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--embed-size", type=int, default=256)
    ap.add_argument("--hidden-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--beam-size", type=int, default=3)
    ap.add_argument("--bleu-images", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images")
    ap.add_argument("--no-pretrained", action="store_true", help="debug only (no ImageNet weights)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | AMP: {use_amp}")

    # ---- data ------------------------------------------------------------
    specs = args.dataset or [DEFAULT_F8K]
    data = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        img_dir, cap_file = rest.rsplit(":", 1)
        d = load_captions(img_dir, cap_file)
        print(f"  {name}: {len(d)} images, {sum(len(v) for v in d.values())} captions")
        data.update(d)
    items = sorted(data.items(), key=lambda kv: str(kv[0]))
    random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[: args.limit]
    n_val = max(1, int(len(items) * args.val_frac))
    val_items, test_items, train_items = items[:n_val], items[n_val:2 * n_val], items[2 * n_val:]
    print(f"Split by image -> train {len(train_items)} | val {len(val_items)} | test {len(test_items)}")

    vocab = Vocabulary(freq_threshold=args.min_freq)  # built from TRAIN captions only
    vocab.build_vocabulary([c for _, caps in train_items for c in caps])
    print(f"Vocabulary size: {len(vocab)}")

    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.25, 0.25, 0.2),
        T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])

    mk = lambda its, tf, sh: DataLoader(
        ImageCaptionDataset(its, vocab, tf), batch_size=args.batch_size, shuffle=sh,
        collate_fn=collate, num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0)
    train_loader, val_loader = mk(train_items, train_tf, True), mk(val_items, eval_tf, False)

    # ---- model -----------------------------------------------------------
    encoder = EncoderCNN(args.embed_size, pretrained=not args.no_pretrained).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), dropout=args.dropout).to(device)
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    train_crit = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
    val_crit = nn.CrossEntropyLoss(ignore_index=0)  # plain NLL so val loss is a true perplexity
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    cfg = dict(embed_size=args.embed_size, hidden_size=args.hidden_size, vocab_size=len(vocab),
               attention="dynamic", image_size=224)
    best, bad, history = float("inf"), 0, []
    ckpt_path = out_dir / "sightvoice_v2.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, encoder, decoder, train_crit, optimizer, scaler, device, True, use_amp, params)
        va = run_epoch(val_loader, encoder, decoder, val_crit, optimizer, scaler, device, False, use_amp, params)
        scheduler.step(va)
        history.append(dict(epoch=epoch, train_loss=round(tr, 4), val_loss=round(va, 4),
                            val_ppl=round(float(np.exp(va)), 2)))
        flag = ""
        if va < best:
            best, bad = va, 0
            torch.save(make_checkpoint(encoder, decoder, vocab, cfg, history), ckpt_path)
            flag = "  <- saved best"
        else:
            bad += 1
        print(f"Epoch {epoch:02d}/{args.epochs} | train {tr:.3f} | val {va:.3f} "
              f"(ppl {np.exp(va):.1f}) | {time.time() - t0:.0f}s{flag}")
        if bad >= args.patience:
            print("Early stopping (validation loss stopped improving).")
            break

    # ---- final test metrics on the best checkpoint -------------------------------
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    encoder.embed.load_state_dict(ck["embed"])
    decoder.load_state_dict(ck["decoder"])
    metrics = evaluate_bleu(encoder, decoder, vocab, test_items, eval_tf, device,
                            args.beam_size, args.bleu_images)
    if metrics:
        print("Test BLEU:", metrics)
    with open(out_dir / "training_report.json", "w") as f:
        json.dump(dict(history=history, test_bleu=metrics, best_val_loss=round(best, 4),
                       args=vars(args)), f, indent=2)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab.to_list(), f)
    print(f"Done. Copy {ckpt_path.name} next to app.py.")


if __name__ == "__main__":
    main()
