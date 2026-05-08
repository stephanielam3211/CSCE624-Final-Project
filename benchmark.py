import argparse
import glob
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import RasterDataset, StrokeDataset
from models import RasterCNN, StrokeCNN, StrokeRNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["raster", "stroke", "stroke_rnn"],
        required=True,
    )
    parser.add_argument("--data_dir", type=str, default="../quickdraw_npy")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def make_dataset_and_model(args, files, device):
    if args.mode == "raster":
        dataset = RasterDataset(files)
        model = RasterCNN(len(files)).to(device)
    elif args.mode == "stroke":
        dataset = StrokeDataset(files, L=args.seq_len)
        model = StrokeCNN(len(files)).to(device)
    else:
        dataset = StrokeDataset(files, L=args.seq_len)
        model = StrokeRNN(len(files)).to(device)

    return dataset, model


def main():
    args = parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = sorted(glob.glob(f"{args.data_dir}/*.npy"))[: args.num_classes]
    if len(files) == 0:
        raise RuntimeError(f"No .npy files found in {args.data_dir}")

    dataset, model = make_dataset_and_model(args, files, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    data_iter = iter(loader)

    def get_batch():
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        return batch

    # Warmup
    for _ in range(5):
        x, y = get_batch()
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x)
            loss = F.cross_entropy(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    for _ in range(args.steps):
        x, y = get_batch()
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x)
            loss = F.cross_entropy(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    elapsed = end - start
    total_samples = args.steps * args.batch_size
    throughput = total_samples / elapsed

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Batch size: {args.batch_size}")
    if args.mode in ["stroke", "stroke_rnn"]:
        print(f"Sequence length: {args.seq_len}")
    print(f"Elapsed time: {elapsed:.4f} sec")
    print(f"Throughput: {throughput:.2f} samples/sec")

    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak memory: {mem_mb:.2f} MB")


if __name__ == "__main__":
    main()