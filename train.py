import argparse
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def make_dataset_and_model(args, files):
    if args.mode == "raster":
        dataset = RasterDataset(files)
        model = RasterCNN(len(files))
    elif args.mode == "stroke":
        dataset = StrokeDataset(files, L=args.seq_len)
        model = StrokeCNN(len(files))
    else:
        dataset = StrokeDataset(files, L=args.seq_len)
        model = StrokeRNN(len(files))

    return dataset, model


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = loss_fn(out, y)

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    files = sorted(glob.glob(f"{args.data_dir}/*.npy"))[: args.num_classes]
    if len(files) == 0:
        raise RuntimeError(f"No .npy files found in {args.data_dir}")
    if len(files) < args.num_classes:
        print(f"Warning: requested {args.num_classes} classes but found only {len(files)} files.")

    dataset, model = make_dataset_and_model(args, files)
    model = model.to(device)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Classes used: {len(files)}")
    print(f"Dataset size: {len(dataset)}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    save_path = f"{args.mode}_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()