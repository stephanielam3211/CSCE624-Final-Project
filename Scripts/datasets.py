import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


def rasterize(drawing, size=64):
    img = Image.new("L", (256, 256), 0)
    draw = ImageDraw.Draw(img)

    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        points = list(zip(xs, ys))
        if len(points) > 1:
            draw.line(points, fill=255, width=2)

    img = img.resize((size, size))
    arr = np.array(img) / 255.0
    return arr[None, :, :]


def strokes_to_seq(drawing):
    seq = []
    prev_x, prev_y = 0, 0

    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        for i, (x, y) in enumerate(zip(xs, ys)):
            dx = x - prev_x
            dy = y - prev_y
            pen = 1 if i == len(xs) - 1 else 0

            seq.append([dx, dy, pen])
            prev_x, prev_y = x, y

    return np.array(seq, dtype=np.float32)


class RasterDataset(Dataset):
    def __init__(self, files):
        self.data = []
        self.labels = []

        for label, f in enumerate(files):
            drawings = np.load(f, allow_pickle=True)
            for d in drawings:
                self.data.append(rasterize(d))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.tensor(self.data[i], dtype=torch.float32), self.labels[i]


class StrokeDataset(Dataset):
    def __init__(self, files, L=150):
        self.data = []
        self.labels = []
        self.L = L

        for label, f in enumerate(files):
            drawings = np.load(f, allow_pickle=True)
            for d in drawings:
                seq = strokes_to_seq(d)

                padded = np.zeros((L, 3))
                n = min(len(seq), L)
                padded[:n] = seq[:n]

                self.data.append(padded.T)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.tensor(self.data[i], dtype=torch.float32), self.labels[i]