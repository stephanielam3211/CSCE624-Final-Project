# CSCE624-Final-Project

# GPU Performance Analysis for Sketch Recognition

This project evaluates how different sketch representations and neural network architectures affect GPU performance. Sketches are represented either as raster images or as stroke-based sequences and benchmarked using CNN and RNN models.

---

# Project Overview

The project compares:

- Raster CNN (2D convolution on images)
- Stroke CNN (1D convolution on stroke sequences)
- Stroke RNN (LSTM on stroke sequences)

The experiments measure:

- Throughput (samples/sec)
- Peak GPU memory usage
- Batch size scaling
- Sequence length scaling
- AMP (mixed precision) behavior

---

# Dataset

Dataset used:
- Google QuickDraw Dataset

Selected classes:
- cat
- dog
- apple
- bicycle
- car
- fish
- tree
- house
- chair
- clock

---

# Project Structure

```text
sketch_gpu_project/
|
|-- quickdraw_raw/
|-- quickdraw_npy/
|
|-- scripts/
|   |-- download_quickdraw.py
|   |-- convert_quickdraw_to_npy.py
|   |-- datasets.py
|   |-- models.py
|   |-- train.py
|   |-- benchmark.py
|
|-- jobs/
|   |-- download_data.sh
|   |-- train_raster.sh
|   |-- train_stroke.sh
|   |-- train_stroke_rnn.sh
|   |-- bench_main.sh
|   |-- bench_seq.sh
|   |-- bench_amp.sh
```

---

# Requirements

## Hardware
- GPU-enabled system
- ROCm-compatible AMD GPU

## Software
- Python 3.10+
- ROCm
- PyTorch (ROCm backend)

## Python Packages

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy pillow matplotlib
```

---

# Setup

## Create directories

```bash
mkdir quickdraw_raw
mkdir quickdraw_npy
mkdir scripts
mkdir jobs
```

---

# Download Dataset

Go to scripts directory:

```bash
cd ~/scripts
```

Run:

```bash
flux batch download_data.sh
```

Output:
- downloads `.ndjson` files into:

```text
quickdraw_raw/
```

---

# Convert Dataset

Output:
- converted `.npy` files in:

```text
quickdraw_npy/
```

Note: the download_data.sh runs both python download_quickdraw.py and convert_quickdraw_to_npy.py

---

# Training Models

## Raster CNN

```bash
cd ~/jobs
flux batch train_raster.sh
```

---

## Stroke CNN

```bash
cd ~/jobs
flux batch train_stroke.sh
```

---

## Stroke RNN

```bash
cd ~/jobs
flux batch train_stroke_rnn.sh
```

---

# Benchmarking

## Main Benchmark
Tests:
- throughput vs batch size
- memory vs batch size

Run:

```bash
cd ~/jobs
flux batch bench_main.flux
```

Output:
- `output_benchmark_main.out`

---

## Sequence Length Benchmark

Tests:
- throughput vs sequence length

Run:

```bash
cd ~/jobs
flux batch bench_seq.flux
```

Output:
- `output_benchmark_seq.out`

---

## AMP Benchmark

Tests:
- mixed precision vs standard precision

Run:

```bash
cd ~/jobs
flux batch bench_amp.flux
```

Output:
- `output_benchmark_amp.out`

---

# Extracting Results

You only need the values from Throughput (samples/sec) and Peak Memory (MB) for the graphs, everything else is to help you organize your data points

---

# Metrics

The following metrics are collected:

- Throughput (samples/sec)
- Peak GPU memory usage (MB)

---

# Key Findings

- Stroke CNN achieved the highest throughput
- Raster CNN used the most memory
- RNN models showed lower GPU efficiency due to sequential computation
- Sequence length significantly impacted RNN performance
- AMP did not improve performance for small models

---

# Future Work

Possible extensions:
- Larger datasets
- Transformer-based models
- Multi-GPU scaling
- Accuracy vs efficiency analysis
- Hybrid raster/stroke representations

---

# Author

Stephanie Lam
