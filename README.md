# ⛳ SwingNet++ : Golf Swing Event Detection using CNN + BiLSTM

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-Supported-green)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)

<div align="center">

<img src="assets/demo.gif" width="800"/>

</div>

---

Real-time golf swing sequencing using a CNN + BiLSTM architecture trained on the GolfDB dataset.

The model predicts eight key swing events and generates an annotated output video with confidence scores.

A production-ready implementation of **SwingNet**, a deep learning model for **golf swing event sequencing**, built on the **GolfDB** dataset.

This project extends the original CVPR Workshop implementation with a cleaner codebase, configurable training pipeline, stronger data augmentation, automatic checkpointing, and an end-to-end inference system capable of generating annotated prediction videos.

---

# Demo

Input

```
sample_videos/test_video.mp4
```

Output

```
results/
│
├── swing_prediction.mp4
├── Address.jpg
├── Toe-up.jpg
├── Mid-backswing.jpg
├── Top.jpg
├── Mid-downswing.jpg
├── Impact.jpg
├── Mid-follow-through.jpg
└── Finish.jpg
```

The generated video contains

- live swing phase prediction
- confidence score
- current frame number
- clean HUD overlay

---

# Project Overview

The objective is to detect the **8 key events** of a golf swing from a monocular video.

The network predicts

1. Address
2. Toe-up
3. Mid-backswing
4. Top
5. Mid-downswing
6. Impact
7. Mid-follow-through
8. Finish

along with a background class.

The architecture consists of

```
Video
   │
   ▼
MobileNetV2
(Frame Feature Extraction)
   │
   ▼
Bi-directional LSTM
(Temporal Modeling)
   │
   ▼
Fully Connected Layer
   │
   ▼
Frame-wise Event Prediction
```

---

# Repository Structure

```
.
├── checkpoints/
├── data/
├── models/
├── results/
├── sample_videos/
├── scripts/
├── swingnet/
│   ├── train.py
│   ├── test_video.py
│   ├── model.py
│   ├── MobileNetV2.py
│   ├── dataloader.py
│   ├── eval.py
│   └── configs.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/SwingNet.git

cd SwingNet
```

Create environment

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

Download

- GolfDB Dataset
- Preprocessed videos (160×160)

Place them as

```
data/

├── golfDB.mat
├── videos_160/
├── train_split_1.pkl
└── val_split_1.pkl
```

Run

```bash
python data/generate_splits.py
```

to generate train/validation splits.

---

# Download Pretrained MobileNetV2

Download the pretrained MobileNetV2 weights from

https://github.com/tonylins/pytorch-mobilenet-v2

Place

```
mobilenet_v2.pth.tar
```

inside

```
models/
```

---

# Training

Train using

```bash
python -m swingnet.train
```

During training the script automatically

- saves latest checkpoint
- saves best checkpoint
- evaluates after every epoch
- reports GPU memory
- reports validation PCE

Example

```
Epoch 20

Loss = 0.2818

PCE = 0.7564

Saved Best Model
```

---

# Testing Your Own Video

Place your golf swing video inside

```
sample_videos/
```

Run

```bash
cd swingnet

python test_video.py --video ../sample_videos/test_video.mp4
```

Outputs

```
results/

swing_prediction.mp4

Address.jpg

...

Finish.jpg
```

---

# Performance

## Original SwingNet

| Configuration | PCE |
|---------------|------|
| Repository Baseline | **71.5%** |
| Paper (with augmentation) | **76.1%** |

---

## Our Implementation

| Configuration | PCE |
|---------------|------|
| Reimplementation | **73.18%** |
| With Data Augmentation | **75.64%** |

Improvement over repository baseline

```
71.5%

↓

75.64%

+4.14 Percentage Points
```

which is approximately a

**5.8% relative improvement**

without modifying the model architecture.

---

# Improvements over Original Repository

This project significantly extends the original implementation.

## 1. Modern Project Structure

The original repository was a research prototype.

The project was reorganized into

- modular configuration
- reusable dataset pipeline
- checkpoint directory
- sample videos
- results directory
- cleaner inference code

making the repository significantly easier to maintain.

---

## 2. Data Augmentation

The original implementation trained **without augmentation**.

We introduced

- Random Horizontal Flip
- Small Random Rotation
- Random Affine Translation

which improves robustness against

- camera angle variation
- player positioning
- slight recording imperfections

Result

```
71.5%

↓

75.64%
```

---

## 3. Improved Training Pipeline

Training now includes

- AdamW optimizer
- Cosine Annealing LR Scheduler
- Automatic mixed precision
- Gradient scaling
- Validation after every epoch
- Best checkpoint saving
- Latest checkpoint saving
- Early stopping support
- GPU memory reporting

making the training process reproducible and stable.

---

## 4. End-to-End Inference Pipeline

A completely new inference pipeline was implemented.

Features

- accepts arbitrary golf swing videos
- automatic preprocessing
- sequence prediction
- confidence estimation
- event frame extraction

---

## 5. Annotated Video Generation

Instead of only printing frame numbers,

the project now generates a fully annotated video.

Each frame displays

- current swing phase
- prediction confidence
- frame counter
- professional HUD overlay

making the model suitable for demonstrations and qualitative analysis.

---

## 6. Automatic Event Snapshot Extraction

The exact predicted frames for

- Address
- Top
- Impact
- Finish

etc.

are automatically exported as JPEG images for further analysis.

---

# Model Architecture

```
Input Video

↓

160 × 160 Frames

↓

MobileNetV2 Backbone

↓

Feature Sequence

↓

Bidirectional LSTM

↓

Fully Connected Layer

↓

9-Class Prediction

↓

8 Swing Events + Background
```

---

# Results

The model successfully identifies

- Address
- Toe-up
- Mid-backswing
- Top
- Mid-downswing
- Impact
- Mid-follow-through
- Finish

from unseen golf swing videos with high confidence.

The inference pipeline also provides a frame-by-frame visualization of predictions.

---

# Future Improvements

Potential extensions include

- Real-time webcam inference
- YOLO-based automatic golfer detection
- Multi-person support
- Streamlit Web Application
- TensorRT optimization
- ONNX export
- Mobile deployment

---

# Reference

This project is based on

**GolfDB: A Video Database for Golf Swing Sequencing**

William McNally et al.

CVPR Workshops 2019

Paper

https://arxiv.org/abs/1903.06528

BibTeX

```text
@InProceedings{McNally_2019_CVPR_Workshops,
author = {McNally, William and Vats, Kanav and Pinto, Tyler and Dulhanty, Chris and McPhee, John and Wong, Alexander},
title = {GolfDB: A Video Database for Golf Swing Sequencing},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```

---

# License

This project follows the same license as the original GolfDB repository.

Creative Commons Attribution-NonCommercial 4.0 International License.