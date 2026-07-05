import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from configs import *
from dataloader import ToTensor, Normalize
from eval import predict_video
from model import EventDetector


EVENT_NAMES = [
    "Address",
    "Toe-up",
    "Mid-backswing",
    "Top",
    "Mid-downswing",
    "Impact",
    "Mid-follow-through",
    "Finish"
]


class SampleVideo(Dataset):

    def __init__(self, video_path, input_size=160, transform=None):
        self.video_path = video_path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        cap = cv2.VideoCapture(str(self.video_path))

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        scale = self.input_size / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        top = (self.input_size - new_h) // 2
        bottom = self.input_size - new_h - top

        left = (self.input_size - new_w) // 2
        right = self.input_size - new_w - left

        frames = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (new_w, new_h))

            frame = cv2.copyMakeBorder(
                frame,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=[
                    0.406 * 255,
                    0.456 * 255,
                    0.485 * 255
                ],
            )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        cap.release()

        sample = {
            "images": np.asarray(frames),
            "labels": np.zeros(len(frames))
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_model():

    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    model = EventDetector(
        pretrain=False,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_DIR / "best_model.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        default=str(ROOT / "sample_videos" / "test_video.mp4"),
        help="Path to input video",
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    transform = transforms.Compose([
        ToTensor(),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
    ])

    dataset = SampleVideo(
        video_path,
        input_size=IMG_SIZE,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    model = load_model()

    print("\nLoaded model.")

    for sample in loader:

        images = sample["images"]

        probs = predict_video(
            model,
            images,
            seq_length=SEQUENCE_LENGTH,
        )

    event_frames = np.argmax(
        probs,
        axis=0
    )[:-1]

    confidences = [
        probs[event_frames[i], i]
        for i in range(8)
    ]

    print("\nDetected Swing Events")
    print("-" * 50)

    for i in range(8):

        print(
            f"{EVENT_NAMES[i]:25s}"
            f" Frame {event_frames[i]:4d}"
            f"   Confidence {confidences[i]:.3f}"
        )

    output_dir = ROOT/"results"
    output_dir.mkdir(exist_ok=True)

    event_dir = output_dir / "event_frames"
    event_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_dir / "swing_prediction.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    saved_events = [False] * len(EVENT_NAMES)

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # --------------------------------------------------
        # Find current swing phase
        # --------------------------------------------------

        current_event = EVENT_NAMES[-1]
        current_conf = confidences[-1]

        for i in range(len(event_frames) - 1):

            if event_frames[i] <= frame_idx < event_frames[i + 1]:

                current_event = EVENT_NAMES[i]
                current_conf = confidences[i]
                break

        # --------------------------------------------------
        # Save exact event frame once
        # --------------------------------------------------

        for i, ef in enumerate(event_frames):

            if frame_idx == ef and not saved_events[i]:

                cv2.imwrite(
                    str(event_dir / f"{i}.{EVENT_NAMES[i]}.jpg"),
                    frame,
                )

                saved_events[i] = True
                
        # --------------------------------------------------
        # Draw overlay (compact)
        # --------------------------------------------------

        overlay = frame.copy()

        # Smaller background box
        cv2.rectangle(
            overlay,
            (8, 8),
            (185, 72),
            (0, 0, 0),
            -1,
        )

        frame = cv2.addWeighted(
            overlay,
            0.35,
            frame,
            0.65,
            0,
        )

        # Title
        cv2.putText(
            frame,
            "SwingNet",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Frame number
        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total_frames}",
            (14, 41),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Event
        cv2.putText(
            frame,
            f"{current_event}",
            (14, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Confidence
        cv2.putText(
            frame,
            f"{current_conf*100:.1f}%",
            (130, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

        frame_idx += 1

    writer.release()
    cap.release()

    print("\nSaved:")
    print(output_dir / "swing_prediction.mp4")
    print(output_dir)

if __name__ == "__main__":
    main()