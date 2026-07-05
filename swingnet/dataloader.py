import cv2
import numpy as np
import pandas as pd

from pathlib import Path

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class GolfDB(Dataset):

    def __init__(
        self,
        data_file,
        video_dir,
        seq_length=64,
        train=True,
        transform=None,
        event_centered=True,
        jitter=8,
    ):
        
        self.df = pd.read_pickle(data_file)
        self.video_dir = Path(video_dir)
        self.seq_length = seq_length
        self.train = train
        self.transform = transform
        self.event_centered = event_centered
        self.jitter = jitter

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.loc[idx]
        events = row["events"].copy()
        events -= events[0]
        cap = cv2.VideoCapture(
            str(self.video_dir / f"{row['id']}.mp4")
        )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        images = []
        labels = []

        # TRAINING
        if self.train:
            if self.event_centered:
                target_event = int(np.random.choice(events[1:-1])) # sample only from the 8 events, skip the start and end markers

                center = target_event + np.random.randint(
                    -self.jitter,
                    self.jitter + 1
                )
                start = center - self.seq_length // 2

            else:
                # start = np.random.randint(
                #     max(events[-1], 1)
                # )
                start = np.random.randint(
                    max(total_frames - self.seq_length, 1)
                )

            start = max(0, start)

            if start + self.seq_length > total_frames:
                start = max(
                    total_frames - self.seq_length,
                    0
                )

            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            pos = start

            while len(images) < self.seq_length:
                ret, img = cap.read()

                if not ret:
                    break

                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )

                images.append(img)

                if pos in events[1:-1]:
                    labels.append(
                        np.where(events[1:-1] == pos)[0][0]
                    )

                else:
                    labels.append(8)

                pos += 1

        # VALIDATION
        else:
            for pos in range(total_frames):
                ret, img = cap.read()

                if not ret:
                    break

                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )

                images.append(img)

                if pos in events[1:-1]:
                    labels.append(
                        np.where(events[1:-1] == pos)[0][0]
                    )

                else:
                    labels.append(8)

        cap.release()

        sample = {
            "images": np.asarray(images),
            "labels": np.asarray(labels),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class VideoAugmentation:

    def __init__(
        self,
        flip_prob=0.5,
        rotation=5,
        translate=0.05,
        scale=(0.95, 1.05),
    ):
        self.flip_prob = flip_prob
        self.rotation = rotation
        self.translate = translate
        self.scale = scale

    def __call__(self, sample):

        images = sample["images"]
        labels = sample["labels"]

        do_flip = np.random.rand() < self.flip_prob

        angle = np.random.uniform(
            -self.rotation,
            self.rotation,
        )

        scale = np.random.uniform(
            self.scale[0],
            self.scale[1],
        )

        h, w = images.shape[1:3]

        max_dx = self.translate * w
        max_dy = self.translate * h

        translate = (
            np.random.uniform(-max_dx, max_dx),
            np.random.uniform(-max_dy, max_dy),
        )

        aug_images = []

        for img in images:

            img = TF.to_pil_image(img)

            if do_flip:
                img = TF.hflip(img)

            img = TF.affine(
                img,
                angle=angle,
                translate=(
                    int(translate[0]),
                    int(translate[1]),
                ),
                scale=scale,
                shear=0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            aug_images.append(np.asarray(img))

        sample["images"] = np.asarray(aug_images)

        return sample


class ToTensor:

    def __call__(self, sample):

        images = sample["images"]
        labels = sample["labels"]
        images = images.transpose((0, 3, 1, 2))

        return {
            "images": torch.from_numpy(images).float() / 255.0,
            "labels": torch.from_numpy(labels).long(),
        }


class Normalize:

    def __init__(self, mean, std):

        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, sample):

        images = sample["images"]
        labels = sample["labels"]
        images.sub_(self.mean[None, :, None, None]).div_(
            self.std[None, :, None, None]
        )

        return {
            "images": images,
            "labels": labels,
        }