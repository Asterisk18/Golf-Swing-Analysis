from model import EventDetector
from dataloader import GolfDB, ToTensor, Normalize
from util import correct_preds
from configs import VIDEO_DIR, VAL_SPLIT, CHECKPOINT_DIR

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_video(model, images, seq_length=64):
    """
    Predicts probabilities for an entire video.

    First tries whole-video inference.
    Falls back to chunked inference if GPU memory is insufficient.
    """

    images = images.to(device)

    try:
        with torch.inference_mode():
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        return probs

    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise

        torch.cuda.empty_cache()
        probs = []

        with torch.inference_mode():
            start = 0

            while start < images.shape[1]:
                end = min(start + seq_length, images.shape[1])
                logits = model(images[:, start:end])
                probs.append(
                    F.softmax(logits, dim=1).cpu().numpy()
                )
                start = end

        return np.concatenate(probs, axis=0)


def eval(model, split=1, seq_length=64, n_cpu=6, disp=False):

    transform = transforms.Compose([
        ToTensor(),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
    ])

    # dataset from the training part
    dataset = GolfDB(
        data_file=VAL_SPLIT,
        video_dir=VIDEO_DIR,
        seq_length=seq_length,
        transform=transform,
        train=False,
    )

    # loads one video, in form of frames
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=n_cpu,
    )


    model.eval()
    correct = []
    progress = tqdm(loader, desc="Validation")

    with torch.inference_mode():
        for idx, sample in enumerate(progress):
            images = sample["images"]
            labels = sample["labels"]

            # returns a vector of size 9, having the probabilies of each event for the given frame
            probs = predict_video(
                model,
                images,
                seq_length,
            )
            
            # 'c' denotes if the prediction was correct or not, 1/0
            _, _, _, _, c = correct_preds(
                probs,
                labels.squeeze()
            )

            if disp:
                print(idx, c)

            correct.append(c)

    return np.mean(correct)


if __name__ == "__main__":

    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
    )

    checkpoint = torch.load(
        f"{CHECKPOINT_DIR}/best_model.pt",
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    pce = eval(
        model,
        split=1,
        disp=True,
    )

    print(f"\nAverage PCE : {pce:.4f}")