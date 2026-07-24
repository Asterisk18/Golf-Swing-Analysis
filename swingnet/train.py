import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from configs import *
from model import EventDetector
from dataloader import GolfDB, ToTensor, Normalize, VideoAugmentation
from eval import eval


# Dataset
transform = transforms.Compose([
    VideoAugmentation(),
    ToTensor(),
    Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

train_dataset = GolfDB(
    data_file=TRAIN_SPLIT,
    video_dir=VIDEO_DIR,
    seq_length=SEQUENCE_LENGTH,
    transform=transform,
    train=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)


# Model
device = torch.device(DEVICE)

model = EventDetector(
    pretrain=True,
    width_mult=1.0,
    lstm_layers=1,
    lstm_hidden=256,
    bidirectional=True,
    dropout=False,
)

model.to(device)

print(f"\nModel on {device}")
print(f"Train videos : {len(train_dataset)}")
print(f"Batches/epoch: {len(train_loader)}")




# Freeze CNN for first few epochs
for i, layer in enumerate(model.cnn):
    if i < 10:
        for param in layer.parameters():
            param.requires_grad = False

# print(f"Frozen CNN for first {FREEZE_EPOCHS} epochs")


# Loss
weights = torch.tensor(
    [
        1/8,
        1/8,
        1/8,
        1/8,
        1/8,
        1/8,
        1/8,
        1/8,
        1/35
    ],
    device=device
)

criterion = nn.CrossEntropyLoss(weight=weights)


# Optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

scaler = torch.amp.GradScaler("cuda")


# Checkpoints
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_pce = 0.0
patience = 10
patience_counter = 0


# Training Loop
for epoch in range(EPOCHS):

    torch.cuda.reset_peak_memory_stats()
    model.train()

    running_loss = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        leave=True,
    )

    for sample in progress:

        images = sample["images"].to(device)
        labels = sample["labels"].to(device)

        labels = labels.view(-1)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print("\nRunning validation...")
    model.eval()

    with torch.no_grad():

        pce = eval(
            model,
            split=1,
            seq_length=SEQUENCE_LENGTH,
            n_cpu=NUM_WORKERS,
            disp=False,
        )

    print(
        f"\nEpoch {epoch+1}"
        f" | Loss = {avg_loss:.4f}"
        f" | PCE = {pce:.4f}"
    )
    print(
        f"GPU Memory : {torch.cuda.max_memory_allocated()/1024**3:.2f} GB"
    )

    # save the latest weights
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pce": pce,
        },
        os.path.join(CHECKPOINT_DIR, "latest.pt"),
    )

    if pce > best_pce:
        best_pce = pce
        patience_counter = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "pce": pce,
            },
            os.path.join(CHECKPOINT_DIR, "best_model.pt"),
        )

        print("Saved Best Model")

    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")

    if patience_counter >= patience:
        print("\nEarly stopping!")
        break

print("\nTraining Finished")
print(f"Best Validation PCE = {best_pce:.4f}")