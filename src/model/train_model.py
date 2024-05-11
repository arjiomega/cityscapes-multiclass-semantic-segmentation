import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data import load_dataset
from src.model.models import model_loader
from src.model.loss_functions import loss_loader

from src.visualization import visualize


def train_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim,
    train_dataset: torch.utils.data.DataLoader,
    epoch: int,
    device: str,
) -> float:
    """Training function that we iteratively run for N number of epochs.

    Args:
        model (torch.nn.Module): model that is going to be used for training.
        loss_fn (torch.nn.Module): loss function.
        optimizer (torch.optim): optimizer.
        train_dataset (torch.utils.data.DataLoader): training set.
        epoch (int): current epoch. Ex. 40 out of 100
    """
    train_loss = 0
    for images, masks in tqdm(train_dataset, desc=f"Epoch {epoch}"):
        model.train()

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Model Prediction
        pred_logits = model(images)

        # Loss Calculation
        loss = loss_fn(pred_logits, masks)
        train_loss += loss

        # Optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataset)

    print(f"train loss: {train_loss}")


def validation_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    validation_dataset: torch.utils.data.DataLoader,
    epoch: int,
    device: str,
):
    val_loss = 0
    model.eval()
    with torch.inference_mode():
        for images, masks in tqdm(validation_dataset, desc=f"Epoch {epoch}"):

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Model Prediction
            pred_logits = model(images)

            # Loss Calculation
            val_loss += loss_fn(pred_logits, masks)

        val_loss /= len(validation_dataset)

    print(f"validation loss: {val_loss}")


def train(
    model_name: str,
    loss_name: str,
    batch_size: int,
    data_dim: int,
    learning_rate: float,
    epochs: int,
    device: str,
):
    NUM_CLASSES = 8

    data_transform = A.Compose(
        [
            A.Resize(height=data_dim, width=data_dim),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_dataset = load_dataset("train", batch_size, NUM_CLASSES, data_transform)
    valid_dataset = load_dataset("valid", batch_size, NUM_CLASSES, data_transform)

    model = model_loader(model_name)(in_channels=3, out_channels=NUM_CLASSES).to(device)
    loss_fn = loss_loader(loss_name)()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_step(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_dataset=train_dataset,
            epoch=epoch,
            device=device,
        )

        validation_step(
            model=model,
            loss_fn=loss_fn,
            validation_dataset=valid_dataset,
            epoch=epoch,
            device=device,
        )

    sample_batch_img, sample_batch_mask = next(iter(valid_dataset))
    sample_img = sample_batch_img[0].unsqueeze(0)
    sample_mask = sample_batch_mask[0].unsqueeze(0)

    visualize.from_prediction(
        model=model, image=sample_img, true_mask=sample_mask, device=device
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 0
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 4
    DATA_DIM = 112
    EPOCHS = 1

    print(
        f"""
    TRAINING ON DEVICE: {device}

    SEED: {RANDOM_SEED}
    LEARNING RATE: {LEARNING_RATE}
    BATCH SIZE: {BATCH_SIZE}
    DATA DIMENSION (WIDTH x HEIGHT): {DATA_DIM} x {DATA_DIM}
    EPOCHS: {EPOCHS}
    """
    )

    train(
        model_name="unet",
        loss_name="dice",
        batch_size=BATCH_SIZE,
        data_dim=DATA_DIM,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
    )
