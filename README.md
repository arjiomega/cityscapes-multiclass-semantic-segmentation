# Cityscapes Semantic Segmentation

## Data
https://www.cityscapes-dataset.com


## Models

Vgg16-Unet
![Vgg16-unet](https://i.imgur.com/qjMPeVL.png)

## Setup

### Environment Variables 
1. (Optional) Create an `.env` file containing your `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`.
2. Adjust `config/train_args.json` according to your needs.

### Data
1. place `gtFine_trainvaltest` and `leftImg8bit_trainvaltest` to `data/raw` directory
2. update `leftImg8bit_trainvaltest/val`  to `leftImg8bit_trainvaltest/valid`
3. update `gtFine_trainvaltest/gtFine/val` to `gtFine_trainvaltest/gtFine/valid`

### Dependencies
```bash
pip install -e ".[dev]"
# OR
pip install -e .
```

### API
```bash
uvicorn src.api.model_api:app --log-level debug --reload 
```

## Usage

### Training

#### 1. Load the labels to update
```python
updated_class_dict = {
    "unlabeled": ["unlabeled", "ego vehicle", "rectification border", "out of roi", "static", "dynamic", "ground"],
    "road": ["road", "parking", "rail track"],
    "vehicles": ["car", "motorcycle", "bus", "truck", "train", "caravan", "trailer"],
    "bike": ["bicycle"],
    "person": ["person"],
    "rider": ["rider"],
    "sky": ["sky"],
    "vegetation": ["vegetation", "terrain"],
    "objects": ["traffic light", "traffic sign", "pole", "polegroup"],
    "walls": ["building", "wall", "fence", "guard rail", "bridge", "tunnel"],
    "sidewalk": ["sidewalk"],
    "license plate": ["license plate"],
}
```

#### 2. Training Config
```python
import torch

epochs = 10
data_dim = 512
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

#### 3. Load Model
```python
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="resnet34",        
    encoder_weights="imagenet",    
    in_channels=3,            
    classes=len(updated_class_dict),
).to(device)
```
[docs](https://github.com/qubvel-org/segmentation_models.pytorch)

#### 4. (Optional) Load Required Preprocessing
```python
from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
```

#### 5. Create Preprocessing Function
```python
from src.data.mask_updater import MaskUpdater
import albumentations as A
from albumentations.pytorch import ToTensorV2

mask_updater = MaskUpdater(updated_class_dict)
data_transform = A.Compose(
    [
        A.Resize(height=data_dim, width=data_dim),
        ToTensorV2(),
    ]
)

import numpy as np
def modified_transform(image:np.array, mask:np.array):
    augment_dict = {}
    preprocessed_image = preprocess_input(image)
    transformed = data_transform(image=preprocessed_image, mask=mask)
    transformed_image, transformed_mask = transformed["image"], transformed["mask"]
    
    augment_dict["image"] = transformed_image
    augment_dict["mask"] = transformed_mask
    
    return augment_dict
```

#### 6. Load Dataset
```python
from config import config
from src.data.load_dataset import SubsetLoader, LoadDataset
from torch.utils.data import DataLoader

subset_loader = SubsetLoader(config.RAW_IMG_DIR, config.RAW_MASK_DIR)

train_dataset = LoadDataset("train", subset_loader, mask_updater, transform=modified_transform)
valid_dataset = LoadDataset("valid", subset_loader, mask_updater, transform=modified_transform)

batch_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
batch_valid = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

```

#### 7. Load Metric(s), Loss Function, Optimizer
```python
from src.train.metrics import IoU

iou_metric = IoU(n_classes=len(updated_class_dict), device=device)
focal_loss = smp.losses.FocalLoss(mode="multiclass")
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
```

#### 8. (Optional) Discord Webhook and Plotter
```python
from src.experiment_tracking.training_notifier import TrainingNotifier
from src.report.report import Report

webhook_url = ""
avatar_url = ""

notifier = TrainingNotifier(webhook_url, avatar_url=avatar_url)
reporter = Report(show_plots=False)
```

#### 9. Train using ModelTrainer
```python
from src.train.train import ModelTrainer

model_trainer = ModelTrainer(
    model=model, 
    metric_fn=iou_metric,
    loss_fn=focal_loss,
    optimizer=optimizer,
    class_dict=updated_class_dict, 
    device=device
)

model_trainer.add_notifier(notifier)
model_trainer.add_reporter(reporter)

model_trainer.train(batch_train, batch_valid, epochs=epochs)
```

#### 10. Save Model
```python
model_trainer.save("model_filename.pth")
```