# Driving AI Assistant

The AI Driving Assistant project aims to develop an intelligent system capable of assisting drivers in various scenarios through the implementation of semantic segmentation technology. By leveraging state-of-the-art deep learning techniques and utilizing the Cityscapes dataset, this project endeavors to enhance driver safety, navigation efficiency, and overall driving experience.

## Data
https://www.cityscapes-dataset.com

### Data Format
{city} _ {seq:0>6} _ {frame:0>6} _ {type}{ext}

- **type** the type/modality of data, e.g. gtFine for fine ground truth, or leftImg8bit for left 8-bit images.
- **city** the city in which this part of the dataset was recorded.
- **seq** the sequence number using 6 digits.
- **frame** the frame number using 6 digits. Note that in some cities very few, albeit very long sequences were recorded, while in some cities - many short sequences were recorded, of which only the 19th frame is annotated.
- **ext** the extension of the file and optionally a suffix, e.g. _polygons.json for ground truth files

## Models

Vgg16-Unet
![Vgg16-unet](https://i.imgur.com/qjMPeVL.png)

## Setup

### API
```bash
uvicorn src.api.model_api:app --log-level debug --reload 
```

### Label Studio
```bash
sudo docker run -it -p 8080:8080 -v ./labelstudio_data:/label-studio/data --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data/images --env DATA_UPLOAD_MAX_NUMBER_FILES=1000 heartexlabs/label-studio:latest
```

<br>

If ever an error occur when running `docker run` related to permissions
```bash
sudo chown 1001:1001 ./labelstudio_data
```

Setup Segment Anything Model for Easier Labeling
>https://labelstud.io/blog/get-started-using-segment-anything/

