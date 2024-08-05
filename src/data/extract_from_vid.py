import logging
from pathlib import Path

import cv2
from tqdm import tqdm

from config import config

logging.basicConfig(level=logging.DEBUG)

source_name = "manila_to_lucena"
vid_format = ".mp4"
filename = source_name + vid_format
data_path = Path(config.SOURCE_DATA_DIR, filename)

save_name = lambda frame_num: f"{source_name}_{frame_num}"
save_format = ".jpg"
save_dir = Path(config.RAW_DATA_DIR, source_name)
save_dir.mkdir(parents=True, exist_ok=True)
save_as = lambda frame_num: str(
    Path(config.RAW_DATA_DIR, save_dir, f"{save_name(frame_num)}{save_format}")
)


cap = cv2.VideoCapture(str(data_path))

# Check if camera opened successfully
if cap.isOpened() == False:
    logging.error("Error opening video file")

fps = cap.get(cv2.CAP_PROP_FPS)
fpm = int(fps * 60)  # frames per minute

total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

pbar = tqdm(total=total_frame_count)

while cap.isOpened():
    ret, frame = cap.read()

    current_frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    pbar.update(1)

    # save every minute
    if ret == True and current_frame_count % (fpm / 2) == 0:
        cv2.imwrite(save_as(current_frame_count), frame)
    elif ret == False:
        break

pbar.close()
cap.release()
