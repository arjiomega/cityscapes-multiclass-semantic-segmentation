"""Contains all the labels used in the dataset."""

class_dict = {
    "unlabeled": 0,
    "ego vehicle": 1,
    "rectification border": 2,
    "out of roi": 3,
    "static": 4,
    "dynamic": 5,
    "ground": 6,
    "road": 7,
    "sidewalk": 8,
    "parking": 9,
    "rail track": 10,
    "building": 11,
    "wall": 12,
    "fence": 13,
    "guard rail": 14,
    "bridge": 15,
    "tunnel": 16,
    "pole": 17,
    "polegroup": 18,
    "traffic light": 19,
    "traffic sign": 20,
    "vegetation": 21,
    "terrain": 22,
    "sky": 23,
    "person": 24,
    "rider": 25,
    "car": 26,
    "truck": 27,
    "bus": 28,
    "caravan": 29,
    "trailer": 30,
    "train": 31,
    "motorcycle": 32,
    "bicycle": 33,
    "license plate": -1,
}

vehicles = [
    "car",
    "truck",
    "bus",
    "caravan",
    "trailer",
    "train",
    "motorcycle",
    "bicycle",
]
human = ["person", "rider"]
other_classes = ["sky", "traffic light", "traffic sign", "road", "sidewalk"]

excluded_classes = [
    class_name
    for class_name in class_dict
    if class_name not in [*other_classes, *human, *vehicles]
]

updated_class_dict = {
    1: "human",
    2: "vehicle",
    3: "sky",
    4: "traffic light",
    5: "traffic sign",
    6: "road",
    7: "sidewalk",
    0: "unlabeled",
}
