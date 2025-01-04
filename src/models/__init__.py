import segmentation_models_pytorch as smp

architectures = {
    'Unet': smp.Unet,
    'UnetPlusPlus': smp.UnetPlusPlus,
    'MAnet': smp.MAnet,
    'Linknet': smp.Linknet,
    'FPN': smp.FPN,
    'PSPNet': smp.PSPNet,
    'PAN': smp.PAN,
    'DeepLabV3': smp.DeepLabV3,
    'DeepLabV3Plus': smp.DeepLabV3Plus,
}

encoders = {
    "resnet": {
        "resnet18": {"weights": ["imagenet"]},
        "resnet34": {"weights": ["imagenet"]},
        "resnet50": {"weights": ["imagenet"]},
        "resnet101": {"weights": ["imagenet"]},
        "resnet152": {"weights": ["imagenet"]},
    },
    "efficientnet": {
        "efficientnet-b0": {"weights": ["imagenet"]},
        "efficientnet-b1": {"weights": ["imagenet"]},
        "efficientnet-b2": {"weights": ["imagenet"]},
        "efficientnet-b3": {"weights": ["imagenet"]},
        "efficientnet-b4": {"weights": ["imagenet"]},
        "efficientnet-b5": {"weights": ["imagenet"]},
        "efficientnet-b6": {"weights": ["imagenet"]},
        "efficientnet-b7": {"weights": ["imagenet"]},
        "timm-efficientnet-b0": {"weights": ["imagenet", "advprop", "noisy-student"]},
        "timm-efficientnet-b1": {"weights": ["imagenet", "advprop", "noisy-student"]},
        "timm-efficientnet-b2": {"weights": ["imagenet", "advprop", "noisy-student"]},
        "timm-efficientnet-b3": {"weights": ["imagenet", "advprop", "noisy-student"]},
        "timm-efficientnet-b4": {"weights": ["imagenet", "advprop", "noisy-student"]},
        "timm-efficientnet-b5": {"weights": ["imagenet", "advprop", "noisy-student"]},
        "timm-efficientnet-b6": {"weights": ["imagenet", "advprop", "noisy-student"]},
        "timm-efficientnet-b7": {"weights": ["imagenet", "advprop", "noisy-student"]},
        "timm-efficientnet-b8": {"weights": ["imagenet", "advprop"]},
        "timm-efficientnet-l2": {"weights": ["noisy-student"]},
        "timm-efficientnet-lite0": {"weights": ["imagenet"]},
        "timm-efficientnet-lite1": {"weights": ["imagenet"]},
        "timm-efficientnet-lite2": {"weights": ["imagenet"]},
        "timm-efficientnet-lite3": {"weights": ["imagenet"]},
        "timm-efficientnet-lite4": {"weights": ["imagenet"]}
    },
    "mobilenet": {
        "mobilenet_v2": {"weights": ["imagenet"]},
        "timm-mobilenetv3_large_075": {"weights": ["imagenet"]},
        "timm-mobilenetv3_large_100": {"weights": ["imagenet"]},
        "timm-mobilenetv3_large_minimal_100": {"weights": ["imagenet"]},
        "timm-mobilenetv3_small_075": {"weights": ["imagenet"]},
        "timm-mobilenetv3_small_100": {"weights": ["imagenet"]},
        "timm-mobilenetv3_small_minimal_100": {"weights": ["imagenet"]}
    },
    "vgg": {
        "vgg11": {"weights": ["imagenet"]},
        "vgg11_bn": {"weights": ["imagenet"]},
        "vgg13": {"weights": ["imagenet"]},
        "vgg13_bn": {"weights": ["imagenet"]},
        "vgg16": {"weights": ["imagenet"]},
        "vgg16_bn": {"weights": ["imagenet"]},
        "vgg19": {"weights": ["imagenet"]},
        "vgg19_bn": {"weights": ["imagenet"]}
    }
}