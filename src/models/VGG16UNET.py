from typing import Literal
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class VGG16UNET(nn.Module):
    def __init__(self, out_channels=8):
        super(VGG16UNET, self).__init__()

        # DECODER
        weights = torchvision.models.VGG16_Weights.DEFAULT
        model = torchvision.models.vgg16(weights=weights)

        for param in model.features.parameters():
        # for param in model.features[:11].parameters(): # Unfreeze the layers after Conv5
            param.requires_grad = False
        self.vgg16_decoder = model.features

        LAST_FILTER_CHANNELS = 512

        self.bottleneck = DoubleConv(
            in_channels=LAST_FILTER_CHANNELS,
            out_channels=LAST_FILTER_CHANNELS * 2,
        )

        FEATURES = [512, 256, 128, 64]

        # ENCODER
        self.encoder = nn.ModuleList()

        for feature in FEATURES:
            input_channels = feature * 2
            output_channels = feature

            self.encoder.append(
                nn.ConvTranspose2d(
                    input_channels,
                    output_channels,
                    kernel_size=2,
                    stride=2,
                    bias=False
                )
            )
            self.encoder.append(DoubleConv(input_channels, output_channels))

        input_channels = 512
        output_channels = 512

        self.encoder.insert(
            index=2,
            module=nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=2,
                stride=2,
                bias=False
            ),
        )

        input_channels = 1024
        output_channels = 512

        self.encoder.insert(index=3, module=DoubleConv(input_channels, output_channels))

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)
        self.bn_final_conv = nn.BatchNorm2d(out_channels)

    def update_vgg16(self, block: Literal["block1", "block2", "block3", "block4", "block5"], trainable: bool):
        # reference https://media.geeksforgeeks.org/wp-content/uploads/20200219152327/conv-layers-vgg16.jpg
        blocks_map = {
            "block1": [0, 2],
            "block2": [5, 7],
            "block3": [10, 12, 14],
            "block4": [17, 19, 21],
            "block5": [24, 26, 28]
        }
    
        for block_idx, layer in self.vgg16_decoder.named_children():
            if (layer.__class__.__name__ == "Conv2d") and (int(block_idx) in blocks_map[block]):
                for param in layer.parameters():
                    param.requires_grad = trainable
                    # print(block_idx, layer.__class__.__name__, param.requires_grad)

    def _encoder_block(self, x):
        skip_connections = []

        for index, param in enumerate(self.vgg16_decoder):

            x = param(x)

            # check if not yet the last
            if index < len(self.vgg16_decoder) - 1:
                next_param_name = self.vgg16_decoder[index + 1].__class__.__name__

                if next_param_name == "MaxPool2d":
                    skip_connections.append(x)

        return skip_connections, x

    def _decoder_block(self, skip_connections, x):

        skip_connections = reversed(skip_connections)
        skip_connections = iter(skip_connections)

        for param in self.encoder:
            param_name = param.__class__.__name__

            if param_name == "ConvTranspose2d":
                x = param(x)
            elif param_name == "DoubleConv":
                skip_connection = next(skip_connections)

                if x.shape != skip_connection.shape:
                    x = TF.resize(x, size=skip_connection.shape[2:])

                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = param(concat_skip)
            else:
                raise ValueError(f"Invalid Parameter {param_name}")

        return x

    def forward(self, x):

        skip_connections, x = self._encoder_block(x)
        x = self.bottleneck(x)
        x = self._decoder_block(skip_connections, x)
        x = self.final_conv(x)
        x = self.bn_final_conv(x)

        return x

