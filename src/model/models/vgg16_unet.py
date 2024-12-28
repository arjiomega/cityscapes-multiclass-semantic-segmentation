import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as TF

from src.model.models.components import DoubleConv


# TODO: Remove input part of vgg16 to receive different channel number
class VGG16UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=8):
        super(VGG16UNET, self).__init__()

        # DECODER
        weights = torchvision.models.VGG16_Weights.DEFAULT
        model = torchvision.models.vgg16(weights=weights)

        for param in model.features.parameters():
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
