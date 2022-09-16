"""MetNet-2 model for weather forecasting"""
from typing import List

import einops
import torch
import torch.nn as nn
import torchvision.transforms
from huggingface_hub import PyTorchModelHubMixin

from models.layers import DownSampler, MetNetPreprocessor, TimeDistributed
from models.layers.ConvLSTM import ConvLSTM
from models.layers.DilatedCondConv import DilatedResidualConv, UpsampleResidualConv


__all__ = ['MetNet2']

class MetNet2(torch.nn.Module, PyTorchModelHubMixin):
    """MetNet-2 model for weather forecasting"""

    def __init__(
        self,
        image_encoder: str = "downsampler",
        input_channels: int = 11, # 12
        input_size: int = 252, # 512
        lstm_channels: int = 64, # 128
        encoder_channels: int = 96, # 384
        upsampler_channels: int = 126, # 512
        lead_time_features: int = 504, #2048
        upsample_method: str = "interp",
        num_upsampler_blocks: int = 2,
        num_context_blocks: int = 3,
        num_input_timesteps: int = 4,
        encoder_dilations: List[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        sat_channels: int = 4, # 12
        output_channels: int = 1, # 12
        kernel_size: int = 3,
        center_crop_size: int = 63, #128
        forecast_steps: int = 32, # 48
        **kwargs,
    ):
        """
        MetNet-2 builds on MetNet-1 to use an even larger context area to predict up to 12 hours ahead.
        Paper: https://arxiv.org/pdf/2111.07470.pdf
        The architecture of MetNet-2 differs from the original MetNet in terms of the axial attention is dropped, and there
        is more dilated convolutions instead.
        Args:
            image_encoder:
            input_channels:
            input_size:
            lstm_channels:
            encoder_channels:
            upsampler_channels:
            lead_time_features:
            upsample_method:
            num_upsampler_blocks:
            num_context_blocks:
            num_input_timesteps:
            encoder_dilations:
            sat_channels:
            output_channels:
            kernel_size:
            center_crop_size:
            forecast_steps:
            **kwargs:
        """
        super(MetNet2, self).__init__()
        config = locals()
        config.pop("self")
        config.pop("__class__")
        self.config = kwargs.pop("config", config)

        sat_channels = self.config["sat_channels"]
        input_size = self.config["input_size"]
        input_channels = self.config["input_channels"]
        lstm_channels = self.config["lstm_channels"]
        image_encoder = self.config["image_encoder"]
        forecast_steps = self.config["forecast_steps"]
        encoder_channels = self.config["encoder_channels"]
        kernel_size = self.config["kernel_size"]
        num_context_blocks = self.config["num_context_blocks"]
        num_upsampler_blocks = self.config["num_upsampler_blocks"]
        encoder_dilations = self.config["encoder_dilations"]
        upsample_method = self.config["upsample_method"]
        output_channels = self.config["output_channels"]
        lead_time_features = self.config["lead_time_features"]
        num_input_timesteps = self.config["num_input_timesteps"]
        center_crop_size = self.config["center_crop_size"]
        upsampler_channels = self.config["upsampler_channels"]

        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.preprocessor = MetNetPreprocessor(
            sat_channels=sat_channels, crop_size=input_size, use_space2depth=True, split_input=True
        )
        # Update number of input_channels with output from MetNetPreprocessor
        new_channels = sat_channels * 4  # Space2Depth
        new_channels *= 2  # Concatenate two of them together
        input_channels = input_channels  # - sat_channels + new_channels
        if image_encoder in ["downsampler", "default"]:
            image_encoder = DownSampler(input_channels, output_channels=input_channels)
        else:
            raise ValueError(f"Image_encoder {image_encoder} is not recognized")
        self.image_encoder = TimeDistributed(image_encoder)
        total_number_of_conv_blocks = num_context_blocks * len(encoder_dilations) + 8
        total_number_of_conv_blocks = (
            total_number_of_conv_blocks + num_upsampler_blocks
            if upsample_method != "interp"
            else total_number_of_conv_blocks
        )

        # ConvLSTM with 13 timesteps, 128 LSTM channels, 18 encoder blocks, 384 encoder channels,
        self.conv_lstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dim=lstm_channels,
            kernel_size=kernel_size,
            num_layers=num_input_timesteps,
        )
        # Convolutional Residual Blocks going from dilation of 1 to 128 with 384 channels
        # 3 stacks of 8 blocks form context aggregating part of arch -> only two shown in image, so have both
        self.context_block_one = nn.ModuleList()
        self.context_block_one.append(
            DilatedResidualConv(
                input_channels=lstm_channels,
                output_channels=encoder_channels,
                kernel_size=kernel_size,
                dilation=1,
            )
        )
        self.context_block_one.extend(
            [
                DilatedResidualConv(
                    input_channels=encoder_channels,
                    output_channels=encoder_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                )
                for d in encoder_dilations[1:]
            ]
        )
        self.context_blocks = nn.ModuleList()
        for block in range(num_context_blocks - 1):
            self.context_blocks.extend(
                nn.ModuleList(
                    [
                        DilatedResidualConv(
                            input_channels=encoder_channels,
                            output_channels=encoder_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )
                        for d in encoder_dilations
                    ]
                )
            )

        # Center crop the output
        self.center_crop = torchvision.transforms.CenterCrop(size=center_crop_size)

        # Then tile 4x4 back to original size
        # This seems like it would mean something like this, with two applications of a simple upsampling
        if upsample_method == "interp":
            self.upsample = nn.Upsample(scale_factor=4, mode="nearest")
            self.upsampler_changer = nn.Conv2d(
                in_channels=encoder_channels, out_channels=upsampler_channels, kernel_size=(1, 1)
            )
        else:
            # The paper though, under the architecture, has 2 upsample blocks with 512 channels, indicating it might be a
            # transposed convolution instead?
            # 2 Upsample blocks with 512 channels
            self.upsample = nn.ModuleList(
                UpsampleResidualConv(
                    input_channels=encoder_channels,
                    output_channels=upsampler_channels,
                    kernel_size=3,
                )
                for _ in range(num_upsampler_blocks)
            )
        self.upsample_method = upsample_method

        # Shallow network of Conv Residual Block Dilation 1 with the lead time MLP embedding added
        self.residual_block_three = nn.ModuleList(
            [
                DilatedResidualConv(
                    input_channels=upsampler_channels,
                    output_channels=upsampler_channels,
                    kernel_size=kernel_size,
                    dilation=1,
                )
                for _ in range(8)
            ]
        )

        self.time_conditioners = nn.ModuleList()
        # Go through each set of blocks and add conditioner
        # Context Stack
        for layer in self.context_block_one:
            self.time_conditioners.append(
                ConditionWithTimeMetNet2(
                    forecast_steps=forecast_steps,
                    hidden_dim=lead_time_features,
                    num_feature_maps=layer.output_channels,
                )
            )
        for layer in self.context_blocks:
            self.time_conditioners.append(
                ConditionWithTimeMetNet2(
                    forecast_steps=forecast_steps,
                    hidden_dim=lead_time_features,
                    num_feature_maps=layer.output_channels,
                )
            )
        if self.upsample_method != "interp":
            for layer in self.upsample:
                self.time_conditioners.append(
                    ConditionWithTimeMetNet2(
                        forecast_steps=forecast_steps,
                        hidden_dim=lead_time_features,
                        num_feature_maps=layer.output_channels,
                    )
                )
        for layer in self.residual_block_three:
            self.time_conditioners.append(
                ConditionWithTimeMetNet2(
                    forecast_steps=forecast_steps,
                    hidden_dim=lead_time_features,
                    num_feature_maps=layer.output_channels,
                )
            )
        # Last layers are a Conv 1x1 with 4096 channels then softmax
        self.head = nn.Conv2d(upsampler_channels, output_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, lead_time: int = 0):
        """
        Compute for all forecast steps
        Args:
            x: Input tensor in [Batch, Time, Channel, Height, Width]
        Returns:
            The output predictions for all future timesteps
        """
        
        # B x C x T x H x W -> B x T x C x H x W
        xs = x.shape
        x = torch.reshape(x, (xs[0], xs[2], xs[1], xs[3], xs[4]))
#         x = x.permute(0,2,1,3,4)
        # Compute all timesteps, probably can be parallelized
        x = self.image_encoder(x)
        # Compute scale and bias
        block_num = 0

        # ConvLSTM
        res, _ = self.conv_lstm(x)
        # Select last state only
        res = res[:, -1]

        # Context Stack
        for layer in self.context_block_one:
            scale, bias = self.time_conditioners[block_num](res, lead_time)
            res = layer(res, scale, bias)
            block_num += 1
        for layer in self.context_blocks:
            scale, bias = self.time_conditioners[block_num](res, lead_time)
            res = layer(res, scale, bias)
            block_num += 1
        # Get Center Crop
        res = self.center_crop(res)
        # Upsample
        if self.upsample_method == "interp":
            res = self.upsample(res)
            res = self.upsampler_changer(res)
        else:
            for layer in self.upsample:
                scale, bias = self.time_conditioners[block_num](res, lead_time)
                res = layer(res, scale, bias)
                block_num += 1

        # Shallow network
        for layer in self.residual_block_three:
            scale, bias = self.time_conditioners[block_num](res, lead_time)
            res = layer(res, scale, bias)
            block_num += 1

        # Return 1x1 Conv
        res = self.head(res)

        # Softmax for rain forecasting
        return res


class ConditionWithTimeMetNet2(nn.Module):
    """Compute Scale and bias for conditioning on time"""

    def __init__(self, forecast_steps: int, hidden_dim: int, num_feature_maps: int):
        """
        Compute the scale and bias factors for conditioning convolutional blocks on the forecast time
        Args:
            forecast_steps: Number of forecast steps
            hidden_dim: Hidden dimension size
            num_feature_maps: Max number of channels in the blocks, to generate enough scale+bias values
                This means extra values will be generated, but keeps implementation simpler
        """
        super().__init__()
        self.forecast_steps = forecast_steps
        self.num_feature_maps = num_feature_maps
        self.lead_time_network = nn.ModuleList(
            [
                nn.Linear(in_features=forecast_steps, out_features=hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=2 * num_feature_maps),
            ]
        )

    def forward(self, x: torch.Tensor, timestep: int) -> [torch.Tensor, torch.Tensor]:
        """
        Get the scale and bias for the conditioning layers
        From the FiLM paper, each feature map (i.e. channel) has its own scale and bias layer, so needs
        a scale and bias for each feature map to be generated
        Args:
            x: The Tensor that is used
            timestep: Index of the timestep to use, between 0 and forecast_steps
        Returns:
            2 Tensors of shape (Batch, num_feature_maps)
        """
        # One hot encode the timestep
        timesteps = torch.zeros(x.size()[0], self.forecast_steps, dtype=x.dtype).to(next(self.lead_time_network.parameters()).device)
        timesteps[:, timestep] = 1
        # Get scales and biases
        for layer in self.lead_time_network:
            timesteps = layer(timesteps)
        scales_and_biases = timesteps
        scales_and_biases = einops.rearrange(
            scales_and_biases, "b (block sb) -> b block sb", block=self.num_feature_maps, sb=2
        )
        
        
        return scales_and_biases[:, :, 0], scales_and_biases[:, :, 1]