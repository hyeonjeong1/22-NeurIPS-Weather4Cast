# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# Baseline model for the Neurips 2022 Weather4cast Competition -
# an adaptation of the CNN ELEKTRONN3 model available under MIT license on
#   https://raw.githubusercontent.com/ELEKTRONN/elektronn3/f754796d861f1cfe1c19dfc7819087972573ce40/elektronn3/models/unet.py
# 
# The adaptations are part of the Weather4cast 2022 Starter Kit.

# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


VERBOSE=False
# VERBOSE=True

__all__ = ['UNetBottle']

import copy
import itertools

from typing import Sequence, Union, Tuple, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

import einops
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def get_conv(dim=3):
    """Chooses an implementation for a convolution layer."""
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_convtranspose(dim=3):
    """Chooses an implementation for a transposed convolution layer."""
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_maxpool(dim=3):
    """Chooses an implementation for a max-pooling layer."""
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_normalization(normtype: str, num_channels: int, dim: int = 3):
    """Chooses an implementation for a batch normalization layer."""
    if normtype is None or normtype == 'none':
        return nn.Identity()
    elif normtype.startswith('group'):
        if normtype == 'group':
            num_groups = 8
        elif len(normtype) > len('group') and normtype[len('group'):].isdigit():
            num_groups = int(normtype[len('group'):])
        else:
            raise ValueError(
                f'normtype "{normtype}" not understood. It should be "group<G>",'
                f' where <G> is the number of groups.'
            )
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normtype == 'instance':
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
        else:
            raise ValueError('dim has to be 2 or 3')
    elif normtype == 'batch':
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
        else:
            raise ValueError('dim has to be 2 or 3')
    else:
        raise ValueError(
            f'Unknown normalization type "{normtype}".\n'
            'Valid choices are "batch", "instance", "group" or "group<G>",'
            'where <G> is the number of groups.'
        )


def planar_kernel(x):
    """Returns a "planar" kernel shape (e.g. for 2D convolution in 3D space)
    that doesn't consider the first spatial dim (D)."""
    if isinstance(x, int):
        return (1, x, x)
    else:
        return x


def planar_pad(x):
    """Returns a "planar" padding shape that doesn't pad along the first spatial dim (D)."""
    if isinstance(x, int):
        return (0, x, x)
    else:
        return x


def conv3(in_channels, out_channels, kernel_size=3, stride=1,
          padding=1, bias=True, planar=False, dim=3):
    """Returns an appropriate spatial convolution layer, depending on args.
    - dim=2: Conv2d with 3x3 kernel
    - dim=3 and planar=False: Conv3d with 3x3x3 kernel
    - dim=3 and planar=True: Conv3d with 1x3x3 kernel
    """
    if planar:
        stride = planar_kernel(stride)
        padding = planar_pad(padding)
        kernel_size = planar_kernel(kernel_size)
    return get_conv(dim)(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias
    )

def upconv2(in_channels, out_channels, mode='transpose', planar=False, dim=3):
    """Returns a learned upsampling operator depending on args."""
    kernel_size = 2
    stride = 2
    if planar:
        kernel_size = planar_kernel(kernel_size)
        stride = planar_kernel(stride)
    if mode == 'transpose':
        return get_convtranspose(dim)(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
    elif 'resizeconv' in mode:
        if 'linear' in mode:
            upsampling_mode = 'trilinear' if dim == 3 else 'bilinear'
        else:
            upsampling_mode = 'nearest'
        rc_kernel_size = 1 if mode.endswith('1') else 3
        return ResizeConv(
            in_channels, out_channels, planar=planar, dim=dim,
            upsampling_mode=upsampling_mode, kernel_size=rc_kernel_size
        )


def conv1(in_channels, out_channels, dim=3):
    """Returns a 1x1 or 1x1x1 convolution, depending on dim"""
    return get_conv(dim)(in_channels, out_channels, kernel_size=1)


def get_activation(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky':
            return nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'prelu':
            return nn.PReLU(num_parameters=1)
        elif activation == 'rrelu':
            return nn.RReLU()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'celu':
            return nn.CELU(alpha=1.0)
        elif activation == 'lin':
            return nn.Identity()
    else:
        # Deep copy is necessary in case of paremtrized activations
        return copy.deepcopy(activation)
    
    
class SEBlock(nn.Module):
    def __init__(self, out_channels, dim, reduction=4):
        super(SEBlock, self).__init__()

        self.fc = nn.Sequential(
            conv1(out_channels, out_channels//reduction, dim=dim),
            nn.ReLU(),
            conv1(out_channels//reduction, out_channels * 2, dim=dim),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool3d(x, 1) # Squeeze
        w = self.fc(x)
        w, b = w.split(w.data.size(1) // 2, dim=1) # Excitation
        w = torch.sigmoid(w)

        return x * w + b # Scale and add bias


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, planar=False, activation='relu',
                 normalization=None, full_norm=True, dim=3, conv_mode='same', transfer = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        self.dim = dim
        padding = 1 if 'same' in conv_mode else 0

        self.conv0 =  conv1(
            self.in_channels, self.out_channels, dim=dim
        )
        
        self.conv1 = conv3(
            self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding
        )
        self.conv2 = conv3(
            self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding
        ) # how about 1x1 conv?
        
        
        self.conv3 = conv1(
            self.out_channels, self.out_channels, dim=dim
        )
        
        self.se = SEBlock(self.out_channels, dim=dim)

        if self.pooling:
            kernel_size = 2
            if planar:
                kernel_size = planar_kernel(kernel_size)
            self.pool = get_maxpool(dim)(kernel_size=kernel_size, ceil_mode=True)
            self.pool_ks = kernel_size
        else:
            self.pool = nn.Identity()
            self.pool_ks = -123  # Bogus value, will never be read. Only to satisfy TorchScript's static type system

        self.dropout = nn.Dropout3d(0.2)
        
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)
        self.act3 = get_activation(activation)

        if full_norm:
            self.norm0 = get_normalization(normalization, self.out_channels, dim=dim)
            if VERBOSE: print("DownConv, full_norm, norm0 =",normalization)
        else:
            self.norm0 = nn.Identity()
            if VERBOSE: print("DownConv, no full_norm")
        self.norm1 = get_normalization(normalization, self.out_channels, dim=dim)
#         self.norm1 = nn.GroupNorm(num_groups=4, num_channels=self.out_channels)
        if VERBOSE: print("DownConv, norm1 =",normalization)
        self.norm2 = get_normalization(normalization, self.out_channels, dim=dim)
        self.transfer = transfer
        self.film_scale = nn.Parameter(torch.ones(self.out_channels))
        self.film_bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x):
        identity = self.conv0(x)
        y = self.conv1(identity)
        y = self.norm0(y)
        y =  self.dropout(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm1(y)
        y =  self.dropout(y)
        y += identity
        y = self.act2(y)
        y = self.se(y)
        before_pool = y
        y = self.pool(y)
        # before_pool = self.se(before_pool)
        
        before_pool = self.act3(self.norm2(self.conv3(before_pool)))
        if self.transfer:
            # before_pool = self.act3(self.norm2(before_pool))
            # before_pool = self.se(before_pool)
            scale = self.film_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(before_pool)
            bias = self.film_bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(before_pool)
            before_pool = (scale * before_pool) + bias
        
        return y, before_pool


@torch.jit.script
def autocrop(from_down: torch.Tensor, from_up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crops feature tensors from the encoder and decoder pathways so that they
    can be combined.

    - If inputs from the encoder pathway have shapes that are not divisible
      by 2, the use of ``nn.MaxPool(ceil_mode=True)`` leads to the 2x
      upconvolution results being too large by one element in each odd
      dimension, so they need to be cropped in these dimensions.

    - If VALID convolutions are used, feature tensors get smaller with each
      convolution, so we need to center-crop the larger feature tensors from
      the encoder pathway to make features combinable with the smaller
      decoder feautures.

    Args:
        from_down: Feature from encoder pathway (``DownConv``)
        from_up: Feature from decoder pathway (2x upsampled)

    Returns:

    """
    ndim = from_down.dim()  # .ndim is not supported by torch.jit

    if from_down.shape[2:] == from_up.shape[2:]:  # No need to crop anything
        return from_down, from_up

    # Step 1: Handle odd shapes

    # Handle potentially odd input shapes from encoder
    #  by cropping from_up by 1 in each dim that is odd in from_down and not
    #  odd in from_up (that is, where the difference between them is odd).
    #  The reason for looking at the shape difference and not just the shape
    #  of from_down is that although decoder outputs mostly have even shape
    #  because of the 2x upsampling, but if anisotropic pooling is used, the
    #  decoder outputs can also be be oddly shaped in the z (D) dimension.
    #  In these cases no cropping should be performed.
    ds = from_down.shape[2:]
    us = from_up.shape[2:]
    upcrop = [u - ((u - d) % 2) for d, u in zip(ds, us)]

    if ndim == 4:
        from_up = from_up[:, :, :upcrop[0], :upcrop[1]]
    if ndim == 5:
        from_up = from_up[:, :, :upcrop[0], :upcrop[1], :upcrop[2]]

    # Step 2: Handle center-crop resulting from valid convolutions
    ds = from_down.shape[2:]
    us = from_up.shape[2:]

    assert ds[0] >= us[0], f'{ds, us}'
    assert ds[1] >= us[1]
    if ndim == 4:
        from_down = from_down[
            :,
            :,
            (ds[0] - us[0]) // 2:(ds[0] + us[0]) // 2,
            (ds[1] - us[1]) // 2:(ds[1] + us[1]) // 2
        ]
    elif ndim == 5:
        assert ds[2] >= us[2]
        from_down = from_down[
            :,
            :,
            ((ds[0] - us[0]) // 2):((ds[0] + us[0]) // 2),
            ((ds[1] - us[1]) // 2):((ds[1] + us[1]) // 2),
            ((ds[2] - us[2]) // 2):((ds[2] + us[2]) // 2),
        ]
    return from_down, from_up


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    att: Optional[torch.Tensor]

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose', planar=False,
                 activation='relu', normalization=None, full_norm=True, dim=3, conv_mode='same',
                 attention=False, transfer=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.normalization = normalization
        padding = 1 if 'same' in conv_mode else 0

        self.upconv = upconv2(self.in_channels, self.out_channels,
                              mode=self.up_mode, planar=planar, dim=dim)

        if self.merge_mode == 'concat':
            self.conv1 = conv3(
                2*self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding
            ) # how about conv 1x1
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3(
                self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding
            )
        self.conv2 = conv3(
            self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding
        )
        self.conv3 = conv1(
            self.out_channels, self.out_channels, dim=dim
        )

        self.act0 = get_activation(activation)
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)
        self.act3 = get_activation(activation)

        if full_norm:
            self.norm0 = get_normalization(normalization, self.out_channels, dim=dim)
            self.norm1 = get_normalization(normalization, self.out_channels, dim=dim)
#             self.norm1 = nn.GroupNorm(num_groups=4, num_channels=self.out_channels)
        else:
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()
        self.norm2 = get_normalization(normalization, self.out_channels, dim=dim)
        self.norm3 = get_normalization(normalization, self.out_channels, dim=dim)
        if attention:
            self.attention = GridAttention(
                in_channels=in_channels, gating_channels=in_channels, dim=dim
            )
        else:
            self.attention = DummyAttention()
        self.att = None  # Field to store attention mask for later analysis
        self.transfer = transfer
        self.film_scale = nn.Parameter(torch.ones(self.out_channels))
        self.film_bias = nn.Parameter(torch.zeros(self.out_channels))
        

    def forward(self, enc, dec):
        """ Forward pass
        Arguments:
            enc: Tensor from the encoder pathway
            dec: Tensor from the decoder pathway (to be upconv'd)
        """

        updec = self.upconv(dec)
        enc, updec = autocrop(enc, updec)
        genc, att = self.attention(enc, dec)
        if not torch.jit.is_scripting():
            self.att = att
        updec = self.norm0(updec)
        updec = self.act0(updec)
        if self.merge_mode == 'concat':
            mrg = torch.cat((updec, genc), 1)
        else:
            mrg = updec + genc
        y = self.conv1(mrg)
        y = self.norm1(y)
        identity = self.act1(y)
        
        y = self.conv2(identity)
        y = self.norm2(y)
        y = self.act2(y)
        y = self.conv3(y)
        y = self.norm3(y)
        y = self.act3(y + identity)
        
        if self.transfer:
            scale = self.film_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(y)
            bias = self.film_bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(y)
            return (scale * y) + bias
        else:
            return y

class ResizeConv(nn.Module):
    """Upsamples by 2x and applies a convolution.

    This is meant as a replacement for transposed convolution to avoid
    checkerboard artifacts. See

    - https://distill.pub/2016/deconv-checkerboard/
    - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, planar=False, dim=3,
                 upsampling_mode='nearest'):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.scale_factor = 2
        if dim == 3 and planar:  # Only interpolate (H, W) dims, leave D as is
            self.scale_factor = planar_kernel(self.scale_factor)
        self.dim = dim
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=self.upsampling_mode)
        # TODO: Investigate if 3x3 or 1x1 conv makes more sense here and choose default accordingly
        # Preliminary notes:
        # - conv3 increases global parameter count by ~10%, compared to conv1 and is slower overall
        # - conv1 is the simplest way of aligning feature dimensions
        # - conv1 may be enough because in all common models later layers will apply conv3
        #   eventually, which could learn to perform the same task...
        #   But not exactly the same thing, because this layer operates on
        #   higher-dimensional features, which subsequent layers can't access
        #   (at least in U-Net out_channels == in_channels // 2).
        # --> Needs empirical evaluation
        if kernel_size == 3:
            self.conv = conv3(
                in_channels, out_channels, padding=1, planar=planar, dim=dim
            )
        elif kernel_size == 1:
            self.conv = conv1(in_channels, out_channels, dim=dim)
        else:
            raise ValueError(f'kernel_size={kernel_size} is not supported. Choose 1 or 3.')

    def forward(self, x):
        return self.conv(self.upsample(x))


class GridAttention(nn.Module):
    """Based on https://github.com/ozan-oktay/Attention-Gated-Networks

    Published in https://arxiv.org/abs/1804.03999"""
    def __init__(self, in_channels, gating_channels, inter_channels=None, dim=3, sub_sample_factor=2):
        super().__init__()

        assert dim in [2, 3]

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dim

        # Default parameter set
        self.dim = dim
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dim == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dim == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplementedError

        # Output transform
        self.w = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels),
        )
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels,
            kernel_size=(1, self.sub_sample_kernel_size[0], self.sub_sample_kernel_size[0]), stride=self.sub_sample_factor, bias=False
        )
        self.phi = conv_nd(
            in_channels=self.gating_channels, out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.psi = conv_nd(
            in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, bias=True
        )

        self.init_weights()

    def forward(self, x, g):
        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x.shape[2:], mode=self.upsample_mode, align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=x.shape[2:], mode=self.upsample_mode, align_corners=False)
        y = sigm_psi_f.expand_as(x) * x
        wy = self.w(y)

        return wy, sigm_psi_f

    def init_weights(self):
        def weight_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(weight_init)


class DummyAttention(nn.Module):
    def forward(self, x, g):
        return x, None


# TODO: Pre-calculate output sizes when using valid convolutions
class UNetBottle(nn.Module):
    """Modified version of U-Net, adapted for 3D biomedical image segmentation

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding, expansive pathway)
    about an input tensor is merged with information representing the
    localization of details (from the encoding, compressive pathway).

    - Original paper: https://arxiv.org/abs/1505.04597
    - Base implementation: https://github.com/jaxony/unet-pytorch


    Modifications to the original paper (@jaxony):

    - Padding is used in size-3-convolutions to prevent loss
      of border pixels.
    - Merging outputs does not require cropping due to (1).
    - Residual connections can be used by specifying
      UNet(merge_mode='add').
    - If non-parametric upsampling is used in the decoder
      pathway (specified by upmode='upsample'), then an
      additional 1x1 convolution occurs after upsampling
      to reduce channel dimensionality by a factor of 2.
      This channel halving happens with the convolution in
      the tranpose convolution (specified by upmode='transpose').

    Additional modifications (@mdraw):

    - Operates on 3D image data (5D tensors) instead of 2D data
    - Uses 3D convolution, 3D pooling etc. by default
    - Each network block pair (the two corresponding submodules in the
      encoder and decoder pathways) can be configured to either work
      in 3D or 2D mode (3D/2D convolution, pooling etc.)
      with the `planar_blocks` parameter.
      This is helpful for dealing with data anisotropy (commonly the
      depth axis has lower resolution in SBEM data sets, so it is not
      as important for convolution/pooling) and can reduce the complexity of
      models (parameter counts, speed, memory usage etc.).
      Note: If planar blocks are used, the input patch size should be
      adapted by reducing depth and increasing height and width of inputs.
    - Configurable activation function.
    - Optional normalization

    Gradient checkpointing can be used to reduce memory consumption while
    training. To make use of gradient checkpointing, just run the
    ``forward_gradcp()`` instead of the regular ``forward`` method.
    This makes the backward pass a bit slower, but the memory savings can be
    huge (usually around 20% - 50%, depending on hyperparameters). Checkpoints
    are made after each network *block*.
    See https://pytorch.org/docs/master/checkpoint.html and
    https://arxiv.org/abs/1604.06174 for more details.
    Gradient checkpointing is not supported in TorchScript mode.

    Args:
        in_channels: Number of input channels
            (e.g. 1 for single-grayscale inputs, 3 for RGB images)
            Default: 1
        out_channels: Number of output channels (in classification/semantic
            segmentation, this is the number of different classes).
            Default: 2
        n_blocks: Number of downsampling/convolution blocks (max-pooling)
            in the encoder pathway. The decoder (upsampling/upconvolution)
            pathway will consist of `n_blocks - 1` blocks.
            Increasing `n_blocks` has two major effects:

            - The network will be deeper
              (n + 1 -> 4 additional convolution layers)
            - Since each block causes one additional downsampling, more
              contextual information will be available for the network,
              enhancing the effective visual receptive field.
              (n + 1 -> receptive field is approximately doubled in each
              dimension, except in planar blocks, in which it is only
              doubled in the H and W image dimensions)

            **Important note**: Always make sure that the spatial shape of
            your input is divisible by the number of blocks, because
            else, concatenating downsampled features will fail.
        start_filts: Number of filters for the first convolution layer.
            Note: The filter counts of the later layers depend on the
            choice of `merge_mode`.
        up_mode: Upsampling method in the decoder pathway.
            Choices:

            - 'transpose' (default): Use transposed convolution
              ("Upconvolution")
            - 'resizeconv_nearest': Use resize-convolution with nearest-
              neighbor interpolation, as proposed in
              https://distill.pub/2016/deconv-checkerboard/
            - 'resizeconv_linear: Same as above, but with (bi-/tri-)linear
              interpolation
            - 'resizeconv_nearest1': Like 'resizeconv_nearest', but using a
              light-weight 1x1 convolution layer instead of a spatial convolution
            - 'resizeconv_linear1': Like 'resizeconv_nearest', but using a
              light-weight 1x1-convolution layer instead of a spatial convolution
        merge_mode: How the features from the encoder pathway should
            be combined with the decoder features.
            Choices:

            - 'concat' (default): Concatenate feature maps along the
              `C` axis, doubling the number of filters each block.
            - 'add': Directly add feature maps (like in ResNets).
              The number of filters thus stays constant in each block.

            Note: According to https://arxiv.org/abs/1701.03056, feature
            concatenation ('concat') generally leads to better model
            accuracy than 'add' in typical medical image segmentation
            tasks.
        planar_blocks: Each number i in this sequence leads to the i-th
            block being a "planar" block. This means that all image
            operations performed in the i-th block in the encoder pathway
            and its corresponding decoder counterpart disregard the depth
            (`D`) axis and only operate in 2D (`H`, `W`).
            This is helpful for dealing with data anisotropy (commonly the
            depth axis has lower resolution in SBEM data sets, so it is
            not as important for convolution/pooling) and can reduce the
            complexity of models (parameter counts, speed, memory usage
            etc.).
            Note: If planar blocks are used, the input patch size should
            be adapted by reducing depth and increasing height and
            width of inputs.
        activation: Name of the non-linear activation function that should be
            applied after each network layer.
            Choices (see https://arxiv.org/abs/1505.00853 for details):

            - 'relu' (default)
            - 'silu': Sigmoid Linear Unit (SiLU, aka Swish)
            - 'leaky': Leaky ReLU (slope 0.1)
            - 'prelu': Parametrized ReLU. Best for training accuracy, but
              tends to increase overfitting.
            - 'rrelu': Can improve generalization at the cost of training
              accuracy.
            - Or you can pass an nn.Module instance directly, e.g.
              ``activation=torch.nn.ReLU()``
        normalization: Type of normalization that should be applied at the end
            of each block. Note that it is applied after the activated conv
            layers, not before the activation. This scheme differs from the
            original batch normalization paper and the BN scheme of 3D U-Net,
            but it delivers better results this way
            (see https://redd.it/67gonq).
            Choices:

            - 'group' for group normalization (G=8)
            - 'group<G>' for group normalization with <G> groups
              (e.g. 'group16') for G=16
            - 'instance' for instance normalization
            - 'batch' for batch normalization (default)
            - 'none' or ``None`` for no normalization
        attention: If ``True``, use grid attention in the decoding pathway,
            as proposed in https://arxiv.org/abs/1804.03999.
            Default: ``False``.
        full_norm: If ``True`` (default), perform normalization after each
            (transposed) convolution in the network (which is what almost
            all published neural network architectures do).
            If ``False``, only normalize after the last convolution
            layer of each block, in order to save resources. This was also
            the default behavior before this option was introduced.
        dim: Spatial dimensionality of the network. Choices:

            - 3 (default): 3D mode. Every block fully works in 3D unless
              it is excluded by the ``planar_blocks`` setting.
              The network expects and operates on 5D input tensors
              (N, C, D, H, W).
            - 2: Every block and every operation works in 2D, expecting
              4D input tensors (N, C, H, W).
        conv_mode: Padding mode of convolutions. Choices:

            - 'same' (default): Use SAME-convolutions in every layer:
              zero-padding inputs so that all convolutions preserve spatial
              shapes and don't produce an offset at the boundaries.
            - 'valid': Use VALID-convolutions in every layer: no padding is
              used, so every convolution layer reduces spatial shape by 2 in
              each dimension. Intermediate feature maps of the encoder pathway
              are automatically cropped to compatible shapes so they can be
              merged with decoder features.
              Advantages:

              - Less resource consumption than SAME because feature maps
                have reduced sizes especially in deeper layers.
              - No "fake" data (that is, the zeros from the SAME-padding)
                is fed into the network. The output regions that are influenced
                be removed in post-processing if possible (see
                ``overlap_shape`` in :py:mod:`elektronn3.inference`).
                Using VALID convolutions prevents the unnecessary computation
                of these regions that need to be cut away anyways for
                high-quality tiled inference.
              - Avoids the issues described in https://arxiv.org/abs/1811.11718.
              - Since the network will not receive zero-padded inputs, it is
                not required to learn a robustness against artificial zeros
                being in the border regions of inputs. This should reduce the
                complexity of the learning task and allow the network to
                specialize better on understanding the actual, unaltered
                inputs (effectively requiring less parameters to fit).

              Disadvantages:

              - Using this mode poses some additional constraints on input
                sizes and requires you to center-crop your targets,
                so it's harder to use in practice than the 'same' mode.
              - In some cases it might be preferable to get low-quality
                outputs at image borders as opposed to getting no outputs at
                the borders. Most notably this is the case if you do training
                and inference not on small patches, but on complete images in
                a single step.
    """
    def __init__(
            self,
            in_channels: int = 11,
            out_channels: int = 32,  ## NEW: number of time slots to predict
            n_blocks: int = 6,
            start_filts: int = 32,
            up_mode: str = 'transpose',
            merge_mode: str = 'concat',
            planar_blocks: Sequence = (),
            batch_norm: str = 'unset',
            attention: bool = False,
            activation: Union[str, nn.Module] = 'celu',
            normalization: str = 'batch',
            full_norm: bool = True,
            dim: int = 3,
            conv_mode: str = 'same',
            transfer = False
    ):
        super().__init__()

        if n_blocks < 1:
            raise ValueError('n_blocks must be > 1.')

        if dim not in {2, 3}:
            raise ValueError('dim has to be 2 or 3')
        if dim == 2 and planar_blocks != ():
            raise ValueError(
                'If dim=2, you can\'t use planar_blocks since everything will '
                'be planar (2-dimensional) anyways.\n'
                'Either set dim=3 or set planar_blocks=().'
            )
        if up_mode in ('transpose', 'upsample', 'resizeconv_nearest', 'resizeconv_linear',
                       'resizeconv_nearest1', 'resizeconv_linear1'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for upsampling".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        # TODO: Remove merge_mode=add. It's just worse than concat
        if 'resizeconv' in self.up_mode and self.merge_mode == 'add':
            raise ValueError("up_mode \"resizeconv\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "n_blocks channels (by half).")

        if len(planar_blocks) > n_blocks:
            raise ValueError('planar_blocks can\'t be longer than n_blocks.')
        if planar_blocks and (max(planar_blocks) >= n_blocks or min(planar_blocks) < 0):
            raise ValueError(
                'planar_blocks has invalid value range. All values have to be'
                'block indices, meaning integers between 0 and (n_blocks - 1).'
            )

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.n_blocks = n_blocks
        self.normalization = normalization
        self.attention = attention
        self.conv_mode = conv_mode
        self.activation = activation
        self.dim = dim
        self.transfer = transfer

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        if batch_norm != 'unset':
            raise RuntimeError(
                'The `batch_norm` option has been replaced with the more general `normalization` option.\n'
                'If you still want to use batch normalization, set `normalization=batch` instead.'
            )

        # Indices of blocks that should operate in 2D instead of 3D mode,
        # to save resources
        self.planar_blocks = planar_blocks

        # create the encoder pathway and add to a list
        for i in range(n_blocks):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < n_blocks - 1 else False
            planar = i in self.planar_blocks

            down_conv = DownConv(
                ins,
                outs,
                pooling=pooling,
                planar=planar,
                activation=activation,
                normalization=normalization,
                full_norm=full_norm,
                dim=dim,
                conv_mode=conv_mode,
                transfer = self.transfer
            )
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires n_blocks-1 blocks
        for i in range(n_blocks - 1):
            ins = outs
            outs = ins // 2
            planar = n_blocks - 2 - i in self.planar_blocks

            up_conv = UpConv(
                ins,
                outs,
                up_mode=up_mode,
                merge_mode=merge_mode,
                planar=planar,
                activation=activation,
                normalization=normalization,
                attention=attention,
                full_norm=full_norm,
                dim=dim,
                conv_mode=conv_mode,
                transfer = self.transfer
            )
            self.up_convs.append(up_conv)
        self.reduce_channels = conv1(outs*4, ## 4= experiment / len_seq_in
                                     self.out_channels, dim=dim)
        
        self.region_conditioners = nn.ModuleList()
        self.conditioner = nn.ModuleList()
        for i in range(n_blocks):
            self.conditioner.append(RegionConditioner())
            self.region_conditioners.append(ConditionWithRegion(
                regions=7, hidden_dim=self.start_filts * (2**i) * 2, num_feature_maps=self.start_filts * (2**i)))
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.dropout = nn.Dropout3d(0.2)  ## read this from config!
        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, GridAttention):
            return
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if getattr(m, 'bias') is not None:
                nn.init.constant_(m.bias, 0)        
    
    

    def forward(self, x, r_idx, r_idx_b=None, lam=None):
        if r_idx_b != None:
            r_idx_label = nn.functional.one_hot(r_idx, num_classes=7).type(torch.FloatTensor)
            r_idx_b_label = nn.functional.one_hot(r_idx_b, num_classes=7).type(torch.FloatTensor)
            r_idx = lam * r_idx_label + (1-lam) * r_idx_b_label
        else:
            r_idx = nn.functional.one_hot(r_idx, num_classes=7).type(torch.FloatTensor)
        

        encoder_outs = []

        # Encoder pathway, save outputs for merging
        i = 0  # Can't enumerate because of https://github.com/pytorch/pytorch/issues/16123
        for module in self.down_convs:
            x, before_pool = module(x)
            coditioner_layer = self.conditioner[i]
            scale, bias = self.region_conditioners[i](before_pool, r_idx)
            before_pool = coditioner_layer(before_pool, scale, bias)
            before_pool =  self.dropout(before_pool)  # for skip connections
            encoder_outs.append(before_pool)
            i += 1

        x =  self.dropout(x)  # at bottom of the U, as in the original U-Net
        # Decoding by UpConv and merging with saved outputs of encoder
        i = 0
        for module in self.up_convs:
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
            i += 1

        # No softmax is used, so you need to apply it in the loss.
        if VERBOSE: print("pre-reshape",x.shape)
        xs = x.shape;
        x = torch.reshape(x,(xs[0],xs[1]*xs[2],1,xs[3],xs[4]));
        
        if VERBOSE: print("pre-reduce",x.shape)
        x = self.reduce_channels(x)
        if VERBOSE: print("post-reduce",x.shape)
        xs = x.shape;
        x = torch.reshape(x,(xs[0],1,xs[1],xs[3],xs[4]));
        if VERBOSE: print("post-reshape",x.shape)
            # Uncomment the following line to temporarily store output for
        #  receptive field estimation using fornoxai/receptivefield:
        # self.feature_maps = [x]  # Currently disabled to save memory
        
        return x

    @torch.jit.unused
    def forward_gradcp(self, x):
        """``forward()`` implementation with gradient checkpointing enabled.
        Apart from checkpointing, this behaves the same as ``forward()``."""
        encoder_outs = []
        i = 0
        for module in self.down_convs:
            x, before_pool = checkpoint(module, x)
            encoder_outs.append(before_pool)
            i += 1
        i = 0
        for module in self.up_convs:
            before_pool = encoder_outs[-(i+2)]
            x = checkpoint(module, before_pool, x)
            i += 1
        x = self.conv_final(x)
        # self.feature_maps = [x]  # Currently disabled to save memory
        return x

class RegionConditioner(nn.Module):
    """
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    Paper: https://arxiv.org/pdf/1709.07871.pdf
    """

    def forward(self, x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Add the conditioning to the input tensor
        Args:
            x: Input tensor
            scale: Scale parameter
            bias: Bias parameter
        Returns:
            Input tensor with the scale multiplied to it and bias added
        """
        scale = scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        bias = bias.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)

        return (scale * x) + bias
    

def test_model(
    batch_size=1,
    in_channels=1,
    out_channels=2,
    n_blocks=3,
    planar_blocks=(),
    merge_mode='concat',
    dim=3
):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = UNetBottle(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=n_blocks,
        planar_blocks=planar_blocks,
        merge_mode=merge_mode,
        dim=dim,
    ).to(device)

    # Minimal test input
    if dim == 3:
        # Each block in the encoder pathway ends with 2x2x2 downsampling, except
        # planar blocks, which only do 1x2x2 downsampling, so the input has to
        # be larger when using more blocks.
        x = torch.randn(
            batch_size,
            in_channels,
            2 ** n_blocks // (2 ** len(planar_blocks)),
            2 ** n_blocks,
            2 ** n_blocks,
            device=device
        )
        expected_out_shape = (
            batch_size,
            out_channels,
            2 ** n_blocks // (2 ** len(planar_blocks)),
            2 ** n_blocks,
            2 ** n_blocks
        )
    elif dim == 2:
        # Each block in the encoder pathway ends with 2x2 downsampling
        # so the input has to be larger when using more blocks.
        x = torch.randn(
            batch_size,
            in_channels,
            2 ** n_blocks,
            2 ** n_blocks,
            device=device
        )
        expected_out_shape = (
            batch_size,
            out_channels,
            2 ** n_blocks,
            2 ** n_blocks
        )

    # Test forward, autograd, and backward pass with test input
    out, reg = model(x)
    loss = torch.sum(out)
    loss.backward()
    assert out.shape == expected_out_shape


def test_2d_config(max_n_blocks=4):
    for n_blocks in range(1, max_n_blocks + 1):
        print(f'Testing 2D U-Net with n_blocks = {n_blocks}...')
        test_model(n_blocks=n_blocks, dim=2)


def test_planar_configs(max_n_blocks=4):
    for n_blocks in range(1, max_n_blocks + 1):
        planar_combinations = itertools.chain(*[
            list(itertools.combinations(range(n_blocks), i))
            for i in range(n_blocks + 1)
        ])  # [(), (0,), (1,), ..., (0, 1), ..., (0, 1, 2, ..., n_blocks - 1)]

        for p in planar_combinations:
            print(f'Testing 3D U-Net with n_blocks = {n_blocks}, planar_blocks = {p}...')
            test_model(n_blocks=n_blocks, planar_blocks=p)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


def _stack_tups(tuples, stack_dim=1):
    "Stack tuple of tensors along `stack_dim`"
    return tuple(
        torch.stack([t[i] for t in tuples], dim=stack_dim) for i in list(range(len(tuples[0])))
    )


class TimeDistributed(nn.Module):
    '''Applies `module` over `tdim` identically for each step,
    use `low_mem` to compute one at a time.'''

    def __init__(self, module, low_mem=False, tdim=1):
        super().__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim

    def forward(self, *tensors, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim != 1:
            return self.low_mem_forward(*tensors, **kwargs)
        else:
            # only support tdim=1
            inp_shape = tensors[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            out = self.module(*[x.view(bs * seq_len, *x.shape[2:]) for x in tensors], **kwargs)
        return self.format_output(out, bs, seq_len)

    def low_mem_forward(self, *tensors, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        seq_len = tensors[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in tensors]
        out = []
        for i in range(seq_len):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        if isinstance(out[0], tuple):
            return _stack_tups(out, stack_dim=self.tdim)
        return torch.stack(out, dim=self.tdim)

    def format_output(self, out, bs, seq_len):
        "unstack from batchsize outputs"
        if isinstance(out, tuple):
            return tuple(out_i.view(bs, seq_len, *out_i.shape[1:]) for out_i in out)
        return out.view(bs, seq_len, *out.shape[1:])

    def __repr__(self):
        return f"TimeDistributed({self.module})"


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class LeFF(nn.Module):
    def __init__(self, dim = 192, scale = 4, depth_kernel = 3):
        super().__init__()
        scale_dim = dim*scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(scale_dim),
                                    nn.GELU(),
                                    Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                    )
        self.depth_conv =  nn.Sequential(nn.Conv2d(
            scale_dim, scale_dim, kernel_size=depth_kernel,
            padding=1, groups=scale_dim, bias=False),
                          nn.BatchNorm2d(scale_dim),
                          nn.GELU(),
                          Rearrange('b c h w -> b (h w) c', h=14, w=14)
                          )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c')
                                    )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class LCAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = q[:, :, -1, :].unsqueeze(2) # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class ConditionWithRegion(nn.Module):
    """Compute Scale and bias for conditioning on region"""

    def __init__(
        self,
        regions: int,
        hidden_dim: int,
        num_feature_maps: int
    ):
        """
        Compute the scale and bias factors for
        conditioning convolutional blocks on the forecast time
        Args:
            forecast_steps: Number of forecast steps
            hidden_dim: Hidden dimension size
            num_feature_maps: Max number of channels in the blocks,
            to generate enough scale+bias values
            This means extra values will be generated, but keeps implementation simpler
        """
        super().__init__()
        self.forecast_steps = regions
        self.num_feature_maps = num_feature_maps
        self.lead_time_network = nn.ModuleList(
            [
                nn.Linear(in_features=self.forecast_steps, out_features=hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=2* num_feature_maps),
            ]
        )

    def forward(self, x: torch.Tensor, region: int) -> [torch.Tensor, torch.Tensor]:
        """
        Get the scale and bias for the conditioning layers
        From the FiLM paper, each feature map (i.e. channel) has its own scale and bias layer,
        so needs a scale and bias for each feature map to be generated
        Args:
            x: The Tensor that is used
            timestep: Index of the timestep to use, between 0 and forecast_steps
        Returns:
            2 Tensors of shape (Batch, num_feature_maps)
        """
        # One hot encode the timestep
        # lead time
        device = next(self.lead_time_network.parameters()).device
        regionsteps = region.to(device)

        for layer in self.lead_time_network:
            regionsteps = layer(regionsteps)
        scales_and_biases = regionsteps
        scales_and_biases = einops.rearrange(
            scales_and_biases, "b (block sb) -> b block sb", block=self.num_feature_maps, sb=2
        )
        return scales_and_biases[:,:,0],scales_and_biases[:,:,1]


if __name__ == '__main__':
    # m = UNet3dLite()
    # x = torch.randn(1, 1, 22, 140, 140)
    # m(x)
    test_2d_config()
    print()
    test_planar_configs()
    print('All tests sucessful!')
    # # TODO: Also test valid convolution architecture.
