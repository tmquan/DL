from typing import Any, Callable, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, AdamW, SGD
import torchvision

from pytorch_lightning import LightningModule

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss

from loss import create_loss

# class UNet(nn.Module):
#     """
#     Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
#     <https://arxiv.org/abs/1505.04597>`_
#     Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
#     Implemented by:
#         - `Annika Brundyn <https://github.com/annikabrundyn>`_
#         - `Akshay Kulkarni <https://github.com/akshaykvnit>`_
#     Args:
#         num_classes: Number of output classes required
#         input_channels: Number of channels in input images (default 3)
#         num_layers: Number of layers in each side of U-net (default 5)
#         features_start: Number of features in first layer (default 64)
#         bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
#     """

#     def __init__(
#         self,
#         num_classes: int,
#         input_channels: int = 3,
#         num_layers: int = 5,
#         features_start: int = 64,
#         bilinear: bool = False,
#     ):

#         if num_layers < 1:
#             raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

#         super().__init__()
#         self.num_layers = num_layers

#         layers = [DoubleConv(input_channels, features_start)]

#         feats = features_start
#         for _ in range(num_layers - 1):
#             layers.append(Down(feats, feats * 2))
#             feats *= 2

#         for _ in range(num_layers - 1):
#             layers.append(Up(feats, feats // 2, bilinear))
#             feats //= 2

#         layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

#         self.layers = nn.ModuleList(layers)

#     def forward(self, x):
#         xi = [self.layers[0](x)]
#         # Down path
#         for layer in self.layers[1 : self.num_layers]:
#             xi.append(layer(xi[-1]))
#         # Up path
#         for i, layer in enumerate(self.layers[self.num_layers : -1]):
#             xi[-1] = layer(xi[-1], xi[-2 - i])
#         return self.layers[-1](xi[-1])


# class DoubleConv(nn.Module):
#     """[ Conv2d => BatchNorm (optional) => ReLU ] x 2."""

#     def __init__(self, in_ch: int, out_ch: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.net(x)


# class Down(nn.Module):
#     """Downscale with MaxPool => DoubleConvolution block."""

#     def __init__(self, in_ch: int, out_ch: int):
#         super().__init__()
#         self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

#     def forward(self, x):
#         return self.net(x)


# class Up(nn.Module):
#     """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
#     map from contracting path, followed by DoubleConv."""

#     def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
#         super().__init__()
#         self.upsample = None
#         if bilinear:
#             self.upsample = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#                 nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
#             )
#         else:
#             self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

#         self.conv = DoubleConv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.upsample(x1)

#         # Pad x1 to the size of x2
#         diff_h = x2.shape[2] - x1.shape[2]
#         diff_w = x2.shape[3] - x1.shape[3]

#         x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

#         # Concatenate along the channels axis
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

class PixelEmbeddingUNetModule(LightningModule):
    def __init__(self):
        super().__init__()
        # self.net = nn.Sequential(   
        #     UNet(
        #         dimensions=2,
        #         in_channels=1,
        #         out_channels=16,
        #         channels=(32, 64, 128, 256, 512),
        #         strides=(2, 2, 2, 2),
        #         num_res_units=2,
        #         norm=Norm.BATCH,
        #         dropout=0.5,
        #     ), 
        #     # nn.Sigmoid()
        # )
        
        # self.loss_func = DiceLoss(to_onehot_y=False, 
        #                               sigmoid=False, 
        #                               squared_pred=False)
        
        self.net = UNet(
            dimensions=2,
            in_channels=1,
            out_channels=16,
            channels=(64, 128, 256, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.5,
        )

        # self.net = UNet(
        #     num_classes=1, 
        #     input_channels=1,
        #     num_layers=5,
        #     features_start=16,
        #     bilinear=False,
        # )

        self.loss_func = create_loss(
            delta_var=0.5, 
            delta_dist=2.0, 
            alpha=1.0, 
            beta=1.0, 
            gamma=0.001, 
            unlabeled_push_weight=0.0, 
            instance_term_weight=1.0,
            consistency_weight=1.0, 
            kernel_threshold=0.5, 
            instance_loss='dice', 
            spoco=False)

    def forward(self, x):
        return self.net(x)
        
    def configure_optimizers(self):
        optimizer = AdamW(self.net.parameters(), lr=1e-4)
        return optimizer
        
    def training_step(self, batch, batch_idx, stage: Optional[str]='train'):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_func(output, labels)
        if batch_idx==0:
            viz = torch.cat([images, labels.unsqueeze(1)], dim=-1)#[:8]
            grid = torchvision.utils.make_grid(viz, nrow=4, normalize=True, scale_each=True, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid, self.current_epoch)
        return {"loss": loss}

    def _shared_step(self, batch, batch_idx, stage: Optional[str]='_shared'):
        images, labels = batch["image"], batch["label"]
        # print(images.shape, labels.shape)
        output = self.forward(images)
        
        loss = self.loss_func(output, labels)
        if batch_idx==0:
            viz = torch.cat([images, labels.unsqueeze(1)], dim=-1)#[:8]
            grid = torchvision.utils.make_grid(viz, nrow=4, normalize=True, scale_each=True, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid, self.current_epoch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage='test')

    def training_epoch_end(self, outputs, stage: Optional[str]='train'):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _shared_epoch_end(self, outputs, stage: Optional[str]='_shared'):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, stage='validation')
    
    def test_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, stage='test')
