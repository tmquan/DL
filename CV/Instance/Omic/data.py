import numpy as np
from natsort import natsorted
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional

import cv2 

from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningDataModule

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged, AddChanneld, Resized, ScaleIntensityd, Flipd, Rotate90d,
    RandAdjustContrastd, RandHistogramShiftd, RandGaussianNoised, RandGaussianSmoothd, RandGaussianSharpend, 
    RandAffined, RandRotate90d, RandFlipd, RandZoomd, RandSpatialCropd, RandCropByPosNegLabeld, 
    ToTensord
)
from monai.data import CacheDataset, list_data_collate, create_test_image_2d
from monai.transforms.utils import rescale_array
from monai.config import print_config
# print_config()

from PIL import Image
import skimage.io
import skimage.measure
import skimage.segmentation

__all__ = [
    "PseudoDataModule", 
    "PairedDataModule", 
    "SingleClassPixelBasedDataModule", 
    "SingleClassRegionBasedDataModule"
]

"""
class CustomDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        pass

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass 

    def train_dataloader(self):
        train_split = Dataset(**kwargs)
        return DataLoader(train_split)

    def val_dataloader(self):
        val_split = Dataset(**kwargs)
        return DataLoader(val_split)

    def test_dataloader(self):
        test_split = Dataset(**kwargs)
        return DataLoader(test_split)
    
    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass
"""

def create_pseudo_image_2d(
    width: int,
    height: int,
    num_objs: int = 8,
    rad_max: int = 30,
    rad_min: int = 5,
    noise_max: float = 0.0,
    num_seg_classes: int = 5,
    channel_dim: Optional[int] = None,
    random_state: Optional[np.random.RandomState] = None,
    shape_type: Optional[str] = "bar",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a noisy 2D image with `num_objs` circles and a 2D mask image. The maximum and minimum radii of the circles
    are given as `rad_max` and `rad_min`. The mask will have `num_seg_classes` number of classes for segmentations labeled
    sequentially from 1, plus a background class represented as 0. If `noise_max` is greater than 0 then noise will be
    added to the image taken from the uniform distribution on range `[0,noise_max)`. If `channel_dim` is None, will create
    an image without channel dimension, otherwise create an image with channel dimension as first dim or last dim.
    Args:
        width: width of the image. The value should be larger than `2 * rad_max`.
        height: height of the image. The value should be larger than `2 * rad_max`.
        num_objs: number of circles to generate. Defaults to `12`.
        rad_max: maximum circle radius. Defaults to `30`.
        rad_min: minimum circle radius. Defaults to `5`.
        noise_max: if greater than 0 then noise will be added to the image taken from
            the uniform distribution on range `[0,noise_max)`. Defaults to `0`.
        num_seg_classes: number of classes for segmentations. Defaults to `5`.
        channel_dim: if None, create an image without channel dimension, otherwise create
            an image with channel dimension as first dim or last dim. Defaults to `None`.
        random_state: the random generator to use. Defaults to `np.random`.
    """

   

    image = np.zeros((height, width))
    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state  # type: ignore

    if shape_type=="circle":
        if rad_max <= rad_min:
            raise ValueError("`rad_min` should be less than `rad_max`.")
        if rad_min < 1:
            raise ValueError("`rad_min` should be no less than 1.")
        min_size = min(height, width)
        if min_size <= 2 * rad_max:
            raise ValueError("the minimal size of the image should be larger than `2 * rad_max`.")
            
        for _ in range(num_objs):
            x = rs.randint(rad_max, width - rad_max)
            y = rs.randint(rad_max, height - rad_max)
            rad = rs.randint(rad_min, rad_max)
            spy, spx = np.ogrid[-x : width - x, -y : height - y]
            circle = (spx * spx + spy * spy) <= rad * rad

            if num_seg_classes > 1:
                image[circle] = np.ceil(rs.random() * num_seg_classes)
            else:
                image[circle] = rs.random() * 0.5 + 0.5
        labels = np.ceil(image).astype(np.int32, copy=False)

        norm = rs.uniform(0, num_seg_classes * noise_max, size=image.shape)
        noisyimage: np.ndarray = rescale_array(np.maximum(image, norm))  # type: ignore

        if channel_dim is not None:
            if not (isinstance(channel_dim, int) and channel_dim in (-1, 0, 2)):
                raise AssertionError("invalid channel dim.")
            if channel_dim == 0:
                noisyimage = noisyimage[None]
                labels = labels[None]
            else:
                noisyimage = noisyimage[..., None]
                labels = labels[..., None]

        return noisyimage, labels
        
    elif shape_type=="bar":
        insts = np.zeros((0, height, width), dtype=np.uint8)
        for _ in range(num_objs):
            x = np.random.randint(int(width/8), int(7*width/8))
            y = np.random.randint(int(height/8), int(7*height/8))
            w = 15
            h = np.random.randint(80, 100)
            theta = np.random.randint(-90, 90)
            rect = ([x, y], [w, h], theta)
            box = np.int0(cv2.boxPoints(rect))

            if num_seg_classes > 1:
                # image = cv2.fillPoly(image, [box], np.ceil(rs.random() * num_seg_classes))
                gt = np.zeros_like(image)
                gt = cv2.fillPoly(gt, [box], np.ceil(rs.random() * num_seg_classes))
                insts[:, gt != 0] = 0
                insts = np.concatenate([insts, gt[np.newaxis]])
                image = cv2.fillPoly(image, [box], 255)
                image = cv2.drawContours(image, [box], 0, 0, 2)
            else:
                # image = cv2.fillPoly(image, [box], rs.random() * 0.5 + 0.5)
                gt = np.zeros_like(image)
                gt = cv2.fillPoly(gt, [box], rs.random() * 0.5 + 0.5)
                insts[:, gt != 0] = 0
                insts = np.concatenate([insts, gt[np.newaxis]])
                image = cv2.fillPoly(image, [box], 255)
                image = cv2.drawContours(image, [box], 0, 0, 2)

        # labels = np.ceil(image).astype(np.int32, copy=False)
        labels = np.max(insts, 0)
        image = cv2.drawContours(image, [box], 0, 0, 2)
        image[labels==0] = 255
        norm = rs.uniform(0, num_seg_classes * noise_max, size=image.shape)
        noisyimage: np.ndarray = rescale_array(np.maximum(image, norm))  # type: ignore
        return noisyimage, labels
    

class PseudoDataModule(LightningDataModule):
    def __init__(self, 
        batch_size: int=32,
    ):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 32.
        """
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_data_dicts = self.make_dict(size=500)
        self.val_data_dicts = self.make_dict(size=100)
        self.test_data_dicts = self.make_dict(size=100)
        set_determinism(seed=0)

    def make_dict(self, size=500) -> Dict[str, List[str]]:
        # Create a dictionary of image and label files
        images = []
        labels = []

        num_objects = 8
        height = 256
        width = 256
        for _ in range(size):
            # image, label = create_test_image_2d(width=256, height=256, channel_dim=0)
            # label = (image > 0).astype(np.float32)
            
            images.extend(image)
            labels.extend(label)
        data_dicts = [
            {"image": image,  
             "label": label} for image, label in zip(images, labels)
        ]
        return data_dicts

    def _shared_dataloader(self, data_dicts, transforms=None, shuffle=True, drop_last=False, num_workers=8):
        dataset = CacheDataset(
            data=data_dicts, 
            cache_rate=1.0, 
            num_workers=num_workers,
            transform=transforms,
        )
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            num_workers=num_workers, 
            collate_fn=list_data_collate,
            shuffle=shuffle,
        )
        return dataloader

    def train_dataloader(self):
        train_transforms = Compose(
            [
                # Basic Augmentationm
                AddChanneld(keys=["image", "label"]),
                ToTensord(keys=["image", "label"]),
            ]
        )
        return self._shared_dataloader(self.train_data_dicts, 
            transforms=train_transforms, 
            shuffle=True,
            drop_last=False,
            num_workers=4
        )
    
    def val_dataloader(self):
        val_transforms = Compose(
            [
                # Basic Augmentationm
                AddChanneld(keys=["image", "label"]),
                ToTensord(keys=["image", "label"]),
            ]
        )
        return self._shared_dataloader(self.val_data_dicts, 
            transforms=val_transforms, 
            shuffle=False,
            drop_last=False,
            num_workers=2
        )

    def test_dataloader(self):
        test_transforms = Compose(
            [
                # Basic Augmentationm
                AddChanneld(keys=["image", "label"]),
                ToTensord(keys=["image", "label"]),
            ]
        )
        return self._shared_dataloader(self.test_data_dicts, 
            transforms=test_transforms, 
            shuffle=False,
            drop_last=False,
            num_workers=2
        )
    
class PairedDataModule(LightningDataModule):
    def __init__(self, 
        batch_size: int=32,
        train_image_dirs: List[str]=['/data/train/images'],
        train_label_dirs: List[str]=['/data/train/labels'], 
        val_image_dirs: List[str]=['/data/val/images'], 
        val_label_dirs: List[str]=['/data/val/labels'],
        test_image_dirs: List[str]=['/data/test/images'],
        test_label_dirs: List[str]=['/data/test/labels'],
    ):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 32.
            train_image_dirs (List[str], optional): [description]. Defaults to ['/data/train/images'].
            train_label_dirs (List[str], optional): [description]. Defaults to ['/data/train/labels'].
            val_image_dirs (List[str], optional): [description]. Defaults to ['/data/val/images'].
            val_label_dirs (List[str], optional): [description]. Defaults to ['/data/val/labels'].
            test_image_dirs (List[str], optional): [description]. Defaults to ['/data/test/images'].
            test_label_dirs (List[str], optional): [description]. Defaults to ['/data/test/labels'].
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_image_dirs = train_image_dirs
        self.train_label_dirs = train_label_dirs
        self.val_image_dirs = val_image_dirs
        self.val_label_dirs = val_label_dirs
        self.test_image_dirs = test_image_dirs
        self.test_label_dirs = test_label_dirs

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
    
    def glob_dict(
        self, 
        image_dirs: List[str], 
        label_dirs: List[str],
        ext: str='*.png',
    ) -> Dict[str, List[str]]:
        assert image_dirs is not None and label_dirs is not None
        assert len(image_dirs) == len(label_dirs)
        
        # Glob all image files in image_dirs
        image_paths = [Path(folder).rglob(ext) for folder in image_dirs]
        image_files = natsorted([str(path) for path_list in image_paths for path in path_list])

        # Glob all label files in label_dirs
        label_paths = [Path(folder).rglob(ext) for folder in label_dirs]
        label_files = natsorted([str(path) for path_list in label_paths for path in path_list])

        # Check that the number of image and label files match
        print(f'Found {len(image_files)} images and {len(label_files)} labels.')
        assert len(image_files) == len(label_files)

        # Create a dictionary of image and label files
        data_dicts = [
            {"image": image_file,  
             "label": label_file} for image_file, label_file in zip(image_files, label_files)
        ]
        return data_dicts

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_data_dicts = self.glob_dict(self.train_image_dirs, self.train_label_dirs, ext='*.png')
        self.val_data_dicts = self.glob_dict(self.val_image_dirs, self.val_label_dirs, ext='*.png')
        self.test_data_dicts = self.glob_dict(self.test_image_dirs, self.test_label_dirs, ext='*.png')
        set_determinism(seed=0)

class SingleClassPixelBasedDataModule(PairedDataModule):
    def __init__(self, 
        batch_size: int=32,
        train_image_dirs: List[str]=['/data/train/images'],
        train_label_dirs: List[str]=['/data/train/labels'], 
        val_image_dirs: List[str]=['/data/val/images'], 
        val_label_dirs: List[str]=['/data/val/labels'],
        test_image_dirs: List[str]=['/data/test/images'],
        test_label_dirs: List[str]=['/data/test/labels'],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_image_dirs = train_image_dirs
        self.train_label_dirs = train_label_dirs
        self.val_image_dirs = val_image_dirs
        self.val_label_dirs = val_label_dirs
        self.test_image_dirs = test_image_dirs
        self.test_label_dirs = test_label_dirs

    def _shared_dataloader(self, data_dicts, transforms=None, shuffle=True, drop_last=False, num_workers=8):
        dataset = CacheDataset(
            data=data_dicts, 
            cache_rate=1.0, 
            num_workers=num_workers,
            transform=transforms,
        )
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            num_workers=num_workers, 
            collate_fn=list_data_collate,
            shuffle=shuffle,
        )
        return dataloader

    def train_dataloader(self):
        train_transforms = Compose(
            [
                # Basic Augmentationm
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Rotate90d(keys=["image", "label"]),
                Flipd(keys=["image", "label"], spatial_axis=0),
                Resized(
                    keys=["image", "label"], 
                    spatial_size=(640, 640),
                    mode=["bilinear", "nearest"],
                ),
                ScaleIntensityd(
                    keys=["image", "label"], 
                    minv=0.0, 
                    maxv=1.0,
                ),
                # Advanced - Heavy Augmentation
                RandAdjustContrastd(
                    keys=["image"],
                    prob=0.25,
                ), 
                RandHistogramShiftd(
                    keys=["image"],
                    prob=0.25,
                ),
                RandGaussianNoised(
                    keys=["image"],
                    prob=0.25,
                ),
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=0.25,
                ),
                RandGaussianSharpend(
                    keys=["image"],
                    prob=0.25,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=["bilinear", "nearest"],
                    prob=0.5,
                    spatial_size=(640, 640),
                    rotate_range=(0, np.pi/4),
                    scale_range=(0.3, 0.3)
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.25,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    prob=0.25,
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    # pos_value=1.0,
                    # neg_value=0.0,
                    spatial_size=(512, 512),                    
                ), 
                ToTensord(keys=["image", "label"]),
            ]
        )
        return self._shared_dataloader(self.train_data_dicts, 
            transforms=train_transforms, 
            shuffle=True,
            drop_last=False,
            num_workers=4
        )
    
    def val_dataloader(self):
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Rotate90d(keys=["image", "label"]),
                Flipd(keys=["image", "label"], spatial_axis=0),
                Resized(
                    keys=["image", "label"], 
                    spatial_size=(512, 512),
                    mode=["bilinear", "nearest"],
                ),
                ScaleIntensityd(
                    keys=["image", "label"], 
                    minv=0.0, 
                    maxv=1.0,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )
        return self._shared_dataloader(self.val_data_dicts, 
            transforms=val_transforms, 
            shuffle=False,
            drop_last=False,
            num_workers=2
        )

    def test_dataloader(self):
        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Rotate90d(keys=["image", "label"]),
                Flipd(keys=["image", "label"], spatial_axis=0),
                Resized(
                    keys=["image", "label"], 
                    spatial_size=(512, 512),
                    mode=["bilinear", "nearest"],
                ),
                ScaleIntensityd(
                    keys=["image", "label"], 
                    minv=0.0, 
                    maxv=1.0,
                ),
            ]
        )
        return self._shared_dataloader(self.test_data_dicts, 
            transforms=test_transforms, 
            shuffle=False,
            drop_last=False,
            num_workers=2
        )

if __name__ == '__main__':
    set_determinism(seed=42)
    datamodule = PairedDataModule(
        train_image_dirs=[],
        train_label_dirs=[],
        val_image_dirs=[],
        val_label_dirs=[],
        test_image_dirs=[],
        test_label_dirs=[],
    )
    datamodule.prepare_data()
    datamodule.setup()
    # print(datamodule.train_dataloader())
    # datamodule.teardown()
    # print(datamodule.train_dataloader())

    ###############################################################################
    set_determinism(seed=42)
    datamodule = SingleClassPixelBasedDataModule(
        train_image_dirs=[],
        train_label_dirs=[],
        val_image_dirs=[],
        val_label_dirs=[],
        test_image_dirs=[],
        test_label_dirs=[],
    )
    datamodule.prepare_data()
    datamodule.setup()
