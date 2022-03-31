import numpy as np
from natsort import natsorted
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
from monai.data import CacheDataset, list_data_collate
from monai.config import print_config
# print_config()

from icevision.imports import *
from icevision import *
from icevision.parsers.parser import *

from PIL import Image
import skimage.io
import skimage.measure
import skimage.segmentation

__all__ = [
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

class PairedDataModule(LightningDataModule):
    def __init__(self, 
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

class RegionBasedParser(Parser):
    def __init__(
        self,
        data_dicts: Dict[str, Any],
        class_map: ClassMap = None,
    ):
        super().__init__(template_record=self.template_record())
        self.class_map = ClassMap(list(["foreground"])) if class_map is None else class_map
        self.data_dicts = data_dicts
   
    def __iter__(self):
        yield from [data_dict["image"] for data_dict in self.data_dicts]

    def __len__(self):
        return len(self.data_dicts)

    def template_record(self) -> BaseRecord:
        return BaseRecord(
            (
                FilepathRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
                InstanceMasksRecordComponent(),
            )
        )
    
    def parse_fields(self, o, record, is_new=True):
        img = Image.open(o).convert("RGB")
        record.set_filepath(o)
        record.set_img_size(img.size)
        record.detection.set_class_map(self.class_map)
        
        # print(data_dict.keys())
        # data_dict = dict(filter(lambda elem: elem['image'] == o, self.data_dicts))
        data_dict = next(item for item in self.data_dicts if item["image"] == o)
        # print(data_dict)
        for key in {key:data_dict[key] for key in data_dict if key!="image"}.keys():
            # print(data_dict["image"], data_dict[key])
            #
            # Process the Mask
            #
            masks = np.array(Image.open(data_dict[key]).convert("L"))
            insts, num_insts = skimage.measure.label(masks, return_num=True)
            self._num_objects = num_insts # Don't count background
            if num_insts>0:
                insts_id = np.arange(1, num_insts+1)
                record.detection.add_masks([ MaskArray( insts == insts_id[:, None, None]  ) ])
                
                #
                # Process the bounding boxes from region props of instancve
                #
                props = skimage.measure.regionprops(insts) 
                for prop in props:
                    # record.detection.add_labels(["foreground"])
                    record.detection.add_labels([key])
                    record.detection.add_bboxes([BBox.from_xyxy(prop.bbox[1]-1, 
                                                                prop.bbox[0]-1, 
                                                                prop.bbox[3]+1, 
                                                                prop.bbox[2]+1)]) #Bounding box (min_row, min_col, max_row, max_col)
            
    def prepare(self, data_dict):
        self._record_id = getattr(self, "_record_id", 0) + 1
        for data_dict in self.data_dicts:
            self._data_dict = data_dict 
            self._filepath = data_dict["image"]
            
    def record_id(self, data_dict) -> int:
        return self._record_id

    def data_dict(self, o) -> Union[str, Dict]:
        return self._data_dict
    
    def filepath(self, o) -> Union[str, Path]:
        return self._filepath


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
