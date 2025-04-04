from itertools import chain
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

# from src.datasets.tsm_scan import CTSlice as _CTSlice
from src.datasets.normalizers import get_normalization_scheme
from src.datasets.tsm_scan import CTScan as _CTScan


class CTScan(_CTScan):
    def __getitem__(self, index):
        s = super().__getitem__(index)
        s['image'] = s['image'].transpose(2, 1).flip((1, 2))
        if s['masks'].shape[0] != 0:
            s['masks'] = s['masks'].transpose(2, 1).flip((1, 2))
        return s


class TUHDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, split: str, split_dir: str = 'splits', limit_scans: int = 99999, **scan_params):
        self.root_dir = Path(root_dir)
        self.split = split

        self.ann_path = self.root_dir / split_dir / f'{split}_scans.csv'
        assert self.ann_path.exists()
        with open(self.ann_path, 'r') as fid:
            scan_names = set(fid.read().splitlines())
        assert scan_names, f'No scans found in split: {self.ann_path}'

        scans_dir = self.root_dir / 'imagesTr'
        labels_dir = self.root_dir / 'labelsTr'
        self.scans = []
        for i, sp in enumerate(scans_dir.rglob('*.nii.gz')):
            if i > limit_scans:
                break
            sname = sp.name.replace('_0000', '')
            if sname not in scan_names:
                continue
            # self.scans.append(CTScan(sp, labels_dir / sp.parent.name / sname, **scan_params))
            self.scans.append(CTScan(sp, labels_dir / sname, **scan_params))

        self.scans_dataset = ConcatDataset(self.scans)
        self.classes = self.scans[0].classes

    def get_sampling_labels(self):
        lbs = list(chain.from_iterable(scan.get_sampling_labels() for scan in self.scans))
        print(f'[TUH dataset] Number of slices with positive sampling label:', sum(lbs))
        return lbs

    def __len__(self):
        return len(self.scans_dataset)

    def __getitem__(self, index):
        return self.scans_dataset[index]


import os
from os.path import join

import imageio
import numpy as np

# class FibDataset(_CTSlice):
#     def __init__(self, root_dir:str, split:str, split_dir:str='splits', **scan_params):
#         self.root_dir = Path(root_dir)
#         self.img_dir = self.root_dir / 'orig'
#         self.label_dir = self.root_dir / 'orig_mask'
#         # self.split_dir = split_dir
#         self.split_dir = Path(split_dir)
#         self.split = split

#         if split == 'train':
#             self.ann_path = join(split_dir,f'{split}.txt')
#             test_path = join(split_dir, f'test.txt')
#             test_names = np.loadtxt(test_path, dtype=str).tolist()
#             train_names = np.loadtxt(self.ann_path, dtype=str).tolist()
#             self.names = train_names + test_names
#         elif split == 'val':
#             self.ann_path = self.root_dir / split_dir / f'{split}.txt'
#             self.names = np.loadtxt(self.ann_path, dtype=str).tolist()

# self.scans = []
# for i, sp in enumerate(self.img_dir.rglob('*.png')):
#     label_name = sp.name.replace('.png', '_mask.png')
#     self.scans.append(CTScan(sp, self.label_dir / label_name, **scan_params))


class FibDataset(torch.utils.data.Dataset):
    from typing import Union

    import albumentations as albu

    slicing_dims = {
        'sagittal': 0,  # side view
        'coronal': 1,  # front view
        'axial': 2,  # top down view
    }

    def __init__(
        self,
        root_dir,
        split,
        split_dir,
        norm_scheme: dict = {'kind': 'minmax'},
        transforms: albu.Compose = None,
        min_max_normalization: bool = True,
        slicing_direction: str = 'axial',
        classes: list[str] = ('empty', 'fibrosis'),
        sampling_class: str = None,
        classify_labels: str = None,
        classify_labels_thresh: int = 32,
        filter_class_slices: str = None,
        filter_class_slices_thresh: int = 32,
        synth_params: dict = None,
        load_masks: Union[bool, list] = False,
        default_label: int = None,
        blur_background_sigma: int = None,
        counterfactual_val: bool = False,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / 'orig'
        self.label_dir = self.root_dir / 'orig_mask'

        self.root_dir_0 = Path('/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/no_fibrosis/fid_dataset')
        self.img_dir_0 = self.root_dir_0 / 'orig_covid'

        self.split_dir = Path(split_dir)
        self.split = split

        self.counterfactual_val = counterfactual_val
        self.names = self.get_split_name()
        if self.names[0].endswith(('.png', '.jpg', '.jpeg')):
            self.names = [Path(name).stem for name in self.names]
        self.min_max_norm = min_max_normalization
        self.transforms = transforms
        self.slicing_dim = self.slicing_dims[slicing_direction]

        # self.name = scan_path.stem
        # self.scan_path = scan_path
        # self.labels_path = labels_path

        self.sampling_class = sampling_class

        self.classify_labels = classify_labels
        self.classify_labels_thresh = classify_labels_thresh
        self.filter_class_slices = filter_class_slices
        self.filter_class_slices_thresh = filter_class_slices_thresh
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.synth_params = synth_params
        self.anomaly_template, self.anomaly_transforms = None, None

        self.load_masks = load_masks

        self.slice_indices = None

        self.labels = None
        if self.classify_labels:
            self.labels = self._get_slices_with_classes_mask(classify_labels, classify_labels_thresh).astype(np.uint8)
            if self.slice_indices is not None:
                self.labels = self.labels[self.slice_indices]
        self.default_label = default_label
        self.blur_background_sigma = blur_background_sigma
        # self.norm = get_normalization_scheme(**norm_scheme)

    def get_slice_indices(self, filter_classes: list[str]) -> list[int]:
        filter_mask = self._get_slices_with_classes_mask(filter_classes, self.filter_class_slices_thresh)
        indices = np.nonzero(filter_mask)[0]
        if indices.shape[0] == 0:
            print(f'{self} has no slices for classes: {filter_classes}')
        return indices

    def get_ith_slice(self, volume: np.ndarray, index: int) -> np.ndarray:
        return volume[self._get_slicer(index)]

    def get_split_name(self):
        if self.split == 'train':
            self.ann_path = self.split_dir / f'{self.split}.txt'

            train_path_0 = self.split_dir / 'train_covid.txt'

            train_names_1 = np.loadtxt(self.ann_path, dtype=str).tolist()
            train_names_0 = np.loadtxt(train_path_0, dtype=str).tolist()

            names = train_names_1 + train_names_0
        elif self.split == 'val':
            self.ann_path = self.split_dir / f'{self.split}.txt'
            val_path_0 = self.split_dir / 'val_covid.txt'

            train_names_1 = np.loadtxt(self.ann_path, dtype=str).tolist()
            train_names_0 = np.loadtxt(val_path_0, dtype=str).tolist()

            if self.counterfactual_val:
                names = train_names_1
            else:
                names = train_names_1 + train_names_0

        elif self.split == 'test':
            self.ann_path = self.split_dir / f'{self.split}.txt'
            test_name = np.loadtxt(self.ann_path, dtype=str).tolist()
            names = test_name

        return names

    def __len__(self):
        # return len(self.slice_indices) if self.slice_indices is not None else self.scan.shape[self.slicing_dim]
        return len(self.names)

    def __getitem__(self, index):
        if 'data/OSIC/processed/fibrosis' in self.root_dir.as_posix():
            # get item for OSIC dataset from new repository 'genai-wsss'. 
            # Now only check for inference for test split.
            self.img_dir = self.root_dir
            img_path = self.img_dir / (self.names[index] + '.png')

            scan_slice = np.asarray(imageio.imread(img_path)).astype(np.uint8)
            scan_slice = scan_slice[:, :, 0]

            sample = {'image': scan_slice}
            if self.transforms:
                sample = self.transforms(**sample)

            sample['image'] = torch.from_numpy(sample['image'])

            sample['scan_name'] = self.names[index]
            sample['slice_index'] = index

            sample['image'] = sample['image'].transpose(1, 0).flip((0, 1))
            sample['image'] = sample['image'].unsqueeze(0)


            return sample
            
        
        else:
            # Old repository, 'fid_dataset/fibrosis/orig, mask'
            if 'cov' in self.names[index]:
                img_path = self.img_dir_0 / (self.names[index] + '.png')
                scan_slice = np.asarray(imageio.imread(img_path)).astype(np.uint8)
                scan_slice = scan_slice[:, :, 0]

                mask = np.zeros_like(scan_slice).astype(np.uint8)
                label = np.zeros(1)
            else:
                img_path = self.img_dir / (self.names[index] + '.png')
                label_path = self.label_dir / (self.names[index] + '_mask.png')
                scan_slice = np.asarray(imageio.imread(img_path)).astype(np.uint8)
                scan_slice = scan_slice[:, :, 0]
                label_slice = np.asarray(imageio.imread(label_path))
                label_slice = (label_slice / 255).astype(np.uint16)
                assert scan_slice.shape == label_slice.shape, 'Shapes of provided scan and labels volumes do not match'

                classes = self.load_masks if isinstance(self.load_masks, list) else self.classes
                mask = (label_slice == self.class_to_idx[classes[-1]]).astype(np.uint8)
                label = np.ones(1)

            sample = {'image': scan_slice, 'mask': mask}
            if self.transforms:
                sample = self.transforms(**sample)

            sample['image'] = torch.from_numpy(sample['image'])
            sample['mask'] = torch.from_numpy(np.stack(sample['mask']))
            sample['label'] = torch.tensor(label).long()
            sample['scan_name'] = self.names[index]
            sample['slice_index'] = index

            sample['image'] = sample['image'].transpose(1, 0).flip((0, 1))
            sample['image'] = sample['image'].unsqueeze(0)

            sample['mask'] = sample['mask'].transpose(1, 0).flip((0, 1))
            sample['mask'] = sample['mask'].unsqueeze(0)

            # plt.imsave('image.png', sample['image'].numpy(), cmap='gray')
            return sample

    def __str__(self) -> str:
        return self.name

class AustraliaDataset(torch.utils.data.Dataset):
    from typing import Union

    import albumentations as albu

    slicing_dims = {
        'sagittal': 0,  # side view
        'coronal': 1,  # front view
        'axial': 2,  # top down view
    }

    def __init__(
        self,
        root_dir,
        split,
        split_dir,
        norm_scheme: dict = {'kind': 'minmax'},
        transforms: albu.Compose = None,
        min_max_normalization: bool = True,
        slicing_direction: str = 'axial',
        classes: list[str] = ('empty', 'fibrosis'),
        sampling_class: str = None,
        classify_labels: str = None,
        classify_labels_thresh: int = 32,
        filter_class_slices: str = None,
        filter_class_slices_thresh: int = 32,
        synth_params: dict = None,
        load_masks: Union[bool, list] = False,
        default_label: int = None,
        blur_background_sigma: int = None,
        counterfactual_val: bool = False,
    ):
        super().__init__()
        # root_dir = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/unet_train_test/quantification_result/02_00019'
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir

        self.counterfactual_val = counterfactual_val

        self.names = os.listdir(self.root_dir)
        self.names = [name.removesuffix('.png') for name in self.names if name.endswith('.png')]

        self.min_max_norm = min_max_normalization
        self.transforms = transforms #resize to 256
        self.slicing_dim = self.slicing_dims[slicing_direction]

        self.sampling_class = sampling_class

        self.load_masks = load_masks




    def get_ith_slice(self, volume: np.ndarray, index: int) -> np.ndarray:
        return volume[self._get_slicer(index)]

    def get_split_name(self):
        if  self.split == 'test':
            self.ann_path = self.split_dir / f'{self.split}.txt'
            test_name = np.loadtxt(self.ann_path, dtype=str).tolist()
            names = test_name
        else:
            raise ValueError(f'Invalid split: {self.split}')

        return names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):

        img_path = self.img_dir / (self.names[index] + '.png')

        scan_slice = np.asarray(imageio.imread(img_path)).astype(np.uint8)
        scan_slice = scan_slice[:, :, 0]

        sample = {'image': scan_slice}
        if self.transforms:
            sample = self.transforms(**sample)

        sample['image'] = torch.from_numpy(sample['image'])

        sample['scan_name'] = self.names[index]
        sample['slice_index'] = index

        sample['image'] = sample['image'].transpose(1, 0).flip((0, 1))
        sample['image'] = sample['image'].unsqueeze(0)


        return sample

    def __str__(self) -> str:
        return self.name