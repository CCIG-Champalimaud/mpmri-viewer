import re
import numpy as np
import SimpleITK as sitk
import cv2
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from skimage.transform import resize
from skimage.feature import canny
from skimage.morphology import binary_dilation

from typing import List, Tuple, Dict, Any


def normalize(image: np.ndarray):
    minimum = image.min()
    maximum = image.max()
    return (image - minimum) / (maximum - minimum)


def read_sitk_array(path: str):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def get_filter_from_str(filter_str: str):
    if "===" in filter_str:
        key, value = filter_str.split("===")
        return lambda x: str(x[key]) == str(value) if key in x else False
    elif "!==" in filter_str:
        key, value = filter_str.split("!==")
        return lambda x: str(x[key]) != str(value) if key in x else True
    if "==" in filter_str:
        key, value = filter_str.split("==")
        return lambda x: x[key] == float(value) if key in x else False
    elif "!=" in filter_str:
        key, value = filter_str.split("!=")
        return lambda x: x[key] != float(value) if key in x else True
    elif ">" in filter_str:
        key, value = filter_str.split(">")
        return lambda x: x[key] > float(value) if key in x else False
    elif ">=" in filter_str:
        key, value = filter_str.split(">=")
        return lambda x: x[key] >= float(value) if key in x else False
    elif "<" in filter_str:
        key, value = filter_str.split("<")
        return lambda x: x[key] < float(value) if key in x else False
    elif "<=" in filter_str:
        key, value = filter_str.split("<=")
        return lambda x: x[key] <= float(value) if key in x else False


def get_filter_functions(filter_list_str: str):
    filters = filter_list_str.split()
    filters = [get_filter_from_str(s) for s in filters]
    return [f for f in filters if f is not None]


@dataclass
class VolumeDataset:
    path: str
    patterns: List[str] = field(default_factory=["*nii.gz", "*nii", "*mha"])
    recursive: bool = True
    image_pattern: str = r"image_[0-9A-Za-z]+"
    study_uid_pattern: str = r"[0-9\.]+\.[0-9\.]+"

    def __post_init__(self):
        self.collect_all_files()
        self.organize_files()

        self.all_study_uids = self.all_study_uids_full
        self.retro_conversion = self.retro_conversion_full
        self.volume_dictionary = self.volume_dictionary_full

    def path_glob(self, path: str, pattern: str) -> List[str]:
        if self.recursive is True:
            out = Path(path).rglob(pattern)
        else:
            out = Path(path).glob(pattern)
        return [str(x) for x in out]

    def collect_all_files(self):
        self.all_files = []
        for pattern in self.patterns:
            self.all_files.extend(self.path_glob(self.path, pattern))

    def organize_files(self):
        self.volume_dictionary_full = {}
        self.image_types = []
        for file in self.all_files:
            study_uid = re.search(self.study_uid_pattern, file)
            image_type = re.search(self.image_pattern, file)
            if study_uid is None or image_type is None:
                continue
            study_uid = study_uid.group()
            image_type = image_type.group()
            if study_uid not in self.volume_dictionary_full:
                self.volume_dictionary_full[study_uid] = {}
            if image_type not in self.image_types:
                self.image_types.append(image_type)
            self.volume_dictionary_full[study_uid][image_type] = file
        self.all_study_uids_full = list(self.volume_dictionary_full.keys())
        self.retro_conversion_full = {
            key: idx for idx, key in enumerate(self.all_study_uids_full)
        }

    def filter_volume_dataset(
        self, filter_str_list: str, metadata: Dict[str, Any]
    ):
        all_study_uids = deepcopy(self.all_study_uids_full)
        retro_conversion = deepcopy(self.retro_conversion_full)
        volume_dictionary = deepcopy(self.volume_dictionary_full)
        if (filter_str_list is not None) and (filter_str_list != ""):
            if len(metadata) > 0:
                filter_fns = get_filter_functions(filter_str_list)
                keys_to_remove = []
                for key in all_study_uids:
                    if key in metadata:
                        keep = all([fn(metadata[key]) for fn in filter_fns])
                        if keep == False:
                            keys_to_remove.append(key)
                    else:
                        keys_to_remove.append(key)
            all_study_uids = [
                study_uid
                for study_uid in all_study_uids
                if study_uid not in keys_to_remove
            ]
            retro_conversion = {
                study_uid: idx
                for idx, study_uid in enumerate(retro_conversion)
                if study_uid not in keys_to_remove
            }
            volume_dictionary = {
                study_uid: volume_dictionary[study_uid]
                for study_uid in volume_dictionary
                if study_uid not in keys_to_remove
            }

        self.all_study_uids = all_study_uids
        self.retro_conversion = retro_conversion
        self.volume_dictionary = volume_dictionary

    def __getitem__(self, key_or_idx: str | List[str | int] | int):
        if isinstance(key_or_idx, int):
            if key_or_idx < len(self.all_study_uids):
                key = self.all_study_uids[key_or_idx]
                return self[key]
        elif isinstance(key_or_idx, str):
            if key_or_idx in self.volume_dictionary:
                return self.volume_dictionary[key_or_idx]
        elif isinstance(key_or_idx, (list, tuple)):
            return [self[key] for key in key_or_idx]
        return {}

    def __len__(self):
        return len(self.all_study_uids)


@dataclass
class ImageLoader:
    size: Tuple[int]
    dataset: VolumeDataset
    mask_dataset: VolumeDataset = None
    display_mode: str = "crop"

    def __post_init__(self):
        self.all_images = []
        self.all_masks = []
        self.image_key = ""
        self.text_coords = self.size[0] // 16, self.size[1] // 4
        self.center_coords = self.size[0] // 2, self.size[1] // 2

    def init_array_if_necessary(self, sqrt_n_images: int):
        n_images = sqrt_n_images**2
        if hasattr(self, "array") is False:
            self.array = np.array([n_images, *self.size])
            self.array_tiles = np.zeros(
                [sqrt_n_images * self.size[0], sqrt_n_images * self.size[1]],
                dtype=np.uint8,
            )
        else:
            if n_images != self.array.shape[0]:
                self.array = np.array([n_images, *self.size])
                self.array_tiles = np.zeros(
                    [
                        sqrt_n_images * self.size[0],
                        sqrt_n_images * self.size[1],
                    ],
                    dtype=np.uint8,
                )

    def load_images_if_necessary(self, start_idx, n_images, image_key):
        self.image_idxs = [
            idx for idx in range(start_idx, start_idx + n_images)
        ]
        self.curr_study_uids = [
            self.dataset.all_study_uids[idx]
            if idx < len(self.dataset)
            else None
            for idx in self.image_idxs
        ]
        # always load if image_key is different
        if image_key != self.image_key:
            self.all_images = {
                idx: read_sitk_array(self.dataset[idx][image_key])
                if image_key in self.dataset[idx]
                else None
                for idx in self.image_idxs
            }
        else:
            idxs_to_delete = [
                idx for idx in self.all_images if idx not in self.image_idxs
            ]
            for idx in idxs_to_delete:
                del self.all_images[idx]
            for idx in self.image_idxs:
                if idx not in self.image_idxs:
                    if image_key in self.dataset[idx]:
                        self.all_images[idx] = read_sitk_array(
                            self.dataset[idx][image_key]
                        )
                    else:
                        self.all_images[idx] = None

    def load_masks_if_necessary(self, mask_key):
        if mask_key != self.image_key:
            self.all_masks = {}
            for idx in self.image_idxs:
                self.all_masks[idx] = None
                if idx < len(self.dataset):
                    curr_masks = self.mask_dataset[
                        self.dataset.all_study_uids[idx]
                    ]
                    if mask_key in curr_masks:
                        self.all_masks[idx] = read_sitk_array(
                            self.mask_dataset[
                                self.dataset.all_study_uids[idx]
                            ][mask_key]
                        )

        else:
            idxs_to_delete = [
                idx for idx in self.all_masks if idx not in self.image_idxs
            ]
            for idx in idxs_to_delete:
                del self.all_masks[idx]
            for idx in self.image_idxs:
                if idx not in self.image_idxs:
                    key = self.dataset.all_study_uids[idx]
                    if mask_key in self.mask_dataset[key]:
                        self.all_masks[idx] = read_sitk_array(
                            self.mask_dataset[key][mask_key]
                        )
                    else:
                        self.all_masks[idx] = None

    def crop_to_size(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape
        side1, side2 = [sh - size for sh, size in zip(shape, self.size)]
        a, b = side1 // 2, side2 // 2
        c, d = shape[0] - a, shape[1] - b
        return image[a:c, b:d]

    def resize_to_size(self, image: np.ndarray) -> np.ndarray:
        return resize(image, self.size)

    def update_array(
        self,
        sqrt_n_images: int,
        slice_idx: int,
        start_idx: int,
        image_key: str,
        mode: str = None,
        numbers: bool = True,
    ):
        if mode is None:
            mode = self.display_mode
        n_images = sqrt_n_images**2
        self.init_array_if_necessary(sqrt_n_images)
        self.load_images_if_necessary(start_idx, n_images, image_key)
        for i in range(sqrt_n_images):
            for j in range(sqrt_n_images):
                image = 0
                is_missing = True
                full_image = self.all_images[
                    self.image_idxs[i * sqrt_n_images + j]
                ]
                if full_image is not None:
                    is_missing = False
                    if slice_idx < full_image.shape[0]:
                        image = full_image[slice_idx]
                        if mode == "crop":
                            image = self.crop_to_size(image)
                        if mode == "resize":
                            image = self.resize_to_size(image)
                        image = normalize(image)
                        image = np.uint8(image * 255)
                x1, x2 = i * self.size[0], i * self.size[0] + self.size[0]
                y1, y2 = j * self.size[1], j * self.size[1] + self.size[1]
                self.array_tiles[x1:x2, y1:y2] = image
                if numbers is True:
                    if is_missing is False:
                        cv2.putText(
                            self.array_tiles,
                            str(i * sqrt_n_images + j + 1),
                            [
                                y1 + self.text_coords[0],
                                x1 + self.text_coords[1],
                            ],
                            cv2.FONT_ITALIC,
                            0.8,
                            thickness=2,
                            color=255,
                            lineType=cv2.LINE_AA,
                        )
                    else:
                        cv2.putText(
                            self.array_tiles,
                            "none",
                            [
                                y1 + self.text_coords[0],
                                x1 + self.text_coords[1],
                            ],
                            cv2.FONT_ITALIC,
                            0.8,
                            thickness=2,
                            color=128,
                            lineType=cv2.LINE_AA,
                        )

    def update_array_masks(
        self,
        sqrt_n_images: int,
        slice_idx: int,
        mask_key: str,
        mode: str = None,
    ):
        if mode is None:
            mode = self.display_mode
        self.load_masks_if_necessary(mask_key)
        for i in range(sqrt_n_images):
            for j in range(sqrt_n_images):
                mask = 0
                add_mask = False
                full_mask = self.all_masks[
                    self.image_idxs[i * sqrt_n_images + j]
                ]
                if full_mask is not None:
                    if slice_idx < full_mask.shape[0]:
                        mask = full_mask[slice_idx]
                        if mode == "crop":
                            mask = self.crop_to_size(mask)
                        if mode == "resize":
                            mask = self.resize_to_size(mask)
                        mask = normalize(mask)
                        mask = np.uint8(mask * 255)
                        add_mask = True
                if add_mask is True:
                    x, y = np.where(canny(mask) > 0)
                    x = x + i * self.size[0]
                    y = y + j * self.size[1]
                    self.array_tiles[x, y] = 255

    def retrieve_pil_image(
        self,
        sqrt_n_images: int,
        slice_idx: int,
        start_idx: int,
        image_key: str,
        mask_key: str,
        mode: str = None,
        numbers: bool = True,
    ):
        self.update_array(
            sqrt_n_images,
            slice_idx,
            start_idx,
            image_key,
            mode=mode,
            numbers=numbers,
        )
        if self.mask_dataset is not None:
            self.update_array_masks(
                sqrt_n_images, slice_idx, mask_key, mode=mode
            )
        return Image.fromarray(self.array_tiles)

    def retrieve_curr_study_uids(self, *args, **kwargs):
        return self.curr_study_uids
