import panel as pn
from src.app_lib import init_app

pn.extension(template="material", notifications=True)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--study_uid_pattern", default=r"[0-9\.]+\.[0-9\.]+")
parser.add_argument("--dataset_path", required=True)
parser.add_argument(
    "--patterns", nargs="+", type=str, default=["*mha", "*nii", "*nii.gz"]
)
parser.add_argument("--image_type_pattern", default=r"image_[a-zA-Z0-9]+")
parser.add_argument("--mask_path", default=None)
parser.add_argument(
    "--mask_patterns", nargs="+", type=str, default=["*mha", "*nii", "*nii.gz"]
)
parser.add_argument("--mask_type_pattern", default=r"mask_[a-zA-Z0-9]+")
parser.add_argument("--metadata_path", default=None)
parser.add_argument("--metadata_keys", nargs="+")
parser.add_argument("--max_slices", type=int, default=32)

args = parser.parse_args()

init_app(
    dataset_path=args.dataset_path,
    patterns=args.patterns,
    image_type_pattern=args.image_type_pattern,
    mask_path=args.mask_path,
    mask_patterns=args.mask_patterns,
    mask_type_pattern=args.mask_type_pattern,
    study_uid_pattern=args.study_uid_pattern,
    metadata_path=args.metadata_path,
    metadata_keys=args.metadata_keys,
    max_slices=args.max_slices,
)
