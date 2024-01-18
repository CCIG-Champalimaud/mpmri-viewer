import panel as pn
from src.app_lib import init_app

pn.extension(template="material", notifications=True)

import argparse

desc = """
Displays a dataset composed of SimpleITK-readable images in a user-friendly 
interface. Supports the display of masks and complementary metadata, as well as
filtering according to the metadata.

PLEASE NOTE this must be used with the `panel serve` CLI and does not work if
used as a regular python script.
"""

parser = argparse.ArgumentParser(
    description=desc, usage="panel serve app.py --args [ARGS]"
)

parser.add_argument(
    "--study_uid_pattern",
    default=r"[0-9\.]+\.[0-9\.]+",
    help="Pattern used to extract the study UID.",
)
parser.add_argument(
    "--dataset_path",
    required=True,
    help="Path to the dataset (can be relative or absolute)",
)
parser.add_argument(
    "--patterns",
    nargs="+",
    type=str,
    default=["*mha", "*nii", "*nii.gz"],
    help="Patterns used to retrieve the different images",
)
parser.add_argument(
    "--image_type_pattern",
    default=r"image_[a-zA-Z0-9]+",
    help="Pattern used to identify different types of sequences",
)
parser.add_argument(
    "--mask_path", default=None, help="Path to masks for the dataset"
)
parser.add_argument(
    "--mask_patterns",
    nargs="+",
    type=str,
    default=["*mha", "*nii", "*nii.gz"],
    help="Patterns used to retrieve the different masks",
)
parser.add_argument(
    "--mask_type_pattern",
    default=r"mask_[a-zA-Z0-9]+",
    help="Pattern used to identify different types of masks",
)
parser.add_argument(
    "--metadata_path",
    default=None,
    help="Path to metadata. Must be a JSON structured like \
        study_uid:{key1:value1, key2:value2}. The study_uid must be identical \
        to that inferred in study_uid_pattern",
)
parser.add_argument(
    "--metadata_keys",
    nargs="+",
    help="Keys in each metadata entry corresponding to specific fields",
)
parser.add_argument(
    "--max_slices",
    type=int,
    default=32,
    help="Maximum number of support slices for 3D volumes",
)

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
