# Volume dataset viewer

Simple viewer implemented in `Panel`, enabling the quick visualization of large volumes of SimpleITK-readable files. This was developed for prostate mpMRI, but few changes are necessary to get it up and running for other data types.

## Usage 

### Installing pre-requisites

```
pip install -r requirements
```

### Running as command line tool

Very little input is necessary to get this up and running. `app.py` is the script orchestrating everything and the relevant code is contained in `src`. To launch this simple visualizer all that is necessary is:

1. Serve a panel app using `panel serve app.py`. A single command line argument is necessary and should be specified with `--args`, i.e. `panel serve app.py --args --dataset_path PATH_TO_FOLDER_CONTAINING_STUDIES`. Together with this, other arguments can be specified:

    * `--patterns` - patterns used to recursively glob file extensions corresponding to images
    * `--study_uid_pattern` - pattern used to collect, from the file path, the study UID (should be regex compatible)
    * `--image_type_pattern` - pattern used to assign volumes to different image types. Volumes should be organized into different folders with characteristic names corresponding to different sequence types (i.e. `study_1/image_type_1.nii.gz`, `study_1/image_type_2.nii.gz`, `study_2/image_type_1.nii.gz`, `study_2/image_type_2.nii.gz`)
    * `--mask_path` - similar to `--dataset_path` but for masks (displayed on top of the images; assumes volumes and masks are correctly registered)
    * `--mask_patterns` - similar to `--patterns` but for masks 
    * `--mask_type_pattern` - similar to `--image_type_pattern` but for masks 
    * `--metadata_path` - path to file containing metadata (`json` file containing one entry for each study UID)
    * `--metadata_keys` - keys for relevant fields in metadata dictionary