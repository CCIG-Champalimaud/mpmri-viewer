import json
import panel as pn
from .lib import VolumeDataset, ImageLoader

from typing import List

def format_metadata(title: str, metadata_str: str):
    style = "word-wrap: break-word; width: 90%;"
    return f"<div style='{style}'><b>{title}</b>: {metadata_str}</div>"

class AccordionReactive(pn.Accordion):
    def __init__(self, 
                 image_loader,
                 metadata_path: str=None, 
                 metadata_keys: List[str]=[],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_loader = image_loader
        self.metadata_path = metadata_path
        self.metadata_keys = metadata_keys

        if metadata_path is not None:
            with open(self.metadata_path) as o:
                self.metadata = json.load(o)

    def add_text(self, *args, **kwargs):
        study_uids = self.image_loader.retrieve_curr_study_uids()
        self.clear()
        to_add = []
        for i,study_uid in enumerate(study_uids):
            curr_metadata = [
                format_metadata("Study UID",study_uid)
            ]
            if hasattr(self,"metadata"):
                if study_uid in self.metadata:
                    for key in self.metadata_keys:
                        if key in self.metadata[study_uid]:
                            curr_metadata.append(
                                format_metadata(key,self.metadata[study_uid][key]))
            element = (str(i + 1),pn.pane.HTML("\n".join(curr_metadata)))
            to_add.append(element)
        self.extend(to_add)
        return self

def init_app(dataset_path: str,
             patterns: List[str],
             image_type_pattern: str,
             study_uid_pattern: str,
             mask_path: str=None,
             mask_patterns: List[str]=[],
             mask_type_pattern: str=None,
             metadata_path: str=None,
             metadata_keys: List[str]=[],
             max_slices: int=32,
             max_number_of_images: int=6):
    volume_dataset = VolumeDataset(
        dataset_path, 
        patterns=patterns,
        image_pattern=image_type_pattern,
        study_uid_pattern=study_uid_pattern)
    if mask_path is not None:
        mask_dataset = VolumeDataset(
            mask_path, 
            patterns=mask_patterns,
            image_pattern=mask_type_pattern,
            study_uid_pattern=study_uid_pattern)
    else:
        mask_dataset = None
    image_loader = ImageLoader([128,128],
                               volume_dataset,
                               mask_dataset,
                               "resize")

    template = pn.template.MaterialTemplate()
    image_type = pn.Column(
        pn.widgets.Select(name="Image type", 
                          options=volume_dataset.image_types),
        pn.widgets.Select(name="Mask type", 
                          options=[]))
    if mask_dataset is None:
        image_type[1].options = []
    else:
        image_type[1].options = mask_dataset.image_types
    slice_idx = pn.widgets.IntSlider(name="Slice index",start=0, end=max_slices)
    sqrt_n_images = pn.widgets.IntSlider(name="Number of images",start=1,
                                         end=max_number_of_images)
    mode = pn.widgets.Select(name="Display mode",options=["resize","crop"])
    page = pn.widgets.IntSlider(name="Volumes",start=0,end=len(volume_dataset),
                                step=pn.bind(lambda x: x**2,sqrt_n_images))
    autocomplete = pn.widgets.AutocompleteInput(
        name='Study UID', options=volume_dataset.all_study_uids,
        placeholder='Skips to this study UID')
    search_button = pn.widgets.Button(name="Search")

    def update_image_from_search(event):
        study_uid = autocomplete.value
        if study_uid in volume_dataset.retro_conversion:
            idx = volume_dataset.retro_conversion[study_uid]
            page.value = idx
            sqrt_n_images.value = 1
            template.notifications.success(
                f'Found study UID {study_uid}',
                duration=1000)
        else:
            template.notifications.error(
                f'Study UID {study_uid} not in dataset',
                duration=1000)

    search_button.on_click(update_image_from_search)

    image = pn.pane.Image(pn.bind(image_loader.retrieve_pil_image,
                                  sqrt_n_images,
                                  slice_idx=slice_idx,
                                  start_idx=page,
                                  image_key=image_type[0],
                                  mask_key=image_type[1],
                                  mode=mode),
                          width=640)
    reactive_col = AccordionReactive(image_loader=image_loader,
                                     metadata_path=metadata_path,
                                     metadata_keys=metadata_keys,
                                     width_policy="fixed",
                                     width=300)
    
    template.param.update(title="Volume dataset explorer")
    template.sidebar.append(
        pn.Column(image_type,
                  slice_idx,
                  sqrt_n_images, 
                  mode,
                  "<br>",
                  autocomplete,
                  search_button,
                  "<br>",
                  pn.Column(
                      pn.bind(reactive_col.add_text, sqrt_n_images, page),
                      height=200
                  )).servable(area="sidebar")
    )

    template.main.append(
        pn.Column(image, page).servable()
        )
