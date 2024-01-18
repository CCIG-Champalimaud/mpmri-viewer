import json
import panel as pn
from .lib import VolumeDataset, ImageLoader

from typing import List, Dict, Any


class AccordionReactive(pn.Accordion):
    def __init__(
        self,
        image_loader: VolumeDataset,
        metadata: Dict[str, Any] = None,
        metadata_keys: List[str] = [],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_loader = image_loader
        self.metadata = metadata
        self.metadata_keys = metadata_keys

    def format_metadata(self, title: str, metadata_str: str):
        style = "word-wrap: break-word; width: 90%;"
        return f"<div style='{style}'><b>{title}</b>: {metadata_str}</div>"

    def add_text(self, *args, **kwargs):
        study_uids = self.image_loader.retrieve_curr_study_uids()
        self.clear()
        to_add = []
        for i, study_uid in enumerate(study_uids):
            curr_metadata = [self.format_metadata("Study UID", study_uid)]
            if hasattr(self, "metadata"):
                if study_uid in self.metadata:
                    for key in self.metadata_keys:
                        if key in self.metadata[study_uid]:
                            curr_metadata.append(
                                self.format_metadata(
                                    key, self.metadata[study_uid][key]
                                )
                            )
            element = (str(i + 1), pn.pane.HTML("\n".join(curr_metadata)))
            to_add.append(element)
        self.extend(to_add)
        return self


def init_app(
    dataset_path: str,
    patterns: List[str],
    image_type_pattern: str,
    study_uid_pattern: str,
    mask_path: str = None,
    mask_patterns: List[str] = [],
    mask_type_pattern: str = None,
    metadata_path: str = None,
    metadata_keys: List[str] = [],
    max_slices: int = 32,
    max_number_of_images: int = 6,
):
    volume_dataset = VolumeDataset(
        dataset_path,
        patterns=patterns,
        image_pattern=image_type_pattern,
        study_uid_pattern=study_uid_pattern,
    )

    if metadata_path is not None:
        with open(metadata_path) as o:
            metadata = json.load(o)
    else:
        metadata = {}

    if mask_path is not None:
        mask_dataset = VolumeDataset(
            mask_path,
            patterns=mask_patterns,
            image_pattern=mask_type_pattern,
            study_uid_pattern=study_uid_pattern,
        )
        for key in mask_dataset.volume_dictionary_full:
            if key not in metadata:
                metadata[key] = {}
            for mask_key in mask_dataset.volume_dictionary_full[key]:
                metadata[key][mask_key] = True
        metadata_keys.extend(mask_dataset.image_types)
    else:
        mask_dataset = None
    image_loader = ImageLoader(
        [128, 128], volume_dataset, mask_dataset, "resize"
    )

    template = pn.template.MaterialTemplate()
    image_type = pn.Column(
        pn.widgets.Select(
            name="Image type", options=volume_dataset.image_types
        ),
        pn.widgets.Select(name="Mask type", options=[]),
    )
    if mask_dataset is None:
        image_type[1].options = [None]
    else:
        image_type[1].options = [None] + mask_dataset.image_types
    slice_idx = pn.widgets.IntSlider(
        name="Slice index", start=0, end=max_slices
    )
    sqrt_n_images = pn.widgets.IntSlider(
        name="Number of images", start=1, end=max_number_of_images
    )
    mode = pn.widgets.Select(name="Display mode", options=["resize", "crop"])
    numbers = pn.widgets.Checkbox(name="Numbers", value=True)
    page = pn.widgets.IntSlider(
        name="Volumes",
        start=0,
        end=len(volume_dataset) - sqrt_n_images.value**2,
        step=pn.bind(lambda x: x**2, sqrt_n_images),
    )
    autocomplete = pn.widgets.AutocompleteInput(
        name="Study UID",
        options=volume_dataset.all_study_uids,
        placeholder="Skips to this study UID",
        width=250,
    )
    search_button = pn.widgets.Button(name="Search")

    filter_instructions = pn.widgets.TextInput(
        name="Filter by metadata key",
        options=metadata_keys,
        placeholder="key==value or key!=value",
        width=250,
    )
    filter_helper_text = pn.pane.Markdown(
        """
    ### Instructions

    Base format: key\<symbol\>value (no spaces between either)

    * For exact string equivalences:
        * Use === for exact string equivalences
        * Use !== for exact string differences
    * For float equivalences:
        * Use == for float equivalences
        * Use != for float differences
        * Use >, <, >=, <= for general float inequalities
    """
    )
    filter_button = pn.widgets.Button(name="Filter")

    def update_image_from_search(event):
        study_uid = autocomplete.value
        if study_uid in image_loader.dataset.retro_conversion:
            idx = image_loader.dataset.retro_conversion[study_uid]
            page.value = idx
            sqrt_n_images.value = 1
            template.notifications.success(
                f"Found study UID {study_uid}", duration=2000
            )
        else:
            template.notifications.error(
                f"Study UID {study_uid} not in dataset", duration=2000
            )

    def update_from_filter(event):
        filters = filter_instructions.value
        previous_autocomplete_options = autocomplete.options
        previous_image_loader_idxs = image_loader.image_idxs
        previous_image_curr_study_uids = image_loader.curr_study_uids
        previous_page_end = page.end
        previous_page_value = page.value
        try:
            image_loader.dataset.filter_volume_dataset(filters, metadata)
            autocomplete.options = image_loader.dataset.all_study_uids
            image_loader.image_idxs = []
            image_loader.curr_study_uids = {}
            page.end = len(image_loader.dataset) - sqrt_n_images.value**2
            page.value = previous_page_value + 1
            page.value = 0
            template.notifications.success(
                f"Found {len(image_loader.dataset)} studies", duration=2000
            )
        except:
            # reset status, raise error
            autocomplete.options = previous_autocomplete_options
            image_loader.image_idxs = previous_image_loader_idxs
            image_loader.curr_study_uids = previous_image_curr_study_uids
            page.end = previous_page_end
            page.value = previous_page_value
            template.notifications.error(
                f"Filtering failed, please check filters are correct",
                duration=2000,
            )

    search_button.on_click(update_image_from_search)
    filter_button.on_click(update_from_filter)

    image = pn.pane.Image(
        pn.bind(
            image_loader.retrieve_pil_image,
            sqrt_n_images.param.value_throttled,
            slice_idx=slice_idx.param.value_throttled,
            start_idx=page.param.value_throttled,
            image_key=image_type[0],
            mask_key=image_type[1],
            mode=mode,
            numbers=numbers,
        ),
        width=640,
    )
    reactive_col = AccordionReactive(
        image_loader=image_loader,
        metadata=metadata,
        metadata_keys=metadata_keys,
        width_policy="fixed",
        width=300,
    )

    template.param.update(title="Volume dataset explorer")
    template.sidebar.append(
        pn.Column(
            image_type,
            slice_idx,
            sqrt_n_images,
            mode,
            numbers,
            pn.Accordion(
                (
                    "Filter",
                    pn.Column(
                        filter_instructions, filter_button, filter_helper_text
                    ),
                ),
                ("Search", pn.Column(autocomplete, search_button)),
                width=300,
                width_policy="fixed",
            ),
            pn.Column(
                pn.bind(
                    reactive_col.add_text,
                    sqrt_n_images.param.value_throttled,
                    page.param.value_throttled,
                ),
                height=200,
            ),
        ).servable(area="sidebar")
    )

    return template.main.append(pn.Column(image, page).servable())
