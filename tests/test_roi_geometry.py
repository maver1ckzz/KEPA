import numpy as np

from kepa.tools.roi_extraction.geometry import (
    composite_bbox,
    composite_bbox_with_provenance,
    keep_anterior_components,
    mask_bbox,
    point_bbox,
)


def test_mask_bbox_uses_half_open_edges_and_clips_to_image():
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:8, 4:10] = True
    assert mask_bbox(mask, image_shape=mask.shape) == (4, 3, 10, 8)
    assert mask_bbox(mask, image_shape=mask.shape, padding_px={"left": 9, "top": 1}) == (0, 2, 10, 8)


def test_reference_size_scales_fixed_pixel_padding_per_axis_and_defaults_to_native_scale():
    mask = np.zeros((256, 512), dtype=bool)
    mask[100:120, 200:220] = True
    padding = {"left": 48, "top": 48, "right": 48, "bottom": 48}
    assert mask_bbox(mask, image_shape=mask.shape, padding_px=padding) == (152, 52, 268, 168)
    assert mask_bbox(mask, image_shape=mask.shape, padding_px=padding,
                     reference_size=[512, 512]) == (152, 76, 268, 144)


def test_reference_size_scales_point_margins_but_does_not_change_fractional_position():
    image_shape = (256, 512)
    margins = {"left": 48, "top": 48, "right": 48, "bottom": 48}
    assert point_bbox((100, 100), image_shape=image_shape, margin_px=margins,
                      reference_size=[512, 512]) == (52, 76, 148, 124)
    mask = np.zeros(image_shape, dtype=bool)
    mask[80:120, 180:220] = True
    assert mask_bbox(mask, image_shape=image_shape, position={"left": 0.25, "right": 0.25}) == \
        mask_bbox(mask, image_shape=image_shape, position={"left": 0.25, "right": 0.25},
                  reference_size=[512, 512])

def test_composite_left_position_minus_four_thirds_reaches_past_mask_right():
    labels = np.zeros((12, 16), dtype=np.int16)
    labels[2:8, 2:6] = 1
    labels[2:8, 8:11] = 2
    bounds = {
        "left": {"label_ids": [1], "source_edge": "min", "position": -4 / 3},
        "top": {"label_ids": [1], "source_edge": "min"},
        "right": {"label_ids": [2], "source_edge": "max"},
        "bottom": {"label_ids": [2], "source_edge": "max"},
    }
    assert composite_bbox(labels, image_shape=labels.shape, bounds=bounds) == (7, 2, 11, 8)


def test_composite_right_uses_canal_only_within_reference_vertebra_span():
    labels = np.zeros((14, 18), dtype=np.int16)
    labels[4:10, 2:6] = 13
    labels[0:3, 12:15] = 2  # Outside C3's vertical extent: must not affect C3 right edge.
    labels[6:8, 8:10] = 2
    bounds = {
        "left": {"label_ids": [13], "source_edge": "min"},
        "top": {"label_ids": [13], "source_edge": "min"},
        "right": {"candidates": [
            {"label_ids": [2], "source_edge": "max", "position": 0,
             "scope": {"axis": "y", "reference_label_ids": [13]}},
            {"label_ids": [13], "source_edge": "max", "position": 1},
        ]},
        "bottom": {"label_ids": [13], "source_edge": "max"},
    }
    bbox, provenance = composite_bbox_with_provenance(
        labels, image_shape=labels.shape, bounds=bounds,
    )
    assert bbox == (2, 4, 10, 10)
    assert provenance["right"]["selected_candidate_index"] == 0
    assert provenance["right"]["attempts"][0]["scope_rows_inclusive"] == [4, 9]


def test_composite_right_falls_back_when_local_canal_is_absent():
    labels = np.zeros((14, 18), dtype=np.int16)
    labels[4:10, 2:6] = 13
    labels[0:3, 12:15] = 2  # A canal elsewhere in the image is not a local candidate.
    bounds = {
        "left": {"label_ids": [13], "source_edge": "min"},
        "top": {"label_ids": [13], "source_edge": "min"},
        "right": {"candidates": [
            {"label_ids": [2], "source_edge": "max", "position": 0,
             "scope": {"axis": "y", "reference_label_ids": [13]}},
            {"label_ids": [13], "source_edge": "max", "position": 1},
        ]},
        "bottom": {"label_ids": [13], "source_edge": "max"},
    }
    bbox, provenance = composite_bbox_with_provenance(
        labels, image_shape=labels.shape, bounds=bounds,
    )
    assert bbox == (2, 4, 10, 10)
    assert provenance["right"]["selected_candidate_index"] == 1
    assert provenance["right"]["attempts"][0]["status"] == "no_pixels_in_scope"


def test_point_bbox_applies_asymmetric_margins():
    assert point_bbox(
        (50, 60),
        image_shape=(100, 100),
        margin_px={"left": 10, "top": 5, "right": 8, "bottom": 12},
    ) == (40, 55, 58, 72)


def test_anterior_filter_removes_posterior_process_before_largest_component():
    labels = np.zeros((12, 16), dtype=np.int16)
    labels[2:10, 2:5] = 13  # anterior vertebral body
    labels[5, 4:11] = 13  # pedicle/process extending across the canal boundary
    labels[0, 0] = 13  # small detached anterior fragment, not 8-connected to the body
    labels[2:10, 7] = 1  # spinal-canal reference; edge at x=7
    filtered, info = keep_anterior_components(
        labels,
        target_label_ids=[13],
        reference_label_ids=[1],
    )
    assert np.all(filtered[:, 7:] != 13)
    assert np.count_nonzero(filtered == 13) == 26
    assert info["per_label"]["13"]["removed_pixels"] > 0
    assert np.count_nonzero(labels == 13) > np.count_nonzero(filtered == 13)
