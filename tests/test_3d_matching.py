import unittest

import numpy as np

from kepa.tools.matching import ImageSeriesGeometry, match_sag2tra, match_tra2sag


def geometry(view, shape, affine, slice_axis):
    return ImageSeriesGeometry(
        view=view,
        shape=shape,
        voxel_to_patient=tuple(tuple(float(v) for v in row) for row in affine),
        slice_axis=slice_axis,
    )


class Matching3DTests(unittest.TestCase):
    def test_sag2tra_finds_nearest_axial_plane(self):
        # Axial planes are z=0, 2, 4, 6, 8 mm. The source level centroid is z=5.
        source_affine = np.diag([1.0, 1.0, 2.0, 1.0])
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2:4] = True
        axial_affine = np.diag([1.0, 1.0, 2.0, 1.0])
        target = geometry("axial", (8, 8, 5), axial_affine, slice_axis=2)

        result = match_sag2tra(mask, source_affine, target, level="C4-5")

        self.assertEqual(result.direction, "sag2tra")
        self.assertEqual(result.candidates[0].slice_index, 2)
        self.assertAlmostEqual(result.candidates[0].distance_mm, 1.0)
        self.assertEqual(result.selected, result.candidates[0])

    def test_tra2sag_uses_oblique_plane_normal_not_array_axis_guess(self):
        axial_affine = np.eye(4)
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        # Target sagittal slice planes are oblique: moving the slice index by
        # one changes patient x by 2 mm and y by 0.5 mm.
        sagittal_affine = np.array(
            [[0.0, 0.0, 2.0, 0.0], [1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0], [0, 0, 0, 1]],
            dtype=float,
        )
        target = geometry("sagittal", (5, 5, 5), sagittal_affine, slice_axis=2)

        result = match_tra2sag(mask, axial_affine, target, level="C5-6")

        self.assertEqual(result.candidates[0].slice_index, 1)
        self.assertLess(result.candidates[0].distance_mm, result.candidates[-1].distance_mm)

    def test_quality_gate_can_reject_all_candidates(self):
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True
        target = geometry("axial", (3, 3, 2), np.diag([1, 1, 5, 1]), slice_axis=2)

        result = match_sag2tra(mask, np.eye(4), target, level="C3", max_distance_mm=0.1)

        self.assertIsNone(result.selected)
        self.assertGreater(result.candidates[0].distance_mm, 0.1)

    def test_empty_mask_and_wrong_view_are_rejected(self):
        mask = np.zeros((2, 2, 2), dtype=bool)
        axial = geometry("axial", (2, 2, 2), np.eye(4), slice_axis=2)
        with self.assertRaisesRegex(ValueError, "empty source mask"):
            match_sag2tra(mask, np.eye(4), axial, level="C3")
        with self.assertRaisesRegex(ValueError, "axial target"):
            match_sag2tra(np.ones((2, 2, 2)), np.eye(4), geometry("sagittal", (2, 2, 2), np.eye(4), 0), level="C3")

    def test_coordinate_convention_must_match(self):
        mask = np.ones((2, 2, 2), dtype=bool)
        target = ImageSeriesGeometry(
            view="axial",
            shape=(2, 2, 2),
            voxel_to_patient=tuple(tuple(float(v) for v in row) for row in np.eye(4)),
            slice_axis=2,
            coordinate_system="DICOM_RAS",
        )
        with self.assertRaisesRegex(ValueError, "coordinate systems differ"):
            match_sag2tra(mask, np.eye(4), target, level="C3")


if __name__ == "__main__":
    unittest.main()
