from __future__ import annotations

import unittest

import numpy as np

from kepa.tools.segmentation.postprocessing import keep_center_nearest_components


class CenterNearestComponentTests(unittest.TestCase):
    def test_selects_center_nearest_eligible_component(self) -> None:
        plane = np.zeros((30, 40), dtype=np.int16)
        plane[2:10, 2:10] = 1       # area 64, far from center
        plane[12:18, 17:23] = 1     # area 36, centered, >= 50% of max
        plane[25:27, 34:36] = 1     # area 4, below candidate threshold

        result, report = keep_center_nearest_components(
            plane, target_label_ids=[1], minimum_area_fraction=0.5
        )

        self.assertEqual(int(np.count_nonzero(result == 1)), 36)
        self.assertEqual(report["labels"][0]["candidate_count"], 2)
        self.assertEqual(report["labels"][0]["selected_component_area_pixels"], 36)

    def test_handles_missing_label_and_rejects_invalid_fraction(self) -> None:
        result, report = keep_center_nearest_components(
            np.zeros((5, 7), dtype=np.int16), target_label_ids=[1]
        )
        self.assertFalse(result.any())
        self.assertEqual(report["labels"][0]["status"], "absent")
        with self.assertRaises(ValueError):
            keep_center_nearest_components(result, target_label_ids=[1], minimum_area_fraction=0.0)


if __name__ == "__main__":
    unittest.main()
