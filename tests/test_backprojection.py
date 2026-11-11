import unittest
import numpy as np
from test_sensors import preview


class BackprojectionTests(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[400., 0, 320], [0, 400, 240], [0, 0, 1]])

    def test_center_and_off_axis_depth_is_not_range(self):
        np.testing.assert_allclose(preview.pixel_to_3d(320, 240, 2, self.k), [0, 0, 2])
        point = preview.pixel_to_3d(520, 140, 2, self.k)
        np.testing.assert_allclose(point, [1, -.5, 2])
        self.assertGreater(np.linalg.norm(point), 2)

    def test_world_rotation_translation_and_reprojection(self):
        transform = np.array([[0, 0, 1, .175], [-1, 0, 0, 0], [0, -1, 0, .115], [0, 0, 0, 1]])
        point = preview.pixel_to_3d(520, 140, 2, self.k, frame='world', camera_to_world=transform)
        np.testing.assert_allclose(point, [2.175, -1, .615])
        camera_point = transform[:3, :3].T @ (point-transform[:3, 3])
        uv = self.k @ camera_point
        np.testing.assert_allclose(uv[:2]/uv[2], [520, 140])

    def test_estimated_depth_controls_result(self):
        first = preview.pixel_to_3d(360, 280, 1.2, self.k)
        second = preview.pixel_to_3d(360, 280, 1.8, self.k)
        np.testing.assert_allclose(second, first*1.5)

    def test_invalid_inputs(self):
        for z in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                preview.pixel_to_3d(320, 240, z, self.k)
        with self.assertRaises(ValueError):
            preview.pixel_to_3d(320, 240, 1, self.k, frame='world')
        with self.assertRaises(ValueError):
            preview.pixel_to_3d(320, 240, 1, np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            preview.pixel_to_3d(320, 240, 1, self.k, frame='other')


if __name__ == '__main__':
    unittest.main()
