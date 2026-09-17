import unittest

from basic_robotics.general import tm
from basic_robotics.kinematics.visual_info import vis_info


class test_kinematics_visual_info(unittest.TestCase):

    def test_visual_info_defaults(self):
        info = vis_info()
        self.assertIsNone(info.geo_type)
        self.assertIsInstance(info.origin, tm)
        self.assertEqual(info.scale, 1.0)
        self.assertIsNone(info.file_name)
        self.assertIsNone(info.radius)
        self.assertIsNone(info.length)
        self.assertIsNone(info.box_size)

    def test_visual_info_setOrigin(self):
        info = vis_info()
        new_origin = tm([1, 2, 3, 0, 0, 0])
        info.setOrigin(new_origin)
        self.assertIs(info.origin, new_origin)
        # Passing None is a no-op, not a reset.
        info.setOrigin(None)
        self.assertIs(info.origin, new_origin)

    def test_visual_info_setScale(self):
        info = vis_info()
        info.setScale(2.5)
        self.assertEqual(info.scale, 2.5)
        info.setScale(None)
        self.assertEqual(info.scale, 2.5)

    def test_visual_info_str_mesh(self):
        info = vis_info()
        info.geo_type = 'mesh'
        info.file_name = 'part.stl'
        self.assertEqual(str(info), 'Mesh: part.stl')

    def test_visual_info_str_cylinder(self):
        info = vis_info()
        info.geo_type = 'cyl'
        info.radius = 0.5
        info.length = 2.0
        self.assertEqual(str(info), 'Cylinder: 0.5 2.0')

    def test_visual_info_str_sphere(self):
        info = vis_info()
        info.geo_type = 'spr'
        info.radius = 1.0
        info.length = None
        self.assertEqual(str(info), 'Sphere: 1.0 None')

    def test_visual_info_str_box(self):
        info = vis_info()
        info.geo_type = 'box'
        info.box_size = [1, 2, 3]
        self.assertEqual(str(info), 'Box: [1, 2, 3]')

    def test_visual_info_str_undefined(self):
        info = vis_info()
        self.assertEqual(str(info), 'Undefined')


if __name__ == '__main__':
    unittest.main()
