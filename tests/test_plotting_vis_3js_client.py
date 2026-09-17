import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from basic_robotics.general import tm, fsr
from basic_robotics.kinematics import Arm, makeSP
from basic_robotics.kinematics.visual_info import vis_info
from basic_robotics.plotting.vis_3js_client import (
    newFloor, newMaterial, newPrimitive, determineAxis,
    DrawClient, VisPlot, PrimitivePlot, CubePlot, AxesPlot, TubePlot,
    RobotPlot, ArmPlot, SPPlot, package_directory,
)


def make_mocked_client():
    client = DrawClient()
    client.ses = MagicMock()
    client.ses.put.return_value = MagicMock(status_code=200)
    client.ses.post.return_value = MagicMock(status_code=200)
    client.ses.get.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})
    return client


def make_test_arm(num_dof=6):
    """Small serial arm fixture (same shape as other test modules' fixture)."""
    Base_T = tm()
    L1, L2, L3, W = 4.5, 3.75, 3.75, 0.1
    basic_arm_end_effector_home = fsr.TAAtoTM(np.array([[L2+L3+W+W+W], [0], [L1], [0], [0], [0]]))
    basic_arm_joint_axes = np.array(
            [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]).conj().T
    basic_arm_joint_homes = np.array(
            [[0, 0, 0], [0, 0, L1], [L2, 0, L1], [L2+L3, 0, L1],
             [L2+L3+W, 0, L1], [L2+L3+2*W, 0, L1]]).conj().T
    basic_arm_screw_list = np.zeros((6, 6))
    for i in range(6):
        basic_arm_screw_list[0:6, i] = np.hstack((
                basic_arm_joint_axes[0:3, i],
                np.cross(basic_arm_joint_homes[0:3, i], basic_arm_joint_axes[0:3, i])))
    box_dims = np.array(
            [[W, W, L1], [L2, W, W], [L3, W, W], [W, W, W], [W, W, W], [W, W, W]]).conj().T

    arm = Arm(Base_T, basic_arm_screw_list, basic_arm_end_effector_home,
            basic_arm_joint_homes, basic_arm_joint_axes)
    arm.setVisColProperties(link_dimensions=box_dims)
    return arm


class test_plotting_vis_3js_client(unittest.TestCase):

    # -- module level helpers --

    def test_client_newFloor_default_resolves_package_path(self):
        floor = newFloor()
        self.assertEqual(floor['Key'], 'Floor')
        self.assertEqual(floor['File'], os.path.join(package_directory, 'ChFloor2.glb'))
        self.assertEqual(floor['Category'], 'Model')

    def test_client_newFloor_custom_filename_unchanged(self):
        floor = newFloor('custom.glb')
        self.assertEqual(floor['File'], 'custom.glb')

    def test_client_newMaterial_defaults(self):
        mat = newMaterial()
        self.assertEqual(mat['Color'], 0x3238a8)
        self.assertTrue(mat['transparent'])
        self.assertEqual(mat['opacity'], .5)

    def test_client_newPrimitive_defaults_and_material(self):
        prim = newPrimitive()
        self.assertEqual(prim['Key'], 'CubeLet')
        self.assertEqual(prim['File'], 'Cube3.glb')
        self.assertEqual(prim['Category'], 'Model')
        self.assertEqual(prim['Scale'], [1.0, 1.0, 1.0])
        self.assertIn('Color', prim['Material'])

    def test_client_newPrimitive_custom_color_not_overwritten(self):
        color = newMaterial(color=0x00ff00)
        prim = newPrimitive(color=color)
        self.assertIs(prim['Material'], color)

    def test_client_determineAxis_identity(self):
        loc = tm([0, 0, 0, 0, 0, 0])
        axis = np.array([0, 0, 1])
        result = determineAxis(loc, axis)
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-9)

    # -- DrawClient: naming / host management --

    def test_client_newName_increments_registry(self):
        client = DrawClient()
        n1 = client.newName()
        n2 = client.newName()
        self.assertEqual(client.unnamed_registry, [n1, n2])
        self.assertNotEqual(n1, n2)

    def test_client_deleteName_not_registered_returns_false(self):
        client = DrawClient()
        self.assertFalse(client.deleteName('NeverRegistered'))

    def test_client_deleteName_registered_calls_delete(self):
        client = make_mocked_client()
        name = client.newName()
        result = client.deleteName(name)
        self.assertTrue(result)
        self.assertNotIn(name, client.unnamed_registry)
        client.ses.put.assert_called_once()

    def test_client_deleteAllUnnamed_resets_registry(self):
        # Regression test: deleteAllUnnamed used to call the misspelled
        # self.delte(...), which raised AttributeError.
        client = make_mocked_client()
        client.newName()
        client.newName()
        result = client.deleteAllUnnamed()
        self.assertTrue(result)
        self.assertEqual(client.unnamed_registry, [])
        self.assertEqual(client.unnamed_counter, 0)

    def test_client_setPort_updates_url(self):
        client = DrawClient()
        client.setPort(6000)
        self.assertEqual(client.port, 6000)
        self.assertIn(':6000/', client.url)

    def test_client_setHost_updates_url(self):
        client = DrawClient()
        client.setHost('192.168.1.5')
        self.assertEqual(client.host, '192.168.1.5')
        self.assertIn('192.168.1.5', client.url)

    def test_client_detHost_default_and_override(self):
        client = DrawClient()
        self.assertEqual(client.detHost(), client.url)
        self.assertEqual(client.detHost('http://alt:1234/api/json'), 'http://alt:1234/api/json')

    def test_client_tryDictMatch_present_and_absent(self):
        client = DrawClient()
        self.assertEqual(client.tryDictMatch({'A': 1}, 'A', 0), 1)
        self.assertEqual(client.tryDictMatch({'A': 1}, 'B', 'default'), 'default')

    # -- DrawClient: sending data --

    def test_client_makeFloor_resolves_default_path(self):
        client = make_mocked_client()
        client.sendFile = MagicMock()
        client.makeFloor()
        client.sendFile.assert_called_once_with(
            os.path.join(package_directory, 'jsons/SceneFloor.json'))

    def test_client_prepAggregated_skips_none(self):
        client = DrawClient()
        agg = client.prepAggregated([{"Key": "a", "V": 1}, None, {"Key": "b", "V": 2}])
        self.assertEqual(agg, {"Keys": {"a": {"Key": "a", "V": 1}, "b": {"Key": "b", "V": 2}}})

    def test_client_send_post_success(self):
        client = make_mocked_client()
        result = client.send({"Key": "foo"}, "POST")
        self.assertTrue(result)
        client.ses.post.assert_called_once()

    def test_client_send_put_failure(self):
        client = make_mocked_client()
        client.ses.put.return_value = MagicMock(status_code=400)
        result = client.send({"Key": "foo"}, "PUT")
        self.assertFalse(result)

    def test_client_send_addtime_single_key(self):
        client = make_mocked_client()
        dat = {"Key": "foo"}
        client.send(dat, "PUT", addtime=True)
        self.assertIn("UnixTime", dat)

    def test_client_send_addtime_multi_keys(self):
        client = make_mocked_client()
        dat = {"Keys": {"a": {"V": 1}, "b": {"V": 2}}}
        client.send(dat, "PUT", addtime=True)
        self.assertIn("UnixTime", dat["Keys"]["a"])
        self.assertIn("UnixTime", dat["Keys"]["b"])

    def test_client_sendAggregated_uses_prep_and_put(self):
        client = make_mocked_client()
        result = client.sendAggregated([{"Key": "a", "V": 1}, None])
        self.assertTrue(result)
        sent_json = client.ses.put.call_args.kwargs['json']
        self.assertEqual(list(sent_json['Keys'].keys()), ['a'])

    def test_client_sendFile_single_key(self):
        client = make_mocked_client()
        client.ses.put.return_value = MagicMock(status_code=200, json='response-json')
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump({"Key": "foo", "Value": 1}, f)
            fname = f.name
        try:
            res = client.sendFile(fname)
        finally:
            os.remove(fname)
        self.assertEqual(res, 'response-json')
        sent_json = client.ses.put.call_args.kwargs['json']
        self.assertEqual(sent_json['Key'], 'foo')
        self.assertIn('UnixTime', sent_json)

    def test_client_sendFile_key_name_override(self):
        client = make_mocked_client()
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump({"Key": "foo", "Value": 1}, f)
            fname = f.name
        try:
            client.sendFile(fname, key_name='renamed')
        finally:
            os.remove(fname)
        sent_json = client.ses.put.call_args.kwargs['json']
        self.assertEqual(sent_json['Key'], 'renamed')

    def test_client_sendFile_no_key_wraps_in_keys(self):
        client = make_mocked_client()
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump({"a": {"V": 1}, "b": {"V": 2}}, f)
            fname = f.name
        try:
            client.sendFile(fname)
        finally:
            os.remove(fname)
        sent_json = client.ses.put.call_args.kwargs['json']
        self.assertIn('Keys', sent_json)
        self.assertIn('UnixTime', sent_json['Keys']['a'])

    def test_client_prepTM_adds_matrix(self):
        client = DrawClient()
        params = client.prepTM(tm([1, 2, 3, 0, 0, 0]), {"Key": "foo"})
        self.assertIn('Matrix', params)
        self.assertEqual(len(params['Matrix']), 16)

    def test_client_sendTM_calls_send_with_matrix(self):
        client = make_mocked_client()
        result = client.sendTM(tm(), {"Key": "foo"})
        self.assertTrue(result)
        sent_json = client.ses.put.call_args.kwargs['json']
        self.assertIn('Matrix', sent_json)

    def test_client_sendAxes_builds_frame_params(self):
        client = make_mocked_client()
        result = client.sendAxes(tm(), name='MyFrame', scale=2.0)
        self.assertTrue(result)
        sent_json = client.ses.put.call_args.kwargs['json']
        self.assertEqual(sent_json['MyFrame']['Scale'], 2.0)
        self.assertEqual(sent_json['MyFrame']['Frame'], 1)

    def test_client_sendLine_arrow_type(self):
        client = make_mocked_client()
        key = client.sendLine(
            [tm([0, 0, 0, 0, 0, 0]), tm([1, 0, 0, 0, 0, 0])], key='MyLine')
        self.assertEqual(key, 'MyLine')
        sent_json = client.ses.put.call_args.kwargs['json']
        self.assertEqual(sent_json['LineParameters']['Arrow'], 1)

    def test_client_sendLine_plain_type_uses_segments(self):
        client = make_mocked_client()
        key = client.sendLine(
            [tm([0, 0, 0, 0, 0, 0]), tm([1, 0, 0, 0, 0, 0]), tm([1, 1, 0, 0, 0, 0])],
            key='PolyLine', type='Line')
        sent_json = client.ses.put.call_args.kwargs['json']
        self.assertEqual(len(sent_json['Segments']), 3)
        self.assertEqual(key, 'PolyLine')

    def test_client_sendLine_auto_name(self):
        client = make_mocked_client()
        key = client.sendLine([tm(), tm([1, 0, 0, 0, 0, 0])])
        self.assertIn(key, client.unnamed_registry)

    # -- DrawClient.get() query construction --

    def test_client_get_default_is_latest(self):
        client = make_mocked_client()
        data, ok = client.get()
        self.assertTrue(ok)
        self.assertEqual(data, {"ok": True})
        called_url = client.ses.get.call_args.kwargs['url']
        self.assertIn('Latest=1', called_url)

    def test_client_get_complete_with_category(self):
        client = make_mocked_client()
        client.get({"Kind": "Complete", "Category": "Model"})
        called_url = client.ses.get.call_args.kwargs['url']
        self.assertIn('Category=Model', called_url)
        self.assertIn('Complete=1', called_url)

    def test_client_get_keys_without_index(self):
        client = make_mocked_client()
        client.get({"Keys": ["a", "b"]})
        called_url = client.ses.get.call_args.kwargs['url']
        self.assertIn('Key=a,b', called_url)

    def test_client_get_keys_with_index(self):
        client = make_mocked_client()
        client.get({"Keys": ["a"], "Index": 2})
        called_url = client.ses.get.call_args.kwargs['url']
        self.assertIn('Key=a', called_url)
        self.assertIn('Index=2', called_url)

    def test_client_get_keys_with_getrange(self):
        client = make_mocked_client()
        client.get({"Keys": ["a"], "GetRange": "X", "ValueRange": [1, 2, 3]})
        called_url = client.ses.get.call_args.kwargs['url']
        self.assertIn('GetRange=X', called_url)
        self.assertIn('ValueRange=1,2,3', called_url)

    def test_client_get_failure_status(self):
        client = make_mocked_client()
        client.ses.get.return_value = MagicMock(status_code=404, json=lambda: {})
        data, ok = client.get()
        self.assertFalse(ok)
        self.assertEqual(data, {})

    # -- DrawClient.delete() --

    def test_client_delete_all_when_no_params(self):
        client = make_mocked_client()
        result = client.delete()
        self.assertTrue(result)
        sent = json.loads(client.ses.put.call_args[0][1])
        self.assertEqual(sent, {"DeleteAll": 1})

    def test_client_delete_single_key(self):
        client = make_mocked_client()
        client.delete({"Keys": ["a"]})
        sent = json.loads(client.ses.put.call_args[0][1])
        self.assertEqual(sent, {"DeleteKey": "a"})

    def test_client_delete_multiple_keys(self):
        client = make_mocked_client()
        client.delete({"Keys": ["a", "b"]})
        sent = json.loads(client.ses.put.call_args[0][1])
        self.assertEqual(sent, {"DeleteKeys": ["a", "b"]})

    def test_client_delete_category(self):
        client = make_mocked_client()
        client.delete({"Category": "Model"})
        sent = json.loads(client.ses.put.call_args[0][1])
        self.assertEqual(sent, {"DeleteCategory": "Model"})

    def test_client_delete_failure_status(self):
        client = make_mocked_client()
        client.ses.put.return_value = MagicMock(status_code=500)
        self.assertFalse(client.delete())


class test_plotting_vis_3js_client_plots(unittest.TestCase):

    # -- VisPlot base behavior --

    def test_visplot_default_client_and_autoname(self):
        plot = VisPlot(None)
        self.assertIsInstance(plot.c, DrawClient)
        self.assertTrue(plot.name.startswith('Object_'))

    def test_visplot_delete_calls_client_delete_twice(self):
        client = make_mocked_client()
        plot = VisPlot('MyPlot', client)
        plot.keys = ['k1', 'k2']
        plot.delete()
        self.assertEqual(client.ses.put.call_count, 2)

    def test_visplot_initialize_and_update_are_noops(self):
        plot = VisPlot('MyPlot', make_mocked_client())
        self.assertIsNone(plot.initialize())
        self.assertIsNone(plot.update())

    # -- CubePlot --

    def test_cubeplot_initializes_and_sends_on_creation(self):
        client = make_mocked_client()
        cube = CubePlot(tm(), [1, 2, 3], client=client, name='MyCube')
        self.assertIn('MyCube', cube.keys)
        self.assertEqual(cube.object['Key'], 'MyCube')
        self.assertEqual(cube.object['File'], 'internal/models/Cube3.glb')
        self.assertEqual(cube.object['Scale'], [1, 2, 3])
        client.ses.put.assert_called_once()

    def test_cubeplot_setTM_send_false_returns_object(self):
        # CubePlot.initialize() unconditionally calls self.update(True), so
        # setTM still triggers exactly one send even with send=False.
        client = make_mocked_client()
        cube = CubePlot(tm(), [1, 1, 1], client=client, name='C2')
        client.ses.put.reset_mock()
        result = cube.setTM(tm([1, 0, 0, 0, 0, 0]), send=False)
        self.assertEqual(result, [cube.object])
        client.ses.put.assert_called_once()

    def test_cubeplot_setTM_send_true_calls_send(self):
        # send=True triggers a second, separate send on top of the one
        # unconditionally issued by initialize().
        client = make_mocked_client()
        cube = CubePlot(tm(), [1, 1, 1], client=client, name='C3')
        client.ses.put.reset_mock()
        cube.setTM(tm([1, 0, 0, 0, 0, 0]), send=True)
        self.assertEqual(client.ses.put.call_count, 2)

    # -- AxesPlot --

    def test_axesplot_initializes_frame_object(self):
        client = make_mocked_client()
        axes = AxesPlot(tm(), client=client, scale=2.5, name='Ax1')
        self.assertEqual(axes.dimensions, 2.5)
        self.assertIn('Ax1', axes.object)
        self.assertEqual(axes.object['Ax1']['Scale'], 2.5)
        self.assertEqual(axes.object['Ax1']['Frame'], 1)

    # -- TubePlot --

    def test_tubeplot_initializes_and_sends_on_creation(self):
        client = make_mocked_client()
        tube = TubePlot(tm(), 2.0, 0.5, client, name='Tube1')
        self.assertIn('Tube1', tube.keys)
        self.assertEqual(tube.object['Key'], 'Tube1')
        self.assertEqual(tube.object['Primitive'], 'Cylinder')
        self.assertEqual(tube.object['Scale'], [0.5, 0.5, 2.0])
        client.ses.put.assert_called_once()

    # -- RobotPlot --

    def test_robotplot_stores_bot_reference(self):
        client = make_mocked_client()
        sentinel_bot = object()
        rp = RobotPlot('Bot1', sentinel_bot, client)
        self.assertIs(rp.bot, sentinel_bot)
        self.assertEqual(rp.name, 'Bot1')

    # -- ArmPlot (cylinder / box-dimension path) --

    def test_armplot_cyl_type_initialize_and_update(self):
        client = make_mocked_client()
        arm = make_test_arm()
        plot = ArmPlot('TestArm', arm, client)

        self.assertTrue(plot.cyl_type)
        self.assertEqual(len(plot.bot_data), 12)  # 6 links + 6 joints
        self.assertEqual(plot.link_end_ind, 6)

        tms = plot.update(send=False)
        self.assertEqual(len(tms), 12)
        # Base link transform is always populated.
        self.assertIsNotNone(tms[0])

        plot.update(send=True)
        client.ses.put.assert_called()

    def test_armplot_cyl_type_diagonal_joint_axis_branch(self):
        # A joint axis that isn't aligned with X/Y/Z falls through to the
        # determineAxis(...) branch in ArmPlot.update().
        client = make_mocked_client()
        arm = make_test_arm()
        arm.joint_axes = arm.joint_axes.copy()
        arm.joint_axes[:, 0] = [0.5, 0.5, np.sqrt(0.5)]
        plot = ArmPlot('DiagArm', arm, client)

        tms = plot.update(send=False)
        self.assertIsNotNone(tms[plot.link_end_ind])

    def test_armplot_mesh_vis_props_path(self):
        client = make_mocked_client()
        arm = make_test_arm()
        vis_props = []
        for name in arm.link_names:
            info = vis_info()
            info.geo_type = 'mesh'
            info.file_name = name + '.glb'
            vis_props.append(info)
        arm.setVisColProperties(vis_props=vis_props)

        plot = ArmPlot('MeshArm', arm, client)
        self.assertFalse(plot.cyl_type)
        self.assertEqual(len(plot.bot_data), 2 * len(vis_props))

        tms = plot.update(send=False)
        self.assertIsNotNone(tms[0])

    # -- SPPlot --

    def test_spplot_initializes_and_updates(self):
        client = make_mocked_client()
        sp, _bottom, _top = makeSP(0.075, 0.045, 6, tm(), .25, 1, 0)

        plot = SPPlot('TestSP', sp, client)
        self.assertEqual(plot.links[0]['Key'], 'TestSPB')
        self.assertEqual(plot.links[1]['Key'], 'TestSPT')
        self.assertEqual(len(plot.legs), 6)

        out = plot.update(send=False)
        self.assertEqual(len(out), 14)
        self.assertIsNotNone(out[0])

        plot.update(send=True)
        client.ses.put.assert_called()


if __name__ == '__main__':
    unittest.main()
