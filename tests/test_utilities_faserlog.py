import json
import os
import tempfile
import unittest

import numpy as np

from basic_robotics.utilities.FaserLog import FaserLog, JSONLog


class test_utilities_faserlog(unittest.TestCase):

    # NOTE: logging.basicConfig() (used internally by FaserLog.__init__) only
    # takes effect on its *first* call in a process - later calls are no-ops.
    # So only a single FaserLog is created across this whole test module,
    # and every assertion below targets that one instance/log file.

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.log = FaserLog(name="TestLog", dir=cls.tmpdir)

    def test_utilities_FaserLog_creates_log_file(self):
        self.assertTrue(os.path.exists(self.log.dirname))
        self.assertTrue(self.log.dirname.startswith(self.tmpdir))

    def test_utilities_FaserLog_writeToLog(self):
        self.log.writeToLog("hello world")
        with open(self.log.dirname) as f:
            contents = f.read()
        self.assertIn("hello world", contents)

    def test_utilities_FaserLog_default_dir_uses_bare_name(self):
        # With dir left as "Default", dirname is just the timestamped name,
        # not prefixed with a directory. logging.basicConfig() is already
        # configured from setUpClass's instance by this point, so this
        # doesn't actually redirect logging or touch the filesystem.
        default_log = FaserLog(name="DefaultDirLog")
        self.assertEqual(default_log.dirname, default_log.name)
        self.assertNotIn("/", default_log.dirname)

    def test_utilities_FaserLog_writeMatrixToLog(self):
        self.log.writeMatrixToLog(np.eye(2), "Identity")
        with open(self.log.dirname) as f:
            contents = f.read()
        self.assertIn("Identity", contents)


class test_utilities_jsonlog(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "log.json")

    def test_utilities_JSONLog_openLog_creates_file_when_missing(self):
        log = JSONLog(self.path)
        self.assertFalse(os.path.exists(self.path))

        log.openLog()

        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(log.log, {'num_items': 0})

    def test_utilities_JSONLog_writeToLog_without_persisting(self):
        log = JSONLog(self.path)
        log.writeToLog({"value": 1})

        self.assertEqual(log.enum, 1)
        self.assertEqual(log.log['0']['data'], {"value": 1})
        self.assertIn('timestamp', log.log['0'])
        # openClose defaults to False, so nothing is written to disk yet.
        self.assertFalse(os.path.exists(self.path))

    def test_utilities_JSONLog_writeToLog_with_openClose_persists(self):
        log = JSONLog(self.path)
        log.writeToLog({"value": 42}, openClose=True)

        self.assertTrue(os.path.exists(self.path))
        with open(self.path) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk['num_items'], 1)
        self.assertEqual(on_disk['0']['data'], {"value": 42})

    def test_utilities_JSONLog_reload_after_save(self):
        writer = JSONLog(self.path)
        writer.writeToLog({"value": "a"}, openClose=True)
        writer.writeToLog({"value": "b"}, openClose=True)

        reader = JSONLog(self.path)
        reader.openLog()

        self.assertEqual(reader.log['num_items'], 2)
        self.assertEqual(reader.log['0']['data'], {"value": "a"})
        self.assertEqual(reader.log['1']['data'], {"value": "b"})


if __name__ == '__main__':
    unittest.main()
