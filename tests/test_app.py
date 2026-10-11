"""Smoke test: the Streamlit dashboard runs a full offline mission without errors."""
import os
import tempfile
import unittest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - streamlit missing
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit not installed")
class AppSmokeTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.env = {k: os.environ.get(k) for k in ("PROOFPILOT_HOME", "PROOFPILOT_LLM")}
        os.environ["PROOFPILOT_HOME"] = self.home.name
        os.environ["PROOFPILOT_LLM"] = "off"

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.home.cleanup()

    def test_offline_mission_renders_report(self):
        at = AppTest.from_file("../app.py", default_timeout=60)
        at.run()
        self.assertFalse(at.exception)
        at.toggle[0].set_value(True)  # offline demo mode
        at.text_area[0].input("Compare open-source agent frameworks for a student project")
        at.button(key="run").click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertTrue(any("Mission report" in s.value for s in at.subheader))
        self.assertTrue(any("Offline demo" in w.value for w in at.warning))
        self.assertEqual(len(at.metric), 6)
        at.run()  # rerun with the stored result (e.g. history now has an entry)
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
