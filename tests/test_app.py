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
        self.env = {k: os.environ.get(k) for k in ("PROOFPILOT_HOME", "PROOFPILOT_LLM", "PROOFPILOT_DEMO_PACE")}
        os.environ["PROOFPILOT_DEMO_PACE"] = "0"
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

    def test_factcheck_mission_and_follow_up_question(self):
        at = AppTest.from_file("../app.py", default_timeout=90)
        at.run()
        at.toggle[0].set_value(True)  # offline demo mode
        mission = next(r for r in at.radio if r.label == "Mission type")
        mission.set_value("⚖️ Fact-check a claim").run()
        at.text_area[0].input("Fact-check: open-source agent frameworks have a learning curve for beginners")
        at.button(key="run").click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertTrue(any("EVIDENCE VERDICT" in md.value for md in at.markdown))
        question = next(t for t in at.text_input if t.label == "Your question")
        question.input("Is there a learning curve for beginners?")
        next(b for b in at.button if b.label == "Ask").click().run()  # form submit button
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertTrue(any(md.value.startswith("**Q:**") for md in at.markdown))


if __name__ == "__main__":
    unittest.main()
