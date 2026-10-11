import unittest
from unittest.mock import patch
from agent import Evidence, plan_questions, run_agent, _fallback

class AgentTests(unittest.TestCase):
    def test_plan_has_three_questions(self):
        qs=plan_questions("Compare vector databases for a student project")
        self.assertEqual(len(qs),3)
        self.assertTrue(all("Compare vector databases" in q for q in qs))
    def test_blank_plan(self): self.assertEqual(plan_questions("  "),[])
    def test_fallback_links_sources(self):
        e=Evidence("Official docs","https://example.org/docs","Documentation snippet","Extracted text","fetched:200","query")
        out=_fallback("Research example docs",[e])
        self.assertIn("https://example.org/docs",out)
        self.assertIn("Unknowns and caveats",out)
    def test_short_goal_rejected(self):
        with self.assertRaises(ValueError): run_agent("short")
    @patch("agent._gemini",return_value=None)
    @patch("agent.search_web",return_value=[])
    def test_bounded_run_has_trace(self,search,_gemini):
        r=run_agent("Research a topic with enough detail",1)
        self.assertEqual(r["metrics"]["searches_planned"],1)
        self.assertTrue(r["events"])
        self.assertIn("summary",r)
        search.assert_called_once()

if __name__=="__main__": unittest.main()
