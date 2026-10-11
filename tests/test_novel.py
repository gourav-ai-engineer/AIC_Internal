"""Tests for the Critic agent, contradiction detection, fact-check mode, evidence Q&A and the evaluation harness."""
import unittest

from proofpilot import run_agent
from proofpilot.conflicts import Conflict, compare, confirm_with_llm, find_conflicts
from proofpilot.critic import heuristic_objections, llm_objections
from proofpilot.evaluate import run as run_eval
from proofpilot.factcheck import claim_of, is_factcheck, judge
from proofpilot.models import Claim, Evidence, SubQuestion
from proofpilot.qa import answer, answer_local
from proofpilot.verify import support_score

from tests.helpers import FakeLLM, FakeToolBox, make_agent

GOAL = "Compare vector databases for a student project"


def ev(sid, domain, text, cred=0.7, sq="Q1", published=""):
    return Evidence(f"{domain} page", f"https://{domain}/{sid}", text, sid=sid, credibility=cred, subquestion=sq,
                    published=published)


class ConflictTests(unittest.TestCase):
    def test_compare_kinds(self):
        self.assertEqual(compare("The library supports Python 3.8 and later versions.",
                                 "The library does not support Python 3.8 or later versions.")[0], "polarity")
        self.assertEqual(compare("Electric cars have lower lifetime emissions than petrol cars.",
                                 "Electric cars have higher lifetime emissions than petrol cars.")[0], "opposite terms")
        self.assertEqual(compare("The submission deadline for the hackathon is 12 October 2026.",
                                 "The submission deadline for the hackathon is 15 October 2026.")[0], "numbers differ")

    def test_compatible_statements_are_not_conflicts(self):
        self.assertIsNone(compare("The framework is free and open-source for students.",
                                  "The framework is free and open-source and popular with students."))
        self.assertIsNone(compare("The deadline is 12 October.", "Python is a programming language."))

    def test_find_conflicts_dedupes_and_leans(self):
        cited = [ev("S1", "a.org", "The submission deadline for the hackathon is 12 October 2026."),
                 ev("S2", "b.org", "The submission deadline for the hackathon is 12 October 2026."),
                 ev("S3", "c.org", "The submission deadline for the hackathon is 15 October 2026.")]
        conflicts = find_conflicts(cited, "hackathon submission deadline")
        self.assertEqual(len(conflicts), 1)
        c = conflicts[0]
        self.assertIn("S3", (c.a_sid, c.b_sid))
        self.assertTrue(c.leaning.startswith(("S1", "S2")))
        self.assertIn("2 vs 1 domains", c.leaning)

    def test_same_domain_never_conflicts_with_itself(self):
        cited = [ev("S1", "a.org", "The tool is free for students."), ev("S2", "a.org", "The tool is paid for students.")]
        self.assertEqual(find_conflicts(cited, "tool students"), [])

    def test_llm_can_veto_false_positive(self):
        c = Conflict("polarity", "S1", "a", "S2", "b", 0.9)

        class Veto(FakeLLM):
            def complete_json(self, prompt, **_):
                return {"pairs": [{"id": 1, "contradict": False}]}
        self.assertEqual(confirm_with_llm([c], Veto()), [])
        self.assertEqual(confirm_with_llm([c], FakeLLM(fail=True)), [c])  # errors keep the heuristic result


class CriticTests(unittest.TestCase):
    def setUp(self):
        self.cited = [ev("S1", "a.org", "Milvus scales to billions of vectors.", sq="Q1"),
                      ev("S2", "b.org", "Chroma is easy to embed in Python apps.", sq="Q2", cred=0.3)]
        self.sqs = [SubQuestion("Q1", "Which vector databases scale?"), SubQuestion("Q2", "Which are easy?")]

    def test_heuristic_objections_cover_weak_points(self):
        claims = [Claim("Milvus scales to billions of vectors", ["S1"], domains=1, confidence="Low"),
                  Claim("Chroma is easy to embed", ["S2"], domains=1, confidence="Low")]
        conflict = Conflict("polarity", "S1", "Milvus scales well", "S2", "Milvus does not scale", 0.8)
        objections = heuristic_objections("Latest vector databases 2026", ["latest"], claims, [conflict], self.sqs,
                                          {"Q1": {"covered": True}, "Q2": {"covered": False}}, self.cited,
                                          ["web_search", "news_search"], limit=10)
        kinds = [o.kind for o in objections]
        self.assertEqual(kinds[0], "conflict")
        for kind in ("single_source", "gap", "stale", "weak_sources"):
            self.assertIn(kind, kinds)
        self.assertEqual(next(o for o in objections if o.kind == "stale").tool, "news_search")
        self.assertEqual(len({o.query.lower() for o in objections}), len(objections))

    def test_no_searchers_no_objections(self):
        self.assertEqual(heuristic_objections("x", [], [], [], self.sqs, {}, self.cited, []), [])

    def test_llm_objections_are_validated(self):
        llm = FakeLLM(critic={"overall": "Thin.", "objections": [
            {"kind": "made_up", "reason": "r", "query": "q1", "tool": "evil_tool", "target": "Q2"},
            {"kind": "gap", "reason": "r", "query": "", "tool": "web_search"},
            "junk"]})
        objections, overall = llm_objections("goal", "exec", [], [], self.sqs, {}, ["web_search"], llm)
        self.assertEqual(overall, "Thin.")
        self.assertEqual(len(objections), 1)
        self.assertEqual((objections[0].kind, objections[0].tool, objections[0].subquestion), ("other", "web_search", "Q2"))

    def test_critic_round_runs_inside_budget_and_reports(self):
        box = FakeToolBox()
        r = make_agent(toolbox=box, budget=12, critic_budget=4).run(GOAL)
        self.assertLessEqual(r["metrics"]["tool_calls"], 12)
        self.assertIn("critic", [e["phase"] for e in r["events"]])
        self.assertIsNotNone(r["critic"])
        self.assertIn("evidence_strength_before_critic", r["metrics"])
        self.assertIn("Critic review", r["markdown"])

    def test_critic_reserve_is_respected_by_researcher(self):
        r = make_agent(budget=9, critic_budget=3, max_reads=10).run(GOAL)
        guard = next(e for e in r["events"] if e["phase"] == "guardrail" and "Researcher budget" in e["message"])
        self.assertIn("3 reserved for the Critic", guard["message"])

    def test_critic_can_be_disabled(self):
        r = make_agent(budget=8, critic=False).run(GOAL)
        self.assertIsNone(r["critic"])
        self.assertNotIn("critic", [e["phase"] for e in r["events"]])

    def test_llm_critic_objection_is_investigated(self):
        llm = FakeLLM()
        r = make_agent(llm=llm, budget=14).run(GOAL)
        self.assertEqual(r["critic"]["by"], "LLM")
        critic_decisions = [e for e in r["events"] if e["phase"] == "decide" and e.get("by") == "critic"]
        self.assertTrue(critic_decisions)


class FactCheckTests(unittest.TestCase):
    def test_detection_and_claim_extraction(self):
        self.assertTrue(is_factcheck("Fact-check: bats are blind"))
        self.assertTrue(is_factcheck("Is it true that coffee dehydrates you?"))
        self.assertFalse(is_factcheck("Compare coffee brands for students"))
        self.assertEqual(claim_of("Is it true that coffee dehydrates you?"), "coffee dehydrates you")

    def test_verdicts(self):
        claim = "Coffee causes dehydration in adults"
        support = [ev("S1", "a.org", "Coffee causes dehydration in adults, one study claims."),
                   ev("S2", "b.org", "Researchers say coffee causes dehydration in adults.")]
        debunk = [ev("S3", "c.gov", "The idea that coffee causes dehydration in adults is a myth.", cred=0.9),
                  ev("S4", "d.edu", "Coffee does not cause dehydration in adults, a review found.", cred=0.85)]
        self.assertEqual(judge(claim, support)["verdict"], "Supported")
        self.assertEqual(judge(claim, debunk)["verdict"], "Refuted")
        self.assertEqual(judge(claim, support[:1])["verdict"], "Weakly supported")
        self.assertEqual(judge(claim, support + debunk)["verdict"], "Disputed")
        self.assertEqual(judge(claim, [ev("S9", "e.org", "Tea is popular in India.")])["verdict"], "Unverified")
        self.assertIn("S3", judge(claim, debunk)["refuting_reasons"])

    def test_factcheck_mission(self):
        r = run_agent("Fact-check: open-source agent frameworks have a learning curve for beginners", 14,
                      offline=True, use_memory=False)
        self.assertEqual(r["task"], "fact-check")
        self.assertEqual(r["planner"], "fact-check")
        self.assertEqual(r["subquestions"][-1]["role"], "skeptic")
        self.assertIn(r["verdict"]["verdict"], ("Supported", "Mostly supported", "Weakly supported", "Disputed",
                                                "Mostly refuted", "Refuted", "Unverified"))
        self.assertIn("## Verdict", r["markdown"])
        self.assertIn("Fact-check Report", r["markdown"])


class QATests(unittest.TestCase):
    def setUp(self):
        self.result = {"goal": "vector dbs", "evidence": [
            ev("S1", "a.org", "Chroma is the easiest vector database for beginners to run locally.").to_dict(),
            ev("S2", "b.org", "Many tutorials say Chroma is the easiest vector database for beginners.").to_dict(),
            ev("S3", "c.org", "Milvus is designed for billion-scale deployments.").to_dict()]}

    def test_extractive_answer_cites_sources(self):
        out = answer("Which vector database is easiest for beginners?", self.result)
        self.assertTrue(out["answerable"])
        self.assertIn("[S1]", out["answer"])
        self.assertEqual(out["confidence"], "Medium")

    def test_unanswerable_question(self):
        out = answer("What is the price of gold in 1850?", self.result)
        self.assertFalse(out["answerable"])

    def test_llm_answer_is_grounding_checked(self):
        good = answer("easiest?", self.result, FakeLLM(qa={"answerable": True, "answer":
                      "Chroma is the easiest vector database for beginners [S1][S2]. Fake fact [S77]."}))
        self.assertNotIn("S77", good["answer"])
        self.assertIn(good["confidence"], ("High", "Medium"))
        bad = answer("q", self.result, FakeLLM(qa={"answerable": True, "answer": "Bananas orbit Jupiter [S3]."}))
        self.assertEqual(bad["confidence"], "Unsupported")
        fallback = answer("Which vector database is easiest for beginners?", self.result, FakeLLM(fail=True))
        self.assertEqual(fallback["mode"], "extractive")

    def test_answer_local_requires_relevance(self):
        self.assertFalse(answer_local("quantum chromodynamics", [Evidence.from_dict(self.result["evidence"][2])])
                         ["answerable"])


class EvaluationTests(unittest.TestCase):
    def test_benchmark_runs_and_is_precise(self):
        r = run_eval()
        self.assertGreaterEqual(r["grounding"]["cases"], 20)
        self.assertGreaterEqual(r["conflict"]["cases"], 20)
        # regression guards: the verifier must stay precise (it may miss paraphrases, but not invent support)
        self.assertGreaterEqual(r["grounding"]["precision"], 0.9)
        self.assertGreaterEqual(r["conflict"]["precision"], 0.9)

    def test_subject_without_predicate_is_not_support(self):
        self.assertLess(support_score("The Great Wall of China is visible from space",
                                      "The Great Wall of China is a popular tourist attraction in China."), 0.45)
        self.assertEqual(judge("The Great Wall of China is visible from space", [
            ev("S1", "a.org", "The Great Wall of China is a popular tourist attraction."),
            ev("S2", "b.org", "The Great Wall of China was built over many centuries.")])["verdict"], "Unverified")

    def test_entity_gate(self):
        self.assertLess(support_score("Rust has a built-in garbage collector",
                                      "Go uses a concurrent garbage collector."), 0.45)


if __name__ == "__main__":
    unittest.main()
