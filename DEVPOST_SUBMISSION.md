# Devpost submission — copy-paste ready

> Fill in the two links marked **TODO** before submitting. Everything below describes features that exist in the
> repository. The only numbers quoted are ones the repository itself measures.

## Project name
ProofPilot — the research agent that doubts itself

## Tagline (≤ 200 chars)
Two AI agents, a Researcher and a Critic, plan, search, argue, and check every claim against its sources before you
see it. Research a topic or fact-check a claim.

## Links
- Source code: https://github.com/gourav-ai-engineer/AIC_Internal
- Demo video: **TODO** (YouTube / Vimeo, unlisted is fine)
- Live app (optional): **TODO** (Streamlit Community Cloud URL)

---

## Inspiration
Every student has asked an AI a research question and received a confident answer with a citation that doesn't say
what the AI claimed, or doesn't exist at all. Chatbots answer once and never check themselves. Real researchers
work differently: they break the problem down, gather evidence from several places, look for reasons they might be
wrong, and have someone else review their work. We built an agent that does all of that.

## What it does
ProofPilot has two missions.

**🔬 Research a topic.** Example: *"Compare LangGraph vs CrewAI vs AutoGen for a student project"*.
1. **Recalls** related past research from long-term memory and reuses trusted sources.
2. **Plans** 3–5 sub-questions, always including a **skeptic pass** that looks for counter-evidence.
3. **Runs a decide → act → observe loop.** At every step the Researcher chooses one tool (web search, news,
   Wikipedia, arXiv, GitHub, or reading a full page) or decides to finish, and explains why.
4. **Adapts:** it reformulates queries, switches tools, uses circuit breakers for failing tools, and falls back
   from the LLM to its heuristic policy if the model is rate-limited.
5. **Verifies** every finding against the source text and detects **contradictions between sources**.
6. **The Critic agent attacks the draft.** It flags single-source findings, disagreements, gaps and stale evidence.
   Each objection becomes a follow-up search the Researcher must run, using a reserved budget. Then everything is
   re-verified, and you see the evidence strength **before → after** and which findings were upgraded.
7. **Reports** a brief with clickable citations and confidence badges (🟢 corroborated by several independent
   domains, 🟠 single source, 🔴 unsupported). Alongside it: an evidence ledger with credibility scores, the full
   decision trace, a coverage view, and an evidence-map graph.
8. **💬 Ask the evidence:** follow-up questions are answered only from the collected sources, and each cited
   sentence is grounding-checked.

**⚖️ Fact-check a claim.** Example: *"Fact-check: The Great Wall of China is visible from space"*. The agent
plans *supporting evidence / primary sources / debunk* searches and returns a verdict: **Supported · Mostly
supported · Weakly supported · Disputed · Mostly refuted · Refuted · Unverified**. The verdict is computed from
credibility-weighted evidence for and against, and every contradicting source is quoted. The LLM writes prose but
never decides the verdict.

## Why it is agentic
- **Planning:** goal decomposition, with per-sub-question tool choices and a rationale.
- **Tool use:** 6 tools with no required API keys, chosen dynamically at each step.
- **Reasoning:** every decision records a "thought" (ReAct style), streamed live in the UI.
- **Multi-agent collaboration:** a Researcher and a Critic with separate roles, budgets and prompts. The Critic's
  objections drive new tool calls.
- **Observation and adaptation:** reformulation, tool switching, circuit breakers and LLM fallback.
- **Reflection and self-verification:** coverage checks, early stopping, claim grounding and contradiction
  detection, followed by a re-verification after the Critic round.
- **Memory:** runs persist, and related runs are recalled to seed new research.
- **Autonomy with guardrails:**
  - a hard tool budget and time limit
  - an override when the agent tries to finish before every sub-question has been researched
  - the LLM may only fetch URLs that a tool actually returned
  - prompt-injection text in web pages is removed
  - an SSRF guard blocks private addresses

## How we built it
- **Python** package `proofpilot/`, one module per responsibility: controller, planner, tools, critic, conflicts,
  fact-check, verifier, credibility, report, Q&A, memory, offline mode, evaluation, CLI.
- **Streamlit** dashboard that streams agent events live, plus a Graphviz evidence map.
- **Tools:** DuckDuckGo search and news (`ddgs`), the Wikipedia API, the arXiv API, the GitHub search API, and a page
  reader built on Requests and Beautiful Soup.
- **LLM (optional):** a provider-agnostic client over plain HTTP for **Gemini**, **Groq**, any OpenAI-compatible
  endpoint, or local **Ollama**, with strictly validated JSON outputs. Without a key, everything still works through
  transparent rule-based policies.
- **Verification:** lexical grounding (sentence-level and whole-text coverage, plural folding, and a subject-entity
  gate so that a source about Go can't "confirm" a claim about Rust), plus contradiction rules for polarity,
  opposite terms and numbers.
- **Quality:** 81 offline tests (including headless UI tests and mocked HTTP for every live tool), CI on Python
  3.10 and 3.12, and a verifier benchmark.

## Measured results
On 45 hand-labelled cases (`python -m proofpilot.evaluate`):

| Check | Accuracy | Precision | Recall |
|---|---|---|---|
| Claim grounding | 83% | 100% | 69% |
| Contradiction detection | 95% | 100% | 91% |

The verifier is conservative. On this set it never accepted a source that doesn't support a claim, and never
invented a contradiction. It does miss some paraphrases. The set is small and was also used to find bugs during
development, so it is a regression check rather than an independent benchmark.

## Challenges we ran into
- **LLMs make up citations.** We stopped trusting the writer and built a verifier that re-reads the sources and
  removes or downgrades anything it can't find.
- **One agent grading its own work is too lenient.** Adding a separate Critic with its own budget made weaknesses
  turn into concrete follow-up work instead of disclaimers.
- **Free APIs fail and rate-limit.** Failure became a normal observation: circuit breakers, LLM fallback, and a
  deterministic offline mode so the demo never depends on Wi-Fi.
- **Web pages can attack agents.** Text like "ignore previous instructions" is detected and stripped, and the
  source's credibility is lowered.

## Accomplishments we're proud of
- It works **without any API key**, and the same loop gets smarter with an LLM.
- Confidence levels and verdicts are **explained**, with quoted evidence on both sides.
- The full trace makes behaviour auditable: judges can see *why* each tool was chosen and what the Critic changed.
- We **measured** the verifier instead of just claiming it works.

## What we learned
Agentic systems are mostly about handling the unhappy path: tools fail, models misbehave, sources disagree, and
content is untrusted. Making the process observable, and giving a second agent the job of finding flaws, made the
system both more reliable and easier to trust.

## What's next
- Semantic (embedding or LLM-judge) grounding, to catch the paraphrases that lexical matching misses.
- "Watch this topic": scheduled re-runs that alert you when the evidence or a verdict changes.
- PDF reading, and exporting the brief as a PDF.

## Built with
python · streamlit · requests · beautifulsoup4 · ddgs · wikipedia-api · arxiv-api · github-api · gemini · groq ·
ollama · graphviz · github-actions

## How to run
See the [README](https://github.com/gourav-ai-engineer/AIC_Internal#quick-start-windows-powershell). In short:
`pip install -r requirements.txt`, then `streamlit run app.py`. No key is needed; add a free Gemini key for the
best results. **Offline demo mode** works without internet.

---

## Final checklist
- [ ] `python -m unittest discover -v` passes on your machine (81 tests).
- [ ] One **live** research run and one **live** fact-check completed with real sources (ideally with a Gemini key).
- [ ] Demo video recorded (see `DEMO_SCRIPT.md`), uploaded, and the link added above.
- [ ] No API key visible in the video, the screenshots or the repository.
- [ ] Optional: deploy to Streamlit Community Cloud and add the URL.
- [ ] Devpost gallery: add the images in `docs/screenshots/` (dashboard, trace, critic review, fact-check verdict,
      evidence map). Note that they were captured in offline demo mode, or replace them with live-run screenshots.
- [ ] Submit before **Oct 12, 2026, 9:30 AM IST**.
