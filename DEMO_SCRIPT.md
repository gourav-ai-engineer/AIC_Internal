# ProofPilot — 3-minute demo script

**Setup before recording**
- Run `python -m unittest discover -v` and make sure everything passes.
- Set a free Gemini key (`$env:GEMINI_API_KEY="..."`) so the LLM Researcher and Critic are used. Without a key the
  rule-based policies run, which is still a valid demo.
- Do one warm-up live run so that memory has something to recall.
- Close personal tabs. Never show your API key: set it in the environment rather than typing it on camera.
- Backup plan: if the Wi-Fi is bad, toggle **Offline demo mode**. The loop is identical and the sources are clearly
  labelled as synthetic. Say so on camera.

---

**0:00–0:15 — The problem**
> "When you ask a chatbot to research something, it answers once, sounds confident, and sometimes cites sources that
> don't say what it claims. ProofPilot is a team of two agents, a Researcher and a Critic, that do the research and
> then try to prove themselves wrong."

**0:15–0:30 — Give it a goal**
- Choose *"Compare LangGraph vs CrewAI vs AutoGen for building a multi-agent study tutor…"*.
- Sidebar: "6 tools, a hard budget of 16 tool calls with 4 reserved for the Critic, a time limit, and memory."
- Press **▶ Launch agent**.

**0:30–1:15 — The Researcher's loop**
While the trace streams, narrate:
- 🧠 Memory, then 🗺️ Plan: "Sub-questions, including a skeptic pass."
- 🤔 Decide → 🔧 Act → 👁️ Observe: "It picks each tool itself and says why: Wikipedia for background, GitHub for
  adoption, arXiv for research."
- 🔁 Adapt / 🛡️ Guardrail, if they appear: "A tool failed, so it rerouted" or "this page tried a prompt
  injection, so the text was stripped."

**1:15–1:50 — The Critic attacks**
- 🕵️ lines in the trace: "Now the Critic reviews the draft. This finding has only one source; these two sources
  disagree. Each objection becomes a new search."
- **Critic & disputes** tab: show an objection with its outcome (*"confirmed — C7 upgraded Low → Medium"*), the
  side-by-side disagreement, and the **Evidence strength** metric with its *+N from Critic* delta.

**1:50–2:20 — Trust, but verify**
- **Verified findings:** "Green means several independent domains agree. Orange means a single source. Red means the
  source didn't support it. Fabricated citations are removed automatically."
- **Evidence map:** "The whole investigation at a glance."
- **Ask the evidence:** type *"Which option is easiest for beginners?"* and show the cited answer and its confidence.

**2:20–2:45 — Fact-check mode**
- Switch to **⚖️ Fact-check a claim** and choose *"The Great Wall of China is visible from space"*.
- Show the verdict card: "The verdict comes from weighing credible sources for and against, with the contradicting
  ones quoted. The LLM never decides it."

**2:45–3:00 — Close**
> "ProofPilot plans, picks its own tools, adapts when things fail, argues with itself through a Critic agent, and
> audits every claim. We even measured the verifier: on our labelled set it never accepted a source that doesn't
> support a claim. Code, tests and setup are on GitHub, and it runs with no API key, or even offline."

---

**Honesty rules for the video:** show real runs only. Do not edit in results, and only quote the measured numbers
from `python -m proofpilot.evaluate`. If you use offline mode, say so on camera.
