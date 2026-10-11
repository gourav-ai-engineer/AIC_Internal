# ProofPilot — 3-minute demo script

**Setup before recording**
- Run `python -m unittest discover -v` and make sure everything passes.
- Set a free Gemini key (`$env:GEMINI_API_KEY="..."`) so the LLM ReAct controller is used. Without a key the
  heuristic controller runs, which is still a valid demo.
- Do one warm-up live run so that memory has something to recall later.
- Close personal tabs. Never show your API key: set it in the environment rather than typing it on camera.
- Backup plan: if the Wi-Fi is bad, toggle **Offline demo mode**. The loop is identical and the sources are clearly
  labelled as synthetic.

---

**0:00–0:20 — The problem**
> "When you ask a chatbot to research something, it answers once, sounds confident, and sometimes cites sources that
> don't say what it claims. ProofPilot is an agent that does the research itself, and then checks its own work."

**0:20–0:40 — Give it a goal**
- Choose the example *"Compare LangGraph vs CrewAI vs AutoGen for building a multi-agent study tutor…"*.
- Point at the sidebar: "It has 6 tools, a hard budget of 14 tool calls, a time limit, and long-term memory."
- Press **▶ Launch agent**.

**0:40–1:30 — Watch it think (the agent loop)**
While the trace streams, narrate:
- 🧠 **Memory:** "It first checks whether it has researched something similar before."
- 🗺️ **Plan:** "It splits the goal into sub-questions, including a skeptic pass that looks for counter-evidence."
- 🤔 **Decide → 🔧 Act → 👁️ Observe:** "At every step it picks the next tool and explains why: Wikipedia for
  background, GitHub for adoption signals, arXiv for research…"
- 🔁 **Adapt / 🛡️ Guardrail** (if any appear): "Here a tool returned nothing, so it reformulated the query" or "this
  page contained a prompt-injection attempt, which was stripped out."
- "It stops on its own when every sub-question is backed by two or more independent sources."

**1:30–2:15 — Trust, but verify**
- **Verified findings** tab: "Every claim was checked against the source text. Green means several independent
  domains agree. Orange means a single source. Red means the source didn't support it. Fabricated citations are
  removed automatically."
- **Evidence ledger:** credibility scores, and click one source to open the original page.
- **Evidence map:** "The whole investigation at a glance: goal, sub-questions and sources, coloured by credibility."

**2:15–2:40 — The deliverable**
- **Brief** tab: executive answer with clickable citations, answers per sub-question, comparison table, next actions,
  unknowns.
- Download the Markdown brief and the JSON trace.
- Optional (10 s): run a related goal and show 🧠 *"Recalled 1 related past run… reusing sources"*.

**2:40–3:00 — Close**
> "ProofPilot plans, chooses its own tools, adapts when things fail, remembers what it learned, and audits its own
> claims before you see them. It's an agent you can actually check. Code and setup are on GitHub, and it runs with
> no API key, or even offline."

---

**Honesty rules for the video:** show real runs only. Do not edit in results, and do not claim accuracy numbers you
have not measured. If you use offline mode, say so on camera.
