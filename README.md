# ProofPilot — Evidence-First Agentic Research

**A goal-to-brief research agent** that plans questions, searches the live web, inspects source pages, reviews evidence coverage, adapts when results are weak, and exports a traceable report.

Built for AIC Internal Competition 2026 (Agentic AI).

## Why it is agentic
ProofPilot runs an observable, bounded loop: **Goal → Plan → Search → Inspect → Review → Adapt → Synthesize**. It uses live search and page-fetch tools, tracks observations and failures, and may issue a targeted follow-up when inspected evidence is sparse. Source text is treated as untrusted data, not instructions.

## Features
- Streamlit dashboard with activity timeline, evidence ledger, and run trace
- Live web search plus page extraction with timeouts and graceful failures
- Bounded research budget and source diversity
- Optional Gemini synthesis; no-key local summary fallback
- Markdown and JSON exports; unit tests

## Run (Windows PowerShell)
```powershell
git clone https://github.com/gourav-ai-engineer/AIC_Internal.git
cd AIC_Internal
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```
If activation is blocked, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m streamlit run app.py`.

Optional Gemini: set `$env:GEMINI_API_KEY="YOUR_KEY"` in PowerShell. Never commit API keys. Without it, the fallback still runs.

## Demo prompt
"Compare open-source agent frameworks for a student project, including trade-offs and official documentation."

## Demo checklist
1. Run a real search and show planning, tool calls, page inspection and adaptive review.
2. Open the evidence ledger; click through to original sources.
3. Show the event trace and download the Markdown/JSON report.
4. Explain that retrieval is not fact-checking and users must verify consequential claims.

## Judging alignment
- **Functionality:** real tools, bounded execution, timeouts, graceful errors, tests.
- **Agentic concepts:** planning, tool use, observations, adaptation, trace.
- **Creativity:** evidence coverage and uncertainty are part of the product.
- **Submission:** reproducible setup, demo script, explicit limitations.
- **Usefulness:** research briefs with sources and next actions.

## Limitations
Results may be incomplete or stale; lexical ranking is a heuristic. ProofPilot does not guarantee correctness or completeness and should not be the sole basis for high-stakes decisions.

## Tests
`python -m unittest discover -s tests -v`

MIT License.