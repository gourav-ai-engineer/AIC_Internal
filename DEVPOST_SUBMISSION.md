# Devpost submission draft — ProofPilot

## Project name
ProofPilot — Evidence-First Agentic Research

## Tagline
A research agent that plans, searches, inspects, adapts, and leaves an evidence trail.

## What did you build?
ProofPilot is a lightweight web app that transforms a broad research goal into a structured brief. It creates research questions, searches the public web, inspects promising pages, tracks source links and fetch status, reviews evidence coverage, optionally performs a follow-up search, and exports a Markdown report plus JSON run trace.

## Why is it agentic?
It follows a bounded observe–decide–act workflow rather than producing a single response from one prompt:
1. Plan research questions from the user's goal.
2. Use live web search as a tool.
3. Fetch and extract readable source-page text.
4. Observe tool results and failures in a trace.
5. Review coverage and adapt with a targeted follow-up when the inspected evidence is sparse.
6. Synthesize a brief and expose the underlying evidence.

The run has a configurable budget, visible steps, and explicit limitations. Gemini synthesis is optional; a no-key fallback makes the workflow demonstrable without a paid API.

## Tools and technology
- Python
- Streamlit
- DuckDuckGo search via the `ddgs` package
- Requests and Beautiful Soup for source retrieval and text extraction
- Optional Gemini API for synthesis
- unittest and GitHub Actions workflow

## Why this project?
AI-generated research can sound confident even when its evidence is weak. ProofPilot makes tool use and evidence provenance visible, so a user can inspect the source, understand where the result came from, and identify what still needs verification.

## What is innovative?
The experience emphasizes the research process as a first-class artifact: plan, tool observations, source diversity, extraction status, uncertainty, and downloadable trace are shown alongside the report. The design prioritizes inspectability over pretending a generated answer is automatically correct.

## How to run
See the repository README. Install `requirements.txt`, run `streamlit run app.py`, enter a goal, and click **Run agent**. Internet access is needed for live search. Gemini is optional.

## Demo video outline
1. Introduce the evidence problem (20 seconds).
2. Enter a research goal and show the planner (20 seconds).
3. Run the agent; point out live search, inspection, and coverage review (50 seconds).
4. Open the evidence ledger and click an original source (45 seconds).
5. Show the event trace and export the Markdown report (30 seconds).
6. Close by explaining the bounded agent loop and limitations (15 seconds).

## Limitations and responsible use
Search results can be incomplete or stale, and relevance scoring is heuristic rather than fact verification. ProofPilot does not guarantee completeness or correctness. Users should verify consequential claims at original sources and must not rely on the prototype alone for high-stakes decisions.

## Final checklist before submission
- [ ] Install dependencies and run the app locally.
- [ ] Run `python -m unittest discover -s tests -v` and fix failures.
- [ ] Complete at least one real live-search demo.
- [ ] Record a real video; do not fabricate outcomes or metrics.
- [ ] Check that no API keys or personal information appear in the repository or recording.
- [ ] Add the real demo video URL and deployed app URL to Devpost if available.
- [ ] Confirm the hackathon rules and submission deadline on Devpost.
