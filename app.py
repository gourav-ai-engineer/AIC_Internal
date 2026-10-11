"""ProofPilot — Streamlit dashboard for the self-verifying research agent."""
from __future__ import annotations

import json
import os

import streamlit as st

from proofpilot import AgentConfig, ResearchAgent, __version__
from proofpilot.llm import LLMClient
from proofpilot.memory import Memory
from proofpilot.report import BADGES
from proofpilot.tools import TOOL_SPECS, ToolBox

st.set_page_config(page_title="ProofPilot · Self-verifying research agent", page_icon="🧭", layout="wide")

ICONS = {"goal": "🎯", "memory": "🧠", "plan": "🗺️", "decide": "🤔", "act": "🔧", "observe": "👁️", "adapt": "🔁",
         "guardrail": "🛡️", "reflect": "🪞", "synthesize": "✍️", "verify": "✅", "report": "📄"}
EXAMPLES = [
    "Compare LangGraph vs CrewAI vs AutoGen for building a multi-agent study tutor as a student project.",
    "Research free online AI hackathons accepting student submissions in the next 30 days; verify deadlines and eligibility.",
    "Investigate how small language models can support multilingual education in India using research papers and official sources.",
    "Should a beginner learn Rust or Go in 2026 for backend development? Include job market and learning curve.",
    "What are the health effects of intermittent fasting according to recent studies?",
]
PROVIDERS = {"Gemini (free tier)": "gemini", "Groq (free tier)": "groq", "OpenAI": "openai", "Ollama (local)": "ollama"}


def load_secrets() -> None:
    """On Streamlit Community Cloud, keys live in st.secrets; expose them as environment variables."""
    try:
        for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                    "OPENAI_MODEL", "GITHUB_TOKEN"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:  # no secrets file configured
        pass


load_secrets()

st.markdown("""
<style>
.block-container {padding-top: 3.2rem; max-width: 1280px;}
.pp-hero {padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: .8rem;
  background: linear-gradient(120deg, rgba(79,70,229,.16), rgba(16,185,129,.12));
  border: 1px solid rgba(128,128,128,.25);}
.pp-hero h1 {margin: 0; font-size: 2.2rem; letter-spacing: -.03em;}
.pp-kicker {font-size: .75rem; letter-spacing: .14em; font-weight: 700; opacity: .7;}
.pp-sub {opacity: .85; margin-top: .25rem;}
.pp-loop {display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .8rem;}
.pp-loop span {padding: .2rem .6rem; border-radius: 999px; font-size: .8rem;
  border: 1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.08);}
.pp-event {padding: .45rem .7rem; border-left: 3px solid rgba(128,128,128,.4); margin: .2rem 0;
  border-radius: 6px; background: rgba(128,128,128,.06); font-size: .92rem;}
.pp-decide {border-left-color: #6366f1;} .pp-act {border-left-color: #0ea5e9;}
.pp-observe {border-left-color: #14b8a6;} .pp-adapt {border-left-color: #f59e0b;}
.pp-guardrail {border-left-color: #ef4444;} .pp-verify {border-left-color: #22c55e;}
.pp-memory {border-left-color: #a855f7;} .pp-plan {border-left-color: #8b5cf6;}
.pp-t {opacity: .6; font-family: monospace; font-size: .8rem; margin-right: .4rem;}
</style>
<div class="pp-hero">
  <div class="pp-kicker">AGENTIC RESEARCH · EVIDENCE FIRST</div>
  <h1>🧭 ProofPilot</h1>
  <div class="pp-sub">A research agent that <b>doubts itself</b>: it plans, chooses its own tools, reads sources,
  hunts for counter-evidence, and checks every claim against the text before you see it.</div>
  <div class="pp-loop"><span>🎯 Goal</span><span>🧠 Recall</span><span>🗺️ Plan</span><span>🤔 Decide</span>
  <span>🔧 Act</span><span>👁️ Observe</span><span>🔁 Adapt</span><span>🪞 Reflect</span><span>✅ Verify</span>
  <span>📄 Report</span></div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("⚙️ Mission settings")
    offline = st.toggle("Offline demo mode", value=False,
                        help="Runs the full agent loop on clearly-labelled synthetic sources. No internet needed.")
    st.subheader("🧠 Reasoning model")
    env_llm = LLMClient.from_env()
    choice = st.selectbox("LLM", ["Auto (environment)", *PROVIDERS, "None (heuristic only)"],
                          help="Optional. Without a model, ProofPilot uses its transparent rule-based policy.")
    llm: LLMClient | None = None
    if choice == "Auto (environment)":
        llm = env_llm
        st.caption(f"Detected: **{env_llm.label}**" if env_llm else
                   "No API key found in the environment — running heuristic-only. Gemini and Groq have free tiers.")
    elif choice in PROVIDERS:
        provider = PROVIDERS[choice]
        key = st.text_input("API key", type="password", help="Used only for this session; never saved to disk.")
        model = st.text_input("Model (optional)", placeholder="provider default")
        if key or provider == "ollama":
            llm = LLMClient(provider, key or "ollama", model.strip() or None)
        else:
            st.caption("Enter a key to enable the LLM.")
    policy = st.radio("Decision policy", ["auto", "llm", "heuristic"], horizontal=True,
                      help="Who chooses the next tool: the LLM (ReAct), or the deterministic heuristic controller.")
    st.subheader("🧰 Tools & guardrails")
    tools = st.multiselect("Enabled tools", list(TOOL_SPECS), default=list(TOOL_SPECS),
                           help="\n".join(f"{n}: {s.description}" for n, s in TOOL_SPECS.items()))
    budget = st.slider("Tool-call budget", 4, 30, 14, help="Hard cap on searches + page reads.")
    max_reads = st.slider("Max full-page reads", 0, 10, 6)
    time_limit = st.slider("Time limit (seconds)", 30, 300, 150, step=10)
    use_memory = st.toggle("Long-term memory", value=True, help="Recall related past runs and reuse trusted sources.")
    memory = Memory() if use_memory else None
    if memory:
        with st.expander("📚 Research history"):
            history = memory.history(12)
            if not history:
                st.caption("No saved runs yet.")
            for h in history:
                score = h["evidence_strength"]
                if st.button(f"{h['goal'][:60]}{'…' if len(h['goal']) > 60 else ''} · {score}/100", key=f"h-{h['id']}"):
                    st.session_state.result = memory.load(h["id"])
            if history and st.button("🗑️ Clear memory"):
                memory.clear()
                st.rerun()
    st.caption(f"ProofPilot v{__version__} · MIT licensed")


# ----------------------------------------------------------------------------- run

st.subheader("What should the agent research?")
example = st.selectbox("Try an example (optional)", ["Write my own"] + EXAMPLES)
goal = st.text_area("Research goal", value="" if example == "Write my own" else example, height=90,
                    placeholder="e.g. Compare free vector databases for a student RAG project")
c1, c2 = st.columns([1, 3])
run_clicked = c1.button("▶ Launch agent", type="primary", key="run")
c2.caption("Web content is treated as untrusted data, never as instructions. "
           "Verify consequential claims at the original source.")


def event_html(e: dict) -> str:
    from html import escape
    return (f"<div class='pp-event pp-{e['phase']}'><span class='pp-t'>{e['t']:5.1f}s</span>"
            f"{ICONS.get(e['phase'], '•')} <b>{e['phase'].upper()}</b> — {escape(e['message'])}</div>")


if run_clicked:
    if len(goal.strip()) < 8:
        st.error("Enter a research goal of at least 8 characters.")
    elif not tools:
        st.error("Enable at least one tool.")
    else:
        config = AgentConfig(budget=budget, time_limit=time_limit, policy=policy, tools=tuple(tools),
                             offline=offline, use_memory=use_memory, max_reads=max_reads)
        progress_bar = st.progress(0.0, text="Starting…")
        with st.status("Agent is working…", expanded=True) as status:
            toolbox = ToolBox(config.tools, offline_mode=offline, pace=0.35 if offline else 0.0)
            agent = ResearchAgent(config, llm=None if offline else llm, memory=memory, toolbox=toolbox)

            def on_event(e: dict) -> None:
                status.markdown(event_html(e), unsafe_allow_html=True)
                progress_bar.progress(min(1.0, agent.calls / config.budget),
                                      text=f"{agent.calls}/{config.budget} tool calls · {e['phase']}")

            agent.progress = on_event
            try:
                st.session_state.result = agent.run(goal)
                m = st.session_state.result["metrics"]
                status.update(label=f"Mission complete — evidence strength {m['evidence_strength']}/100 in "
                                    f"{m['elapsed_s']}s", state="complete", expanded=False)
            except ValueError as ex:
                status.update(label=str(ex), state="error")
            except Exception as ex:  # keep the UI alive and explain what to do
                status.update(label=f"Run failed: {type(ex).__name__}: {ex}. Try Offline demo mode.", state="error")


# ----------------------------------------------------------------------------- results


def dot_escape(text: str, limit: int = 42) -> str:
    text = text if len(text) <= limit else text[: limit - 1] + "…"
    return text.replace("\\", "\\\\").replace('"', "'")


def evidence_map(r: dict) -> str:
    colors = {"High": "#22c55e", "Medium": "#eab308", "Low": "#f97316"}
    lines = ['digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", '
             'fontname="Helvetica", fontsize=10, color="#64748b", fillcolor="#e2e8f0"]; edge [color="#94a3b8"];',
             f'goal [label="🎯 {dot_escape(r["goal"], 60)}", fillcolor="#c7d2fe"];']
    for sq in r["subquestions"]:
        fill = "#bbf7d0" if sq["coverage"]["covered"] else "#fde68a"
        lines.append(f'{sq["id"]} [label="{sq["id"]} · {dot_escape(sq["question"])}", fillcolor="{fill}"];')
        lines.append(f"goal -> {sq['id']};")
    for e in r["evidence"]:
        if e["sid"]:
            lines.append(f'{e["sid"]} [label="{e["sid"]} · {dot_escape(e["domain"], 30)}", '
                         f'fillcolor="{colors.get(e["credibility_label"], "#e2e8f0")}"];')
            if e["subquestion"] in {sq["id"] for sq in r["subquestions"]}:
                lines.append(f'{e["subquestion"]} -> {e["sid"]};')
    lines.append("}")
    return "\n".join(lines)


r = st.session_state.get("result")
if r:
    st.markdown("---")
    st.subheader("📄 Mission report")
    st.caption(f"**Goal:** {r['goal']}  \nMode: {r['mode']} · Planner: {r['planner']} · Policy: {r['policy']} · "
               f"Synthesis: {r['synthesis_mode']} · {r['created']}")
    if r["mode"].startswith("offline"):
        st.warning("Offline demo: every source below is synthetic (reserved `.example` domains). "
                   "Switch off Offline demo mode for real research.")
    m = r["metrics"]
    cols = st.columns(6)
    cols[0].metric("Evidence strength", f"{m['evidence_strength']}/100")
    cols[1].metric("Tool calls", f"{m['tool_calls']}/{m['budget']}")
    cols[2].metric("Sources (cited)", f"{m['unique_sources']} ({m['cited_sources']})")
    cols[3].metric("Independent domains", m["unique_domains"])
    cols[4].metric("Verified findings", f"{m['claims']} · {m['high_confidence_claims']} high")
    cols[5].metric("Adaptations · guards", f"{m['adaptations']} · {m['guardrails']}")

    tabs = st.tabs(["📄 Brief", "✅ Verified findings", "🔎 Evidence ledger", "🧠 Agent trace",
                    "🗺️ Plan & coverage", "🕸️ Evidence map"])
    with tabs[0]:
        st.markdown(r["markdown"])
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download brief (Markdown)", r["markdown"], file_name="proofpilot_brief.md",
                           mime="text/markdown")
        d2.download_button("⬇ Download full run (JSON)", json.dumps(r, ensure_ascii=False, indent=2),
                           file_name="proofpilot_run.json", mime="application/json")
    with tabs[1]:
        st.caption("Each finding was checked against the text of its cited sources. Confidence rises with the number "
                   "of independent domains that contain it; fabricated or unsupported citations are removed.")
        urls = {e["sid"]: e["url"] for e in r["evidence"] if e["sid"]}
        if not r["claims"]:
            st.info("No findings could be verified in this run.")
        for c in r["claims"]:
            with st.container(border=True):
                st.markdown(f"**{BADGES[c['confidence']]}** · {c['domains']} independent domain(s) · "
                            f"grounding {c['grounding']:.0%}")
                st.write(c["text"])
                links = " ".join(f"[{s}]({urls[s]})" for s in c["sources"] if s in urls)
                if links:
                    st.markdown(f"Sources: {links}")
                if c["note"]:
                    st.caption(f"Verifier: {c['note']}")
    with tabs[2]:
        rows = [{"ID": e["sid"] or "—", "Title": e["title"], "URL": e["url"], "Type": e["source_type"],
                 "Credibility": e["credibility"], "Q": e["subquestion"], "Tool": e["tool"],
                 "Read": "✅" if e["page_text"] else "—", "Published": e["published"],
                 "Flags": "🛡️ injection" if e["flags"] else ""} for e in r["evidence"]]
        st.dataframe(rows, hide_index=True, column_config={
            "URL": st.column_config.LinkColumn("URL", width="medium"),
            "Credibility": st.column_config.ProgressColumn("Credibility", min_value=0.0, max_value=1.0, format="%.2f"),
        })
        st.caption(f"{len(r['evidence'])} sources collected; IDs mark the diversity-balanced citation set.")
        for e in [x for x in r["evidence"] if x["sid"]]:
            with st.expander(f"[{e['sid']}] {e['title']} · {e['credibility_label']} credibility"):
                st.markdown(f"**URL:** [{e['url']}]({e['url']})  \n**Found by:** `{e['tool']}` for {e['subquestion']} "
                            f"with query “{e['query']}”  \n**Status:** {e['status']}")
                st.write(e["page_text"][:2000] if e["page_text"] else e["snippet"])
    with tabs[3]:
        phases = [e["phase"] for e in r["events"]]
        st.caption(" · ".join(f"{ICONS.get(p, '•')} {p}: {phases.count(p)}" for p in dict.fromkeys(phases)))
        show_details = st.toggle("Show raw event data", value=False)
        for e in r["events"]:
            st.markdown(event_html(e), unsafe_allow_html=True)
            extra = {k: v for k, v in e.items() if k not in ("i", "t", "time", "phase", "message")}
            if show_details and extra:
                st.json(extra, expanded=False)
    with tabs[4]:
        for sq in r["subquestions"]:
            cov = sq["coverage"]
            with st.container(border=True):
                status_ = "✅ corroborated" if cov["covered"] else "⚠️ weak"
                role = " · 🕵️ skeptic pass" if sq["role"] == "skeptic" else ""
                st.markdown(f"**{sq['id']} · {sq['question']}**{role}  \n{status_} — {cov['sources']} sources, "
                            f"{cov['domains']} domains, {cov['read']} read, {cov['attempts']} attempt(s)")
                st.progress(min(1.0, cov["domains"] / 3), text=f"Independent domains: {cov['domains']} (target 2+)")
                if sq["rationale"]:
                    st.caption(f"Why: {sq['rationale']}")
                st.caption("Tools: " + ", ".join(sq["tools"]) + " · Tried: " +
                           (", ".join(f"`{t.replace('::', ': ')}`" for t in sq["tried"]) or "nothing"))
    with tabs[5]:
        st.caption("Goal → sub-questions (green = corroborated) → cited sources (coloured by credibility).")
        st.graphviz_chart(evidence_map(r))
else:
    st.info("Pick an example or write a goal, then press **Launch agent**. No API key? It still works — and "
            "Offline demo mode works without internet.")
