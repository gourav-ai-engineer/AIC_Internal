import json
from datetime import datetime
import streamlit as st
from agent import run_agent

st.set_page_config(page_title="ProofPilot | Agentic Research",page_icon="🧭",layout="wide")
st.markdown("""<style>.block-container{padding-top:2rem;max-width:1240px}.hero{padding:1.5rem;border:1px solid rgba(128,128,128,.3);border-radius:18px;margin-bottom:1rem}.hero h1{margin:0;letter-spacing:-.04em}.muted{opacity:.75}</style><div class="hero"><div>AGENTIC RESEARCH WORKSPACE</div><h1>🧭 ProofPilot</h1><div class="muted">From a broad goal to a sourced brief — plan, search, inspect, adapt, explain.</div></div>""",unsafe_allow_html=True)
examples=["Compare open-source agent frameworks for a student project, including trade-offs and official documentation.","Research free online AI hackathons accepting student submissions in the next 14 days; verify deadlines and eligibility.","Investigate how small language models can support multilingual education in Northeast India using research papers and official sources."]
with st.sidebar:
    st.header("Run settings")
    budget=st.slider("Search budget",1,8,5)
    st.caption("Goal → Plan → Search → Inspect → Review → Synthesize")
    st.caption("Gemini is optional; a local fallback works without an API key.")
    st.caption("Prototype: verify important details at original sources.")
if "last_run" not in st.session_state: st.session_state.last_run=None
st.subheader("Give your agent a goal")
choice=st.selectbox("Start with an example (optional)",["Write my own"]+examples)
goal=st.text_area("Research goal",value="" if choice=="Write my own" else choice,height=105,placeholder="What do you need to research or decide?")
c1,c2=st.columns([1,3])
with c1: clicked=st.button("▶ Run agent",type="primary",use_container_width=True)
with c2: st.caption("Live search needs internet. External pages are treated as evidence, not instructions.")
if clicked:
    if len(goal.strip())<8: st.error("Enter a goal with at least 8 characters.")
    else:
        status=st.empty(); trace=st.empty(); live=[]
        def progress(ev):
            live.append(ev); status.info(f"**{ev['step'].upper()}** · {ev['status']}")
            with trace.container():
                for x in live[-5:]: st.caption(f"{x.get('time','')} · {x['step']}: {x['status']}")
        with st.spinner("Running bounded agent loop…"):
            try:
                st.session_state.last_run=run_agent(goal,budget,progress); status.success("Run complete.")
            except Exception as ex: st.error(f"Run failed: {type(ex).__name__}. Check connectivity and retry.")
r=st.session_state.last_run
if r:
    st.markdown("---"); st.subheader("Mission report"); st.caption(f"Goal: {r['goal']} · Synthesis: {r['synthesis_mode']}")
    m=r["metrics"]; cols=st.columns(4)
    for col,label,key in zip(cols,["Sources","Pages inspected","Unique domains","Trace events"],["unique_sources","pages_inspected","unique_domains","agent_steps"]): col.metric(label,m[key])
    t1,t2,t3,t4=st.tabs(["📄 Research brief","🔎 Evidence ledger","🧠 Agent trace","🧭 Plan"])
    with t1:
        st.markdown(r["summary"])
        report=f"# ProofPilot Research Brief\n\nGenerated: {datetime.now().astimezone().isoformat(timespec='seconds')}\n\n**Goal:** {r['goal']}\n\n**Synthesis:** {r['synthesis_mode']}\n\n{r['summary']}\n\n## Evidence ledger\n"
        for i,e in enumerate(r["evidence"],1): report+=f"\n### [S{i}] {e['title']}\n- URL: {e['url']}\n- Status: {e['status']}\n- Query: {e['query']}\n\n{e['snippet']}\n"
        st.download_button("⬇ Download Markdown brief",report,file_name="proofpilot_report.md",mime="text/markdown")
        st.download_button("⬇ Download run data (JSON)",json.dumps(r,ensure_ascii=False,indent=2),file_name="proofpilot_run.json",mime="application/json")
    with t2:
        st.write("Retrieved evidence is not automatically verified.")
        for i,e in enumerate(r["evidence"],1):
            with st.expander(f"[S{i}] {e['title']} · {e['status']}",expanded=i<=3):
                st.markdown(f"**Source:** [{e['url']}]({e['url']})"); st.caption(f"Query: {e['query']}"); st.write(e["snippet"])
                if e["page_text"]: st.markdown("**Extracted text**"); st.write(e["page_text"][:1800])
                else: st.warning("No page text extracted; open the source directly.")
    with t3:
        st.write("Observable tool and decision trace")
        for e in r["events"]:
            with st.container(border=True):
                st.markdown(f"**{e.get('time','')} · {e['step'].upper()}**"); st.write(e["status"])
                extra={k:v for k,v in e.items() if k not in ("time","step","status")}
                if extra: st.json(extra,expanded=False)
    with t4:
        for i,q in enumerate(r["questions"],1): st.markdown(f"**{i}.** {q}")
        st.caption("The planner is transparent and bounded; low inspection coverage can trigger a targeted follow-up.")
st.markdown("---")
st.caption("ProofPilot · Evidence first, autonomy with guardrails. Verify consequential claims at original sources.")
