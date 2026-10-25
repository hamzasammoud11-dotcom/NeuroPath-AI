"""NeuroPath AI v005: v003 visual design with Analytics and Explainability pages."""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[1]; INF=ROOT/"inference-v003"; V2=ROOT/"app-v002"
sys.path.insert(0,str(INF)); sys.path.insert(0,str(V2)); sys.path.insert(0,str(Path(__file__).parent))
from inference import EXPECTED_CHECKPOINT_NAME,EXPECTED_CHECKPOINT_SHA256,THRESHOLD,load_model,predict_subject
from validation import validate_uploads
from explainability import TARGET_LAYER_NAME,gradcam,heatmap_rgb,overlay_rgb
from logic import report_bytes,state_signature

CHECKPOINT=INF/EXPECTED_CHECKPOINT_NAME; AUROC=0.6975; N_VALIDATION=40
Z=("z052","z061","z070","z079","z088","z096","z105","z114","z122")
st.set_page_config(page_title="NeuroPath AI · v005",page_icon="🧠",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{--panel:rgba(14,29,52,.86);--line:rgba(110,220,255,.24);--cyan:#62e6ff;--muted:#b3c5d8;--white:#f3f8ff}
.stApp{background:radial-gradient(circle at 78% 0%,#172b56 0%,transparent 34%),linear-gradient(145deg,#040b15,#071424 55%,#0b1022);color:var(--white);font-family:Inter,Segoe UI,sans-serif}.block-container{max-width:1450px;padding:2rem 2.5rem 4rem}.glass{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:1.25rem;box-shadow:0 14px 50px #0005}.eyebrow{color:var(--cyan);font-family:monospace;font-size:.72rem;letter-spacing:.13em;text-transform:uppercase}.notice{border:1px solid #efb7ff88;background:#6d3c8150;border-radius:12px;padding:.85rem 1rem;color:#ffe4fc}.muted{color:var(--muted)}.mono{font-family:monospace;font-size:.72rem;overflow-wrap:anywhere;color:#c9f3ff}[data-testid="stSidebar"]{background:#050d19;border-right:1px solid var(--line)}div[data-testid="stMetric"]{background:#0b1a2c;border:1px solid var(--line);border-radius:14px;padding:1rem}
</style>""",unsafe_allow_html=True)
for key,default in (("uploads",None),("result",None),("cams",None),("subject",None)): st.session_state.setdefault(key,default)
with st.sidebar:
    st.markdown('<div class="eyebrow">NeuroPath AI · v005</div>',unsafe_allow_html=True)
    page=st.radio("Navigation",["Overview","MRI Workspace","AI Analysis","Results","Analytics","Explainability","Model Information","Research Limitations"],label_visibility="collapsed")
    st.divider(); st.success("Checkpoint verified" if CHECKPOINT.is_file() else "Checkpoint missing"); st.caption("Local-only · no telemetry · no image persistence")
    if st.button("Reset analysis",use_container_width=True):
        for k in ("uploads","result","cams","subject"): st.session_state[k]=None
        st.rerun()
st.markdown('<div class="eyebrow">Exploratory neuroimaging workstation · v005</div><h1>NeuroPath AI</h1>',unsafe_allow_html=True)
st.markdown('<div class="notice">Research use only — not for medical diagnosis. Anatomical laterality remains UNVERIFIED.</div>',unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def verified_model(): return load_model(CHECKPOINT,device="cpu")

if page=="Overview":
    a,b,c=st.columns(3); a.metric("Selected model","CNN v003"); b.metric("Validation AUROC",f"{AUROC:.4f}"); c.metric("Explainability target",TARGET_LAYER_NAME)
    st.markdown('<div class="glass"><h3>Workflow</h3><p class="muted">Upload nine training-partition slices → validate → run the frozen CNN → inspect real outputs in Analytics and exploratory Grad-CAM maps in Explainability.</p></div>',unsafe_allow_html=True)
elif page=="MRI Workspace":
    st.subheader("MRI Workspace"); st.write("Upload exactly nine original-view 208×208 grayscale PNGs from one frozen training subject.")
    uploaded=st.file_uploader("MRI slices",type=["png"],accept_multiple_files=True)
    if uploaded:
        try: ordered=validate_uploads(uploaded)
        except ValueError as exc: st.error(f"Validation blocked: {exc}"); ordered=None
        if ordered:
            signature=state_signature(ordered)
            if signature!=state_signature(st.session_state.uploads): st.session_state.result=st.session_state.cams=None
            st.session_state.uploads=ordered; st.session_state.subject=ordered[0][0].split("_MR1_")[0]
            st.success(f"Validated training subject {st.session_state.subject} · sequence locked")
            cols=st.columns(3)
            for i,(name,image) in enumerate(ordered):
                with cols[i%3]: st.image(image,caption=name,use_container_width=True)
            if st.button("Run verified CNN",type="primary",use_container_width=True):
                try:
                    if not CHECKPOINT.is_file(): raise FileNotFoundError(f"Verified checkpoint missing: {CHECKPOINT}")
                    model=verified_model(); result=predict_subject(model,[image for _,image in ordered],subject_id=st.session_state.subject)
                    cams=[gradcam(model,image) for _,image in ordered]
                    if any(not (cam.shape==(128,128) and cam.dtype==__import__('numpy').float32 and __import__('numpy').isfinite(cam).all()) for cam,_ in cams): raise RuntimeError("Invalid Grad-CAM output")
                    if any(abs(cams[i][1]-result["slice_positive_probabilities"][i])>1e-6 for i in range(9)): raise RuntimeError("Grad-CAM score differs from prediction forward pass")
                    st.session_state.result=result; st.session_state.cams=cams; st.success("Analysis complete · cached for all pages")
                except Exception as exc: st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
elif page=="AI Analysis":
    st.subheader("AI Analysis"); st.info("Run inference from MRI Workspace. Results and Grad-CAM maps are retained in session state; changing pages does not rerun the model.")
    if st.session_state.result: st.success("Prediction and explainability outputs are ready.")
elif page=="Results":
    st.subheader("Results"); result=st.session_state.result
    if not result: st.info("No analysis yet. Open MRI Workspace to upload and analyze a training subject.")
    else:
        a,b,c=st.columns(3); a.metric("Subject-level CDR score",f"{result['cdr_score']:.4f}"); b.metric("Predicted class",f"Class {result['prediction']}"); c.metric("Fixed threshold",f"{THRESHOLD:.1f}")
        st.caption("The score is the mean of nine positive-class outputs. It is not clinical confidence.")
        st.write("Class 0 = CDR 0. Class 1 = CDR > 0 in the frozen exploratory protocol. This is not a diagnosis.")
        st.download_button("Download research report",report_bytes(result,st.session_state.subject),"neuropath-v005-report.json","application/json")
elif page=="Analytics":
    st.subheader("Analytics"); result=st.session_state.result
    if not result: st.info("Run an analysis first; no artificial measurements are shown.")
    else:
        st.markdown('<div class="glass"><div class="eyebrow">Nine real model outputs</div><h3>Slice-level positive-class scores</h3><p class="muted">These are per-slice CNN outputs. The subject classification is their arithmetic mean, compared with the fixed threshold 0.5.</p></div>',unsafe_allow_html=True)
        choices=[f"{i+1} · {Z[i]} · {st.session_state.uploads[i][0]}" for i in range(9)]; selected=st.selectbox("Inspect slice",choices)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=list(Z),y=result["slice_positive_probabilities"],mode="lines+markers",customdata=[st.session_state.uploads[i][0] for i in range(9)],hovertemplate="%{x}<br>%{customdata}<br>score=%{y:.4f}<extra></extra>",line=dict(color="#62e6ff",width=3),marker=dict(color="#a78bfa",size=9))); fig.add_hline(y=.5,line_dash="dash",line_color="#ffb86b",annotation_text="threshold 0.5"); fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(6,16,30,.75)",font_color="#eaf7ff",height=360,margin=dict(l=20,r=20,t=20,b=20),xaxis_title="Slice index",yaxis_title="Positive-class score",hovermode="x unified"); st.plotly_chart(fig,use_container_width=True)
        st.metric("Subject-level mean score",f"{result['cdr_score']:.4f}"); st.caption(f"Selected: {selected}. Validation AUROC {AUROC:.4f} is separate model performance on {N_VALIDATION} validation subjects.")
elif page=="Explainability":
    st.subheader("Explain prediction"); result=st.session_state.result; cams=st.session_state.cams; ordered=st.session_state.uploads
    st.warning("Grad-CAM highlights image regions that influenced the CNN prediction. It does not establish lesion location, anatomical laterality, or a medical diagnosis.")
    if not result or not cams or not ordered: st.info("Run an analysis first; no fabricated heatmaps are shown.")
    else:
        index=st.slider("Browse MRI slice",0,8,0,format="%d · "+Z[0]); opacity=st.slider("Overlay opacity",.1,1.,.55,.05); cam,_=cams[index]; image=ordered[index][1]
        tabs=st.tabs(["Original","Heatmap","Overlay"])
        with tabs[0]: st.image(image,caption=ordered[index][0],use_container_width=True)
        with tabs[1]: st.image(heatmap_rgb(cam),caption=f"Exploratory model explainability map · {ordered[index][0]}",use_container_width=True)
        with tabs[2]: st.image(overlay_rgb(image,cam,opacity),caption=f"Regions influencing model output · opacity {opacity:.2f}",use_container_width=True)
        st.caption(f"Target layer: {TARGET_LAYER_NAME} · positive-class output (class 1) · normalized display map 128×128.")
elif page=="Model Information":
    st.subheader("Model Information"); st.markdown('<div class="glass">',unsafe_allow_html=True); st.write("**ImageClassifier2D / image2d_v1** · horizontal_flip · epoch 18"); st.write(f"Validation AUROC: **{AUROC:.4f}** on {N_VALIDATION} validation subjects"); st.write(f"Grad-CAM target: `{TARGET_LAYER_NAME}` — final convolution before pooling."); st.markdown(f'<div class="mono">SHA-256 · {EXPECTED_CHECKPOINT_SHA256}</div>',unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
elif page=="Research Limitations":
    st.subheader("Research Limitations"); st.warning("Research use only — not for medical diagnosis."); st.write("Grad-CAM identifies regions that influenced the model score under this frozen preprocessing. It is not a validated lesion localization, does not establish causality or laterality, and may reflect shortcuts. Validation AUROC is not clinical performance or prediction confidence. Held-out test subjects remain inaccessible to this app.")
