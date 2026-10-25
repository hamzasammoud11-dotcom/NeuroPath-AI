"""NeuroPath AI v006: v005 design with Overview curves and 3x3 explainability."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[1]; INF=ROOT/"inference-v003"; V2=ROOT/"app-v002"; HERE=Path(__file__).parent
sys.path.insert(0,str(INF)); sys.path.insert(0,str(V2)); sys.path.insert(0,str(HERE))
from inference import EXPECTED_CHECKPOINT_NAME,EXPECTED_CHECKPOINT_SHA256,THRESHOLD,load_model,predict_subject
from validation import validate_uploads
from explainability import TARGET_LAYER_NAME,gradcam,heatmap_rgb,overlay_rgb
from logic import report_bytes,signature
from training_curves import load_curves
CHECKPOINT=INF/EXPECTED_CHECKPOINT_NAME; AUROC=.6975; N_VALIDATION=40; Z=("z052","z061","z070","z079","z088","z096","z105","z114","z122")
st.set_page_config(page_title="NeuroPath AI · v006",page_icon="🧠",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>:root{--panel:rgba(14,29,52,.86);--line:rgba(110,220,255,.24);--cyan:#62e6ff;--muted:#b3c5d8;--white:#f3f8ff}.stApp{background:radial-gradient(circle at 78% 0%,#172b56 0%,transparent 34%),linear-gradient(145deg,#040b15,#071424 55%,#0b1022);color:var(--white);font-family:Inter,Segoe UI,sans-serif}.block-container{max-width:1450px;padding:2rem 2.5rem 4rem}.glass{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:1.25rem;box-shadow:0 14px 50px #0005}.eyebrow{color:var(--cyan);font-family:monospace;font-size:.72rem;letter-spacing:.13em;text-transform:uppercase}.notice{border:1px solid #efb7ff88;background:#6d3c8150;border-radius:12px;padding:.85rem 1rem;color:#ffe4fc}.muted{color:var(--muted)}.mono{font-family:monospace;font-size:.72rem;overflow-wrap:anywhere;color:#c9f3ff}[data-testid="stSidebar"]{background:#050d19;border-right:1px solid var(--line)}div[data-testid="stMetric"]{background:#0b1a2c;border:1px solid var(--line);border-radius:14px;padding:1rem}</style>""",unsafe_allow_html=True)
st.markdown("""<style>
.stAppViewContainer,[data-testid="stAppViewContainer"],section.main{background:transparent}
.stAppViewContainer .main .block-container{max-width:1450px;padding:0 2.5rem 4rem!important;margin-top:0!important}
.stAppViewContainer [data-testid="stHeader"]{height:0!important;background:transparent!important;visibility:hidden}
.stAppViewContainer [data-testid="stToolbar"],div[data-testid="stDecoration"]{display:none!important}
.stApp h1{font-size:clamp(3.6rem,8vw,6.8rem);line-height:.95;letter-spacing:-.075em;font-weight:800;margin:.15rem 0 1.15rem;background:linear-gradient(110deg,#f3f8ff 5%,#62e6ff 38%,#a78bfa 68%,#f3f8ff 92%);background-size:220% auto;-webkit-background-clip:text;background-clip:text;color:transparent;text-shadow:0 0 28px #62e6ff35;animation:brand-enter .8s ease-out both,brand-shimmer 9s ease-in-out 1s infinite}
@keyframes brand-enter{from{opacity:0;transform:translateY(-12px);filter:blur(5px)}to{opacity:1;transform:translateY(0);filter:blur(0)}}
@keyframes brand-shimmer{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
@media (prefers-reduced-motion:reduce){.stApp h1{animation:none}}
</style>""",unsafe_allow_html=True)
st.markdown("""<style>
/* High-contrast presentation layer: scientific accent colors remain distinct. */
.stApp,.stApp p,.stApp label,.stApp [data-testid="stMarkdownContainer"]{color:#f1f5f9}
.stApp .muted,.stApp small,.stApp [data-testid="stCaptionContainer"]{color:#e2e8f0!important}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] [role="radio"],[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *{color:#f8fafc!important}
[data-testid="stSidebar"] [aria-checked="true"]{color:#62e6ff!important;font-weight:700}
div[data-testid="stMetricLabel"]{color:#e2e8f0!important;font-weight:600;letter-spacing:.04em}
div[data-testid="stMetricValue"]{color:#ffffff!important;font-weight:800;text-shadow:0 0 14px #62e6ff28}
div[data-testid="stMetricDelta"]{color:#f1f5f9!important}
.stApp button,.stApp button p{color:#ffffff!important;font-weight:650}
.stApp input,.stApp textarea,.stApp [data-baseweb="select"] *,.stApp [data-baseweb="slider"] *{color:#f8fafc!important}
.stApp [data-testid="stFileUploader"] label,.stApp [data-testid="stFileUploaderDropzone"] *{color:#f1f5f9!important}
.stApp [aria-disabled="true"],.stApp button:disabled{color:#cbd5e1!important;opacity:.9}
.stApp [data-testid="stAlert"] p,.stApp [data-testid="stStatusWidget"] *{color:#f8fafc!important}
</style>""",unsafe_allow_html=True)
for k in ("uploads","result","cams","subject"): st.session_state.setdefault(k,None)
with st.sidebar:
    st.markdown('<div class="eyebrow">NeuroPath AI · v006</div>',unsafe_allow_html=True); page=st.radio("Navigation",["Overview","MRI Workspace","AI Analysis","Results","Analytics","Explainability","Model Information","Research Limitations"],label_visibility="collapsed"); st.divider(); st.success("Checkpoint verified" if CHECKPOINT.is_file() else "Checkpoint missing"); st.caption("Local-only · no telemetry · no image persistence")
    if st.button("Reset analysis",use_container_width=True):
        for k in ("uploads","result","cams","subject"): st.session_state[k]=None
        st.rerun()
st.markdown('<div class="eyebrow">Exploratory neuroimaging workstation · v006</div><h1>NeuroPath AI</h1>',unsafe_allow_html=True); st.markdown('<div class="notice">Research use only — not for medical diagnosis. Anatomical laterality remains UNVERIFIED.</div>',unsafe_allow_html=True)
@st.cache_resource(show_spinner=False)
def verified_model(): return load_model(CHECKPOINT,device="cpu")
def run_analysis(ordered,subject_id):
    """Run the real frozen CNN and Grad-CAM once for the validated session inputs."""
    if not CHECKPOINT.is_file(): raise FileNotFoundError(f"Verified checkpoint missing: {CHECKPOINT}")
    model=verified_model(); result=predict_subject(model,[image for _,image in ordered],subject_id=subject_id); cams=[gradcam(model,image) for _,image in ordered]
    if any(cam.shape!=(128,128) or not np.isfinite(cam).all() for cam,_ in cams): raise RuntimeError("Invalid Grad-CAM output")
    if any(abs(cams[i][1]-result["slice_positive_probabilities"][i])>1e-6 for i in range(9)): raise RuntimeError("Grad-CAM score differs from prediction pass")
    return result,cams
def curve_chart(curves):
    rows=curves["rows"]; fig=go.Figure(); epochs=[r["epoch"] for r in rows]
    for key,name,color in (("training_loss","Training loss","#a78bfa"),("validation_auroc","Validation AUROC","#62e6ff"),("validation_balanced_accuracy","Validation balanced accuracy","#7ee787")):
        values=[r.get(key) for r in rows]
        if any(v is not None for v in values): fig.add_trace(go.Scatter(x=epochs,y=values,mode="lines+markers",name=name,line=dict(color=color,width=3),marker=dict(size=7)))
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(6,16,30,.75)",font_color="#edf8ff",height=390,margin=dict(l=20,r=20,t=20,b=20),xaxis_title="Epoch",yaxis_title="Metric",hovermode="x unified",legend=dict(orientation="h",y=1.12)); return fig
if page=="Overview":
    a,b,c,d=st.columns(4); a.metric("Selected model","CNN v003"); b.metric("Variant","horizontal_flip"); c.metric("Best epoch",18); d.metric("Best validation AUROC",f"{AUROC:.4f}")
    st.markdown('<div class="glass"><div class="eyebrow">Training trajectory</div><h3>Selected model curves</h3><p class="muted">Real metrics are read-only from the selected run artifacts. No values are fabricated when files are unavailable.</p></div>',unsafe_allow_html=True)
    curves=load_curves(ROOT)
    if curves["available"]:
        st.plotly_chart(curve_chart(curves),use_container_width=True); st.caption(f"Run source: {curves['path']}")
    else: st.info("No per-epoch validation artifacts were found locally. Add the selected run directory containing validation-*.json, summary.json, or history.json to display the real curves.")
    st.markdown('<div class="glass"><h3>Workflow</h3><p class="muted">Upload nine training-partition slices → validate → run the frozen CNN → inspect Results, Analytics, and the 3×3 Explainability grid.</p></div>',unsafe_allow_html=True)
elif page=="MRI Workspace":
    st.subheader("MRI Workspace"); uploaded=st.file_uploader("Drop exactly nine original-view 208×208 grayscale PNG slices",type=["png"],accept_multiple_files=True)
    if uploaded:
        try: ordered=validate_uploads(uploaded)
        except ValueError as exc: st.error(f"Validation blocked: {exc}"); ordered=None
        if ordered:
            if signature(ordered)!=signature(st.session_state.uploads): st.session_state.result=st.session_state.cams=None
            st.session_state.uploads=ordered; st.session_state.subject=ordered[0][0].split("_MR1_")[0]; st.success(f"Validated training subject {st.session_state.subject} · sequence locked"); cols=st.columns(3)
            for i,(name,image) in enumerate(ordered):
                with cols[i%3]: st.image(image,caption=name,use_container_width=True)
            if st.button("Run verified CNN",type="primary",use_container_width=True):
                try:
                    model=verified_model(); result=predict_subject(model,[image for _,image in ordered],subject_id=st.session_state.subject); cams=[gradcam(model,image) for _,image in ordered]
                    if any(cam.shape!=(128,128) or not np.isfinite(cam).all() for cam,_ in cams): raise RuntimeError("Invalid Grad-CAM output")
                    if any(abs(cams[i][1]-result["slice_positive_probabilities"][i])>1e-6 for i in range(9)): raise RuntimeError("Grad-CAM score differs from prediction pass")
                    st.session_state.result=result; st.session_state.cams=cams; st.success("Analysis complete · cached for all pages")
                except Exception as exc: st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
elif page=="AI Analysis":
    st.subheader("AI Analysis")
    ordered=st.session_state.uploads; result=st.session_state.result
    if not ordered:
        st.warning("No MRI images are available. Open MRI Workspace, upload and validate exactly nine training-subject slices first.")
    else:
        subject_id=st.session_state.subject; st.success(f"Uploaded MRI status: 9 validated slices for {subject_id}")
        a,b,c=st.columns(3); a.metric("Image validation","PASS"); b.metric("Checkpoint","VERIFIED" if CHECKPOINT.is_file() else "MISSING"); c.metric("Inference state","CACHED" if result else "READY")
        st.caption("The checkpoint SHA-256 is checked by the existing loader. Training-partition restrictions remain enforced by the upload validator.")
        if result:
            st.info("A real analysis is already cached in session state. Navigation does not rerun inference.")
        if st.button("Run AI Analysis" if not result else "Re-run AI Analysis",type="primary",use_container_width=True):
            progress=st.progress(0,text="Preparing validated MRI inputs…")
            try:
                progress.progress(25,text="Loading and verifying the CNN checkpoint…")
                computed,cams=run_analysis(ordered,subject_id)
                progress.progress(100,text="Analysis complete")
                st.session_state.result,st.session_state.cams=computed,cams
                st.success("AI analysis completed with real CNN outputs. Results, Analytics and Explainability are now available.")
            except Exception as exc:
                progress.empty(); st.error(f"AI analysis failed: {type(exc).__name__}: {exc}")
        if st.session_state.result:
            result=st.session_state.result; a,b,c=st.columns(3); a.metric("Subject-level CDR score",f"{result['cdr_score']:.4f}"); b.metric("Predicted class",f"Class {result['prediction']}"); c.metric("Fixed threshold",f"{THRESHOLD:.1f}")
            st.caption("The score is the mean of nine positive-class outputs; it is not clinical confidence.")
elif page=="Results":
    st.subheader("Results"); result=st.session_state.result
    if not result: st.info("Run an analysis first.")
    else:
        a,b,c=st.columns(3); a.metric("Subject-level CDR score",f"{result['cdr_score']:.4f}"); b.metric("Predicted class",f"Class {result['prediction']}"); c.metric("Fixed threshold",f"{THRESHOLD:.1f}"); st.caption("Mean of nine positive-class outputs; not clinical confidence."); st.write("Class 0 = CDR 0. Class 1 = CDR > 0 in the frozen exploratory protocol. This is not a diagnosis."); st.download_button("Download research report",report_bytes(result,st.session_state.subject),"neuropath-v006-report.json","application/json")
elif page=="Analytics":
    st.subheader("Analytics"); result=st.session_state.result
    if not result: st.info("Run an analysis first; no artificial measurements are shown.")
    else:
        fig=go.Figure(go.Scatter(x=list(Z),y=result["slice_positive_probabilities"],mode="lines+markers",customdata=[st.session_state.uploads[i][0] for i in range(9)],hovertemplate="%{x}<br>%{customdata}<br>score=%{y:.4f}<extra></extra>",line=dict(color="#62e6ff",width=3),marker=dict(color="#a78bfa",size=9))); fig.add_hline(y=.5,line_dash="dash",line_color="#ffb86b",annotation_text="threshold 0.5"); fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(6,16,30,.75)",font_color="#edf8ff",height=360,xaxis_title="Slice index",yaxis_title="Positive-class score",hovermode="x unified"); st.plotly_chart(fig,use_container_width=True); st.metric("Subject-level mean score",f"{result['cdr_score']:.4f}"); st.caption(f"Nine real slice outputs · AUROC {AUROC:.4f} is separate validation performance on {N_VALIDATION} subjects.")
elif page=="Explainability":
    st.subheader("Explainability"); st.warning("Grad-CAM highlights image regions that influenced the CNN prediction. It does not establish lesion location, anatomical laterality, or a medical diagnosis."); result=st.session_state.result; cams=st.session_state.cams; ordered=st.session_state.uploads
    if not result or not cams or not ordered: st.info("Run an analysis first; no fabricated heatmaps are shown.")
    else:
        opacity=st.slider("Overlay opacity",.1,1.,.55,.05); st.caption("All nine explanatory overlays are shown simultaneously. Display only; prediction tensors are unchanged.")
        for start in (0,3,6):
            cols=st.columns(3)
            for i,col in enumerate(cols,start):
                with col:
                    cam,score=cams[i]; st.image([ordered[i][1],overlay_rgb(ordered[i][1],cam,opacity)],caption=["Original","Grad-CAM overlay"],width=145); st.caption(f"{ordered[i][0]} · slice score {score:.4f}")
        st.info("These maps show regions that influenced the model prediction. They are exploratory explainability maps, not validated lesion localizations.")
elif page=="Model Information":
    st.subheader("Model Information"); st.markdown('<div class="glass">',unsafe_allow_html=True); st.write("**ImageClassifier2D / image2d_v1** · horizontal_flip · epoch 18"); st.write(f"Validation AUROC: **{AUROC:.4f}** on {N_VALIDATION} validation subjects"); st.write(f"Grad-CAM target: `{TARGET_LAYER_NAME}`"); st.markdown(f'<div class="mono">SHA-256 · {EXPECTED_CHECKPOINT_SHA256}</div>',unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
elif page=="Research Limitations":
    st.subheader("Research Limitations"); st.warning("Research use only — not for medical diagnosis."); st.write("Training curves and metrics are shown only when real run artifacts exist. Grad-CAM maps indicate regions influencing the model output, not lesions, causality, anatomy, or clinical utility. Laterality remains UNVERIFIED; held-out test subjects remain inaccessible.")
