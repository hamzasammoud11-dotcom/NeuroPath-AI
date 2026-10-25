"""NeuroPath AI v004: local exploratory CNN dashboard with Grad-CAM."""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]; INFERENCE_DIR = ROOT / "inference-v003"
sys.path.insert(0, str(INFERENCE_DIR)); sys.path.insert(0, str(ROOT / "app-v002"))
from inference import EXPECTED_CHECKPOINT_NAME, EXPECTED_CHECKPOINT_SHA256, THRESHOLD, load_model, predict_subject
from validation import validate_uploads
from explainability import TARGET_LAYER_NAME, gradcam, heatmap_rgb, overlay_rgb
from logic import report_bytes

CHECKPOINT = INFERENCE_DIR / EXPECTED_CHECKPOINT_NAME; AUROC, VALIDATION_SUBJECTS = .6975, 40
SLICE_NAMES = ("z052", "z061", "z070", "z079", "z088", "z096", "z105", "z114", "z122")
st.set_page_config(page_title="NeuroPath AI · v004", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{--panel:rgba(14,29,52,.8);--line:rgba(110,220,255,.2);--cyan:#62e6ff;--muted:#9cb1c8}
.stApp{background:radial-gradient(circle at 78% 0%,#16264b 0%,transparent 34%),linear-gradient(145deg,#040b15,#071424 55%,#0b1022);color:#edf8ff;font-family:Inter,Segoe UI,sans-serif}
.block-container{max-width:1450px;padding:2rem 2.5rem 4rem}.glass{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:1.25rem;box-shadow:0 14px 50px #0003}.eyebrow{color:var(--cyan);font-family:monospace;font-size:.72rem;letter-spacing:.13em;text-transform:uppercase}.notice{border:1px solid #efb7ff66;background:#6d3c8140;border-radius:12px;padding:.8rem 1rem;color:#ffdafa}.mono{font-family:monospace;font-size:.72rem;overflow-wrap:anywhere;color:#beeefe}
[data-testid="stSidebar"]{background:#050d19;border-right:1px solid var(--line)} div[data-testid="stMetric"]{background:#0b1a2c;border:1px solid var(--line);border-radius:14px;padding:1rem}
</style>""", unsafe_allow_html=True)
if "uploads" not in st.session_state: st.session_state.uploads = None
if "result" not in st.session_state: st.session_state.result = None
if "cams" not in st.session_state: st.session_state.cams = None
with st.sidebar:
    st.markdown('<div class="eyebrow">NeuroPath AI · v004</div>', unsafe_allow_html=True)
    page = st.radio("Workspace", ["Overview", "MRI Workspace", "AI Analysis", "Results", "Model Information", "Research Limitations"], label_visibility="collapsed")
    st.divider(); st.success("Checkpoint verified" if CHECKPOINT.is_file() else "Checkpoint missing")
    st.caption("Local-only · no telemetry · no image persistence")
    if st.button("Reset analysis", use_container_width=True):
        st.session_state.uploads = st.session_state.result = st.session_state.cams = None; st.rerun()
st.markdown('<div class="eyebrow">Exploratory neuroimaging workstation · explainability v004</div><h1>NeuroPath AI</h1>', unsafe_allow_html=True)
st.markdown('<div class="notice">Research use only — not for medical diagnosis. Anatomical laterality remains UNVERIFIED.</div>', unsafe_allow_html=True)

if page == "Overview":
    a,b,c = st.columns(3)
    a.metric("Selected model", "CNN v003"); b.metric("Validation AUROC", f"{AUROC:.4f}"); c.metric("Grad-CAM target", TARGET_LAYER_NAME)
    st.info("Workflow: upload nine training-partition slices → validate → run the frozen CNN → inspect exploratory explanatory maps.")
elif page in ("MRI Workspace", "AI Analysis", "Results"):
    st.subheader("MRI Workspace")
    uploaded = st.file_uploader("Drop exactly nine original-view 208×208 grayscale PNG slices", type=["png"], accept_multiple_files=True)
    if uploaded:
        try: ordered = validate_uploads(uploaded)
        except ValueError as exc: st.error(f"Validation blocked: {exc}"); ordered = None
        if ordered:
            st.session_state.uploads = ordered; subject_id = ordered[0][0].split("_MR1_")[0]
            st.success(f"Validated training subject {subject_id} · sequence locked")
            cols = st.columns(3)
            for i,(name,image) in enumerate(ordered):
                with cols[i%3]: st.image(image, caption=name, use_container_width=True)
            if st.button("Run verified CNN and prepare explanation", type="primary", use_container_width=True):
                try:
                    if not CHECKPOINT.is_file(): raise FileNotFoundError(f"Verified checkpoint missing: {CHECKPOINT}")
                    with st.spinner("Verifying SHA-256 and loading CNN on CPU…"): model = load_model(CHECKPOINT, device="cpu")
                    result = predict_subject(model, [image for _,image in ordered], subject_id=subject_id)
                    cams = [gradcam(model, image) for _,image in ordered]
                    if [round(x[1],8) for x in cams] != [round(x,8) for x in result["slice_positive_probabilities"]]:
                        raise RuntimeError("Explainability forward scores did not match prediction scores")
                    st.session_state.result, st.session_state.cams = result, cams
                    st.success("Analysis complete · real model outputs and Grad-CAM maps ready")
                except Exception as exc: st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
    result, cams, ordered = st.session_state.result, st.session_state.cams, st.session_state.uploads
    if page in ("AI Analysis", "Results") and result and cams and ordered:
        st.subheader("Subject-level result")
        a,b,c = st.columns(3); a.metric("CDR score", f"{result['cdr_score']:.4f}"); b.metric("Predicted class", f"Class {result['prediction']}"); c.metric("Threshold", f"{THRESHOLD:.1f}")
        st.caption("The score is the arithmetic mean of nine positive-class outputs; it is not clinical confidence.")
        st.bar_chart({"Positive-class output": result["slice_positive_probabilities"]}, height=220)
        st.subheader("Explain prediction")
        st.info("These are regions that most influenced the model prediction. They are an exploratory model explainability map, not a validated lesion localization. Anatomical laterality is UNVERIFIED.")
        idx = st.slider("Browse slice", 0, 8, 0, format="%d · " + SLICE_NAMES[0]); opacity = st.slider("Overlay opacity", .1, 1., .55, .05)
        tab_original, tab_heatmap, tab_overlay = st.tabs(["Original", "Heatmap", "Overlay"])
        heat, _ = cams[idx]; original = ordered[idx][1]
        with tab_original: st.image(original, caption=ordered[idx][0], use_container_width=True)
        with tab_heatmap: st.image(heatmap_rgb(heat), caption=f"Exploratory map · {ordered[idx][0]}", use_container_width=True)
        with tab_overlay: st.image(overlay_rgb(original, heat, opacity), caption=f"Overlay · opacity {opacity:.2f}", use_container_width=True)
        st.download_button("Download verified research report", report_bytes(result, result["subject_id"]), "neuropath-v004-research-report.json", "application/json")
elif page == "Model Information":
    st.subheader("Model Information"); st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("**Architecture:** ImageClassifier2D / image2d_v1 · **Variant:** horizontal_flip · **Epoch:** 18")
    st.write(f"**Validation AUROC:** {AUROC:.4f} on {VALIDATION_SUBJECTS} validation subjects")
    st.write(f"**Grad-CAM target:** `{TARGET_LAYER_NAME}` (final convolution before the third max-pool and deterministic grid pool)")
    st.markdown(f'<div class="mono">SHA-256 · {EXPECTED_CHECKPOINT_SHA256}</div>', unsafe_allow_html=True); st.markdown('</div>', unsafe_allow_html=True)
elif page == "Research Limitations":
    st.subheader("Research Limitations"); st.warning("Research use only — not for medical diagnosis.")
    st.write("Grad-CAM highlights spatial regions that most influenced the selected class score under this model and input preprocessing. It does not prove anatomy, causality, lesion presence, disease mechanism, or clinical utility. Maps can be diffuse, resolution-limited, and sensitive to model shortcuts. No laterality claim, clinical diagnosis, or validated lesion localization is supported.")
