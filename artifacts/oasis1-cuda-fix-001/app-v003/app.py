"""NeuroPath AI v003 futuristic local Streamlit research workstation."""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = ROOT / "inference-v003"
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(ROOT / "app-v002"))
from inference import EXPECTED_CHECKPOINT_NAME, EXPECTED_CHECKPOINT_SHA256, THRESHOLD, load_model, predict_subject
from validation import validate_uploads
from logic import preview_image, report_bytes

CHECKPOINT = INFERENCE_DIR / EXPECTED_CHECKPOINT_NAME
AUROC, VALIDATION_SUBJECTS = 0.6975, 40
SLICE_NAMES = ("z052", "z061", "z070", "z079", "z088", "z096", "z105", "z114", "z122")

st.set_page_config(page_title="NeuroPath AI · v003", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{--ink:#07111f;--panel:rgba(14,29,52,.78);--line:rgba(110,220,255,.19);--cyan:#62e6ff;--violet:#a78bfa;--muted:#9cb1c8}
.stApp{background:radial-gradient(circle at 78% 0%,#16264b 0%,transparent 34%),linear-gradient(145deg,#040b15 0%,#071424 52%,#0b1022 100%);color:#edf8ff;font-family:Inter,Segoe UI,sans-serif}
.block-container{max-width:1450px;padding:2rem 2.5rem 4rem}.glass{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:1.25rem;box-shadow:0 14px 50px rgba(0,0,0,.18)}
.eyebrow{font-family:'DM Mono',monospace;color:var(--cyan);font-size:.72rem;letter-spacing:.13em;text-transform:uppercase}.hero{padding:1.2rem 0 1.6rem}.hero h1{font-size:2.7rem;letter-spacing:-.06em;margin:.35rem 0}.hero p{color:var(--muted);max-width:720px}.notice{border:1px solid #efb7ff66;background:#6d3c8140;border-radius:12px;padding:.8rem 1rem;color:#ffdafa}.step{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0}.step span{border:1px solid var(--line);border-radius:999px;padding:.45rem .7rem;color:var(--muted);font-size:.82rem}.step .active{border-color:var(--cyan);color:var(--cyan);background:#62e6ff12}.metric-label{color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.08em}.metric-value{font-size:1.7rem;font-weight:800}.mono{font-family:'DM Mono',monospace;font-size:.75rem;overflow-wrap:anywhere;color:#beeefe}
[data-testid="stSidebar"]{background:#050d19;border-right:1px solid var(--line)}
div[data-testid="stMetric"]{background:#0b1a2c;border:1px solid var(--line);border-radius:14px;padding:1rem}
</style>
""", unsafe_allow_html=True)

if "result" not in st.session_state: st.session_state.result = None
if "ordered" not in st.session_state: st.session_state.ordered = None

with st.sidebar:
    st.markdown('<div class="eyebrow">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Workspace", ["Overview", "MRI Workspace", "AI Analysis", "Results", "Model Information", "Research Limitations"], label_visibility="collapsed")
    st.divider(); st.markdown('<div class="eyebrow">System status</div>', unsafe_allow_html=True)
    st.success("Checkpoint verified" if CHECKPOINT.is_file() else "Checkpoint missing")
    st.caption("Local-only mode · no telemetry · no image persistence")
    if st.button("Reset analysis", use_container_width=True):
        st.session_state.result = None
        st.session_state.ordered = None
        st.rerun()

st.markdown('<div class="hero"><div class="eyebrow">Neuroimaging research workstation · v003</div><h1>NeuroPath AI</h1><p>Explore a frozen OASIS-1 CNN workflow with strict partition controls and transparent subject-level aggregation.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="notice">Research use only — not for medical diagnosis. Anatomical laterality is UNVERIFIED.</div>', unsafe_allow_html=True)

if page == "Overview":
    st.markdown('<div class="step"><span class="active">01 Upload MRI</span><span>02 Validate</span><span>03 Run CNN</span><span>04 View results</span></div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a: st.markdown('<div class="glass"><div class="metric-label">Selected model</div><div class="metric-value">CNN v003</div><p>horizontal_flip · epoch 18</p></div>', unsafe_allow_html=True)
    with b: st.markdown(f'<div class="glass"><div class="metric-label">Validation performance</div><div class="metric-value">{AUROC:.4f}</div><p>AUROC on {VALIDATION_SUBJECTS} validation subjects</p></div>', unsafe_allow_html=True)
    with c: st.markdown('<div class="glass"><div class="metric-label">Inference policy</div><div class="metric-value">9 → 1</div><p>Mean positive-class output · threshold 0.5</p></div>', unsafe_allow_html=True)
    st.info("Use the MRI Workspace to upload one training-partition subject and begin the guarded workflow.")

if page in ("MRI Workspace", "AI Analysis", "Results"):
    st.markdown('<div class="step"><span class="active">01 Upload MRI</span><span class="active">02 Validate</span><span class="active">03 Run CNN</span><span class="active">04 View results</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("MRI Workspace")
    uploaded = st.file_uploader("Drop exactly nine original-view PNG slices", type=["png"], accept_multiple_files=True, help="208×208 grayscale; one frozen training subject; z052 through z122.")
    if uploaded:
        try: ordered = validate_uploads(uploaded)
        except ValueError as exc: st.error(f"Validation blocked: {exc}"); ordered = None
        if ordered:
            st.session_state.ordered = ordered; subject_id = ordered[0][0].split("_MR1_")[0]
            st.success(f"Validated training subject {subject_id} · 9/9 slices · sequence locked")
            left, right = st.columns([1,2])
            with left:
                selected = st.slider("Slice explorer", 0, 8, 0, format="%d · " + SLICE_NAMES[0])
                brightness = st.slider("Display brightness", 0.5, 1.8, 1.0, .05)
                contrast = st.slider("Display contrast", 0.5, 1.8, 1.0, .05)
                st.caption("Brightness and contrast affect this preview only. CNN tensors always use frozen preprocessing.")
                st.image(preview_image(ordered[selected][1], brightness, contrast), caption=ordered[selected][0], use_container_width=True)
            with right:
                cols = st.columns(3)
                for i,(name,image) in enumerate(ordered):
                    with cols[i%3]: st.image(image, caption=name, use_container_width=True)
            if st.button("Run verified CNN", type="primary", use_container_width=True):
                try:
                    if not CHECKPOINT.is_file(): raise FileNotFoundError(f"Verified checkpoint missing: {CHECKPOINT}")
                    with st.spinner("Verifying SHA-256 and loading CNN on CPU…"): model = load_model(CHECKPOINT, device="cpu")
                    st.session_state.result = predict_subject(model, [image for _,image in ordered], subject_id=subject_id)
                    st.success("Analysis complete · real model output")
                except Exception as exc: st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
    else: st.caption("No images uploaded. The checkpoint is never loaded until validation succeeds.")
    st.markdown('</div>', unsafe_allow_html=True)
    result = st.session_state.result
    if page in ("Results", "AI Analysis") and result:
        st.subheader("Subject-level result")
        a,b,c = st.columns(3)
        a.metric("CDR score", f"{result['cdr_score']:.4f}")
        b.metric("Predicted class", f"Class {result['prediction']}")
        c.metric("Fixed threshold", f"{THRESHOLD:.1f}")
        st.caption("This score is the arithmetic mean of nine positive-class outputs; it is not clinical confidence.")
        st.line_chart({"Slice positive output": result["slice_positive_probabilities"]}, height=220)
        st.write("Class 0 = CDR 0. Class 1 = CDR > 0 (CDR 0.5, 1, or 2 in the frozen protocol). This does not diagnose Alzheimer’s disease.")
        st.download_button("Download verified research report (JSON)", report_bytes(result, result["subject_id"]), "neuropath-v003-research-report.json", "application/json")

if page == "Model Information":
    st.subheader("Model Information")
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("**Architecture:** ImageClassifier2D / image2d_v1 with deterministic 4×4 pooling on the guarded 16×16 feature map.")
    st.write("**Variant:** horizontal_flip · **epoch:** 18 · **validation AUROC:** 0.6975 on 40 validation subjects")
    st.write("**Preprocessing:** grayscale, bilinear resize to 128×128, scale to [0,1], mean 0.5, standard deviation 0.5.")
    st.markdown(f'<div class="mono">SHA-256 · {EXPECTED_CHECKPOINT_SHA256}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if page == "Research Limitations":
    st.subheader("Research Limitations")
    st.warning("This is an exploratory research prototype, not a medical device or diagnostic system.")
    st.write("The score is a model output under one frozen preprocessing and split protocol. Validation AUROC is a historical validation metric, not an estimate of clinical performance or confidence. Laterality remains unverified; no hemisphere-specific interpretation is supported. Test subjects are rejected by the frozen partition guard. No Grad-CAM, explainability map, telemetry, or external upload is implemented.")
