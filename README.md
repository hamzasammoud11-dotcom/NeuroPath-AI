# NeuroPath-AI
NeuroPath AI is a professional-grade, multimodal diagnostic platform designed to assist neurologists in the early detection and progression tracking of Alzheimer’s Disease.

## Brain Stack (Phase 1)
This repo now contains the first pass of the AI brain scaffold: multimodal data loading, imaging model, fusion head, and a simple training loop for smoke testing.

### Project Structure
- src/neuro_path_ai: Core package
- src/neuro_path_ai/data: Dataset loaders and manifest helpers
- src/neuro_path_ai/models: Imaging, fusion, and time-series modules
- src/neuro_path_ai/training: Training loop and CLI
- configs: Experiment configuration
- scripts: Entry points for local runs

### Quickstart (Smoke Test)
1. Create a Python environment and install dependencies:
	- pip install -r requirements.txt
	- pip install -e .
2. Run the synthetic training loop:
	- python scripts/train_brain.py --synthetic

### Checkpoints
Use the checkpoint-only helper to move the model files between PCs:

```bash
python scripts/checkpoints.py pull
```

To send updated checkpoints from the current machine:

```bash
python scripts/checkpoints.py push --message "Update checkpoints"
```

### Dashboard + API
1. Install the local package so `neuro_path_ai` is importable:
	- pip install -e .
   If you do not want to install it, make sure `src` is on `PYTHONPATH`.
	- Windows: `set PYTHONPATH=src`
	- macOS/Linux: `export PYTHONPATH=src`
2. Start the FastAPI server (optional if using localhost auto-start):
	- python -m uvicorn neuro_path_ai.api.app:app --host 0.0.0.0 --port 8000
3. Launch the Streamlit dashboard:
	- streamlit run src/neuro_path_ai/ui/dashboard.py
4. When the API URL is set to http://localhost:8000, the dashboard will auto-start the API if it is not already running.

### Next Steps
- Wire real ADNI manifests into data loaders
- Replace synthetic MRI loader with nibabel/monai pipeline
- Add explainability hooks (Grad-CAM/SHAP)
