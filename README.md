# MedVision PACS — Neuro‑Symbolic Radiology Report Drafting (BiomedCLIP + RAG)

**Authors:** Samira Sadeghi, Daniel Noroozi

A PACS‑style Streamlit dashboard that takes a chest X‑ray, predicts **multi‑label** findings using **BiomedCLIP image embeddings**, retrieves **similar historical cases** (RAG), and drafts a radiology report that is **checked (and optionally auto‑revised)** against a small **knowledge graph** (neuro‑symbolic verification).

> **Not** a clinical device.

---

## Demo (Dashboard)

![Dashboard overview](assets/dashboard_page1.png)

![LLM revision trail + final report](assets/dashboard_page2.png)

A printable version is included as `assets/FinalDashboard.pdf`.

---

## What the project does (end‑to‑end)

1. **Dataset prep (Notebook)**
   - Loads Indiana CXR metadata (`indiana_projections.csv`, `indiana_reports.csv`) and resolves valid image paths.
   - Repairs common filename mismatches by scanning the image folder and keeping only valid files.
   - Builds the “caption” text by joining *Findings + Impression*.

2. **Embeddings + multi‑label classifier (Notebook)**
   - Computes **BiomedCLIP** image embeddings (ViT‑B/16 backbone).
   - Trains a **One‑vs‑Rest Random Forest** for multi‑label prediction over 8 labels:
     `Cardiomegaly, Pneumonia, Atelectasis, Edema, PleuralEffusion, Fracture, Pneumothorax, Normal`.
   - Calibrates per‑class probabilities with **sigmoid / Platt scaling** on a held‑out calibration split.
   - Tunes **per‑class decision thresholds** on a validation split (no test leakage).

3. **Neuro‑symbolic layer (Notebook + App)**
   - Constructs a small RDF knowledge graph (definitions, anatomy tree, disease→location hints).
   - Validates predicted labels and provides KG facts (definitions + expected locations).

4. **RAG evidence retrieval (App)**
   - Retrieves top‑K similar cases by cosine similarity in embedding space.
   - Highlights detected pathology terms inside the retrieved radiologist report (rule‑based entity mapping).

5. **Report drafting (App)**
   - Optionally calls a **local LLM** (DeepSeek‑R1 via Ollama) to draft a report from:
     predicted labels + retrieved case snippets + KG definitions.
   - Runs a **KG verifier** over the draft; if it violates constraints, the app attempts **automatic revision** (up to 2 passes) and shows a revision trail.

6. **Explainability (App)**
   - **CLIP embedding saliency** (not disease‑specific) to visualize what influenced the embedding.
   - **Label‑specific occlusion sensitivity** heatmap (slower, grid‑based).

---

## Quickstart

### 1) Install dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

> **GPU is optional.** If you have CUDA, PyTorch will use it automatically.

### 2) Put the dataset in the expected place

The notebook expects this structure (dataset is **not** included in this repo):

```
dataset/
  images/
    images_normalized/
      <image files...>
    indiana_projections.csv
    indiana_reports.csv
```

### 3) Run the notebook to generate the app artifacts

Open and run:

- `notebooks/AI_RAG_train_test_split_Final.ipynb`

It will generate several `.pkl` artifacts (dataset + embeddings + models + thresholds).

To keep the repo clean, move them into `./artifacts`:

```bash
python scripts/collect_artifacts.py
```

The Streamlit app is already configured to prefer `./artifacts/*.pkl` automatically.

### 4) Run the dashboard

```bash
streamlit run app/app.py
```

---

## Local LLM (optional): DeepSeek‑R1 via Ollama

If you want the **LLM report drafting** feature:

1. Install Ollama
2. Pull a model (example):
   ```bash
   ollama pull deepseek-r1
   ```
3. Make sure Ollama’s OpenAI‑compatible endpoint is reachable at:
   `http://localhost:11434/v1`

If the LLM is not available, the app falls back to a rule‑based report layout.

---

## Notes & limitations

- This is a course/portfolio project; outputs must be reviewed by qualified professionals.
- Embedding saliency ≠ diagnosis explanation; it shows sensitivity of the **embedding**, not a certified clinical rationale.
- RAG evidence is similarity‑based and may retrieve imperfect matches.

---

## Acknowledgements

- **BiomedCLIP** (Microsoft) via `open_clip`
- Indiana University chest X‑ray dataset (refer to the dataset’s original license/terms)
