# CRAG-MM: Adaptive Multimodal RAG with 5-Phase Pipeline

Fine-tuning **Llama 3.2 11B Vision** for smart-glasses question answering using the [CRAG-MM 2025](https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public) benchmark. The system implements a complete five-phase pipeline with adaptive retrieval planning and hallucination mitigation, evaluated across three system variants to answer two research questions.

---

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Research Questions & System Variants](#research-questions--system-variants)
- [Phase Details](#phase-details)
  - [Phase 1: Input Preprocessing](#phase-1-input-preprocessing)
  - [Phase 2: Query Analysis & Classification](#phase-2-query-analysis--classification)
  - [Phase 3: Evidence Retrieval](#phase-3-evidence-retrieval)
  - [Phase 4: Answer Generation](#phase-4-answer-generation)
  - [Phase 5: Hallucination Detection & Mitigation](#phase-5-hallucination-detection--mitigation)
- [Dataset](#dataset)
- [Model Configuration](#model-configuration)
- [Training Details](#training-details)
- [Evaluation Methodology](#evaluation-methodology)
- [Getting Started](#getting-started)
- [Notebook Structure](#notebook-structure)
- [Outputs & Artifacts](#outputs--artifacts)
- [Quick Inference](#quick-inference)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Pipeline Architecture

```
┌────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Phase 1   │───▶│   Phase 2    │───▶│    Phase 3    │───▶│   Phase 4    │───▶│     Phase 5     │
│   Input    │    │    Query     │    │   Evidence    │    │   Answer     │    │  Hallucination  │
│ Preprocess │    │ Classifier   │    │  Retrieval    │    │ Generation   │    │   Detection     │
└────────────┘    └──────────────┘    └───────────────┘    └──────────────┘    └─────────────────┘
  Image norm.       TF-IDF +           Simulated KG        LoRA fine-tuned      Confidence scoring
  Text cleaning     regex rules        + web search         Llama 3.2 11B       Self-consistency
  Metadata                                                  Vision               Mitigation
```

| Phase | Component | Description |
|-------|-----------|-------------|
| **Phase 1** | Input Preprocessing | Image normalisation (RGB, max 1280px), text cleaning, metadata extraction |
| **Phase 2** | Query Analysis | Classifies query type to select retrieval strategy and generation parameters |
| **Phase 3** | Evidence Retrieval | Simulated knowledge-graph and web-search retrieval using dataset-provided results |
| **Phase 4** | Answer Generation | LoRA fine-tuned VLM produces context-aware responses with query-adaptive decoding |
| **Phase 5** | Hallucination Detection | Linguistic confidence scoring, evidence grounding checks, and answer mitigation |

---

## Research Questions & System Variants

The notebook investigates two research questions by comparing three system variants:

- **RQ1:** Does adaptive retrieval planning (Phase 2) improve accuracy over a static baseline?
- **RQ2:** Does hallucination detection (Phase 5) reduce incorrect answers?

| Variant | Phase 2 (Query Classifier) | Phase 5 (Hallucination Detection) | Purpose |
|---------|:--------------------------:|:---------------------------------:|---------|
| **Baseline** | — | — | Control group |
| **Adaptive** | ✓ | — | Isolates RQ1 |
| **Adaptive + Mitigation** | ✓ | ✓ | Tests RQ1 + RQ2 |

All three variants share the same fine-tuned Phase 4 model; only the surrounding pipeline modules differ.

---

## Phase Details

### Phase 1: Input Preprocessing

Handles raw inputs before they enter the pipeline:

- **Image processing:** Converts to RGB, resizes images exceeding 1280px (preserving aspect ratio) using Lanczos resampling. Target resolution follows the CRAG-MM standard of 960 × 1280.
- **Text cleaning:** Normalises whitespace, strips control characters, and converts to string.
- **Metadata extraction:** Pulls `domain`, `timestamp`, and `source_type` from the sample for downstream context.

### Phase 2: Query Analysis & Classification

A two-stage classifier that routes queries into five types, each mapped to a distinct retrieval strategy and generation parameter set:

| Query Type | Examples | KG Retrieval | Web Search | Temperature |
|------------|----------|:------------:|:----------:|:-----------:|
| `simple_recognition` | "What brand is this?", "Read the text" | — | — | 0.05 |
| `simple_knowledge` | "How much does it cost?", "When was it released?" | ✓ | ✓ | 0.10 |
| `multi_hop` | "What other films did this director make?" | ✓ | ✓ | 0.15 |
| `comparison` | "Which is cheaper?", "Compare these two" | ✓ | ✓ | 0.10 |
| `reasoning` | "Is this compatible with…?", "Explain how it works" | ✓ | ✓ | 0.20 |

**Stage 1 — Rule-based:** Regex pattern matching against ~30 query templates per type.
**Stage 2 — ML:** TF-IDF vectoriser + Logistic Regression trained on rule-labelled training queries. Falls back to rule-based if fewer than 2 classes or 10 samples are available.

### Phase 3: Evidence Retrieval

Simulated retrieval that uses the search results provided in the CRAG-MM dataset:

- **Knowledge Graph retrieval:** Extracts up to 5 results from `search_results` / `kg_results` fields.
- **Web search retrieval:** Extracts up to 5 results from `web_results` / `web_search` fields.
- Retrieval is selectively activated based on the Phase 2 query-type strategy (e.g., `simple_recognition` queries skip retrieval entirely).

In a production deployment, these would be replaced by calls to real KG and web APIs.

### Phase 4: Answer Generation

The core VLM, fine-tuned with LoRA:

- **Base model:** `unsloth/Llama-3.2-11B-Vision-Instruct` loaded in 16-bit (bfloat16).
- **LoRA targets:** All attention projections (`q`, `k`, `v`, `o`) and MLP layers (`gate`, `up`, `down`) across both vision and language modules.
- **Adaptive decoding:** Temperature and `top_p` are adjusted per query type (set by Phase 2), ranging from very focused (0.05 / 0.85 for recognition) to exploratory (0.20 / 0.95 for reasoning).
- **System prompt:** `"You are an expert multimodal assistant for smart glasses. Answer questions about images accurately and concisely."`
- Responses are generated with `repetition_penalty=1.2` and `max_new_tokens=128`.

### Phase 5: Hallucination Detection & Mitigation

A post-generation filter that scores and optionally modifies answers:

- **Linguistic confidence scoring:** Counts uncertainty markers (e.g., "maybe", "probably", "I think") vs. confidence markers (e.g., "is", "shows", "the answer is") to produce a 0–1 score.
- **Refusal detection:** Recognises explicit refusals ("I don't know", "cannot determine") and treats them as confident abstentions (score = 1.0).
- **Evidence grounding:** Checks whether the generated answer references terms from retrieved evidence.
- **Mitigation:** When confidence falls below the threshold (default 0.7), the system can flag low-confidence answers, append caveats, or substitute with "I'm not sure" to reduce hallucinated incorrect responses.

---

## Dataset

The [CRAG-MM 2025](https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public) benchmark for multimodal question answering, designed for smart-glasses scenarios.

| Source | HuggingFace ID | Revision |
|--------|----------------|----------|
| Single-turn | `crag-mm-2025/crag-mm-single-turn-public` | `v0.1.2` |
| Multi-turn | `crag-mm-2025/crag-mm-multi-turn-public` | `v0.1.2` |

**Preprocessing pipeline:**
1. Both splits are loaded and feature-aligned (column types cast to match).
2. Concatenated into a single dataset; samples without images are filtered out.
3. Split 80 / 10 / 10 into train, validation, and test sets (seeded).
4. Training queries are extracted and labelled via rule-based classification to train the Phase 2 ML classifier.

**Data format per sample:**
- `image` — PIL Image
- `turns.query` — list of question strings (one per conversation turn)
- `answers.ans_full` — list of ground-truth answer strings
- `search_results` / `web_results` — pre-retrieved evidence (used by Phase 3)
- `domain` — question domain category

---

## Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `unsloth/Llama-3.2-11B-Vision-Instruct` | 11B parameter VLM |
| Precision | bfloat16 (16-bit) | Full 16-bit, no quantisation |
| LoRA rank (r) | 16 | |
| LoRA alpha (α) | 32 | α = 2r |
| LoRA dropout | 0.05 | |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | All attention + MLP layers |
| Vision layers fine-tuned | ✓ | Both vision and language modules |
| Gradient checkpointing | Unsloth optimised | Memory-efficient backprop |
| Max sequence length | 2048 tokens | |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Effective batch size | 16 (2 per device × 8 gradient accumulation) |
| Optimiser | AdamW 8-bit |
| Learning rate | 2e-4 |
| LR scheduler | Cosine decay |
| Warmup | 3% of max steps |
| Weight decay | 0.01 |
| Max training steps | 500 |
| Evaluation interval | Every 50 steps |
| Checkpoint interval | Every 100 steps |
| Best model selection | Lowest `eval_loss` |
| Training precision | bf16 |
| Trainer | TRL `SFTTrainer` with `UnslothVisionDataCollator` |

**Latency constraints:** 30 seconds max per query, 10 seconds per individual phase.

---

## Evaluation Methodology

### Scoring Function

Each prediction is scored against the ground truth using a multi-tier matching system:

| Score | Category | Matching Criteria |
|------:|----------|-------------------|
| **+1.0** | Perfect | Exact match, containment (either direction), or ≥40% keyword overlap with key entity match |
| **+0.5** | Acceptable | ≥20% keyword overlap, or ≥2 overlapping words with number match |
| **0.0** | Missing | Explicit refusal detected ("I don't know", "cannot determine", etc.) |
| **−1.0** | Incorrect | No meaningful overlap with ground truth |

Matching preprocesses both strings by lowercasing, stripping stop words, and extracting numbers and capitalised entities for targeted comparison.

### Aggregate Metrics

- **Truthfulness:** Mean score across all samples (range: −1.0 to +1.0)
- **Accuracy:** Fraction of responses scoring ≥ 0.5 (Perfect or Acceptable)
- **Average latency:** Mean wall-clock time per sample across the full pipeline
- **Category distribution:** Counts of Perfect / Acceptable / Missing / Incorrect

All three variants are evaluated on the same 50 test samples for a controlled comparison.

### Visualisations

The notebook produces the following evaluation charts:

- **Truthfulness comparison** — bar chart across all three variants
- **Accuracy comparison** — bar chart with target threshold
- **Category breakdown** — grouped bar chart (Perfect / Acceptable / Missing / Incorrect)
- **Visual prediction comparison** — side-by-side image + predictions from all variants with scoring labels
- **Before vs after training** — baseline model predictions compared with fine-tuned output

---

## Getting Started

### Requirements

- **Hardware:** Google Colab Pro+ with **A100 GPU (80 GB)** and High-RAM runtime
- **Python:** 3.10+

### Installation

```bash
pip install unsloth datasets transformers trl scikit-learn pillow matplotlib
```

### HuggingFace Authentication

The notebook downloads gated models. You'll need a HuggingFace access token:

```python
from huggingface_hub import login
login(token="hf_YOUR_TOKEN")
```

> ⚠️ **Security note:** Never commit tokens to version control. Use environment variables or Colab secrets instead.

### Running

1. Open the notebook in **Google Colab**.
2. Set the runtime to **A100 GPU** with **High-RAM**.
3. Run all cells sequentially — the notebook handles installation, training, evaluation, and export.

---

## Notebook Structure

| Section | Cells | Description |
|---------|-------|-------------|
| **1. Environment Setup** | 1–4 | OS config, dependency installation, GPU verification, HuggingFace auth |
| **2. Configuration** | 5 | `CRAGMMConfig` dataclass with all hyperparameters |
| **3. Phase 1** | 6 | `Phase1_InputPreprocessor` — image and text normalisation |
| **4. Phase 2** | 7 | `Phase2_QueryClassifier` — regex + TF-IDF/LogReg classification |
| **5. Phase 3** | 8 | `Phase3_EvidenceRetrieval` — simulated KG and web retrieval |
| **6. Phase 5** | 9 | `Phase5_HallucinationDetector` — confidence scoring and mitigation |
| **7. Model Loading** | 10–11 | Load Llama 3.2 11B Vision in 16-bit, attach LoRA adapters |
| **8. Dataset Prep** | 12–14 | Load CRAG-MM, 80/10/10 split, train query classifier, convert to chat format |
| **9. Training** | 15–17 | Baseline predictions, SFT training (500 steps), model saving |
| **10. Pipeline Assembly** | 18 | `CRAGMMPipeline` class wiring all five phases together |
| **11. Evaluation** | 19–23 | Three-variant evaluation, results tables, bar charts, visual prediction comparison |
| **12. Export** | 24–27 | Save JSON results, zip LoRA weights, download archive |
| **13. Inference API** | 25 | Standalone inference example for all three variants |

---

## Outputs & Artifacts

| Artifact | Description |
|----------|-------------|
| `crag_mm_lora/` | Fine-tuned LoRA adapter weights (saved via Unsloth) |
| `query_classifier.pkl` | Trained Phase 2 TF-IDF + LogReg classifier |
| `hallucination_detector.pkl` | Phase 5 hallucination detection model |
| `crag_mm_results.json` | Full evaluation results for all three variants |
| `crag_mm_complete.zip` | Downloadable archive of all output artifacts |

---

## Quick Inference

```python
from unsloth import FastVisionModel
from PIL import Image

# Load fine-tuned model
model, tokenizer = FastVisionModel.from_pretrained("crag_mm_lora")
FastVisionModel.for_inference(model)

# Build pipeline (full variant with all phases enabled)
pipeline = CRAGMMPipeline(
    model, tokenizer, config,
    enable_query_classifier=True,       # Phase 2
    enable_hallucination_detection=True, # Phase 5
)

# Run on a sample
result = pipeline(sample)
print(result["answer"])        # Generated answer
print(result["confidence"])    # Confidence score (0–1)
print(result["query_type"])    # Classified query type
print(result["total_latency"]) # End-to-end latency in seconds
```

---

## Tech Stack

| Library | Role |
|---------|------|
| [Unsloth](https://github.com/unslothai/unsloth) | Memory-efficient LoRA fine-tuning and fast inference for VLMs |
| [Transformers](https://huggingface.co/docs/transformers) | Model loading and tokenisation |
| [TRL](https://huggingface.co/docs/trl) | `SFTTrainer` for supervised fine-tuning |
| [Datasets](https://huggingface.co/docs/datasets) | HuggingFace CRAG-MM benchmark loading and splitting |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF vectorisation and Logistic Regression for query classification |
| [Matplotlib](https://matplotlib.org/) | Evaluation charts, prediction visualisations, before/after comparisons |
| [Pillow](https://pillow.readthedocs.io/) | Image loading and preprocessing |

---

## License

This project is for educational and research purposes.
