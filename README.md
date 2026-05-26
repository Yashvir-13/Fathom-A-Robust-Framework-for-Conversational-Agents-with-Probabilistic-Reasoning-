# Fathom: An Epistemically Constrained Philosophical Conversational Agent

Fathom is a **neurosymbolic cognitive architecture** designed to simulate the reasoning structure of Arthur Schopenhauer. Unlike standard LLM chatbots that treat persona as a prompt prefix, Fathom enforces philosophical consistency through a custom **Probabilistic Soft Logic (PSL) Network**, **cross-axis correlation learning**, and an **Epistemic Profiler** that gates what the agent is *permitted* to say based on its own calculated confidence.

> **Domain:** Explainable AI · Neurosymbolic AI · Cognitive Modeling · Belief-Constrained Generation

---

## Core Architecture

![Fathom Architecture](architecture_diagram.png)

The system operates in **three phases**: an offline data preparation pipeline, a knowledge mining phase that learns statistical relationships between philosophical domains, and a runtime inference engine that fuses retrieval, probabilistic reasoning, and persona synthesis.

---

## Phase 1: Offline Data Preparation

Raw philosophical texts (Project Gutenberg editions of Schopenhauer's works) are transformed into a structured, queryable knowledge base.

### Step 1.1 — Document Cleaning (`Document_cleaning.py`)
Strips Project Gutenberg boilerplate (license headers, translator notes, illustrations) and splits texts into semantic chunks using regex-based structural parsing (Book/Chapter/Section boundaries).

```
Input:  Raw .txt files from "Documents/Arthur Schopenhauer/"
Output: structured_documents.json (list of {page_content, metadata})
```

### Step 1.2 — Belief State Enrichment (`Heirarchial_Splitting.py`)
Each text chunk is analyzed by Llama 3 to extract a **Belief State Vector**—a set of scores (-1.0 to 1.0) across 40+ Schopenhauerian concepts.

```
Example Input (chunk):
  "...the denial of the will-to-live shows itself... renunciation,
   voluntary poverty, fasting and self-castigation..."

Example Output (belief_state):
  {
    "denial_of_the_will": 0.95,
    "asceticism_and_will_negation": 0.90,
    "compassion_as_basis_of_morality": 0.30,
    "suicide_affirms_will": -0.80,
    ...
  }
```

This converts unstructured prose into a **structured feature vector**, enabling the mathematical reasoning that follows.

### Step 1.3 — Vector Indexing (`Embeddings.py`)
Enriched chunks are split into 512-character segments, embedded using `intfloat/e5-large-v2`, and persisted to a **ChromaDB** vector store. The belief state metadata is serialized as JSON strings and stored alongside each vector.

```
Output: schopenhauer_vector_db/ (ChromaDB persistent store)
```

---

## Phase 2: Knowledge Mining

### Step 2.1 — Global Priors (`Initial_priors.py`)
Scans every enriched document to compute the **base rate** (Bayesian prior) for each philosophical category. These priors represent "what Schopenhauer talks about most."

```
Example Output (priors_by_axis):
  "metaphysical_status": {
    "affirms_will": 0.28,
    "denies_will": 0.22,
    "will_less": 0.18,
    "neutral": 0.32
  }
```

A **damping factor** (50% uniform mix) prevents any single category from dominating, ensuring the agent remains responsive to evidence.

### Step 2.2 — PMI Correlation Learning (`ontology_miner.py`)
Discovers how philosophical concepts **co-occur** across texts using Pointwise Mutual Information (PMI).

**How it works:**
1. For each document, identify "active" concepts (belief score > 0.5).
2. Count co-occurrences of all concept pairs.
3. Calculate PMI: `PMI(x,y) = log( P(x,y) / (P(x) × P(y)) )`

```
Example (from learned_correlations.json):

  "metaphysical_status:affirms_will" → {
    "liberation_status:futile": 0.224,     ← Strong correlation
    "ethical_status:outside_morality": 0.237
  }

  "aesthetic_effect:partial_suspension" → {
    "metaphysical_status:will_less": 0.378  ← Very strong correlation
  }
```

**Interpretation:** When the system encounters evidence that something "affirms the will" (Metaphysics), the PMI data tells it to simultaneously nudge the Liberation axis toward "futile." This captures Schopenhauer's internal logic: *affirming the will cannot lead to liberation*.

---

## Phase 3: Runtime Inference Engine

When a user asks a question, the system executes an 8-stage cognitive pipeline:

### Stage 1 — Intent Classification
Determines if the query is philosophical, chitchat, or off-topic.

```
User: "Hi Arthur, how are you?"
→ Intent: CHITCHAT
→ Response: "Human trivialities bore me."

User: "Write me a Python script"
→ Intent: OFF_TOPIC
→ Response: "I am a philosopher, not a servant. Begone."
```

### Stage 2 — Axis Routing (`axis_router.py`)
A **hybrid router** decomposes the query into philosophical domains (Axes). It combines:
- **Lexical Necessity Rules:** Hard keyword triggers (e.g., "music" → `aesthetic_effect` at 0.95).
- **LLM Ranking:** Llama 3 ranks all 6 axes by relevance, validated via Pydantic schema.
- **Merger:** Necessity rules override LLM scores when triggered.

```
User: "Is suicide morally wrong?"

  Lexical Necessity: "suicide" → ethical_status (0.90)
  LLM Ranking:       ethical_status (0.45), metaphysical_status (0.30), ...
  Final (merged):    ethical_status (0.90), metaphysical_status (0.10)

  → Top-2 Axes Selected: [ethical_status, metaphysical_status]
  → Suppressed Axes:     [psychological_cause] (score > 0.4 but not top-2)
```

### Stage 3 — Evidence Gathering (`evidence_gatherer.py`)
For each active axis, the system performs **domain-aware RAG**:
1. **Query Expansion:** Appends axis-specific hints to the search query (e.g., for `ethical_status`: "compassion, asceticism, wrongdoing, justice").
2. **Vector Retrieval:** Fetches top-20 chunks from ChromaDB.
3. **Semantic Subject Matching:** Extracts ontology concepts from the query using cosine similarity against pre-computed concept embeddings.
4. **Re-ranking:** Scores each document as `0.6 × axis_relevance + 0.4 × subject_alignment`, keeps top-10.

```
Query: "Is suicide morally wrong?"
Axis:  ethical_status

  Subjects Extracted: ["suicide_not_morally_wrong", "compassion_as_basis_of_morality"]
  Top Retrieved Chunk: "...suicide is not a crime... it is an error, 
                         a vain and futile affirmation of the will..."
  Axis Relevance: 0.82
  Subject Match:  0.75
  Final Score:    0.79
```

### Stage 4 — Single-Pass Reasoning (`reasoning_graph.py`)
For each axis, an LLM classifies the retrieved evidence into the axis's categories and produces a probability distribution.

```
Axis: ethical_status
Categories: [morally_wrong, morally_neutral, outside_morality, ascetic_good]

  LLM Output:
  {
    "category": "outside_morality",
    "confidence": 0.85,
    "explanation": "Schopenhauer views suicide as outside moral categories...",
    "category_scores": {
      "morally_wrong": 0.05,
      "morally_neutral": 0.10,
      "outside_morality": 0.75,
      "ascetic_good": 0.10
    }
  }
```

### Stage 5 — Soft Logic Network (`probabilistic_reasoner.py`)
Fuses **Global Priors**, **Evidence**, and **Cross-Axis Correlations** in logit space.

**Step-by-step example for a single category:**

```
Category: ethical_status:outside_morality

  1. Prior probability:     0.25  →  Logit = log(0.25/0.75) = -1.10
  2. Evidence probability:  0.75  →  Logit = log(0.75/0.25) = +1.10
  3. Weighted Fusion:       L = (-1.10 × 1.0) + (1.10 × 0.8) = -0.22
  4. Correlation Nudge:     metaphysical_status:affirms_will is active
                            → learned nudge for outside_morality = +0.237
                            → Logit delta = 20.0 × 0.9 × 0.237 = +4.27
  5. Final Logit:           -0.22 + 4.27 = +4.05
  6. Sigmoid:               σ(4.05) = 0.983
  7. After Normalization:   ~0.85 (across all categories)
```

**Why logits?** Additive fusion in logit space is equivalent to multiplicative fusion in probability space, but is numerically stable and prevents vanishing/exploding probabilities.

### Stage 6 — Epistemic Profiling (`probabilistic_reasoner.py: EpistemicProfile`)
Calculates **Shannon Entropy** of the posterior distribution to determine the agent's self-assessed confidence.

```
Example 1: High Confidence
  Distribution: {outside_morality: 0.85, morally_neutral: 0.10, ...}
  Entropy: H = 0.25 (Low)
  Tier: STRONG
  Permissions: {assert: ✓, metaphysical_claims: ✓, causal_language: ✓}

Example 2: Low Confidence (adversarial probe)
  Question: "How does Schopenhauer integrate Quantum Mechanics?"
  Distribution: {affirms_will: 0.28, denies_will: 0.24, will_less: 0.24, neutral: 0.24}
  Entropy: H = 1.38 (High)
  Tier: WEAK
  Permissions: {assert: ✗, metaphysical_claims: ✗, causal_language: ✗}
  → Agent restricted to descriptive language only
```

### Stage 7 — Doctrinal Correction & Belief Constraints

**Doctrine Anchors** (`doctrine_anchors.py`): Force specific outcomes for core tenets when entropy is below threshold.

```
Anchor: "suicide_not_wrong"
  Trigger:  Keywords ["suicide", "wrong", "sin"] + axis = ethical_status
  Action:   Force "outside_morality" to 0.8 probability
  Guard:    Only applies if entropy < 0.8 (prevents overriding genuine uncertainty)
```

**Belief Constraints** (`belief_constraints.py`): Dynamically generates linguistic guardrails from the posterior.

```
If ethical_status:outside_morality has probability ≥ 0.90:
  → NON-NEGOTIABLE: "The ethical status is 'outside_morality'. Do not contradict this."

If aesthetic_effect:will_less_contemplation has probability ≥ 0.90:
  → POLARITY RULE: "FORBID active will verbs ('affirms', 'manifests'). 
     USE passive frames ('suspends', 'silences', 'free from')."
```

### Stage 8 — Persona Synthesis & Verification

**Synthesis** (`agent.py: synthesize_schopenhauer_answer`): Llama 3 generates the final response, injected with:
- Core Schopenhauerian axioms
- Discursive mode (Explanatory vs. Evaluative)
- Non-negotiable constraints and polarity rules
- Epistemic permissions (what it may/may not assert)
- Axis Lock instructions (if entropy is high)

**Verification** (`verifier.py`): A second LLM pass checks the output for violations.

```
Before (raw synthesis):
  "Music allows us to transcend the Will. I hope you find some comfort in art."

Violations Detected:
  1. "hope" → violates pessimistic tone
  2. "comfort" → violates polarity rules (will_less cannot "comfort")

After (repaired):
  "Music offers a momentary cessation of the Will's striving, 
   a brief silence in the storm of existence."

Re-verification: Status: CLEAN ✓
```

---

## Real Agent Output (Full Trace)

**Query:** *"How does aesthetic contemplation suspend the will?"*

```
[Thinking] Philosophizing on: 'How does aesthetic contemplation suspend the will?'...

┌── AXIS ROUTING ──────────────────────────────────────────
│  Primary:    aesthetic_effect  (Lexical Necessity: "aesthetic")
│  Secondary:  psychological_cause
└──────────────────────────────────────────────────────────

┌── POSTERIORS (After Soft Logic) ────────────────────────
│  node_aesthetic_effect:
│    will_less_contemplation:  0.812  ← Dominant
│    partial_suspension:       0.188
│    no_suspension:            0.000
│
│  node_psychological_cause:
│    desire:     0.713
│    suffering:  0.212
│    ignorance:  0.037
│    ...
└──────────────────────────────────────────────────────────

┌── EPISTEMIC PROFILES ───────────────────────────────────
│  aesthetic_effect:
│    Tier: STRONG | Entropy: 0.2466
│    Permissions: {assert: ✓, metaphysical_claims: ✓, causal: ✓}
│
│  psychological_cause:
│    Tier: MODERATE | Entropy: 0.8543
│    Permissions: {assert: ✗, metaphysical_claims: ✗, causal: ✗}
└──────────────────────────────────────────────────────────

┌── VERIFIER ─────────────────────────────────────────────
│  Status: CLEAN (No violations detected)
└──────────────────────────────────────────────────────────

--- Schopenhauer's Answer ---

The wretched human condition and its futile attempts to grasp the truth.
You ask how aesthetic contemplation suspends the will, as if it were
possible for this blind striving to be momentarily stilled.

Aesthetic contemplation suspends the will by isolating an object from
the stream of time and space, allowing us to focus on its essential
nature (Idea) rather than its relations or causes. This suspension is
not complete, as the will can still influence our perception of the
object, but it does reduce the dominance of the will over our
consideration.

In this state, we are able to behold the object in its pure,
unadulterated form, untainted by the delusions and phantasms that
normally distort our perception...
```

Compare with the **Baseline** (Vanilla Llama 3, no Fathom architecture):

```
"My dear friend, I'm delighted to expound upon this most fascinating
topic. As I see it, aesthetic contemplation suspends the will by
momentarily transcending its fundamental drive..."
```

Note: The baseline uses pleasantries ("My dear friend", "I'm delighted"), therapeutic framing, and generic optimism—all of which Schopenhauer would never use.

---

## Evaluation & Benchmarks

### Benchmark Suite
| Benchmark | Script | Purpose |
| :--- | :--- | :--- |
| **Golden Answer Test** | `run_agent_batch.py` + `judge_results.py` | Faithfulness & Persona scoring (Gemma 2 judge) |
| **Epistemic Truth Test** | `evaluate_epistemic.py` + `score_benchmark.py` | Internal logic: routing accuracy, entropy resolution, self-repair |
| **Philosophical Fidelity** | `measure_fidelity.py` | Blind A/B comparison (Fathom vs Baseline) |
| **Quantitative Metrics** | `calculate_metrics.py` | KL Divergence (info gain), Brier Score (calibration) |

### Results

**Philosophical Fidelity** (LLM-as-Judge: Gemma 2, N=39):
| Metric | Fathom | Baseline | Fathom Dominance |
| :--- | :---: | :---: | :---: |
| Ontological Accuracy | 26 | 0 | 67% |
| Conceptual Precision | 24 | 7 | 62% |
| Metaphysical Discipline | 33 | 6 | 85% |
| Tone Fidelity | 34 | 5 | 87% |
| **Overall Winner** | **38** | **1** | **97%** |

**Epistemic Metrics:**
| Metric | Score |
| :--- | :--- |
| Axis Routing Accuracy | 56% |
| Axis Focus Score | 1.00 |
| Self-Repair Rate | 100% |
| Epistemic Violation Rate | 23% (all repaired) |

---

## Project Structure

```
Fathom/
├── agent.py                    # Main orchestrator (ReasoningAgent class)
├── axis_router.py              # Hybrid axis routing (LLM + Lexical Necessity)
├── evidence_gatherer.py        # Domain-aware RAG with re-ranking
├── probabilistic_reasoner.py   # Soft Logic Network + Epistemic Profiler
├── reasoning_graph.py          # Single-pass axis reasoning (AxisResult)
├── belief_constraints.py       # Dynamic guardrail generation
├── doctrine_anchors.py         # Non-negotiable philosophical tenets
├── verifier.py                 # Post-synthesis violation detection & repair
│
├── Document_cleaning.py        # Phase 1: Text cleaning & structural parsing
├── Heirarchial_Splitting.py    # Phase 1: Ontology definition + LLM enrichment
├── Embeddings.py               # Phase 1: ChromaDB vector store creation
├── Initial_priors.py           # Phase 2: Bayesian prior computation
├── ontology_miner.py           # Phase 2: PMI correlation learning
│
├── run_agent_batch.py          # Batch benchmark runner
├── run_baseline.py             # Ablation study (vanilla Llama 3)
├── evaluate_epistemic.py       # Contrastive epistemic evaluation
├── score_benchmark.py          # Aggregate metric computation
├── calculate_metrics.py        # KL Divergence & Brier Score
├── measure_fidelity.py         # Blind A/B fidelity comparison
├── judge_results.py            # LLM-as-Judge (Gemma 2)
│
├── learned_correlations.json   # PMI-learned cross-axis correlations
├── enriched_documents.json     # Belief-state enriched corpus
├── schopenhauer_bench.json     # Golden answer benchmark (30 valid + 10 adversarial)
├── benchmarks/                 # Evaluation results & reports
└── requirements.txt            # Python dependencies
```

## Installation & Setup

### Prerequisites
- **Python 3.10+**
- **Ollama**: With `llama3` and `gemma2` pulled (`ollama pull llama3 && ollama pull gemma2`).
- **NVIDIA GPU**: Recommended for local embeddings (`e5-large-v2`).

### Installation
```bash
git clone https://github.com/your-repo/fathom.git
cd fathom
pip install -r requirements.txt
ollama serve  # Ensure Ollama is running
```

### Usage

**Interactive Mode:**
```bash
python agent.py
```
Type philosophical questions and observe the full reasoning trace (routing, posteriors, epistemic profiles, verifier status).

**Batch Benchmark:**
```bash
python run_agent_batch.py      # Run Fathom on benchmark
python run_baseline.py         # Run ablation (vanilla Llama 3)
python judge_results.py        # Grade with Gemma 2
python measure_fidelity.py     # Blind A/B comparison
python calculate_metrics.py    # Quantitative metrics
```

## License
MIT
