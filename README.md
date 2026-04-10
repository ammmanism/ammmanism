<div align="center">

<br/>
<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:161b22,100:0d1117&height=120&section=header&text=&animation=fadeIn"/>

```
░█████╗░███╗░░░███╗███╗░░░███╗░█████╗░███╗░░██╗
██╔══██╗████╗░████║████╗░████║██╔══██╗████╗░██║
███████║██╔████╔██║██╔████╔██║███████║██╔██╗██║
██╔══██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║██║╚████║
██║░░██║██║░╚═╝░██║██║░╚═╝░██║██║░░██║██║░╚███║
╚═╝░░╚═╝╚═╝░░░░╚═╝╚═╝░░░░╚═╝╚═╝░░╚═╝╚═╝░░╚══╝
```

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=15&duration=2500&pause=1000&color=58A6FF&center=true&vCenter=true&multiline=false&repeat=true&width=560&lines=AI+Engineer+%E2%80%94+I+build+systems+that+actually+ship.;LLM+%7C+RAG+%7C+MLOps+%7C+Transformers+%7C+Evaluation;Not+wrappers.+Not+tutorials.+Production+systems.;From+mathematical+first+principles+to+deployment." />

<br/>
<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ammmanism/)
[![Twitter](https://img.shields.io/badge/X%20%2F%20Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://twitter.com/ammmanism)
[![Email](https://img.shields.io/badge/ammanism@yahoo.com-6001D2?style=for-the-badge&logo=yahoo&logoColor=white)](mailto:ammanism@yahoo.com)
[![GitHub](https://img.shields.io/badge/GitHub-161b22?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ammmanism)

<br/>

![](https://img.shields.io/badge/India-remote%2C%20globally%20available-58A6FF?style=flat-square&labelColor=0d1117)
&nbsp;
![](https://img.shields.io/badge/status-building-238636?style=flat-square&labelColor=0d1117)
&nbsp;
![](https://img.shields.io/badge/open%20source-contributor-8957e5?style=flat-square&labelColor=0d1117)

<br/>
<br/>

</div>

---

<div align="center">

### `// who I am`

</div>

<br/>

I'm an AI engineer who builds from the ground up. Not from YouTube tutorials. Not from copy-pasted notebooks. From derivations — math first, then code, then production.

Every system I build is instrumented, benchmarked, and hardened for failure. I care about *why* things work, not just *that* they work. That means understanding backpropagation before touching PyTorch, understanding attention math before using Transformers, and understanding retrieval theory before plugging in a vector database.

My work sits at the intersection of research depth and engineering rigor. I don't ship demos. I ship systems.

<br/>

```python
class Amman:
    stack      = ["LLMs", "RAG", "MLOps", "Transformers", "Evaluation"]
    languages  = ["Python", "SQL", "Bash"]
    approach   = "derive → implement from scratch → harden → benchmark → ship"
    based_in   = "India"
    available  = "remote, globally"
    building   = True
```

<br/>

---

<div align="center">

### `// what I've shipped`

</div>

<br/>

<details>
<summary><strong>✅ &nbsp; ml-from-scratch &nbsp;·&nbsp; completed</strong></summary>

<br/>

> *10 ML algorithms. Pure NumPy. Zero sklearn. Every formula derived by hand.*

Before touching any framework, I sat down with the mathematics and built everything from scratch — linear models, kernel methods, ensemble methods, dimensionality reduction. Each algorithm comes with a full derivation document, visual comparisons against sklearn, and benchmarks proving identical outputs.

This repo exists to prove one thing: I understand the math, not just the API.

```
algorithms   →   Linear Regression (OLS + gradient descent + Ridge + Lasso)
                 Logistic Regression (binary + multiclass + regularized)
                 K-Nearest Neighbors (classification + regression)
                 K-Means Clustering (elbow method + silhouette analysis)
                 Naive Bayes (Gaussian + Multinomial + Bernoulli)
                 Decision Trees (CART + pruning)
                 Random Forests (bagging + feature importance)
                 Support Vector Machines (linear + kernel)
                 Principal Component Analysis
                 Gradient Boosting

testing      →   100% unit tested against sklearn — identical outputs verified
docs         →   every algorithm has derivation → intuition → code → result
```

`Python` `NumPy` `Matplotlib` `Math-first` `Unit tested`

<br/>

</details>

---

<div align="center">

### `// what I'm building`

</div>

<br/>

<details>
<summary><strong>🔥 &nbsp; agentic-ai-production-system &nbsp;·&nbsp; active</strong></summary>

<br/>

> *A multi-agent orchestration system built for production — not a demo, not a prototype.*

Most "agentic AI" projects are chains wrapped in Streamlit. This is different. It's a full production system with instrumentation, safety, evaluation gates, and a feedback loop that fine-tunes the model on real user interactions.

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Request  ──▶  FastAPI  ──▶  LangGraph Orchestrator   │
│                                    │                    │
│                    ┌───────────────┼───────────────┐   │
│                    ▼               ▼               ▼   │
│                 Planner        Executor        Reflector│
│                    │               │               │   │
│                    └───────────────┼───────────────┘   │
│                                    │                    │
│              ┌─────────────────────┼──────────────┐    │
│              ▼                     ▼              ▼    │
│         RAG Pipeline         Tool Sandbox    Safety    │
│         (hybrid search)      (Docker)       Guards    │
│              │                     │              │    │
│              └─────────────────────┼──────────────┘    │
│                                    │                    │
│        Prometheus ── Langfuse ── Audit Logs (S3)       │
│                                    │                    │
│                        Human Approval Gate             │
│                                    │                    │
│                     LoRA Fine-tuning on Feedback       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**What makes it production-grade:**

- Circuit breakers on every external call — no silent failures
- PII scrubbing before any data touches the LLM
- RAGAS evaluation runs on every PR — merge blocked on faithfulness regression
- Human-in-the-loop approval gate before irreversible tool actions
- Every interaction logged to S3 for compliance and replay
- LoRA fine-tuning loop trained on collected thumbs-up/down feedback
- Multi-tenant rate limiting with token bucket per API key

`LangGraph` `FastAPI` `Qdrant` `Docker` `Kubernetes` `RAGAS` `Prometheus` `Langfuse` `LoRA` `Redis`

<br/>

</details>

<br/>

<details>
<summary><strong>🔥 &nbsp; llm-gateway-platform &nbsp;·&nbsp; active</strong></summary>

<br/>

> *A routing layer that sits in front of any LLM provider. Optimized routing. Semantic caching. Automatic fallback.*

The problem: you're calling OpenAI directly, paying full price on cache-able queries, and one provider outage takes your whole system down. This gateway solves all three.

```python
# route by strategy — gateway picks the optimal provider automatically
response = gateway.complete(prompt, strategy="cost")    # → cheapest model available
response = gateway.complete(prompt, strategy="speed")   # → lowest p99 latency
response = gateway.complete(prompt, strategy="safe")    # → circuit-broken fallback chain

# semantic cache — similar queries return cached response
# "what is gradient descent?" and "explain gradient descent" → same cache hit
```

**How the routing works:**

```
Incoming Request
      │
      ▼
  Auth + Rate Limit
      │
      ▼
  Semantic Cache  ──── HIT ──────────────────▶  Return cached response
      │
     MISS
      │
      ▼
  Router (cost / speed / safe)
      │
      ├──▶  OpenAI
      ├──▶  Anthropic
      ├──▶  Together AI
      └──▶  Local vLLM
      │
      ▼
  Circuit Breaker  ──── OPEN ──▶  Fallback chain
      │
     CLOSED
      │
      ▼
  Response + Metrics (Prometheus) + Traces (OpenTelemetry)
```

**Chaos engineering included** — a test suite that randomly kills providers mid-run, verifies circuit breakers open, and confirms fallback activates within SLA. Because a gateway you haven't deliberately broken isn't a gateway you can trust.

`FastAPI` `Redis` `OpenTelemetry` `Grafana` `Locust` `Terraform` `Kubernetes`

<br/>

</details>

<br/>

<details>
<summary><strong>🔥 &nbsp; gpt-engineer-kit &nbsp;·&nbsp; active</strong></summary>

<br/>

> *GPT-2 implemented twice. Once for clarity, once for performance. BPE tokenizer from scratch. Benchmarked.*

Two complete implementations in one repo:

```
legacy/        →  clean, annotated, readable
                  every operation mapped to the paper
                  for understanding the architecture

optimized/     →  FlashAttention v2
                  Rotary Position Embeddings (RoPE)
                  PagedAttention-style KV cache
                  SwiGLU MLP
                  torch.compile
                  FP8 quantization stubs
                  FSDP distributed training wrapper
```

BPE tokenizer built from scratch — merge rules, vocabulary, encode/decode — before touching HuggingFace tokenizers.

Also includes stubs for alternative architectures: Mamba (selective SSMs), Hyena operators, RWKV — for when attention isn't the answer.

```
benchmarks vs nanoGPT:

  perplexity   →  WikiText-2, measured at every checkpoint
  throughput   →  tokens/sec at batch sizes 1, 8, 32, 128
  memory       →  peak GPU memory per optimization added
  compilation  →  torch.compile speedup measured independently
```

`PyTorch` `CUDA` `FlashAttention` `FSDP` `FP8` `torch.compile` `Mamba` `RWKV`

<br/>

</details>

<br/>

<details>
<summary><strong>⚡ &nbsp; llm-evaluation-framework &nbsp;·&nbsp; building</strong></summary>

<br/>

> *Evaluate any LLM system in 3 lines. Block any deployment that regresses.*

Most teams deploy LLMs and hope quality holds. This framework makes quality a hard gate.

```python
from llm_eval import Evaluator

# run evaluation
results = Evaluator(
    metrics=["faithfulness", "hallucination", "relevancy", "answer_correctness"]
).run(predictions, references)

# block CI on regression
results.assert_threshold(faithfulness=0.85, hallucination=0.05)

# compare two model versions
dashboard.compare(results_v1, results_v2)  # opens Streamlit diff view
```

**What it evaluates:**

```
offline    →  RAGAS (faithfulness, context recall, answer relevancy)
              DeepEval (GEval, answer correctness, hallucination detection)
              custom metrics (tool call accuracy, cost per query, latency)

online     →  stream real queries to Kafka/S3
              monitor input distribution drift
              log real-world failure cases

ci/cd      →  assert_threshold() blocks merges on regression
              nightly benchmark runs with variance analysis
              Streamlit dashboard: compare any two model versions
```

**Why this closes the loop:** this framework runs against every other repo I build. The agentic system is evaluated here. The gateway is benchmarked here. The GPT kit's generations are scored here. One place to know if quality is holding.

`RAGAS` `DeepEval` `Streamlit` `Kafka` `Prometheus` `FastAPI` `Langfuse`

<br/>

</details>

---

<div align="center">

### `// tech arsenal`

</div>

<br/>

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Python · PyTorch · NumPy · HuggingFace Transformers
  LangChain · LangGraph · FastAPI · Pydantic

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LLM ENGINEERING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Fine-tuning (LoRA · QLoRA) · RLHF · RAG Pipelines
  Prompt Engineering · LLM-as-judge · Speculative Decoding
  FlashAttention · RoPE · KV Cache · FP8 Quantization

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RAGAS · DeepEval · Custom Metrics · Langfuse · Prometheus

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VECTOR SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FAISS · Qdrant · Pinecone · Weaviate
  Dense + Sparse + Hybrid Retrieval · Cross-encoder Reranking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MLOPS & INFRA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Docker · Kubernetes · Helm · GitHub Actions · Terraform
  Prometheus · Grafana · OpenTelemetry · Locust

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLOUD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AWS — EC2 · S3 · Lambda · SageMaker · EKS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PostgreSQL · MongoDB · Redis · Celery · Kafka

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MATHEMATICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Calculus · Linear Algebra · Probability Theory · Statistics
```

<br/>

---

<div align="center">

### `// open source`

</div>

<br/>

I contribute to the ecosystem, not just consume it. Every repo I build is designed to be forked, extended, and built on — with derivations others can follow, benchmarks others can reproduce, and post-mortems others can learn from.

**Actively looking to contribute to:**

```
  HuggingFace Transformers   →   evaluation, documentation, reproducibility
  RAGAS                      →   custom metrics, edge case coverage
  DeepEval                   →   metric implementations, CI integrations
  vLLM                       →   inference optimization experiments
  LangGraph                  →   production patterns, reliability improvements
```

The goal: leave every project I touch more testable, more documented, and more honest about its failure modes than I found it.

<br/>

---

<div align="center">

### `// how I build`

</div>

<br/>

```
  Every repo I ship clears five gates before merge:

  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  01  WHY DOES THIS WORK?                            │
  │      Mathematical derivation lives in docs/         │
  │      No black boxes. No "trust the framework."      │
  │                                                     │
  │  02  HOW DOES IT WORK?                              │
  │      Scratch implementation before any library      │
  │      If I can't write it in NumPy, I don't use it  │
  │                                                     │
  │  03  DOES IT ACTUALLY WORK?                         │
  │      Benchmarks with real numbers, not vibes        │
  │      Tested against reference implementations       │
  │                                                     │
  │  04  WHAT BROKE?                                    │
  │      Post-mortems documented in docs/failures.md    │
  │      Failures are first-class content, not hidden   │
  │                                                     │
  │  05  CAN IT HANDLE PRODUCTION?                      │
  │      Failure modes mapped. Fallbacks implemented.   │
  │      Load tested. Circuit breakers in place.        │
  │                                                     │
  └─────────────────────────────────────────────────────┘
```

<br/>

---

<div align="center">


<br/>

---

<div align="center">

<br/>

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AMMAN HUSSAIN ANSARI
  AI Engineer  ·  MLOps  ·  Open Source Contributor
  India  ·  Remote  ·  Globally Available
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

<br/>

[![Email](https://img.shields.io/badge/ammanism@yahoo.com-6001D2?style=for-the-badge&logo=yahoo&logoColor=white)](mailto:ammanism@yahoo.com)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ammmanism/)
&nbsp;
[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://twitter.com/ammmanism)

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:161b22,100:0d1117&height=100&section=footer&animation=fadeIn"/>

</div>
