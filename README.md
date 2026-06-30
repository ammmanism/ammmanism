<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:161b22,100:0d1117&height=120&section=header&text=&animation=fadeIn"/>

```
░█████╗░███╗░░░███╗███╗░░░███╗░█████╗░███╗░░██╗
██╔══██╗████╗░████║████╗░████║██╔══██╗████╗░██║
███████║██╔████╔██║██╔████╔██║███████║██╔██╗██║
██╔══██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║██║╚████║
██║░░██║██║░╚═╝░██║██║░╚═╝░██║██║░░██║██║░╚███║
╚═╝░░╚═╝╚═╝░░░░░╚═╝╚═╝░░░░░╚═╝╚═╝░░╚═╝╚═╝░░╚══╝
```

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=15&duration=2500&pause=1000&color=58A6FF&center=true&vCenter=true&multiline=false&repeat=true&width=650&lines=Production+AI+Engineer+%E2%80%94+I+ship+systems+that+survive+real+traffic.;LLM+Inference+%7C+RAG+%7C+MLOps+%7C+Transformers+%7C+Evaluation;From+math+derivation+%E2%86%92+scratch+implementation+%E2%86%92+hardened+deployment." />

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ammmanism/)
[![Twitter](https://img.shields.io/badge/X%20%2F%20Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://twitter.com/ammmanism)
[![Email](https://img.shields.io/badge/ammanism@yahoo.com-6001D2?style=for-the-badge&logo=yahoo&logoColor=white)](mailto:ammanism@yahoo.com)
[![GitHub](https://img.shields.io/badge/GitHub-161b22?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ammmanism)

<br/>

![](https://img.shields.io/badge/🇮🇳_India-remote%2C_globally_available-58A6FF?style=flat-square&labelColor=0d1117)
&nbsp;
![](https://img.shields.io/badge/🚀_status-building_production-238636?style=flat-square&labelColor=0d1117)
&nbsp;
![](https://img.shields.io/badge/✨_open_source-contributor-8957e5?style=flat-square&labelColor=0d1117)

</div>

---

<div align="center">

### `// who I am`

</div>

> I don't follow tutorials. I derive equations.  
> I don't ship demos. I ship systems that survive production.  
> I don't guess. I benchmark, instrument, and iterate.

I'm an AI engineer who builds from **first principles → scratch implementation → hardened deployment**. Every system I create is mathematically grounded, rigorously tested, and engineered for failure resilience.

```python
class Amman:
    stack      = ["LLMs", "RAG", "MLOps", "Transformers", "Evaluation"]
    languages  = ["Python", "SQL", "Bash"]
    approach   = "derive → implement from scratch → harden → benchmark → ship"
    based_in   = "India"
    available  = "remote, globally"
    building   = True  # always
```

---

<div align="center">

### `// open source impact`

</div>

I contribute to the ecosystem, not just consume it. Every PR I ship fixes real bugs that affect production users.

#### 🐍 LangChain Monorepo — 5 Bugs Resolved

| Issue # | Package | Bug Impact | Fix | Status |
|:---|:---|:---|:---|:---|
| **#37049** | `text-splitters` | `Language.PERL` threw `ValueError` due to missing separators | Added full Perl grammar support + unit tests | ✅ Merged |
| **#37369** | `langchain_v1` | `glob_search` returned arbitrary filesystem order despite `mtime-desc` promise | Added `mtime` descending sort + regression test | ✅ Merged |
| **#37058** | `partners` & `core` | Async `asimilarity_search` docstrings missing `await` | Fixed documentation across Qdrant, Chroma, InMemoryVectorStore | ✅ Merged |
| **#36993** | `core` | LangSmith metadata concatenated on every chunk (`openaiopenaiopenai`) | Extended skip-concatenation guard + 5 regression tests | ✅ Merged |
| **#36747** | `core` | `parse_partial_json` failed on raw `\t` / `\r` in streaming JSON | Patched character parser to escape control characters | ✅ Merged |

#### 🔗 physical-ai-toolchain — 2 Critical Fixes

| Issue # | Component | Problem | Resolution | SHA |
|:---|:---|:---|:---|:---|
| **#500** | Dev Container | DB cleanup scripts failed due to missing `psql`/`redis-cli` binaries | Added `postgresql-client` + `redis-tools` to `onCreateCommand` | `94678f1` |
| **#688** | CI/CD | PR validation silently skipped fuzz tests due to double-escaped regex | Fixed `grep -E` path filter regex (`\\` → `\`) | `d1606b8` |

---

<div align="center">

### `// what I've shipped`

</div>

<br/>

<div align="center">

```
╔═══════════════════════════════════════════════════════════╗
║         🚀  PRODUCTION SYSTEMS  ·  ACTIVE  ·  PUBLIC      ║
╚═══════════════════════════════════════════════════════════╝
```

</div>

---

<details>
<summary><strong>⚡ &nbsp; fast-gpt-lab &nbsp;·&nbsp; active</strong></summary>

<br/>

> *GPT architecture implemented twice — once for clarity, once for performance. BPE tokenizer from scratch. Benchmarked against nanoGPT.*

Bridges the gap between theoretical deep learning and hardware-level optimization. Two complete implementations in one repo:

```
legacy/        →  clean, annotated, readable
                  every operation mapped to the Attention Is All You Need paper
                  for understanding the architecture deeply

optimized/     →  FlashAttention v2
                  Rotary Position Embeddings (RoPE)
                  PagedAttention-style KV cache
                  SwiGLU MLP
                  torch.compile
                  FP8 quantization stubs
                  FSDP distributed training wrapper
```

BPE tokenizer built from scratch — merge rules, vocabulary, encode/decode — before touching HuggingFace tokenizers.

```
benchmarks vs nanoGPT:

  perplexity   →  WikiText-2, measured at every checkpoint
  throughput   →  tokens/sec at batch sizes 1, 8, 32, 128
  memory       →  peak GPU memory per optimization added
  compilation  →  torch.compile speedup measured independently
```

`PyTorch` `CUDA` `FlashAttention` `FSDP` `FP8` `torch.compile` `RoPE` `BPE`

<br/>

</details>

---

<details>
<summary><strong>🔥 &nbsp; cost-aware-llm &nbsp;·&nbsp; active</strong></summary>

<br/>

> *A high-performance LLM Gateway that dynamically routes requests across multiple providers using cost, latency, and reliability signals.*

The problem: calling OpenAI directly means paying full price on cacheable queries, and one provider outage takes your whole system down. This gateway solves all three.

```python
# Route by strategy — gateway picks the optimal provider automatically
response = gateway.complete(prompt, strategy="cost")    # → cheapest model available
response = gateway.complete(prompt, strategy="speed")   # → lowest p99 latency
response = gateway.complete(prompt, strategy="safe")    # → circuit-broken fallback chain

# Semantic cache — similar queries return cached response
# "what is gradient descent?" and "explain gradient descent" → same cache hit
```

**Routing architecture:**

```
Incoming Request
      │
      ▼
  Auth + Rate Limit (token bucket per API key)
      │
      ▼
  Semantic Cache ──── HIT ──────────────────▶ Return cached response
      │
     MISS
      │
      ▼
  Router (cost / speed / safe signal scoring)
      │
      ├──▶  OpenAI
      ├──▶  Anthropic
      ├──▶  Together AI
      └──▶  Local vLLM
      │
      ▼
  Circuit Breaker ──── OPEN ──▶ Fallback chain
      │
     CLOSED
      │
      ▼
  Response + Prometheus metrics + OpenTelemetry traces
```

**Chaos engineering included** — a test suite that randomly kills providers mid-run, verifies circuit breakers open, and confirms fallback activates within SLA.

`FastAPI` `Redis` `OpenTelemetry` `Grafana` `Locust` `Terraform` `Kubernetes` `Multi-tenant`

<br/>

</details>

---

<details>
<summary><strong>🤖 &nbsp; agentic-ai-production-system &nbsp;·&nbsp; active</strong></summary>

<br/>

> *A multi-agent orchestration system built for production — LLMs, tool-use, and workflow orchestration for autonomous reasoning and execution.*

Most "agentic AI" projects are chains wrapped in Streamlit. This is different — a **full production system** with instrumentation, safety gates, evaluation, and a feedback loop that fine-tunes the model on real user interactions.

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
│              │                                    │    │
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

- ✅ Circuit breakers on every external call — no silent failures
- ✅ PII scrubbing before any data touches the LLM
- ✅ RAGAS evaluation runs on every PR — merge blocked on faithfulness regression
- ✅ Human-in-the-loop approval gate before irreversible tool actions
- ✅ Every interaction logged to S3 for compliance and replay
- ✅ LoRA fine-tuning loop trained on collected thumbs-up/down feedback

`LangGraph` `FastAPI` `Qdrant` `Docker` `Kubernetes` `RAGAS` `Prometheus` `Langfuse` `LoRA` `Redis`

<br/>

</details>

---

<details>
<summary><strong>🧠 &nbsp; neurocache &nbsp;·&nbsp; active</strong></summary>

<br/>

> *Production-grade semantic caching layer with FAISS + pgvector, multi-tenant billing, and real-time analytics.*

A FastAPI-based LLM proxy that slashes API costs through intelligent semantic caching, with full tenant isolation and usage metering.

**Key Features:**
- **Native Anthropic Streaming** — SSE chunk mapping with sub-50ms TTFT
- **Exact Token Counting** — `tiktoken` BPE tokenizer for precise billing
- **Cache Warmer** — `POST /v1/cache/warm` to pre-populate from JSONL datasets
- **Tenant CRUD** — SHA-256 hashed API keys with Redis 1h TTL fallback
- **Real-Time Stats** — `GET /v1/cache/stats` with hit rates, latency gains, USD saved

`FastAPI` `FAISS` `pgvector` `Redis` `PostgreSQL` `Prometheus` `tiktoken` `Multi-tenant`

<br/>

</details>

---

<details>
<summary><strong>📐 &nbsp; pure-ml &nbsp;·&nbsp; completed</strong></summary>

<br/>

> *Mathematical Foundations → Algorithms → Neural Networks → Research Engineering. Machine Learning implemented from scratch using NumPy.*

Before touching any framework, I sat down with the mathematics and built everything from scratch. Each algorithm comes with a full derivation document, visual comparisons against sklearn, and benchmarks proving identical outputs.

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
docs         →   every algorithm: derivation → intuition → code → result
```

This repo exists to prove one thing: **I understand the math, not just the API.**

`Python` `NumPy` `Matplotlib` `Math-first` `Unit tested`

<br/>

</details>

---

<div align="center">

### `// tech arsenal`

</div>

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
  FlashAttention · RoPE · KV Cache · FP8 Quantization · BPE

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

---

<div align="center">

### `// how I build`

</div>

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

---

<div align="center">

### `// currently seeking`

</div>

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║  🎯  SENIOR AI / ML ENGINEER ROLES                      ║
║                                                           ║
║  • LLM inference optimization & production serving       ║
║  • RAG system architecture at scale                      ║
║  • Agentic AI orchestration & safety engineering           ║
║  • Open-source AI infrastructure                         ║
║                                                           ║
║  📍  Remote · Global   |   📧  ammanism@yahoo.com       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

**Actively looking to contribute to:**

```
  HuggingFace Transformers   →   evaluation, documentation, reproducibility
  RAGAS                      →   custom metrics, edge case coverage
  DeepEval                   →   metric implementations, CI integrations
  vLLM                       →   inference optimization experiments
  LangGraph                  →   production patterns, reliability improvements
```

---

<div align="center">

### `// in the lab 🔐`

</div>

> *Some projects live in private repos. Some are in closed beta. Being stress-tested with real users before the world sees them.*

```
╔═══════════════════════════════════════════════════════════╗
║  🎯  CURRENT FOCUS: PRODUCTION-GRADE AI PLATFORM          ║
║                                                           ║
║  • Multi-tenant architecture with usage metering          ║
║  • Real-user feedback loops driving model iteration       ║
║  • End-to-end observability: logs, traces, metrics        ║
║  • Auth, billing, and rate-limiting baked in from day 1   ║
║                                                           ║
║  Status: 🚧 Private beta · Invite-only · Real traffic    ║
╚═══════════════════════════════════════════════════════════╝
```

<details>
<summary><strong>🔍 &nbsp; Research Prototypes (not public yet)</strong></summary>

<br/>

```
🧪 multimodal-data-interpreter
   ├─ PDF + Excel + images + audio → unified query interface
   ├─ Natural language → SQL / Python / charts
   ├─ Auto-dashboard generation with live data refresh
   └─ Scalable backend: DuckDB/Spark for >RAM datasets

🧪 autonomous-code-reviewer
   ├─ Agentic PR analysis: bugs, perf, security, style
   ├─ Test generation + sandboxed execution
   ├─ Human-in-loop approval gates (reuse production patterns)
   └─ GitHub API integration + CI/CD hooks

🧪 real-time-meeting-copilot
   ├─ Live transcription + action item extraction
   ├─ Post-meeting RAG: "What did John say about the deadline?"
   └─ Privacy-first: local inference + on-prem LLM fallback
```

*These are research prototypes. If they survive benchmarking, hardening, and real-user testing — they'll graduate to production repos.*

</details>

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
