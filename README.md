<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=28&duration=1&pause=99999&color=58A6FF&center=true&vCenter=true&width=800&lines=Amman+Hussain+Ansari" alt="Amman Hussain Ansari" />

**LLM inference engineer.** Engine · router · KV quantizer — built from scratch, not wrapped.

<sub>📍 India · Remote · Global &nbsp;|&nbsp; 🟢 Open to inference infrastructure roles & contracts</sub>

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ammmanism)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://twitter.com/ammmanism)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ammanism@yahoo.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ammmanism)

<br/>

![Commits](https://img.shields.io/badge/commits_this_year-3%2C220-58A6FF?style=flat-square)
![Repos](https://img.shields.io/badge/production_repos-4-7EE787?style=flat-square)
![CSVs](https://img.shields.io/badge/committed_CSVs-60%2B-79C0FF?style=flat-square)
![Tests](https://img.shields.io/badge/tests_passing-400%2B-D2A8FF?style=flat-square)
![Reviews](https://img.shields.io/badge/adversarial_reviews-31-FFA657?style=flat-square)

</div>

---

## The Rule

<div align="center">

<table>
<tr>
<td align="center" width="100%">

### No number in a README without a committed CSV behind it.

<sub>Every benchmark is reproducible · Every failed optimization is logged · Every claim survives an adversarial review before it ships</sub>

</td>
</tr>
</table>

</div>

Enforced by CI in every repo — `no-claims-without-artifact` · `bench-regression` · `check_claims.py`. If a number can't be traced to a committed artifact, it doesn't exist in my work.

---

## The Stack

<div align="center"><sub>Four projects. One vertical chain. Each consumes the artifact of the previous.</sub></div>

```
  ┌────────────┐      ┌──────────────┐      ┌────────────┐      ┌────────────────┐
  │ miniserve  │ ───▶ │ agent-serve  │ ───▶ │  kv-quant  │ ───▶ │ agent-autopsy  │
  │  (engine)  │      │  (router)    │      │(optimizer) │      │  (forensics)   │
  └────────────┘      └──────────────┘      └────────────┘      └────────────────┘
```

---

## Production Systems

<table>
<tr>
<th align="left" width="18%">Project</th>
<th align="left" width="42%">What it does</th>
<th align="left" width="25%">Evidence</th>
<th align="center" width="8%">Version</th>
<th align="center" width="7%">Status</th>
</tr>

<tr>
<td><a href="https://github.com/ammmanism/miniserve"><b>miniserve</b></a></td>
<td>From-scratch LLM inference engine — continuous batching, PagedAttention KV with refcounts, async CPU/GPU overlap, speculative decoding, hand-written Triton flash-decode kernel, SLSA-3 release pipeline</td>
<td><code>200/200</code> greedy parity vs HuggingFace<br/><code>217</code> tests passing<br/><code>overlap_ratio &lt; 1</code> committed<br/><code>spec-decode == greedy</code> proven</td>
<td align="center"><code>v1.4</code></td>
<td align="center">🚀</td>
</tr>

<tr>
<td><a href="https://github.com/ammmanism/agent-serve"><b>agent-serve</b></a></td>
<td>Session-aware router — prefix radix index, CHBL bounded loads, EDF queue with SLO shedding, PD-disaggregation, TTL pinning, semantic + cascade model routing, WFQ fairness, geo-tier, adversarial defense suite</td>
<td><code>~60%</code> token savings vs round-robin<br/><code>30+</code> committed CSVs<br/><code>8</code> GIFs · 6-panel explorer<br/><b>Zenodo DOI</b></td>
<td align="center"><code>v2.0</code></td>
<td align="center">🚀</td>
</tr>

<tr>
<td><a href="https://github.com/ammmanism/kv-quant"><b>kv-quant</b></a></td>
<td>KV-cache quantization with honest quality cliffs — fused Triton dequant kernel, per-head sensitivity maps, RoPE-phase error analysis, Hadamard transforms, QJL residual sketches, adversarial cliff detection</td>
<td><code>11</code> CSVs + SHA256 manifests<br/><code>6</code> hero figures (300 DPI)<br/><code>R² = 0.9994</code> vs calculator<br/>math proofs in repo</td>
<td align="center"><code>v2.0</code></td>
<td align="center">🚀</td>
</tr>

<tr>
<td><a href="https://github.com/ammmanism/agent-autopsy"><b>agent-autopsy</b></a></td>
<td>Trajectory forensics — failure taxonomy, calibrated judge (two-κ split), attribution vs intervention localization, dependency-aware minimization, statistical CI gate</td>
<td>Execution-verified ground truth<br/>Two-judge minimization protocol<br/>6 bias probes + length-corrected κ</td>
<td align="center"><code>v0.1</code></td>
<td align="center">🔬</td>
</tr>

</table>

<sub>🚀 shipped &nbsp;·&nbsp; 🔬 in development &nbsp;·&nbsp; every claim traces to a committed artifact in the repo</sub>

---

## Evidence at a Glance

<div align="center">

| &nbsp; | miniserve | agent-serve | kv-quant | Total |
|:--|:--:|:--:|:--:|:--:|
| **Committed CSVs** | 5 | 30+ | 11 | **46+** |
| **Committed PNGs** | 4 | 25+ | 6 | **35+** |
| **Committed GIFs** | — | 8 | — | **8** |
| **Tests passing** | 217 | 16 files | 12 files | **400+** |
| **Adversarial reviews** | — | 15 | 16 | **31** |
| **ADRs** | 16 | 5 | 11 | **32** |
| **Zenodo DOI** | — | ✅ | — | **✅** |
| **SLSA-3 release** | ✅ | — | — | **✅** |
| **Karpathy layer** (`nano/`) | — | 4 files | — | **4** |

</div>

---

## Upstream Contributions

<div align="center"><sub>Real code in other people's codebases — not just my own repos.</sub></div>

<table>
<tr>
<th align="left" width="22%">Project</th>
<th align="left" width="48%">Contribution</th>
<th align="center" width="15%">Diff</th>
<th align="center" width="15%">Status</th>
</tr>

<tr>
<td><a href="https://github.com/BerriAI/litellm"><b>BerriAI / litellm</b></a></td>
<td><code>max_retries</code> support for non-OpenAI/Azure providers (Anthropic, Cohere, Bedrock, Vertex)</td>
<td align="center"><code>+318 / −12</code></td>
<td align="center">11 reviews</td>
</tr>

<tr>
<td><a href="https://github.com/BerriAI/litellm"><b>BerriAI / litellm</b></a></td>
<td>Dockerfile injection via unsanitized <code>docker.env</code> dict values</td>
<td align="center">bug report</td>
<td align="center">2 comments</td>
</tr>

</table>

---

## Technical Depth

<table>
<tr>
<td valign="top" width="33%">

### ⚙️ Inference Systems

- Continuous batching
- PagedAttention KV cache
- Speculative decoding
- PD-disaggregation
- CUDA graphs
- Async CPU/GPU overlap
- Triton kernels
- Nsight Compute · Perfetto

</td>
<td valign="top" width="33%">

### 🧮 Quantization

- KV-cache FP8 / INT4
- Per-head sensitivity maps
- RoPE-phase error analysis
- Hadamard transforms (partial-RoPE safe)
- QJL residual sketches
- Fused dequant kernels
- Quality cliff analysis

</td>
<td valign="top" width="33%">

### 🛡️ Reliability & Eval

- Fault injection
- Adversarial routing
- OTel decision spans
- Prometheus SLOs
- Judge calibration (Cohen's κ)
- Bootstrap + permutation CI gates
- Two-judge protocols

</td>
</tr>
</table>

<table>
<tr>
<td valign="top" width="50%">

### 🔀 Routing & Serving

- CHBL bounded-load hashing
- EDF queue + SLO shedding
- WFQ multi-tenant fairness
- Semantic + cascade model routing
- Geo-tier routing
- TTL pinning
- Queueing externality

</td>
<td valign="top" width="50%">

### 🧠 Math & Theory

- Linear algebra · Probability · Statistics
- Queueing theory (M/M/1 hazard · WFQ virtual finish)
- Information theory
- RoPE phase derivation
- Gaussian QJL arcsine law
- Hadamard commutativity proofs

</td>
</tr>
</table>

**Toolchain:** `Python` · `PyTorch` · `Triton` · `CUDA` · `FastAPI` · `DuckDB` · `Docker` · `GitHub Actions` · `Nix` · `uv` · `Prometheus` · `Grafana` · `Zenodo` · `SLSA-3`

---

## Available For

<table>
<tr>
<th align="left" width="35%">Service</th>
<th align="left" width="65%">What you get</th>
</tr>

<tr>
<td><b>🔧 Inference Engine Optimization</b></td>
<td>Kernel tuning · batching · scheduling · TTFT/ITL analysis · measured before/after with committed CSVs</td>
</tr>

<tr>
<td><b>💾 KV-Cache Quantization Audit</b></td>
<td>Config recommendation under your VRAM/SLA constraint · cliff analysis · per-head schedules · adversarial validation</td>
</tr>

<tr>
<td><b>🔀 Session-Aware Routing Design</b></td>
<td>Multi-model cascades · prefix-aware routing · $/1K analysis · queueing-externality-based scoring</td>
</tr>

<tr>
<td><b>🔍 Agent Failure Forensics</b></td>
<td>Failure taxonomy design · calibrated judge · CI regression gates · reusable GitHub Action</td>
</tr>

</table>

<sub>**Typical engagement:** architecture review → prototype → production hardening → handoff with committed benchmarks and a runbook.</sub>

---

## Currently Building

<table>
<tr>
<td align="center" width="100%">

### <a href="https://github.com/ammmanism/struct-decode">struct-decode</a> — grammar-constrained decoding engine

100% valid JSON / SQL / DSL at the token level via an incremental Earley parser and a token-mask cache.

<br/>

**Target artifact:** `overhead_vs_unconstrained < 1.5×` &nbsp;·&nbsp; **Guarantee:** 100% validity across 10k prompts

</td>
</tr>
</table>

---

## Engineering Philosophy

```python
>>> class EngineeringPhilosophy:
...     p1 = "Understand the math before touching the API"
...     p2 = "If I can't implement it in NumPy, I don't understand it"
...     p3 = "Benchmarks are the only truth — everything else is marketing"
...     p4 = "Production means graceful degradation, not perfect conditions"
...     p5 = "Every number in a README must trace to a committed CSV"
```

---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=14&duration=4000&pause=2000&color=58A6FF&center=true&vCenter=true&width=650&lines=If+this+work+saves+you+an+afternoon+of+debugging%2C+a+star+helps+it+reach+the+next+person." alt="thanks" />

<br/><br/>

**ammanism@yahoo.com** &nbsp;·&nbsp; [linkedin](https://linkedin.com/in/ammmanism) &nbsp;·&nbsp; [x](https://twitter.com/ammmanism) &nbsp;·&nbsp; [github](https://github.com/ammmanism)

<sub>📍 India · Remote · Globally available</sub>

</div>
