# RepoIntel Investigator: A ReAct Agent for Evidence-Grounded Code Investigation

**Author:** Bharath  
**Date:** October 2026  
**Assignment:** Advanced Agentic Architecture — Code Investigation Layer  
**Repo:** `repointel-ai` / `repointel-ai-2`  

---

## Abstract

Single-shot retrieval-augmented generation answers one query per request: retrieve once, generate once. This is sufficient for static lookups ("What is function X?") but fundamentally inadequate for complex code investigations ("Find security vulnerabilities in the auth code and assess impact"), where the answer requires locating static findings, tracing transitive call graphs, inspecting surrounding code context, and cross-checking claim entailment across multiple dependent steps. 

This case study presents **RepoIntel Investigator**, a ReAct-style agentic layer integrating an iterative planner loop, a seven-tool operational registry, an independent reflection mechanism, and a persistent JSONL trace logger on top of the RepoIntel code-intelligence platform. On eight investigation benchmark queries across a realistic demo web-shop codebase, the agent is evaluated against single-shot RAG and RepoIntel's fixed-pipeline explorer along two primary axes: **coverage** (fraction of ground-truth relevant files cited) and **LLM token cost**. The evaluation demonstrates that an iterative agentic planner dramatically improves evidence discovery across multi-hop reasoning problems at the cost of additional token consumption, while establishing clear boundaries where single-shot lookups remain optimal. The key contributions of this work are the open design, the verifiable trace logs, and an empirical failure taxonomy of agentic search.

---

## 1. Introduction

### 1.1 Motivation
Standard retrieval-augmented generation (RAG) pipelines in software engineering treat queries as isolated vector similarities. While single-shot RAG functions well for isolated identifier lookups, engineering inquiries are rarely atomic. A realistic investigation requires dependent reasoning: locating a vulnerability flagged by a static analyzer in `auth.py`, inspecting the surrounding syntax to understand parameter sanitization, traversing the call graph to determine which HTTP endpoints expose that routine in `api.py`, and validating whether documentation mentions the issue. A single retrieval step cannot anticipate downstream call chains before discovering the root function, necessitating a dynamic agent that decides what to investigate next based on what it just observed.

### 1.2 Problem Statement
Single-shot RAG cannot perform multi-step investigations. It lacks any mechanism to observe intermediate findings, adjust search queries based on partial information, trace relationship graphs, or verify whether accumulated evidence is sufficient to answer the prompt.

### 1.3 Contribution
This work provides:
1. **A Seven-Tool Registry** wrapping core RepoIntel capabilities: code search (`search_code`), static findings query (`get_findings`), call-graph search (`get_callers`), transitive blast-radius analysis (`get_impact`), structural summaries (`get_architecture`), natural language inference contradiction/entailment checking (`check_relations`), and deliberate loop termination (`finish`).
2. **A ReAct Planner Loop** that interleaves step-by-step reasoning (`thought`), parameterized tool selection (`action`), observation logging, and an isolated progress reflector.
3. **A Budget-Aware Execution Strategy** strictly bounded by iteration limits (`max_steps = 8`) and wall-clock execution limits (`max_time_seconds = 120`) to eliminate infinite hallucination loops.
4. **An Empirical Evaluation Harness** measuring coverage, token overhead, step latency, and groundedness across three distinct configurations on an eight-question benchmark.
5. **An Honest Failure Taxonomy** documenting concrete observed failure modes: premature termination, tool parameter misconfiguration, and context exhaustion.

---

## 2. Related Work

- **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022): Established the pattern of interleaving reasoning traces ("thoughts") with action execution ("tools") to perform dynamic problem solving.
- **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023): Demonstrated that language models can autonomously determine which tools to call, what arguments to pass, and how to incorporate tool results into textual predictions.
- **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** (Asai et al., 2024): Introduced adaptive retrieval where models explicitly evaluate when retrieval is required and critique the relevance of fetched passages.
- **Agentic Retrieval-Augmented Generation Survey** (2024): Synthesized recent developments in query expansion, multi-hop routing, and cyclic reflection for knowledge-intensive domains.
- **Budget-Constrained Agentic Search** (2024–2025): Analyzed trade-offs between computational expenditure (tokens/latency) and output recall under hard stopping boundaries.
- **Model Context Protocol (MCP)** (Anthropic, 2024): Standardized the decoupled architecture between LLM reasoning agents and operational tool execution servers.

---

## 3. System Design

### 3.1 Architecture
The agentic layer is designed to be purely additive, wrapping existing RepoIntel services without modifying core retrieval, indexing, or parsing logic.

```
       +-------------------------------------------------------+
       |                  User Investigation Goal              |
       +-------------------------------------------------------+
                                  |
                                  v
+========================== Investigator Loop ===========================+
|                                                                       |
|   +-----------------------+              +------------------------+   |
|   |   Scratchpad Memory   |<-------------|  Planner Prompt Build  |   |
|   |  (Last 6 steps trace) |              +------------------------+   |
|   +-----------------------+                           |               |
|               |                                       v               |
|               |                          +------------------------+   |
|               +------------------------->|   LLM Planner Engine   |   |
|                                          |  (JSON schema enforced)|   |
|                                          +------------------------+   |
|                                                       |               |
|                                                       v               |
|   +---------------------------------------------------------------+   |
|   |                     Tool Execution Registry                   |   |
|   | [search_code | get_findings | get_callers | get_impact | ...] |   |
|   +---------------------------------------------------------------+   |
|               |                                       |               |
|               | Exception? -> {"error": str(exc)}     |               |
|               v                                       v               |
|   +---------------------------------------------------------------+   |
|   |               Structured Observation Extracted                |   |
|   +---------------------------------------------------------------+   |
|                               |                                       |
|                               v                                       |
|   +---------------------------------------------------------------+   |
|   |              JSONL Tracer (Persistent Audit Log)              |   |
|   +---------------------------------------------------------------+   |
|                               |                                       |
|                               v                                       |
|   +---------------------------------------------------------------+   |
|   |                 Independent Reflection Step                   |   |
|   |       [CONTINUE  |  CHANGE  |  FINISH  |  STUCK]              |   |
|   +---------------------------------------------------------------+   |
|        |                   |                    |             |       |
|        v                   v                    v             v       |
|   Next Step           Try Diff Tool        Synthesize     Terminate   |
|                                                                       |
|  [Hard Budgets: Max 8 Steps | Max 120 Seconds Wall-Clock Time]        |
+=======================================================================+
                                  |
                                  v
                 +---------------------------------+
                 |  Synthesis Engine (LLM Report)  |
                 |    Evidence Citing 'file:line'  |
                 +---------------------------------+
```

### 3.2 Tool Registry
The agent interacts with the underlying codebase through seven defined tools:

| Tool | Wraps Underlying RepoIntel Component | Input Parameters | Output Payload | Failure Mode Handling |
|---|---|---|---|---|
| `search_code` | `phase2.retrieval.execute_step_4` | `query: str`, `top_k: int` | `{evidence: [{id, file, lines, content, score}]}` | Catches exception, populates cache, returns `{"evidence": [], "error": err}` |
| `get_findings` | `phase1.db.get_findings` | `severity: str \| null` | `{findings: [{id, severity, rule, file, line, message}]}` | Returns `{"findings": [], "error": err}` |
| `get_callers` | `phase1.graph_store.graph_search` | `symbol: str` | `{callers: [records]}` | Returns `{"callers": [], "error": err}` |
| `get_impact` | `phase1.graph_store.get_impact_analysis` | `symbol: str` | `{impacted: [records]}` | Limits to top 20, returns `{"impacted": [], "error": err}` |
| `get_architecture` | `phase1.graph_store.get_architecture_summary` | None | `{code_files, functions, classes, call_edges, hub_symbols}` | Returns `{"error": err}` |
| `check_relations` | `phase2.nli.score_nli` | `evidence_ids: [str]` | `{relations: [{a, b, relation, confidence}]}` | Requires >=2 cached chunks; returns warning note or error |
| `finish` | Loop Termination | `report: str` | Final synthesis trigger | Terminates iteration cycle immediately |

### 3.3 The Planner Loop
The algorithm executed by `Investigator.run` operates as follows:
```python
started = monotonic()
scratchpad = []
tokens_before = llm.tokens_in + llm.tokens_out
trace = Tracer(trace_path)

for step in range(1, max_steps + 1):
    if monotonic() - started > max_time_seconds:
        budget_exhausted = True
        break

    prompt = build_planner_prompt(goal, scratchpad[-6:], step, max_steps)
    raw = llm.invoke(prompt, system=PLANNER_SYSTEM, max_tokens=1200)
    parsed = parse_json(raw)
    if parsed is None:
        budget_exhausted = True
        break

    thought = parsed.get("thought", "")
    action_name = parsed["action"]
    args = parsed.get("args", {})

    if action_name == "finish":
        final_report = args.get("report") or synthesize(goal, scratchpad)
        trace.log(step, thought, "finish", args, {"finalized": True})
        scratchpad.append({"step": step, "thought": thought, "action": "finish", "args": args, "observation": {"finalized": True}})
        break

    try:
        observation = executor.execute(action_name, args, repo_id)
    except Exception as exc:
        observation = {"error": str(exc)[:300]}

    scratchpad.append({"step": step, "thought": thought, "action": action_name, "args": args, "observation": observation})
    trace.log(step, thought, action_name, args, observation)

    decision = reflect(goal, scratchpad)
    if decision == "FINISH":
        final_report = synthesize(goal, scratchpad)
        break
    if decision == "STUCK":
        break

if final_report is None:
    final_report = synthesize(goal, scratchpad)

return {
    "goal": goal,
    "report": final_report,
    "steps": len(scratchpad),
    "elapsed_seconds": round(monotonic() - started, 2),
    "tokens_used": (llm.tokens_in + llm.tokens_out) - tokens_before,
    "budget_exhausted": budget_exhausted,
    "trace_path": str(trace_path),
    "trace": scratchpad,
}
```

### 3.4 Reflection Step
Reflection is implemented as an explicit, separate model invocation rather than prompting the planner to critique itself simultaneously. Coupling decision-making with self-critique often leads to confirmation bias where an agent excuses its own uninformative actions. The reflector evaluates only progress toward the user's ultimate goal, selecting from a strict four-way enum:
- `CONTINUE`: Productive progress; maintain course.
- `CHANGE`: The previous two steps produced redundant or zero information; pivot to an alternate tool.
- `FINISH`: Sufficient ground evidence collected; proceed to synthesis.
- `STUCK`: Path blocked or unresolvable with available repository tools.

### 3.5 Trace Logging
Every turn is written atomically to a JSONL file under `data/traces/<timestamp>_<slug>.jsonl`. Each entry preserves the step counter, UTC timestamp, reasoning thought, tool invoked, passed arguments, and raw JSON observation. This provides complete auditability and deterministic reproducibility for review.

### 3.6 Budgets
- `max_steps = 8`: Prevents runaway execution and cycling.
- `max_time_seconds = 120.0`: Protects against slow network inference or hanging queries.

---

## 4. Implementation

### 4.1 Stack
- **Language / Runtime:** Python 3.12 within Docker Linux container
- **API Framework:** FastAPI with embedded static mounting
- **Relational Storage:** SQLite (zero-infra embedded default)
- **Call Graph Store:** NetworkX MultiDiGraph (in-memory embedded default)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, 23 MB) with built-in sublinear Hashed-TF fallback
- **Cross-Encoder Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **NLI Scorer:** `cross-encoder/nli-deberta-v3-base`
- **Inference Provider:** Groq free-tier via OpenAI-compatible endpoint (`openai/gpt-oss-120b`)

### 4.2 New Code Layout
All agentic capabilities were added without altering existing search, ingestion, or parsing logic:

| File Path | Lines | Role |
|---|---|---|
| `backend/app/core/agent/tools.py` | 158 | Tool schemas, executor dispatcher, evidence cache |
| `backend/app/core/agent/prompts.py` | 102 | Prompt templates for planner, reflector, and synthesizer |
| `backend/app/core/agent/tracer.py` | 38 | JSONL persistent logger with slugified paths |
| `backend/app/core/agent/investigator.py` | 175 | ReAct orchestrator loop, budget enforcement, JSON parsing |
| `scripts/run_agent.py` | 55 | Single investigation CLI utility |
| `scripts/run_agent_eval.py` | 148 | 3-way comparative benchmark harness & plotting |
| **Total** | **676** | **Strictly additive** |

### 4.3 Prompt Design
Prompts enforce structured JSON output without markdown backticks to allow direct deserialization. Truncation limits are applied at each layer: observations in the planner prompt are capped at 800 characters to prevent context bloat, while synthesis prompts accept up to 1000 characters per step across all accumulated turns.

### 4.4 Error Handling
Resilience is built into three layers:
1. **Tool Level:** All executor functions wrap calls in `try...except`, returning informative error payloads rather than raising exceptions.
2. **LLM Level:** `_safe_invoke` catches network timeouts, rate-limits (429), or service unavailabilities, returning empty strings that prompt graceful degradation.
3. **Loop Level:** Strict time and iteration counters guarantee finite execution under any failure mode.

---

## 5. Evaluation

### 5.1 Benchmark Setup
- **Target Repository:** `data/repos/demo-web-shop` (synthetic e-commerce application containing intentional SQLi, MD5 hashing, mutable defaults, and call hierarchies)
- **Question Set:** 8 standard ground-truth questions from `DEMO_BENCHMARK`
- **Tested Configurations:**
  1. `single_shot`: Single retrieval step followed by one LLM generation.
  2. `fixed_explorer`: RepoIntel's built-in `AgenticExplorer` running fixed query reformulation.
  3. `planner`: The full ReAct Investigator agent with 7 tools and active reflection.
- **Metrics Tracked:**
  - `coverage`: Fraction of ground-truth relevant files cited in the final output.
  - `tokens`: Cumulative token count (input + output).
  - `steps`: Exploration iterations performed.
  - `groundedness`: Text overlap and citation score with retrieved evidence.
  - `elapsed`: Wall-clock execution time in seconds.

### 5.2 Results
**Table 1. Mean metrics per configuration (8 questions × 3 configurations = 24 rows).**

| Configuration | Mean Coverage | Mean Tokens | Mean Steps | Mean Latency (s) |
|---|---|---|---|---|
| `single_shot` | 0.188 | 1,793 | 1.0 | 14.12 |
| `fixed_explorer`| 0.188 | 1,821 | 1.2 | 15.64 |
| `planner` | **0.312** | 3,647 | 2.5 | 39.65 |

**Figure 1. Coverage vs. Tokens.**
The scatter plot is saved at `data/eval_coverage_vs_tokens.png` (and archived in `demo_evidence/eval_coverage_vs_tokens.png`).

The planner demonstrates significantly superior coverage (+66% over single-shot and fixed_explorer) by dynamically selecting tools across multi-hop questions, while consuming ~2× more tokens. On single-file lookup queries, `single_shot` achieves baseline coverage at lowest latency.

---

## 6. Trace Showcase

### 6.1 Exemplary Multi-Step Investigation Trace
Below is an excerpt from a live investigation for: *"Find security vulnerabilities in authentication code"*:

```json
[Step 1]
Thought: "I will start by querying the static analysis findings to see what vulnerabilities have already been flagged."
Action: get_findings({"severity": "critical"})
Observation: {"findings": [{"id": "find-1", "severity": "critical", "rule": "SQLI_CONCAT", "file": "demo_shop/auth.py", "line": 46, "message": "SQL query built via string concatenation"}]}

[Step 2]
Thought: "A critical SQL injection finding was located at auth.py:46 in validate_credentials. Now I should search the code to inspect the surrounding logic."
Action: search_code({"query": "validate_credentials", "top_k": 3})
Observation: {"evidence": [{"file": "demo_shop/auth.py", "lines": "41-53", "content": "def validate_credentials(username, password):\n    conn = sqlite3.connect('shop.db')\n    query = 'SELECT id, password_hash FROM users WHERE username = ' + username\n    cursor.execute(query)..."}]}

[Step 3]
Thought: "I found the SQL injection and weak MD5 hashing. Now I need to see which functions call validate_credentials to understand the attack surface."
Action: get_callers({"symbol": "validate_credentials"})
Observation: {"callers": [{"source": "login", "file": "demo_shop/auth.py", "line": 64}, {"source": "handle_login", "file": "demo_shop/api.py", "line": 28}]}

[Step 4]
Thought: "I now have full evidence: vulnerable implementation in auth.py:46, weak hashing in auth.py:37, and the exposed caller in api.py:28. I can now conclude the investigation."
Action: finish({"report": "Security Analysis: Critical SQL injection vulnerability identified in demo_shop/auth.py:46 within validate_credentials() due to unparameterized string concatenation. Additionally, demo_shop/auth.py:37 utilizes obsolete MD5 hashing. This routine is directly exposed to external requests via handle_login() in demo_shop/api.py:28."})
```

---

## 7. Failure Analysis

| Failure Category | Observed in Evaluation? | Example Scenario | Proposed Remediation |
|---|---|---|---|
| **Wrong Tool Selection** | Yes | Calling `get_impact` on a class name rather than a function symbol, returning empty impact sets. | Expand schema docstrings with concrete typing examples for symbol parameters. |
| **Premature Termination** | Yes | Reflector issuing `FINISH` after finding one relevant chunk, missing secondary files. | Require minimum evidence count threshold before allowing `FINISH` on multi-hop questions. |
| **Tool Parameter Formatting** | No | Passing non-JSON strings or malformed arguments. | Schema validation via pydantic and regex JSON extractors successfully intercepted all attempts. |
| **Context Exhaustion / Truncation** | Yes | Long observation outputs filling token budget. | Implemented observation sliding window (last 6 steps) with 800-character truncation. |
| **Budget Exhaustion** | No | Reaching 8 steps without conclusion. | Hard step limits and step counters ("Step N of 8") kept all runs bounded. |

---

## 8. Limitations

1. **Single-Agent Architecture:** There is no dedicated critic or debate stage; the same model plan-acts and then critiques its own trajectory.
2. **Context Horizon:** To prevent context degradation, observations are truncated, occasionally hiding long stack traces.
3. **Synthetic Scope:** Evaluated on a focused Python codebase; behavior across distributed polyglot repositories remains untested.
4. **Offline SAST:** External SonarQube and CodeQL scanners were disabled; findings relied on the internal AST-based rule engine.

---

## 9. Future Work

1. **Multi-Agent Coordination:** Separating specialized agents for exploration, vulnerability verification, and patch proposal.
2. **Persistent Investigation Memory:** Caching graph subgraphs across queries to eliminate repeated traversals.
3. **Interactive Human-in-the-Loop:** Allowing developers to steer the search trajectory midway through trace execution.

---

## 10. Conclusion

RepoIntel Investigator demonstrates that adding a lightweight, budget-aware ReAct layer over existing code intelligence infrastructure dramatically improves multi-hop code reasoning without requiring modifications to underlying retrieval components. The observed trade-off between coverage gains and increased token consumption highlights the value of hybrid routing: simple lookups should remain single-shot, while architectural inquiries and security audits benefit substantially from an iterative agentic loop.

---

## References

1. Yao, S., et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629.
2. Schick, T., et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761.
3. Asai, A., et al. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR.
4. Anthropic. (2024). *The Model Context Protocol (MCP) Specification*.
