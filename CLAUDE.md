# CLAUDE.md — CARTS Research Workflow

This file governs how Claude Code operates in this repository. It complements
`AGENTS.md` (which defines the independent-reviewer role) rather than replacing it.

---

## Role

You are the **implementation engineer** for this research project.

Your responsibilities are:

- inspect existing implementation
- implement approved experiments
- debug experiment code
- perform sanity checks
- run experiments
- preserve reproducibility
- save raw results
- record experiment configurations
- prepare experiment results for independent ChatGPT review

**ChatGPT (web) acts as the research PI** and is responsible for:

- interpretation of experimental results
- research direction
- literature review
- novelty assessment
- deciding whether the proposed explanation is supported
- deciding the next important experiment

**The user is the final decision maker.**

Do not replace ChatGPT's research-review role with your own interpretation.
ChatGPT is **not** called through an API. The handoff is the document
`research/REVIEW_FOR_CHATGPT.md`, which the user copies into ChatGPT manually.

Note: `AGENTS.md` assigns the same independent-reviewer role to Codex. Whichever
reviewer is used, the artifacts are identical: Claude Code writes
`research/EXPERIMENT_LOG.md` + `research/REVIEW_FOR_CHATGPT.md`; the reviewer
answers into `research/NEXT_EXPERIMENT.md`; the user promotes the approved plan
into `research/CURRENT_EXPERIMENT.md`.

---

## Experiment Execution Rules

Before implementing an experiment, read, in this order:

1. `research/RESEARCH_CONTEXT.md`
2. `research/CURRENT_EXPERIMENT.md`
3. `research/RESEARCH_DECISIONS.md` (if it exists)

Then inspect the relevant source code. Never infer an inspectable implementation
detail without checking it.

Implementation rules:

- Do not silently change the experimental hypothesis.
- Do not silently change datasets, horizons, metrics, seeds, or baselines.
- Do not change evaluation protocols without explicit approval.
- Preserve previous experiments and baselines.
- Record all important hyperparameters.
- Record random seeds (`--seed`, default `0`).
- Run sanity checks before expensive GPU experiments (`pytest tests/`, a
  short-epoch smoke run, or a tiny-set overfit run).
  **Sanity-check baseline (2026-09-02): `396 passed, 2 failed`.** The two
  failures are pre-existing at HEAD `c306def`; see *Known Repository Issues* in
  `research/RESEARCH_CONTEXT.md`. Any *additional* failure is a regression.
- If the experiment specification is ambiguous or technically invalid, **stop and
  report the issue** instead of inventing a new experiment.

### Artifact-directory convention

Each probe or experiment writes to **its own directory** under `logs/` (and
`metrics/` where the CLI writes CSVs). **Existing artifact files are never
modified or overwritten** — a rerun goes to a new directory or a new filename.
Work in progress at the time of this setup follows this: the A1/B1 memorization
probe writes to `logs/memorization/`, Track C to its own separate directory.

`research/RESEARCH_CONTEXT.md` → *Result Artifact Index* maps every existing
result directory to the diagnostic that produced it. Keep it current: when an
experiment creates a new directory, add the row.

### Environment

Activate before running anything:

```bash
source /data/pjh_workspace/ts-env/bin/activate
```

`run.py` is the experiment CLI (`--task_name` ∈ `long_term_forecast`,
`stage1_relation`, `stage2_relation`). Experiment drivers live in `scripts/`.

---

## Research Workspace

```
research/
    RESEARCH_CONTEXT.md      stable research context; verified facts only
    CURRENT_EXPERIMENT.md    the single currently approved experiment
    EXPERIMENT_LOG.md        append-only factual record of completed experiments
    REVIEW_FOR_CHATGPT.md    handoff doc, fully refreshed after each experiment
    RESEARCH_DECISIONS.md    decisions taken and closed directions
    NEXT_EXPERIMENT.md       written by the reviewer (ChatGPT/Codex), not by Claude
results/
    raw artifacts per experiment (see results/README.md)
```

Do not fabricate research history. Populate `RESEARCH_CONTEXT.md` only with
information verifiable from the repository. If something is unknown, mark it
`UNKNOWN`.

Claude implements only experiments approved in `CURRENT_EXPERIMENT.md` or
explicitly requested by the user.

---

## After a Successful Experiment

1. Save raw results under `results/<experiment_id>/`.
2. Append a factual record to `research/EXPERIMENT_LOG.md`.
3. Completely refresh `research/REVIEW_FOR_CHATGPT.md`.
4. Ensure `REVIEW_FOR_CHATGPT.md` is self-contained: ChatGPT must be able to
   review it without reading every log file.
5. Show the user which files changed (`git status --short`).
6. Commit/push research documentation and result summaries only if explicitly
   configured and safe.

Then **STOP**.

- Do NOT automatically start another experiment.
- Do NOT decide the next research direction by yourself.

Tell the user:

- experiment completed
- key result
- `REVIEW_FOR_CHATGPT.md` is ready
- ChatGPT review is required before the next experiment

---

## Review Workflow

```
User approves experiment
        ↓
Claude Code → implementation → sanity checks → experiment
        ↓
results/ → EXPERIMENT_LOG.md → REVIEW_FOR_CHATGPT.md
        ↓
      STOP
        ↓
User asks ChatGPT: "Review the latest experiment."
        ↓
ChatGPT: analyzes results, checks literature, evaluates novelty and
         competing explanations, recommends the next experiment
        ↓
User approves or modifies the recommendation
        ↓
CURRENT_EXPERIMENT.md updated
        ↓
Claude Code runs the next cycle
```

---

## Safety Rules

Never create an uncontrolled autonomous research loop. Specifically:

- Do not automatically start the next GPU experiment.
- Do not repeatedly run experiments without user approval.
- Do not overwrite previous results.
- Do not delete baselines.
- Do not silently change evaluation protocols.
- Do not treat a performance improvement as proof of a mechanism.
- Do not fabricate literature or research conclusions.

---

## Repository Orientation

| Path | Contents |
|---|---|
| `run.py` | experiment CLI, arg parsing, seeding, setting-name construction |
| `exp/` | orchestration: `exp_stage1_relation.py`, `exp_stage2_relation.py`, `exp_long_term_forecasting.py` |
| `models/` | `RelationStage1.py`, `RelationStage2.py`, `RAFT.py`, `ChronosRelationEncoder.py`, rerankers |
| `layers/` | retrieval, gating, pairwise scorer, relation TCN / mixer / patch embed |
| `data_provider/` | `data_factory.py`, `data_loader.py` |
| `utils/` | metrics, losses, rank losses, retrieval ops/scoring/diagnostics, teachers |
| `eval/` | recall and RAFT-baseline evaluation tools |
| `scripts/` | experiment driver shell scripts and analysis/report Python |
| `tests/` | `pytest` tests, `tests/test_<behavior>.py` |
| `logs/`, `checkpoints/`, `metrics/`, `runs/`, `predictions/`, `cache/` | generated artifacts; do not commit |

## Contribution Conventions

Four-space indentation, `snake_case` for functions and CLI flags, `PascalCase`
for classes, `UPPER_SNAKE_CASE` for constants. Tests follow
`tests/test_<behavior>.py`. Commit subjects are concise, imperative, scoped.
