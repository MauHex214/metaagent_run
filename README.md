# MetaAgent Run

A 6-stage LLM-assisted pipeline that discovers, defines, and extracts
structured environmental metadata from full-text hydrosphere-research
literature, associating the extracted values with biosample accessions.

## Pipeline stages

| Step | Module | Purpose |
|------|--------|---------|
| **Step 1** | `steps/step1` | Paper-level target-environment classification (regex prefilter + LLM 5-way hydrosphere categorisation) |
| **Step 2** | `steps/step2` | Paragraph-level accession / label / metadata relation extraction (+ INSDC verification, builds Step 3 input) |
| **Step 3** | `steps/step3` | Schema discovery — iterative metadata-field discovery + LLM semantic de-duplication |
| **Step 4-accession** | `steps/step4_accession` | Sample-level BioSample hydrosphere-sub-environment classification |
| **Step 4-metadata** | `steps/step4_metadata` | Metadata tier / MIxS alignment + Fisher-test environmental profiling |
| **Step 5** | `steps/step5` | Structured metadata extraction — Identity Resolution → Table+LLM extraction → Normalization finalisation |
| **env_field_pipeline** | `steps/env_field_pipeline` | Produces the `env6_extraction_targets.json` schema that Step 5 consumes. See [hydrosphere_meta_norm](https://github.com/MauHex214/hydrosphere_meta_norm) for the standalone view |

`steps/step3_old`, `steps/step6`, `steps/step7` are historical designs and
experimental extensions kept here for traceability.

## Design

- Python 3.10+, fully async (`asyncio`)
- LLM: DeepSeek-V3 via Huawei ModelArts (OpenAI-compatible, bearer-token auth)
- Key deps: `httpx`, `pandas`, `pydantic`, `tqdm`, `scipy`, `matplotlib`
- Core helpers in `metaagent_run/core/`:
  - LLM client with ramp-up concurrency + retry-with-jitter
  - Stream → truncation-continue → JSON-repair → Pydantic-validate → non-stream fallback
  - JSONL checkpointing for long-running stages (resumable)
  - Batch → single-item fallback for bulk LLM calls

## Runtime configuration

All stages read `ALL_API_KEY` from the environment (ModelArts bearer token).
Each step has a `config.py` with endpoint URL, concurrency caps, temperature,
and `max_tokens`; the specifics differ slightly per step.

## Usage pattern

Every step exposes a unified CLI entrypoint through its `new.py`:

```bash
export ALL_API_KEY=...
cd <project_root>
python3 -m metaagent_run.steps.step1.new          # target-env classification
python3 -m metaagent_run.steps.step2.new          # relation extraction
python3 -m metaagent_run.steps.step3.new          # schema discovery
python3 -m metaagent_run.steps.step4_accession.new
python3 -m metaagent_run.steps.step4_metadata.new
python3 -m metaagent_run.steps.step5.new

# env_field_pipeline has its own phase-by-phase entrypoint
python3 -m metaagent_run.steps.env_field_pipeline.new 0
python3 -m metaagent_run.steps.env_field_pipeline.new 1
# ... through phase 6
```

## Inputs not in this repo

Each step reads upstream artefacts that users provide locally (they are
not checked in due to size and data-sensitivity):

- `relation_v1_step2_relation_output.json` — Step 2 paragraph records
- `paper_env_map.json` — PMID → hydrosphere sub-env(s) mapping
- `pmid_run_merged_data_expanded.json` — PMID → full-text sections
- `mixs_hydrosphere_slots.json` — 89 MIxS water-sphere slots (candidate vocabulary)
- `design_review_package/` — historical curation decisions for the offline eval only

The `env_field_pipeline` sub-pipeline additionally expects:
- `env_field_pipeline_output/phase6_schema.yaml` — curator decisions
- `env_field_pipeline_output/raw_key_expansion_table.csv` — 115-entry normalization table
- `env_field_pipeline_output/mixs_hydrosphere_slots.json`

## Outputs

Each stage writes into an `..._output/` directory (gitignored); the
`env_field_pipeline` produces:

- `env6_extraction_targets.json` — consumed by Step 5's `upstream_loader`
- `env6_main_schema.csv` (77 targets) + `env6_signature_schema.csv` (81 targets)
- `env6_excluded_trace.csv` (100 R1-R4 rejections, for the paper appendix)
- Full pipeline traceability (raw_key → canonical → target)

## Repo layout

```
metaagent_run/
├── README.md
├── .gitignore
└── metaagent_run/                # Python package
    ├── __init__.py
    ├── core/                     # shared LLM client + JSON + retry + checkpoint
    └── steps/
        ├── env_field_pipeline/   # 7 phases + viz_final + embedding_viz + phase6 renderer
        ├── step1/                # Target-env classification
        ├── step2/                # Relation extraction
        ├── step3/                # Schema discovery
        ├── step3_old/            # Legacy schema discovery (kept for reference)
        ├── step4_accession/      # BioSample env classification
        ├── step4_metadata/       # Metadata tier + MIxS alignment
        ├── step5/                # Structured metadata extraction
        ├── step6/                # Experimental
        └── step7/                # Experimental
```

## Related repos

- [`hydrosphere_meta_norm`](https://github.com/MauHex214/hydrosphere_meta_norm) —
  standalone view of `steps/env_field_pipeline` plus `docs/` (design philosophy,
  decision YAML, expansion table). Updates should be kept in sync with this
  repo's `steps/env_field_pipeline/` subtree.
