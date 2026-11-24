# Frozen coding-agent decode comparison set

[manifest.json](manifest.json) pins 48 independent requests from the same SWE-agent 1.1.0 submission as the [three historical controls](../coding_agent/README.md). The first entry in each cell is one of the 12 pilot requests. Replay each selected query in isolation with the [baseline protocol](../../../../docs/user/contributing/coding-agent-benchmark.md). Use the same frozen inputs for both versions. The baseline server revision is `e25780e2`.

## Selection

The submission index listed 500 tasks. Removing the three historical controls left 497. Across the 497 tasks with eligible snapshots, 16,451 snapshots met the stage and length rules. The length cuts are the first token counts reaching 25%, 50% and 75% of the candidate distribution, with equal total weight for each task and equal weight for its eligible turns. The inclusive upper bounds are **13,344, 21,961, 32,475** tokens. Candidate lengths range from 1,864 to 140,640.

Stage evidence comes only from the last assistant tool call and its corresponding visible tool result in the selected query. Explore requires a successful file or directory view. Edit requires a confirmed replacement or insertion in a project subdirectory. Verify/debug requires output from a recognizable test runner or reproduction, check or debug script. Ambiguous turns were skipped. These rules select stages with concrete evidence and do not estimate their frequency in production. Each eligible request must fit the 262,144-token context with the 8,192-token output allowance.

Each length/stage cell contributes four distinct tasks. Seed `ignis-5080-coding-agent-48-v1` orders task IDs and turns by SHA-256. In each of four rounds, fill the cells with the fewest eligible tasks first, breaking ties by quartile then stage name. Within a cell, prefer repositories with the fewest picks overall, then the fewest picks in the cell, then the smallest seeded task hash. Among the task's eligible turns in that cell choose the smallest seeded turn hash. The chosen task is removed globally. Task ranks hash `seed|task|quartile|stage|task_id` and turn ranks hash `seed|turn|quartile|stage|task_id|query_index` in UTF-8. No subsequent response, task outcome or Ignis decode measurement enters the selection.

## Requests

All 48 input lengths were independently checked with the baseline commit's Rust `/v1/tokenize` handler, using the real artifact frontend and a CPU mock scheduler. The same check reproduced the three historical counts of 19,954, 22,420 and 16,717 tokens.

Query indices start at zero. The manifest gives the repository revision, tool evidence, exact input length, request SHA-256 and source trajectory SHA-256 for each row.

| Quartile | Stage | Task | Query | Input tokens | Pilot |
|---|---|---|---:|---:|:---:|
| Q1 | Explore | mwaskom__seaborn-3187 | 1 | 2,978 | ✓ |
| Q1 | Explore | mwaskom__seaborn-3069 | 2 | 3,042 |  |
| Q1 | Explore | scikit-learn__scikit-learn-14087 | 2 | 4,724 |  |
| Q1 | Explore | django__django-16950 | 8 | 10,180 |  |
| Q1 | Edit | django__django-11099 | 7 | 8,489 | ✓ |
| Q1 | Edit | astropy__astropy-7606 | 17 | 9,790 |  |
| Q1 | Edit | matplotlib__matplotlib-24970 | 8 | 7,456 |  |
| Q1 | Edit | pytest-dev__pytest-6202 | 20 | 11,404 |  |
| Q1 | Verify/debug | astropy__astropy-8872 | 12 | 11,201 | ✓ |
| Q1 | Verify/debug | scikit-learn__scikit-learn-25973 | 8 | 10,939 |  |
| Q1 | Verify/debug | pytest-dev__pytest-10051 | 16 | 13,318 |  |
| Q1 | Verify/debug | pylint-dev__pylint-7277 | 5 | 12,667 |  |
| Q2 | Explore | pydata__xarray-4687 | 7 | 15,059 | ✓ |
| Q2 | Explore | pytest-dev__pytest-10081 | 7 | 15,524 |  |
| Q2 | Explore | psf__requests-1921 | 27 | 21,275 |  |
| Q2 | Explore | sympy__sympy-14248 | 10 | 21,818 |  |
| Q2 | Edit | pytest-dev__pytest-7324 | 20 | 15,534 | ✓ |
| Q2 | Edit | matplotlib__matplotlib-25332 | 17 | 16,263 |  |
| Q2 | Edit | pydata__xarray-7233 | 15 | 17,120 |  |
| Q2 | Edit | sphinx-doc__sphinx-9320 | 12 | 13,661 |  |
| Q2 | Verify/debug | matplotlib__matplotlib-24627 | 33 | 19,417 | ✓ |
| Q2 | Verify/debug | psf__requests-6028 | 16 | 20,776 |  |
| Q2 | Verify/debug | django__django-14017 | 18 | 21,521 |  |
| Q2 | Verify/debug | pydata__xarray-6992 | 22 | 15,156 |  |
| Q3 | Explore | pylint-dev__pylint-6528 | 31 | 25,243 | ✓ |
| Q3 | Explore | pydata__xarray-3677 | 51 | 30,726 |  |
| Q3 | Explore | scikit-learn__scikit-learn-14629 | 37 | 22,766 |  |
| Q3 | Explore | psf__requests-1724 | 21 | 22,120 |  |
| Q3 | Edit | scikit-learn__scikit-learn-25102 | 25 | 26,320 | ✓ |
| Q3 | Edit | sympy__sympy-12096 | 28 | 24,000 |  |
| Q3 | Edit | sphinx-doc__sphinx-8593 | 32 | 28,878 |  |
| Q3 | Edit | pydata__xarray-7393 | 51 | 25,993 |  |
| Q3 | Verify/debug | pallets__flask-5014 | 29 | 22,444 | ✓ |
| Q3 | Verify/debug | django__django-14311 | 25 | 22,846 |  |
| Q3 | Verify/debug | sympy__sympy-15875 | 52 | 31,323 |  |
| Q3 | Verify/debug | astropy__astropy-7166 | 29 | 22,607 |  |
| Q4 | Explore | sympy__sympy-13091 | 77 | 38,575 | ✓ |
| Q4 | Explore | sphinx-doc__sphinx-11445 | 38 | 34,038 |  |
| Q4 | Explore | pylint-dev__pylint-4551 | 46 | 32,495 |  |
| Q4 | Explore | astropy__astropy-13977 | 51 | 61,256 |  |
| Q4 | Edit | sphinx-doc__sphinx-7590 | 50 | 63,564 | ✓ |
| Q4 | Edit | django__django-11138 | 33 | 33,979 |  |
| Q4 | Edit | sympy__sympy-13551 | 38 | 37,820 |  |
| Q4 | Edit | matplotlib__matplotlib-23299 | 54 | 35,330 |  |
| Q4 | Verify/debug | psf__requests-1142 | 39 | 47,081 | ✓ |
| Q4 | Verify/debug | pylint-dev__pylint-4970 | 31 | 40,819 |  |
| Q4 | Verify/debug | astropy__astropy-14096 | 31 | 39,629 |  |
| Q4 | Verify/debug | sphinx-doc__sphinx-7748 | 57 | 51,177 |  |

## Materializing the frozen requests

Download each entry in the manifest's `sources` to its `local_file` in one scratch directory. From the repository root, use the [existing extractor](../../../../scripts/extract_coding_agent.py) with PyYAML 6.0.3. It verifies every source hash, checks the archived tool documentation and verifies each generated request before writing it.

```sh
python3 scripts/extract_coding_agent.py \
  --manifest crates/bench/fixtures/coding_agent_48/manifest.json \
  --source-dir .scratch/coding-agent-48-sources \
  --out .scratch/coding-agent-48-requests
```

The request hashes cover the frozen messages and tool schemas using the historical fixture encoding. Add the baseline generation options at replay time. The stock three-request runner has its original fixture list compiled in.
