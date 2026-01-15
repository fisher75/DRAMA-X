# DRAMA-X CoC QA Gate Report
- input: `/workspace/chz/code/DRAMA-X/annotation_coc/drama_x_coc_qwen3vl_2b_v2_full_rule_based_20260108.jsonl`
- total: **5679**
- low_margin_thr: **0.1**
- low_margin_count: **1309**
- low_info_count: **0**

## Risk label distribution
- high: 5528
- low: 151

## Driving decision distribution
- SLOW: 2955
- STOP: 2510
- GO: 214

## Consistency (heuristic)
- ok: 5613
- conflict_high_go: 65
- conflict_low_stop: 1

## Trace length (chars)
- min: 115
- p50: 206
- p90: 269
- max: 427

## Next actions (recommended)
- Manually review `samples_for_review.jsonl` (random + low-margin + low-info).
- If low-margin samples often have wrong primary_vru: consider v3 (temporal / relative-motion cues).
- If low-info traces dominate: selective re-annotation with larger model on low-info subset.
