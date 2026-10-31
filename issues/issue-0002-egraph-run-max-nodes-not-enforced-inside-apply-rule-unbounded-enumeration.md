+++
id = "I-002"
title = "EGraph.run — max_nodes not enforced inside apply_rule; unbounded enumeration"
status = "open"
created = "2026-10-31"
owner = "devin"
priority = "medium"
+++

Found via I-001 investigation (2026-10-31). EGraph.run checks n_enodes >= max_nodes only BETWEEN iterations (core.py ~1725). Inside apply_rule, with enode_budget=None (no rule_budgets), match_cap=None → _iter_matches enumerates an unbounded substitution space per class, and each firing can add enodes without bound. Combined effect: one apply_rule call can run for hours and grow the graph to multiple GB — the div/recip/pow rsqrt spellings do exactly this under the current ALL_RULES bank (143 rules, ~57 prim:synop folds promoted recently). Frozen-read epoch journaling (ep.jph/added per union per find-step) makes per-union cost grow superlinearly with overlay size. Fix candidates: enforce a node cap inside apply_rule's match loop (honor run's max_nodes per-call), apply _MATCH_CAP even when enode_budget=None, and/or bound epoch overlay growth. The slow tests are deselected from the push gate (gates.sh dev lane) — this is the underlying engine follow-up.
