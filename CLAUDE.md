# CQEC ML Decoder — Claude Context File

**Project:** Continuous Quantum Error Correction ML Decoder
**Team:** Clark Enge & Pranav Reddy · UCSB Data Science Club Project Showcase
**Repo:** https://github.com/pkarakala/cqec-ml-decoder
**Hard Deadline:** Tue Apr 28, 11:59 PM — post to team Slack (poster + repo + blurb)
**Midpoint Showcase:** Tue Apr 21, 6:30 PM · TD-W 2600 (mandatory, ≥2 members)

---

## Project Structure

```
cqec-ml-decoder/
├── notebooks/
│   ├── 01_phase1_setup.ipynb          # Baseline / sanity check
│   ├── 02_phase2_dynamics.ipynb       # Bayesian filter & drift modeling
│   ├── 03_phase3_nonideal.ipynb       # Static GRU decoder (has overfitting issue)
│   └── 04_phase4_adaptive_decoding.ipynb  # Adaptive GRU — PRIMARY NOTEBOOK
├── outputs/                           # Figures saved here after re-run
├── presentation/
│   ├── slides_content.md              # Source of truth for slides (has 22 ____ blanks)
│   └── build_slides.py                # Run this to regenerate .pptx
├── scripts/
├── src/
├── README.md                          # Phase 4 section now has real numbers (Apr 19 re-run)
└── requirements.txt
```

---

## Phase Summary

| Phase | Notebook | What it does |
|-------|----------|-------------|
| 1 | `01_phase1_setup.ipynb` | Threshold decoder — baseline sanity check |
| 2 | `02_phase2_dynamics.ipynb` | Bayesian filter — optimal under static assumptions |
| 3 | `03_phase3_nonideal.ipynb` | Static GRU — trained offline, no adaptation |
| 4 | `04_phase4_adaptive_decoding.ipynb` | Adaptive GRU — pseudo-label + hybrid supervised |

---

## Current Results (Apr 19 re-run, N=100, T=400, tuned adapt params)

Sandbox CPU run with `adapt_lr=0.005`, `ema_decay=0.5`, `supervised_every=20`.
Full spec (N=200, T=1000, E=50) still owed on Pranav's Mac — will supersede
these when available; numbers are directionally consistent with prior N=60 run.
Raw JSON: `outputs/phase4_results.json`.

### Overall Test Accuracy
| Decoder | Overall | Seg 1 (low drift) | Seg 5 (high drift) | Drop (pp) |
|---------|---------|-------------------|--------------------|-----------|
| Threshold | 0.700 | 0.760 | 0.600 | 16.0 |
| Bayesian Filter | 0.765 | 0.901 | 0.650 | 25.1 |
| Static GRU | **0.861** | 0.892 | 0.760 | 13.2 |
| Adaptive (pseudo-labels) | 0.784 | 0.458 | 0.264 | **(collapses across all segs)** |
| Adaptive (hybrid, every 20) | 0.857 | 0.882 | 0.735 | 14.7 |

**Hybrid-vs-static gap:** −0.4 pp overall, −2.5 pp at seg 5 (static edges
hybrid under this aggressive adapt setting). Narrative has been pivoted
per plan: lead with pseudo-label collapse, frame hybrid as the obvious
mitigation that prevents collapse.

**Supervision sweep:** every 10 → 0.864, every 20 → 0.842, every 50 → 0.807,
every 100 → 0.736, every 200 → 0.809, every 500 → 0.583, static → 0.861.
Still mildly non-monotonic at 100/200 — likely single-seed noise; full-spec
re-run should smooth it.

---

## Status Snapshot (Apr 19 EOD)

### Completed today
- [x] Phase 4 re-run with tuned adapt params → `outputs/phase4_results.json` + figs
- [x] Phase 3 overfitting: notebook already uses `train_gru(..., dropout=0.3, patience=10)`; fix is in place
- [x] Filled all 22 slide placeholders in `presentation/slides_content.md`
- [x] Updated README Phase 4 section (was "results pending")
- [x] Updated `build_slides.py` data rows + softened slide 10/16 narrative
- [x] Regenerated `presentation/adaptive_qec_slides.pptx` (22 slides)
- [x] Produced `cqec_project_tracker_apr19.xlsx` with updated statuses

### Still open
- [ ] **Full-spec Phase 4 re-run** (N=200, T=1000, E=50) on Pranav's Mac — ~20 min.
  Re-run command: `P4_N=200 P4_T=1000 P4_EPOCHS=50 python3 scripts/run_phase4_rerun.py`.
  If numbers shift materially, re-propagate through slides/README/tracker.
- [ ] **Run-All on `04_phase4_adaptive_decoding.ipynb`** so the notebook's
  inline cell outputs match the re-run JSON. Notebook itself is already
  regenerated from `_build_phase4.py` with tuned params.
- [ ] **Phase 3 notebook re-run** — fix is in the cell, just execute.
- [ ] **Delete stale tracker** `cqec_project_tracker.xlsx` and rename
  `cqec_project_tracker_apr19.xlsx` → original name (Clark — fuse mount
  blocks agent from overwriting existing files).
- [ ] **Poster v1** — ready to build; hero figure is `phase4_accuracy_over_time.png`.

---

## Key Findings to Preserve in All Outputs

1. **Pseudo-label collapse is the hero finding** — under tuned adapt params the
   adaptive GRU stays ~99.9% confident while accuracy crashes to 12–46% across
   every segment. Dramatic distribution-shift failure of self-training.
2. **Bayesian has best early performance but worst degradation** — 90% → 65%
   (25.1pp). Motivates data-driven decoders.
3. **Static GRU is surprisingly robust** — only 13.2pp drop. Acknowledge
   honestly; don't hide it. It even edges hybrid overall in this run.
4. **Hybrid prevents collapse, doesn't exceed static** — framing should be
   "stability/safety under self-training" not "hybrid wins on accuracy".
5. **Supervision sweep still noisy at single seed** — every-200 > every-100 is
   noise; full-spec re-run or 3-seed average will smooth.

---

## Narrative Strategy (DECIDED: pseudo-label collapse lead)

Hybrid-static gap is tight (<2pp), so we're on the pivot path:
- **Lead:** pseudo-label collapse (99.9% confident, 26% accurate) as headline.
- **Middle:** hybrid as the obvious mitigation — matches static, beats pseudo.
- **Close:** future work = smarter supervision scheduling to make hybrid
  *beat* static, not just match it.

---

## Slide Blanks (all 22 filled Apr 19 — re-fill if numbers change)

All `____` placeholders in `presentation/slides_content.md` now have real
values. To change: edit `slides_content.md` AND the matching cells in
`build_slides.py`, then re-run `python presentation/build_slides.py`.

- **Slide 11:** 5 overall accuracies (Th/BF/Static/Pseudo/Hybrid)
- **Slide 12:** 5×5 temporal segment table + Static drop + Hybrid drop + gap-at-seg-5
- **Slide 13:** pseudo avg confidence at seg 5, pseudo accuracy at seg 5
- **Slide 14:** 6 supervision sweep accuracies + static baseline
- **Slide 15:** Phase 4 column (Hybrid/Static final numbers)

---

## Judge Q&A Prep (Known Tough Questions)

| Question | Answer |
|----------|--------|
| "Hybrid degrades MORE than static over time — why?" | "Hybrid trades overall stability for online plasticity; it still dramatically beats pure pseudo-label. Better supervision scheduling is future work." |
| "Why only 3-qubit repetition code?" | "Architecture generalizes; surface code extension is future work listed in README." |
| "No comparison to Chamberland / Varbanov?" | "Undergraduate project focused on analog/continuous setting; prior ML decoder work is mostly on discrete syndromes. Direct comparison is future work." |
| "Phase 3 overfits?" | "We report final-epoch for fair comparison; peak val was 86.7%." |

---

## Task Ownership at a Glance

| Task | Owner | Priority | Due | Status |
|------|-------|----------|-----|--------|
| Re-run Phase 4 notebook (sandbox) | Agent | 🔴 Blocker | Apr 19 | ✅ Done |
| Decide narrative from re-run results | Both | 🔴 Blocker | Apr 19 | ✅ Done (pseudo-label collapse lead) |
| Fix Phase 3 overfitting | Clark | 🟠 High | Apr 19 | ✅ In notebook (needs Run-All) |
| Fill slide blanks | Clark | 🟠 High | Apr 19 | ✅ All 22 filled |
| Update README | Pranav | 🟠 High | Apr 19 | ✅ Done |
| Regenerate .pptx | Pranav | 🟠 High | Apr 19 | ✅ Done |
| Full-spec Phase 4 re-run (Mac) | Pranav | 🟠 High | Apr 20 | ⏳ Open |
| Run-All Phase 3 & 4 notebooks | Pranav | 🟠 High | Apr 20 | ⏳ Open |
| Build poster v1 | Clark | 🟠 High | Apr 20 | ⏳ Open |
| Write brochure blurb | Pranav | 🟠 High | Apr 20 | ⏳ Open |
| Attend Midpoint Showcase | Both | 🔴 Blocker | Apr 21 | ⏳ Open |
| Latency analysis (Phase 3 & 4) | Pranav | 🟡 Medium | Apr 24 | ⏳ Open |
| Drift-rate robustness sweep | Clark | 🟡 Medium | Apr 25 | ⏳ Open |
| Final poster v2 | Both | 🟠 High | Apr 26 | ⏳ Open |
| Cross-check all numbers | Pranav | 🟠 High | Apr 27 | ⏳ Open |
| Submit to Slack | Pranav | 🔴 Blocker | Apr 28 | ⏳ Open |

---

## Important Notes for Claude

- **Single source of truth = notebook outputs.** Never assume slide/README/poster numbers match until explicitly cross-checked.
- **Don't fix what isn't asked.** Scope changes to the specific file/cell mentioned.
- When editing `slides_content.md`, search for `____` — there are 22 of them.
- The `outputs/figures/` directory populates only after a full notebook re-run.
- `requirements.txt` has all dependencies; use the `.venv` in the repo root.
- `scripts/run_phase4_rerun.py` is a standalone re-runner (env vars `P4_N`, `P4_T`, `P4_EPOCHS`); use it when full notebook execution is too slow, or to refresh `outputs/phase4_results.json` after tweaking adapt params.
