# IEEE TVT targeted reconstruction report

This is a separate author-review draft derived from `manuscript/` at the current working-tree state. The T-ITS source was not edited for this reconstruction. The TVT draft uses the unmodified scientific model, hard-QP construction, certificate definitions, theory, supervisor logic, experimental tables, and figure data. It does not add experiments or retune parameters.

1. **Title:** changed from “Feasibility Certification and Predictive Monitoring for Collision--Thermal Conflict and Dual-Brake Allocation in Heavy-Duty Platoons” to “Feasibility-Certified Dual-Brake Control With Predictive Monitoring for Thermally Constrained Heavy-Duty Platoons.”
2. **Abstract:** rewritten vehicle-first, retaining 43/1,275 and 192/288 initial-state findings and the negative same-plant ablation. It distinguishes the exact instantaneous sign test, conditional theoretical predictive certificate, and unverified numerical monitor.
3. **Contributions:** recast as physical command geometry, exact current certificate, conditional predictive/V2V monitoring, and four-mode supervisory validation. MAPPO and KKT differentiation remain secondary.
4. **Introduction:** now opens with heavy-vehicle brake hardware, fade, lag, collision demand, and remaining two-actuator authority. Figure 1 remains the early architecture figure.
5. **Related work:** reordered into heavy-duty braking, safety-constrained vehicle control, cooperative/V2V control, and secondary learning tools. No new reference was invented or added. The two uncited BibTeX records `amos2017optnet` and `emam2022robust` were removed from the TVT-only bibliography; 28 cited records remain.
6. **Section titles:** “Dynamics and Constraint Geometry” became “Heavy-Duty Dual-Brake Dynamics and Physical Constraints.” Problem formulation now introduces realized brake forces and their distinct commands before the parameter vector; the mathematical model is unchanged.
7. **Learning/KKT:** retained with a clearly secondary role. No differentiable-QP performance claim was promoted.
8. **Validation/results:** compressed repeated protocol and result narration while retaining numerical evidence and negative findings, including no demonstrated certificate-loss reduction, failed registered string-stability criterion, adapter-sensitive external transfer, and baseline runtime advantage. Frozen generated evidence files were not edited; manuscript-local result-table copies were redesigned typographically without changing values.
9. **Figures:** Fig. 1's editable TikZ artwork received a local typography/spacing revision without changing its scientific formulas or topology and is now included as vector PDF; Figs. 2--5 were unchanged. Fig. 2 is compiled from TikZ; Figs. 3--5 are vector PDFs. Source/rendered asset hashes are recorded below. The final PDF's Fig. 1 was rendered at 300 dpi and visually inspected at two-column width; text, formulas, arrowheads and routing passed the print-readability audit. See `TVT_FIGURE_TYPOGRAPHY_VERIFICATION.md`.
10. **Tables:** Tables I--IX were structurally reformatted for 10-point table bodies. No quantitative values, provenance categories, or frozen generated evidence files changed. See `TABLE_REDESIGN_REPORT.md`.
11. **References:** zero undefined citations or references after BibTeX build; 28 cited entries, no uncited entries in the TVT `.bib`. No citation metadata was fabricated.
12. **Pages:** 15-page original to 14-page TVT PDF. The reduction is from prose/appendix-check compression, not body font, margin, or line-spacing changes. Fig. 1 received a separate graphics-type recovery while remaining within the standard two-column width.
13. **Unchanged claims:** exact current hard-set feasibility; predictive one-sided guarantee only under sound enclosure; current floating-point predictor not formally certified; negative predictive values inconclusive; delayed upstream quantity not instantaneous fleet-global truth; no global recursive feasibility, string-stability theorem, real-vehicle validation, or general DiffQP superiority.
14. **Remaining submission blockers:** No identified format blocker remains after Fig. 1 lettering recovery; the administrative items are real author/affiliation/funding metadata and author approval of the AI-use disclosure draft and its placement. Main-paper tables contain neither `\scriptsize` nor `\tiny`; their bodies use the normal 10-point IEEEtran size. The final PDF is 14 pages with zero overfull boxes. Editorial scope judgment belongs to TVT, not this audit.

## Official TVT basis

- [VTS TVT scope](https://vtsociety.org/publication/ieee-transactions-vehicular-technology): vehicle safety controls and collision avoidance lie within Vehicular Electronics and Systems; connected/autonomous vehicles and platooning also fit.
- [VTS TVT instructions](https://vtsociety.org/publication/ieee-transactions-vehicular-technology/guidelines-authors/instructions): initial regular-paper limit is 14 double-column pages including references and biographies. The instructions also require author responsibility and disclosure for AI-assisted modification of author-written text. No author metadata or disclosure was invented in this draft.

## Figure asset register

| Figure | Editable source | PDF inclusion/rendered asset | Format and dimensions | SHA-256 (rendered) |
|---|---|---|---|---|
| 1 | `figures/fig1_desktop66_final.tex` | `figures/fig1_desktop66_final.pdf` | vector PDF, 514.072 × 372.603 pt asset box | `bec6cda26782fc18b738ffe37fa6d2ab862044c656d0ab96402f4e4c3500e223` |
| 2 | `figures/coupled_dynamics.tex` | compiled TikZ (`figures/coupled_dynamics_rendered.pdf` preview) | vector PDF preview, 284.97 × 94.842 pt | `881e2f53240216b7b3c805c9f999899c16ec141fcd19eb572dd5da7ac10199b1` |
| 3 | `../scripts/generate_submission_figures.py` | `../generated/submission_revision/figures/feasibility_frontier.pdf` | vector PDF, 675 × 408.75 pt | `538a782bc68fc05e9edf4856a45c16959e78f4923343ae11921304475c55a618` |
| 4 | `../scripts/generate_submission_figures.py` | `../generated/submission_revision/figures/predictive_examples.pdf` | vector PDF, 675 × 510 pt | `d584e4256a0abe344f71b86a910894ade7ff72d6aa56bb3bf5600a2c1034c540` |
| 5 | `../scripts/generate_submission_figures.py` | `../generated/submission_revision/figures/communication_age.pdf` | vector PDF, 675 × 382.5 pt | `3ccef2b96f63923b36a5901c8d32b2af7806cd5cf743fe9efbe6667a6af85ff3` |

The generator scripts and source data remain in the repository; the source ZIP contains the figure assets required for a clean LaTeX build. The Fig. 1 editable TikZ source hash is `0710921cac5dfc13304b9c7c68664d2cadee40a65664cce6e9ad75f8fab0fc2b`; Fig. 2 TikZ source hash is `bc20dab778a08f5bd414a7a07a902dcdcfed0226d6ae5472a540f18a65d414a4`.

## Verification

- Original PDF: 15 pages. Final TVT review PDF: 14 pages, with unchanged IEEEtran journal class, base font, margins, and line spacing. Active main-paper table bodies are 10 point; normal IEEE captions and figure lettering are audited separately.
- Full scientific regression: 33/33 registered scripts passed. These were regression/debug checks, not a rerun of frozen paper experiments.
- TVT LaTeX/BibTeX: PASS. Undefined references: 0; undefined citations: 0; LaTeX errors: 0; overfull boxes: 0.
- TVT source ZIP and SHA-256 manifest: rebuilt after the format patch; clean extraction and build findings are in `TVT_FORMAT_COMPLIANCE_REPORT.md`.
- Visual review: the actual final PDF was rendered at 300 dpi for Fig. 1 and at print scale for Figs. 2--5; all five figures passed readability, clipping, and text/line checks.
