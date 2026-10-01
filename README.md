# TVT dual-brake minimal reproduction package

This is a **public reproducibility snapshot** for *Feasibility-Certified Dual-Brake Control With Predictive Monitoring for Thermally Constrained Heavy-Duty Platoons*. The bundled 14-page manuscript is a historical snapshot, not the current IEEE TVT submission version; consult the submitted paper and supplementary material for final wording, metadata, and disclosures. The package contains no new or synthetic paper result.

The package has two intentionally separate roots:

- `paper/`: the active 14-page IEEE TVT LaTeX source, bibliography, vector figures, frozen figure inputs, deterministic figure generator, and source/claim mapping. `paper/manuscript_tvt/main.tex` is the entry point.
- `science/`: the accepted research-code snapshot, test registry, frozen configuration, parameter provenance, 92-run model campaign, 82-run follow-up, 20 formal-learning runs (including `.pt` checkpoints), and adapted-baseline evidence. `science/scripts/run_regression_suite.py` is the 33-script entry point.

The additional `science/results/paper_candidate/phase2m-paper-d10d13eb-d945f0ca/result_manifest.json` is included **only** because the frozen formal-learning integrity hash covers every historical Phase 2M manifest. Its raw run is not included and it is not used as accepted paper evidence. The accepted paper campaign is `phase2m-paper-d10d13eb-6634280c`.

## Reproduce the verified checks

Use Linux/WSL2/VM with Python and `venv`, TeX Live including `IEEEtran`, `latexmk`, `librsvg2-bin` (`rsvg-convert`), and Poppler (`pdfinfo`). CPU PyTorch is sufficient; CUDA is optional. The verified host used WSL2, Python 3.14.4, official PyTorch 2.14.0+cpu, NumPy 2.5.2, and SymPy 1.14.0. A different platform may use the appropriate official PyTorch wheel; no unverified wheel is bundled.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install official CPU/CUDA PyTorch using https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
bash scripts/verify.sh
```

`verify.sh` checks `SHA256SUMS`, genuine autograd, all 33 registered regression scripts, byte-identical regeneration of the three SVG figures from frozen inputs, a clean TeX build, 14 pages, 28 bibliography entries, and zero undefined citations/references, overfull boxes, or LaTeX/BibTeX errors. The paper rebuild is performed in a temporary copy so the checked-in frozen assets remain untouched.

This verifies the package and recomputes the display figures from archived inputs; it does **not** rerun the registered model-based experiments or ten-seed training. Those raw logs and manifests are supplied for independent inspection and deeper reproduction. Re-running scientific campaigns can depend on platform, PyTorch, and numerical environment and must not be reported as frozen paper evidence without a new provenance audit. The floating-point predictive output is a numerical monitor, not a formally sound interval certificate; a negative value is warning/inconclusive only. The external comparison is a domain-transfer stress test, not a native-domain ranking.

The snapshot originated from the audited research working tree at Git commit `682493bf13f0b2a1099eeacf6f1b6b7395bf6565`, plus the later TVT manuscript and figure revisions. The exact packaged bytes are identified by `SHA256SUMS`, not by that commit alone. The package intentionally omits virtual environments, caches, historical figure variants, unrelated drafts, and newly generated DEBUG artifacts. No reuse license is included with this public release.
