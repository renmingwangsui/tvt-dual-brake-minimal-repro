# Linux/WSL/VM autodiff migration workflow

Phase 2C-1.5 is blocked on the audited Windows host because Windows Application Control rejects the unsigned DLLs in the official PyTorch wheel. Do not disable Smart App Control or WDAC and do not use unsigned-binary workarounds. Use an administrator-approved Linux, WSL2, or Linux VM environment.

## Exact workflow

1. Obtain this repository or source archive and enter its root directory. Preserve filename case; in particular, the canonical claims file is exactly `docs/unverified_claims.md`.
2. Create and activate an isolated environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. Install official PyTorch for the selected target. CPU is sufficient for Phase 2C-1.5 and Phase 2C-2 correctness validation:

   ```bash
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

   For CUDA, obtain the exact command from the official PyTorch selector for that approved host. Do not reuse the Windows wheel or guess a CUDA build.

4. Install the repository dependency specification:

   ```bash
   python -m pip install -r environment/linux/requirements.txt
   ```

5. Verify real PyTorch autograd, module backward, optimizer update, device, version and dtype:

   ```bash
   python scripts/check_autodiff_backend.py
   ```

6. Run the complete accepted regression suite on the case-sensitive filesystem:

   ```bash
   python scripts/run_regression_suite.py
   ```

7. Only when both commands exit zero, record the verified PyTorch/Python/platform versions in the environment lock and restart Phase 2C-1.5 actor/critic/PPO migration.

Do not implement DiffQP, KKT differentiation, matched stop-gradient training, or predictive BPTT during this preparation. No output from these steps is paper eligible.

## Portability audit outcome

- Five host-specific absolute Windows path occurrences were found in generated/historical reports and removed or converted to repository-relative descriptions. The readiness validator now emits repository-relative POSIX paths for repository-owned files.
- Runtime Python under `src/`, `tests/`, `scripts/`, and `experiments/` contains no absolute Windows drive path and uses `pathlib` for repository paths.
- The previous PowerShell-only test command in the root README now calls the platform-neutral Python regression entrypoint. `scripts/build_manuscript.py` is the platform-neutral build entrypoint; `build.ps1` remains only as an optional Windows convenience wrapper.
- The exact lowercase `docs/unverified_claims.md` spelling remains enforced, including a duplicate-by-case check. The regression registry uses exact POSIX-style relative names.
- Text reads/writes use explicit UTF-8 where repository content is parsed. Python universal-newline behavior is used, and `.gitattributes` declares LF for portable text/source files while preserving CRLF only for the optional PowerShell wrapper.
- No simulator, safety, controller, MAPPO, physical parameter, formula, or experiment-claim behavior was changed by this portability preparation.
