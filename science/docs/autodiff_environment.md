# Autodiff environment audit

Pre-installation snapshot: 17 September 2026.

| item | observed value |
|---|---|
| Python | 3.12.14 |
| executable / environment prefix | host-managed Codex Python runtime (historical Windows audit; not a repository dependency) |
| virtual environment | no separate `VIRTUAL_ENV`; executable prefix equals base prefix |
| OS | Windows 11, build 26200 |
| architecture | AMD64, 64 bit |
| pip | 26.2.1 |
| NumPy | 2.3.5 |
| SymPy | 1.14.0 |
| physical RAM | 16,905,965,568 bytes total; 4,963,860,480 bytes free at audit time |
| detected GPU | NVIDIA GeForce MX450, 2,048 MiB |
| NVIDIA driver | 581.95 |
| installed CUDA-enabled Python backend | none before installation |
| installed autodiff modules | PyTorch/JAX/JAXLIB/TensorFlow/Autograd all absent |
| existing QP solver | repository-local analytical weighted 2-D projection; no externally versioned QP package |

## Backend selection

PyTorch is selected as the sole canonical neural-training/autodiff backend. Python 3.12 is supported by current Windows PyTorch releases. Phase 2C-1.5 uses the official CPU wheel, so it does not depend on a CUDA toolkit or on the detected low-memory GPU. CUDA is optional and is not required for the Phase 2C-1.5/2C-2 DEBUG validation path.

The registered default training policy is `torch.float32` on CPU. Tests that compare migration numerics or future DiffQP Jacobians may explicitly request `torch.float64`; dtype changes must be deliberate rather than implicit.

## Installation outcome

The official CPU wheel `torch==2.14.0+cpu` was downloaded from `https://download.pytorch.org/whl/cpu`. Its first import initially reported the missing Microsoft Visual C++ Redistributable. Microsoft Visual C++ v14 Redistributable x64 version 14.51.36247.0 was then installed successfully through the Microsoft WinGet source, and all required `vcruntime140`/`msvcp140` DLLs became available.

Import still failed with Windows error 4551 while loading PyTorch's `shm.dll`: the active Windows Application Control/Smart App Control policy rejects the unsigned PyTorch core DLLs. Authenticode inspection reported `NotSigned` for `shm.dll`, `c10.dll`, `torch_cpu.dll`, `torch_python.dll`, and the other PyTorch core DLLs. This is an externally enforced code-integrity policy, not a missing Python package or CUDA requirement.

No attempt was made to disable Smart App Control, alter WDAC policy, remove security metadata, or substitute untrusted DLLs. The unusable PyTorch package was uninstalled so the accepted NumPy Phase 2C-1 environment remains operational. Consequently Phase 2C-1.5 is `BLOCKED_BACKEND_INSTALLATION`; no actor, critic, PPO, checkpoint, or DiffQP source migration was performed and no dependency lock was changed.
