"""Fail-fast preflight for an approved PyTorch execution environment."""
from __future__ import annotations

import json
import platform
import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:  # Import can fail with DLL/code-integrity errors, not only ImportError.
        print(json.dumps({
            "status": "BLOCKED_AUTODIFF_BACKEND",
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, indent=2))
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260917)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260917)

    x = torch.tensor(2.0, dtype=torch.float32, device=device, requires_grad=True)
    y = x * x
    y.backward()
    scalar_ok = x.grad is not None and torch.isfinite(x.grad) and torch.isclose(
        x.grad, torch.tensor(4.0, dtype=x.dtype, device=device), rtol=1e-6, atol=1e-6
    )

    module = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 1),
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    inputs = torch.ones((3, 2), dtype=torch.float32, device=device)
    loss = module(inputs).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in module.parameters()]
    module_backward_ok = all(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in gradients
    ) and any(bool(torch.count_nonzero(gradient)) for gradient in gradients if gradient is not None)
    before = tuple(parameter.detach().clone() for parameter in module.parameters())
    optimizer.step()
    optimizer_ok = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, module.parameters())
    )

    report = {
        "status": "PASS" if scalar_ok and module_backward_ok and optimizer_ok else "FAIL",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "default_dtype": str(torch.get_default_dtype()),
        "scalar_autograd_gradient": float(x.grad.detach().cpu()),
        "scalar_autograd_ok": bool(scalar_ok),
        "module_backward_ok": bool(module_backward_ok),
        "optimizer_step_ok": bool(optimizer_ok),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
