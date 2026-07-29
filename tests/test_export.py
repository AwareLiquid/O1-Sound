import subprocess
import sys
from pathlib import Path

BUDGET_MB = 7.0


def test_export_fits_budget_and_matches_torch(tmp_path):
    """The published artefact must fit 7 MB and agree with PyTorch.

    Run as a subprocess so this exercises the real CLI a release would use, not
    an in-process shortcut around it.
    """
    out = tmp_path / "o1sound.onnx"
    r = subprocess.run(
        [sys.executable, "export_onnx.py", "--out", str(out)],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    assert r.returncode == 0, r.stdout + r.stderr
    size_mb = out.stat().st_size / 1e6
    assert size_mb <= BUDGET_MB, f"{size_mb:.2f} MB exceeds {BUDGET_MB} MB"
    assert "max |onnx - torch|" in r.stdout


def test_export_fails_when_budget_is_impossible(tmp_path):
    """The gate must actually fail the build, not just print a warning."""
    out = tmp_path / "tiny.onnx"
    r = subprocess.run(
        [sys.executable, "export_onnx.py", "--out", str(out), "--budget-mb", "0.001"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    assert r.returncode == 1
    assert "exceeds" in r.stdout
