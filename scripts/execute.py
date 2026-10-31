"""Execute notebooks in fresh kernels using the active Python interpreter.

No user kernel registration, API credentials, model download or private input
is required. The temporary kernelspec is removed when execution finishes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def execute_all():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib"))
    receipt = []
    with tempfile.TemporaryDirectory(prefix="research-preview-kernel-") as directory:
        kernels = Path(directory) / "kernels"
        spec = kernels / "research-preview"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python (research preview)", "language": "python"
        }))

        class ActivePythonClient(NotebookClient):
            def create_kernel_manager(self):
                self.km = KernelManager(kernel_name="research-preview", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
                return self.km

        for path in sorted((ROOT / "notebooks").glob("*_EXECUTED.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            client = ActivePythonClient(notebook, timeout=180, kernel_name="research-preview", resources={"metadata": {"path": str(ROOT)}})
            client.execute()
            code = [cell for cell in notebook.cells if cell.cell_type == "code"]
            errors = [output for cell in code for output in cell.outputs if output.output_type == "error"]
            if errors or any(cell.execution_count is None for cell in code):
                raise RuntimeError(f"Incomplete notebook: {path.name}")
            notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            receipt.append({"notebook": str(path.relative_to(ROOT)), "executed_code_cells": len(code), "errors": len(errors)})
            print(f"Executed {path.name}: {len(code)} code cells, no errors")
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "execution_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    execute_all()
