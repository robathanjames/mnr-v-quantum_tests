import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from mnr_quantum import HybridModel, SimpleCNN, quantum_layer
from mnr_quantum.demo import main


def test_training_demo_is_reproducible_and_reduces_loss(capsys):
    arguments = [
        "--batch-size",
        "4",
        "--image-size",
        "16",
        "--qubits",
        "2",
        "--steps",
        "5",
        "--json",
    ]
    main(arguments)
    first = json.loads(capsys.readouterr().out)
    main(arguments)
    second = json.loads(capsys.readouterr().out)
    assert first == second
    assert first["final_loss"] < first["initial_loss"]
    assert len(first["training_losses"]) == 5
    assert len(first["logits"]) == 4
    assert all(len(row) == 2 for row in first["logits"])
    assert math.isfinite(first["final_loss"])


def test_forward_only_demo(capsys):
    main(["--batch-size", "1", "--json"])
    output = json.loads(capsys.readouterr().out)
    assert output["steps"] == 0
    assert output["training_losses"] == []
    assert len(output["logits"]) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["--batch-size", "0"],
        ["--qubits", "9"],
        ["--image-size", "3"],
        ["--steps", "-1"],
        ["--shots", "0"],
        ["--seed", "-1"],
        ["--seed", str(2**32)],
        ["--learning-rate", "nan"],
        ["--learning-rate", "0"],
    ],
)
def test_invalid_cli_arguments(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_original_script_still_exports_public_components(capsys):
    script = Path(__file__).resolve().parents[1] / "mnr-v_c.py"
    spec = importlib.util.spec_from_file_location("legacy_entry", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.HybridModel is HybridModel
    assert module.SimpleCNN is SimpleCNN
    assert module.quantum_layer is quantum_layer
    assert capsys.readouterr().out == ""
    assert module.HybridModel()(torch.zeros(1, 1, 28, 28)).shape == (1, 10)
