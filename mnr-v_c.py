"""Original entry point; reusable implementations live in :mod:`mnr_quantum`."""

from mnr_quantum import HybridModel, SimpleCNN, quantum_layer  # noqa: F401
from mnr_quantum.demo import main

if __name__ == "__main__":
    main()
