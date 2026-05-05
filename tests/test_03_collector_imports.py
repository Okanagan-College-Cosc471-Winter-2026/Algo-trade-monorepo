"""
TEST 03 — Collector import regression
─────────────────────────────────────
Protects the collector ORM compatibility layer by verifying that the
models module and the scripts that depend on it still import cleanly.
"""
import sys
from pathlib import Path


COLLECTOR_SRC = Path(__file__).parents[1] / "services/collector/src"
if str(COLLECTOR_SRC) not in sys.path:
    sys.path.insert(0, str(COLLECTOR_SRC))


def test_collector_model_runtime_imports():
    import model.models as models

    assert models.PipelineLog.__tablename__ == "pipeline_logs"
    assert "src.model.models" in sys.modules
    assert sys.modules["src.model.models"] is models


def test_collector_entrypoints_import():
    import run_15min_pipeline
    import run_scheduled_operations

    assert run_15min_pipeline is not None
    assert run_scheduled_operations is not None
