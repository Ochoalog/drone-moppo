"""POA configs point only to the POA folders; the Taquari defaults are unchanged."""
from dal.config import DATA, RESULTS, ROOT, RUNS, default_config, load_config, paths


def test_poa_paths_are_separate():
    cfg = load_config("configs_poa/poa.yaml")
    p = paths(cfg)
    assert p["data"] == ROOT / "data_poa" and p["results"] == ROOT / "results_poa" and p["runs"] == ROOT / "runs_poa"
    for k in ("data", "results", "runs"):
        assert p[k] not in (ROOT / "data", ROOT / "results", ROOT / "runs", ROOT / "runs" / "overnight")
    assert cfg["paths"]["report"] == "REPORT_POA.md"


def test_poa_sources_and_scenarios_replace_taquari():
    cfg = load_config("configs_poa/poa.yaml")
    assert set(cfg["data"]["sources"]) == {"poa_post", "poa_pre_feb24", "poa_pre_dec22"}
    assert set(cfg["scenarios"]) == {"P0", "P1", "P2"}
    assert cfg["env"]["descriptor"] == "hand"
    assert set(cfg["regions"]) == {"ilhas", "centro"}


def test_taquari_defaults_unchanged():
    p = paths(default_config())
    assert p["data"] == DATA == ROOT / "data"
    assert p["results"] == RESULTS == ROOT / "results"
    assert p["runs"] == RUNS == ROOT / "runs"
