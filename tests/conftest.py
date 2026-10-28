import pytest

from dal.config import DATA


def pytest_collection_modifyitems(config, items):
    if (DATA / "masks.npz").exists():
        return
    skip = pytest.mark.skip(reason="data/ not prepared (run scripts/prepare_data.sh)")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
