import pickle

import pytest

from slv._files import replacing
from slv.inversion.cache import fips_cache


def test_a_failed_write_leaves_the_old_file(tmp_path):
    target = tmp_path / "obs.parquet"
    target.write_text("old")
    with pytest.raises(RuntimeError), replacing(target) as tmp:
        tmp.write_text("half")
        raise RuntimeError("killed mid-write")
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_a_cache_entry_that_fails_to_pickle_is_not_left_behind(tmp_path):
    class Config:
        cache = tmp_path
        cache_overwrite = []

    @fips_cache("unpicklable")
    def build(config):
        return lambda: None  # a lambda cannot be pickled

    with pytest.raises((pickle.PicklingError, AttributeError)):
        build(Config())
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]
