"""Tests for InversionConfig and MDM component configuration."""

import numpy as np
import pandas as pd
import pytest

from slv.inversion.config import (
    DEFAULT_MDM_CONFIG,
    InversionConfig,
    get_mdm_comp_configs,
)

# ---------------------------------------------------------------------------
# get_mdm_comp_configs
# ---------------------------------------------------------------------------


class TestGetMdmCompConfigs:
    def test_returns_all_default_components(self):
        components = get_mdm_comp_configs({})
        assert len(components) == len(DEFAULT_MDM_CONFIG)

    def test_component_order_matches_defaults(self):
        components = get_mdm_comp_configs({})
        assert [c["name"] for c in components] == list(DEFAULT_MDM_CONFIG.keys())

    def test_default_values_are_preserved(self):
        components = get_mdm_comp_configs({})
        for comp in components:
            defaults = DEFAULT_MDM_CONFIG[comp["name"]]
            for key, val in defaults.items():
                assert comp[key] == val

    def test_override_single_std(self):
        components = get_mdm_comp_configs({"part": {"std": 0.001}})
        part = next(c for c in components if c["name"] == "part")
        assert part["std"] == 0.001

    def test_override_does_not_affect_other_keys_in_same_component(self):
        components = get_mdm_comp_configs({"part": {"std": 0.001}})
        part = next(c for c in components if c["name"] == "part")
        assert part["correlated"] == DEFAULT_MDM_CONFIG["part"]["correlated"]

    def test_override_does_not_affect_other_components(self):
        components = get_mdm_comp_configs({"part": {"std": 0.001}})
        instr = next(c for c in components if c["name"] == "instr")
        assert instr["std"] == DEFAULT_MDM_CONFIG["instr"]["std"]

    def test_override_multiple_components(self):
        components = get_mdm_comp_configs(
            {"part": {"std": 0.001}, "bg": {"scale": "10d"}}
        )
        part = next(c for c in components if c["name"] == "part")
        bg = next(c for c in components if c["name"] == "bg")
        assert part["std"] == 0.001
        assert bg["scale"] == "10d"
        assert bg["std"] == DEFAULT_MDM_CONFIG["bg"]["std"]

    def test_extra_key_in_override_is_added(self):
        components = get_mdm_comp_configs({"part": {"new_key": "foo"}})
        part = next(c for c in components if c["name"] == "part")
        assert part["new_key"] == "foo"


# ---------------------------------------------------------------------------
# InversionConfig — basic defaults
# ---------------------------------------------------------------------------


class TestInversionConfigDefaults:
    def test_default_instantiation(self):
        config = InversionConfig()
        assert config.tstart == "2015-06-01"
        assert config.tend == "2025-02-01"
        assert config.sites == ["wbb"]
        assert config.flux_freq == "MS"
        assert config.cache is False

    def test_default_mdm_config_is_empty_dict(self):
        config = InversionConfig()
        assert config.mdm_config == {}

    def test_default_mdm_components_count(self):
        config = InversionConfig()
        assert len(config.mdm_components) == len(DEFAULT_MDM_CONFIG)

    def test_default_cache_overwrite_is_empty_list(self):
        config = InversionConfig()
        assert config.cache_overwrite == []

    def test_bias_std_is_none_by_default(self):
        config = InversionConfig()
        assert config.bias_std is None


# ---------------------------------------------------------------------------
# InversionConfig — mdm_config override
# ---------------------------------------------------------------------------


class TestInversionConfigMdm:
    def test_mdm_override_std(self):
        config = InversionConfig(mdm_config={"part": {"std": 0.001}})
        part = next(c for c in config.mdm_components if c["name"] == "part")
        assert part["std"] == 0.001

    def test_mdm_override_does_not_change_other_components(self):
        config = InversionConfig(mdm_config={"part": {"std": 0.001}})
        bg = next(c for c in config.mdm_components if c["name"] == "bg")
        assert bg["std"] == DEFAULT_MDM_CONFIG["bg"]["std"]


# ---------------------------------------------------------------------------
# InversionConfig — time properties
# ---------------------------------------------------------------------------


class TestInversionConfigTime:
    def test_time_range_returns_timestamps(self):
        config = InversionConfig(tstart="2020-01-01", tend="2021-01-01")
        t0, t1 = config.time_range
        assert t0 == pd.Timestamp("2020-01-01")
        assert t1 == pd.Timestamp("2021-01-01")

    def test_flux_times_monthly_length(self):
        config = InversionConfig(tstart="2020-01-01", tend="2020-04-01", flux_freq="MS")
        assert len(config.flux_times) == 3

    def test_flux_times_monthly_values(self):
        config = InversionConfig(tstart="2020-01-01", tend="2020-04-01", flux_freq="MS")
        assert config.flux_times[0] == pd.Timestamp("2020-01-01")
        assert config.flux_times[-1] == pd.Timestamp("2020-03-01")

    def test_flux_time_bins_count(self):
        config = InversionConfig(tstart="2020-01-01", tend="2020-04-01", flux_freq="MS")
        assert len(config.flux_time_bins) == 3

    def test_subset_hours_utc_conversion(self):
        # Mountain Standard Time is UTC-7
        config = InversionConfig(subset_hours=[12, 13], utc_offset=-7)
        assert config.subset_hours_utc == [19, 20]

    def test_subset_hours_utc_wraps_midnight(self):
        config = InversionConfig(subset_hours=[23], utc_offset=-1)
        # 23 - (-1) = 24 -> wraps to 0
        assert config.subset_hours_utc == [0]


# ---------------------------------------------------------------------------
# InversionConfig — spatial properties
# ---------------------------------------------------------------------------


class TestInversionConfigSpatial:
    @pytest.fixture
    def config(self):
        return InversionConfig(
            xmin=-112.0, xmax=-110.0, ymin=39.0, ymax=41.0, dx=0.1, dy=0.05
        )

    def test_bbox(self, config):
        assert config.bbox == (-112.0, 39.0, -110.0, 41.0)

    def test_extent(self, config):
        assert config.extent == (-112.0, -110.0, 39.0, 41.0)

    def test_map_extent_adds_buffer(self, config):
        xmin, xmax, ymin, ymax = config.map_extent
        assert xmin < config.xmin
        assert xmax > config.xmax
        assert ymin < config.ymin
        assert ymax > config.ymax

    def test_resolution(self, config):
        assert config.resolution == "0.1x0.05"


# ---------------------------------------------------------------------------
# InversionConfig — grid / state_grid
# ---------------------------------------------------------------------------

RESOLUTIONS = [0.002, 0.005, 0.01, 0.02, 0.025, 0.03, 0.05, 0.1, 0.15]


class TestInversionConfigGrid:
    """The prior's grid and the Jacobian's state_grid must enumerate the same cells:
    fips zero-fills Jacobian columns the prior has and the Jacobian lacks."""

    @pytest.mark.parametrize("d", RESOLUTIONS)
    def test_grid_matches_state_grid(self, d):
        config = InversionConfig(dx=d, dy=d)
        x, y = config.state_grid.axes
        np.testing.assert_array_equal(config.grid["lon"].values, x)
        np.testing.assert_array_equal(config.grid["lat"].values, y)

    @pytest.mark.parametrize("d", RESOLUTIONS)
    def test_cells_cover_domain_snapped_outward(self, d):
        config = InversionConfig(dx=d, dy=d)
        x, y = config.state_grid.axes
        for centres, lo, hi in (
            (x, config.xmin, config.xmax),
            (y, config.ymin, config.ymax),
        ):
            assert centres[0] - d / 2 == pytest.approx(lo)
            assert centres[-1] + d / 2 >= hi - 1e-9  # the domain is covered
            assert centres[-1] - d / 2 < hi - 1e-9  # by no more than one extra cell

    @pytest.mark.parametrize("d", [0.05, 0.1])
    def test_production_resolutions_unchanged(self, d):
        # The grid used to come from lair.geo.generate_regular_grid; keep those cells.
        from lair.geo import generate_regular_grid

        config = InversionConfig(dx=d, dy=d)
        old = generate_regular_grid(
            config.xmin,
            config.xmax,
            d,
            config.ymin,
            config.ymax,
            d,
            x_label="lon",
            y_label="lat",
        )
        np.testing.assert_array_equal(config.grid["lon"].values, old["lon"].values)
        np.testing.assert_array_equal(config.grid["lat"].values, old["lat"].values)

    def test_float_error_does_not_drop_last_row(self):
        # (40.93 - 40.45) / 0.01 = 47.99999999999969: stilt used to count 47 rows.
        config = InversionConfig(dx=0.01, dy=0.01)
        assert len(config.state_grid.axes[1]) == 48

    def test_centres_are_not_rounded_off(self):
        # lair rounded 0.025-deg centres to 3 decimals (40.9125 -> 40.912).
        lat = InversionConfig(dx=0.025, dy=0.025).grid["lat"].values
        assert 40.9125 in lat
        assert lat[-1] == pytest.approx(40.9375)

    def test_grid_crs(self):
        grid = InversionConfig().grid
        assert grid.dims == ("lat", "lon")
        assert grid.rio.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# InversionConfig — cache_overwrite variants
# ---------------------------------------------------------------------------


class TestInversionConfigCacheOverwrite:
    def test_cache_overwrite_list(self):
        config = InversionConfig(cache_overwrite=["obs", "prior_error"])
        assert config.cache_overwrite == ["obs", "prior_error"]

    def test_cache_overwrite_all_string(self):
        config = InversionConfig(cache_overwrite="all")
        assert config.cache_overwrite == "all"

    def test_cache_overwrite_single_string(self):
        config = InversionConfig(cache_overwrite="obs")
        assert config.cache_overwrite == "obs"


def test_stilt_project_dir_env_override(monkeypatch):
    from pathlib import Path

    from slv.inversion.config import DEFAULT_STILT_PROJECT, stilt_project_dir

    monkeypatch.delenv("SLV_STILT_DIR", raising=False)
    assert stilt_project_dir() == Path(DEFAULT_STILT_PROJECT)
    monkeypatch.setenv("SLV_STILT_DIR", "/tmp/other_stilt")
    assert stilt_project_dir() == Path("/tmp/other_stilt")


# ---------------------------------------------------------------------------
# InversionConfig — validation and derived values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"flux_freq": "W"},
        {"bias_grouping": "org"},
        {"background": "ctstilt"},
        {"prior": "vulcan"},
        {"mdm_config": {"transport_pbl": {"std": 0.1}}},
        {"dx": 0.0},
        {"xmin": -111.0, "xmax": -112.0},
    ],
)
def test_invalid_settings_fail_at_construction(bad):
    # these used to fail only after the Jacobian build or the solve
    with pytest.raises(ValueError):
        InversionConfig(**bad)


def test_derived_values_follow_field_changes():
    config = InversionConfig(dx=0.05, dy=0.05)
    assert config.grid.sizes["lon"] == 9
    config.dx = 0.1
    assert config.grid.sizes["lon"] == 5
    assert config.state_grid.xres == 0.1


def test_location_site_map_matches_within_a_metre():
    from slv.inversion.config import build_location_site_map
    from slv.measurements.sites import load_site_config

    site_config = load_site_config()
    exact = "-111.847672_40.766189_35"
    off_50m = "-111.848272_40.766189_35"  # 0.0006 deg of longitude
    mapper = build_location_site_map([exact, off_50m], site_config)
    assert mapper == {exact: "wbb"}
