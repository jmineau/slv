"""Bias state block for :class:`~slv.inversion.pipelines.SLVMethaneInversion`."""

import pandas as pd
from fips import Vector

from slv.inversion.config import InversionConfig


class BiasMixin:
    """Bias terms grouped by time, site or site organization (``config.bias_std`` /
    ``config.bias_grouping``). Expects the host pipeline's ``_resolve_sites``."""

    config: InversionConfig

    def get_site_group(self, site: str) -> str:
        """Map site to organization group."""
        return self.config.site_config.organization.to_dict().get(site, "unknown")

    def get_bias(self) -> pd.Series:
        """Build the bias prior based on config.bias_grouping.

        Returns a zero-valued Series with index determined by bias_grouping:
          - None or "time": one bias per time interval
          - "site": one bias per (time, obs_location)
          - "site_group": one bias per (time, organization)

        Override this method for non-zero initial values or custom groupings.
        """
        grouping = self.config.bias_grouping

        if grouping in (None, "time"):
            # Time-only bias (default)
            index = pd.Index(self.config.flux_times, name="time")

        elif grouping == "site":
            # Per-site bias
            index = pd.MultiIndex.from_product(
                [self.config.flux_times, self.config.sites],
                names=["time", "obs_location"],
            )

        elif grouping == "site_group":
            # Per-site-group (organization) bias - only for configured sites
            site_groups = (
                self.config.site_config.loc[self.config.sites, "organization"]
                .unique()
                .tolist()
            )
            index = pd.MultiIndex.from_product(
                [self.config.flux_times, site_groups], names=["time", "site_group"]
            )

        else:
            raise ValueError(
                f"Unknown bias_grouping: {grouping}. "
                f"Expected None, 'time', 'site', or 'site_group'"
            )

        return pd.Series(0.0, index=index, name="bias")

    def _obs_bias_sites(self, obs_index, location_mapper=None) -> pd.Index:
        """Each obs's site: STILT location IDs through the location map, then receptors
        to the mobile site (:meth:`obs_sites`)."""
        mapper = location_mapper or self.config.location_site_map or {}
        locations = obs_index.get_level_values("obs_location")
        return self._resolve_sites(locations.map(lambda loc: mapper.get(loc, loc)))

    def get_bias_jacobian(
        self, obs: Vector, prior: Vector, location_mapper: dict | None = None
    ) -> pd.DataFrame:
        """Build the obs × bias Jacobian based on config.bias_grouping.

        Maps each observation to its corresponding bias term:
          - time: match by time interval only
          - site: match by (time, obs_location)
          - site_group: match by (time, organization)

        ``location_mapper`` (STILT location ID -> site) defaults to
        ``config.location_site_map``.
        """
        obs_index = obs["concentration"].index
        bias_index = prior["bias"].index
        obs_times = obs_index.get_level_values("obs_time")
        grouping = self.config.bias_grouping

        # Bin obs times into flux intervals
        cut = pd.cut(obs_times, bins=self.config.flux_time_bins)
        flux_times = cut.map(lambda iv: iv.left if pd.notna(iv) else None)

        if grouping in (None, "time"):
            # Time-only: simple one-hot encoding
            jac = pd.get_dummies(cut, dtype=float)
            jac.columns = jac.columns.map(lambda iv: iv.left)
            jac.index = obs_index

        elif grouping == "site":
            # Per-site: match (time, site); a mobile receptor belongs to its mobile site
            obs_sites = self._obs_bias_sites(obs_index, location_mapper)
            bias_keys = pd.Series(
                list(zip(flux_times, obs_sites, strict=True)),
                index=obs_index,
                dtype=object,
            )
            jac = pd.get_dummies(bias_keys, dtype=float)
            jac.columns = pd.MultiIndex.from_tuples(
                jac.columns, names=["time", "obs_location"]
            )

        elif grouping == "site_group":
            # Per-site-group: match (time, organization) of each obs's site
            obs_sites = self._obs_bias_sites(obs_index, location_mapper)
            obs_site_groups = obs_sites.map(self.get_site_group)
            bias_keys = pd.Series(
                list(zip(flux_times, obs_site_groups, strict=True)),
                index=obs_index,
                dtype=object,
            )
            jac = pd.get_dummies(bias_keys, dtype=float)
            jac.columns = pd.MultiIndex.from_tuples(
                jac.columns, names=["time", "site_group"]
            )

        else:
            raise ValueError(f"Unknown bias_grouping: {grouping}")

        # Align to bias index (handles any time-range trimming)
        return jac.reindex(columns=bias_index, fill_value=0.0)
