"""Parameter sweep harness for SLV methane inversions.

Run many :class:`~slv.inversion.pipelines.SLVMethaneInversion` configurations
in one pass, collect goodness-of-fit metrics (targeting reduced chi² ≈ 1), and
analyse which parameter combinations produce stable, well-fitting inversions.

Quick start — single-node sweep
--------------------------------
.. code-block:: python

    from slv.inversion.sweep import Sweep, SweepResults
    from slv.inversion.pipelines import SLVMethaneInversion

    # Create and run a parameter sweep
    sweep = Sweep(
        cache="/scratch/my_sweep/cache",
        pipeline_cls=SLVMethaneInversion,
        prior_base_std=[0.01, 0.019, 0.03],
        prior_std_frac=[0.3, 0.5, 0.7],
        bg_baseline_window=["7d", "14d", "21d"],
        mdm_config=[
            {},  # all defaults
            {"transport_pbl": {"std": 0.10}},
            {"transport_pbl": {"std": 0.20}},
        ],
    )
    # 3 × 3 × 3 × 3 = 81 configs
    results = sweep.run(results_dir="/scratch/my_sweep", n_jobs=8)

    best = results.best(tol=0.1)  # chi² ∈ [0.9, 1.1]
    results.plot_chi2("cfg_prior_base_std")
    cfg, cls = results.to_config(best.iloc[0])

SLURM job-array usage
----------------------
.. code-block:: bash

    # 1. Generate the grid without running (n_jobs=0)
    python -c "
    from slv.inversion.sweep import Sweep
    sweep = Sweep(cache='/scratch/sweep/cache', ...)
    sweep.run(results_dir='/scratch/sweep', n_jobs=0)
    "

    # 2. Submit one job per config
    sbatch --array=0-80 job.sh

    # job.sh:
    # python -c "
    # from slv.inversion.sweep import run_sweep_job
    # run_sweep_job('/scratch/sweep')
    # "
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import itertools
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from slv.inversion.config import InversionConfig
from slv.inversion.pipelines import (
    SLVMethaneInversion,
)

logger = logging.getLogger(__name__)

# Fields excluded from config_id and swept_params capture (non-scientific)
_NON_SCIENTIFIC_FIELDS = frozenset(
    {
        "cache",
        "cache_overwrite",
        "tiler",
        "tiler_zoom",
        "plot_inputs",
        "plot_results",
        "plot_diagnostics",
        "stilt_path",
        "num_processes",
        "timeout",
    }
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_json(v: Any) -> Any:
    """Recursively convert a value to a JSON-serializable form."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_to_json(i) for i in v]
    if isinstance(v, dict):
        return {k: _to_json(vv) for k, vv in sorted(v.items())}
    return str(v)


# ---------------------------------------------------------------------------
# Config identification
# ---------------------------------------------------------------------------


def config_id(config: InversionConfig) -> str:
    """Return a stable 16-char hex identifier for a config.

    Excludes non-scientific fields (cache paths, plotting flags, etc.) so that
    changing only where to write outputs does not produce a new ID.
    """
    fields = {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(config)
        if f.name not in _NON_SCIENTIFIC_FIELDS
    }
    serialized = json.dumps(_to_json(fields), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Sweep class
# ---------------------------------------------------------------------------


class Sweep:
    """Object-oriented interface for parameter sweeps.

    Examples
    --------
    **Basic usage:**

    .. code-block:: python

        sweep = Sweep(
            cache="./cache",
            pipeline_cls=SLVMethaneInversion,
            base_config=base,
            prior_base_std=[0.01, 0.02],
            gamma=[1.0, 3.0],
        )
        results = sweep.run(results_dir="./my_sweep", n_jobs=40)
        problem = sweep.get_problem(results.best().iloc[0]["config_id"])

    **Resume/merge with existing grid:**

    .. code-block:: python

        sweep = Sweep(
            cache="./cache",
            grid_path="./my_sweep/sweep_grid.json",  # Load existing
            prior_base_std=[0.03, 0.04],  # Add new configs
        )
        results = sweep.run(results_dir="./my_sweep", n_jobs=40, resume=True)

    **SLURM job array:**

    .. code-block:: python

        # Generate grid without running
        sweep = Sweep(cache="./cache", prior_base_std=[...], ...)
        sweep.run(results_dir="./my_sweep", n_jobs=0)

        # Submit SLURM job array (uses run_sweep_job)
        # sbatch --array=0-N job.sh
    """

    def __init__(
        self,
        cache: str | Path,
        pipeline_cls: type[SLVMethaneInversion] = SLVMethaneInversion,
        base_config: InversionConfig | None = None,
        grid_path: str | Path | None = None,
        **sweep_kwargs: list[Any],
    ):
        """Initialize sweep with configs.

        Parameters
        ----------
        cache :
            Cache directory for inversion components.
        pipeline_cls :
            Pipeline class to use for inversions.
        base_config :
            Base configuration for all configs.
        grid_path :
            Optional path to existing grid JSON. If provided, configs are loaded
            from the grid and merged with any new configs from ``**sweep_kwargs``.
        **sweep_kwargs :
            Parameter sweep ranges (e.g., ``prior_base_std=[0.01, 0.02]``).
        """
        self.cache = Path(cache)
        self.pipeline_cls = pipeline_cls
        self.base_config = base_config
        self.sweep_kwargs = sweep_kwargs

        self.configs: list[tuple[InversionConfig, type]] = []
        self._config_map: dict[str, tuple[InversionConfig, type]] = {}

        self._build_configs(grid_path)

    def _build_configs(self, grid_path: str | Path | None) -> None:
        """Build configs from grid (if exists) and/or sweep_kwargs."""
        existing_ids: set[str] = set()

        # Load from existing grid if provided
        if grid_path:
            grid_path = Path(grid_path)
            if grid_path.exists():
                with open(grid_path) as f:
                    grid = json.load(f)

                for entry in grid:
                    fields = entry["fields"]
                    valid = {
                        f.name
                        for f in dataclasses.fields(InversionConfig)
                        if f.name != "tiler"
                    }
                    cfg = InversionConfig(
                        **{k: v for k, v in fields.items() if k in valid}
                    )

                    module_path, cls_name = entry["pipeline_cls"].rsplit(".", 1)
                    cls = getattr(importlib.import_module(module_path), cls_name)

                    cid = config_id(cfg)
                    self.configs.append((cfg, cls))
                    self._config_map[cid] = (cfg, cls)
                    existing_ids.add(cid)

                print(f"Loaded {len(grid)} configs from {grid_path}")

        # Generate new configs from sweep_kwargs
        if self.sweep_kwargs:
            # Validate sweep parameters are InversionConfig fields
            valid_fields = {f.name for f in dataclasses.fields(InversionConfig)}
            unknown = set(self.sweep_kwargs.keys()) - valid_fields
            if unknown:
                raise ValueError(
                    f"Unknown InversionConfig field(s): {', '.join(sorted(unknown))}"
                )

            # Cartesian product of sweep parameters
            base_fields = (
                {
                    f.name: getattr(self.base_config, f.name)
                    for f in dataclasses.fields(self.base_config)
                }
                if self.base_config
                else {}
            )

            param_names = list(self.sweep_kwargs.keys())
            param_values = list(self.sweep_kwargs.values())

            added = 0
            for combo in itertools.product(*param_values):
                overrides = dict(zip(param_names, combo, strict=True))
                all_fields = {**base_fields, **overrides, "cache": str(self.cache)}
                cfg = InversionConfig(
                    **{k: v for k, v in all_fields.items() if k in valid_fields}
                )

                cid = config_id(cfg)
                if cid not in existing_ids:
                    self.configs.append((cfg, self.pipeline_cls))
                    self._config_map[cid] = (cfg, self.pipeline_cls)
                    existing_ids.add(cid)
                    added += 1

            if added > 0:
                print(f"Added {added} new configs from sweep parameters")

        if not self.configs:
            raise ValueError(
                "No configs to sweep. Provide either grid_path or **sweep_kwargs."
            )

        print(f"Sweep initialized with {len(self.configs)} total configs")

    def run(
        self,
        results_dir: str | Path,
        n_jobs: int = 1,
        swept_params: list[str] | None = None,
        resume: bool = True,
        config_ids: list[str] | None = None,
    ) -> SweepResults:
        """Run the sweep.

        Parameters
        ----------
        results_dir :
            Directory for results CSV and grid JSON.
        n_jobs :
            Number of parallel workers (0 = write grid only, no runs).
        swept_params :
            Config fields to capture as columns. None = all scientific fields.
        resume :
            Skip configs already in results CSV.
        config_ids :
            Optional list of config IDs to run. If None, run all configs.
            Use this to run a random sample of the full sweep.

        Returns
        -------
        SweepResults wrapping the output CSV.
        """
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "sweep_results.csv"
        grid_path = results_dir / "sweep_grid.json"

        # Write grid for SLURM compatibility
        self._write_grid(grid_path)

        if n_jobs == 0:
            print("n_jobs=0: grid written, no runs executed.")
            return (
                SweepResults(csv_path)
                if csv_path.exists()
                else SweepResults.__new__(SweepResults)
            )

        # Determine which configs still need to run
        done_ids: set[str] = set()
        if resume and csv_path.exists():
            existing = pd.read_csv(csv_path)
            if "config_id" in existing.columns:
                done_ids = set(existing["config_id"].dropna().tolist())
            skipped = sum(1 for cfg, _ in self.configs if config_id(cfg) in done_ids)
            remaining = len(self.configs) - skipped
            print(f"Resuming: {skipped} already done, {remaining} remaining.")

        # Filter by config_ids if provided
        target_ids = set(config_ids) if config_ids is not None else None
        if target_ids is not None:
            print(f"Targeting {len(target_ids)} specific config IDs")

        pending = [
            (cfg, cls, str(results_dir), swept_params, n_jobs > 1)
            for cfg, cls in self.configs
            if config_id(cfg) not in done_ids
            and (target_ids is None or config_id(cfg) in target_ids)
        ]

        if not pending:
            print("All configs already completed.")
            return SweepResults(csv_path)

        # Run sweep (sequential or parallel)
        if n_jobs == 1:
            print("Running sweep sequentially (n_jobs=1)...")
            try:
                from tqdm import tqdm

                iterator = tqdm(pending, desc="Sweep", unit="run")
            except ImportError:
                iterator = iter(pending)
            for args in iterator:
                self._append_row(csv_path, self._run_single(args))
        else:
            print(
                f"Running sweep in parallel (n_jobs={n_jobs}). "
                f"Output suppressed - check errors.log for failures."
            )
            # Use loky backend for robust nested parallelism
            rows = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
                delayed(self._run_single)(args) for args in pending
            )
            try:
                from tqdm import tqdm

                rows = tqdm(rows, total=len(pending), desc="Sweep", unit="run")
            except ImportError:
                pass
            for row in rows:
                self._append_row(csv_path, row)

        print(f"Sweep complete. Results at {csv_path}")
        return SweepResults(csv_path)

    def _run_single(
        self,
        args: tuple[InversionConfig, type, str, list[str] | None, bool],
    ) -> dict[str, Any]:
        """Run one inversion and return a metrics dict.

        Catches all exceptions so a single failed run does not abort the whole sweep.
        """
        config, pipeline_cls, results_dir, swept_params, suppress_output = args
        cid = config_id(config)

        # Suppress plots in sweep runs
        config.plot_inputs = False
        config.plot_results = False
        config.plot_diagnostics = False

        try:
            # Suppress stdout/stderr when running in parallel
            if suppress_output:
                with _suppress_output():
                    pipeline = pipeline_cls(config)
                    problem = pipeline.run()
            else:
                pipeline = pipeline_cls(config)
                problem = pipeline.run()
            return collect_metrics(problem, config, pipeline, swept_params=swept_params)
        except Exception as exc:
            tb = traceback.format_exc()
            # Log to file instead of printing full traceback
            error_log = Path(results_dir) / "errors.log"
            with open(error_log, "a") as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"Config: {cid}\n")
                f.write(f"Error: {exc}\n")
                f.write(tb)
            logger.error(f"[{cid}] Run failed: {exc}")
            if not suppress_output:
                print(f"[{cid}] Error: {exc}")

            # Return same column structure as collect_metrics, with None values
            error_row: dict[str, Any] = {
                "config_id": cid,
                "error": str(exc),
                "runtime_seconds": None,
                "reduced_chi2": None,
                "chi2_distance": None,
                "R2": None,
                "RMSE": None,
                "DOFS": None,
                "uncertainty_reduction": None,
                "prior_flux_mean": None,
                "posterior_flux_mean": None,
                "posterior_flux_std": None,
                "flux_change_pct": None,
                "prior_conc_rmse": None,
                "posterior_conc_rmse": None,
                "prior_conc_bias": None,
                "posterior_conc_bias": None,
                "n_obs": None,
                "n_state": None,
            }

            # Add swept config params
            if swept_params is None:
                swept_params = [
                    f.name
                    for f in dataclasses.fields(config)
                    if f.name not in _NON_SCIENTIFIC_FIELDS
                ]
            for p in swept_params:
                val = getattr(config, p, None)
                if isinstance(val, (dict, list)):
                    val = json.dumps(_to_json(val), sort_keys=True)
                else:
                    val = _to_json(val)
                error_row[f"cfg_{p}"] = val

            return error_row

    @staticmethod
    def _append_row(csv_path: Path, row: dict | None) -> None:
        """Atomically append one row to the results CSV."""
        if row is None:
            return
        pd.DataFrame([row]).to_csv(
            csv_path, mode="a", header=not csv_path.exists(), index=False
        )

    def _write_grid(self, path: Path) -> None:
        """Write configs to grid JSON for SLURM job arrays."""
        grid = []
        for i, (cfg, cls) in enumerate(self.configs):
            entry = {
                "index": i,
                "config_id": config_id(cfg),
                "pipeline_cls": f"{cls.__module__}.{cls.__qualname__}",
                "fields": _to_json(
                    {
                        f.name: getattr(cfg, f.name)
                        for f in dataclasses.fields(cfg)
                        if f.name != "tiler"
                    }
                ),
            }
            grid.append(entry)

        with open(path, "w") as f:
            json.dump(grid, f, indent=2)

    def get_problem(self, config_id: str):
        """Reconstruct and run a single config by ID.

        Parameters
        ----------
        config_id :
            Config ID from results CSV.

        Returns
        -------
        FluxProblem instance from running the pipeline.

        Raises
        ------
        ValueError
            If config_id not found in sweep.
        """
        if config_id not in self._config_map:
            raise ValueError(
                f"Config {config_id} not found in sweep. "
                f"Available IDs: {list(self._config_map.keys())[:10]}..."
            )

        cfg, cls = self._config_map[config_id]
        pipeline = cls(cfg)
        return pipeline.run()

    def __len__(self) -> int:
        """Return number of configs in sweep."""
        return len(self.configs)

    def __repr__(self) -> str:
        return f"Sweep({len(self)} configs, pipeline={self.pipeline_cls.__name__})"


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------


def collect_metrics(
    problem: Any,
    config: InversionConfig,
    pipeline: SLVMethaneInversion,
    swept_params: list[str] | None = None,
) -> dict[str, Any]:
    """Extract goodness-of-fit and flux metrics from a solved ``FluxProblem``.

    The primary target metric is ``reduced_chi2`` — ideally close to 1.0,
    which indicates the error covariances are well-specified.

    Parameters
    ----------
    problem :
        Solved ``FluxProblem`` returned by ``pipeline.run()``.
    config :
        :class:`InversionConfig` used for this run.
    pipeline :
        The pipeline instance (used for ``calculate_total_flux``).
    swept_params :
        Config field names to capture as ``cfg_*`` columns in the output row.
        If *None*, all non-scientific fields are excluded and the rest included.

    Returns
    -------
    Flat :class:`dict` ready to append as a row to the results CSV.
    """
    est = problem.estimator

    metrics: dict[str, Any] = {
        "config_id": config_id(config),
        # Primary chi² metric — target ≈ 1.0
        "reduced_chi2": float(est.reduced_chi2),
        "chi2_distance": abs(float(est.reduced_chi2) - 1.0),
        # Supplementary fit metrics
        "R2": float(est.R2),
        "RMSE": float(est.RMSE),
        "DOFS": float(est.DOFS),
        "uncertainty_reduction": float(est.uncertainty_reduction),
    }

    # Domain-integrated flux totals
    try:
        total_prior = pipeline.calculate_total_flux(
            problem.prior_fluxes, units=config.output_units
        )
        total_posterior = pipeline.calculate_total_flux(
            problem.posterior_fluxes, units=config.output_units
        )
        metrics["prior_flux_mean"] = float(total_prior.mean())
        metrics["posterior_flux_mean"] = float(total_posterior.mean())
        metrics["posterior_flux_std"] = float(total_posterior.std())
        prior_mean = total_prior.mean()
        if prior_mean != 0:
            metrics["flux_change_pct"] = float(
                (total_posterior.mean() - prior_mean) / prior_mean * 100
            )
        else:
            metrics["flux_change_pct"] = None
    except Exception:
        logger.debug("collect_metrics: flux totals failed", exc_info=True)
        metrics["prior_flux_mean"] = metrics["posterior_flux_mean"] = None
        metrics["posterior_flux_std"] = metrics["flux_change_pct"] = None

    # Concentration residuals
    try:
        obs_enh = problem.enhancement
        prior_c = problem.prior_concentrations
        post_c = problem.posterior_concentrations
        rp = obs_enh - prior_c
        rq = obs_enh - post_c
        metrics["prior_conc_rmse"] = float((rp**2).mean() ** 0.5)
        metrics["posterior_conc_rmse"] = float((rq**2).mean() ** 0.5)
        metrics["prior_conc_bias"] = float(rp.mean())
        metrics["posterior_conc_bias"] = float(rq.mean())
        metrics["n_obs"] = int(len(obs_enh))
        metrics["n_state"] = int(problem.prior_fluxes.shape[0])
    except Exception:
        logger.debug("collect_metrics: concentration residuals failed", exc_info=True)

    # Capture swept config field values as cfg_* columns
    if swept_params is None:
        swept_params = [
            f.name
            for f in dataclasses.fields(config)
            if f.name not in _NON_SCIENTIFIC_FIELDS
        ]
    for p in swept_params:
        val = getattr(config, p, None)
        # JSON-encode complex types so they survive a CSV round-trip
        if isinstance(val, (dict, list)):
            metrics[f"cfg_{p}"] = json.dumps(_to_json(val))
        else:
            metrics[f"cfg_{p}"] = val

    return metrics


# ---------------------------------------------------------------------------
# Output suppression context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# ---------------------------------------------------------------------------
# SLURM job-array entry point
# ---------------------------------------------------------------------------


def run_sweep_job(results_dir: str | Path) -> None:
    """Run a single config from a pre-serialised sweep grid (SLURM job arrays).

    Reads ``$SLURM_ARRAY_TASK_ID`` (falls back to ``--index`` CLI argument) to
    pick which config to run.  Writes one row to
    ``{results_dir}/sweep_results.csv``.

    Typical usage::

        python -c "
        from slv.inversion.sweep import run_sweep_job
        run_sweep_job('/scratch/my_sweep')
        "

    Submitted as::

        sbatch --array=0-<N-1> job.sh
    """
    import argparse

    results_dir = Path(results_dir)
    grid_path = results_dir / "sweep_grid.json"

    if not grid_path.exists():
        raise FileNotFoundError(
            f"sweep_grid.json not found at {grid_path}. "
            "Call run_sweep(..., n_jobs=0) first to generate the grid."
        )

    # Resolve index: SLURM env var > --index CLI flag
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx_str is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--index", type=int, required=True)
        args, _ = parser.parse_known_args()
        idx_str = str(args.index)
    idx = int(idx_str)

    with open(grid_path) as fh:
        grid = json.load(fh)

    if idx >= len(grid):
        raise IndexError(f"Index {idx} out of range (grid has {len(grid)} entries).")

    entry = grid[idx]
    fields = entry["fields"]

    # Rehydrate InversionConfig (skip tiler — not JSON-serialisable)
    valid = {f.name for f in dataclasses.fields(InversionConfig) if f.name != "tiler"}
    cfg = InversionConfig(**{k: v for k, v in fields.items() if k in valid})

    # Rehydrate pipeline class
    module_path, cls_name = entry["pipeline_cls"].rsplit(".", 1)
    pipeline_cls = getattr(importlib.import_module(module_path), cls_name)

    cid = config_id(cfg)
    csv_path = results_dir / "sweep_results.csv"

    # Skip if already done (e.g. job resubmitted after partial failure)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        if "config_id" in existing.columns and cid in existing["config_id"].values:
            print(f"[{idx}] {cid} already done — skipping.")
            return

    # Run inversion and collect metrics
    cfg.plot_inputs = False
    cfg.plot_results = False
    cfg.plot_diagnostics = False

    try:
        pipeline = pipeline_cls(cfg)
        problem = pipeline.run()
        row = collect_metrics(problem, cfg, pipeline, swept_params=None)
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        error_log = results_dir / "errors.log"
        with open(error_log, "a") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Config: {cid}\n")
            f.write(f"Error: {exc}\n")
            f.write(tb)
        print(f"[{idx}] {cid} → error: {exc}")
        row = {"config_id": cid, "error": str(exc)}

    pd.DataFrame([row]).to_csv(
        csv_path, mode="a", header=not csv_path.exists(), index=False
    )
    status = "error" if row.get("error") else "ok"
    print(f"[{idx}] {cid} → {status}")


# ---------------------------------------------------------------------------
# Results analysis
# ---------------------------------------------------------------------------


class SweepResults:
    """Wraps ``sweep_results.csv`` for post-sweep analysis.

    Attributes
    ----------
    path : pathlib.Path
        Path to the results CSV.
    df : pandas.DataFrame
        All completed rows, sorted by ``chi2_distance`` (closest to 1 first).

    Examples
    --------
    .. code-block:: python

        results = SweepResults("/scratch/my_sweep/sweep_results.csv")

        # Best-fitting configs (chi² ∈ [0.9, 1.1])
        results.best(tol=0.1)

        # Which values of prior_base_std give a stable posterior flux?
        results.stability("cfg_prior_base_std")

        # Are results more sensitive to prior or MDM params?
        results.sensitivity()

        # 1-D scatter: chi² vs a single parameter
        results.plot_chi2("cfg_prior_base_std")

        # 2-D heatmap: chi² over two parameters
        results.plot_heatmap("cfg_prior_base_std", "cfg_prior_std_frac")

        # Rehydrate best config for a reproduction run
        cfg, cls = results.to_config(results.best().iloc[0])
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._reload()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _reload(self) -> None:
        self.df = pd.read_csv(self.path)
        if "chi2_distance" not in self.df.columns and "reduced_chi2" in self.df.columns:
            self.df["chi2_distance"] = (self.df["reduced_chi2"] - 1.0).abs()
        if "chi2_distance" in self.df.columns:
            self.df = self.df.sort_values("chi2_distance").reset_index(drop=True)

    def reload(self) -> SweepResults:
        """Re-read from disk (useful while a sweep is still running)."""
        self._reload()
        return self

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def best(self, target: float = 1.0, tol: float = 0.1) -> pd.DataFrame:
        """Return rows where ``|reduced_chi2 - target| < tol``, best first.

        Parameters
        ----------
        target :
            Ideal chi² value (default 1.0).
        tol :
            Acceptance window around *target* (default ±0.1).
        """
        if "reduced_chi2" not in self.df.columns:
            return self.df.iloc[:0]
        dist = (self.df["reduced_chi2"] - target).abs()
        return self.df[dist < tol].copy().reset_index(drop=True)

    def failed(self) -> pd.DataFrame:
        """Return rows that errored during the sweep."""
        if "error" not in self.df.columns:
            return self.df.iloc[:0]
        return self.df[self.df["error"].notna()].copy()

    # ------------------------------------------------------------------
    # Stability / sensitivity analysis
    # ------------------------------------------------------------------

    def stability(
        self,
        groupby_params: str | list[str],
        metric: str = "posterior_flux_mean",
    ) -> pd.DataFrame:
        """Standard deviation of *metric* grouped by *groupby_params*.

        A low std indicates those parameter values stabilise the posterior.

        Parameters
        ----------
        groupby_params :
            One or more ``cfg_*`` column names to group by.
        metric :
            Column to compute variability over (default ``"posterior_flux_mean"``).
        """
        if isinstance(groupby_params, str):
            groupby_params = [groupby_params]
        return (
            self.df.groupby(groupby_params)[metric]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
            .sort_values(f"{metric}_std")
        )

    def sensitivity(self, metric: str = "reduced_chi2") -> pd.DataFrame:
        """Rank swept parameters by their effect on *metric*.

        Computes the range (max − min) of the group-mean *metric* when
        grouped by each ``cfg_*`` column.  A larger range means the result
        is more sensitive to that parameter.

        Returns
        -------
        DataFrame sorted by sensitivity (most influential first).
        """
        cfg_cols = [c for c in self.df.columns if c.startswith("cfg_")]
        rows = []
        for col in cfg_cols:
            try:
                grp = self.df.groupby(col)[metric].mean()
                rows.append(
                    {
                        "param": col.removeprefix("cfg_"),
                        "range": float(grp.max() - grp.min()),
                        "best_value": grp.idxmin(),  # value giving lowest chi²
                    }
                )
            except Exception:
                pass
        return (
            pd.DataFrame(rows)
            .sort_values("range", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_chi2(
        self,
        param: str,
        ax=None,
        target_band: tuple[float, float] = (0.9, 1.1),
    ):
        """Scatter plot of reduced chi² vs a swept parameter.

        Parameters
        ----------
        param :
            Column name (with or without the ``cfg_`` prefix).
        ax :
            Existing :class:`matplotlib.axes.Axes`.  If *None*, creates a figure.
        target_band :
            Shade this chi² range in green (default ``(0.9, 1.1)``).
        """
        import matplotlib.pyplot as plt

        if not param.startswith("cfg_"):
            param = f"cfg_{param}"

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        valid = self.df[["reduced_chi2", param]].dropna()
        ax.scatter(valid[param], valid["reduced_chi2"], alpha=0.7, zorder=3)
        ax.axhline(1.0, color="k", lw=1.5, label="chi² = 1")
        ax.axhspan(
            *target_band, color="green", alpha=0.12, label=f"chi² ∈ {target_band}"
        )
        ax.set_xlabel(param.removeprefix("cfg_"))
        ax.set_ylabel("Reduced chi²")
        ax.set_title(f"chi² vs {param.removeprefix('cfg_')}")
        ax.legend()
        plt.tight_layout()
        return ax

    def plot_heatmap(
        self,
        param_x: str,
        param_y: str,
        metric: str = "reduced_chi2",
        ax=None,
    ):
        """Heatmap of *metric* over two swept parameters.

        Parameters
        ----------
        param_x, param_y :
            Column names (with or without ``cfg_`` prefix) for the x / y axes.
        metric :
            CSV column to use as cell values (default ``"reduced_chi2"``).
        ax :
            Existing :class:`matplotlib.axes.Axes`.  If *None*, creates a figure.
        """
        import matplotlib.pyplot as plt

        if not param_x.startswith("cfg_"):
            param_x = f"cfg_{param_x}"
        if not param_y.startswith("cfg_"):
            param_y = f"cfg_{param_y}"

        pivot = self.df.pivot_table(
            values=metric, index=param_y, columns=param_x, aggfunc="mean"
        )

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        im = ax.imshow(pivot.values, aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(param_x.removeprefix("cfg_"))
        ax.set_ylabel(param_y.removeprefix("cfg_"))
        ax.set_title(f"{metric} heatmap")
        plt.colorbar(im, ax=ax, label=metric)
        plt.tight_layout()
        return ax

    def plot_sensitivity(self, metric: str = "reduced_chi2", ax=None):
        """Horizontal bar chart of parameter sensitivity (range of group means).

        Parameters
        ----------
        metric :
            Column to measure sensitivity over (default ``"reduced_chi2"``).
        ax :
            Existing :class:`matplotlib.axes.Axes`.
        """
        import matplotlib.pyplot as plt

        sens = self.sensitivity(metric=metric)
        if sens.empty:
            return ax

        if ax is None:
            _, ax = plt.subplots(figsize=(7, max(3, len(sens) * 0.4)))

        ax.barh(sens["param"][::-1], sens["range"][::-1])
        ax.set_xlabel(f"Range of mean {metric} across parameter values")
        ax.set_title("Parameter sensitivity")
        plt.tight_layout()
        return ax

    # ------------------------------------------------------------------
    # Config rehydration
    # ------------------------------------------------------------------

    def to_config(
        self,
        row: pd.Series,
        pipeline_cls: type = SLVMethaneInversion,
        cache: str | Path | None = None,
    ) -> tuple[InversionConfig, type]:
        """Rehydrate an :class:`InversionConfig` from a results row.

        Parameters
        ----------
        row :
            A row from :attr:`df` (e.g. ``results.best().iloc[0]``).
        pipeline_cls :
            Pipeline class to associate.
        cache :
            Override cache path.  If *None*, keeps the original run's cache.

        Returns
        -------
        ``(InversionConfig, pipeline_cls)`` tuple ready to instantiate and run.
        """
        valid = {f.name for f in dataclasses.fields(InversionConfig)}
        raw = {
            k.removeprefix("cfg_"): v
            for k, v in row.items()
            if k.startswith("cfg_") and k.removeprefix("cfg_") in valid
        }
        # Decode JSON-encoded dicts/lists
        parsed: dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, str) and v[:1] in ("{", "["):
                with contextlib.suppress(json.JSONDecodeError):
                    v = json.loads(v)
            parsed[k] = v

        if cache is not None:
            parsed["cache"] = str(cache)

        return InversionConfig(**parsed), pipeline_cls
