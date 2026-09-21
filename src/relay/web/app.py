"""The dashboard's HTTP layer: one static page plus a handful of JSON endpoints.

This module is deliberately the thinnest thing in relay. It does exactly two
jobs:

  * read SQLite and hand the numbers back as JSON, and
  * serve one HTML file.

It does *not* talk to the cluster, parse event lines, or know that Slurm
exists. That is the same rule the rest of relay follows (no component calls
another component's functions; the database and the event log are the only
shared contracts), and it has a practical payoff: you can point the dashboard
at a database file and develop the whole front end with no cluster, no daemon
and no network.

The one thing a dashboard genuinely cannot do by reading the database is
*cancel* a run, because cancelling means running `scancel` over SSH. Rather
than importing a backend here and dragging config loading, SSH and Slurm into
the web layer, `create_app` takes a `cancel_fn` callable. Whoever builds the
app (the CLI's `relay dash` command) already has a configured backend and can
hand us a one-line closure. If nobody does, the cancel endpoint answers 503
with a sentence explaining that, which is honest and harmless.

Two smaller decisions worth naming:

* **A `Store` per request.** `Store` owns a sqlite3 connection and its
  docstring says it is not thread-safe; uvicorn runs sync endpoints in a
  thread pool, so a single shared connection would eventually be used from two
  threads at once. Opening one per request costs well under a millisecond
  (SQLite "connecting" is just opening a file) and removes the whole class of
  problem. The sophisticated alternative is a thread-local connection pool,
  which would be worth it if this served real traffic. It serves one browser.

* **Polling, not websockets.** The browser asks every two seconds and sends
  the last step / last seq it already has, so each poll transfers only new
  points. Websockets are explicitly out of scope for relay, and for a page
  that shows numbers arriving every few seconds anyway, polling is simpler to
  reason about and reconnects for free.

* **The GPU-hour rate is injected, not loaded.** `create_app` takes an
  optional `gpu_hour_rate` because this module reads SQLite and nothing else;
  importing `config` here would give the web layer a second way to fail (a
  missing or malformed config file) for a page whose entire job is to display
  rows. The CLI already has a loaded `Config` when it starts the dashboard, so
  it passes the one number down. When it is None the API simply omits the
  `cost` key -- relay never invents a price.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse

from ..store import Store

# The static page lives next to this file and is listed in pyproject.toml's
# package-data, so it ships with an installed relay rather than only working
# from a source checkout.
STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Loopback only. There is no authentication anywhere in relay and this runs on
# a university network, so binding 0.0.0.0 would publish every experiment (and
# a cancel button) to anyone on the same subnet. This constant exists so the
# address is stated once and is hard to change by accident.
BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _age_seconds(iso_ts: str | None) -> float | None:
    """Seconds between an ISO-8601 UTC timestamp and now, or None.

    Computed on the server so the browser never has to reason about clock
    skew or timezone parsing: it just prints the number we give it. Python
    3.11's `fromisoformat` understands the trailing "Z" that relay writes.
    """
    if not iso_ts:
        return None
    try:
        when = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds()


def create_app(
    store_path: str | os.PathLike | None = None,
    cancel_fn: Callable[[str], None] | None = None,
    gpu_hour_rate: float | None = None,
) -> FastAPI:
    """Build the dashboard app.

    `store_path` is the SQLite database to read; None means relay's default
    location. `cancel_fn(run_id)` is an optional hook the CLI wires to a
    backend's `cancel`; without it the cancel endpoint refuses politely.
    `gpu_hour_rate` is the configured currency-per-GPU-hour, or None: with
    None the usage endpoint carries no `cost` key at all and the page shows
    hours only.
    """
    app = FastAPI(title="relay dashboard", docs_url=None, redoc_url=None)

    def get_store() -> Iterator[Store]:
        """FastAPI dependency: one short-lived Store per request.

        The `yield` form guarantees `close()` runs even if the endpoint
        raises, so we never leak a file handle on a 404.
        """
        store = Store(store_path)
        try:
            yield store
        finally:
            store.close()

    def _require_run(store: Store, run_id: str) -> dict:
        """Fetch a run or raise a 404 whose detail is a plain sentence.

        Same rule as the CLI: the message says what happened and what to do
        next, because a user reading it has no traceback to fall back on.
        """
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No run {run_id!r} in relay's database. "
                    f"Run `relay ls` to see the runs relay knows about."
                ),
            )
        return run

    # ---------------------------------------------------------------- page

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the one static page.

        `FileResponse` rather than mounting StaticFiles: there is exactly one
        file, and serving it by name means no directory is ever exposed.
        """
        if not INDEX_HTML.exists():
            raise HTTPException(
                status_code=500,
                detail=(
                    f"The dashboard page is missing from the installed package "
                    f"(expected {INDEX_HTML}). Reinstall relay."
                ),
            )
        return FileResponse(INDEX_HTML, media_type="text/html")

    # ---------------------------------------------------------------- runs

    @app.get("/api/runs")
    def list_runs(store: Store = Depends(get_store)) -> dict:
        """Every run, newest first, each with its latest metric values.

        The latest metrics are joined in here rather than fetched per-run by
        the browser: the list view wants one small payload, not N+1 requests.
        The sync timestamp rides in the same envelope so the page can always
        say how old what it is showing is -- stale data the user knows is
        stale is fine, stale data presented as current is not.

        `daemon_state` rides along for the same reason. The page needs it on
        every poll to decide whether to show "Authentication needed. Run
        `relay connect`.", and this endpoint is already polled every two
        seconds; a second endpoint would double the request count to carry
        five fields that are always the same size.
        """
        runs = store.list_runs()
        for run in runs:
            run["latest_metrics"] = store.latest_metrics(run["run_id"])
        last_synced = store.last_synced()
        return {
            "runs": runs,
            "daemon_state": store.get_daemon_state(),
            "last_synced": last_synced,
            "last_synced_age_seconds": _age_seconds(last_synced),
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, store: Store = Depends(get_store)) -> dict:
        """One run, plus the names of the metrics it has logged.

        The names are what the page needs to decide how many charts to draw;
        the points themselves come from the metrics endpoint, one call per
        chart, so that each chart can poll independently with its own
        `since`.

        `usage` is one row per scheduler attempt. It is small -- a run
        preempted five times has six rows -- so it rides in the detail payload
        rather than getting an endpoint of its own. `attempts` comes from the
        run row itself (counted from `run_started` events at ingest) and can
        differ from `len(usage)`: the sidecar knows an attempt started the
        moment it runs, while sacct only learns about it on the next usage
        fetch.
        """
        run = _require_run(store, run_id)
        run["metric_names"] = store.metric_names(run_id)
        run["latest_metrics"] = store.latest_metrics(run_id)
        run["usage"] = store.list_usage(run_id)
        last_synced = store.last_synced()
        run["last_synced"] = last_synced
        run["last_synced_age_seconds"] = _age_seconds(last_synced)
        return run

    @app.get("/api/runs/{run_id}/metrics/{name}")
    def get_metric(
        run_id: str,
        name: str,
        since: int | None = None,
        store: Store = Depends(get_store),
    ) -> dict:
        """Points for one metric, optionally only those after step `since`.

        `since` is exclusive and passes straight through to the store's
        `since_step`. It is optional rather than defaulting to 0 because step
        0 is a real step: a first fetch must not silently drop it.

        This parameter is the whole reason the dashboard can watch a long run.
        A 200k-point loss curve re-sent every two seconds would be tens of
        megabytes a minute; with `since` it is a handful of points.
        """
        _require_run(store, run_id)
        points = store.metric_series(run_id, name, since_step=since)
        return {
            "run_id": run_id,
            "name": name,
            "since": since,
            "points": points,
            # The highest step in this batch, so the browser can send it back
            # as the next `since` without scanning the array itself.
            "last_step": points[-1]["step"] if points else since,
        }

    @app.get("/api/runs/{run_id}/logs")
    def get_logs(
        run_id: str,
        since_seq: int = 0,
        limit: int | None = None,
        store: Store = Depends(get_store),
    ) -> dict:
        """Log events for a run, in order, after `since_seq` (exclusive).

        Same incremental trick as metrics, keyed on the event sequence number
        instead of a step. Payloads are passed through exactly as the store
        decoded them: what is inside a log payload belongs to the event
        schema, and the web layer should not need editing when it grows a
        field.
        """
        _require_run(store, run_id)
        logs = store.list_logs(run_id, since_seq=since_seq, limit=limit)
        return {
            "run_id": run_id,
            "since_seq": since_seq,
            "logs": logs,
            "last_seq": logs[-1]["seq"] if logs else since_seq,
        }

    # --------------------------------------------------------------- usage

    @app.get("/api/usage")
    def get_usage(
        by: str = "run",
        since: str | None = None,
        group: str | None = None,
        store: Store = Depends(get_store),
    ) -> dict:
        """GPU- and CPU-hours, the partition split, and the wasted-hours breakdown.

        Three store queries in one response because the page shows all three
        together and they share the same filters; splitting them would mean
        three round trips that could disagree about `since`.

        Every number here is derived at query time from the `usage` table --
        nothing is stored -- so this endpoint is always consistent with the
        rows the daemon last fetched, and `last_synced` says how old those are.

        `by`, `since` and `group` are the same three filters `relay usage`
        takes, deliberately: the CLI and the dashboard should never be able to
        answer the same question differently.
        """
        try:
            rows = store.usage_by(by, since=since, group=group)
            totals = store.usage_totals(since=since, group=group)
            wasted = store.wasted_hours(since=since, group=group)
        except ValueError as exc:
            # The store raises ValueError for both an unknown `by` and an
            # unreadable `since`. Both are the caller's mistake, so 400 rather
            # than 500, and the store's message is already a plain sentence.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        last_synced = store.last_synced()
        body = {
            "by": by,
            "rows": rows,
            "totals": totals,
            "wasted": wasted,
            "last_synced": last_synced,
            "last_synced_age_seconds": _age_seconds(last_synced),
        }

        # The `cost` key exists only when someone told relay what an hour
        # costs. No key at all, rather than nulls or a guessed rate: a made-up
        # dollar figure on a dashboard is worse than no dollar figure, because
        # it looks like it came from the invoice.
        if gpu_hour_rate is not None:
            body["cost"] = {
                "gpu_hour_rate": gpu_hour_rate,
                "total": totals["gpu_hours"] * gpu_hour_rate,
                "wasted_failed": wasted["failed"]["gpu_hours"] * gpu_hour_rate,
                "wasted_lost": wasted["lost_to_preemption"]["gpu_hours"] * gpu_hour_rate,
            }
        return body

    # -------------------------------------------------------------- cancel

    # POST, never GET. A GET that cancels is a live grenade: browsers, link
    # prefetchers and "open all bookmarks" would all happily fire it, and the
    # thing on the other end is someone's eight-hour training job. Registering
    # only POST also means a stray GET gets a 405 instead of doing anything.
    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str, store: Store = Depends(get_store)) -> dict:
        """Ask the injected backend hook to cancel a run.

        We check that the run exists before checking whether cancelling is
        even wired up, so a typo'd run ID gets the more useful of the two
        errors.

        Note what this does *not* do: write "cancelled" into the database.
        Status is scheduler-owned. We asked Slurm to stop the job; the daemon
        will notice it stopped on its next cycle and record that. Writing the
        status here would mean the dashboard and the scheduler could disagree.
        """
        _require_run(store, run_id)

        if cancel_fn is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "This dashboard was started without a way to cancel runs, so "
                    "it can only read. Use `relay cancel " + run_id + "` from the "
                    "command line instead."
                ),
            )

        try:
            cancel_fn(run_id)
        except Exception as exc:  # noqa: BLE001 - any backend failure, reported as text
            # Transport failures are normal, not exceptional. The user gets a
            # sentence; the run is untouched and they can try again.
            raise HTTPException(
                status_code=502,
                detail=f"Could not cancel {run_id}: {exc}. Check `relay doctor` and try again.",
            ) from exc

        return {
            "run_id": run_id,
            "ok": True,
            # Deliberately not "cancelled": we have only asked. The daemon
            # reports what actually happened once the scheduler agrees.
            "status": "cancel requested",
        }

    return app


def serve(
    store_path: str | os.PathLike | None = None,
    cancel_fn: Callable[[str], None] | None = None,
    port: int = DEFAULT_PORT,
    gpu_hour_rate: float | None = None,
) -> None:
    """Run the dashboard until interrupted.

    Bound to 127.0.0.1 and nothing else. relay has no authentication and no
    concept of users, and this typically runs on a university network where
    "the local network" means a few thousand strangers. A dashboard on
    0.0.0.0 would hand all of them your experiment data and a cancel button.
    If you need it from another machine, forward the port over SSH -- that
    way the authentication is SSH's problem, which is where it belongs.
    """
    import uvicorn  # imported here so `import relay.web.app` stays cheap

    uvicorn.run(
        create_app(store_path=store_path, cancel_fn=cancel_fn, gpu_hour_rate=gpu_hour_rate),
        host=BIND_HOST,
        port=port,
        log_level="warning",
    )
