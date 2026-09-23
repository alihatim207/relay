"""Tests for the dashboard's HTTP layer.

Everything here works the same way: build a small SQLite database in `tmp_path`
by hand, hand its path to `create_app`, and drive the result with FastAPI's
`TestClient` (which calls the app in-process -- no server, no port, no network).

The database is seeded through the real `Store`, not by writing SQL, so if the
schema changes these tests fail loudly instead of quietly testing a fiction.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from relay.store import Store
from relay.web.app import create_app

RUN_ID = "vr_s1_7f3a"


def _event(seq: int, type: str, payload: dict, ts: str) -> dict:
    """One parsed event, shaped the way the daemon hands them to the store."""
    return {"v": 1, "run": RUN_ID, "seq": seq, "ts": ts, "type": type, "payload": payload}


@pytest.fixture()
def db_path(tmp_path):
    """A seeded database: one run with three metric steps and three log lines.

    Steps are 0, 100 and 200 on purpose. Step 0 is a real step, and a
    `since`-style filter that treats 0 as "nothing yet" would silently drop it,
    so the first fetch below checks it comes back.
    """
    path = tmp_path / "relay.db"
    with Store(path) as store:
        store.create_run(
            RUN_ID,
            backend="local",
            name="baseline",
            job_id="4242",
            status="running",
            run_dir=str(tmp_path / "runs" / RUN_ID),
        )
        events = [
            _event(1, "run_started", {"cmd": ["python", "train.py"]}, "2026-09-16T18:22:00.000Z"),
            _event(2, "metric", {"step": 0, "values": {"loss": 2.0, "sps": 100.0}},
                   "2026-09-16T18:22:01.000Z"),
            _event(3, "log", {"stream": "stdout", "line": "epoch 0"}, "2026-09-16T18:22:02.000Z"),
            _event(4, "metric", {"step": 100, "values": {"loss": 1.0, "sps": 110.0}},
                   "2026-09-16T18:22:03.000Z"),
            _event(5, "log", {"stream": "stdout", "line": "epoch 1"}, "2026-09-16T18:22:04.000Z"),
            _event(6, "metric", {"step": 200, "values": {"loss": 0.5, "sps": 120.0}},
                   "2026-09-16T18:22:05.000Z"),
            _event(7, "log", {"stream": "stderr", "line": "warning: slow"},
                   "2026-09-16T18:22:06.000Z"),
        ]
        store.ingest(RUN_ID, "events.jsonl", events, new_offset=1234)
        store.mark_synced()

        # A second run, so "list" is genuinely listing rather than returning
        # the only row that exists.
        store.create_run("vr_s2_0001", backend="local", name="other", status="completed")
    return path


@pytest.fixture()
def client(db_path):
    """App with cancel wired to a stub that just records what it was asked to do."""
    cancelled: list[str] = []
    app = create_app(store_path=str(db_path), cancel_fn=cancelled.append)
    with TestClient(app) as test_client:
        # Hang the recording list off the client so tests can inspect it.
        test_client.cancelled = cancelled
        yield test_client


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def test_runs_list_shape(client):
    response = client.get("/api/runs")
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "runs",
        "daemon_state",
        "last_synced",
        "last_synced_age_seconds",
        # The scheduler poll is a second, slower clock; see the freshness
        # tests at the bottom of this file.
        "last_scheduler",
        "last_scheduler_age_seconds",
        "active_runs",
    }
    assert body["last_synced"] is not None
    # The daemon synced a moment ago, so the age is a small non-negative number.
    assert 0 <= body["last_synced_age_seconds"] < 60

    runs = {run["run_id"]: run for run in body["runs"]}
    assert set(runs) == {RUN_ID, "vr_s2_0001"}

    run = runs[RUN_ID]
    assert run["status"] == "running"
    assert run["name"] == "baseline"
    assert run["stale"] is False
    assert run["terminal"] is False
    assert run["last_step"] == 200
    # Latest metrics come joined into the list so the page needs one request,
    # not one per run.
    assert run["latest_metrics"] == {"loss": 0.5, "sps": 120.0}


def test_run_detail(client):
    body = client.get(f"/api/runs/{RUN_ID}").json()

    assert body["run_id"] == RUN_ID
    assert body["backend"] == "local"
    assert body["job_id"] == "4242"
    assert body["metric_names"] == ["loss", "sps"]
    assert body["latest_metrics"]["loss"] == 0.5
    assert body["last_synced"] is not None


def test_unknown_run_detail_is_404_with_a_sentence(client):
    response = client.get("/api/runs/vr_nope")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "vr_nope" in detail
    # A plain sentence that says what to do next, not a traceback or a code.
    assert "relay ls" in detail


# --------------------------------------------------------------------------
# Metrics: the incremental fetch that makes long runs affordable
# --------------------------------------------------------------------------


def test_metric_series_full_then_incremental(client):
    full = client.get(f"/api/runs/{RUN_ID}/metrics/loss").json()
    assert [point["step"] for point in full["points"]] == [0, 100, 200]
    assert [point["value"] for point in full["points"]] == [2.0, 1.0, 0.5]
    assert full["last_step"] == 200

    # Now poll the way the browser does: send back the last step we have.
    # `since` is exclusive, so only genuinely newer points come across.
    incremental = client.get(f"/api/runs/{RUN_ID}/metrics/loss", params={"since": 100}).json()
    assert [point["step"] for point in incremental["points"]] == [200]
    assert incremental["since"] == 100
    assert incremental["last_step"] == 200

    # Caught up: nothing new, and `last_step` stays where the caller was so the
    # next poll asks the same question rather than rewinding to the start.
    caught_up = client.get(f"/api/runs/{RUN_ID}/metrics/loss", params={"since": 200}).json()
    assert caught_up["points"] == []
    assert caught_up["last_step"] == 200


def test_metric_first_fetch_includes_step_zero(client):
    """Step 0 is a real step; omitting `since` must not behave like since=0."""
    points = client.get(f"/api/runs/{RUN_ID}/metrics/sps").json()["points"]
    assert points[0]["step"] == 0


def test_metric_for_unknown_run_is_404(client):
    assert client.get("/api/runs/vr_nope/metrics/loss").status_code == 404


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------


def test_logs_since_seq(client):
    everything = client.get(f"/api/runs/{RUN_ID}/logs").json()
    assert [event["seq"] for event in everything["logs"]] == [3, 5, 7]
    # Only log events, and the payload is passed through untouched.
    assert {event["type"] for event in everything["logs"]} == {"log"}
    assert everything["logs"][0]["payload"] == {"stream": "stdout", "line": "epoch 0"}
    assert everything["last_seq"] == 7

    newer = client.get(f"/api/runs/{RUN_ID}/logs", params={"since_seq": 5}).json()
    assert [event["seq"] for event in newer["logs"]] == [7]
    assert newer["logs"][0]["payload"]["stream"] == "stderr"

    caught_up = client.get(f"/api/runs/{RUN_ID}/logs", params={"since_seq": 7}).json()
    assert caught_up["logs"] == []
    assert caught_up["last_seq"] == 7


# --------------------------------------------------------------------------
# Cancel
# --------------------------------------------------------------------------


def test_cancel_calls_the_injected_hook(client):
    response = client.post(f"/api/runs/{RUN_ID}/cancel")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert client.cancelled == [RUN_ID]


def test_cancel_does_not_write_the_status_itself(client, db_path):
    """Status is scheduler-owned: we asked, the daemon records what happened."""
    client.post(f"/api/runs/{RUN_ID}/cancel")
    with Store(db_path) as store:
        assert store.get_run(RUN_ID)["status"] == "running"


def test_cancel_without_a_hook_is_503(db_path):
    app = create_app(store_path=str(db_path), cancel_fn=None)
    with TestClient(app) as client:
        response = client.post(f"/api/runs/{RUN_ID}/cancel")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "relay cancel" in detail  # tells the user how to do it instead


def test_cancel_unknown_run_is_404(client):
    response = client.post("/api/runs/vr_nope/cancel")
    assert response.status_code == 404
    assert client.cancelled == []  # the backend was never asked


def test_cancel_is_post_only(client):
    """A GET that cancels would be fired by any link prefetcher. 405, not 200."""
    response = client.get(f"/api/runs/{RUN_ID}/cancel")
    assert response.status_code == 405
    assert client.cancelled == []


# --------------------------------------------------------------------------
# The page itself
# --------------------------------------------------------------------------


def test_index_serves_the_html_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>relay</title>" in response.text
    # The page is self-contained apart from the charting library.
    assert "chart.js" in response.text


# --------------------------------------------------------------------------
# Daemon state: the auth banner's only input
# --------------------------------------------------------------------------


def test_runs_envelope_carries_daemon_state(client):
    """Unset means every field is None, not a missing key.

    The page reads `daemon_state.last_error_kind` on every poll, so the
    envelope always has the same shape whether or not the daemon has ever
    written a row.
    """
    state = client.get("/api/runs").json()["daemon_state"]
    assert state["last_error_kind"] is None
    assert state["last_error_ts"] is None
    assert state["last_ok_ts"] is None


def test_daemon_state_error_is_reflected(client, db_path):
    with Store(db_path) as store:
        store.set_daemon_state(
            last_error_kind="auth_required",
            last_error_ts="2026-09-16T19:00:00.000Z",
            last_error_detail="Permission denied (publickey).",
        )

    state = client.get("/api/runs").json()["daemon_state"]
    assert state["last_error_kind"] == "auth_required"
    assert state["last_error_ts"] == "2026-09-16T19:00:00.000Z"


# --------------------------------------------------------------------------
# Usage
# --------------------------------------------------------------------------
#
# The fixture below is the whole point of these tests, so the arithmetic is
# spelled out once here and the assertions just check it:
#
#   vr_u1  account mlopt, partition ckpt-all, status failed, 2 GPUs, 8 CPUs
#     attempt 1  PREEMPTED  3600s, checkpoint 900s before the end
#     attempt 2  FAILED     1800s
#   vr_u2  account other, partition gpu-a40, status completed, 1 GPU, 4 CPUs
#     attempt 1  PREEMPTED  1800s, no checkpoint events at all
#
#   total gpu-hours  (3600*2 + 1800*2 + 1800*1) / 3600 = 3.5
#   total cpu-hours  (3600*8 + 1800*8 + 1800*4) / 3600 = 14.0
#   wasted failed    every attempt of vr_u1: 3.0 gpu-h, 12.0 cpu-h
#   wasted lost      vr_u1 tail 900s*2 = 0.5 gpu-h, vr_u2 whole 1800s*1 = 0.5

USAGE_RUN_A = "vr_u1"
USAGE_RUN_B = "vr_u2"


@pytest.fixture()
def usage_db(tmp_path):
    """A database with two runs and three scheduler attempts, seeded via Store."""
    path = tmp_path / "usage.db"
    with Store(path) as store:
        store.create_run(
            USAGE_RUN_A,
            backend="slurm",
            name="preempted-then-crashed",
            job_id="1001",
            status="failed",
            spec={"account": "mlopt", "partition": "ckpt-all"},
        )
        store.create_run(
            USAGE_RUN_B,
            backend="slurm",
            name="no-checkpoints",
            job_id="2002",
            status="completed",
            spec={"account": "other", "partition": "gpu-a40"},
        )

        # One checkpoint, fifteen minutes before the first attempt was
        # preempted. Only the tail after it is lost.
        store.ingest(
            USAGE_RUN_A,
            "events.jsonl",
            [
                {"v": 2, "run": USAGE_RUN_A, "seq": 1, "ts": "2026-09-10T00:00:00.000Z",
                 "type": "run_started", "payload": {"attempt": 0}},
                {"v": 2, "run": USAGE_RUN_A, "seq": 2, "ts": "2026-09-10T00:45:00.000Z",
                 "type": "checkpoint", "payload": {"step": 5000}},
            ],
            new_offset=200,
        )
        # vr_u2 reports no checkpoint events at all, which is the case the
        # dashboard has to call out by name.
        store.ingest(
            USAGE_RUN_B,
            "events.jsonl",
            [
                {"v": 2, "run": USAGE_RUN_B, "seq": 1, "ts": "2026-09-12T00:00:00.000Z",
                 "type": "run_started", "payload": {"attempt": 0}},
            ],
            new_offset=100,
        )

        store.upsert_usage(
            [
                {"job_id": "1001", "run_id": USAGE_RUN_A, "state": "PREEMPTED",
                 "partition": "ckpt-all", "start": "2026-09-10T00:00:00",
                 "end": "2026-09-10T01:00:00", "elapsed_s": 3600, "gpus": 2, "cpus": 8},
                {"job_id": "1001", "run_id": USAGE_RUN_A, "state": "FAILED",
                 "partition": "ckpt-all", "start": "2026-09-10T02:00:00",
                 "end": "2026-09-10T02:30:00", "elapsed_s": 1800, "gpus": 2, "cpus": 8},
                {"job_id": "2002", "run_id": USAGE_RUN_B, "state": "PREEMPTED",
                 "partition": "gpu-a40", "start": "2026-09-12T00:00:00",
                 "end": "2026-09-12T00:30:00", "elapsed_s": 1800, "gpus": 1, "cpus": 4},
            ]
        )
        store.mark_synced()
    return path


@pytest.fixture()
def usage_client(usage_db):
    with TestClient(create_app(store_path=str(usage_db))) as test_client:
        yield test_client


def test_usage_default_shape(usage_client):
    body = usage_client.get("/api/usage").json()

    assert body["by"] == "run"
    assert body["last_synced"] is not None
    assert 0 <= body["last_synced_age_seconds"] < 60

    totals = body["totals"]
    assert totals["gpu_hours"] == pytest.approx(3.5)
    assert totals["cpu_hours"] == pytest.approx(14.0)
    assert totals["attempts"] == 3
    assert totals["runs"] == 2

    # Grouped by run, biggest first.
    rows = body["rows"]
    assert [row["key"] for row in rows] == [USAGE_RUN_A, USAGE_RUN_B]
    assert rows[0]["gpu_hours"] == pytest.approx(3.0)
    assert rows[0]["attempts"] == 2
    assert rows[1]["gpu_hours"] == pytest.approx(0.5)


def test_usage_partition_split_rides_in_totals(usage_client):
    split = {
        row["partition"]: row["gpu_hours"]
        for row in usage_client.get("/api/usage").json()["totals"]["by_partition"]
    }
    assert split["ckpt-all"] == pytest.approx(3.0)
    assert split["gpu-a40"] == pytest.approx(0.5)


def test_usage_wasted_covers_both_categories(usage_client):
    wasted = usage_client.get("/api/usage").json()["wasted"]

    # Every attempt of a failed run is wasted, including the ones before it
    # crashed.
    assert wasted["failed"]["gpu_hours"] == pytest.approx(3.0)
    assert wasted["failed"]["cpu_hours"] == pytest.approx(12.0)
    assert wasted["failed"]["runs"] == [USAGE_RUN_A]

    lost = wasted["lost_to_preemption"]
    assert lost["gpu_hours"] == pytest.approx(1.0)   # 0.5 tail + 0.5 whole attempt
    assert lost["cpu_hours"] == pytest.approx(4.0)

    attempts = {attempt["run_id"]: attempt for attempt in lost["attempts"]}
    assert set(attempts) == {USAGE_RUN_A, USAGE_RUN_B}

    # The attempt with a mid-attempt checkpoint loses only the tail after it.
    assert attempts[USAGE_RUN_A]["lost_s"] == 900
    assert attempts[USAGE_RUN_A]["no_checkpoints"] is False

    # The attempt that reported no checkpoint loses everything, and is
    # flagged so the user learns their script is not reporting them.
    assert attempts[USAGE_RUN_B]["lost_s"] == 1800
    assert attempts[USAGE_RUN_B]["no_checkpoints"] is True


def test_usage_by_partition(usage_client):
    body = usage_client.get("/api/usage", params={"by": "partition"}).json()
    assert body["by"] == "partition"
    assert [row["key"] for row in body["rows"]] == ["ckpt-all", "gpu-a40"]
    assert body["rows"][0]["runs"] == 1
    assert body["rows"][0]["attempts"] == 2


def test_usage_by_group(usage_client):
    body = usage_client.get("/api/usage", params={"by": "group"}).json()
    rows = {row["key"]: row["gpu_hours"] for row in body["rows"]}
    assert rows["mlopt"] == pytest.approx(3.0)
    assert rows["other"] == pytest.approx(0.5)


def test_usage_since_filters_by_attempt_start(usage_client):
    body = usage_client.get("/api/usage", params={"since": "2026-09-11"}).json()
    # Only vr_u2's attempt started on or after the 11th.
    assert [row["key"] for row in body["rows"]] == [USAGE_RUN_B]
    assert body["totals"]["attempts"] == 1
    assert body["totals"]["gpu_hours"] == pytest.approx(0.5)
    # The wasted numbers use the same window, so the failed run drops out too.
    assert body["wasted"]["failed"]["gpu_hours"] == pytest.approx(0.0)
    assert body["wasted"]["lost_to_preemption"]["gpu_hours"] == pytest.approx(0.5)


def test_usage_group_filter_narrows_everything(usage_client):
    body = usage_client.get("/api/usage", params={"group": "mlopt"}).json()
    assert [row["key"] for row in body["rows"]] == [USAGE_RUN_A]
    assert body["totals"]["gpu_hours"] == pytest.approx(3.0)
    assert [part["partition"] for part in body["totals"]["by_partition"]] == ["ckpt-all"]
    assert body["wasted"]["failed"]["runs"] == [USAGE_RUN_A]


def test_usage_has_no_cost_key_without_a_configured_rate(usage_client):
    """No rate, no dollar figures. relay never invents a price."""
    assert "cost" not in usage_client.get("/api/usage").json()


def test_usage_cost_when_a_rate_is_injected(usage_db):
    app = create_app(store_path=str(usage_db), gpu_hour_rate=2.0)
    with TestClient(app) as client:
        cost = client.get("/api/usage").json()["cost"]

    assert cost["gpu_hour_rate"] == 2.0
    assert cost["total"] == pytest.approx(7.0)           # 3.5 gpu-h * 2
    assert cost["wasted_failed"] == pytest.approx(6.0)   # 3.0 gpu-h * 2
    assert cost["wasted_lost"] == pytest.approx(2.0)     # 1.0 gpu-h * 2


def test_usage_rejects_an_unknown_grouping(usage_client):
    response = usage_client.get("/api/usage", params={"by": "gpu"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "gpu" in detail
    assert "partition" in detail  # says what the valid choices are


def test_usage_rejects_an_unreadable_since(usage_client):
    response = usage_client.get("/api/usage", params={"since": "last tuesday"})
    assert response.status_code == 400
    assert "last tuesday" in response.json()["detail"]


def test_run_detail_carries_attempts_and_usage(usage_db):
    with TestClient(create_app(store_path=str(usage_db))) as client:
        body = client.get(f"/api/runs/{USAGE_RUN_A}").json()

    # Counted from run_started events at ingest, not from the usage table.
    assert body["attempts"] == 1
    # One row per scheduler attempt, oldest first.
    assert [row["state"] for row in body["usage"]] == ["PREEMPTED", "FAILED"]
    assert body["usage"][0]["partition"] == "ckpt-all"
    assert body["usage"][0]["elapsed_s"] == 3600
    assert body["usage"][0]["gpus"] == 2


# --------------------------------------------------------------------------
# The page's new furniture
# --------------------------------------------------------------------------


def test_page_has_the_usage_section(client):
    page = client.get("/").text
    assert 'id="usage-section"' in page
    assert 'id="usage-totals"' in page
    assert 'id="usage-partitions"' in page
    assert 'id="wasted"' in page
    # The filters that refetch.
    assert 'id="usage-by"' in page
    assert 'id="usage-since"' in page
    # Usage is not live data, so it has its own slow timer.
    assert "USAGE_POLL_MS = 30000" in page
    assert "const POLL_MS = 2000" in page


def test_page_has_the_connect_banner(client):
    page = client.get("/").text
    assert 'id="banner"' in page
    assert "Authentication needed. Run `relay connect`." in page
    assert "Cluster unreachable since" in page


def test_page_shows_no_checkpoints_and_per_attempt_usage(client):
    page = client.get("/").text
    assert "no checkpoints reported" in page
    assert 'id="attempts-body"' in page


# --------------------------------------------------------------------------
# Two freshness clocks
# --------------------------------------------------------------------------
#
# The daemon polls two things on two independent schedules: it tails the event
# logs on the shared filesystem every couple of seconds, and it asks the
# scheduler for job state no more often than every thirty seconds, backing off
# from there. So there are two "how old is this" numbers, and the page has to
# show both -- metrics two seconds old next to job state forty seconds old is
# the normal healthy state, and one number would hide it.
#
# These tests never call `mark_synced`, only `mark_scheduler_synced`, which is
# what proves the two are genuinely independent rather than two names for the
# same write.

UNSYNCED_RUN = "vr_fresh"


@pytest.fixture()
def unsynced_db(tmp_path):
    """A database the daemon has never synced: one run, no sync timestamps."""
    path = tmp_path / "unsynced.db"
    with Store(path) as store:
        store.create_run(UNSYNCED_RUN, backend="local", name="fresh", status="queued")
    return path


@pytest.fixture()
def unsynced_client(unsynced_db):
    with TestClient(create_app(store_path=str(unsynced_db))) as test_client:
        test_client.db_path = unsynced_db
        yield test_client


def _freshness_payloads(client):
    """The three payloads that carry freshness, keyed by a readable name."""
    return {
        "runs": client.get("/api/runs").json(),
        "detail": client.get(f"/api/runs/{UNSYNCED_RUN}").json(),
        "usage": client.get("/api/usage").json(),
    }


def test_scheduler_freshness_is_null_before_the_first_poll(unsynced_client):
    """Never polled is a null, not a missing key or a zero.

    A missing key would make the page's null check depend on whether the
    daemon had ever run; a zero would read as "just now", which is the exact
    opposite of the truth.
    """
    for where, body in _freshness_payloads(unsynced_client).items():
        assert body["last_scheduler"] is None, where
        assert body["last_scheduler_age_seconds"] is None, where
        # Nothing has been synced at all yet, so the other clock is null too.
        assert body["last_synced"] is None, where
        assert body["last_synced_age_seconds"] is None, where


def test_scheduler_freshness_moves_without_the_metrics_clock(unsynced_client):
    """A scheduler poll advances one clock and leaves the other alone."""
    with Store(unsynced_client.db_path) as store:
        store.mark_scheduler_synced()

    for where, body in _freshness_payloads(unsynced_client).items():
        assert isinstance(body["last_scheduler"], str), where
        assert 0 <= body["last_scheduler_age_seconds"] < 60, where
        # `mark_synced` was never called, so the filesystem clock has not
        # moved. If these two shared a timestamp this is where it would show.
        assert body["last_synced"] is None, where
        assert body["last_synced_age_seconds"] is None, where


def test_page_shows_both_ages_in_one_line(client):
    """The header says which number belongs to which clock, in plain words."""
    page = client.get("/").text
    assert "function renderSync(metricsAge, schedulerAge, activeRuns)" in page
    assert "'metrics '" in page
    assert "job state never polled" in page
    # Both call sites pass the scheduler age through.
    assert "renderSync(data.last_synced_age_seconds, data.last_scheduler_age_seconds, data.active_runs)" in page
    assert "renderSync(run.last_synced_age_seconds, run.last_scheduler_age_seconds, run.active_runs)" in page
    assert "no active runs · daemon alive" in page


def test_active_runs_counts_only_non_terminal_runs(client, db_path):
    """`active_runs` is what lets the page say "no active runs" honestly.

    Zero tells the page that neither age is cluster freshness -- the daemon is
    cycling locally and making no cluster calls at all -- so it shows one
    "daemon alive" age instead of two that look like polling for nothing. A
    completed run must not count: a user whose last job finished an hour ago
    is idle, not watching.
    """
    body = client.get("/api/runs").json()
    assert isinstance(body["active_runs"], int)
    active = sum(1 for run in body["runs"] if run["status"] not in ("completed", "failed", "cancelled"))
    assert body["active_runs"] == active

    with Store(db_path) as store:
        for run_id in [run["run_id"] for run in body["runs"]]:
            store.update_run_status(run_id, "completed")
    assert client.get("/api/runs").json()["active_runs"] == 0
    assert client.get("/api/usage").json()["active_runs"] == 0
