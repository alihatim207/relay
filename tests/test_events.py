"""Tests for the event log format.

These are the first tests in the project, so they double as the executable
specification of the contract every later stage depends on: the sidecar writes
this format, the daemon reads it, the store persists it.
"""

import json
import os
import re

import pytest

from relay import events


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


def test_schema_version_is_two():
    assert events.SCHEMA_VERSION == 2


def test_both_schema_versions_are_accepted_for_reading():
    """New relay writes v2 and still reads v1.

    `events.jsonl` is append-only and one file can hold both: a run that
    started under v1 and was requeued after an upgrade has v1 lines followed by
    v2 lines. Refusing the old ones would drop the first attempt's history.
    """
    assert events.ACCEPTED_VERSIONS == frozenset({1, 2})
    assert events.SCHEMA_VERSION in events.ACCEPTED_VERSIONS


def test_run_ended_statuses_are_the_four_in_the_spec():
    # `interrupted` is the one that matters: SIGTERM is preemption *or* cancel
    # and the sidecar cannot tell which, so it claims neither.
    assert events.RUN_ENDED_STATUSES == (
        "completed",
        "failed",
        "requeued",
        "interrupted",
    )


def test_event_types_are_exactly_the_seven_in_the_spec():
    assert set(events.EVENT_TYPES) == {
        "run_started",
        "metric",
        "log",
        "heartbeat",
        "checkpoint",
        "signal",
        "run_ended",
    }


def test_envelope_key_order_is_fixed():
    assert events.ENVELOPE_KEYS == ("v", "run", "seq", "ts", "type", "payload")


# --------------------------------------------------------------------------
# build_event
# --------------------------------------------------------------------------


def test_build_event_has_exactly_the_envelope_keys_in_order():
    event = events.build_event("vr_s1_7f3a", 2, "metric", {"step": 1000, "values": {}})
    assert list(event.keys()) == list(events.ENVELOPE_KEYS)
    assert event["v"] == 2
    assert event["run"] == "vr_s1_7f3a"
    assert event["seq"] == 2
    assert event["type"] == "metric"


def test_build_event_stamps_iso_utc_millisecond_timestamp():
    event = events.build_event("r1", 1, "heartbeat")
    # e.g. 2026-09-16T18:22:09.003Z -- exactly three fractional digits, Z suffix.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", event["ts"])


def test_build_event_accepts_an_explicit_timestamp():
    event = events.build_event("r1", 1, "log", {}, ts="2026-09-16T18:22:09.003Z")
    assert event["ts"] == "2026-09-16T18:22:09.003Z"


def test_build_event_defaults_payload_to_empty_dict():
    assert events.build_event("r1", 1, "heartbeat")["payload"] == {}


@pytest.mark.parametrize("bad_type", ["metrics", "", "RUN_STARTED", "unknown"])
def test_build_event_rejects_unknown_type(bad_type):
    with pytest.raises(events.EventParseError):
        events.build_event("r1", 1, bad_type)


@pytest.mark.parametrize("bad_seq", [0, -1, 1.0, "1", None, True])
def test_build_event_rejects_non_positive_or_non_int_seq(bad_seq):
    # True is included on purpose: bool subclasses int, so an unguarded check
    # would silently accept it as seq == 1.
    with pytest.raises(events.EventParseError):
        events.build_event("r1", bad_seq, "log")


def test_build_event_rejects_non_dict_payload():
    with pytest.raises(events.EventParseError):
        events.build_event("r1", 1, "log", ["not", "a", "dict"])


# --------------------------------------------------------------------------
# serialize + parse_event round trip
# --------------------------------------------------------------------------


def test_serialize_is_compact_and_single_line():
    line = events.serialize(events.build_event("r1", 1, "log", {"text": "hello"}))
    assert "\n" not in line
    assert ", " not in line and '": ' not in line


def test_serialize_matches_the_example_in_the_spec():
    # The example line in CLAUDE.md shows v:1, which is what relay wrote when it
    # was written; the envelope is otherwise byte-for-byte what we still emit.
    # A freshly built event carries the current version.
    event = events.build_event(
        "vr_s1_7f3a",
        2,
        "metric",
        {"step": 1000, "values": {"loss": 0.412, "sps": 18400}},
        ts="2026-09-16T18:22:09.003Z",
    )
    assert events.serialize(event) == (
        '{"v":2,"run":"vr_s1_7f3a","seq":2,"ts":"2026-09-16T18:22:09.003Z",'
        '"type":"metric","payload":{"step":1000,"values":{"loss":0.412,"sps":18400}}}'
    )


def test_build_serialize_parse_round_trip():
    original = events.build_event(
        "vr_s1_7f3a", 7, "checkpoint", {"step": 2000, "path": "/tmp/ckpt.pt"}
    )
    assert events.parse_event(events.serialize(original)) == original


def test_parse_event_tolerates_a_trailing_newline():
    line = events.serialize(events.build_event("r1", 1, "log", {"text": "x"})) + "\n"
    assert events.parse_event(line)["seq"] == 1


# --------------------------------------------------------------------------
# parse_event rejections
# --------------------------------------------------------------------------


def valid_envelope(**overrides):
    """A known-good envelope dict, with fields overridden for negative tests."""
    base = {
        "v": 1,
        "run": "r1",
        "seq": 3,
        "ts": "2026-09-16T18:22:09.003Z",
        "type": "log",
        "payload": {"text": "hi"},
    }
    base.update(overrides)
    return base


def test_parse_event_rejects_bad_json():
    with pytest.raises(events.EventParseError):
        events.parse_event('{"v":1,"run":')


def test_parse_event_rejects_empty_line():
    with pytest.raises(events.EventParseError):
        events.parse_event("   ")


def test_parse_event_rejects_non_object_json():
    for text in ["[1,2,3]", '"a string"', "42", "null"]:
        with pytest.raises(events.EventParseError):
            events.parse_event(text)


@pytest.mark.parametrize("missing", ["v", "run", "seq", "ts", "type", "payload"])
def test_parse_event_rejects_missing_envelope_key(missing):
    obj = valid_envelope()
    del obj[missing]
    with pytest.raises(events.EventParseError):
        events.parse_event(json.dumps(obj))


def test_parse_event_error_names_the_missing_key():
    obj = valid_envelope()
    del obj["seq"]
    with pytest.raises(events.EventParseError, match="seq"):
        events.parse_event(json.dumps(obj))


def test_parse_event_rejects_unknown_type():
    with pytest.raises(events.EventParseError):
        events.parse_event(json.dumps(valid_envelope(type="explosion")))


@pytest.mark.parametrize("version", [1, 2])
def test_parse_event_accepts_every_known_schema_version(version):
    assert events.parse_event(json.dumps(valid_envelope(v=version)))["v"] == version


def test_parse_event_reads_the_v1_example_line_from_the_spec():
    """The exact line CLAUDE.md documents, written by a pre-v2 relay."""
    event = events.parse_event(
        '{"v":1,"run":"vr_s1_7f3a","seq":2,"ts":"2026-09-16T18:22:09.003Z",'
        '"type":"metric","payload":{"step":1000,"values":{"loss":0.412,"sps":18400}}}'
    )
    assert event["seq"] == 2
    assert event["payload"]["values"]["loss"] == 0.412


@pytest.mark.parametrize("version", [0, 3, "2", None, 2.0, True])
def test_parse_event_rejects_unsupported_schema_version(version):
    # A version we have never heard of could use a different envelope shape
    # entirely, so guessing is worse than refusing. The last two are the
    # interesting ones: `2.0 == 2` and `True == 1` in Python, so a bare
    # membership test would let both through.
    with pytest.raises(events.EventParseError):
        events.parse_event(json.dumps(valid_envelope(v=version)))


@pytest.mark.parametrize("bad_seq", [0, -5, "3", 3.5, None, True])
def test_parse_event_rejects_bad_seq(bad_seq):
    with pytest.raises(events.EventParseError):
        events.parse_event(json.dumps(valid_envelope(seq=bad_seq)))


@pytest.mark.parametrize("bad_run", ["", None, 17])
def test_parse_event_rejects_bad_run(bad_run):
    with pytest.raises(events.EventParseError):
        events.parse_event(json.dumps(valid_envelope(run=bad_run)))


def test_parse_event_rejects_non_object_payload():
    with pytest.raises(events.EventParseError):
        events.parse_event(json.dumps(valid_envelope(payload=[1, 2])))


def test_parse_event_rejects_bytes():
    # The daemon decodes; parse_event takes str only.
    with pytest.raises(events.EventParseError):
        events.parse_event(b'{"v":1}')


def test_event_parse_error_is_a_value_error():
    # So a caller that only knows about ValueError still catches it.
    assert issubclass(events.EventParseError, ValueError)


# --------------------------------------------------------------------------
# parse_metric_line -- the ##relay## convention
# --------------------------------------------------------------------------


def test_parse_metric_line_happy_path():
    payload = events.parse_metric_line('##relay## {"step": 1000, "loss": 0.412, "sps": 18400}')
    assert payload == {"step": 1000, "values": {"loss": 0.412, "sps": 18400}}


def test_parse_metric_line_preserves_the_number_type_the_user_printed():
    # So an event built from this payload serializes as "sps":18400, matching
    # the example line in the spec, rather than "sps":18400.0.
    payload = events.parse_metric_line('##relay## {"step": 1000, "loss": 0.412, "sps": 18400}')
    event = events.build_event("vr_s1_7f3a", 2, "metric", payload, ts="2026-09-16T18:22:09.003Z")
    assert events.serialize(event) == (
        '{"v":2,"run":"vr_s1_7f3a","seq":2,"ts":"2026-09-16T18:22:09.003Z",'
        '"type":"metric","payload":{"step":1000,"values":{"loss":0.412,"sps":18400}}}'
    )


def test_parse_metric_line_tolerates_trailing_newline():
    assert events.parse_metric_line('##relay## {"step": 5, "loss": 1.0}\n') == {
        "step": 5,
        "values": {"loss": 1.0},
    }


def test_parse_metric_line_allows_a_metric_with_no_values():
    assert events.parse_metric_line('##relay## {"step": 3}') == {"step": 3, "values": {}}


def test_parse_metric_line_coerces_float_step_to_int():
    assert events.parse_metric_line('##relay## {"step": 1000.0, "loss": 1.5}')["step"] == 1000


@pytest.mark.parametrize(
    "line",
    [
        "just a normal print statement",
        "",
        "Epoch 1: loss=0.4",
        "  ##relay## {\"step\": 1}",  # marker must start the line
        '##relay##{"step": 1}',  # the marker includes a trailing space
        'prefix ##relay## {"step": 1}',
    ],
)
def test_parse_metric_line_returns_none_for_non_metric_lines(line):
    assert events.parse_metric_line(line) is None


@pytest.mark.parametrize(
    "line",
    [
        "##relay## {not json}",
        "##relay## ",
        "##relay## [1, 2, 3]",  # JSON, but not an object
        '##relay## {"loss": 0.4}',  # no step
        '##relay## {"step": "1000", "loss": 0.4}',  # step is not numeric
        '##relay## {"step": null}',
        '##relay## {"step": true}',  # bool is not a step
    ],
)
def test_parse_metric_line_returns_none_for_malformed_metrics(line):
    # A typo in a user's print statement costs them a metric, never a crash.
    assert events.parse_metric_line(line) is None


def test_parse_metric_line_skips_non_numeric_values():
    payload = events.parse_metric_line(
        '##relay## {"step": 10, "loss": 0.4, "note": "warmup", "tags": [1], "meta": {"a": 1}}'
    )
    assert payload == {"step": 10, "values": {"loss": 0.4}}


def test_parse_metric_line_skips_boolean_values():
    # isinstance(True, int) is True, so bools have to be excluded explicitly.
    payload = events.parse_metric_line('##relay## {"step": 10, "converged": true, "loss": 0.4}')
    assert payload == {"step": 10, "values": {"loss": 0.4}}


def test_parse_metric_line_keeps_ints_as_numbers():
    payload = events.parse_metric_line('##relay## {"step": 10, "tokens": 4096}')
    assert payload["values"]["tokens"] == 4096


# --------------------------------------------------------------------------
# EventWriter
# --------------------------------------------------------------------------


def read_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def test_writer_appends_one_parseable_line_per_event(tmp_path):
    path = str(tmp_path / "events.jsonl")
    writer = events.EventWriter.open(path, "vr_s1_7f3a")
    writer.emit("run_started", {"cmd": "python train.py"})
    writer.emit("metric", {"step": 1, "values": {"loss": 0.5}})
    writer.close()

    lines = read_lines(path)
    assert len(lines) == 2
    parsed = [events.parse_event(line) for line in lines]
    assert [e["type"] for e in parsed] == ["run_started", "metric"]
    assert all(e["run"] == "vr_s1_7f3a" for e in parsed)


def test_writer_file_ends_with_a_newline(tmp_path):
    # The daemon relies on a complete line ending in \n; anything else is
    # treated as a partial write and re-read next cycle.
    path = str(tmp_path / "events.jsonl")
    with events.EventWriter.open(path, "r1") as writer:
        writer.emit("heartbeat", {"step": 0})
    with open(path, "rb") as handle:
        assert handle.read().endswith(b"\n")


def test_writer_assigns_increasing_seq_starting_at_one(tmp_path):
    path = str(tmp_path / "events.jsonl")
    with events.EventWriter.open(path, "r1") as writer:
        for _ in range(5):
            writer.emit("log", {"text": "x"})
    seqs = [events.parse_event(line)["seq"] for line in read_lines(path)]
    assert seqs == [1, 2, 3, 4, 5]


def test_emit_returns_the_event_that_was_written(tmp_path):
    path = str(tmp_path / "events.jsonl")
    with events.EventWriter.open(path, "r1") as writer:
        returned = writer.emit("log", {"text": "hello"})
    on_disk = events.parse_event(read_lines(path)[0])
    assert returned == on_disk


def test_reopening_continues_seq_instead_of_restarting(tmp_path):
    """The requeue case: a preempted attempt restarts and must not reuse seqs.

    Restarting at 1 would collide with the first attempt's (run_id, seq)
    primary key and the store's INSERT OR IGNORE would silently drop the
    resumed run's events.
    """
    path = str(tmp_path / "events.jsonl")
    with events.EventWriter.open(path, "r1") as first:
        for _ in range(3):
            first.emit("log", {"text": "attempt 1"})

    with events.EventWriter.open(path, "r1") as second:
        second.emit("run_started", {"attempt": 2})
        second.emit("metric", {"step": 100, "values": {"loss": 0.1}})

    seqs = [events.parse_event(line)["seq"] for line in read_lines(path)]
    assert seqs == [1, 2, 3, 4, 5]


def test_reopening_appends_rather_than_truncating(tmp_path):
    path = str(tmp_path / "events.jsonl")
    with events.EventWriter.open(path, "r1") as first:
        first.emit("log", {"text": "keep me"})
    with events.EventWriter.open(path, "r1") as second:
        second.emit("log", {"text": "and me"})
    lines = read_lines(path)
    assert len(lines) == 2
    assert events.parse_event(lines[0])["payload"]["text"] == "keep me"


def test_writer_continues_from_highest_seq_not_last_seq(tmp_path):
    """A SIGKILL can leave the highest seq somewhere other than the last line.

    highest_seq scans the whole file and takes the max, so an out-of-order tail
    still cannot cause a collision.
    """
    path = tmp_path / "events.jsonl"
    path.write_text(
        events.serialize(events.build_event("r1", 9, "log", {"text": "a"}))
        + "\n"
        + events.serialize(events.build_event("r1", 4, "log", {"text": "b"}))
        + "\n",
        encoding="utf-8",
    )
    with events.EventWriter.open(str(path), "r1") as writer:
        assert writer.emit("log", {"text": "c"})["seq"] == 10


def test_writer_ignores_a_truncated_final_line_when_resuming(tmp_path):
    """Exactly what SIGKILL mid-write leaves behind -- must not be fatal."""
    path = tmp_path / "events.jsonl"
    good = events.serialize(events.build_event("r1", 2, "log", {"text": "ok"}))
    path.write_text(good + "\n" + '{"v":1,"run":"r1","seq":3,"ts":"20', encoding="utf-8")

    with events.EventWriter.open(str(path), "r1") as writer:
        event = writer.emit("log", {"text": "resumed"})
    assert event["seq"] == 3


def test_highest_seq_on_missing_file_is_zero(tmp_path):
    assert events.highest_seq(str(tmp_path / "nope.jsonl")) == 0


def test_highest_seq_on_empty_file_is_zero(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")
    assert events.highest_seq(str(path)) == 0


def test_highest_seq_skips_garbage_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "this is not json\n"
        + events.serialize(events.build_event("r1", 6, "log", {}))
        + "\n"
        + "\n",
        encoding="utf-8",
    )
    assert events.highest_seq(str(path)) == 6


def test_writer_creates_the_file_if_absent(tmp_path):
    path = tmp_path / "events.jsonl"
    assert not path.exists()
    events.EventWriter.open(str(path), "r1").close()
    assert path.exists()


def test_close_is_idempotent(tmp_path):
    writer = events.EventWriter.open(str(tmp_path / "events.jsonl"), "r1")
    writer.close()
    writer.close()


def test_emit_after_close_raises(tmp_path):
    writer = events.EventWriter.open(str(tmp_path / "events.jsonl"), "r1")
    writer.close()
    with pytest.raises(ValueError):
        writer.emit("log", {"text": "too late"})


def test_writer_survives_the_file_being_replaced_underneath_it(tmp_path):
    """O_APPEND writes go to the open descriptor, so events are never lost.

    This documents the behaviour rather than demanding a particular recovery:
    the point is that the write does not raise.
    """
    path = tmp_path / "events.jsonl"
    with events.EventWriter.open(str(path), "r1") as writer:
        writer.emit("log", {"text": "one"})
        os.remove(path)
        writer.emit("log", {"text": "two"})  # must not raise


def test_full_metric_pipeline_stdout_line_to_event_file(tmp_path):
    """End to end for stage 1: a training script's print becomes a log line.

    This is the exact path the sidecar will take in stage 2.
    """
    path = str(tmp_path / "events.jsonl")
    stdout_lines = [
        "starting training",
        '##relay## {"step": 100, "loss": 0.9}',
        "warning: something",
        '##relay## {"step": 200, "loss": 0.5, "note": "skip me"}',
    ]
    with events.EventWriter.open(path, "vr_s1_7f3a") as writer:
        for line in stdout_lines:
            payload = events.parse_metric_line(line)
            if payload is None:
                writer.emit("log", {"text": line})
            else:
                writer.emit("metric", payload)

    parsed = [events.parse_event(line) for line in read_lines(path)]
    assert [e["type"] for e in parsed] == ["log", "metric", "log", "metric"]
    assert [e["seq"] for e in parsed] == [1, 2, 3, 4]
    assert parsed[1]["payload"] == {"step": 100, "values": {"loss": 0.9}}
    assert parsed[3]["payload"] == {"step": 200, "values": {"loss": 0.5}}
