"""The paired Honcho gate counts only executed testcases and fails on runtime skips.

``scripts/ci/validate_honcho_pair.py`` judges the plugin suite from per-file JUnit XML.
Here real pytest children, run through the validator's own ``_pytest``/``_run``, write
that XML for small generated plugin test trees, and the real ``_run_file``/``_suite``/
``_tally``/``_problems`` judge it. Only duplicate testcase ids and the generated
ID-less bot control, which real pytest cannot produce here deterministically, are
answered at the ``_run`` subprocess boundary with hand-written pytest-shaped JUnit.
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.ci import validate_honcho_pair as pair

INI = "[pytest]\nmarkers =\n    allow_network: opens the network fence\n"

NETWORK_ONLY = """
import pytest

pytestmark = pytest.mark.allow_network


def test_fetch_wheel():
    pass


def test_fetch_toolchain():
    pass
"""

CONTROL_NAMES = [
    "test_idless_bot_turn_writes_nothing",
    "test_human_turn_still_writes",
    "test_legacy_turn_without_author_still_writes",
    "test_core_loader_and_memory_manager_route_idless_bot_to_no_writes",
]


def _args(tmp_path, expect_network):
    return argparse.Namespace(python=sys.executable, state=tmp_path / "state",
                              expect_network=expect_network)


def _plugin_tests(tmp_path, files):
    """A plugin ``tests/`` root that always carries a wholly network-marked file."""
    tests = tmp_path / "plugin" / "tests"
    tests.mkdir(parents=True)
    (tests / "pytest.ini").write_text(INI)
    for name, body in {"test_network_only.py": NETWORK_ONLY, **files}.items():
        (tests / name).write_text(textwrap.dedent(body))
    return tests


def _gate(args, tests):
    """The suite as cmd_test judges it, with the control held green."""
    suite = pair._suite(args, {"pythonpath": ""}, tests)
    suite.update(pair._tally(suite))
    problems = pair._problems(args, {"suite": suite, "idless_bot_control": {"ok": True}})
    return suite, {r["file"]: r for r in suite["file_results"]}, problems


def _junit(cases):
    children = {
        "passed": "",
        "skipped": '<skipped type="pytest.skip" message="skip">skip</skipped>',
        "failure": '<failure message="assert False">assert False</failure>',
    }
    body = "".join(f'<testcase classname="{c}" name="{n}" time="0.001">{children[o]}'
                   "</testcase>" for c, n, o in cases)
    count = {o: sum(1 for *_x, k in cases if k == o) for o in children}
    return ('<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
            f'<testsuite name="pytest" errors="0" failures="{count["failure"]}" '
            f'skipped="{count["skipped"]}" tests="{len(cases)}" time="0.01" '
            f'timestamp="2026-11-09T00:00:00.000000+00:00" hostname="ci">{body}'
            "</testsuite></testsuites>")


def _answer_pytest(monkeypatch, runs):
    """Answer each pytest child by its JUnit name: (exit, stdout, cases or None)."""
    seen = []

    def run(argv, *, env, cwd, timeout=pair.FILE_TIMEOUT):
        junit = Path(next(a for a in argv if a.startswith("--junitxml="))[len("--junitxml="):])
        seen.append((junit.stem, sorted(p.name for p in Path(cwd).iterdir())))
        rc, stdout, cases = runs[junit.stem]
        if cases is not None:
            junit.write_text(_junit(cases))
        return subprocess.CompletedProcess(argv, rc, stdout, "")

    monkeypatch.setattr(pair, "_run", run)
    return seen


def test_ordinary_suite_with_a_wholly_network_marked_file_is_green(tmp_path):
    args = _args(tmp_path, expect_network=3)
    suite, files, problems = _gate(args, _plugin_tests(tmp_path, {
        "test_plain.py": """
            def test_one():
                pass


            def test_two():
                pass
            """,
        "test_mixed_network.py": """
            import pytest


            def test_offline():
                pass


            @pytest.mark.allow_network
            def test_online():
                pass
            """,
    }))

    assert (suite["collected"], suite["network_deselected"], suite["files"]) == (6, 3, 3)
    assert (suite["executed"], suite["unique_executed"]) == (3, 3)
    network_only = files["test_network_only.py"]
    assert (network_only["exit"], network_only["ok"], network_only["cases"],
            network_only["expected"], network_only["network_deselected"]) == (5, True, 0, 0, 2)
    assert (files["test_mixed_network.py"]["ok"], files["test_mixed_network.py"]["cases"]) == (
        True, 1)
    assert suite["failed_or_error"] == [] and suite["bad_files"] == []
    assert problems == []


def test_runtime_skip_beside_a_pass_is_not_executed_and_fails_the_gate(tmp_path):
    args = _args(tmp_path, expect_network=2)
    suite, files, problems = _gate(args, _plugin_tests(tmp_path, {
        "test_mixed_skip.py": """
            import pytest


            def test_runs():
                pass


            def test_skips():
                pytest.skip("runtime skip")
            """,
    }))

    mixed = files["test_mixed_skip.py"]
    assert (mixed["exit"], mixed["cases"], mixed["expected"]) == (0, 2, 2)
    assert (suite["collected"], suite["network_deselected"]) == (4, 2)
    assert (mixed["ok"], suite["executed"], suite["unique_executed"],
            "file not green: test_mixed_skip.py" in problems) == (False, 1, 1, True)
    assert any("test_mixed_skip::test_skips" in p for p in problems)
    assert mixed["failed"] == [] and suite["failed_or_error"] == []
    assert mixed["skipped"] == ["test_mixed_skip::test_skips"]
    assert suite["skipped"] == ["test_mixed_skip::test_skips"]


def test_wholly_skipped_file_exits_zero_but_fails_the_gate(tmp_path):
    args = _args(tmp_path, expect_network=2)
    suite, files, problems = _gate(args, _plugin_tests(tmp_path, {
        "test_all_skip.py": """
            import pytest

            pytestmark = pytest.mark.skip(reason="whole module")


            def test_a():
                pass


            def test_b():
                pass
            """,
    }))

    skipped = files["test_all_skip.py"]
    assert (skipped["exit"], skipped["cases"], skipped["expected"]) == (0, 2, 2)
    assert suite["collected"] == 4
    assert (skipped["ok"], suite["executed"], suite["unique_executed"],
            "file not green: test_all_skip.py" in problems) == (False, 0, 0, True)
    assert skipped["failed"] == [] and suite["failed_or_error"] == []
    assert sorted(skipped["skipped"]) == ["test_all_skip::test_a", "test_all_skip::test_b"]
    assert sorted(suite["skipped"]) == ["test_all_skip::test_a", "test_all_skip::test_b"]


def test_failure_and_setup_error_still_fail_the_gate(tmp_path):
    args = _args(tmp_path, expect_network=2)
    suite, files, problems = _gate(args, _plugin_tests(tmp_path, {
        "test_fails.py": """
            def test_bad():
                assert False
            """,
        "test_errors.py": """
            import pytest


            @pytest.fixture
            def broken():
                raise RuntimeError("boom")


            def test_needs(broken):
                pass
            """,
    }))

    assert files["test_fails.py"]["failed"] == ["test_fails::test_bad failure"]
    assert files["test_errors.py"]["failed"] == ["test_errors::test_needs error"]
    assert sorted(suite["failed_or_error"]) == ["test_errors::test_needs error",
                                                "test_fails::test_bad failure"]
    assert (suite["executed"], suite["unique_executed"]) == (2, 2)
    assert sorted(suite["bad_files"]) == ["test_errors.py", "test_fails.py"]
    assert "file not green: test_fails.py" in problems
    assert "file not green: test_errors.py" in problems


def test_duplicate_junit_ids_still_fail_the_gate(tmp_path, monkeypatch):
    _answer_pytest(monkeypatch, {
        "collect-all": (0, "test_dup.py::test_x\ntest_net.py::test_n\n\n2 tests collected\n",
                        None),
        "collect-allow_network": (0, "test_net.py::test_n\n\n1/2 tests collected\n", None),
        "test_dup.py": (0, "", [("test_dup", "test_x", "passed")] * 2),
        "test_net.py": (5, "", []),
    })
    tests = tmp_path / "tests"
    tests.mkdir()
    suite, files, problems = _gate(_args(tmp_path, expect_network=1), tests)

    assert (suite["collected"], suite["executed"], suite["unique_executed"]) == (2, 2, 1)
    assert files["test_dup.py"]["ok"] is False
    assert files["test_net.py"]["ok"] is True
    assert "duplicate testcase ids in JUnit results" in problems


@pytest.mark.parametrize("outcomes, ok", [
    (["passed"] * 4, True),
    (["passed", "passed", "passed", "skipped"], False),
], ids=["all-four-pass", "one-skipped"])
def test_control_is_ok_only_when_all_four_cases_pass(tmp_path, monkeypatch, outcomes, ok):
    seen = _answer_pytest(monkeypatch, {"control": (0, "", [
        ("test_pair_ci_idless_bot_control", name, outcome)
        for name, outcome in zip(CONTROL_NAMES, outcomes)])})
    tests = tmp_path / "tests"
    tests.mkdir()

    result = pair._run_control(_args(tmp_path, expect_network=0), {"pythonpath": ""}, tests)

    assert seen == [("control", ["test_pair_ci_idless_bot_control.py"])]
    assert list(tests.iterdir()) == []
    assert (result["exit"], result["cases"], result["ok"]) == (0, 4, ok)
    assert result["passed"] == CONTROL_NAMES[:outcomes.count("passed")]
