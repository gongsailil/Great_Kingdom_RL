import hashlib
import tempfile
from pathlib import Path

from run_alternating_generation import (
    learner_player_number,
    model_manifest,
    planned_child_models,
    validate_generation_paths,
)


def expect_exception(exception_type, function):
    try:
        function()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def test_invalid_learner_player_is_rejected():
    expect_exception(ValueError, lambda: learner_player_number("green"))


def test_player_mapping():
    assert learner_player_number("blue") == 1
    assert learner_player_number("red") == 2


def test_paths_hashes_overwrite_and_single_child_plan():
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        parent = temp_dir / "parent.zip"
        latest = temp_dir / "latest.zip"
        previous = temp_dir / "previous.zip"
        output = temp_dir / "child.zip"
        report = temp_dir / "report"
        parent.write_bytes(b"parent")
        latest.write_bytes(b"latest")
        previous.write_bytes(b"previous")

        validate_generation_paths(
            parent_model=parent,
            latest_opponent=latest,
            previous_opponent=previous,
            output_model=output,
            report_dir=report,
        )
        manifest = model_manifest(parent)
        assert manifest == {
            "path": str(parent),
            "sha256": hashlib.sha256(b"parent").hexdigest(),
        }
        assert planned_child_models(output) == [output]

        missing = temp_dir / "missing.zip"
        expect_exception(
            FileNotFoundError,
            lambda: validate_generation_paths(
                parent_model=missing,
                latest_opponent=latest,
                previous_opponent=previous,
                output_model=output,
                report_dir=report,
            ),
        )

        output.write_bytes(b"do not overwrite")
        expect_exception(
            FileExistsError,
            lambda: validate_generation_paths(
                parent_model=parent,
                latest_opponent=latest,
                previous_opponent=previous,
                output_model=output,
                report_dir=report,
            ),
        )


if __name__ == "__main__":
    test_invalid_learner_player_is_rejected()
    test_player_mapping()
    test_paths_hashes_overwrite_and_single_child_plan()
    print("alternating one-generation runner tests: PASS")
