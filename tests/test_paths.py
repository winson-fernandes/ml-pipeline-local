import time

from src.common.paths import latest_file, latest_run_dir


def test_latest_file_returns_none_for_an_empty_directory(tmp_path):
    assert latest_file(str(tmp_path), ".csv") is None


def test_latest_file_returns_none_for_a_missing_directory(tmp_path):
    missing_dir = tmp_path / "does-not-exist"
    assert latest_file(str(missing_dir), ".csv") is None


def test_latest_file_returns_the_most_recently_modified_match(tmp_path):
    older = tmp_path / "data_2020.csv"
    newer = tmp_path / "data_2021.csv"
    ignored = tmp_path / "notes.txt"

    older.write_text("old")
    time.sleep(0.01)
    ignored.write_text("ignored")
    time.sleep(0.01)
    newer.write_text("new")

    result = latest_file(str(tmp_path), ".csv")

    assert result == str(newer)


def test_latest_run_dir_returns_the_most_recently_modified_subdirectory(tmp_path):
    first_run = tmp_path / "run_1"
    second_run = tmp_path / "run_2"
    first_run.mkdir()
    time.sleep(0.01)
    second_run.mkdir()

    result = latest_run_dir(str(tmp_path))

    assert result == str(second_run)


def test_latest_run_dir_ignores_files(tmp_path):
    (tmp_path / "not_a_run.txt").write_text("ignored")

    assert latest_run_dir(str(tmp_path)) is None
