from typer.testing import CliRunner

from brandsentinel import __version__
from brandsentinel.cli import app

runner = CliRunner()


def test_version():
    assert __version__


def test_cli_help():
    assert runner.invoke(app, ["--help"]).exit_code == 0


def test_cli_run_stub():
    args = [
        "run",
        "--stage",
        "preprocess",
        "--category",
        "Baby_Products",
        "--start",
        "2023-01-01",
        "--end",
        "2023-03-31",
    ]
    assert runner.invoke(app, args).exit_code == 0
