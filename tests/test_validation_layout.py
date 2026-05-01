from pathlib import Path


def test_validation_scripts_do_not_use_pytest_names():
    repo_root = Path(__file__).resolve().parents[1]

    script_style_tests = [
        path
        for directory in (repo_root / "src", repo_root / "scripts")
        for path in directory.glob("test_*.py")
    ]

    assert script_style_tests == []
