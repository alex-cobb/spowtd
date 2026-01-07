"""Test for valid version string"""

import spowtd
import pathlib


def test_version_matches_file():
    """Test that package version matches version string in VERSION.txt"""
    root_dir = pathlib.Path(__file__).parents[2]
    version_file = root_dir / 'spowtd' / 'VERSION.txt'

    if version_file.exists():
        expected = version_file.read_text().strip()
        assert spowtd.__version__ == expected, 'Version matches file'
    else:
        # Running from installed package
        assert spowtd.__version__ != 'unknown'
