"""Tests for version module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import version


class TestGetCurrentVersion:
    """Tests for get_current_version function."""

    def test_returns_version_from_file(self, tmp_path: Path) -> None:
        """Test reading version from VERSION file."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("1.2.3")
        
        with patch.object(version, "VERSION_PATHS", [version_file]):
            version.get_current_version.cache_clear()
            result = version.get_current_version()
            
        assert result == "1.2.3"

    def test_returns_unknown_when_no_file(self, tmp_path: Path) -> None:
        """Test returns 'unknown' when VERSION file doesn't exist."""
        non_existent = tmp_path / "nonexistent" / "VERSION"
        
        with patch.object(version, "VERSION_PATHS", [non_existent]):
            version.get_current_version.cache_clear()
            result = version.get_current_version()
            
        assert result == "unknown"

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        """Test that version is stripped of whitespace."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("  1.0.0\n\n")
        
        with patch.object(version, "VERSION_PATHS", [version_file]):
            version.get_current_version.cache_clear()
            result = version.get_current_version()
            
        assert result == "1.0.0"


class TestParseVersion:
    """Tests for parse_version function."""

    def test_parses_simple_version(self) -> None:
        """Test parsing simple version string."""
        assert version.parse_version("1.2.3") == (1, 2, 3)

    def test_parses_version_with_v_prefix(self) -> None:
        """Test parsing version with 'v' prefix."""
        assert version.parse_version("v1.2.3") == (1, 2, 3)

    def test_parses_two_part_version(self) -> None:
        """Test parsing two-part version."""
        assert version.parse_version("1.0") == (1, 0)

    def test_parses_version_with_suffix(self) -> None:
        """Test parsing version with suffix like -beta."""
        result = version.parse_version("1.2.3-beta")
        assert result == (1, 2, 3)

    def test_handles_empty_string(self) -> None:
        """Test handling empty string returns zero tuple."""
        result = version.parse_version("")
        assert result == (0,)

    def test_version_comparison(self) -> None:
        """Test that parsed versions compare correctly."""
        v1 = version.parse_version("1.0.0")
        v2 = version.parse_version("1.0.1")
        v3 = version.parse_version("2.0.0")
        
        assert v1 < v2
        assert v2 < v3
        assert v1 < v3


class TestGetBuildInfo:
    """Tests for get_build_info function."""

    def test_returns_basic_info_when_not_docker(self, tmp_path: Path) -> None:
        """Test returns basic info when not in Docker."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("1.0.0")
        
        with patch.object(version, "VERSION_PATHS", [version_file]):
            version.get_current_version.cache_clear()
            result = version.get_build_info()
            
        assert result["version"] == "1.0.0"
        assert result["is_docker"] is False
        assert result["build_date"] is None
        assert result["git_commit"] is None


class TestCheckForUpdates:
    """Tests for check_for_updates function."""

    def test_returns_none_when_version_unknown(self) -> None:
        """Test returns None when current version is unknown."""
        with patch.object(version, "get_current_version", return_value="unknown"):
            result = version.check_for_updates()
            
        assert result is None

    def test_returns_update_available_when_newer(self) -> None:
        """Test returns update available when newer version exists."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tag_name": "v2.0.0",
            "html_url": "https://github.com/test/release",
            "body": "Release notes",
        }
        
        with patch.object(version, "get_current_version", return_value="1.0.0"):
            with patch("requests.get", return_value=mock_response):
                result = version.check_for_updates()
        
        assert result is not None
        assert result["update_available"] is True
        assert result["current_version"] == "1.0.0"
        assert result["latest_version"] == "2.0.0"

    def test_returns_no_update_when_current(self) -> None:
        """Test returns no update when on current version."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tag_name": "v1.0.0",
            "html_url": "https://github.com/test/release",
            "body": "Release notes",
        }
        
        with patch.object(version, "get_current_version", return_value="1.0.0"):
            with patch("requests.get", return_value=mock_response):
                result = version.check_for_updates()
        
        assert result is not None
        assert result["update_available"] is False

    def test_handles_no_releases(self) -> None:
        """Test handles case when no releases exist."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        with patch.object(version, "get_current_version", return_value="1.0.0"):
            with patch("requests.get", return_value=mock_response):
                result = version.check_for_updates()
        
        assert result is not None
        assert result["update_available"] is False

    def test_handles_network_error(self) -> None:
        """Test handles network errors gracefully."""
        import requests
        
        with patch.object(version, "get_current_version", return_value="1.0.0"):
            with patch("requests.get", side_effect=requests.RequestException("Network error")):
                result = version.check_for_updates()
        
        assert result is None


class TestGetVersionDisplay:
    """Tests for get_version_display function."""

    def test_returns_development_when_unknown(self) -> None:
        """Test returns 'Development Version' when version unknown."""
        with patch.object(version, "get_build_info", return_value={
            "version": "unknown",
            "git_commit": None,
            "build_date": None,
            "is_docker": False,
        }):
            result = version.get_version_display()
            
        assert result == "Development Version"

    def test_returns_version_with_v_prefix(self) -> None:
        """Test returns version with 'v' prefix."""
        with patch.object(version, "get_build_info", return_value={
            "version": "1.0.0",
            "git_commit": None,
            "build_date": None,
            "is_docker": False,
        }):
            result = version.get_version_display()
            
        assert result == "v1.0.0"

    def test_includes_short_commit_hash(self) -> None:
        """Test includes shortened commit hash when available."""
        with patch.object(version, "get_build_info", return_value={
            "version": "1.0.0",
            "git_commit": "abc1234567890",
            "build_date": None,
            "is_docker": True,
        }):
            result = version.get_version_display()
            
        assert result == "v1.0.0 (abc1234)"
