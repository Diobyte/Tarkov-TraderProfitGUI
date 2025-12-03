"""
Version information and update checking for Tarkov Trader Profit GUI.

This module provides utilities to:
- Read the current version from the VERSION file
- Check if a newer version is available on GitHub
- Display version info (used by app.py and Docker containers)
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# GitHub repository information
GITHUB_OWNER = "Diobyte"
GITHUB_REPO = "Tarkov-TraderProfitGUI"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"

# Version file locations (check Docker paths first, then local)
VERSION_PATHS = [
    Path("/app/.version"),  # Docker container path
    Path(__file__).parent / "VERSION",  # Local development path
]


@lru_cache(maxsize=1)
def get_current_version() -> str:
    """
    Get the current version of the application.
    
    Returns:
        str: The current version string (e.g., "1.0.0"), or "unknown" if not found.
    """
    for path in VERSION_PATHS:
        try:
            if path.exists():
                version = path.read_text().strip()
                if version:
                    return version
        except Exception as e:
            logger.debug(f"Could not read version from {path}: {e}")
    
    return "unknown"


def get_build_info() -> dict:
    """
    Get build information (available in Docker containers).
    
    Returns:
        dict: Dictionary with version, build_date, and git_commit.
    """
    info = {
        "version": get_current_version(),
        "build_date": None,
        "git_commit": None,
        "is_docker": False,
    }
    
    # Check if running in Docker container
    docker_paths = {
        "build_date": Path("/app/.build_date"),
        "git_commit": Path("/app/.git_commit"),
    }
    
    for key, path in docker_paths.items():
        try:
            if path.exists():
                info[key] = path.read_text().strip()
                info["is_docker"] = True
        except Exception:
            pass
    
    return info


def parse_version(version_str: str) -> Tuple[int, ...]:
    """
    Parse a version string into a tuple of integers for comparison.
    
    Args:
        version_str: Version string like "1.2.3" or "v1.2.3"
        
    Returns:
        Tuple of integers representing the version.
    """
    # Remove 'v' prefix if present
    version_str = version_str.lstrip("v")
    
    # Split and convert to integers
    parts = []
    for part in version_str.split("."):
        try:
            # Handle versions like "1.2.3-beta" by taking only the numeric part
            numeric = "".join(c for c in part if c.isdigit())
            parts.append(int(numeric) if numeric else 0)
        except ValueError:
            parts.append(0)
    
    return tuple(parts)


def check_for_updates(timeout: int = 5) -> Optional[dict]:
    """
    Check GitHub for newer versions of the application.
    
    Args:
        timeout: Request timeout in seconds.
        
    Returns:
        dict with update info if available, None if current or error.
        {
            "update_available": bool,
            "current_version": str,
            "latest_version": str,
            "release_url": str,
            "release_notes": str,
        }
    """
    current = get_current_version()
    
    if current == "unknown":
        logger.warning("Cannot check for updates: current version unknown")
        return None
    
    try:
        # Check latest release from GitHub API
        response = requests.get(
            f"{GITHUB_API_URL}/releases/latest",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=timeout,
        )
        
        if response.status_code == 404:
            # No releases yet
            logger.debug("No releases found on GitHub")
            return {
                "update_available": False,
                "current_version": current,
                "latest_version": current,
                "release_url": None,
                "release_notes": None,
            }
        
        response.raise_for_status()
        release_data = response.json()
        
        latest = release_data.get("tag_name", "").lstrip("v")
        
        if not latest:
            return None
        
        # Compare versions
        current_tuple = parse_version(current)
        latest_tuple = parse_version(latest)
        
        update_available = latest_tuple > current_tuple
        
        return {
            "update_available": update_available,
            "current_version": current,
            "latest_version": latest,
            "release_url": release_data.get("html_url"),
            "release_notes": release_data.get("body", "")[:500],  # Truncate notes
        }
        
    except requests.RequestException as e:
        logger.warning(f"Failed to check for updates: {e}")
        return None
    except Exception as e:
        logger.error(f"Error checking for updates: {e}")
        return None


def check_docker_image_update(timeout: int = 10) -> Optional[dict]:
    """
    Check if a newer Docker image is available on GHCR.
    
    This compares the local image's version label against the latest on GHCR.
    
    Args:
        timeout: Request timeout in seconds.
        
    Returns:
        dict with update info if available, None if current or error.
    """
    build_info = get_build_info()
    
    if not build_info["is_docker"]:
        logger.debug("Not running in Docker, skipping Docker image update check")
        return None
    
    # For Docker, we use the GitHub releases check as the source of truth
    # since the VERSION file in the image matches the release version
    return check_for_updates(timeout=timeout)


def get_version_display() -> str:
    """
    Get a formatted version string for display in the UI.
    
    Returns:
        str: Formatted version string like "v1.0.0" or "v1.0.0 (abc1234)"
    """
    info = get_build_info()
    version = info["version"]
    
    if version == "unknown":
        return "Development Version"
    
    display = f"v{version}"
    
    if info["git_commit"]:
        # Show first 7 characters of commit hash
        short_commit = info["git_commit"][:7]
        display += f" ({short_commit})"
    
    return display


if __name__ == "__main__":
    # CLI usage for quick version check
    import json
    
    print(f"Current Version: {get_version_display()}")
    print()
    
    build_info = get_build_info()
    print("Build Info:")
    print(json.dumps(build_info, indent=2))
    print()
    
    print("Checking for updates...")
    update_info = check_for_updates()
    if update_info:
        print(json.dumps(update_info, indent=2))
        if update_info["update_available"]:
            print(f"\n🆕 Update available! {update_info['latest_version']}")
            print(f"   Download: {update_info['release_url']}")
    else:
        print("Could not check for updates")
