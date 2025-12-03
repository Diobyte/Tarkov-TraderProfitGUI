#!/usr/bin/env python3
"""
Generate a reference file of all Tarkov items with their Flea Market level requirements.

This script fetches item data from the tarkov.dev GraphQL API and generates a
JSON/CSV file mapping each item to its required player level for Flea Market access.

Usage:
    python generate_flea_levels.py

Output:
    - data/flea_level_requirements.json
    - data/flea_level_requirements.csv
"""

import json
import csv
import os
import sys
from datetime import datetime, timezone
from typing import Any, List

import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CATEGORY_LOCKS, ITEM_LOCKS, FLEA_MARKET_UNLOCK_LEVEL
from utils import get_flea_level_requirement


API_URL = "https://api.tarkov.dev/graphql"

QUERY = """
{
    items {
        id
        name
        shortName
        basePrice
        types
        category {
            name
        }
        handbookCategories {
            name
        }
        sellFor {
            vendor {
                name
            }
            price
        }
    }
}
"""


def fetch_all_items() -> list[dict[str, Any]]:
    """Fetch all items from the tarkov.dev API.
    
    Returns:
        List of item dictionaries from the API.
    """
    print("Fetching items from tarkov.dev API...")
    
    response = requests.post(
        API_URL,
        json={"query": QUERY},
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    response.raise_for_status()
    
    data = response.json()
    items = data.get("data", {}).get("items", [])
    print(f"Fetched {len(items):,} items from API")
    
    return items


def process_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process items and calculate flea level requirements.
    
    Args:
        items: Raw item data from the API.
        
    Returns:
        Processed list of items with flea level requirements.
    """
    processed = []
    
    for item in items:
        item_id = item.get("id", "")
        name = item.get("name", "")
        short_name = item.get("shortName", "")
        base_price = item.get("basePrice", 0)
        types = item.get("types", [])
        category = item.get("category", {}).get("name", "Unknown") if item.get("category") else "Unknown"
        
        # Extract handbook categories
        handbook_categories = []
        handbook_data = item.get("handbookCategories") or []
        if isinstance(handbook_data, list):
            for hc in handbook_data:
                if isinstance(hc, dict) and hc.get("name"):
                    handbook_categories.append(hc.get("name"))
        
        # Check for noFlea and markedOnly flags
        no_flea = "noFlea" in types if types else False
        marked_only = "markedOnly" in types if types else False
        
        # Get best trader sell price
        best_trader = None
        best_price = 0
        for sell in item.get("sellFor", []):
            vendor = sell.get("vendor", {}).get("name", "")
            price = sell.get("price", 0)
            if vendor != "Flea Market" and price > best_price:
                best_price = price
                best_trader = vendor
        
        # Calculate flea level requirement (pass handbook categories for better matching)
        flea_level = get_flea_level_requirement(name, category, handbook_categories)
        
        processed.append({
            "item_id": item_id,
            "name": name,
            "short_name": short_name,
            "category": category,
            "handbook_categories": ", ".join(handbook_categories) if handbook_categories else "",
            "types": ", ".join(types) if types else "",
            "no_flea": no_flea,
            "marked_only": marked_only,
            "base_price": base_price,
            "best_trader": best_trader or "None",
            "trader_price": best_price,
            "flea_level_required": flea_level,
            "flea_locked": flea_level > FLEA_MARKET_UNLOCK_LEVEL
        })
    
    # Sort by flea level (descending), then by name
    processed.sort(key=lambda x: (-x["flea_level_required"], x["name"]))
    
    return processed


def generate_output_files(items: list[dict[str, Any]]) -> None:
    """Generate JSON and CSV output files.
    
    Args:
        items: Processed item data.
    """
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Generate statistics
    total = len(items)
    locked_items = [i for i in items if i["flea_locked"]]
    level_distribution = {}
    for item in items:
        level = item["flea_level_required"]
        level_distribution[level] = level_distribution.get(level, 0) + 1
    
    # Create metadata
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_items": total,
        "locked_items_count": len(locked_items),
        "base_flea_level": FLEA_MARKET_UNLOCK_LEVEL,
        "level_distribution": dict(sorted(level_distribution.items())),
        "category_locks": CATEGORY_LOCKS,
        "item_locks": ITEM_LOCKS
    }
    
    # Write JSON file
    json_path = "data/flea_level_requirements.json"
    json_output = {
        "metadata": metadata,
        "items": items
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"Generated: {json_path}")
    
    # Write CSV file
    csv_path = "data/flea_level_requirements.csv"
    fieldnames = ["item_id", "name", "short_name", "category", "types", 
                  "base_price", "best_trader", "trader_price", 
                  "flea_level_required", "flea_locked"]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)
    print(f"Generated: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FLEA MARKET LEVEL REQUIREMENTS - SUMMARY")
    print("=" * 60)
    print(f"Total Items: {total:,}")
    print(f"Locked Items (>{FLEA_MARKET_UNLOCK_LEVEL}): {len(locked_items):,}")
    print(f"Unlocked at Base Level ({FLEA_MARKET_UNLOCK_LEVEL}): {total - len(locked_items):,}")
    print("\nLevel Distribution:")
    for level in sorted(level_distribution.keys(), reverse=True):
        count = level_distribution[level]
        bar = "█" * min(count // 10, 40)
        locked_marker = " 🔒" if level > FLEA_MARKET_UNLOCK_LEVEL else ""
        print(f"  Level {level:2d}: {count:4d} items {bar}{locked_marker}")
    
    print("\n" + "=" * 60)
    print("Top 20 Highest Level Requirements:")
    print("=" * 60)
    for item in items[:20]:
        print(f"  Lvl {item['flea_level_required']:2d} | {item['name'][:50]:<50} | {item['category']}")


def main() -> None:
    """Main entry point."""
    try:
        items = fetch_all_items()
        processed = process_items(items)
        generate_output_files(processed)
        print("\n✅ Successfully generated flea level requirements files!")
    except requests.RequestException as e:
        print(f"❌ API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
