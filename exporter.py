"""
Data Export Module for Tarkov Trader Profit Analysis.

Provides functionality to export market data, analysis results,
and trading recommendations in various formats (CSV, JSON, Excel).
"""

import json
import csv
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Final
from io import StringIO, BytesIO

import pandas as pd

__all__: List[str] = ['DataExporter', 'ExportFormat', 'get_exporter']

logger = logging.getLogger(__name__)

BASE_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
EXPORTS_DIR: Final[str] = os.path.join(BASE_DIR, 'exports')


class ExportFormat:
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "xlsx"
    MARKDOWN = "md"


class DataExporter:
    """
    Handles data export in multiple formats.
    
    Thread-safe singleton for exporting market data, analysis results,
    and trading recommendations. Automatically manages export directory
    and filename generation.
    
    Supports exporting:
    - Current market data (CSV, JSON, Excel, Markdown)
    - Analysis results with formatting
    - Trading recommendations with profit formatting  
    - Historical trends
    - Market snapshots in all formats
    
    Example:
        >>> exporter = get_exporter()
        >>> filepath = exporter.export_to_csv(df, prefix='market')
    """
    
    def __init__(self, export_dir: Optional[str] = None) -> None:
        """Initialize the exporter."""
        self.export_dir = export_dir or EXPORTS_DIR
        os.makedirs(self.export_dir, exist_ok=True)
    
    def _generate_filename(self, prefix: str, format: str) -> str:
        """Generate a timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.export_dir, f"{prefix}_{timestamp}.{format}")
    
    def export_to_csv(self, df: pd.DataFrame, filename: Optional[str] = None,
                      prefix: str = "export") -> str:
        """
        Export DataFrame to CSV file.
        
        Args:
            df: DataFrame to export.
            filename: Optional specific filename.
            prefix: Prefix for auto-generated filename.
            
        Returns:
            Path to the exported file.
        """
        if filename is None:
            filename = self._generate_filename(prefix, "csv")
        
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info("Exported %d rows to %s", len(df), filename)
        return filename
    
    def export_to_json(self, data: Any, filename: Optional[str] = None,
                       prefix: str = "export", pretty: bool = True) -> str:
        """
        Export data to JSON file.
        
        Args:
            data: Data to export (dict, list, or DataFrame).
            filename: Optional specific filename.
            prefix: Prefix for auto-generated filename.
            pretty: Whether to pretty-print JSON.
            
        Returns:
            Path to the exported file.
        """
        if filename is None:
            filename = self._generate_filename(prefix, "json")
        
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        
        with open(filename, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)
        
        logger.info("Exported data to %s", filename)
        return filename
    
    def export_to_excel(self, df: pd.DataFrame, filename: Optional[str] = None,
                        prefix: str = "export", sheet_name: str = "Data") -> str:
        """
        Export DataFrame to Excel file.
        
        Requires openpyxl package for Excel support. Falls back to CSV if
        openpyxl is not installed or if Excel export fails.
        
        Args:
            df: DataFrame to export.
            filename: Optional specific filename. Auto-generated if not provided.
            prefix: Prefix for auto-generated filename.
            sheet_name: Name of the Excel sheet (max 31 chars, special chars replaced).
            
        Returns:
            Path to the exported file (may be CSV if Excel fails).
            
        Raises:
            No exceptions are raised; errors fall back to CSV export.
        """
        if filename is None:
            filename = self._generate_filename(prefix, "xlsx")
        
        try:
            # Limit sheet name to 31 characters (Excel limit)
            safe_sheet_name = str(sheet_name)[:31] if sheet_name else "Data"
            # Also ensure sheet name doesn't contain invalid characters
            invalid_chars = [':', '\\', '/', '?', '*', '[', ']', "'", '"']
            for char in invalid_chars:
                safe_sheet_name = safe_sheet_name.replace(char, '_')
            # Sheet name cannot start or end with apostrophe, and cannot be empty
            safe_sheet_name = safe_sheet_name.strip().strip("'").strip()
            if not safe_sheet_name:
                safe_sheet_name = "Data"
            # Excel also doesn't allow sheet names that are just whitespace
            if not safe_sheet_name.replace('_', '').strip():
                safe_sheet_name = "Data"
            df.to_excel(filename, index=False, sheet_name=safe_sheet_name, engine='openpyxl')
            logger.info("Exported %d rows to %s", len(df), filename)
        except ImportError:
            # Fallback to CSV if openpyxl not installed
            logger.warning("openpyxl not installed, falling back to CSV")
            filename = filename.replace('.xlsx', '.csv')
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info("Exported %d rows to %s (CSV fallback)", len(df), filename)
        except Exception as e:
            logger.error("Excel export failed: %s, falling back to CSV", e)
            filename = filename.replace('.xlsx', '.csv')
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info("Exported %d rows to %s (CSV fallback)", len(df), filename)
        
        return filename
    
    def export_to_markdown(self, df: pd.DataFrame, filename: Optional[str] = None,
                           prefix: str = "export", title: str = "Data Export") -> str:
        """
        Export DataFrame to Markdown table.
        
        Args:
            df: DataFrame to export.
            filename: Optional specific filename.
            prefix: Prefix for auto-generated filename.
            title: Title for the markdown document.
            
        Returns:
            Path to the exported file.
        """
        if filename is None:
            filename = self._generate_filename(prefix, "md")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(f"Total items: {len(df)}\n\n")
            
            # Write table header
            headers = df.columns.tolist()
            f.write("| " + " | ".join(str(h) for h in headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            
            # Write rows (limit to 100 for readability)
            for _, row in df.head(100).iterrows():
                values = [str(v)[:50] for v in row.values]  # Truncate long values
                f.write("| " + " | ".join(values) + " |\n")
            
            if len(df) > 100:
                f.write(f"\n*... and {len(df) - 100} more rows*\n")
        
        logger.info("Exported markdown to %s", filename)
        return filename
    
    def export_recommendations(self, df: pd.DataFrame, 
                               format: str = ExportFormat.CSV) -> str:
        """
        Export trading recommendations in a user-friendly format.
        
        Args:
            df: DataFrame with recommendations.
            format: Export format (csv, json, xlsx, md).
            
        Returns:
            Path to the exported file.
            
        Raises:
            ValueError: If format is not supported.
        """
        if df.empty:
            # Handle empty DataFrame gracefully
            empty_df = pd.DataFrame(columns=['name', 'profit', 'roi'])
            if format == ExportFormat.CSV:
                return self.export_to_csv(empty_df, prefix="recommendations")
            elif format == ExportFormat.JSON:
                return self.export_to_json(empty_df, prefix="recommendations")
            elif format == ExportFormat.EXCEL:
                return self.export_to_excel(empty_df, prefix="recommendations")
            elif format == ExportFormat.MARKDOWN:
                return self.export_to_markdown(empty_df, prefix="recommendations",
                                               title="Trading Recommendations")
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        # Select key columns for recommendations
        rec_columns = [
            'name', 'profit', 'roi', 'flea_price', 'trader_price',
            'trader_name', 'category', 'last_offer_count',
            'rec_score', 'rec_tier', 'risk_level'
        ]
        
        # Only include columns that exist
        available_cols = [c for c in rec_columns if c in df.columns]
        export_df = df[available_cols].copy()
        
        # Format for readability - handle potential NaN values
        if 'profit' in export_df.columns:
            export_df['profit'] = export_df['profit'].apply(
                lambda x: f"₽{x:,.0f}" if pd.notna(x) and not (isinstance(x, float) and (x == float('inf') or x == float('-inf'))) else "₽0"
            )
        if 'roi' in export_df.columns:
            export_df['roi'] = export_df['roi'].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) and not (isinstance(x, float) and (x == float('inf') or x == float('-inf'))) else "0.0%"
            )
        if 'flea_price' in export_df.columns:
            export_df['flea_price'] = export_df['flea_price'].apply(
                lambda x: f"₽{x:,.0f}" if pd.notna(x) and not (isinstance(x, float) and (x == float('inf') or x == float('-inf'))) else "₽0"
            )
        if 'trader_price' in export_df.columns:
            export_df['trader_price'] = export_df['trader_price'].apply(
                lambda x: f"₽{x:,.0f}" if pd.notna(x) and not (isinstance(x, float) and (x == float('inf') or x == float('-inf'))) else "₽0"
            )
        
        if format == ExportFormat.CSV:
            return self.export_to_csv(export_df, prefix="recommendations")
        elif format == ExportFormat.JSON:
            return self.export_to_json(export_df, prefix="recommendations")
        elif format == ExportFormat.EXCEL:
            return self.export_to_excel(export_df, prefix="recommendations")
        elif format == ExportFormat.MARKDOWN:
            return self.export_to_markdown(export_df, prefix="recommendations",
                                           title="Trading Recommendations")
        else:
            logger.warning("Unsupported export format '%s', falling back to CSV", format)
            return self.export_to_csv(export_df, prefix="recommendations")
    
    def export_market_snapshot(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Export a complete market snapshot in all formats.
        
        Args:
            df: Full market data DataFrame.
            
        Returns:
            Dict mapping format to filepath.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"market_snapshot_{timestamp}"
        
        results = {}
        
        # CSV for data analysis
        csv_path = os.path.join(self.export_dir, f"{prefix}.csv")
        df.to_csv(csv_path, index=False)
        results['csv'] = csv_path
        
        # JSON for programmatic access
        json_path = os.path.join(self.export_dir, f"{prefix}.json")
        df.to_json(json_path, orient='records', indent=2)
        results['json'] = json_path
        
        # Summary markdown
        summary = self._generate_summary(df)
        md_path = os.path.join(self.export_dir, f"{prefix}_summary.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        results['summary'] = md_path
        
        logger.info("Exported market snapshot: %d items", len(df))
        return results
    
    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate a markdown summary of market data."""
        lines = [
            "# Market Snapshot Summary",
            f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "## Overview",
            f"- Total items: {len(df)}",
        ]
        
        if 'profit' in df.columns:
            profitable = (df['profit'] > 0).sum()
            avg_profit = df[df['profit'] > 0]['profit'].mean()
            max_profit = df['profit'].max()
            lines.extend([
                f"- Profitable items: {profitable}",
                f"- Average profit: ₽{avg_profit:,.0f}",
                f"- Max profit: ₽{max_profit:,.0f}",
            ])
        
        if 'category' in df.columns:
            lines.append("\n## Top Categories by Item Count")
            top_cats = df['category'].value_counts().head(10)
            for cat, count in top_cats.items():
                lines.append(f"- {cat}: {count} items")
        
        if 'trader_name' in df.columns:
            lines.append("\n## Items by Trader")
            trader_counts = df['trader_name'].value_counts()
            for trader, count in trader_counts.items():
                lines.append(f"- {trader}: {count} items")
        
        lines.append("\n## Top 10 Most Profitable Items")
        if 'profit' in df.columns and 'name' in df.columns:
            top_items = df.nlargest(10, 'profit')[['name', 'profit', 'category']]
            for _, item in top_items.iterrows():
                lines.append(f"- **{item['name']}**: ₽{item['profit']:,.0f} ({item.get('category', 'N/A')})")
        
        return "\n".join(lines)
    
    def get_csv_string(self, df: pd.DataFrame) -> str:
        """Get CSV as string (for clipboard/download)."""
        return df.to_csv(index=False)
    
    def get_json_string(self, data: Any) -> str:
        """Get JSON as string."""
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        return json.dumps(data, indent=2, default=str)
    
    def cleanup_old_exports(self, days: int = 7) -> int:
        """
        Remove export files older than specified days.
        
        Args:
            days: Delete files older than this many days. 0 means delete all files,
                  negative values are invalid.
            
        Returns:
            Number of files deleted. Returns 0 if days is invalid.
        """
        from datetime import timedelta
        
        # Validate days parameter - negative values are invalid, 0 means "all files"
        if days < 0:
            logger.warning("cleanup_old_exports called with negative days: %d, skipping", days)
            return 0
        
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0
        
        if not os.path.exists(self.export_dir):
            return 0
        
        for filename in os.listdir(self.export_dir):
            filepath = os.path.join(self.export_dir, filename)
            if os.path.isfile(filepath):
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if mtime < cutoff:
                        os.remove(filepath)
                        deleted += 1
                except (OSError, ValueError) as e:
                    logger.warning("Failed to delete %s: %s", filepath, e)
        
        if deleted:
            logger.info("Cleaned up %d old export files", deleted)
        
        return deleted


# Singleton instance with thread-safe initialization
import threading
_exporter: Optional[DataExporter] = None
_exporter_lock: threading.Lock = threading.Lock()


def get_exporter() -> DataExporter:
    """Get or create the data exporter singleton (thread-safe)."""
    global _exporter
    if _exporter is None:
        with _exporter_lock:
            # Double-check locking pattern
            if _exporter is None:
                _exporter = DataExporter()
    return _exporter
