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
from typing import Dict, Any, Optional, List
from io import StringIO, BytesIO

import pandas as pd

__all__ = ['DataExporter', 'ExportFormat']

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORTS_DIR = os.path.join(BASE_DIR, 'exports')


class ExportFormat:
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "xlsx"
    MARKDOWN = "md"


class DataExporter:
    """
    Handles data export in multiple formats.
    
    Supports exporting:
    - Current market data
    - Analysis results
    - Trading recommendations
    - Historical trends
    - Alert history
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
        logger.info(f"Exported {len(df)} rows to {filename}")
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
        
        logger.info(f"Exported data to {filename}")
        return filename
    
    def export_to_excel(self, df: pd.DataFrame, filename: Optional[str] = None,
                        prefix: str = "export", sheet_name: str = "Data") -> str:
        """
        Export DataFrame to Excel file.
        
        Args:
            df: DataFrame to export.
            filename: Optional specific filename.
            prefix: Prefix for auto-generated filename.
            sheet_name: Name of the Excel sheet.
            
        Returns:
            Path to the exported file.
        """
        if filename is None:
            filename = self._generate_filename(prefix, "xlsx")
        
        try:
            df.to_excel(filename, index=False, sheet_name=sheet_name, engine='openpyxl')
            logger.info(f"Exported {len(df)} rows to {filename}")
        except ImportError:
            # Fallback to CSV if openpyxl not installed
            logger.warning("openpyxl not installed, falling back to CSV")
            filename = filename.replace('.xlsx', '.csv')
            df.to_csv(filename, index=False, encoding='utf-8')
        
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
        
        logger.info(f"Exported markdown to {filename}")
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
        """
        # Select key columns for recommendations
        rec_columns = [
            'name', 'profit', 'roi', 'flea_price', 'trader_price',
            'trader_name', 'category', 'last_offer_count',
            'rec_score', 'rec_tier', 'risk_level'
        ]
        
        # Only include columns that exist
        available_cols = [c for c in rec_columns if c in df.columns]
        export_df = df[available_cols].copy()
        
        # Format for readability
        if 'profit' in export_df.columns:
            export_df['profit'] = export_df['profit'].apply(lambda x: f"₽{x:,.0f}")
        if 'roi' in export_df.columns:
            export_df['roi'] = export_df['roi'].apply(lambda x: f"{x:.1f}%")
        if 'flea_price' in export_df.columns:
            export_df['flea_price'] = export_df['flea_price'].apply(lambda x: f"₽{x:,.0f}")
        if 'trader_price' in export_df.columns:
            export_df['trader_price'] = export_df['trader_price'].apply(lambda x: f"₽{x:,.0f}")
        
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
            raise ValueError(f"Unsupported format: {format}")
    
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
        
        logger.info(f"Exported market snapshot: {len(df)} items")
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
            days: Delete files older than this many days.
            
        Returns:
            Number of files deleted.
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0
        
        for filename in os.listdir(self.export_dir):
            filepath = os.path.join(self.export_dir, filename)
            if os.path.isfile(filepath):
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if mtime < cutoff:
                    try:
                        os.remove(filepath)
                        deleted += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {filepath}: {e}")
        
        if deleted:
            logger.info(f"Cleaned up {deleted} old export files")
        
        return deleted


# Create a default exporter instance
_exporter: Optional[DataExporter] = None


def get_exporter() -> DataExporter:
    """Get or create the data exporter singleton."""
    global _exporter
    if _exporter is None:
        _exporter = DataExporter()
    return _exporter
