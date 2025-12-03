"""
Tests for the Data Exporter module.
"""

import pytest
import os
import json
import tempfile
import pandas as pd
from exporter import DataExporter, ExportFormat, get_exporter


class TestDataExporter:
    """Tests for the DataExporter class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'name': ['Item A', 'Item B', 'Item C'],
            'profit': [5000, 10000, -2000],
            'roi': [25.0, 50.0, -10.0],
            'flea_price': [20000, 20000, 20000],
            'trader_price': [25000, 30000, 18000],
            'trader_name': ['Therapist', 'Mechanic', 'Ragman'],
            'category': ['Meds', 'Weapons', 'Gear'],
            'last_offer_count': [50, 100, 25],
        })
    
    @pytest.fixture
    def exporter(self, tmp_path):
        """Create an exporter with temporary directory."""
        return DataExporter(export_dir=str(tmp_path))
    
    def test_export_to_csv(self, exporter, sample_df):
        """Test CSV export."""
        filepath = exporter.export_to_csv(sample_df, prefix="test")
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.csv')
        
        # Verify content
        loaded = pd.read_csv(filepath)
        assert len(loaded) == 3
        assert 'name' in loaded.columns
    
    def test_export_to_json(self, exporter, sample_df):
        """Test JSON export with DataFrame."""
        filepath = exporter.export_to_json(sample_df, prefix="test")
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.json')
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        assert len(data) == 3
        assert data[0]['name'] == 'Item A'
    
    def test_export_dict_to_json(self, exporter):
        """Test JSON export with dictionary."""
        data = {'key': 'value', 'number': 42, 'list': [1, 2, 3]}
        filepath = exporter.export_to_json(data, prefix="dict_test")
        
        assert os.path.exists(filepath)
        
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        assert loaded['key'] == 'value'
        assert loaded['number'] == 42
    
    def test_export_to_markdown(self, exporter, sample_df):
        """Test Markdown export."""
        filepath = exporter.export_to_markdown(sample_df, prefix="test", title="Test Export")
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.md')
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        assert '# Test Export' in content
        assert 'Item A' in content
        assert '| name |' in content
    
    def test_export_recommendations(self, exporter, sample_df):
        """Test recommendations export."""
        # Add recommendation columns
        sample_df['rec_score'] = [80, 90, 30]
        sample_df['rec_tier'] = ['Great', 'Excellent', 'Consider']
        sample_df['risk_level'] = ['Low', 'Low', 'High']
        
        filepath = exporter.export_recommendations(sample_df, format=ExportFormat.CSV)
        
        assert os.path.exists(filepath)
        assert 'recommendations' in filepath
    
    def test_export_recommendations_json(self, exporter, sample_df):
        """Test recommendations export to JSON."""
        filepath = exporter.export_recommendations(sample_df, format=ExportFormat.JSON)
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.json')
    
    def test_export_recommendations_markdown(self, exporter, sample_df):
        """Test recommendations export to Markdown."""
        filepath = exporter.export_recommendations(sample_df, format=ExportFormat.MARKDOWN)
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.md')
    
    def test_export_market_snapshot(self, exporter, sample_df):
        """Test full market snapshot export."""
        results = exporter.export_market_snapshot(sample_df)
        
        assert 'csv' in results
        assert 'json' in results
        assert 'summary' in results
        
        assert os.path.exists(results['csv'])
        assert os.path.exists(results['json'])
        assert os.path.exists(results['summary'])
    
    def test_get_csv_string(self, exporter, sample_df):
        """Test getting CSV as string."""
        csv_str = exporter.get_csv_string(sample_df)
        
        assert isinstance(csv_str, str)
        assert 'name' in csv_str
        assert 'Item A' in csv_str
    
    def test_get_json_string(self, exporter, sample_df):
        """Test getting JSON as string."""
        json_str = exporter.get_json_string(sample_df)
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert len(data) == 3
    
    def test_cleanup_old_exports(self, exporter, sample_df):
        """Test cleaning up old exports."""
        import time
        
        # Create some files
        file1 = exporter.export_to_csv(sample_df, prefix="old1")
        file2 = exporter.export_to_json(sample_df, prefix="old2")
        
        # Files should exist
        assert os.path.exists(file1)
        assert os.path.exists(file2)
        
        # Set file modification time to be old (1 second ago is enough since we use days=0)
        old_time = time.time() - 1
        os.utime(file1, (old_time, old_time))
        os.utime(file2, (old_time, old_time))
        
        # Cleanup (0 days = delete all files modified before now)
        deleted = exporter.cleanup_old_exports(days=0)
        assert deleted >= 2
        
        # Check files are gone
        assert not os.path.exists(file1)
        assert not os.path.exists(file2)
    
    def test_invalid_format_raises(self, exporter, sample_df):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError):
            exporter.export_recommendations(sample_df, format="invalid")


class TestExporterSingleton:
    """Test the singleton exporter."""
    
    def test_get_exporter(self):
        """Test getting the singleton exporter."""
        exporter = get_exporter()
        assert isinstance(exporter, DataExporter)
    
    def test_get_exporter_same_instance(self):
        """Test that get_exporter returns same instance."""
        exporter1 = get_exporter()
        exporter2 = get_exporter()
        assert exporter1 is exporter2


class TestExportFormat:
    """Test export format constants."""
    
    def test_formats_defined(self):
        """Test all formats are defined."""
        assert ExportFormat.CSV == "csv"
        assert ExportFormat.JSON == "json"
        assert ExportFormat.EXCEL == "xlsx"
        assert ExportFormat.MARKDOWN == "md"
