# Test package for Tarkov Trader Profit GUI
"""
Test suite for Tarkov Trader Profit application.

Test modules:
- test_calculations.py: Unit tests for utility functions and metrics calculation
- test_database.py: Database operations, schema, and data persistence tests
- test_alerts.py: Alert system functionality and persistence tests
- test_ml_engine.py: ML engine feature engineering, scoring, and predictions
- test_model_persistence.py: Model state persistence and learning statistics
- test_exporter.py: Data export functionality in various formats
- test_cleanup_and_trends.py: Data cleanup and trend calculation tests

Run all tests:
    pytest tests/

Run with coverage:
    pytest tests/ --cov=. --cov-report=term-missing

Run specific test file:
    pytest tests/test_database.py -v
"""
