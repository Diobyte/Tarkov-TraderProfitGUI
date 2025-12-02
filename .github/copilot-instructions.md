# GitHub Copilot Instructions for Tarkov-TraderProfitGUI

This repository contains a Python-based tool for analyzing Escape from Tarkov market data. It consists of a background data collector and a Streamlit dashboard.

## Project Structure

- **`collector.py`**: The "Producer". Fetches data from the GraphQL API and writes to SQLite. Runs as a background process.
- **`app.py`**: The "Consumer". A Streamlit web application that visualizes the data.
- **`database.py`**: Shared database logic. Handles SQLite connections, retries, and schema management.
- **`utils.py`**: Shared business logic and calculations.

## Coding Standards

- **Python Version**: 3.10+ (Target 3.12).
- **Type Hinting**: MUST use type hints (`typing.List`, `typing.Optional`, etc.) for all function arguments and return values.
- **Error Handling**: Use `try/except` blocks generously, especially around database operations and API calls. Log errors using the `logging` module, not `print`.
- **Database**: Always use the `@retry_db_op` decorator from `database.py` for any DB interaction to handle SQLite locking (WAL mode is enabled).

## Key Architectural Decisions

- **Producer-Consumer**: The collector and app are separate processes communicating solely through the SQLite database.
- **WAL Mode**: SQLite Write-Ahead Logging is enabled to allow concurrent reads/writes.
- **Streamlit Fragments**: Use `@st.fragment` for components that need frequent updates (like the status bar or logs) to avoid re-rendering the entire page.
- **Plotly**: Use `plotly.express` and `plotly.graph_objects` for all visualizations. Use the `plotly_dark` template.

## Testing

- Run tests using `pytest`.
- Tests are located in `tests/`.
- Mock external dependencies (like the API or Database) when writing unit tests.

## UI/UX Guidelines

- **Theme**: Dark mode is enforced via `.streamlit/config.toml`.
- **Colors**:
  - Profit/Good: `#4CAF50` (Green)
  - Cost/Bad: `#FF5252` (Red)
  - Background: `#0E1117`
- **Feedback**: Use `st.toast` for transient messages and `st.success`/`st.error` for persistent status updates.
