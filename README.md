# 💰 Tarkov Trader Profit GUI

A powerful, automated dashboard for **Escape from Tarkov** that identifies profitable trading flips between the Flea Market and Traders.

**Stop guessing. Start profiting.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

## 🌟 Features

- **Real-Time Data**: Fetches live market data from the [tarkov.dev API](https://tarkov.dev/api/) every 5 minutes.
- **Smart Analysis**: Automatically calculates **Profit**, **ROI**, and **Profit Per Slot**.
- **Machine Learning**: Uses K-Means clustering to categorize items into **"Stable"** (Safe) vs **"Volatile"** (Risky) investments.
- **Visual Dashboard**: Built with Streamlit for interactive sorting, filtering, and charting.
- **Historical Trends**: Tracks price history to help you spot market dips and spikes.
- **Background Collector**: Runs silently in the background to build a local database of price history.

## 🚀 Quick Start (Windows)

### Option 1: Dashboard + Collector (Recommended)

Double-click **`run.bat`**.

- Launches the GUI dashboard.
- Automatically starts the data collector in the background.
- **Note**: Closing the dashboard window will stop the data collector.

### Option 2: 24/7 Data Collection (Advanced)

Double-click **`run_collector.bat`**.

- Runs _only_ the data collector in a terminal window.
- Keeps collecting data even if you close the dashboard.
- You can still open the dashboard using `run.bat` to view the data.

## 🛠️ Manual Installation (Linux / macOS / Advanced)

If you prefer the command line:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the Dashboard
streamlit run app.py
```

_Note: The dashboard allows you to start/stop the background data collector directly from the sidebar. Alternatively, you can run `python collector.py` in a separate terminal._

## 📊 Usage Guide

### The Dashboard

Once the app is running, your browser will open to `http://localhost:8501`.

- **Filters (Sidebar)**:
  - **Min Profit**: Filter out low-value items.
  - **Min ROI**: Ensure your investment yields a good percentage return.
  - **Trend Window**: Adjust how far back (in hours) the ML model looks for volatility analysis.
- **Tabs**:
  - **Market Table**: The main view. Sort by Profit, ROI, or Discount.
  - **Visual Analysis**: Scatter plots showing Risk vs. Reward.
  - **Item History**: Deep dive into a specific item's price over time.
  - **Data Review**: View raw data, check all metrics, and export to CSV.
  - **Console**: View system logs for debugging.

### The Collector

The `collector.py` script runs in the background.

- It fetches data every **5 minutes**.
- It stores data in `tarkov_data.db` (SQLite).
- It automatically cleans up data older than **7 days** to keep the database small.
- You can start/stop it directly from the Dashboard sidebar.

## 🤝 Attribution

This project relies on the excellent [Tarkov.dev API](https://github.com/the-hideout/tarkov-api).

- **Data Source**: [tarkov.dev](https://tarkov.dev/)
- **API License**: GPL-3.0 (Note: This project uses the public API and does not bundle their source code).

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

_Not affiliated with Battlestate Games._

## 🔒 Security

Please review our security policy in [SECURITY.md](SECURITY.md).
If you find a vulnerability, submit a private advisory via GitHub Security
Advisories for this repository or open an issue labeled `security` without
public exploit details.
