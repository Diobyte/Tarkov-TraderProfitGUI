# 💰 Tarkov Trader Profit GUI

A powerful, ML-enhanced dashboard for **Escape from Tarkov** that identifies profitable trading flips between the Flea Market and Traders.

**Stop guessing. Start profiting.**

![Dashboard Demo](tarkov.gif)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![Version](https://img.shields.io/badge/version-3.0-green)

---

## 🚀 Quick Start (Windows)

### Prerequisites

- **Windows 10/11** (64-bit)
- **Internet connection** (for API data and initial setup)

That's it! The launcher will automatically install Python and all dependencies if needed.

### Step 1: Download the Project

**Option A: Clone with Git**

```bash
git clone https://github.com/Diobyte/Tarkov-TraderProfitGUI.git
cd Tarkov-TraderProfitGUI
```

**Option B: Download ZIP**

1. Click the green **"Code"** button above → **"Download ZIP"**
2. Extract the ZIP to any folder (e.g., `C:\TarkovProfit`)

### Step 2: Run the Application

Double-click **`run.bat`** — that's it!

The launcher will:

1. ✅ Check for Python (installs Python 3.12 via Winget if missing)
2. ✅ Create a virtual environment (`.venv`)
3. ✅ Install all dependencies automatically
4. ✅ Start the data collector in the background
5. ✅ Open the dashboard in your browser

> **First run takes 2-5 minutes** to download and install everything. Subsequent runs are instant.

---

## 📋 Usage Options

### Option 1: Dashboard + Collector (Recommended)

Double-click **`run.bat`**

- Launches the GUI dashboard at `http://localhost:8501`
- Automatically starts the data collector in the background
- **Note**: Closing the terminal window stops both the dashboard and collector

### Option 2: 24/7 Data Collection

Double-click **`run_collector.bat`**

- Runs _only_ the data collector in a terminal window
- Keeps collecting data 24/7, even when the dashboard is closed
- Use `run.bat` separately to view collected data

### Option 3: PowerShell Scripts

If you prefer PowerShell directly:

```powershell
# Full application (dashboard + collector)
.\run.ps1

# Collector only
.\run_collector.ps1
```

---

## 🌟 Features

### Core Features

- **Real-Time Data**: Fetches live market data from the [tarkov.dev API](https://tarkov.dev/api/) every 5 minutes
- **Smart Analysis**: Automatically calculates **Profit**, **ROI**, **Profit Per Slot**, and **Risk Scores**
- **Visual Dashboard**: Built with Streamlit for interactive sorting, filtering, and charting
- **Historical Trends**: Tracks price history to help you spot market dips and spikes
- **Background Collector**: Runs silently in the background to build a local database of price history
- **Flea Market Level Filtering**: Filter items by player level requirements (Patch 1.0+ support)

### 🤖 ML-Powered Analysis (v3.0)

- **Adaptive Opportunity Scoring**: Multi-factor ML scoring that weighs profit, ROI, liquidity, and price position
- **Anomaly Detection**: Isolation Forest identifies unusual pricing patterns that may indicate arbitrage opportunities
- **Smart Clustering**: K-Means clustering groups items into strategy tiers (Elite, High Value, Solid, Avoid)
- **Risk Assessment**: Comprehensive risk scoring based on volatility, liquidity, momentum, and margin
- **Trend Prediction**: Linear regression-based trend forecasting with confidence intervals
- **Similar Item Finder**: KNN-based similarity search to find alternative trading opportunities

### 📊 Enhanced Visualizations

- **Risk vs Reward Quadrant**: Visual strategy guide with annotated zones
- **Correlation Heatmaps**: Discover relationships between price factors
- **Market Intelligence Dashboard**: Comprehensive multi-panel analysis view
- **Price History Charts**: Track any item's price over time

### 📥 Data Export

- **CSV & JSON Export**: Export filtered data from any table view
- **Customizable Exports**: Export Top Trades, Market Explorer, or Category Summaries
- **Local Storage**: Exports saved to `/exports` directory

### 🔔 Alert System

- **High Profit Alerts**: Automatic notifications when items exceed profit thresholds
- **High ROI Alerts**: Track items with exceptional return on investment
- **Configurable Thresholds**: Customize via environment variables

---

## 🛠️ Manual Installation (Linux / macOS / Advanced)

If you prefer manual setup or are on Linux/macOS:

```bash
# 1. Clone the repository
git clone https://github.com/Diobyte/Tarkov-TraderProfitGUI.git
cd Tarkov-TraderProfitGUI

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start the data collector (in background or separate terminal)
python collector.py &

# 6. Start the Dashboard
streamlit run app.py
```

---

## 📊 Dashboard Guide

Once the app is running, your browser will open to `http://localhost:8501`.

### Sidebar Filters

- **Min Profit**: Filter out low-value items
- **Min ROI**: Ensure your investment yields a good percentage return
- **Player Level**: Filter by Flea Market level requirements (Patch 1.0+)
- **Category**: Focus on specific item types
- **Search**: Find specific items by name
- **Show Negative Profit**: Toggle to include/exclude losing trades
- **Quick Actions**: Refresh data or Start/Stop the collector

### Main Tabs

| Tab                     | Description                                    |
| ----------------------- | ---------------------------------------------- |
| 🏆 **Top Trades**       | Best profit opportunities sorted by profit     |
| 🌐 **Market Explorer**  | Browse ALL items with advanced filtering       |
| 🎨 **Visual Analytics** | Charts, heatmaps, and market visualizations    |
| 📈 **ML Insights**      | Machine learning analysis and risk scoring     |
| 🔍 **Item Details**     | Deep dive into specific item price history     |
| ⚙️ **System**           | Collector status, database stats, logs, alerts |

### The Collector

- Fetches data every **5 minutes**
- Stores data in user's Documents folder (see [Data Storage](#-data-storage))
- Automatically cleans up data older than **7 days**
- Start/stop directly from the **System** tab

---

## 📁 Data Storage

All generated data files are stored **outside the project folder** to keep the Git repository clean:

```
Documents/
└── TarkovTraderProfit/
    ├── tarkov_data.db          # SQLite database
    ├── ml_model_state.pkl      # ML model state
    ├── ml_learned_history.json # Learned patterns
    ├── collector.pid           # Process ID file
    ├── exports/                # Exported CSV/JSON files
    └── logs/                   # Application logs
        ├── app.log
        ├── collector.log
        └── collector_startup.log
```

To use a custom location, set the `TARKOV_DATA_DIR` environment variable:

```powershell
$env:TARKOV_DATA_DIR = "D:\TarkovData"
.\run.ps1
```

---

## 🧪 Running Tests

```bash
# Activate virtual environment first, then:
pytest tests/ -v
```

---

## ⚙️ Configuration

All settings can be customized via environment variables with the `TARKOV_` prefix:

| Variable                             | Default                          | Description                   |
| ------------------------------------ | -------------------------------- | ----------------------------- |
| `TARKOV_DATA_DIR`                    | `~/Documents/TarkovTraderProfit` | Data storage location         |
| `TARKOV_COLLECTION_INTERVAL_MINUTES` | `5`                              | Data fetch frequency          |
| `TARKOV_DATA_RETENTION_DAYS`         | `7`                              | Days to keep historical data  |
| `TARKOV_API_TIMEOUT_SECONDS`         | `30`                             | API request timeout           |
| `TARKOV_ALERT_HIGH_PROFIT_THRESHOLD` | `10000`                          | Profit alert trigger (₽)      |
| `TARKOV_ALERT_HIGH_ROI_THRESHOLD`    | `50.0`                           | ROI alert trigger (%)         |
| `TARKOV_ML_ANOMALY_CONTAMINATION`    | `0.05`                           | Anomaly detection sensitivity |

Example:

```powershell
$env:TARKOV_COLLECTION_INTERVAL_MINUTES = "10"
$env:TARKOV_DATA_RETENTION_DAYS = "14"
.\run.ps1
```

---

## 🔧 Troubleshooting

### "Python not found" Error

The launcher should install Python automatically. If it fails:

1. Install [Python 3.12](https://www.python.org/downloads/) manually
2. During installation, check **"Add Python to PATH"**
3. Re-run `run.bat`

### "Winget not found" Error

Winget comes with Windows 10 (1809+) and Windows 11. If missing:

1. Install from [Microsoft Store](https://apps.microsoft.com/store/detail/app-installer/9NBLGGH4NNS1)
2. Or install Python manually from [python.org](https://www.python.org/downloads/)

### Dashboard won't open in browser

- Check the terminal for the URL (usually `http://localhost:8501`)
- Try opening it manually in your browser
- Check if port 8501 is blocked by firewall

### "Database locked" errors

- Make sure only one collector instance is running
- Check the **System** tab to manage collector processes

---

## 🤝 Attribution

This project relies on the excellent [Tarkov.dev API](https://github.com/the-hideout/tarkov-api).

- **Data Source**: [tarkov.dev](https://tarkov.dev/)
- **API License**: GPL-3.0 (Note: This project uses the public API and does not bundle their source code)

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🔒 Security

Please review our security policy in [SECURITY.md](SECURITY.md).
If you find a vulnerability, submit a private advisory via GitHub Security Advisories or open an issue labeled `security`.

---

_Not affiliated with Battlestate Games._
