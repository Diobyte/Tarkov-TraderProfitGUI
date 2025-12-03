# 💰 Tarkov Trader Profit GUI

A powerful, ML-enhanced dashboard for **Escape from Tarkov** that identifies profitable trading flips between the Flea Market and Traders.

**Stop guessing. Start profiting.**

![Dashboard Demo](tarkov.gif)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![Version](https://img.shields.io/badge/version-3.0.0-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)

---

## 📑 Table of Contents

- [Quick Start (Windows)](#-quick-start-windows)
- [Usage Options](#-usage-options)
- [Features](#-features)
- [Manual Installation](#️-manual-installation-linux--macos--advanced)
- [Dashboard Guide](#-dashboard-guide)
- [Data Storage](#-data-storage)
- [Running Tests](#-running-tests)
- [Docker / Self-Hosting](#-docker--self-hosting)
- [GraphQL API](#-graphql-api)
- [Configuration](#️-configuration)
- [Troubleshooting](#-troubleshooting)
- [Attribution](#-attribution)

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

1. ✅ Check for Python (installs Python 3.13 via Winget if missing)
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

- **CSV, JSON & Excel Export**: Export filtered data from any table view
- **Customizable Exports**: Export Top Trades, Market Explorer, or Category Summaries
- **Local Storage**: Exports saved to `Documents/TarkovTraderProfit/exports/`

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

> **Note**: The GraphQL API dependencies (FastAPI, Strawberry, uvicorn) are included in `requirements.txt` but are only used for Docker deployments. The core dashboard works without them.

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

- Fetches data every **5 minutes** (configurable)
- Stores data in user's Documents folder (see [Data Storage](#-data-storage))
- Automatically cleans up data older than **7 days** (configurable)
- Start/stop directly from the **System** tab

---

## 📁 Data Storage

All generated data files are stored **outside the project folder** to keep the Git repository clean:

```
Documents/
└── TarkovTraderProfit/
    ├── tarkov_data.db              # SQLite database (WAL mode)
    ├── ml_model_state.pkl          # ML model state (scikit-learn)
    ├── ml_learned_history.json     # Learned patterns history
    ├── collector.pid               # Process ID file
    ├── collector_standalone.pid    # Standalone collector PID
    ├── exports/                    # Exported CSV/JSON/Excel files
    └── logs/                       # Application logs
        ├── app.log
        ├── collector.log
        ├── collector_startup.log
        └── api.log                 # GraphQL API logs (Docker only)
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

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

The test suite includes 124+ tests covering database operations, calculations, ML engine, alerts, exports, and more.

---

## 🐳 Docker / Self-Hosting

Run the entire application in Docker for easy deployment on your home server or local network.

### Quick Start (Pre-built Images)

The fastest way to get started using pre-built images from GitHub Container Registry:

```bash
# Pull and run (single container with all services)
docker run -d \
  --name tarkov-profit \
  -p 8501:8501 \
  -p 4000:4000 \
  -v tarkov-data:/data \
  -e TZ=America/New_York \
  ghcr.io/diobyte/tarkov-traderprofitgui:latest
```

**Access:**

- **Dashboard**: `http://localhost:8501`
- **GraphQL Playground**: `http://localhost:4000/graphql`
- **API Health Check**: `http://localhost:4000/health`

### Build Locally

If you prefer to build the image yourself:

```bash
# Build the image
docker build -t tarkov-profit .

# Run with persistent data volume
docker run -d \
  --name tarkov-profit \
  -p 8501:8501 \
  -p 4000:4000 \
  -v tarkov-data:/data \
  -e PUID=$(id -u) \
  -e PGID=$(id -g) \
  -e TZ=America/New_York \
  tarkov-profit
```

### Docker Compose (Recommended for Production)

For production deployments, use Docker Compose with separate containers for each service. This provides better isolation, independent scaling, and easier debugging.

```bash
# Start all services (uses pre-built images from GHCR)
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

The included `docker-compose.yml` provides:

- **3 separate services**: Collector, Dashboard, and GraphQL API
- **Pre-built images** from GitHub Container Registry
- **Health checks** for all services
- **Log rotation** (10MB max, 3 files)
- **Shared data volume** for SQLite database
- **LinuxServer.io-style** PUID/PGID/TZ configuration

#### docker-compose.yml Overview

```yaml
# Uses pre-built images from ghcr.io/diobyte/tarkov-traderprofitgui:latest
services:
  collector: # Background data collection (no ports exposed)
  dashboard: # Streamlit UI on port 8501
  api: # GraphQL API on port 4000

volumes:
  tarkov-data: # Persistent SQLite database

networks:
  tarkov-net: # Isolated bridge network
```

#### Environment Variables (Docker)

| Variable                             | Default   | Description                         |
| ------------------------------------ | --------- | ----------------------------------- |
| `PUID`                               | `1000`    | User ID for file permissions        |
| `PGID`                               | `1000`    | Group ID for file permissions       |
| `TZ`                                 | `Etc/UTC` | Timezone (e.g., `America/New_York`) |
| `TARKOV_DATA_DIR`                    | `/data`   | Data directory inside container     |
| `TARKOV_COLLECTION_INTERVAL_MINUTES` | `5`       | API fetch interval                  |
| `TARKOV_DATA_RETENTION_DAYS`         | `7`       | Days to keep historical data        |
| `TARKOV_LOG_LEVEL`                   | `INFO`    | Log verbosity (DEBUG/INFO/WARNING)  |
| `API_HOST`                           | `0.0.0.0` | GraphQL API bind address            |
| `API_PORT`                           | `4000`    | GraphQL API port                    |

### LAN Access

To access from other devices on your network, use your host's IP address:

```bash
# Find your IP
# Windows: ipconfig
# Linux/macOS: ip addr or hostname -I

# Access from other devices:
# Dashboard: http://192.168.1.100:8501
# GraphQL:   http://192.168.1.100:4000/graphql
```

### Docker Commands Reference

```bash
# Start all services
docker compose up -d

# View all logs (follow mode)
docker compose logs -f

# View specific service logs
docker logs tarkov-collector -f
docker logs tarkov-dashboard -f
docker logs tarkov-api -f

# Restart a specific service
docker compose restart dashboard

# Stop all services
docker compose down

# Stop and remove volumes (deletes data!)
docker compose down -v

# Rebuild after code changes (local build only)
docker compose build --no-cache
docker compose up -d

# Check service health
docker compose ps

# Execute command in container
docker exec -it tarkov-dashboard /bin/bash
```

---

## 🔌 GraphQL API

The GraphQL API provides programmatic access to all market data. It's automatically included in Docker deployments.

### Endpoints

| Endpoint   | Method | Description                         |
| ---------- | ------ | ----------------------------------- |
| `/graphql` | GET    | GraphQL Playground (interactive UI) |
| `/graphql` | POST   | GraphQL queries                     |
| `/health`  | GET    | Health check                        |
| `/stats`   | GET    | Quick statistics                    |
| `/docs`    | GET    | OpenAPI documentation               |

### Example Queries

**Get Top 10 Profitable Items:**

```graphql
query {
  profitableItems(minProfit: 5000, limit: 10) {
    name
    profit
    roi
    fleaPrice
    traderPrice
    traderName
    category
  }
}
```

**Search for Items:**

```graphql
query {
  search(query: "GPU", limit: 5) {
    name
    profit
    fleaPrice
    traderPrice
    profitPerSlot
  }
}
```

**Get All Items with Filters:**

```graphql
query {
  items(
    minProfit: 1000
    minRoi: 10.0
    category: "Barter"
    sortBy: "profit"
    sortDesc: true
    limit: 50
  ) {
    itemId
    name
    profit
    roi
    fleaPrice
    traderPrice
    traderName
  }
}
```

**Get Category Statistics:**

```graphql
query {
  categories {
    category
    itemCount
    avgProfit
    totalProfit
    profitableCount
  }
}
```

**Get Trader Statistics:**

```graphql
query {
  traders {
    traderName
    itemCount
    avgProfit
    totalProfit
  }
}
```

**Get Market Trends:**

```graphql
query {
  trends(hours: 24, limit: 20) {
    itemId
    avgProfit
    minProfit
    maxProfit
    dataPoints
    volatility
  }
}
```

**Check Database Health:**

```graphql
query {
  health {
    status
    totalRecords
    uniqueItems
    dataAgeHours
    fileSizeMb
  }
}
```

### Using with cURL

```bash
# Get profitable items
curl -X POST http://localhost:4000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ profitableItems(limit: 5) { name profit roi } }"}'

# Health check
curl http://localhost:4000/health

# Quick stats
curl http://localhost:4000/stats
```

### Using with Python

```python
import requests

GRAPHQL_URL = "http://localhost:4000/graphql"

query = """
query GetProfitableItems($minProfit: Int!, $limit: Int!) {
  profitableItems(minProfit: $minProfit, limit: $limit) {
    name
    profit
    roi
    fleaPrice
    traderPrice
    traderName
  }
}
"""

response = requests.post(
    GRAPHQL_URL,
    json={
        "query": query,
        "variables": {"minProfit": 5000, "limit": 10}
    }
)

data = response.json()
for item in data["data"]["profitableItems"]:
    print(f"{item['name']}: {item['profit']:,}₽ ({item['roi']:.1f}% ROI)")
```

### Using with JavaScript

```javascript
const GRAPHQL_URL = "http://localhost:4000/graphql";

async function getProfitableItems() {
  const response = await fetch(GRAPHQL_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: `{
        profitableItems(minProfit: 5000, limit: 10) {
          name
          profit
          roi
          fleaPrice
          traderName
        }
      }`,
    }),
  });

  const { data } = await response.json();
  return data.profitableItems;
}

// Usage
getProfitableItems().then((items) => {
  items.forEach((item) => {
    console.log(`${item.name}: ${item.profit.toLocaleString()}₽`);
  });
});
```

---

## ⚙️ Configuration

All settings can be customized via environment variables with the `TARKOV_` prefix.

### Core Settings

| Variable                             | Default                          | Description                          |
| ------------------------------------ | -------------------------------- | ------------------------------------ |
| `TARKOV_DATA_DIR`                    | `~/Documents/TarkovTraderProfit` | Data storage location                |
| `TARKOV_COLLECTION_INTERVAL_MINUTES` | `5`                              | Data fetch frequency (minutes)       |
| `TARKOV_DATA_RETENTION_DAYS`         | `7`                              | Days to keep historical data         |
| `TARKOV_API_TIMEOUT_SECONDS`         | `30`                             | API request timeout                  |
| `TARKOV_LOG_LEVEL`                   | `INFO`                           | Log level (DEBUG/INFO/WARNING/ERROR) |

### Alert Settings

| Variable                                | Default | Description              |
| --------------------------------------- | ------- | ------------------------ |
| `TARKOV_ALERT_HIGH_PROFIT_THRESHOLD`    | `10000` | Profit alert trigger (₽) |
| `TARKOV_ALERT_HIGH_ROI_THRESHOLD`       | `50.0`  | ROI alert trigger (%)    |
| `TARKOV_ALERT_DEFAULT_COOLDOWN_MINUTES` | `30`    | Alert cooldown period    |
| `TARKOV_ALERT_MAX_HISTORY`              | `500`   | Max alerts to keep       |

### ML Engine Settings

| Variable                           | Default | Description                           |
| ---------------------------------- | ------- | ------------------------------------- |
| `TARKOV_ML_ANOMALY_CONTAMINATION`  | `0.05`  | Anomaly detection sensitivity (0-0.5) |
| `TARKOV_ML_ESTIMATORS`             | `100`   | Number of estimators for ML models    |
| `TARKOV_ML_MIN_ITEMS_FOR_ANALYSIS` | `10`    | Minimum items for ML analysis         |
| `TARKOV_ML_MIN_ITEMS_FOR_ANOMALY`  | `20`    | Minimum items for anomaly detection   |

### Trend Analysis Settings

| Variable                              | Default | Description                        |
| ------------------------------------- | ------- | ---------------------------------- |
| `TARKOV_TREND_LOOKBACK_HOURS`         | `24`    | Hours of data for trend analysis   |
| `TARKOV_TREND_MIN_DATA_POINTS`        | `6`     | Minimum data points for trends     |
| `TARKOV_TREND_PROFIT_MOMENTUM_WEIGHT` | `0.20`  | Weight for profit momentum scoring |
| `TARKOV_TREND_VOLATILITY_PENALTY`     | `0.15`  | Penalty for high volatility        |
| `TARKOV_TREND_CONSISTENCY_BONUS`      | `0.25`  | Bonus for consistent profits       |

### Volume/Liquidity Settings

| Variable                               | Default | Description                    |
| -------------------------------------- | ------- | ------------------------------ |
| `TARKOV_VOLUME_MIN_FOR_RECOMMENDATION` | `5`     | Min offers for recommendations |
| `TARKOV_VOLUME_LOW_THRESHOLD`          | `10`    | Low volume threshold           |
| `TARKOV_VOLUME_MEDIUM_THRESHOLD`       | `50`    | Medium volume threshold        |
| `TARKOV_VOLUME_HIGH_THRESHOLD`         | `100`   | High volume threshold          |
| `TARKOV_VOLUME_VERY_HIGH_THRESHOLD`    | `200`   | Very high volume threshold     |
| `TARKOV_VOLUME_WEIGHT_IN_SCORE`        | `0.15`  | Volume weight in scoring       |

### Database Settings

| Variable                             | Default | Description                  |
| ------------------------------------ | ------- | ---------------------------- |
| `TARKOV_DATABASE_CONNECTION_TIMEOUT` | `30`    | Connection timeout (seconds) |
| `TARKOV_DATABASE_RETRY_ATTEMPTS`     | `5`     | Retry attempts on lock       |
| `TARKOV_DATABASE_RETRY_DELAY`        | `1.0`   | Delay between retries (sec)  |
| `TARKOV_DATABASE_BUSY_TIMEOUT_MS`    | `30000` | SQLite busy timeout (ms)     |

### UI Settings

| Variable                             | Default | Description                |
| ------------------------------------ | ------- | -------------------------- |
| `TARKOV_UI_REFRESH_INTERVAL_SECONDS` | `60`    | Auto-refresh interval      |
| `TARKOV_UI_MAX_TABLE_ROWS`           | `100`   | Max rows in tables         |
| `TARKOV_UI_CHART_HEIGHT`             | `400`   | Default chart height (px)  |
| `TARKOV_STREAMLIT_CACHE_TTL_SECONDS` | `60`    | Cache TTL for data queries |

### Export Settings

| Variable                     | Default | Description              |
| ---------------------------- | ------- | ------------------------ |
| `TARKOV_EXPORT_MAX_ROWS`     | `1000`  | Max rows per export      |
| `TARKOV_EXPORT_CLEANUP_DAYS` | `7`     | Days to keep old exports |

### Example Configuration

**PowerShell:**

```powershell
$env:TARKOV_COLLECTION_INTERVAL_MINUTES = "10"
$env:TARKOV_DATA_RETENTION_DAYS = "14"
$env:TARKOV_ML_ANOMALY_CONTAMINATION = "0.03"
$env:TARKOV_ALERT_HIGH_PROFIT_THRESHOLD = "15000"
.\run.ps1
```

**Linux/macOS:**

```bash
export TARKOV_COLLECTION_INTERVAL_MINUTES=10
export TARKOV_DATA_RETENTION_DAYS=14
export TARKOV_LOG_LEVEL=DEBUG
python collector.py &
streamlit run app.py
```

---

## 🔧 Troubleshooting

### "Python not found" Error

The launcher should install Python automatically. If it fails:

1. Install [Python 3.13](https://www.python.org/downloads/) manually
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
- The app uses SQLite WAL mode for better concurrency

### API shows import errors (local development)

The GraphQL API dependencies (FastAPI, Strawberry, uvicorn) are optional for local use. They're only required for Docker deployments. The core dashboard works without them.

### Docker container won't start

- Check logs: `docker logs tarkov-dashboard`
- Verify port availability: `netstat -an | findstr 8501`
- Check disk space for the data volume

### Data not updating

- Verify the collector is running (check **System** tab)
- Check `Documents/TarkovTraderProfit/logs/collector.log` for errors
- Ensure internet connectivity for API access

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
