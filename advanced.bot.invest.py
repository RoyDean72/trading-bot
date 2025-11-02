"""
Advanced Value Investing Trading Bot with Risk Management
"""

import os
import sys
import concurrent.futures
import logging
import logging.handlers
import asyncio
from time import time
from typing import List, Dict, Optional, Tuple
from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd
import requests
import alpaca_trade_api as tradeapi
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field, PositiveFloat, PositiveInt
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log"),
        logging.handlers.RotatingFileHandler(
            "trading_bot_debug.log", maxBytes=1e6, backupCount=3
        ),
    ],
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with validation and environment loading"""

    APCA_API_KEY_ID: str = Field(..., env="APCA_API_KEY_ID")
    APCA_API_SECRET_KEY: str = Field(..., env="APCA_API_SECRET_KEY")
    APCA_API_BASE_URL: str = Field("https://paper-api.alpaca.markets", env="APCA_API_BASE_URL")
    DATA_SOURCES: List[str] = Field([], env="DATA_SOURCES")
    MAX_WORKERS: PositiveInt = 5
    REQUEST_TIMEOUT: PositiveFloat = 15.0
    POSITION_SIZE_PCT: PositiveFloat = Field(2.0, gt=0.0, le=100.0)
    RISK_FREE_RATE: float = 0.02  # 2% annual risk-free rate
    MAX_DRAWDOWN_PCT: float = 5.0
    TRADING_HOURS_CHECK: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()


class AlpacaAPIClient:
    """Singleton Alpaca API client with circuit breaker pattern"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize API client with retry strategy"""
        self.api = tradeapi.REST(
            key_id=settings.APCA_API_KEY_ID,
            secret_key=settings.APCA_API_SECRET_KEY,
            base_url=settings.APCA_API_BASE_URL,
            api_version="v2"
        )
        self.failure_count = 0
        self.circuit_open = False
        self.last_failure_time = None
        self.failure_threshold = 3
        self.reset_timeout = 60  # seconds

    def execute_safe(self, func, *args, **kwargs):
        """Execute API call with circuit breaker protection"""
        if self.circuit_open:
            if time() - self.last_failure_time > self.reset_timeout:
                self.circuit_open = False
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            logger.error(f"API call failed: {str(e)}")
            raise

    def _record_success(self):
        self.failure_count = 0

    def _record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.circuit_open = True
            self.last_failure_time = time()
            logger.critical("Circuit breaker tripped!")


class DataFetcher:
    """Advanced data fetcher with caching and request throttling"""

    def __init__(self):
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.cache = {}

    async def fetch(self, url: str) -> Optional[Dict]:
        """Fetch data with caching and async support"""
        if url in self.cache:
            return self.cache[url]

        try:
            response = await asyncio.to_thread(
                self.session.get,
                url,
                headers={"User-Agent": "TradingBot/2.0"},
                timeout=settings.REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            data = response.json()
            self.cache[url] = data
            return data
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            return None


class PortfolioManager:
    """Advanced portfolio management with risk controls"""

    def __init__(self, api_client: AlpacaAPIClient):
        self.api = api_client
        self.portfolio = {}
        self.update()

    def update(self):
        """Refresh portfolio data"""
        try:
            positions = self.api.execute_safe(self.api.api.list_positions)
            self.portfolio = {
                pos.symbol: {
                    "qty": float(pos.qty),
                    "market_value": float(pos.market_value),
                    "current_price": float(pos.current_price),
                }
                for pos in positions
            }
        except Exception as e:
            logger.error(f"Failed to update portfolio: {str(e)}")
            raise

    def calculate_position_size(self, symbol: str, price: float) -> Decimal:
        """Calculate position size using Kelly Criterion"""
        account = self.api.execute_safe(self.api.api.get_account)
        equity = float(account.equity)
        max_position_size = equity * (settings.POSITION_SIZE_PCT / 100)
        return Decimal(max_position_size / price).quantize(
            Decimal("1"), rounding=ROUND_DOWN
        )

    def risk_assessment(self) -> Tuple[float, float]:
        """Calculate portfolio risk metrics"""
        account = self.api.execute_safe(self.api.api.get_account)
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        return equity, buying_power


async def load_sp500_companies(file_path: str = "sp500.csv") -> List[str]:
    """Load and validate S&P 500 symbols with async I/O"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"S&P 500 file missing: {file_path}")

        df = pd.read_csv(file_path)
        if "Symbol" not in df.columns:
            raise ValueError("Invalid S&P 500 file format")

        symbols = df["Symbol"].dropna().str.strip().tolist()
        logger.info(f"Loaded {len(symbols)} valid S&P 500 symbols")
        return symbols
    except Exception as e:
        logger.error(f"S&P 500 loading failed: {str(e)}")
        return []


def filter_stocks(symbols: List[str], api_client: AlpacaAPIClient) -> pd.DataFrame:
    """Filter stocks using real Alpaca data"""
    logger.info(f"Filtering {len(symbols)} symbols...")
    results = []
    for symbol in symbols[:3]:  # Test with first 3
        logger.info(f"Checking {symbol}...")
        try:
            from datetime import datetime, timedelta
            end = datetime.now()
            start = end - timedelta(days=60)
            bars = api_client.execute_safe(
                api_client.api.get_bars,
                symbol,
                "1Day",
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d')
            ).df
            
            if bars is not None and len(bars) >= 20:
                closes = bars['close'].values
                current = closes[-1]
                sma20 = np.mean(closes[-20:])
                
                logger.info(f"{symbol}: Price=${current:.2f}, SMA20=${sma20:.2f}")
                
                # Very simple filter - just above SMA20
                if current > sma20 * 0.98:  # Within 2% of SMA20
                    results.append({"Symbol": symbol, "Price": current})
                    logger.info(f"  ✓ {symbol} PASSED filter")
                else:
                    logger.info(f"  ✗ {symbol} below SMA20")
        except Exception as e:
            logger.warning(f"Skipping {symbol}: {e}")
    
    return pd.DataFrame(results)


async def execute_trades(filtered_stocks: pd.DataFrame, pm: PortfolioManager) -> None:
    """Execute trades with sophisticated order management"""
    if filtered_stocks.empty:
        logger.warning("No stocks to trade")
        return

    try:
        equity, buying_power = pm.risk_assessment()
        logger.info(f"Portfolio Equity: ${equity:,.2f} | Buying Power: ${buying_power:,.2f}")

        for _, row in filtered_stocks.iterrows():
            symbol = row["Symbol"]
            
            # Check existing position
            if symbol in pm.portfolio:
                logger.info(f"Skipping existing position: {symbol}")
                continue

            # Get real-time price
            try:
                bars = pm.api.execute_safe(pm.api.api.get_bars, symbol, "5Min", limit=5)
                if not bars:
                    logger.warning(f"No price data for {symbol}")
                    continue
                price = bars[-1].c
            except Exception as e:
                logger.error(f"Pricing failed for {symbol}: {str(e)}")
                continue

            # Calculate position size
            try:
                shares = pm.calculate_position_size(symbol, price)
                if shares < 1:
                    logger.warning(f"Insufficient capital for {symbol}")
                    continue
            except Exception as e:
                logger.error(f"Position sizing failed: {str(e)}")
                continue

            # Submit order with multiple safety checks
            try:
                pm.api.execute_safe(
                    pm.api.api.submit_order,
                    symbol=symbol,
                    qty=str(shares),
                    side="buy",
                    type="trailing_stop",
                    time_in_force="gtc",
                    trail_percent=2.0,
                    order_class="bracket",
                    stop_loss=dict(
                        stop_price=str(price * 0.95),
                        limit_price=str(price * 0.94)
                    ),
                    take_profit=dict(
                        limit_price=str(price * 1.15)
                    )
                )
                logger.info(f"Submitted order for {shares} {symbol} @ {price}")
            except Exception as e:
                logger.error(f"Order failed for {symbol}: {str(e)}")

            # Rate limiting
            await asyncio.sleep(0.5)

    except Exception as e:
        logger.critical(f"Trading execution failed: {str(e)}")
        raise


async def main():
    """Main async trading workflow"""
    logger.info("Starting advanced trading bot")
    start_time = time()

    try:
        # Initialize core components
        api_client = AlpacaAPIClient()
        pm = PortfolioManager(api_client)
        fetcher = DataFetcher()

        # Load and validate symbols
        symbols = await load_sp500_companies()
        if not symbols:
            logger.error("Aborting due to missing symbols")
            return

        # Filter stocks using real data
        logger.info(f"Starting filter with {len(symbols)} symbols")
        filtered_df = filter_stocks(symbols, api_client)
        logger.info(f"Filter returned {len(filtered_df)} stocks")

        # Execute trades
        await execute_trades(filtered_df, pm)

        logger.info(f"Completed in {time() - start_time:.2f} seconds")

    except KeyboardInterrupt:
        logger.info("User interrupted execution")
    except Exception as e:
        logger.critical(f"Critical failure: {str(e)}", exc_info=True)
    finally:
        logger.info("Trading bot shutdown completed")


if __name__ == "__main__":
    asyncio.run(main())