import json
import os
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class DatabaseEngine:
    """
    🗄️ Database 資料庫中心 (SQLite)

    避免每次都重新呼叫 yfinance 下載歷史資料：
      - 第一次查詢某檔股票時，下載完整歷史並寫入 SQLite
      - 之後在「新鮮期限」內再次查詢，直接從本地資料庫讀取，速度更快、也減少對 yfinance 的請求量
      - 過了新鮮期限（預設 6 小時）才會重新向 yfinance 抓取最新資料並覆寫快取

    資料庫檔案預設位置：<專案根目錄>/data/tqai.db

    ⚠️ 修正說明（v2.6）：
    新增 is_intraday_estimate 欄位（配合 DataEngine 的即時股價覆蓋修正）。
    這個欄位標記某一筆資料是否為「盤中即時估計值」而非正式收盤資料，
    讓 RiskEngine 在做 Beta 對齊等跨資料源比較時，可以選擇排除掉這種
    尚未定案的最後一筆，避免跟大盤基準（沒有即時覆蓋）的時間點錯位比較。
    使用 ALTER TABLE ... ADD COLUMN 做向後相容的 schema migration，
    舊資料庫升級時不會遺失既有快取資料。
    """

    DEFAULT_DB_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tqai.db"
    )

    # ==========================================
    # 連線與資料表初始化
    # ==========================================
    @staticmethod
    def get_connection(db_path: str = None):
        db_path = db_path or DatabaseEngine.DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        DatabaseEngine._init_schema(conn)
        return conn

    @staticmethod
    def _init_schema(conn):
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                is_intraday_estimate INTEGER DEFAULT 0,
                PRIMARY KEY (ticker, date)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sync_metadata (
                ticker TEXT PRIMARY KEY,
                resolved_symbol TEXT,
                last_updated TEXT,
                row_count INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_directory (
                code TEXT NOT NULL,
                market TEXT NOT NULL,
                name TEXT,
                industry TEXT,
                is_etf INTEGER DEFAULT 0,
                yf_suffix TEXT,
                last_updated TEXT,
                PRIMARY KEY (code, market)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kv_cache (
                key TEXT PRIMARY KEY,
                updated_at TEXT,
                payload_json TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shareholding_snapshot_history (
                date TEXT NOT NULL,
                code TEXT NOT NULL,
                large_holder_pct REAL,
                total_holders INTEGER,
                total_shares INTEGER,
                PRIMARY KEY (date, code)
            )
        """)
        # 向後相容：舊資料庫可能沒有這個欄位，補上去但不影響既有資料
        cur.execute("PRAGMA table_info(stock_prices)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "is_intraday_estimate" not in existing_cols:
            try:
                cur.execute("ALTER TABLE stock_prices ADD COLUMN is_intraday_estimate INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
        conn.commit()

    # ==========================================
    # 寫入快取
    # ==========================================
    @staticmethod
    def save_prices(ticker: str, df: pd.DataFrame, resolved_symbol: str = None, db_path: str = None):
        if df is None or df.empty:
            return

        conn = DatabaseEngine.get_connection(db_path)
        try:
            cols = ["date", "open", "high", "low", "close", "volume"]
            save_df = df[cols].copy()
            if "is_intraday_estimate" in df.columns:
                save_df["is_intraday_estimate"] = df["is_intraday_estimate"].astype(bool).astype(int)
            else:
                save_df["is_intraday_estimate"] = 0
            save_df["date"] = pd.to_datetime(save_df["date"]).dt.strftime("%Y-%m-%d")
            save_df.insert(0, "ticker", str(ticker))

            save_df.to_sql("stock_prices_staging", conn, if_exists="replace", index=False)

            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO stock_prices
                    (ticker, date, open, high, low, close, volume, is_intraday_estimate)
                SELECT ticker, date, open, high, low, close, volume, is_intraday_estimate
                FROM stock_prices_staging
            """)
            cur.execute("DROP TABLE stock_prices_staging")
            cur.execute("""
                INSERT OR REPLACE INTO sync_metadata (ticker, resolved_symbol, last_updated, row_count)
                VALUES (?, ?, ?, ?)
            """, (str(ticker), resolved_symbol, datetime.now().isoformat(), len(save_df)))
            conn.commit()
        finally:
            conn.close()

    # ==========================================
    # 讀取快取
    # ==========================================
    @staticmethod
    def load_prices(ticker: str, db_path: str = None) -> pd.DataFrame:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume, is_intraday_estimate FROM stock_prices "
                "WHERE ticker = ? ORDER BY date",
                conn, params=(str(ticker),)
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            if "is_intraday_estimate" in df.columns:
                df["is_intraday_estimate"] = df["is_intraday_estimate"].fillna(0).astype(bool)
        return df

    @staticmethod
    def get_resolved_symbol(ticker: str, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT resolved_symbol FROM sync_metadata WHERE ticker = ?", (str(ticker),))
            row = cur.fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    @staticmethod
    def get_last_updated(ticker: str, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT last_updated FROM sync_metadata WHERE ticker = ?", (str(ticker),))
            row = cur.fetchone()
        finally:
            conn.close()

        if not row or not row[0]:
            return None
        try:
            return datetime.fromisoformat(row[0])
        except Exception:
            return None

    @staticmethod
    def is_fresh(ticker: str, max_age_hours: float = 6, db_path: str = None) -> bool:
        last_updated = DatabaseEngine.get_last_updated(ticker, db_path)
        if last_updated is None:
            return False
        return (datetime.now() - last_updated) < timedelta(hours=max_age_hours)

    # ==========================================
    # 工具方法：清除單一股票快取 / 查詢資料庫狀態
    # ==========================================
    @staticmethod
    def clear_cache(ticker: str, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM stock_prices WHERE ticker = ?", (str(ticker),))
            cur.execute("DELETE FROM sync_metadata WHERE ticker = ?", (str(ticker),))
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def get_db_stats(db_path: str = None) -> dict:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(DISTINCT ticker) FROM stock_prices")
            ticker_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM stock_prices")
            row_count = cur.fetchone()[0]
        finally:
            conn.close()
        return {"cached_tickers": ticker_count, "total_rows": row_count}

    # ==========================================
    # 股票代碼/名稱目錄快取（配合 StockDirectoryEngine）
    # ==========================================
    @staticmethod
    def save_stock_directory(df: pd.DataFrame, db_path: str = None):
        """寫入/覆蓋某個市場別的代碼/名稱清單快取。"""
        if df is None or df.empty:
            return
        conn = DatabaseEngine.get_connection(db_path)
        try:
            save_df = df[["code", "market", "name", "industry", "is_etf", "yf_suffix"]].copy()
            save_df["is_etf"] = save_df["is_etf"].astype(bool).astype(int)
            save_df["last_updated"] = datetime.now().isoformat()

            markets_in_df = save_df["market"].unique().tolist()
            cur = conn.cursor()
            for m in markets_in_df:
                cur.execute("DELETE FROM stock_directory WHERE market = ?", (m,))

            save_df.to_sql("stock_directory_staging", conn, if_exists="replace", index=False)
            cur.execute("""
                INSERT OR REPLACE INTO stock_directory
                    (code, market, name, industry, is_etf, yf_suffix, last_updated)
                SELECT code, market, name, industry, is_etf, yf_suffix, last_updated
                FROM stock_directory_staging
            """)
            cur.execute("DROP TABLE stock_directory_staging")
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def load_stock_directory(markets=None, db_path: str = None) -> pd.DataFrame:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            if markets:
                placeholders = ",".join(["?"] * len(markets))
                query = f"SELECT * FROM stock_directory WHERE market IN ({placeholders})"
                df = pd.read_sql_query(query, conn, params=list(markets))
            else:
                df = pd.read_sql_query("SELECT * FROM stock_directory", conn)
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()

        if not df.empty and "is_etf" in df.columns:
            df["is_etf"] = df["is_etf"].fillna(0).astype(bool)
        return df

    @staticmethod
    def lookup_stock_name(code: str, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM stock_directory WHERE code = ? LIMIT 1", (str(code),))
            row = cur.fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    # ==========================================
    # 通用小型快取 (Generic Key-Value Cache)
    # ==========================================
    # ⚠️ 用途：給「不是逐日K線、也不是股票代碼目錄」的其他小型、需要新鮮期限
    # 的資料使用（目前用途：ChipEngine.get_market_wide_institutional_ranking
    # 的全市場當日排行原始資料）。比照 get_stock_data() 系列的快取設計精神
    # （新鮮期限內直接吃快取、過期才重新抓取），但用一個通用的 key/JSON
    # payload 儲存方式，避免每多一種需要快取的資料就要新增一張專屬資料表。
    #
    # ⚠️ JSON 序列化陷阱：payload 內若含有 DataFrame.to_dict() 轉出來的數值，
    # 底層型別通常是 numpy.int64／numpy.float64／numpy.bool_，這些型別
    # `json.dumps` 預設不認得，會直接拋 TypeError。這裡用 default=
    # 參數統一轉型，呼叫端不需要自己在存入前手動轉換每一欄位。
    @staticmethod
    def _json_default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    @staticmethod
    def set_cache(key: str, payload, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            payload_json = json.dumps(payload, ensure_ascii=False, default=DatabaseEngine._json_default)
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO kv_cache (key, updated_at, payload_json)
                VALUES (?, ?, ?)
            """, (key, datetime.now().isoformat(), payload_json))
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def get_cache(key: str, max_age_hours: float = 6, db_path: str = None):
        """回傳 None 代表快取不存在或已過期（呼叫端應改為重新產生資料並呼叫
        set_cache 更新）。命中時回傳 {'updated_at':.., 'payload':..}。"""
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT updated_at, payload_json FROM kv_cache WHERE key = ?", (key,))
            row = cur.fetchone()
        except Exception:
            row = None
        finally:
            conn.close()

        if not row:
            return None

        updated_at, payload_json = row
        try:
            updated_dt = datetime.fromisoformat(updated_at)
        except Exception:
            return None
        if (datetime.now() - updated_dt) >= timedelta(hours=max_age_hours):
            return None

        try:
            payload = json.loads(payload_json)
        except Exception:
            return None

        return {"updated_at": updated_at, "payload": payload}

    # ==========================================
    # 大戶持股(千張大戶)歷史快照（配合 ChipEngine.get_shareholding_distribution）
    # ==========================================
    # ⚠️ 說明：TDCC 集保戶股權分散表開放資料每次抓到的都只是「當週最新
    # 快照」，官方沒有提供逐週歷史（TDCC自家查詢介面雖然有個股近1年歷史，
    # 但那是另一組不同的查詢方式，不在這次開放資料 CSV 範圍內）。這裡改成
    # 每次成功抓到新一週的資料時，自己把「大戶持股佔比」精簡摘要
    # （不是整份原始CSV）存進這張小表，隨著這個系統被持續使用，自然
    # 累積出屬於這個系統自己的歷史趨勢，讓使用者可以看到「大戶持股比例
    # 最近幾週的變化」，而不是每次都只看得到單一時間點的快照。
    @staticmethod
    def save_shareholding_snapshot(date: str, code: str, large_holder_pct: float,
                                    total_holders: int, total_shares: int, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO shareholding_snapshot_history
                    (date, code, large_holder_pct, total_holders, total_shares)
                VALUES (?, ?, ?, ?, ?)
            """, (date, str(code), large_holder_pct, total_holders, total_shares))
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def load_shareholding_history(code: str, weeks: int = 12, db_path: str = None) -> pd.DataFrame:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            df = pd.read_sql_query(
                "SELECT date, large_holder_pct, total_holders, total_shares "
                "FROM shareholding_snapshot_history WHERE code = ? ORDER BY date DESC LIMIT ?",
                conn, params=(str(code), weeks)
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df