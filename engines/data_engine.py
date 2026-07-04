import yfinance as yf
import pandas as pd
import numpy as np

class DataEngine:
    """
    🚀 TQAI Pro 數據清洗與即時覆蓋引擎

    ⚠️ 修正說明（v2.7 興櫃/ETF資料擴充）：
    原本只嘗試 .TW（上市）失敗後改試 .TWO（上櫃）一次，對「上市/上櫃/ETF」
    已經夠用（ETF 本質上也是掛在 .TW 或 .TWO 底下的數字代碼，原本邏輯就
    支援），但錯誤訊息完全沒提到「興櫃」這個可能性，使用者查詢興櫃股票
    失敗時只會看到一句「查無資料」，不知道原因。

    這裡把候選後綴改成清單，依序嘗試（目前仍是 .TW / .TWO，因為 yfinance
    本身對台灣興櫃股票的資料覆蓋率非常低，即使加上其他後綴用猜的大機率
    仍抓不到 K 線，沒有意義），但在最終失敗時的錯誤訊息中明確點出「若為
    興櫃股票，yfinance 目前對興櫃資料覆蓋率低」這個限制，讓使用者理解
    失敗原因而不是誤以為程式壞掉。ETF 部分因為原邏輯已相容，不需要特別
    改資料抓取流程，只需要在顯示端（NameEngine／ScannerEngine）補上市場
    別標籤。
    """

    # 依序嘗試的上市/上櫃後綴（興櫃目前無可靠的 yfinance 後綴可補，見上方說明）
    _SUFFIX_CANDIDATES = [".TW", ".TWO"]

    @staticmethod
    def get_stock_data(ticker: str, use_cache: bool = True, max_age_hours: float = 6):
        from engines.db_engine import DatabaseEngine
        ticker = str(ticker).strip()

        # 0. 資料庫快取讀取
        if use_cache:
            try:
                resolved_symbol = DatabaseEngine.get_resolved_symbol(ticker)
                if resolved_symbol and DatabaseEngine.is_fresh(ticker, max_age_hours):
                    cached_df = DatabaseEngine.load_prices(ticker)
                    if not cached_df.empty:
                        return cached_df.sort_values('date').reset_index(drop=True)
            except Exception:
                pass 
        
        # 1. 判斷上市與上櫃符號（依序嘗試 .TW / .TWO，兩者皆涵蓋一般股票與ETF）
        if ticker.isdigit():
            tkr = None
            ticker_try = None
            for suffix in DataEngine._SUFFIX_CANDIDATES:
                candidate = ticker + suffix
                cand_tkr = yf.Ticker(candidate)
                try:
                    hist = cand_tkr.history(period="1d")
                except Exception:
                    hist = pd.DataFrame()
                if not hist.empty:
                    tkr, ticker_try = cand_tkr, candidate
                    break
            if tkr is None:
                # 兩種後綴都查無即時資料：可能是興櫃股票（yfinance覆蓋率低）
                # 或代碼輸入錯誤，先用最後一個候選繼續往下嘗試 yf.download，
                # 讓下面統一的錯誤處理輸出明確訊息。
                ticker_try = ticker + DataEngine._SUFFIX_CANDIDATES[-1]
                tkr = yf.Ticker(ticker_try)
        else:
            ticker_try = ticker
            tkr = yf.Ticker(ticker_try)
            
        # 2. 下載歷史數據
        df = yf.download(ticker_try, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            raise Exception(
                f"【錯誤】查無股票代碼 [{ticker}] 的資料。"
                f"若為興櫃股票，yfinance 目前對興櫃資料覆蓋率偏低，可能沒有歷史K線可用；"
                f"一般上市/上櫃股票與ETF應可正常查詢，請確認代碼是否正確。"
            )
            
        # 安全壓平 MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
        # 清除不完整的盤後零交易列
        if len(df) > 1:
            last_row = df.iloc[-1]
            if (last_row['volume'] == 0) or pd.isna(last_row['close']):
                df = df.iloc[:-1].reset_index(drop=True)
                
        df = df.dropna(subset=['close', 'open', 'high', 'low']).reset_index(drop=True)

        # 3. 盤中即時價格覆蓋機制 (安全賦值)
        # ⚠️ 修正說明：原本只覆蓋 close，沒有同步調整 high/low，會出現
        # close > high 或 close < low 這種違反 OHLC 定義的資料，汙染下游所有
        # 指標（ATR、布林通道、RSI...）與K線圖。現在覆蓋 close 的同時，
        # 同步把 high/low 撐開到至少涵蓋新的 close，並且用 is_intraday_estimate
        # 欄位標記「這根K棒尚未收盤、是即時估計值」，讓下游（例如快取新鮮度判斷、
        # Beta/VaR 對齊）可以知道最後一筆不是正式收盤資料。
        df['is_intraday_estimate'] = False
        try:
            real_price = None
            if hasattr(tkr, 'fast_info'):
                real_price = getattr(tkr.fast_info, 'last_price', None)
            if real_price is None and hasattr(tkr, 'info'):
                real_price = tkr.info.get('currentPrice', None)

            if real_price is not None and real_price > 0 and not df.empty:
                real_price = float(real_price)
                close_col = df.columns.get_loc('close')
                high_col = df.columns.get_loc('high')
                low_col = df.columns.get_loc('low')
                flag_col = df.columns.get_loc('is_intraday_estimate')

                current_high = float(df.iloc[-1, high_col])
                current_low = float(df.iloc[-1, low_col])

                # 採用絕對位置 iloc 修改最後一筆，徹底阻斷 MultiIndex 或 index 不連續引發的 Bug
                df.iloc[-1, close_col] = real_price
                # 同步撐開 high/low，維持 OHLC 內部一致性（high >= close >= low）
                df.iloc[-1, high_col] = max(current_high, real_price)
                df.iloc[-1, low_col] = min(current_low, real_price)
                df.iloc[-1, flag_col] = True
        except Exception:
            pass

        # 4. 快取寫入
        # 注意：即時估計的最後一筆仍會寫入快取，is_intraday_estimate 會一併保存，
        # 下游可依此欄位判斷是否要排除該筆（例如 Beta 對齊計算）。
        if use_cache:
            try:
                DatabaseEngine.save_prices(ticker, df, resolved_symbol=ticker_try)
            except Exception:
                pass

        return df

    @staticmethod
    def get_benchmark_data(symbol: str = "^TWII"):
        try:
            df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df.columns = [str(c).strip().lower() for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            return df.dropna(subset=['close']).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()