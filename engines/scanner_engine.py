import pandas as pd

from engines.data_engine import DataEngine
from engines.indicator_engine import IndicatorEngine
from engines.structure_engine import StructureEngine
from engines.risk_engine import RiskEngine
from engines.strategy_engine import StrategyEngine
from engines.evidence_engine import EvidenceEngine

class ScannerEngine:
    """
    📡 全台股掃描引擎 (Market Scanner) - 前沿科技特化版
    """

    # 專屬觀察名單：整合跨界生醫、生物運算與大盤權值指標
    DEFAULT_WATCHLIST = [
        # --- 跨界生醫與生物電腦 ---
        "2330", "3374", "6223", "3711", # 生物晶片與先進封測
        "6841", "2382", "3231", "2356", # AI醫療與智慧生醫
        "6472", "6901", "4743", "6712", # 合成生物與前沿創投
        
        # --- 宏觀資金動向對照組 (保留部分關鍵權值股觀察大盤健康度) ---
        "2317", "2454", "2308", "2379", "3034", "2412", "2881", "2603", "1515", "1101"
    ]

    @staticmethod
    def _run_single_pipeline(ticker: str, use_cache: bool = True, max_age_hours: float = 6):
        df = DataEngine.get_stock_data(ticker, use_cache=use_cache, max_age_hours=max_age_hours)
        df = IndicatorEngine.add_indicators(df)
        df = StructureEngine.add_swing_points(df)
        df = RiskEngine.add_risk_metrics(df)
        df = StrategyEngine.generate_signals(df)
        df = EvidenceEngine.add_evidence(df)
        return df

    @staticmethod
    def scan(tickers=None, use_cache: bool = True, max_age_hours: float = 6, progress_callback=None):
        tickers = tickers or ScannerEngine.DEFAULT_WATCHLIST
        results = []
        errors = []

        for i, ticker in enumerate(tickers):
            ticker = str(ticker).strip()
            if not ticker:
                continue
            try:
                df = ScannerEngine._run_single_pipeline(ticker, use_cache=use_cache, max_age_hours=max_age_hours)
                latest = df.iloc[-1]

                close_val = latest["close"]
                if hasattr(close_val, "iloc"):
                    close_val = close_val.iloc[0]

                results.append({
                    "代碼": ticker,
                    "收盤價": round(float(close_val), 2),
                    "市場狀態": latest.get("market_regime", "N/A"),
                    "AI Score": round(float(latest.get("ai_score", 0)), 1),
                    "信心度": round(float(latest.get("confidence_pct", 0)), 0),
                    "資料品質": round(float(latest.get("data_quality_pct", 0)), 0),
                    "操作建議": latest.get("action_guide", "N/A"),
                    "年化波動率": round(float(latest.get("volatility_annualized", float("nan"))), 1)
                        if pd.notna(latest.get("volatility_annualized", float("nan"))) else None,
                    "60日回撤": round(float(latest.get("rolling_mdd_60d", float("nan"))), 1)
                        if pd.notna(latest.get("rolling_mdd_60d", float("nan"))) else None,
                })
            except Exception as e:
                errors.append({"代碼": ticker, "錯誤訊息": str(e)})

            if progress_callback:
                progress_callback(i + 1, len(tickers), ticker)

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values("AI Score", ascending=False).reset_index(drop=True)
            result_df.insert(0, "排名", range(1, len(result_df) + 1))

        error_df = pd.DataFrame(errors)
        return result_df, error_df

    @staticmethod
    def get_top_n(result_df: pd.DataFrame, n: int = 10):
        if result_df is None or result_df.empty:
            return result_df
        return result_df.head(n)