import pandas as pd

from engines.data_engine import DataEngine
from engines.indicator_engine import IndicatorEngine
from engines.structure_engine import StructureEngine
from engines.risk_engine import RiskEngine
from engines.divergence_engine import DivergenceEngine
from engines.strategy_engine import StrategyEngine
from engines.momentum_engine import MomentumEngine
from engines.evidence_engine import EvidenceEngine
from engines.name_engine import NameEngine

class ScannerEngine:
    """
    📡 全台股掃描引擎 (Market Scanner) - 前沿科技特化版

    ⚠️ 修正說明（v2.7）：
    1. Pipeline 新增 DivergenceEngine（背離/誘多誘空防禦）與 MomentumEngine
       （飆股七層過濾＋100分評分）兩個階段。DivergenceEngine 必須放在
       StrategyEngine 之前（避免 StrategyEngine/MomentumEngine 用到還沒算
       好的背離欄位），MomentumEngine 則放在 StrategyEngine 之後、
       EvidenceEngine 之前均可（兩者互不相依）。
    2. scan() 回傳結果新增 [代碼][名稱] 全系統標籤化欄位（來自 NameEngine），
       以及「飆股評分／飆股等級／誘盤警報」三個新欄位，供 Dashboard 的
       「A級飆股候選」與「誘盤警報雷達」區塊使用。
    3. DEFAULT_WATCHLIST 新增兩檔常見 ETF（0050、006208）作為 ETF 擴充功能
       的示範標的；ETF 沒有「產業」可言，IndustryEngine 已將其獨立歸類為
       「ETF」類別，不會污染其他產業的平均分數。
    """

    # 專屬觀察名單：整合跨界生醫、生物運算、大盤權值指標與常見 ETF
    DEFAULT_WATCHLIST = [
        # --- 跨界生醫與生物電腦 ---
        "2330", "3374", "6223", "3711", # 生物晶片與先進封測
        "6841", "2382", "3231", "2356", # AI醫療與智慧生醫
        "6472", "6901", "4743", "6712", # 合成生物與前沿創投
        
        # --- 宏觀資金動向對照組 (保留部分關鍵權值股觀察大盤健康度) ---
        "2317", "2454", "2308", "2379", "3034", "2412", "2881", "2603", "1515", "1101",

        # --- ETF 示範標的（v2.7 興櫃/ETF資料擴充）---
        "0050", "006208",
    ]

    @staticmethod
    def _run_single_pipeline(ticker: str, use_cache: bool = True, max_age_hours: float = 6):
        df = DataEngine.get_stock_data(ticker, use_cache=use_cache, max_age_hours=max_age_hours)
        df = IndicatorEngine.add_indicators(df)
        df = StructureEngine.add_swing_points(df)
        df = RiskEngine.add_risk_metrics(df)
        df = DivergenceEngine.add_defense_signals(df)
        df = StrategyEngine.generate_signals(df)
        df = MomentumEngine.add_momentum_score(df)
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

                trap_flag = bool(latest.get("trap_alert", False))

                results.append({
                    "代碼": ticker,
                    "名稱": NameEngine.get_name(ticker),
                    "標的": NameEngine.get_tag(ticker),
                    "市場別": NameEngine.get_market_type(ticker),
                    "收盤價": round(float(close_val), 2),
                    "市場狀態": latest.get("market_regime", "N/A"),
                    "AI Score": round(float(latest.get("ai_score", 0)), 1),
                    "飆股評分": round(float(latest.get("momentum_score", 0)), 1),
                    "飆股等級": latest.get("momentum_grade", "D"),
                    "誘盤警報": "⚠️ 是" if trap_flag else "—",
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

    @staticmethod
    def get_a_grade_candidates(result_df: pd.DataFrame):
        """
        🚀 A級飆股候選 (Momentum A-Grade Candidates)

        篩選 MomentumEngine 七層過濾＋100分評分系統中，飆股等級為 A
        （總分 >= 85，且已通過年線濾網硬性關卡）的標的，依飆股評分排序。
        """
        if result_df is None or result_df.empty or "飆股等級" not in result_df.columns:
            return pd.DataFrame()
        a_grade = result_df[result_df["飆股等級"] == "A"].copy()
        a_grade = a_grade.sort_values("飆股評分", ascending=False).reset_index(drop=True)
        return a_grade

    @staticmethod
    def get_trap_alerts(result_df: pd.DataFrame):
        """
        🛡️ 誘盤警報雷達 (Fake-Signal / Trap Radar)

        篩選近期觸發 DivergenceEngine 背離或假突破/假跌破警報的標的，
        依 AI Score 排序（分數越高但同時觸發警報的標的，代表『看起來還不錯
        但動能可能已經在減弱』，最值得優先留意）。
        """
        if result_df is None or result_df.empty or "誘盤警報" not in result_df.columns:
            return pd.DataFrame()
        alerts = result_df[result_df["誘盤警報"] != "—"].copy()
        alerts = alerts.sort_values("AI Score", ascending=False).reset_index(drop=True)
        return alerts