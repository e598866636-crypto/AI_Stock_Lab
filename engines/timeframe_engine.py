import numpy as np
import pandas as pd


class TimeframeEngine:
    """
    ⏳ 多週期策略與未來走向引擎 (Timeframe & Outlook Engine) - TQAI Pro v2.7

    背景與動機：
    StrategyEngine.ai_score 是「單一時間尺度」的綜合裁決，回答的是波段
    (2週~2個月) 這個尺度上「值不值得操作」。但使用者實務上依照自己的
    操作週期（當沖/短線、波段、長線存股）需要不同的判斷角度 —— 同一檔
    股票很常出現「短線過熱要小心、但長線結構仍偏多」這種分歧狀態，如果
    只看單一 ai_score 容易誤判操作時機。

    設計原則：
    - 本引擎完全「不重新計算」任何技術指標，只讀取 IndicatorEngine /
      StrategyEngine / DivergenceEngine 已經算好的欄位做二次判斷與整合，
      避免又冒出一套獨立、可能互相矛盾的指標定義。
    - 三個週期分別對應不同的既有欄位組合：
        短線 (約1~5個交易日)：EMA8 / RSI / KD / MACD 這類反應快的短週期
            指標，並納入近5日內的誘多/背離防禦訊號。
        波段 (約2週~2個月)：直接沿用 StrategyEngine 的 ai_score /
            market_regime，這正是 ai_score 原本設計要回答的問題，並額外
            納入近10日的防禦訊號滾動視窗（比短線視窗更寬）。
        長線 (半年以上)：偏重 sma_120 / sma_200 年線結構與其斜率，不受
            短期雜訊影響。
    - 三週期結果會再彙整成一個「未來走向 (outlook)」情境推演。

    ⚠️ 關於「未來走向 (Outlook)」的誠實揭露（務必保留此區塊）：
    這裡的 outlook 純粹是「如果目前技術結構延續，可能出現的情境」，是
    條件式的技術面推論 (if-then)，**不是**對未來價格的預測或保證，也不
    構成投資建議。技術分析無法保證未來走勢：總體經濟、產業消息、法人
    動向、公司基本面變化、國際情勢等因素都可能在任何時間點推翻目前的
    技術結構。使用者應自行判斷風險、獨立做出投資決策並自負盈虧，本引擎
    與其呼叫端 (Dashboard) 都不應把這個區塊呈現成「保證會發生的走勢」。

    ⚠️ 使用前提與容錯：
    使用前應先跑過 IndicatorEngine + StrategyEngine（提供 ai_score /
    market_regime 等欄位）；若同時跑過 DivergenceEngine，短線/波段的
    風險判斷會更準確（會納入誘多/背離防禦訊號），沒有的話則以中性、
    不加分/不扣分處理，不會拋出例外中斷呼叫端。
    """

    _SHORT_LOOKBACK = 5    # 短線防禦訊號視窗（交易日）
    _SWING_LOOKBACK = 10   # 波段防禦訊號視窗（交易日）
    _LONG_SLOPE_WINDOW = 20  # 長線年線斜率觀察視窗（交易日）

    # ==========================================
    # 工具方法
    # ==========================================
    @staticmethod
    def _get(row, col, default=np.nan):
        try:
            if col not in row.index:
                return default
            val = row[col]
            if pd.isna(val):
                return default
            return float(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _recent_flag(history: pd.DataFrame, col: str, lookback: int) -> bool:
        """檢查歷史資料近 lookback 天內，某個布林欄位是否曾經為 True。
        欄位不存在時視為「無資料、不觸發」，維持防禦性設計風格。"""
        if col not in history.columns or history.empty:
            return False
        window = history[col].tail(lookback)
        return bool(window.fillna(False).astype(bool).any())

    # ==========================================
    # 1. 短線判斷 (約1~5個交易日)
    # ==========================================
    @staticmethod
    def _assess_short_term(history: pd.DataFrame, row: pd.Series) -> dict:
        close = TimeframeEngine._get(row, 'close')
        ema8 = TimeframeEngine._get(row, 'ema_8')
        rsi = TimeframeEngine._get(row, 'rsi_14')
        k9, d9 = TimeframeEngine._get(row, 'k_9'), TimeframeEngine._get(row, 'd_9')
        macd_hist = TimeframeEngine._get(row, 'macd_hist')

        bull_pts, bear_pts, notes = 0, 0, []

        if pd.notna(close) and pd.notna(ema8):
            if close > ema8:
                bull_pts += 1
                notes.append("站上EMA8短均線")
            else:
                bear_pts += 1
                notes.append("跌破EMA8短均線")

        if pd.notna(macd_hist):
            if macd_hist > 0:
                bull_pts += 1
                notes.append("MACD柱狀體翻正")
            else:
                bear_pts += 1
                notes.append("MACD柱狀體翻負")

        if pd.notna(k9) and pd.notna(d9):
            if k9 > d9:
                bull_pts += 1
                notes.append("KD呈多頭排列")
            else:
                bear_pts += 1
                notes.append("KD呈空頭排列")

        if pd.notna(rsi):
            if rsi >= 80:
                bear_pts += 1
                notes.append(f"RSI={rsi:.0f}過熱，短線拉回風險升高")
            elif rsi <= 20:
                bull_pts += 1
                notes.append(f"RSI={rsi:.0f}超賣，短線可能出現反彈")

        defense_triggered = (
            TimeframeEngine._recent_flag(history, 'bull_trap_confirmed', TimeframeEngine._SHORT_LOOKBACK)
            or TimeframeEngine._recent_flag(history, 'bearish_divergence', TimeframeEngine._SHORT_LOOKBACK)
        )
        if defense_triggered:
            bear_pts += 1
            notes.append(f"近{TimeframeEngine._SHORT_LOOKBACK}日內出現誘多假突破/頂背離警報")

        if bull_pts - bear_pts >= 2:
            view = "偏多"
        elif bear_pts - bull_pts >= 2:
            view = "偏空"
        else:
            view = "中性/觀望"

        action_map = {
            "偏多": "短線動能偏多，已進場者可續抱並嚴設短線停利/停損，未進場者不建議追高，可等待回測支撐再評估",
            "偏空": "短線動能偏弱，不建議短線搶反彈，留意是否持續跌破近期支撐",
            "中性/觀望": "短線多空訊號不明確，建議觀望，避免頻繁進出增加交易成本",
        }

        return {
            "view": view,
            "horizon": f"約1~{TimeframeEngine._SHORT_LOOKBACK}個交易日",
            "reason": "、".join(notes) if notes else "資料不足，暫無法判斷短線動能",
            "action": action_map[view],
        }

    # ==========================================
    # 2. 波段判斷 (約2週~2個月) — 沿用 ai_score / market_regime
    # ==========================================
    @staticmethod
    def _assess_swing(history: pd.DataFrame, row: pd.Series) -> dict:
        ai_score = TimeframeEngine._get(row, 'ai_score', 50.0)
        regime = row['market_regime'] if 'market_regime' in row.index else 'N/A'
        action_guide = row['action_guide'] if 'action_guide' in row.index else ''

        defense_bear = (
            TimeframeEngine._recent_flag(history, 'bull_trap_confirmed', TimeframeEngine._SWING_LOOKBACK)
            or TimeframeEngine._recent_flag(history, 'bearish_divergence', TimeframeEngine._SWING_LOOKBACK)
        )
        reversal_watch = (
            TimeframeEngine._recent_flag(history, 'bear_trap_confirmed', TimeframeEngine._SWING_LOOKBACK)
            or TimeframeEngine._recent_flag(history, 'bullish_divergence', TimeframeEngine._SWING_LOOKBACK)
        )

        if ai_score >= 70 and not defense_bear:
            view = "偏多"
        elif ai_score <= 45 or defense_bear:
            view = "偏空"
        else:
            view = "中性/觀望"

        notes = [f"AI Score={ai_score:.1f}", f"市場狀態：{regime}"]
        if defense_bear:
            notes.append(f"近{TimeframeEngine._SWING_LOOKBACK}日內有背離/誘多警報，波段風險升高")
        if reversal_watch:
            notes.append(f"近{TimeframeEngine._SWING_LOOKBACK}日內有底背離/誘空反轉觀察訊號，可留意是否醞釀反彈（僅供觀察，非確認訊號）")

        return {
            "view": view,
            "horizon": "約2週~2個月",
            "reason": "、".join(notes),
            "action": action_guide if action_guide else "請參考AI主審裁決 (ai_score)",
        }

    # ==========================================
    # 3. 長線判斷 (半年以上) — 年線與長期均線結構
    # ==========================================
    @staticmethod
    def _assess_long_term(history: pd.DataFrame, row: pd.Series) -> dict:
        close = TimeframeEngine._get(row, 'close')
        sma200 = TimeframeEngine._get(row, 'sma_200')

        if pd.isna(sma200) or 'sma_200' not in history.columns:
            return {
                "view": "資料不足",
                "horizon": "半年以上",
                "reason": "尚無足夠交易日計算年線(SMA200)，暫不評估長線結構",
                "action": "資料不足，暫不提供長線建議",
            }

        sma200_series = history['sma_200'].dropna()
        sma200_slope = np.nan
        if len(sma200_series) > TimeframeEngine._LONG_SLOPE_WINDOW:
            sma200_slope = sma200_series.iloc[-1] - sma200_series.iloc[-TimeframeEngine._LONG_SLOPE_WINDOW]

        above_year_line = pd.notna(close) and close > sma200
        year_line_rising = pd.notna(sma200_slope) and sma200_slope > 0
        year_line_falling = pd.notna(sma200_slope) and sma200_slope < 0

        if above_year_line and year_line_rising:
            view = "偏多"
            reason = "股價站上年線且年線向上，長期趨勢結構健康"
        elif above_year_line and not year_line_rising:
            view = "中性偏多(轉弱觀察)"
            reason = "股價仍在年線之上，但年線走平或斜率轉弱，留意長期動能是否減弱"
        elif (not above_year_line) and (year_line_falling or pd.isna(sma200_slope)):
            view = "偏空"
            reason = "股價跌破年線，且年線走平或向下，長期趨勢結構偏弱"
        else:
            view = "中性/觀望"
            reason = "股價與年線關係尚不明確，長期結構仍在轉換中"

        action_map = {
            "偏多": "長線結構偏多，可列入長期存股/波段布局觀察名單，仍建議分批進場以平滑成本",
            "中性偏多(轉弱觀察)": "長線結構仍偏多但動能可能減弱，不建議大幅加碼，留意是否跌破年線",
            "偏空": "長線結構偏弱，現階段不建議長線佈局，待股價站回年線且年線翻揚後再重新評估",
            "中性/觀望": "長線結構尚不明確，建議持續觀察，暫緩大幅加減碼",
        }

        return {
            "view": view,
            "horizon": "半年以上",
            "reason": reason,
            "action": action_map.get(view, "建議持續觀察"),
        }

    # ==========================================
    # 4. 未來走向 (Outlook) — 條件式情境推演，非預測保證
    # ==========================================
    @staticmethod
    def _build_outlook(row: pd.Series, short_term: dict, swing: dict, long_term: dict) -> dict:
        close = TimeframeEngine._get(row, 'close')
        atr = TimeframeEngine._get(row, 'atr_14')
        sma60 = TimeframeEngine._get(row, 'sma_60')

        views = [short_term['view'], swing['view'], long_term['view']]
        bull_votes = sum(1 for v in views if v.startswith('偏多'))
        bear_votes = sum(1 for v in views if v.startswith('偏空'))

        if bull_votes >= 2 and bear_votes == 0:
            bias = "偏多"
            bias_note = "短、波、長三個週期多數呈現偏多訊號，趨勢一致性較高"
        elif bear_votes >= 2 and bull_votes == 0:
            bias = "偏空"
            bias_note = "短、波、長三個週期多數呈現偏空訊號，趨勢一致性較高"
        elif short_term['view'] == '偏空' and long_term['view'].startswith('偏多'):
            bias = "短空長多(分歧)"
            bias_note = "短線出現拉回訊號，但長線結構仍偏多，較可能是波段回檔而非長期趨勢反轉，惟仍需觀察是否進一步破壞長線結構"
        elif short_term['view'] == '偏多' and long_term['view'] == '偏空':
            bias = "短多長空(分歧)"
            bias_note = "短線出現反彈動能，但長線結構偏空，反彈力道與延續性宜保守看待，慎防僅為逃命波或反彈出貨"
        else:
            bias = "中性/多空拉鋸"
            bias_note = "三個週期訊號不一致或均不明確，趨勢方向尚未明朗，建議耐心等待訊號收斂"

        resistance = close + 2.5 * atr if pd.notna(close) and pd.notna(atr) else np.nan
        support = close - 2.0 * atr if pd.notna(close) and pd.notna(atr) else np.nan

        scenarios = []
        if pd.notna(resistance):
            scenarios.append({
                "condition": f"若價格站穩並突破約 {resistance:.2f}（依近期ATR推算的壓力區）",
                "implication": "動能可能延續，短線偏多情境獲得驗證，惟仍應留意量能是否同步放大",
            })
        if pd.notna(support):
            scenarios.append({
                "condition": f"若價格跌破約 {support:.2f}（依近期ATR推算的支撐區）",
                "implication": "偏多結構可能轉弱，建議依原訂停損計畫執行，避免心存僥倖",
            })
        if pd.notna(sma60):
            scenarios.append({
                "condition": f"季線(60MA)目前約在 {sma60:.2f} 附近",
                "implication": "為觀察波段多空轉折的重要參考位置，跌破常伴隨波段轉弱訊號",
            })

        return {
            "bias": bias,
            "bias_note": bias_note,
            "key_levels": {
                "support_est": None if pd.isna(support) else round(float(support), 2),
                "resistance_est": None if pd.isna(resistance) else round(float(resistance), 2),
                "sma_60": None if pd.isna(sma60) else round(float(sma60), 2),
            },
            "scenarios": scenarios,
            "disclaimer": (
                "⚠️ 以上「未來走向」為根據目前技術結構所做的條件式情境推演（若...則...），"
                "並非對未來價格的預測或保證，也不構成投資建議。技術面隨時可能因基本面消息、"
                "法人動向、總體經濟或市場情緒轉變而失效，請勿單獨依賴本區塊做出投資決策，"
                "並自行評估風險與部位控管。"
            ),
        }

    # ==========================================
    # 主要進入點
    # ==========================================
    @staticmethod
    def build_report(df: pd.DataFrame, idx: int = -1) -> dict:
        """
        對單一時間點（預設最新一筆）產出「短線 / 波段 / 長線」三週期判斷，
        以及彙整後的「未來走向」情境推演報告，供 Dashboard 顯示。

        使用前提：df 應已跑過 IndicatorEngine + StrategyEngine（至少要有
        ai_score / market_regime 欄位）；若同時跑過 DivergenceEngine，
        短線/波段的風險判斷會納入誘多/背離防禦訊號、更為準確。

        Parameters
        ----------
        df  : 完整 pipeline 處理後的 DataFrame
        idx : 要評估的資料列位置（預設 -1 = 最新一筆），主要保留給未來
              想針對歷史某一天做回顧分析時使用
        """
        if df is None or df.empty:
            return {"error": "資料不足，無法產生多週期策略報告"}

        n = len(df)
        pos = idx if idx >= 0 else n + idx
        if pos < 0 or pos >= n:
            return {"error": "指定的資料位置超出範圍"}

        row = df.iloc[pos]
        history = df.iloc[: pos + 1]

        short_term = TimeframeEngine._assess_short_term(history, row)
        swing = TimeframeEngine._assess_swing(history, row)
        long_term = TimeframeEngine._assess_long_term(history, row)
        outlook = TimeframeEngine._build_outlook(row, short_term, swing, long_term)

        return {
            "short_term": short_term,
            "swing": swing,
            "long_term": long_term,
            "outlook": outlook,
        }