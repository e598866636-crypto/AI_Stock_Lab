import numpy as np
import pandas as pd


class MomentumEngine:
    """
    🚀 飆股動能引擎 (Momentum / Hot-Stock Engine) - TQAI Pro v2.7

    「飆股七層過濾 + 100分評分系統（含年線濾網）」

    設計動機：
    StrategyEngine.ai_score 回答的是「當下多空優劣勢＋風控之後，這檔股票值不
    值得操作」，是給一般波段決策用的主審裁決。MomentumEngine 回答的是更窄、
    更嚴格的問題 ——「這檔股票現在的技術結構，像不像一檔正要噴出的飆股？」，
    門檻設計得比 ai_score 更嚴苛，並且明確把「假突破／背離失效」的懲罰算進
    總分，而不是像 ai_score 那樣做連續型風控懲罰。

    七層過濾（對應 100 分配分）：
      1. 年線多頭濾網 (15分，硬性關卡)：收盤必須站上年線 (sma_200)，且年線
         本身走平或向上，否則視為不具備飆股的基本結構，即使其他層分數很
         高，總分也會被封頂（見下方 yearline gate 說明），確保「跌破年線」
         永遠評不到 A 級。
      2. 均線多頭排列 (15分)：close > sma_20 > sma_60 > sma_120，短中長期
         排列一致，代表趨勢結構健康、不是短線雜訊。
      3. 價量齊揚 (15分)：近期量能明顯放大 (RVOL) 且股價同步上漲，代表買盤
         有實際成交量支撐，不是無量假拉抬。
      4. MACD 動能強度 (15分)：柱狀體翻正並持續擴張，代表動能仍在加速，
         而非已經開始鈍化。
      5. 相對強度且未過熱 (10分)：RSI 落在「強勢但未過熱」的甜蜜區間，
         避免選到已經嚴重超買、隨時要回檔的標的。
      6. 籌碼慣性代理指標 (10分)：用 OBV 相對其均線的位置代理法人/大戶
         資金流向。刻意不呼叫 ChipEngine 的外部 TWSE API —— 全市場掃描時
         若每檔股票都對外發送籌碼查詢請求，會產生大量對外連線且拖慢
         掃描速度，這裡改用已經在 IndicatorEngine 算好的 OBV 做免費代理。
      7. 誘多/背離防禦 (20分，來自 DivergenceEngine)：近期若出現頂背離或
         誘多假突破，代表這段上漲動能可能已經是尾聲甚至陷阱，大幅扣分。

    ⚠️ 與 ai_score 的關係：
    momentum_score 刻意「不」跟 ai_score 加權平均在一起，避免重蹈
    strategy_engine.py 修正說明中提到的雙重計分問題（同一組原始事實被
    不同 Agent 各自算一次分、再疊加放大）。Dashboard 上兩者並列顯示，
    讓使用者自行比對「AI主審怎麼看」跟「這檔像不像飆股」兩個不同角度。

    ⚠️ 資料相依性與容錯：
    使用前應先跑過 IndicatorEngine（均線/RSI/MACD/RVOL/OBV）與
    DivergenceEngine.add_defense_signals（背離、誘多/誘空防禦旗標）。
    若相依欄位缺漏，該層一律以「中性、不加分」處理並繼續往下算，不會
    拋出例外中斷整條 pipeline（沿用本專案一貫的防禦性設計風格）。
    """

    _DIVERGENCE_LOOKBACK = 10  # 檢查最近幾天內是否有背離/誘多誘空訊號

    @staticmethod
    def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(default, index=df.index)

    @staticmethod
    def add_momentum_score(df: pd.DataFrame):
        df = df.copy()
        n = len(df)
        if n == 0 or 'close' not in df.columns:
            df['momentum_score'] = pd.Series(dtype=float)
            df['momentum_grade'] = pd.Series(dtype=object)
            df['is_a_grade_candidate'] = pd.Series(dtype=bool)
            df['momentum_penalty_alert'] = pd.Series(dtype=bool)
            df['reversal_watch'] = pd.Series(dtype=bool)
            df['trap_alert'] = pd.Series(dtype=bool)
            return df

        close = df['close']
        sma20 = MomentumEngine._col(df, 'sma_20')
        sma60 = MomentumEngine._col(df, 'sma_60')
        sma120 = MomentumEngine._col(df, 'sma_120')
        sma200 = MomentumEngine._col(df, 'sma_200')
        rvol = MomentumEngine._col(df, 'rvol', 1.0).fillna(1.0)
        macd_hist = MomentumEngine._col(df, 'macd_hist', 0.0).fillna(0.0)
        rsi = MomentumEngine._col(df, 'rsi_14', 50.0).fillna(50.0)
        obv = MomentumEngine._col(df, 'obv', 0.0)
        obv_sma = MomentumEngine._col(df, 'obv_sma', 0.0)

        # ---- 第①層：年線多頭濾網（硬性關卡，15分）----
        sma200_slope = sma200.diff(10)
        yearline_pass = (close > sma200) & (sma200_slope >= 0)
        yearline_above_only = (close > sma200) & ~yearline_pass
        yearline_score = np.select([yearline_pass, yearline_above_only], [15, 7], default=0)

        # ---- 第②層：均線多頭排列（15分）----
        alignment_full = (close > sma20) & (sma20 > sma60) & (sma60 > sma120)
        alignment_partial = (close > sma20) & (sma20 > sma60) & ~alignment_full
        alignment_score = np.select([alignment_full, alignment_partial], [15, 8], default=0)

        # ---- 第③層：價量齊揚（15分）----
        price_up = close > close.shift(1)
        volume_score = np.select(
            [(rvol >= 2.0) & price_up, (rvol >= 1.3) & price_up],
            [15, 8], default=0
        )

        # ---- 第④層：MACD動能強度（15分）----
        macd_growing = macd_hist > macd_hist.shift(1)
        macd_score = np.select(
            [(macd_hist > 0) & macd_growing, macd_hist > 0],
            [15, 7], default=0
        )

        # ---- 第⑤層：相對強度且未過熱（10分）----
        rsi_sweet_spot = (rsi >= 55) & (rsi <= 80)
        rsi_overheated = rsi > 80
        rsi_score = np.select([rsi_sweet_spot, rsi_overheated], [10, 3], default=0)

        # ---- 第⑥層：籌碼慣性代理指標（OBV，10分）----
        obv_score = np.where(obv > obv_sma, 10, 0)

        raw_score = yearline_score + alignment_score + volume_score + macd_score + rsi_score + obv_score

        # ---- 第⑦層：誘多/背離防禦（20分）----
        bearish_div_recent = MomentumEngine._col(df, 'bearish_divergence', False).fillna(False).astype(bool) \
            .rolling(MomentumEngine._DIVERGENCE_LOOKBACK, min_periods=1).max().astype(bool)
        bull_trap_recent = MomentumEngine._col(df, 'bull_trap_confirmed', False).fillna(False).astype(bool) \
            .rolling(MomentumEngine._DIVERGENCE_LOOKBACK, min_periods=1).max().astype(bool)
        bullish_div_recent = MomentumEngine._col(df, 'bullish_divergence', False).fillna(False).astype(bool) \
            .rolling(MomentumEngine._DIVERGENCE_LOOKBACK, min_periods=1).max().astype(bool)
        bear_trap_recent = MomentumEngine._col(df, 'bear_trap_confirmed', False).fillna(False).astype(bool) \
            .rolling(MomentumEngine._DIVERGENCE_LOOKBACK, min_periods=1).max().astype(bool)

        defense_score = np.select(
            [bull_trap_recent, bearish_div_recent],
            [0, 8],
            default=20
        )

        total_score = np.clip(raw_score + defense_score, 0, 100)
        momentum_score = pd.Series(total_score, index=df.index).astype(float)

        # 年線未過關（yearline_pass 為 False）屬於結構性缺陷：即使其他層分數
        # 很高，也不應該被歸類為「飆股」，因此對總分做一次硬性封頂修正，
        # 確保「跌破年線或年線走弱」的標的永遠評不到 A 級（>=85）。
        momentum_score = momentum_score.where(yearline_pass, np.minimum(momentum_score, 65))

        df['momentum_score'] = momentum_score
        df['momentum_grade'] = np.select(
            [momentum_score >= 85, momentum_score >= 70, momentum_score >= 55],
            ["A", "B", "C"], default="D"
        )
        df['is_a_grade_candidate'] = df['momentum_grade'] == "A"

        # ⚠️ 修正說明：原本 trap_alert 把「會扣分的誘多警報」(bull_trap_recent /
        # bearish_div_recent) 跟「不影響本評分的底部反轉訊號」(bullish_div_recent /
        # bear_trap_recent，這兩個是看漲反轉訊號，對『飆股動能』評分而言是中性/
        # 甚至偏正面的資訊，defense_score 完全沒有把它們算進扣分) 全部 OR 在一起，
        # 導致只要單純出現底背離或誘空反轉這種「跟這次扣分無關」的訊號，
        # trap_alert 也會被設成 True。這樣一來：
        #   (a) get_momentum_breakdown 的「⑦ 誘多/背離防禦」會顯示「未通過」，
        #       但 defense_score 實際上是滿分 20，兩者互相矛盾；
        #   (b) app.py 顯示的「本層評分已扣分」文字，在這種情況下是錯的
        #       （分數根本沒被扣）。
        # 現在拆成兩個獨立欄位：
        #   momentum_penalty_alert：只反映「真的造成本層扣分」的訊號，
        #       用這個欄位判斷「⑦ 誘多/背離防禦」是否通過、以及是否該顯示
        #       「評分已扣分」。
        #   reversal_watch：純粹資訊性質的底部反轉觀察旗標（底背離／誘空
        #       確認），不影響飆股評分，但保留給想額外關注反轉訊號的使用者。
        #   trap_alert：維持原本「廣義警報雷達」用途（掃描戰情室的誘盤警報
        #       雷達本來就設計成不分方向、任何背離/假突破都想顯示），語意
        #       改為「近期市場結構出現值得留意的背離/假突破事件」，不再
        #       暗示「一定會扣分」。
        df['momentum_penalty_alert'] = bull_trap_recent | bearish_div_recent
        df['reversal_watch'] = bullish_div_recent | bear_trap_recent
        df['trap_alert'] = df['momentum_penalty_alert'] | df['reversal_watch']

        return df

    # ==========================================
    # 取得單一時間點（預設最新一筆）的七層評分明細
    # ==========================================
    @staticmethod
    def get_momentum_breakdown(df: pd.DataFrame, idx: int = -1):
        """回傳最新一筆的七層評分明細，供 Dashboard 顯示清單/雷達。"""
        if df is None or df.empty:
            return []

        row = df.iloc[idx]

        def gf(col, default=np.nan):
            try:
                val = row[col] if col in row.index else default
                return float(val)
            except (TypeError, ValueError):
                return default

        close = gf('close')
        sma200 = gf('sma_200')
        sma20, sma60, sma120 = gf('sma_20'), gf('sma_60'), gf('sma_120')
        rvol = gf('rvol')
        macd_hist = gf('macd_hist')
        rsi = gf('rsi_14')
        obv, obv_sma = gf('obv'), gf('obv_sma')

        yearline_ok = pd.notna(close) and pd.notna(sma200) and close > sma200
        alignment_ok = (pd.notna(close) and pd.notna(sma20) and pd.notna(sma60) and pd.notna(sma120)
                        and close > sma20 > sma60 > sma120)
        obv_ok = pd.notna(obv) and pd.notna(obv_sma) and obv > obv_sma

        # ⚠️ 修正：第⑦層的「通過與否」改用 momentum_penalty_alert（只反映真的
        # 會扣分的誘多假突破／頂背離），不再用籠統的 trap_alert（那個還包含
        # 不影響本層分數的底部反轉觀察訊號），避免顯示「未通過」但實際上
        # defense_score 是滿分的矛盾。
        penalty_active = bool(row.get('momentum_penalty_alert', row.get('trap_alert', False)))
        reversal_watch_active = bool(row.get('reversal_watch', False))
        note = row.get('trap_note', '') if 'trap_note' in row.index else ''
        if not note and penalty_active:
            note = row.get('divergence_note', '') if 'divergence_note' in row.index else ''
        if not note and penalty_active:
            # trap_alert 是「近 N 天內曾觸發」的滾動旗標，觸發當天才會寫入
            # trap_note/divergence_note 文字，往後幾天旗標仍為 True 但當天
            # 沒有新事件文字，這裡補一句通用說明，避免顯示「近期無警報」
            # 卻同時判定本層未過關的矛盾訊息。
            note = f"近期（{MomentumEngine._DIVERGENCE_LOOKBACK}天內）曾觸發誘多假突破/頂背離警報，本層評分已扣分"
        elif not note and reversal_watch_active:
            note = f"近期（{MomentumEngine._DIVERGENCE_LOOKBACK}天內）出現底背離/誘空反轉觀察訊號，不影響本層分數，僅供留意"
        elif not note:
            note = "近期無誘多/背離警報"

        breakdown = [
            {"layer": "① 年線多頭濾網", "passed": bool(yearline_ok),
             "detail": f"收盤 {close:.2f} vs 年線 {sma200:.2f}" if pd.notna(close) and pd.notna(sma200) else "資料不足"},
            {"layer": "② 均線多頭排列", "passed": bool(alignment_ok),
             "detail": "close > 20MA > 60MA > 120MA" if alignment_ok else "均線排列尚未完全多頭"},
            {"layer": "③ 價量齊揚", "passed": pd.notna(rvol) and rvol >= 1.3,
             "detail": f"RVOL={rvol:.2f}" if pd.notna(rvol) else "資料不足"},
            {"layer": "④ MACD動能強度", "passed": pd.notna(macd_hist) and macd_hist > 0,
             "detail": f"柱狀體={macd_hist:.3f}" if pd.notna(macd_hist) else "資料不足"},
            {"layer": "⑤ 相對強度未過熱", "passed": pd.notna(rsi) and 55 <= rsi <= 80,
             "detail": f"RSI={rsi:.1f}" if pd.notna(rsi) else "資料不足"},
            {"layer": "⑥ 籌碼慣性(OBV代理)", "passed": bool(obv_ok),
             "detail": "OBV在均線之上" if obv_ok else "OBV在均線之下或資料不足"},
            {"layer": "⑦ 誘多/背離防禦", "passed": not penalty_active,
             "detail": note},
        ]
        return breakdown