import numpy as np
import pandas as pd


class PortfolioEngine:
    """
    💼 投資組合建構引擎 (Portfolio Construction Engine) - TQAI Pro v2.9

    對應多因子決策文件「二十、AI 股票決策系統架構」裡的 PortfolioEngine：
    選股、資金配置與再平衡。

    ⚠️ 設計取捨與誠實揭露（避免功能包裝過度）：
      1. 這是「規則式」的資金配置建議（反波動度加權 + 產業集中度上限），
         不是嚴謹的均值-方差最佳化（Markowitz）或風險平價最佳化——那需要
         完整、穩定的共變異數矩陣估計，樣本數在台股中小型觀察名單上很
         容易估得不穩定/失真。規則式方法比較保守、透明，每一步怎麼算的
         都能講清楚，但不是「最優」資產配置，僅供參考起點。
      2. 這裡只是「建議權重」，不是實際下單指令：沒有考慮手續費對小額
         單筆交易的相對影響、台股以「張」（1000股）為最小交易單位可能
         導致實際持股比例跟建議權重有落差、稅務成本，以及使用者自身
         的風險承受度與既有部位，使用前務必自行覆核。
      3. 完全複用 ScannerEngine.scan() 已經算好的欄位（AI Score、年化
         波動率、產業），不需要任何新的外部資料源、不增加額外對外請求。
      4. 產業集中度上限用簡單的「等比例縮放 + 缺口重分配」近似解法
         （見 _apply_industry_cap 說明），不是精確的凸最佳化解，多數
         情況下 1~2 輪迭代就能收斂到接近上限，極端情況（例如候選股集中
         在單一產業超過總上限）就直接如實顯示「無法在此上限下分散」，
         不會硬湊出一個假裝合規的數字。
    """

    @staticmethod
    def _apply_industry_cap(weights: pd.Series, industries: pd.Series, max_industry_weight_pct: float,
                             max_iterations: int = 8) -> pd.Series:
        """
        反覆執行「超過上限的產業等比例縮到上限，把縮掉的權重按剩餘標的
        目前權重比例重新分配」，直到沒有產業超過上限，或達到迭代上限
        （極端情況下可能無法完全收斂，例如候選股全部集中在1個產業且
         上限設定不合理地低，這時就讓迴圈自然結束，回傳當下最接近的結果）。
        """
        w = weights.copy().astype(float)
        cap = max_industry_weight_pct / 100.0

        for _ in range(max_iterations):
            group_sum = w.groupby(industries).transform("sum")
            over_mask = group_sum > cap + 1e-9
            if not over_mask.any():
                break

            # 超過上限的產業，成員權重等比例縮到剛好等於上限
            scale = np.where(over_mask, cap / group_sum.replace(0, np.nan), 1.0)
            scale = pd.Series(scale, index=w.index).fillna(1.0)
            new_w = w * scale
            freed = w.sum() - new_w.sum()  # 這一輪縮減出來、需要重新分配的權重

            under_mask = ~over_mask
            under_total = new_w[under_mask].sum()
            if freed <= 1e-9 or under_total <= 1e-9:
                w = new_w
                break

            # 依「未超標標的」目前權重比例，把釋出的權重重新分配回去
            redistribute = (new_w[under_mask] / under_total) * freed
            new_w.loc[under_mask] = new_w.loc[under_mask] + redistribute
            w = new_w

        # 數值誤差修正，確保最終總和仍是100%
        total = w.sum()
        if total > 0:
            w = w / total
        return w

    @staticmethod
    def build_portfolio(
        result_df: pd.DataFrame,
        top_n: int = 10,
        min_ai_score: float = 70,
        max_industry_weight_pct: float = 30.0,
        capital: float = 1_000_000,
        check_correlation: bool = True,
    ) -> dict:
        """
        參數：
            result_df               ScannerEngine.scan() 的回傳結果（需含
                                     代碼/標的/產業/AI Score/收盤價欄位；
                                     「年化波動率」若存在會用來做反波動度
                                     加權，缺失則該股退化為等權重）
            top_n                   最多選幾檔進投資組合
            min_ai_score            候選門檻：AI Score 需達到此分數才會被
                                     考慮進投資組合
            max_industry_weight_pct 單一產業的權重上限（%），避免整個組合
                                     過度集中在單一族群
            capital                 預計投入的總資金（新台幣），用來換算
                                     建議金額與建議張數（台股以1張=1000股
                                     為交易單位）
            check_correlation       是否額外計算候選股之間的歷史報酬相關性
                                     （見 _check_correlation_concentration()
                                     說明）。預設開啟；會需要額外抓取每檔
                                     候選股的歷史股價（通常已有快取），若
                                     想要更快的回應可以關閉。

        回傳：
            {
                'status': 'ok' / 'empty',
                'weights_table': DataFrame（代碼/標的/產業/AI Score/年化波動率/
                                  建議權重%/建議金額/建議張數/剩餘零股金額）,
                'industry_breakdown': DataFrame（產業別合計權重%，用來檢查
                                  是否真的有壓在集中度上限內）,
                'total_allocated': 實際配置金額合計,
                'cash_remaining': 因為台股以整張為單位、無條件捨去零股後
                                  剩餘未配置的金額,
                'note': 使用限制與方法論的簡短提醒,
            }
        """
        empty_result = {
            "status": "empty",
            "weights_table": pd.DataFrame(),
            "industry_breakdown": pd.DataFrame(),
            "total_allocated": 0.0,
            "cash_remaining": capital,
            "note": "候選名單為空（可能是掃描結果為空，或沒有股票達到 min_ai_score 門檻）。",
        }

        if result_df is None or result_df.empty:
            return empty_result

        required_cols = {"代碼", "標的", "產業", "AI Score", "收盤價"}
        missing = required_cols - set(result_df.columns)
        if missing:
            empty_result["note"] = f"⚠️ 缺少必要欄位：{', '.join(missing)}，請確認傳入的是 ScannerEngine.scan() 的回傳結果。"
            return empty_result

        candidates = result_df[result_df["AI Score"] >= min_ai_score].copy()
        candidates = candidates.sort_values("AI Score", ascending=False).head(top_n).reset_index(drop=True)
        if candidates.empty:
            return empty_result

        # 反波動度加權：波動率越低，基礎權重越高（風險平價的簡化近似）。
        # 缺 "年化波動率" 欄位或值為0/NaN 的股票，用候選組合的波動率中位數
        # 頂替，避免單一缺值股票因為除以極小值/NaN而拿到不合理的極端權重。
        if "年化波動率" in candidates.columns:
            vol = pd.to_numeric(candidates["年化波動率"], errors="coerce")
        else:
            vol = pd.Series(np.nan, index=candidates.index)

        median_vol = vol.median()
        fallback_vol = median_vol if pd.notna(median_vol) and median_vol > 0 else 20.0
        vol = vol.fillna(fallback_vol)
        vol = vol.where(vol > 0, fallback_vol)

        raw_weight = 1.0 / vol
        base_weight = raw_weight / raw_weight.sum()

        capped_weight = PortfolioEngine._apply_industry_cap(
            base_weight, candidates["產業"], max_industry_weight_pct
        )

        candidates["建議權重%"] = (capped_weight * 100).round(2)
        candidates["建議金額"] = (capped_weight * capital).round(0)

        # 台股以1張(1000股)為最小交易單位，無條件捨去零股，
        # 剩餘的零股金額如實列出（不假裝可以精確買到建議權重）。
        close = pd.to_numeric(candidates["收盤價"], errors="coerce")
        lot_value = close * 1000
        candidates["建議張數"] = np.where(
            (lot_value > 0) & pd.notna(lot_value),
            np.floor(candidates["建議金額"] / lot_value.replace(0, np.nan)),
            0,
        ).astype(int)
        candidates["實際配置金額"] = candidates["建議張數"] * lot_value.fillna(0)
        candidates["零股剩餘金額"] = (candidates["建議金額"] - candidates["實際配置金額"]).round(0)

        show_cols = [c for c in [
            "代碼", "標的", "產業", "AI Score", "年化波動率",
            "建議權重%", "建議金額", "建議張數", "實際配置金額", "零股剩餘金額",
        ] if c in candidates.columns]
        weights_table = candidates[show_cols].sort_values("建議權重%", ascending=False).reset_index(drop=True)

        industry_breakdown = (
            candidates.groupby("產業")["建議權重%"].sum().round(2)
            .reset_index().rename(columns={"建議權重%": "產業合計權重%"})
            .sort_values("產業合計權重%", ascending=False).reset_index(drop=True)
        )

        total_allocated = float(weights_table["實際配置金額"].sum())
        cash_remaining = float(capital - total_allocated)

        note = (
            "⚠️ 這是規則式（反波動度加權＋產業集中度上限）的資金配置建議，"
            "不是嚴謹的最佳化結果，也不是下單指令；台股以1張=1000股為單位，"
            "實際配置金額與建議權重會有零股落差，下單前請自行覆核。"
        )

        # ⚠️ 誠實揭露邊界情況：若候選股集中在過少的產業（例如只有3個產業卻
        # 都設 30% 上限，數學上限只有90% < 100%），迭代式縮放/重分配無法讓
        # 每個產業都真正壓在上限內——與其讓迴圈跑完後默默回傳一個實際上
        # 超過上限的表格，這裡明確檢查並告知使用者「這次沒有完全壓在
        # 上限內」，而不是假裝合規。
        max_industry_actual = float(industry_breakdown["產業合計權重%"].max()) if not industry_breakdown.empty else 0.0
        if max_industry_actual > max_industry_weight_pct + 0.5:
            note += (
                f" ⚠️ 候選股的產業集中度過高（{len(industry_breakdown)}個產業 × "
                f"{max_industry_weight_pct}%上限，數學上限僅"
                f"{min(len(industry_breakdown) * max_industry_weight_pct, 100):.0f}%，不足以覆蓋100%資金），"
                f"目前最高的產業實際權重為 {max_industry_actual:.1f}%，無法在此上限下完全分散，"
                f"建議放寬 max_industry_weight_pct 或擴大候選股的產業多樣性。"
            )

        correlation_check = {"status": "unavailable", "message": "未執行相關性檢查（check_correlation=False）。"}
        if check_correlation:
            try:
                correlation_check = PortfolioEngine._check_correlation_concentration(
                    weights_table["代碼"].tolist()
                )
            except Exception as e:
                correlation_check = {"status": "unavailable", "message": f"相關性檢查時發生錯誤：{e}"}

        return {
            "status": "ok",
            "weights_table": weights_table,
            "industry_breakdown": industry_breakdown,
            "total_allocated": total_allocated,
            "cash_remaining": cash_remaining,
            "note": note,
            "correlation_check": correlation_check,
        }

    # ==========================================
    # 相關性集中風險檢查（v2.9.3 新增，專業風控觀點）
    # ==========================================
    @staticmethod
    def _check_correlation_concentration(codes: list, use_cache: bool = True, corr_threshold: float = 0.7) -> dict:
        """
        ⚠️ 專業風控觀點：產業集中度上限只防得住「掛在同一個官方產業分類」的
        集中風險，防不住「不同產業分類、但實際上齊漲齊跌」的相關性風險——
        例如半導體設備廠跟IC設計廠官方產業分類不同，卻常常同步反應同一個
        總經事件（例如 AI 需求變化、地緣政治），真實的風險分散程度可能比
        「產業別有幾種」這個數字所暗示的更差。這裡額外計算候選股歷史報酬
        的兩兩相關係數，平均相關係數過高時額外示警，這是產業集中度上限
        機制本身無法涵蓋的風險維度，概念上類似量化風控實務裡「有效分散
        股數 (effective number of bets)」的簡化版本。

        ⚠️ 誠實揭露：
          1. 只用歷史相關係數，相關性本身會隨時間改變——尤其市場壓力期間
             相關係數常常系統性上升（「你最需要分散的時候，分散效果反而
             最差」是資產配置實務裡的常見現象），這裡的歷史相關係數不保證
             反映未來、尤其是市場壓力時期的真實相關性。
          2. 需要額外抓取每檔候選股的歷史股價（通常已有 DataEngine 快取，
             不會大幅增加額外請求），若抓取失敗會直接回傳 unavailable，
             不影響 build_portfolio() 主流程——這是加分的風險提示，
             不是決定投資組合是否成立的必要條件。
        """
        from engines.data_engine import DataEngine

        if len(codes) < 2:
            return {"status": "unavailable", "message": "候選股不足2檔，無法計算相關性。"}

        returns_data = {}
        for code in codes:
            try:
                df = DataEngine.get_stock_data(code, use_cache=use_cache)
                if df is None or df.empty or "close" not in df.columns or "date" not in df.columns:
                    continue
                s = df.set_index("date")["close"].astype(float).pct_change().dropna()
                if len(s) >= 20:
                    returns_data[code] = s
            except Exception:
                continue

        if len(returns_data) < 2:
            return {"status": "unavailable", "message": "有效歷史報酬資料不足2檔，無法計算相關性。"}

        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr()

        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        pairwise_corrs = corr_matrix.where(mask).stack()

        if pairwise_corrs.empty:
            return {"status": "unavailable", "message": "相關係數計算結果為空（可能是樣本重疊天數不足）。"}

        avg_corr = float(pairwise_corrs.mean())
        max_corr_pair = pairwise_corrs.idxmax()
        max_corr_val = float(pairwise_corrs.max())

        flags = []
        if avg_corr >= corr_threshold:
            flags.append(
                f"🔴 候選股之間歷史報酬平均相關係數達 {avg_corr:.2f}（門檻{corr_threshold}），"
                f"即使產業標籤不同，實際漲跌可能高度同步，真實分散效果比表面上的產業數量更差。"
            )
        elif avg_corr >= corr_threshold - 0.2:
            flags.append(f"🟡 候選股之間歷史報酬平均相關係數 {avg_corr:.2f}，中等偏高，建議留意。")
        else:
            flags.append(f"✅ 候選股之間歷史報酬平均相關係數 {avg_corr:.2f}，相關性尚屬合理範圍。")

        flags.append(f"相關性最高的一對：{max_corr_pair[0]} 與 {max_corr_pair[1]}（相關係數 {max_corr_val:.2f}）")
        flags.append("⚠️ 歷史相關係數不保證反映未來，尤其市場壓力期間相關性常常系統性上升，僅供參考，不構成投資建議。")

        return {
            "status": "ok",
            "avg_correlation": round(avg_corr, 2),
            "max_correlation_pair": list(max_corr_pair),
            "max_correlation": round(max_corr_val, 2),
            "flags": flags,
        }

    # ==========================================
    # 再平衡 (Rebalancing) — v2.9.1 新增，補齊原本缺的部分
    # ==========================================
    @staticmethod
    def build_rebalance_plan(current_holdings: dict, target_weights_table: pd.DataFrame,
                              min_adjust_lots: int = 1) -> dict:
        """
        再平衡計畫：比較「目前實際持股」與 build_portfolio() 算出來的
        「目標配置」，計算需要加碼/減碼哪些股票、大約幾張，才能讓實際
        持股貼近目標權重。

        參數：
            current_holdings     {'2330': 張數, '2317': 張數, ...}，使用者
                                  手動輸入目前實際持有的股票代碼與張數
                                  （整數，1張=1000股）
            target_weights_table build_portfolio() 回傳的 weights_table
                                  （需含 代碼/標的/建議張數 欄位）

        ⚠️ 誠實揭露：
          1. 這是「靜態快照比較」，不是動態最佳化再平衡——不考慮交易
             成本最小化、稅務影響（例如短期頻繁調整可能墊高證交稅與
             手續費占比）、或零股/部分成交等實務限制。
          2. 目前持有但「不在」這次目標配置名單裡的股票，會被列為
             「建議全數賣出」——這只代表這檔股票這次沒有通過選股門檻，
             不代表這檔股票基本面變差。如果你有其他理由想繼續持有
             （例如長期存股、還沒到你自己設定的停利停損點），請自行
             判斷是否保留，不要照單全收，本功能不構成投資建議。
          3. 統一以「張」為單位（1張=1000股），沒有處理零股。
          4. min_adjust_lots（預設1張）：調整幅度小於這個門檻的標的會
             標示「維持不變」，避免為了1、2張的微小差距頻繁交易、徒增
             手續費占比。
        """
        if not current_holdings:
            return {"status": "empty", "note": "尚未輸入目前持股，無法計算再平衡建議。"}

        if target_weights_table is None or target_weights_table.empty:
            return {"status": "empty", "note": "目標投資組合為空，請先建立投資組合建議，再計算再平衡。"}

        if "代碼" not in target_weights_table.columns or "建議張數" not in target_weights_table.columns:
            return {"status": "empty", "note": "⚠️ 目標投資組合缺少必要欄位，請確認傳入的是 build_portfolio() 的 weights_table。"}

        target_map = dict(zip(target_weights_table["代碼"].astype(str), target_weights_table["建議張數"]))
        tag_map = dict(zip(target_weights_table["代碼"].astype(str), target_weights_table.get("標的", target_weights_table["代碼"])))

        current_holdings = {str(k).strip(): int(v) for k, v in current_holdings.items() if str(k).strip()}
        all_codes = set(current_holdings.keys()) | set(target_map.keys())

        rows = []
        for code in sorted(all_codes):
            current_lots = int(current_holdings.get(code, 0))
            target_lots = int(target_map.get(code, 0))
            delta = target_lots - current_lots

            if abs(delta) < max(min_adjust_lots, 1):
                action = "➖ 維持不變（差距在門檻內）"
            elif delta > 0:
                action = f"🟢 買進 {delta} 張"
            elif delta < 0:
                action = f"🔴 賣出 {abs(delta)} 張"
            else:
                action = "➖ 不變"

            rows.append({
                "代碼": code,
                "標的": tag_map.get(code, f"[{code}]"),
                "目前張數": current_lots,
                "目標張數": target_lots,
                "調整張數": delta,
                "動作": action,
            })

        result_df = pd.DataFrame(rows)
        result_df["_abs_delta"] = result_df["調整張數"].abs()
        result_df = result_df.sort_values("_abs_delta", ascending=False).drop(columns="_abs_delta").reset_index(drop=True)

        exit_only = result_df[(result_df["目標張數"] == 0) & (result_df["目前張數"] > 0)]

        note = (
            "⚠️ 這是靜態快照比較，不是動態最佳化再平衡，沒有考慮交易成本或稅務影響；"
            "不在這次目標配置名單裡的持股會建議全數賣出——不代表基本面變差，"
            "如果你有其他理由想繼續持有，請自行判斷是否保留，不構成投資建議。"
        )
        if not exit_only.empty:
            note += f" 有 {len(exit_only)} 檔目前持有但這次未入選目標名單：{', '.join(exit_only['代碼'].tolist())}。"

        return {
            "status": "ok",
            "rebalance_table": result_df,
            "note": note,
        }
