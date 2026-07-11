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

        return {
            "status": "ok",
            "weights_table": weights_table,
            "industry_breakdown": industry_breakdown,
            "total_allocated": total_allocated,
            "cash_remaining": cash_remaining,
            "note": note,
        }