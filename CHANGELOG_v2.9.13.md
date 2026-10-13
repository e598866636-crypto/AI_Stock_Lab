# TQAI Pro Enterprise v2.9.13 — Changelog

## 核心升級：即時資料閘門、樣本外去洩漏、統計校正

- `DecisionEngine.build_consensus()` 新增逐標的行情日期檢查；行情日期缺失、無法解析、晚於檢查日、超過設定新鮮度門檻，或最新 K 棒為未收盤盤中估計時，決策不得輸出 `BUY READY` / `AVOID`，改為 `WATCH`。已確認的風險/流動性硬性否決仍優先回傳 `NO TRADE`。
- 個股深度分析把實際最新行情日期與 `is_intraday_estimate` 傳入決策層；UI 明確顯示日期、資料年齡及閘門原因。
- `DataQualityEngine` 現在會檢查每檔股票是否過期，即使資料庫全域最新日期看似新鮮也不再漏報；另新增未來日期、必要欄位缺失、`ticker/date` 重複鍵、成交量缺漏的檢查。
- 資料品質 JSON 預設使用專案相對資料庫路徑，避免在打包報告中保留建置環境的絕對路徑。
- Phase B `compute_forward_returns()` 新增 `label_end_date_{h}d`，記錄每個前瞻報酬標籤實際使用到的終點交易日。
- Phase D 樣本外切分新增 purging：訓練樣本的 forward-return 標籤終點若跨越 split date 即排除；舊 CSV 缺少標籤終點日期時，以 `2 × horizon` 曆日緩衝保守近似，並明確寫入報告。
- Phase D 增加各 horizon 的 Benjamini–Hochberg FDR 校正後 p-value；校正範圍只涵蓋本次單一因子的 horizons，不能宣稱不同因子/不同研究之間已完成全域多重比較校正。
- Phase D 對常數/無變異因子回傳 `degenerate_feature`，而不是以 NaN IC 假裝算出有效結果；若個別 horizon 的報酬欄位不存在，回傳 `unavailable` 而不是因 KeyError 中斷整個研究。
- ETF 身分對照不再使用危險的 `astype(bool)`；文字型 `"False"`、`"0"`、`"no"` 會正確解析為 False，未知值保留 None。
- 新增 Windows `啟動TQAI.bat` 與 `執行測試.bat`，降低初次安裝/測試操作錯誤。

## 驗證與限制

- 離線 pytest：89 passed。
- Python `compileall`：通過。
- `python -m research.data_quality_report`：可產生報告，包內資料狀態為 `stale`。
- 真實行情同步、Streamlit 瀏覽器啟動、Golden regression（缺少 frozen Parquet）及樣本外策略績效仍未驗證。
- FDR 校正不處理同一股票的事件群聚與重疊報酬窗口；研究報告會明確揭露這個限制。
