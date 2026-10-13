# TQAI Pro Enterprise v2.9.13 — 升級紀錄

## 版本目標

在 v2.9.12 的資料品質、研究輸出一致性與決策覆蓋率基礎上，新增兩個重要控制：

1. **防止過期或尚未定案行情直接產生方向性可交易結論。**
2. **防止時間切分的前瞻報酬標籤跨入樣本外期間。**

## 修改檔案

- `app.py`
  - 版本標記更新為 v2.9.13。
  - 個股共識將最新行情日期與盤中估計標記傳入 `DecisionEngine`。
  - 顯示資料新鮮度閘門結果與行情日期。
- `engines/decision_engine.py`
  - 加入行情日期解析、年齡計算、未來日期偵測、過期檢查及未收盤 K 棒閘門。
  - 輸出 `market_data_date`、`market_data_age_days`、`market_data_freshness`、`market_data_trade_gate` 等可追蹤欄位。
- `engines/data_quality_engine.py`
  - 加入逐檔過期資料提醒、未來行情日期、必要欄位、重複 `ticker/date`、成交量缺漏檢查。
  - 報告中資料庫預設顯示相對路徑，提高封裝可攜性。
- `research/data_quality_report.py`
  - 報告增加逐檔新鮮度、重複行情鍵與必要欄位狀態。
- `research/phase_b_forward_return.py`
  - 每個 horizon 寫出 `label_end_date_{h}d`，用實際價格序列記錄標籤終點日期。
- `research/phase_d_alpha_validation.py`
  - 加入 purged out-of-sample；標籤終點跨 split 的訓練資料會被排除。
  - 加入 Benjamini–Hochberg FDR 校正、常數因子防護與缺少 horizon 欄位的明確錯誤回傳。
  - 在輸出中記錄 purge 數、方法及研究限制。
- `research/event_inventory.py`
  - 正確解析數字型及字串型 ETF 旗標，不把文字 `False` 誤判為 True。
- `tests/`、`research/test_*.py`
  - 新增行情新鮮度閘門、盤中估計閘門、逐檔過期資料、重複行情鍵、壞 schema、標籤終點日期、FDR 校正與樣本外 purge 等測試。
- `啟動TQAI.bat`、`執行測試.bat`
  - Windows 一鍵啟動/驗收腳本。

## 本次驗收結果

- `python -m pytest -q`：**89 passed**。
- `python -m compileall -q .`：通過。
- `python -m research.data_quality_report`：成功；目前包內資料狀態是 `stale`，因行情資料最新日期為 2026-07-31。
- 確認打包前後再次執行測試，確保交付 ZIP 本身而不只是工作目錄可通過測試。

## 尚未驗證項目

- 本地資料庫包含 12,227 筆、25 個 ticker；最晚行情日期 2026-07-31。
- `stock_directory` 與歷史大股東持股快照表仍為空。
- 突破事件 CSV 的 186 筆事件 ETF 身分仍全部未知。
- 沒有外部網路同步真實市場資料；未確認 TWSE/yfinance 即時資料源端到端成功。
- Golden regression 沒有凍結的 `tests/golden/raw/*.parquet` 作為比較基準。
- Streamlit 未在此建置環境啟動瀏覽器做互動測試。
- 89 項測試是程式邏輯與合成資料的驗證，不代表預測準確率、樣本外超額報酬或獲利能力已獲得證明。
