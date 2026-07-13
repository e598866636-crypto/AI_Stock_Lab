import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as fgo
from engines.data_engine import DataEngine
from engines.indicator_engine import IndicatorEngine
from engines.structure_engine import StructureEngine
from engines.divergence_engine import DivergenceEngine
from engines.strategy_engine import StrategyEngine
from engines.momentum_engine import MomentumEngine
from engines.evidence_engine import EvidenceEngine
from engines.risk_engine import RiskEngine
from engines.backtest_engine import BacktestEngine
from engines.db_engine import DatabaseEngine
from engines.scanner_engine import ScannerEngine
from engines.chip_engine import ChipEngine
from engines.industry_engine import IndustryEngine
from engines.name_engine import NameEngine
from engines.timeframe_engine import TimeframeEngine
from engines.fundamental_engine import FundamentalEngine
from engines.stock_academy_engine import StockAcademyEngine
from engines.stock_directory_engine import StockDirectoryEngine
from engines.macro_engine import MacroEngine
from engines.options_engine import OptionsEngine
from engines.sector_rotation_engine import SectorRotationEngine
from engines.portfolio_engine import PortfolioEngine
from engines.seasonality_engine import SeasonalityEngine

# 設置企業級寬螢幕版面
st.set_page_config(layout="wide", page_title="TQAI Pro Enterprise v2.5", page_icon="🏦")


def _resolve_stock_search(raw_input: str):
    """
    ⚠️ 新增功能：查找股票支援輸入中文名稱（不再只能輸入代碼）。

    解析規則：
      1. 輸入本身若符合台股代碼格式（含槓桿/反向ETF字母尾碼，見
         DataEngine.is_tw_code）→ 直接當代碼使用，不做名稱搜尋。
      2. 否則視為名稱搜尋：先查 NameEngine 內建觀察名單（快、不需要
         資料庫或網路），再查 StockDirectoryEngine 的全市場快取（需要
         先執行過 refresh_all() 才有資料，涵蓋範圍更廣，包含觀察名單
         以外的股票）。兩邊結果依代碼去重合併。

    回傳：(resolved_code, candidates, error_message)
      - 剛好一筆符合 → resolved_code 有值，其餘為 None
      - 找到多筆 → candidates 為 list，需要使用者從下拉選單挑選
      - 一筆都找不到 → error_message 說明原因
    """
    raw = str(raw_input).strip()
    if not raw:
        return None, None, "請輸入股票代碼或名稱"

    if DataEngine.is_tw_code(raw):
        return raw, None, None

    matches = list(NameEngine.search_by_name(raw))
    seen_codes = {m["code"] for m in matches}

    try:
        dir_df = StockDirectoryEngine.search_by_name(raw)
        for _, row in dir_df.iterrows():
            code = str(row["code"])
            if code not in seen_codes:
                matches.append({
                    "code": code,
                    "name": row.get("name", ""),
                    "market": row.get("market", ""),
                })
                seen_codes.add(code)
    except Exception:
        pass  # 全市場目錄尚未建立或查詢失敗，不影響內建觀察名單的搜尋結果

    if len(matches) == 1:
        return matches[0]["code"], None, None
    elif len(matches) > 1:
        return None, matches, None
    else:
        return None, None, (
            f"⚠️ 查無名稱包含「{raw}」的股票。可能是：(1) 全市場名稱目錄尚未建立"
            f"（需先執行過 StockDirectoryEngine.refresh_all()），只能搜尋到內建觀察"
            f"名單裡的股票；或 (2) 名稱打錯字。請確認名稱，或改用代碼查詢。"
        )


def _resolve_custom_scan_list(raw_list_str: str):
    """
    ⚠️ 新增功能：「全台股掃描」的自訂清單原本只吃逗號分隔的代碼
    （custom_list.split(",")），輸入中文名稱一律查無資料。現在跟個股搜尋框
    共用同一套解析邏輯（見上方 _resolve_stock_search），逗號分隔的每一項都
    可以是代碼或中文名稱，兩者也可以混用（例如 "2330,鴻海,00631L"）。

    批次輸入的情境跟單一搜尋框不同——沒有版面可以讓使用者針對每個模糊的
    項目跳出下拉選單挑選，所以規則是：
      1. 符合台股代碼格式 → 直接當代碼使用。
      2. 名稱剛好解析出唯一一檔 → 自動轉成代碼。
      3. 名稱解析出多筆候選 → 不猜測，直接跳過這一項，並回傳警告訊息列出
         候選清單，讓使用者自己改成更精確的名稱或直接填代碼，避免掃描到
         使用者沒有意圖要看的股票。
      4. 完全查無 → 跳過這一項並回傳警告。

    回傳 (resolved_tickers: list[str], warnings: list[str])
    """
    items = [x.strip() for x in str(raw_list_str).split(",") if x.strip()]
    resolved = []
    warnings = []
    for item in items:
        if DataEngine.is_tw_code(item):
            resolved.append(item)
            continue

        code, candidates, _err = _resolve_stock_search(item)
        if code:
            resolved.append(code)
        elif candidates:
            shown = candidates[:5]
            option_str = "、".join(f"[{c['code']}]{c['name']}" for c in shown)
            more = f" 等共 {len(candidates)} 筆" if len(candidates) > 5 else ""
            warnings.append(f"⚠️「{item}」找到多筆符合的股票（{option_str}{more}），為避免誤判已跳過此項，請改用代碼或更精確的名稱。")
        else:
            warnings.append(f"⚠️ 查無「{item}」對應的股票，已跳過此項。")

    return resolved, warnings



with st.sidebar:
    st.header("⚙️ TQAI 決策中樞")
    mode = st.radio("功能選擇", ["🔍 個股深度分析", "📡 全台股掃描", "🏆 全市場排行榜", "🌍 總經戰情室"])
    st.markdown("---")
    
    if mode == "🔍 個股深度分析":
        search_input = st.text_input(
            "輸入股票代碼或中文名稱 (例: 2330 或 台積電)", value="2330"
        )

        # ⚠️ 新增：中文名稱搜尋解析（見上方 _resolve_stock_search 說明）。
        # 這裡在每次腳本重跑時都重新解析（Streamlit 的標準模式），不需要
        # 額外的 session_state 狀態機，行為簡單且不會有過期快取的問題。
        ticker, name_candidates, resolve_error = _resolve_stock_search(search_input)

        if name_candidates:
            option_labels = [
                f"[{c['code']}] {c['name']}" + (f"（{c['market']}）" if c.get('market') else "")
                for c in name_candidates
            ]
            chosen_label = st.selectbox(
                f"🔎「{search_input.strip()}」找到 {len(name_candidates)} 筆符合的股票，請選擇：",
                option_labels,
            )
            ticker = name_candidates[option_labels.index(chosen_label)]["code"]
        elif resolve_error:
            st.warning(resolve_error)
        elif ticker and not str(search_input).strip().isdigit():
            # 純數字輸入不需要提示（使用者本來就是打代碼），但用中文名稱
            # 解析出唯一結果時，顯示解析結果讓使用者確認打的是對的股票。
            resolved_name = NameEngine.get_name(ticker)
            st.caption(f"🔎 已將「{search_input.strip()}」解析為 [{ticker}] {resolved_name}")

        run_btn = st.button(
            "🚀 啟動 AI 多智能體分析", use_container_width=True, type="primary",
            disabled=(not ticker),
        )
        use_cache = st.checkbox("🗄️ 使用資料庫快取 (建議開啟)", value=True)
        force_refresh = st.button("🔄 強制重新抓取最新資料", use_container_width=True, disabled=(not ticker))
        scan_btn = False
        rank_btn = False
        macro_btn = False
    elif mode == "📡 全台股掃描":
        st.caption("預設使用台股常見權值股／熱門股觀察名單，也可以自行輸入想掃描的股票代碼。")
        custom_list = st.text_area(
            "自訂股票清單（逗號分隔，可混用代碼與中文名稱，留空則使用預設清單）",
            value="", placeholder="例如：2330,鴻海,00631L"
        )
        top_n = st.slider("顯示前 N 名", min_value=5, max_value=30, value=10)
        use_cache_scan = st.checkbox("🗄️ 使用資料庫快取 (建議開啟)", value=True, key="scan_cache")
        scan_btn = st.button("📡 啟動全台股掃描", use_container_width=True, type="primary")
        run_btn = False
        force_refresh = False
        rank_btn = False
        macro_btn = False
    elif mode == "🏆 全市場排行榜":
        st.caption("抓取當日（或最近交易日）全市場三大法人買賣超排行榜，僅涵蓋上市股票。")
        rank_top_n = st.slider("每類別顯示前 N 名", min_value=5, max_value=50, value=20, key="rank_top_n")
        rank_btn = st.button("🏆 抓取最新排行榜", use_container_width=True, type="primary")
        run_btn = False
        scan_btn = False
        force_refresh = False
        macro_btn = False
    else:
        st.caption(
            "🌍 總經戰情室：抓取美元指數／美元台幣／VIX／黃金／原油／美國十年期公債殖利率／"
            "那斯達克／標普500／費半(SOX)／台灣加權指數，作為判斷大盤環境的背景參考。"
        )
        st.caption("⚠️ 這裡的訊號是常見經驗法則的方向性參考，不是嚴謹統計檢定，也不會用來計算任何個股的 AI Score。")
        macro_btn = st.button("🌍 抓取最新總經數據", use_container_width=True, type="primary")
        run_btn = False
        scan_btn = False
        rank_btn = False
        force_refresh = False
    
    st.markdown("---")
    try:
        db_stats = DatabaseEngine.get_db_stats()
        st.caption(f"📦 快取股票數: {db_stats['cached_tickers']} 檔 ／ {db_stats['total_rows']} 筆K線")
    except Exception:
        st.caption("📦 資料庫尚未初始化（首次查詢後會自動建立）")
    
    st.markdown("---")
    st.caption("架構版本: TQAI Pro v2.5 (Bridge)")
    st.caption("核心引擎: 13-Layer Multi-Agent")

if mode == "🔍 個股深度分析" and force_refresh and ticker:
    try:
        DatabaseEngine.clear_cache(str(ticker).strip())
        st.sidebar.success(f"已清除 {ticker} 的快取，請重新點擊分析按鈕")
    except Exception:
        pass

# ==========================================
# 📡 全台股掃描模式
# ==========================================
# ⚠️ 修正說明：原本整個結果顯示區塊都包在 `if scan_btn:` 底下。st.button() 的
# True 狀態只維持「按下當下觸發的那一次 rerun」，使用者接下來只要跟頁面上
# 任何其他元件互動（例如下面的產業下拉選單），Streamlit 就會重新執行整個
# script，這時候 scan_btn 已經變回 False，整個 if 區塊（包含下拉選單本身
# 依賴的資料）就不會被渲染，導致「切換產業看不到結果」。
# 修正做法：按下按鈕時把掃描結果寫進 st.session_state，顯示邏輯改成
# 「只要 session_state 裡有資料就渲染」，不再綁定按鈕的瞬間狀態。
if mode == "📡 全台股掃描":
    if scan_btn:
        if custom_list.strip():
            scan_tickers, resolve_warnings = _resolve_custom_scan_list(custom_list)
            for w in resolve_warnings:
                st.warning(w)
            if not scan_tickers:
                st.error("⚠️ 自訂清單解析後沒有任何有效的股票代碼，請確認輸入內容，或留空使用預設清單。")
        else:
            scan_tickers = ScannerEngine.DEFAULT_WATCHLIST

        if scan_tickers:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _update_progress(done, total, current_ticker):
                progress_bar.progress(done / total)
                status_text.caption(f"掃描中：{current_ticker} ({done}/{total})")

            result_df, error_df = ScannerEngine.scan(
                tickers=scan_tickers, use_cache=use_cache_scan, progress_callback=_update_progress
            )

            progress_bar.empty()
            status_text.empty()

            st.session_state["scan_result_df"] = result_df
            st.session_state["scan_error_df"] = error_df
            st.session_state["scan_ticker_count"] = len(scan_tickers)

    if "scan_result_df" in st.session_state:
        result_df = st.session_state["scan_result_df"]
        error_df = st.session_state["scan_error_df"]

        st.markdown(f"## 📡 全台股掃描戰情室 (共 {st.session_state['scan_ticker_count']} 檔)")

        if result_df.empty:
            st.error("掃描失敗，所有股票皆無法取得資料，請確認代碼是否正確或網路連線。")
        else:
            st.success(f"✅ 掃描完成，成功 {len(result_df)} 檔，失敗 {len(error_df)} 檔。")

            st.markdown(f"### 🏆 Top {top_n} 排行榜（依 AI Score 排序）")
            top_df = ScannerEngine.get_top_n(result_df, top_n)
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 🚀 A級飆股候選 (Momentum A-Grade Candidates)")
            st.caption("依「飆股七層過濾＋100分評分系統（含年線濾網）」篩選，總分需 ≥ 85 分且已站穩年線才會列為 A 級。")
            a_grade_df = ScannerEngine.get_a_grade_candidates(result_df)
            if a_grade_df.empty:
                st.caption("目前掃描名單中沒有符合 A 級門檻的標的。")
            else:
                # 版本相容性保護：只顯示目前 result_df 實際存在的欄位，避免
                # scanner_engine.py 版本沒對齊（例如缺少新版才有的短線/波段/
                # 長線/未來走向欄位）時整頁直接 KeyError 崩潰。
                preferred_cols = ["排名", "標的", "市場別", "收盤價", "飆股評分", "飆股等級", "AI Score",
                                  "買進訊號", "誘盤警報", "操作建議", "短線建議", "波段建議", "長線建議", "未來走向"]
                show_cols = [c for c in preferred_cols if c in a_grade_df.columns]
                missing_cols = [c for c in preferred_cols if c not in a_grade_df.columns]
                if missing_cols:
                    st.caption(f"⚠️ 偵測到 scanner_engine.py 版本較舊，缺少欄位：{', '.join(missing_cols)}（請確認已更新 engines/scanner_engine.py 與 engines/timeframe_engine.py 並重啟服務）")
                st.dataframe(a_grade_df[show_cols], use_container_width=True, hide_index=True)
                st.caption("⚠️ A級代表『當下技術結構符合飆股七層過濾條件』，不代表保證持續噴出，仍請留意下方誘盤警報雷達與個股風險。")

            st.markdown("---")
            st.markdown("### 🛡️ 誘盤警報雷達 (Fake-Signal / Trap Radar)")
            st.caption("彙整近期觸發 MACD 背離或假突破/假跌破（誘多/誘空）警報的標的，依 AI Score 排序。")
            trap_df = ScannerEngine.get_trap_alerts(result_df)
            if trap_df.empty:
                st.caption("✅ 掃描名單中目前無標的觸發背離或假突破/假跌破警報。")
            else:
                preferred_cols_trap = ["排名", "標的", "市場別", "收盤價", "AI Score", "飆股評分",
                                       "誘盤警報", "賣出訊號", "操作建議", "短線建議", "波段建議", "未來走向"]
                show_cols_trap = [c for c in preferred_cols_trap if c in trap_df.columns]
                missing_cols_trap = [c for c in preferred_cols_trap if c not in trap_df.columns]
                if missing_cols_trap:
                    st.caption(f"⚠️ 偵測到 scanner_engine.py 版本較舊，缺少欄位：{', '.join(missing_cols_trap)}（請確認已更新 engines/scanner_engine.py 與 engines/timeframe_engine.py 並重啟服務）")
                st.dataframe(trap_df[show_cols_trap], use_container_width=True, hide_index=True)
                st.caption("⚠️ 誘盤警報代表近期偵測到MACD背離或價格假突破/假跌破收回，動能可能已經或即將失效，請留意追高/殺低風險。")

            st.markdown("---")
            st.markdown("### 🏭 產業中心 (Industry Center) — 產業強弱排名")
            industry_df = IndustryEngine.rank_industries(result_df)
            if industry_df.empty:
                st.caption("⚠️ 尚無法產生產業排名（掃描結果為空，或股票代碼皆不在預設產業對照表中）。")
            else:
                st.dataframe(industry_df, use_container_width=True, hide_index=True)

                industry_options = industry_df["產業"].tolist()
                selected_industry = st.selectbox(
                    "🔍 點選查看該產業內的成分股表現", industry_options, key="industry_select"
                )
                if selected_industry:
                    constituents = IndustryEngine.get_industry_constituents(result_df, selected_industry)
                    st.dataframe(constituents, use_container_width=True, hide_index=True)

                st.markdown("#### 🔁 產業輪動觀察（時間序列版）")
                st.caption(
                    "⚠️ 上面的產業排名是「這次掃描當下」的單一快照；這裡改用本次掃描名單裡每檔股票"
                    "的歷史股價，計算「等權重」平均報酬曲線，觀察最近5日/20日/60日哪個產業動能"
                    "轉強或轉弱。這不是市值加權的真正產業指數，樣本也僅限本次掃描名單，僅供參考。"
                )
                rotation_btn = st.button("🔁 計算產業輪動（讀取歷史股價）", key="rotation_btn")
                if rotation_btn:
                    with st.spinner("計算產業輪動中..."):
                        try:
                            scan_codes = result_df["代碼"].astype(str).tolist()
                            st.session_state["rotation_report"] = SectorRotationEngine.build_rotation_report(
                                tickers=scan_codes, use_cache=use_cache_scan
                            )
                        except Exception as e:
                            st.session_state["rotation_report"] = {"status": "unavailable", "error": str(e)}

                rotation_report = st.session_state.get("rotation_report")
                if rotation_report and rotation_report.get("status") == "ok":
                    st.dataframe(rotation_report["rotation_table"], use_container_width=True, hide_index=True)
                    st.line_chart(rotation_report["return_curves"])
                    if rotation_report.get("failed_tickers"):
                        st.caption(f"⚠️ 以下股票歷史股價抓取失敗，未列入產業輪動計算：{', '.join(rotation_report['failed_tickers'])}")
                elif rotation_report and rotation_report.get("status") == "unavailable":
                    st.caption("⚠️ 目前資料不足以計算產業輪動（可能是掃描名單股票太少、都是ETF/未分類，或歷史資料不足）。")

            st.markdown("---")
            st.markdown("### 🎓 選股大師 Top 候選 (五維度評分，快速模式)")
            st.caption("⚠️ 批次掃描為了效能，刻意不對每檔股票額外呼叫籌碼/基本面外部資料源，籌碼面改用OBV/量能代理、基本面與財務面顯示中性分數，僅供初步篩選；決定下單前建議點進「個股深度分析」查看該股完整的五維度評分（會自動帶入真實籌碼與基本面資料）。")
            academy_top_df = ScannerEngine.get_academy_top_n(result_df, n=top_n, min_grade="B")
            if academy_top_df.empty:
                st.caption("目前掃描名單中沒有符合 B 級以上門檻的標的（快速模式下基本面/財務面為中性分數，門檻天生較難達到）。")
            else:
                preferred_cols_ac = ["排名", "標的", "市場別", "收盤價", "選股評級", "選股評分",
                                     "市場面評分", "基本面評分", "技術面評分", "籌碼面評分", "財務面評分", "AI Score", "飆股等級"]
                show_cols_ac = [c for c in preferred_cols_ac if c in academy_top_df.columns]
                st.dataframe(academy_top_df[show_cols_ac], use_container_width=True, hide_index=True)

            dim_weakest = ScannerEngine.get_dimension_weakest(result_df)
            if dim_weakest:
                st.caption(f"📊 本次掃描名單維度分析：最弱維度為「{dim_weakest['最弱維度']}」(平均 {dim_weakest['最弱維度平均分']} 分)，最強維度為「{dim_weakest['最強維度']}」(平均 {dim_weakest['最強維度平均分']} 分)。")

            st.markdown("---")
            st.markdown("### ⚡ 三信號共識 (AI Score + 飆股評分 + 選股評級 三者皆看好)")
            st.caption("三套評分系統分別回答短期/動能/中長期三個不同時間尺度的問題，三者同時共識代表短中長期角度一致，非保證獲利。")
            consensus_df = ScannerEngine.get_multi_signal_consensus(result_df)
            if consensus_df.empty:
                st.caption("目前掃描名單中沒有三個信號同時共識看好的標的。")
            else:
                preferred_cols_cs = ["排名", "標的", "市場別", "收盤價", "AI Score", "飆股評分", "選股評分", "共識強度", "選股評級", "飆股等級"]
                show_cols_cs = [c for c in preferred_cols_cs if c in consensus_df.columns]
                st.dataframe(consensus_df[show_cols_cs], use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 💼 投資組合建構 (Portfolio Construction)")
            st.caption(
                "⚠️ 規則式（反波動度加權＋產業集中度上限）資金配置建議，不是嚴謹的最佳化結果，"
                "也不是下單指令，僅供參考起點，下單前請自行覆核。"
            )
            pf_col1, pf_col2, pf_col3, pf_col4 = st.columns(4)
            with pf_col1:
                pf_top_n = st.number_input("最多持股檔數", min_value=2, max_value=30, value=10, key="pf_top_n")
            with pf_col2:
                pf_min_score = st.number_input("AI Score 門檻", min_value=0, max_value=100, value=70, key="pf_min_score")
            with pf_col3:
                pf_industry_cap = st.number_input("單一產業權重上限(%)", min_value=10, max_value=100, value=30, key="pf_industry_cap")
            with pf_col4:
                pf_capital = st.number_input("預計投入資金(元)", min_value=10000, value=1000000, step=10000, key="pf_capital")

            if st.button("💼 建立建議投資組合", key="pf_build_btn"):
                try:
                    st.session_state["portfolio_result"] = PortfolioEngine.build_portfolio(
                        result_df, top_n=pf_top_n, min_ai_score=pf_min_score,
                        max_industry_weight_pct=pf_industry_cap, capital=pf_capital,
                    )
                except Exception as e:
                    st.session_state["portfolio_result"] = {"status": "empty", "note": f"⚠️ 建立投資組合時發生錯誤：{e}"}

            portfolio_result = st.session_state.get("portfolio_result")
            if portfolio_result:
                if portfolio_result["status"] != "ok":
                    st.warning(portfolio_result["note"])
                else:
                    st.dataframe(portfolio_result["weights_table"], use_container_width=True, hide_index=True)
                    pf_dcol1, pf_dcol2 = st.columns(2)
                    with pf_dcol1:
                        st.markdown("**產業集中度分布**")
                        st.dataframe(portfolio_result["industry_breakdown"], use_container_width=True, hide_index=True)
                    with pf_dcol2:
                        st.metric("實際配置金額", f"NT$ {portfolio_result['total_allocated']:,.0f}")
                        st.metric("零股/未配置餘額", f"NT$ {portfolio_result['cash_remaining']:,.0f}")
                    st.caption(portfolio_result["note"])

                    st.markdown("#### 🔄 投資組合再平衡")
                    st.caption(
                        "⚠️ 輸入你目前實際持有的股票與張數，跟上面的目標配置比對，算出需要加碼/減碼的標的。"
                        "這是靜態快照比較，不考慮交易成本或稅務影響，不構成投資建議。"
                    )
                    holdings_input = st.text_area(
                        "目前持股（代碼:張數，逗號分隔）", value="",
                        placeholder="例如：2330:5,2317:10,3711:3", key="rebalance_holdings_input",
                    )
                    min_adjust_input = st.number_input(
                        "調整門檻（張數差距小於此不建議調整）", min_value=1, max_value=50, value=1,
                        key="rebalance_min_lots",
                    )

                    if st.button("🔄 計算再平衡建議", key="rebalance_btn"):
                        try:
                            holdings = {}
                            for item in holdings_input.split(","):
                                item = item.strip()
                                if not item or ":" not in item:
                                    continue
                                code, lots = item.split(":", 1)
                                code = code.strip()
                                if code:
                                    holdings[code] = int(float(lots.strip()))
                            st.session_state["rebalance_result"] = PortfolioEngine.build_rebalance_plan(
                                holdings, portfolio_result["weights_table"], min_adjust_lots=min_adjust_input,
                            )
                        except Exception as e:
                            st.session_state["rebalance_result"] = {"status": "empty", "note": f"⚠️ 計算再平衡時發生錯誤：{e}"}

                    rebalance_result = st.session_state.get("rebalance_result")
                    if rebalance_result:
                        if rebalance_result.get("status") != "ok":
                            st.warning(rebalance_result.get("note", "⚠️ 無法計算再平衡。"))
                        else:
                            st.dataframe(rebalance_result["rebalance_table"], use_container_width=True, hide_index=True)
                            st.caption(rebalance_result["note"])

            with st.expander("📋 查看完整掃描結果"):
                st.dataframe(result_df, use_container_width=True, hide_index=True)

            if not error_df.empty:
                with st.expander(f"⚠️ 掃描失敗清單 ({len(error_df)} 檔)"):
                    st.dataframe(error_df, use_container_width=True, hide_index=True)
    else:
        st.info("👈 請在左側設定掃描清單，然後點擊「📡 啟動全台股掃描」。")

# ==========================================
# 🏆 全市場排行榜模式（v2.8 新增，對應選股學院「排行榜選股法」）
# ==========================================
if mode == "🏆 全市場排行榜":
    if rank_btn:
        with st.spinner("抓取 TWSE 全市場三大法人買賣超資料中..."):
            try:
                st.session_state["rank_result"] = ChipEngine.get_market_wide_institutional_ranking(top_n=rank_top_n)
            except Exception as e:
                st.session_state["rank_result"] = {"status": "unavailable", "message": f"⚠️ 排行榜抓取失敗：{e}"}

    rank_result = st.session_state.get("rank_result")

    if rank_result:
        if rank_result.get("status") != "ok":
            st.warning(rank_result.get("message", "⚠️ 暫時無法取得排行榜資料。"))
        else:
            st.success(f"📅 資料日期：{rank_result['date']}（共 {rank_result['total_stocks']} 檔上市股票有三大法人買賣超資料）")
            st.caption("⚠️ 僅涵蓋「上市」股票，上櫃（TPEx）因資料來源不同，暫不支援此排行榜。資料來源：TWSE 三大法人買賣超日報 (T86)。")

            category = st.selectbox("選擇法人類別", ["三大法人合計", "外資", "投信", "自營商"])
            rankings = rank_result["rankings"][category]

            # 把「標的」（[代碼] 名稱，跟其他頁面格式一致）排到最前面顯示，
            # 原始的「代碼」「名稱」欄位保留在 DataFrame 裡（CSV 下載會完整
            # 匯出全部欄位），只是顯示順序上讓「標的」優先。
            display_cols = ["標的", "外資買賣超", "投信買賣超", "自營商買賣超", "三大法人合計買賣超"]

            def _reorder(d):
                cols = [c for c in display_cols if c in d.columns]
                rest = [c for c in d.columns if c not in cols]
                return d[cols + rest]

            buy_df = _reorder(rankings["買超前N名"])
            sell_df = _reorder(rankings["賣超前N名"])

            rk_col1, rk_col2 = st.columns(2)
            with rk_col1:
                st.markdown(f"#### 🟢 {category} 買超前 N 名")
                st.dataframe(buy_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ 下載買超名單 (CSV)",
                    data=buy_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"institutional_ranking_{category}_buy_{rank_result['date']}.csv",
                    mime="text/csv",
                    key="dl_buy_csv",
                )
            with rk_col2:
                st.markdown(f"#### 🔴 {category} 賣超前 N 名")
                st.dataframe(sell_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ 下載賣超名單 (CSV)",
                    data=sell_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"institutional_ranking_{category}_sell_{rank_result['date']}.csv",
                    mime="text/csv",
                    key="dl_sell_csv",
                )

            st.caption("💡 提示：連續多日出現在同一類別買超前段班的個股，依選股學院文件經驗，較具備波段上漲潛力；建議搭配「個股深度分析」頁面查看該股的技術面/基本面是否同步支持。")
    else:
        st.info("👈 請在左側點擊「🏆 抓取最新排行榜」。")

# ==========================================
# 🌍 總經戰情室模式（v2.9 新增）
# ==========================================
# ⚠️ 範圍說明：這裡只涵蓋能用 yfinance 真實取得的總經代理指標子集合，
# 詳見 macro_engine.py 的 class docstring。這是「背景參考儀表板」，
# 不會影響任何個股的 ai_score/momentum_score 計算。
if mode == "🌍 總經戰情室":
    if macro_btn:
        with st.spinner("抓取總經與跨市場報價中..."):
            try:
                snapshot = MacroEngine.get_snapshot()
                flags = MacroEngine.build_macro_flags(snapshot)
                st.session_state["macro_snapshot"] = snapshot
                st.session_state["macro_flags"] = flags
            except Exception as e:
                st.session_state["macro_snapshot"] = None
                st.session_state["macro_flags"] = [f"⚠️ 總經資料抓取失敗：{e}"]

    snapshot = st.session_state.get("macro_snapshot")
    flags = st.session_state.get("macro_flags")

    if snapshot:
        st.markdown("## 🌍 總經戰情室")

        rows = []
        for name, item in snapshot.items():
            if item.get("status") != "ok":
                rows.append({"指標": name, "代碼": item.get("symbol", ""), "最新值": "N/A",
                             "日變動%": None, "5日變動%": None, "20日變動%": None})
            else:
                rows.append({
                    "指標": name, "代碼": item["symbol"], "最新值": item["latest"],
                    "日變動%": item["chg_1d_pct"], "5日變動%": item["chg_5d_pct"],
                    "20日變動%": item["chg_20d_pct"],
                })
        macro_df = pd.DataFrame(rows)
        st.dataframe(macro_df, use_container_width=True, hide_index=True)

        failed_items = {name: item for name, item in snapshot.items() if item.get("status") != "ok"}
        if failed_items:
            with st.expander(f"⚠️ 有 {len(failed_items)} 項指標抓取失敗，點擊查看技術細節"):
                for name, item in failed_items.items():
                    st.caption(f"**{name}**（{item.get('symbol', '')}）：{item.get('message', '未知原因')}")

        st.markdown("#### 📋 方向性訊號（經驗法則，僅供參考）")
        for f in (flags or []):
            st.markdown(f"- {f}")

        st.caption("⚠️ 以上規則為市場常見經驗法則，不是嚴謹統計檢定結果，不構成投資建議；資料來源為 Yahoo Finance 即時/近期報價。")

        st.markdown("---")
        st.markdown("#### 📐 臺指選擇權 Put/Call Ratio")
        st.caption(
            "⚠️ 資料源：TAIFEX官方公開網頁，僅涵蓋臺指選擇權(TXO)，不含個股選擇權/Greeks/IV；"
            "PCR偏高/偏低在市場上有兩種相反的解讀角度（避險需求 vs 情緒過度反轉訊號），僅供參考。"
        )
        if st.button("📐 抓取最新PCR資料", key="pcr_btn"):
            with st.spinner("抓取TAIFEX臺指選擇權Put/Call Ratio中..."):
                try:
                    st.session_state["pcr_report"] = OptionsEngine.build_pcr_report()
                except Exception as e:
                    st.session_state["pcr_report"] = {"status": "unavailable", "message": f"⚠️ PCR資料抓取時發生錯誤：{e}"}

        pcr_report = st.session_state.get("pcr_report")
        if pcr_report:
            if pcr_report.get("status") != "ok":
                st.warning(pcr_report.get("message", "⚠️ PCR資料暫時無法使用。"))
            else:
                pcr_col1, pcr_col2 = st.columns(2)
                pcr_col1.metric("未平倉量PCR%", f"{pcr_report['pcr_oi_pct']}%")
                pcr_col2.metric("成交量PCR%", f"{pcr_report['pcr_volume_pct']}%")
                st.line_chart(pcr_report["history"].set_index("date")[["pcr_oi_pct", "pcr_volume_pct"]])
                for f in pcr_report["flags"]:
                    st.markdown(f"- {f}")
    elif flags:
        for f in flags:
            st.warning(f)
    else:
        st.info("👈 請在左側點擊「🌍 抓取最新總經數據」。")

# ==========================================
# 🔍 個股深度分析模式
# ==========================================
# ⚠️ 修正說明：比照上面「📡 全台股掃描模式」已經修過的同一種問題。
# run_btn 的 True 只在按下當下觸發的那一次 rerun 有效；這個分析頁面底下
# 的大戶持股／董監持股／季節循環分析都是額外的按鈕，點擊後會觸發新的
# rerun，這時候 run_btn 已經變回 False。如果直接用 run_btn 當作整個
# 分析結果區塊的顯示條件，會導致「點擊這些子按鈕之後，上面整份分析
# 結果都消失、看起來像沒東西」。改用 session_state 記住「目前正在分析
# 哪一檔」，顯示條件改成檢查這個持久化的狀態，而不是按鈕的瞬間狀態。
if run_btn and ticker:
    st.session_state["analysis_ticker"] = ticker

analysis_ticker = st.session_state.get("analysis_ticker")

if mode == "🔍 個股深度分析" and analysis_ticker:
    ticker = analysis_ticker  # 讓下面既有的渲染邏輯繼續沿用 ticker 這個變數名稱
    with st.spinner("啟動底層特徵引擎與 Agent 辯論中..."):
        # AI Decision Pipeline 流水線執行
        # ⚠️ 修正說明：這段原本完全沒有 try/except。DataEngine.get_stock_data()
        # 依設計會在查無資料時主動 raise Exception（見 data_engine.py），
        # 例如代碼打錯、興櫃股票資料覆蓋率低、網路不穩或yfinance暫時異常
        # 都會觸發——原本這個例外會直接讓整個 Streamlit 頁面顯示一坨原始
        # Python traceback崩潰，而不是友善的錯誤訊息。現在包起來，失敗時
        # 顯示清楚的錯誤原因並用 st.stop() 乾淨地中止這次執行，不影響
        # 使用者下次操作。
        try:
            df = DataEngine.get_stock_data(ticker, use_cache=use_cache)
            df = IndicatorEngine.add_indicators(df)
            df = StructureEngine.add_swing_points(df)
            df = RiskEngine.add_risk_metrics(df)
            df = DivergenceEngine.add_defense_signals(df)
            df = StrategyEngine.generate_signals(df)
            df = MomentumEngine.add_momentum_score(df)
            df = EvidenceEngine.add_evidence(df)
        except Exception as e:
            st.error(f"⚠️ 無法取得或處理「{ticker}」的股價資料：{e}")
            st.stop()

        # ==========================================
        # 終極防禦工程：欄位對齊與安全檢查 (避免 KeyError)
        # ==========================================
        # ⚠️ 修正說明：這個欄位補齊區塊原本放在 BacktestEngine.run_backtest(df)
        # 之後才執行。若上游引擎因為資料不足（例如 atr_14 缺失）而沒有算出
        # stop_loss/target_1，代表 BacktestEngine 執行當下這兩欄根本不存在
        # （ATR 停損停利機制被靜默關閉，只能靠 ai_score 出場），但畫面上
        # 事後才補上的「假停損價」卻會顯示出來，兩者完全對不上。現在把這個
        # 區塊移到 BacktestEngine 之前，確保回測引擎跟畫面顯示看到的是
        # 同一組欄位/數值。
        #
        # 同時修正：原本 fallback 用的停損/停利倍數 (1.5x / 2.0x / 4.0x) 跟
        # StrategyEngine 實際公式 (2.0x / 2.5x / 5.0x) 不一致，但下方 UI 卡片
        # 的標籤是寫死的「2.0x / 2.5x / 5.0x ATR」——一旦 fallback 觸發，數字
        # 與標籤會互相矛盾，對風控工具而言是會誤導下單的顯示錯誤。現在統一
        # 成跟 StrategyEngine 相同的倍數，維持全系統公式一致。
        fallback_atr = df['atr_14'] if 'atr_14' in df.columns else df['close'] * 0.02

        required_columns_with_defaults = {
            'market_regime': '🔄 盤整或未知狀態 (請更新 strategy_engine.py)',
            'ai_score': 50.0,
            'confidence': 'Medium (中性)',
            'action_guide': '👀 震盪觀望 (多空拉鋸，建議控制倉位)',
            'bull_reason': '⚠️ 多頭辯護因子未完全載入，請確認底層引擎版本。',
            'bear_reason': '⚠️ 空頭辯護因子未完全載入，請確認底層引擎版本。',
            'risk_reason': '⚠️ 風控審查因子未完全載入，請確認底層引擎版本。',
            'stop_loss': df['close'] - (2.0 * fallback_atr),
            'target_1': df['close'] + (2.5 * fallback_atr),
            'target_2': df['close'] + (5.0 * fallback_atr),
            'zigzag': np.nan,
            'confidence_pct': 50.0,
            'data_quality_pct': 100.0,
            'confidence_label': '🟡 中性信心（請更新 evidence_engine.py）',
            'volatility_annualized': np.nan,
            'rolling_mdd_60d': np.nan,
            'var_95_pct': np.nan,
            'var_99_pct': np.nan,
            'bearish_divergence': False,
            'bullish_divergence': False,
            'divergence_note': '',
            'breakout_up': False,
            'breakout_down': False,
            'bull_trap_confirmed': False,
            'bear_trap_confirmed': False,
            'trap_note': '',
            'momentum_score': 50.0,
            'momentum_grade': 'C (請更新 momentum_engine.py)',
            'is_a_grade_candidate': False,
            'momentum_penalty_alert': False,
            'reversal_watch': False,
            'trap_alert': False,
            'defense_risk_flag': False,
            'entry_signal': '⚪ 無明確買進訊號 (請更新 strategy_engine.py)',
            'exit_signal': '⚪ 無明確賣出訊號 (請更新 strategy_engine.py)',
        }

        for col, default_val in required_columns_with_defaults.items():
            if col not in df.columns:
                df[col] = default_val

        # ==========================================
        # 回測資料準備
        # ==========================================
        # ⚠️ 修正說明：df 的最後一筆可能是 DataEngine 用「盤中即時報價」撐開
        # OHLC 產生的估計值（is_intraday_estimate=True），本質上是尚未收盤
        # 定案的資料。RiskEngine.compute_beta() 已經會排除這一筆再計算，
        # 但原本 BacktestEngine 卻直接把它當一般歷史K棒跑進全樣本回測（含
        # 停損/停利觸價判斷與報酬率計算），等於用未定案資料做決策，跟
        # RiskEngine 的處理方式不一致。這裡讓回測輸入資料也排除掉這一筆，
        # 並且刻意不覆蓋主要的 df（Dashboard 顯示仍然要看得到最新的即時估計）。
        if 'is_intraday_estimate' in df.columns and bool(df['is_intraday_estimate'].iloc[-1]):
            backtest_input_df = df.iloc[:-1].copy()
        else:
            backtest_input_df = df

        if len(backtest_input_df) >= 30:
            _, report = BacktestEngine.run_backtest(backtest_input_df)
        else:
            report = {
                'total_return': 0.0, 'win_rate': 0.0, 'max_drawdown': 0.0,
                'total_fees_paid': 0.0,
                'exit_breakdown': {'停損出場': 0, '停利出場': 0, 'AI Score轉弱出場': 0},
                'note': "⚠️ 排除盤中即時估計值後，可用歷史資料不足，暫不執行回測。",
            }

        # 大盤基準資料（用於 Beta 計算），抓取失敗不影響主流程
        try:
            benchmark_df = DataEngine.get_benchmark_data()
        except Exception:
            benchmark_df = None
        
        try:
            risk_report = RiskEngine.build_risk_report(df, benchmark_df)
        except Exception:
            risk_report = None
        
        try:
            chip_report = ChipEngine.build_chip_report(ticker)
        except Exception:
            chip_report = {"status": "unavailable", "message": "⚠️ 籌碼中心初始化失敗。"}

        try:
            fundamental_report = FundamentalEngine.build_fundamental_report(ticker)
        except Exception:
            fundamental_report = {"status": "unavailable", "message": "⚠️ 基本面與財務面中心初始化失敗。"}

        # 個股深度分析頁面：帶入真實的 chip_report / fundamental_report，
        # 算出完整的選股大師五維度評分（跟批次掃描的「快速模式」不同，
        # 那邊為了效能刻意不呼叫這兩個外部資料源，見 scanner_engine.py 說明）。
        try:
            academy_report = StockAcademyEngine.build_report(
                df, chip_report=chip_report, fundamental_report=fundamental_report
            )
        except Exception:
            academy_report = {}

        try:
            tf_report = TimeframeEngine.build_report(df)
        except Exception:
            tf_report = {}

        latest = df.iloc[-1]
        
        try:
            evidence_list = EvidenceEngine.get_evidence_list(df, -1)
        except Exception:
            evidence_list = []
        
        # ==========================================
        # Dashboard Layer: 戰情室視覺化
        # ==========================================
        st.markdown(f"## 📊 標的 `{NameEngine.get_tag(ticker)}` 企業級分析戰情室")
        st.caption(f"市場別：{NameEngine.get_market_type(ticker)}")
        
        # --- 頂部：宏觀狀態與最終裁決 ---
        r1, r2, r3, r4 = st.columns(4)
        
        # 兼容處理單值或 Series 形式的 close
        raw_close = latest['close']
        latest_close = float(raw_close.iloc[0] if isinstance(raw_close, (np.ndarray, list, pd.Series)) else raw_close)
        
        r1.metric("即時收盤價", f"{latest_close:.2f}")
        r2.metric("🌍 市場狀態 (Market Regime)", f"{latest['market_regime']}")
        r3.metric("🤖 綜合 AI 評分 (AI Score)", f"{latest['ai_score']:.1f} / 100")
        r4.metric("🛡️ 決策信心 (Confidence)", f"{latest['confidence']}")
        
        st.info(f"**⚖️ 主審裁決 (Judge Action Guide)：** {latest['action_guide']}")

        sig_col1, sig_col2 = st.columns(2)
        sig_col1.markdown(f"**🎯 買進訊號：** {latest.get('entry_signal', 'N/A')}")
        sig_col2.markdown(f"**🚪 賣出訊號：** {latest.get('exit_signal', 'N/A')}")
        st.caption("ℹ️ 買賣訊號已納入近期背離/誘多假突破防禦訊號（見下方風險中心與誘盤警報），與 AI Score 為互補資訊，仍請自行評估風險。")

        if bool(latest.get('trap_alert', False)):
            trap_msg = latest.get('trap_note', '') or latest.get('divergence_note', '') or "近期偵測到背離或假突破/假跌破警報。"
            st.warning(f"🛡️ **誘盤警報：** {trap_msg}")

        st.markdown("---")
        
        # --- 證據與信心模型 (Evidence & Confidence Engine) ---
        st.markdown("### 🧾 證據與信心模型 (Evidence & Confidence Engine)")
        
        e1, e2, e3 = st.columns([1, 1, 2])
        e1.metric("🤖 AI Score", f"{latest['ai_score']:.1f}")
        e2.metric("📊 資料品質 (Data Quality)", f"{latest['data_quality_pct']:.0f}%")
        e3.metric("🧠 信心程度 (Confidence)", f"{latest['confidence_pct']:.0f}%  ·  {latest['confidence_label']}")
        
        if evidence_list:
            bull_ev = [e for e in evidence_list if e['polarity'] == 'bull']
            bear_ev = [e for e in evidence_list if e['polarity'] == 'bear']
            risk_ev = [e for e in evidence_list if e['polarity'] == 'risk']
            
            ev_col1, ev_col2, ev_col3 = st.columns(3)
            
            def render_evidence(col, title, items, empty_msg):
                with col:
                    st.markdown(f"**{title}**")
                    if not items:
                        st.caption(empty_msg)
                    for item in items:
                        stars = "★" * item['stars'] + "☆" * (5 - item['stars'])
                        detail = f"　_{item['detail']}_" if item.get('detail') else ""
                        st.markdown(f"- {item['label']}　`{stars}`{detail}")
            
            render_evidence(ev_col1, "🟢 多頭證據", bull_ev, "目前無明顯多頭證據")
            render_evidence(ev_col2, "🔴 空頭證據", bear_ev, "目前無明顯空頭證據")
            render_evidence(ev_col3, "🛡️ 風險證據", risk_ev, "目前無異常風險證據")
        else:
            st.caption("⚠️ 尚無足夠資料生成證據清單。")
        
        st.markdown("---")

        # --- 經典技術指標：MTM 動量指標／寶塔線 (v2.8 新增) ---
        st.markdown("### 📐 經典技術指標 (MTM 動量指標／寶塔線)")
        mt1, mt2 = st.columns(2)

        with mt1:
            mtm_val = latest.get('mtm', np.nan)
            mtm_val = float(mtm_val.iloc[0] if isinstance(mtm_val, (np.ndarray, list, pd.Series)) else mtm_val)
            if pd.notna(mtm_val):
                direction = "🟢 動能為正（近期股價高於n日前）" if mtm_val > 0 else ("🔴 動能為負（近期股價低於n日前）" if mtm_val < 0 else "🟡 動能持平")
                st.metric("📊 MTM 動量指標", f"{mtm_val:.2f}", help="MTM = 今日收盤價 - n日前收盤價，衡量股價變動的速度")
                st.caption(direction)
            else:
                st.caption("⚠️ MTM 資料不足（K線筆數不足n天）")
            st.caption("⚠️ MTM 訊號容易在0上下反覆穿越、雜訊較多，建議搭配均線/MACD等趨勢指標一起判讀，不宜單獨作為買賣依據。")

        with mt2:
            pagoda_trend = latest.get('pagoda_trend', None)
            if pagoda_trend == "red":
                st.success("🔴➡️🟩 寶塔線：目前為「翻紅」狀態（趨勢向上已確立）")
            elif pagoda_trend == "black":
                st.error("⬛ 寶塔線：目前為「翻黑」狀態（趨勢向下已確立）")
            else:
                st.caption("⚠️ 寶塔線資料不足，尚未產生翻轉訊號")
            st.caption("⚠️ 寶塔線屬於落後指標，翻紅/翻黑代表趨勢「已經」確立，而非預測轉折，訊號通常會落後真正的高低點數天，盤整格局容易出現進出頻繁但無獲利的情況。")

        st.markdown("---")
        
        # --- 風險中心 (Risk Center) ---
        st.markdown("### 🛡️ 風險中心 (Risk Center)")
        
        if risk_report:
            rk1, rk2, rk3, rk4, rk5 = st.columns(5)
            
            vol = risk_report['volatility_annualized']
            mdd = risk_report['max_drawdown_60d']
            beta = risk_report['beta']
            var95 = risk_report['var_95_pct']
            rr = risk_report['reward_risk_ratio']
            
            rk1.metric("📈 年化波動率", f"{vol:.1f}%" if pd.notna(vol) else "N/A")
            rk2.metric("📉 60日最大回撤", f"{mdd:.1f}%" if pd.notna(mdd) else "N/A")
            rk3.metric("β Beta (相對大盤)", f"{beta:.2f}" if pd.notna(beta) else "N/A")
            rk4.metric("🎲 VaR (95%, 單日)", f"{var95:.1f}%" if pd.notna(var95) else "N/A")
            rk5.metric("⚖️ 報酬風險比 (RR)", f"{rr:.2f}" if pd.notna(rr) else "N/A")
            
            st.markdown(f"**綜合風險等級：{risk_report['risk_level']}**")
            for flag in risk_report['risk_flags']:
                st.caption(flag)
        else:
            st.caption("⚠️ 風險中心計算失敗，請確認 risk_engine.py 是否已正確放入 engines 資料夾。")
        
        st.markdown("---")
        
        # --- 籌碼中心 (Chip Center) ---
        st.markdown("### 🏦 籌碼中心 (Chip Center)")
        
        if chip_report and chip_report.get("status") == "ok":
            inst = chip_report.get("institutional")
            margin = chip_report.get("margin")
            
            ch1, ch2, ch3, ch4 = st.columns(4)
            if inst:
                ch1.metric("🌍 外資買賣超 (張)", f"{inst['foreign_net']/1000:,.0f}", help=f"資料日期：{inst['date']}")
                ch2.metric("🏛️ 投信買賣超 (張)", f"{inst['trust_net']/1000:,.0f}")
                ch3.metric("💼 自營商買賣超 (張)", f"{inst['dealer_net']/1000:,.0f}")
                ch4.metric("📊 三大法人合計 (張)", f"{inst['total_net']/1000:,.0f}")
            else:
                st.caption("ℹ️ 暫無三大法人買賣超資料（可能為上櫃股票或近期非交易日）。")
            
            if margin:
                cm1, cm2 = st.columns(2)
                cm1.metric("💳 融資餘額變化 (張)", f"{margin['margin_change']/1000:,.1f}", help=f"目前餘額 {margin['margin_balance']/1000:,.0f} 張")
                cm2.metric("📉 融券餘額變化 (張)", f"{margin['short_change']/1000:,.1f}", help=f"目前餘額 {margin['short_balance']/1000:,.0f} 張")
            
            for flag in chip_report.get("flags", []):
                st.caption(flag)
        else:
            msg = chip_report.get("message", "⚠️ 暫時無法取得籌碼資料。") if chip_report else "⚠️ 暫時無法取得籌碼資料。"
            st.caption(msg)
            st.caption("ℹ️ 籌碼中心僅支援上市股票，資料來源為 TWSE 公開資訊，需要對外網路連線。")
        
        st.markdown("---")

        # --- 基本面與財務面中心 (Fundamental & Financial Center) ---
        st.markdown("### 📑 基本面與財務面中心 (Fundamental & Financial Center)")
        st.caption("資料來源：yfinance 財務摘要（Yahoo Finance）。台股（尤其中小型股/上櫃/興櫃）覆蓋率與更新頻率有限，缺值一律顯示「資料不足」，不做估計填補。與上方技術面/籌碼面為互補視角，非取代關係。")

        if fundamental_report.get("status") == "not_applicable":
            st.info(fundamental_report.get("message"))
        elif fundamental_report.get("status") != "ok":
            st.caption(fundamental_report.get("message", "⚠️ 暫時無法取得基本面資料。"))
        else:
            snap = fundamental_report["snapshot"]

            def _fmt(val, suffix="", multiplier=1, decimals=1):
                if pd.isna(val):
                    return "資料不足"
                return f"{val * multiplier:.{decimals}f}{suffix}"

            fd1, fd2, fd3, fd4 = st.columns(4)
            fd1.metric("💰 EPS (近四季, TTM)", _fmt(snap["eps_ttm"], decimals=2))
            fd2.metric("📊 本益比 (TTM)", _fmt(snap["pe_ttm"], decimals=1))
            fd3.metric("📈 本益比 (預估, Forward)", _fmt(snap["pe_forward"], decimals=1))
            fd4.metric("📘 股價淨值比 (P/B)", _fmt(snap["price_to_book"], decimals=2))

            fd5, fd6, fd7, fd8 = st.columns(4)
            mc = snap["market_cap"]
            fd5.metric("🏢 總市值", f"{mc/1e8:,.1f} 億" if pd.notna(mc) else "資料不足")
            ec = snap["estimated_capital"]
            fd6.metric("🧮 股本 (概略估算)", f"{ec/1e8:,.1f} 億" if pd.notna(ec) else "資料不足",
                       help="以流通在外股數 × 台股常見面額10元概略估算，非精確財報數字，詳見引擎說明")
            fd7.metric("📖 每股淨值", _fmt(snap["book_value_per_share"], decimals=2))
            rev = snap["revenue_ttm"]
            fd8.metric("💵 營收 (近四季)", f"{rev/1e8:,.1f} 億" if pd.notna(rev) else "資料不足")

            fd9, fd10, fd11, fd12 = st.columns(4)
            fd9.metric("🚀 營收年增率", _fmt(snap["revenue_growth_yoy"], suffix="%", multiplier=100))
            fd10.metric("🏦 ROA (資產報酬率)", _fmt(snap["roa"], suffix="%", multiplier=100))
            fd11.metric("💎 ROE (股東權益報酬率)", _fmt(snap["roe"], suffix="%", multiplier=100))
            debt_ratio = snap.get("debt_ratio_pct", np.nan)
            if pd.notna(debt_ratio):
                fd12.metric("⚖️ 負債比例 (精算)", _fmt(debt_ratio, suffix="%", decimals=1),
                            help="以真實資產負債表精算：負債總額 / 資產總額（嚴謹定義）")
            else:
                fd12.metric("⚖️ 負債權益比 (近似)", _fmt(snap["debt_to_equity"], suffix="%", decimals=0),
                            help="抓不到資產負債表精算值，退而求其次改用 yfinance 的負債/股東權益比近似，兩者分母不同")

            fd13, fd14 = st.columns(2)
            fd13.metric("🏭 毛利率", _fmt(snap["gross_margin"], suffix="%", multiplier=100))
            fd14.metric("⚙️ 營業利益率", _fmt(snap["operating_margin"], suffix="%", multiplier=100))

            st.markdown("**綜合觀察：**")
            for flag in fundamental_report.get("flags", []):
                st.caption(flag)

            st.caption("ℹ️ 「成立時間」「主力買賣超」「集保庫存」「董監持股比例」等選股學院提及的指標，yfinance 無對應資料來源，本區塊不假造這些欄位；如需要請改查詢公開資訊觀測站 (MOPS) 或券商看盤軟體。")

        st.markdown("---")

        # --- 選股大師五維度評分 (Stock Academy Engine) ---
        st.markdown("### 🎓 選股大師 (五維度評分：市場面／基本面／技術面／籌碼面／財務面)")

        if academy_report:
            comp = academy_report.get("綜合評級", {})
            ac1, ac2, ac3 = st.columns([1, 1, 2])
            ac1.metric("🎓 綜合評分", f"{comp.get('總分', 0)} / 100")
            ac2.metric("🏅 學級", f"{comp.get('學級', 'F')}")
            ac3.metric("💬 評語", f"{comp.get('評語', '')}")

            st.caption(academy_report.get("資料完整度提示", ""))

            dims = academy_report.get("五維度明細", {})
            dim_cols = st.columns(5)
            dim_order = ["市場面", "基本面", "技術面", "籌碼面", "財務面"]
            for col, dim_name in zip(dim_cols, dim_order):
                data = dims.get(dim_name, {"分數": 0, "分析": []})
                with col:
                    st.markdown(f"**{dim_name}**　`{data['分數']}/20`")
                    for point in data.get("分析", []):
                        st.caption(point)

            st.markdown("**🎯 關鍵提示：**")
            for tip in academy_report.get("關鍵提示", []):
                st.markdown(f"- {tip}")

            st.caption("⚠️ 選股大師評級是「中長期體質」視角，跟上方 AI Score（短期波段）、飆股評分（動能結構）是三套獨立、互不加權混合的評分系統，刻意分開呈現，避免同一組事實被重複計分。三者請並參，不應只依賴其中一項下單。")
        else:
            st.caption("⚠️ 選股大師評分計算失敗，請確認 stock_academy_engine.py 是否已正確放入 engines 資料夾。")

        st.markdown("---")
        
        # --- 核心：多智能體辯論面板 (Multi-Agent Debate) ---
        st.markdown("### 🗣️ 多智能體對抗辯論 (Multi-Agent Debate)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success(f"**🟢 Bull Agent (多頭辯護)**\n\n💡 觀點：\n\n{latest['bull_reason']}")
        with c2:
            st.error(f"**🔴 Bear Agent (空頭辯護)**\n\n💡 觀點：\n\n{latest['bear_reason']}")
        with c3:
            st.warning(f"**🛡️ Risk Agent (風控審查)**\n\n💡 觀點：\n\n{latest['risk_reason']}")
            
        st.markdown("---")

        # --- 多週期策略與未來走向 (Timeframe & Outlook Engine) ---
        st.markdown("### ⏳ 多週期策略判斷 (短線／波段／長線)")
        if tf_report and "error" not in tf_report:
            tf_col1, tf_col2, tf_col3 = st.columns(3)

            def _render_tf_card(col, title, info):
                with col:
                    st.markdown(f"**{title}**　`{info.get('view', 'N/A')}`")
                    st.caption(f"操作週期：{info.get('horizon', 'N/A')}")
                    st.markdown(f"依據：{info.get('reason', '')}")
                    st.markdown(f"建議：{info.get('action', '')}")

                    plan = info.get("trade_plan", {})
                    if plan.get("available"):
                        st.markdown("**📍 進場／出場價位參考**")
                        st.caption(f"進場參考價：約 {plan['entry_ref_price']}｜{plan['entry_condition']}")
                        target_line = f"出場目標：約 {plan['exit_target_price']}｜{plan['exit_target_condition']}"
                        st.caption(target_line)
                        st.caption(f"停損價：約 {plan['exit_stop_price']}｜{plan['exit_stop_condition']}")
                        st.caption(f"ℹ️ {plan['timing_note']}")
                    elif plan.get("note"):
                        st.caption(f"ℹ️ {plan['note']}")

            _render_tf_card(tf_col1, "⚡ 短線", tf_report.get("short_term", {}))
            _render_tf_card(tf_col2, "🌊 波段", tf_report.get("swing", {}))
            _render_tf_card(tf_col3, "🏔️ 長線", tf_report.get("long_term", {}))

            with st.expander("📊 短線進出場公式的簡化版歷史命中率檢查"):
                st.caption(
                    "⚠️ 這是簡化統計，不是跟 BacktestEngine 同等嚴謹的回測——用當天收盤價/ATR"
                    "當基準（比實際交易更樂觀），沒有計算手續費。僅供參考，不代表未來表現，不構成投資建議。"
                )
                if st.button("📊 執行短線命中率檢查", key="short_term_hitrate_btn"):
                    try:
                        st.session_state["short_term_hitrate"] = TimeframeEngine.backtest_short_term_hit_rate(df)
                    except Exception as e:
                        st.session_state["short_term_hitrate"] = {"status": "unavailable", "message": f"檢查時發生錯誤：{e}"}

                hitrate = st.session_state.get("short_term_hitrate")
                if hitrate:
                    if hitrate.get("status") != "ok":
                        st.warning(hitrate.get("message", "⚠️ 無法計算命中率。"))
                    else:
                        hr1, hr2, hr3 = st.columns(3)
                        hr1.metric("目標優先命中", f"{hitrate['target_hit']} 次")
                        hr2.metric("停損優先觸發", f"{hitrate['stop_hit']} 次")
                        hr3.metric("目標命中率", f"{hitrate['win_rate_pct']}%")
                        st.caption(hitrate["note"])

            st.warning(
                "⚠️ 以上進場／出場價位都是「依目前技術結構與波動率(ATR)推算出的參考價位」，"
                "不是保證會被觸及的價位，也不是對未來日期的預測，市場可能直接跳空穿越、"
                "或永遠不回測到這個價位，不構成投資建議，請自行評估風險。"
            )

            st.markdown("#### 🔮 未來走向 (情境推演)")
            outlook = tf_report.get("outlook", {})
            st.markdown(f"**綜合傾向：`{outlook.get('bias', 'N/A')}`** — {outlook.get('bias_note', '')}")

            levels = outlook.get("key_levels", {})
            lv1, lv2, lv3 = st.columns(3)
            lv1.metric("推算支撐區", f"{levels.get('support_est')}" if levels.get('support_est') is not None else "N/A")
            lv2.metric("推算壓力區", f"{levels.get('resistance_est')}" if levels.get('resistance_est') is not None else "N/A")
            lv3.metric("季線(60MA)", f"{levels.get('sma_60')}" if levels.get('sma_60') is not None else "N/A")

            for scenario in outlook.get("scenarios", []):
                st.markdown(f"- **{scenario['condition']}** → {scenario['implication']}")

            st.warning(outlook.get("disclaimer", "⚠️ 本區塊僅供技術面情境參考，不構成投資建議。"))
        else:
            st.caption("⚠️ 尚無足夠資料生成多週期策略報告（請確認 timeframe_engine.py 是否已正確放入 engines 資料夾）。")

        st.markdown("---")

        # --- 飆股動能引擎 (Momentum Engine) ---
        st.markdown("### 🚀 飆股動能引擎 (七層過濾 + 100分評分系統)")
        mo1, mo2, mo3 = st.columns([1, 1, 2])
        mo1.metric("🚀 飆股評分 (Momentum Score)", f"{float(latest['momentum_score']):.1f} / 100")
        mo2.metric("🏅 飆股等級", f"{latest['momentum_grade']}")
        with mo3:
            # 修正：原本只要 trap_alert 為 True 就顯示「本層評分已扣分」，但
            # trap_alert 也包含不影響本層分數的底部反轉觀察訊號（底背離/誘空
            # 確認），會出現「顯示扣分警告，但分數其實是滿分」的矛盾。改用
            # momentum_penalty_alert（真正造成扣分的誘多假突破/頂背離）判斷。
            penalty_active = bool(latest.get('momentum_penalty_alert', latest.get('trap_alert', False)))
            reversal_watch_active = bool(latest.get('reversal_watch', False))
            if penalty_active:
                st.error("🛡️ 誘多/背離防禦層：近期觸發誘多假突破/頂背離警報，本層評分已扣分")
            elif reversal_watch_active:
                st.info("👀 觀察：近期出現底背離/誘空反轉訊號（不影響飆股評分，僅供留意可能的底部訊號）")
            else:
                st.success("🛡️ 誘多/背離防禦層：近期無警報，本層滿分")

        try:
            breakdown = MomentumEngine.get_momentum_breakdown(df, -1)
        except Exception:
            breakdown = []

        if breakdown:
            for layer in breakdown:
                icon = "✅" if layer["passed"] else "⬜"
                st.markdown(f"{icon} **{layer['layer']}**　_{layer['detail']}_")
        else:
            st.caption("⚠️ 尚無足夠資料生成飆股評分明細。")

        st.caption("⚠️ 飆股評分與 AI Score 是兩套獨立邏輯，刻意不互相加權平均：AI Score 回答『值不值得操作』，飆股評分回答『技術結構像不像飆股』，請兩者並參，不應單獨依賴其中一項下單。")

        st.markdown("---")
        
        # --- 數據面板：回測與風險預算 ---
        # ⚠️ 修正說明：原標題「Walk-Forward 樣本內回測」用詞矛盾且誤導 ——
        # Walk-Forward 指的是滾動式樣本外驗證，這裡實際上是全樣本內回測，
        # 進出場門檻與策略顯示邏輯相同，不構成獨立的樣本外驗證。已改用
        # 正確名稱，並把 BacktestEngine 回傳的 note／出場原因拆解／已扣費用
        # 一併顯示，避免使用者誤信這是嚴謹的策略驗證結果。
        col_bt1, col_bt2 = st.columns([1, 1])
        with col_bt1:
            st.markdown("#### 📈 全樣本內回測 (近兩年，已扣手續費/證交稅)")
            m1, m2, m3 = st.columns(3)
            m1.metric("策略總報酬率", f"{report['total_return']:.2f}%")
            m2.metric("交易勝率", f"{report['win_rate']:.1f}%")
            m3.metric("最大歷史回撤 (MDD)", f"{report['max_drawdown']:.2f}%")

            if 'total_fees_paid' in report:
                st.caption(f"💸 已扣總交易成本：約 {report['total_fees_paid']:,.0f} 元")
            if 'exit_breakdown' in report:
                breakdown = report['exit_breakdown']
                st.caption(
                    f"出場原因分布 — 停損: {breakdown.get('停損出場', 0)} 次 / "
                    f"停利: {breakdown.get('停利出場', 0)} 次 / "
                    f"AI Score轉弱: {breakdown.get('AI Score轉弱出場', 0)} 次"
                )
            if 'note' in report:
                st.caption(report['note'])

        with col_bt2:
            st.markdown("#### 🎯 動態風險預算 (基於 ATR 動能)")
            n1, n2, n3 = st.columns(3)
            
            # 兼容處理停損停利值
            raw_sl = latest['stop_loss']
            raw_t1 = latest['target_1']
            raw_t2 = latest['target_2']
            val_sl = float(raw_sl.iloc[0] if isinstance(raw_sl, (np.ndarray, list, pd.Series)) else raw_sl)
            val_t1 = float(raw_t1.iloc[0] if isinstance(raw_t1, (np.ndarray, list, pd.Series)) else raw_t1)
            val_t2 = float(raw_t2.iloc[0] if isinstance(raw_t2, (np.ndarray, list, pd.Series)) else raw_t2)

            # 修正：標籤倍數原本與 StrategyEngine 實際公式（2.0x / 2.5x / 5.0x ATR）不符，
            # 已修正為正確倍數，避免使用者依錯誤標示的風險預算下單。
            n1.metric("🛡️ 建議停損點 (2.0x ATR)", f"{val_sl:.2f}")
            n2.metric("🎯 目標獲利 1 (2.5x ATR)", f"{val_t1:.2f}")
            n3.metric("🏆 目標獲利 2 (5.0x ATR)", f"{val_t2:.2f}")

        # ==========================================
        # 互動式 K 線圖與斐波那契矩陣
        # ==========================================
        st.markdown("---")
        recent_df = df.tail(150)
        
        fig = fgo.Figure()
        
        # 繪製 K 線
        fig.add_trace(fgo.Candlestick(
            x=recent_df['date'], open=recent_df['open'], 
            high=recent_df['high'], low=recent_df['low'], close=recent_df['close'], name="K線"
        ))
        
        # 繪製 ZigZag 市場結構線 (過濾掉 NaN)
        if 'zigzag' in recent_df.columns:
            zigzag_df = recent_df.dropna(subset=['zigzag'])
            if not zigzag_df.empty:
                fig.add_trace(fgo.Scatter(
                    x=zigzag_df['date'], y=zigzag_df['zigzag'],
                    mode='lines+markers', name='市場結構 (ZigZag)',
                    line=dict(color='cyan', width=1, dash='dot'),
                    marker=dict(size=6, color='cyan')
                ))

        # 標註 MACD 背離與誘多/誘空防禦警報（均為因果安全的「確認當下」標記，
        # 不是回填到轉折發生的那一天，詳見 divergence_engine.py 說明）
        marker_specs = [
            ('bearish_divergence', '🔴 頂背離', 'high', 12, 'triangle-down', '#ff4d4d'),
            ('bullish_divergence', '🟢 底背離', 'low', -12, 'triangle-up', '#33cc33'),
            ('bull_trap_confirmed', '⚠️ 誘多假突破', 'high', 20, 'x', '#ffcc00'),
            ('bear_trap_confirmed', '⚠️ 誘空假跌破', 'low', -20, 'x', '#ff9933'),
        ]
        for col, label, ref_col, offset, symbol, color in marker_specs:
            if col in recent_df.columns:
                hits = recent_df[recent_df[col] == True]
                if not hits.empty:
                    y_vals = hits[ref_col] + offset if offset > 0 else hits[ref_col] + offset
                    fig.add_trace(fgo.Scatter(
                        x=hits['date'], y=y_vals, mode='markers', name=label,
                        marker=dict(size=11, symbol=symbol, color=color, line=dict(width=1, color='white'))
                    ))
        
        # 自動繪製近期斐波那契回撤線
        max_h = recent_df['high'].max()
        min_l = recent_df['low'].min()
        diff = max_h - min_l
        
        fib_levels = {
            "100% (High)": max_h,
            "61.8% (Golden)": min_l + 0.618 * diff,
            "50.0% (Mid)": min_l + 0.500 * diff,
            "38.2% (Support)": min_l + 0.382 * diff,
            "0% (Low)": min_l
        }
        
        colors = ['#ff4d4d', '#ffcc00', '#33cc33', '#3399ff', '#cccccc']
        for (label, price), color in zip(fib_levels.items(), colors):
            # 確保價格為純數值類型
            val_price = float(price.iloc[0] if isinstance(price, (np.ndarray, list, pd.Series)) else price)
            fig.add_hline(
                y=val_price, line_dash="dash", line_color=color, opacity=0.4,
                annotation_text=f"Fib {label}  {val_price:.1f}", 
                annotation_position="top left",
                annotation_font=dict(color=color, size=10)
            )

        fig.update_layout(
            template="plotly_dark", 
            height=700,
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        with st.expander("🏦 大戶持股／千張大戶分析 — 資料源：TDCC集保戶股權分散表，點擊展開後執行"):
            st.caption(
                "⚠️ 每週更新一次（反映集保帳戶而非實質受益人，非即時資料），"
                "「大戶」採業界慣例定義：持股1,000張以上；歷史趨勢是本系統自己逐週累積的記錄，"
                "剛開始使用時筆數會很少，屬正常現象。"
            )
            if st.button("🏦 查詢大戶持股分布（抓取TDCC開放資料）", key="shareholding_btn"):
                with st.spinner("下載並解析集保戶股權分散表中（全市場單一檔案，第一次查詢較慢）..."):
                    try:
                        st.session_state["shareholding_report"] = ChipEngine.get_shareholding_distribution(ticker)
                    except Exception as e:
                        st.session_state["shareholding_report"] = {"status": "unavailable", "message": f"⚠️ 大戶持股資料查詢時發生錯誤：{e}"}

            shareholding_report = st.session_state.get("shareholding_report")
            if shareholding_report:
                if shareholding_report.get("status") != "ok":
                    st.warning(shareholding_report.get("message", "⚠️ 大戶持股資料暫時無法使用。"))
                else:
                    sh_col1, sh_col2, sh_col3 = st.columns(3)
                    sh_col1.metric("資料日期", shareholding_report["date"])
                    sh_col2.metric("千張大戶持股佔比", f"{shareholding_report['large_holder_pct']}%")
                    sh_col3.metric("千張大戶人數", f"{shareholding_report['large_holder_count']:,}")

                    st.dataframe(shareholding_report["tiers"], use_container_width=True, hide_index=True)
                    for f in shareholding_report["flags"]:
                        st.markdown(f"- {f}")

                    trend = ChipEngine.get_shareholding_trend(ticker)
                    if len(trend) >= 2:
                        st.markdown("**大戶持股佔比歷史趨勢（本系統累積記錄）**")
                        st.line_chart(trend.set_index("date")["large_holder_pct"])
                    else:
                        st.caption("ℹ️ 目前只有本次查詢的單一筆記錄，趨勢圖需要多次（跨週）查詢後才會累積出來。")

        with st.expander("🏛️ 董監事／大股東持股與設質分析 — 資料源：證交所董監持股開放資料，僅支援上市股票"):
            st.caption(
                "⚠️ 只支援上市股票（上櫃另有不同資料檔案，尚未驗證，暫不支援），每月更新一次，"
                "反映申報當下持股，不是即時資料；設質比例高不代表一定有問題，需搭配其他資訊綜合判斷，不構成投資建議。"
            )
            if st.button("🏛️ 查詢董監事持股與設質狀況（抓取證交所開放資料）", key="insider_btn"):
                with st.spinner("下載並解析董監事持股資料中（全市場單一檔案，第一次查詢較慢）..."):
                    try:
                        st.session_state["insider_report"] = ChipEngine.get_insider_holdings(ticker)
                    except Exception as e:
                        st.session_state["insider_report"] = {"status": "unavailable", "message": f"⚠️ 董監事持股資料查詢時發生錯誤：{e}"}

            insider_report = st.session_state.get("insider_report")
            if insider_report:
                if insider_report.get("status") != "ok":
                    st.warning(insider_report.get("message", "⚠️ 董監事持股資料暫時無法使用。"))
                else:
                    in_col1, in_col2 = st.columns(2)
                    in_col1.metric("資料年月", insider_report["data_month"])
                    in_col2.metric("設質比例（個人/關係人較高者）", f"{insider_report['max_pledge_pct']}%")

                    st.dataframe(insider_report["detail"], use_container_width=True, hide_index=True)
                    for f in insider_report["flags"]:
                        st.markdown(f"- {f}")

                    if not insider_report["high_pledge_table"].empty:
                        st.markdown("**⚠️ 個人設質比例超過50%的內部人**")
                        st.dataframe(insider_report["high_pledge_table"], use_container_width=True, hide_index=True)

        with st.expander("📅 季節循環分析 (Seasonality) — 需額外抓取10年歷史資料，點擊展開後執行"):
            st.caption(
                "⚠️ 這裡只統計「月份別歷史報酬」，不做除權息行情/選舉行情等需要額外事件資料源的分析；"
                "樣本數很小（10年資料每個月份也只有約10個獨立觀察值），任何規律都可能只是雜訊，"
                "不是穩定可複製的效應，僅供參考，不能單獨依此做進出場決策。"
            )
            if st.button("📅 執行季節循環分析（抓取10年歷史資料）", key="seasonality_btn"):
                with st.spinner("抓取長天期歷史資料並計算月份統計中..."):
                    try:
                        st.session_state["seasonality_report"] = SeasonalityEngine.build_seasonality_report(ticker)
                    except Exception as e:
                        st.session_state["seasonality_report"] = {"status": "unavailable", "message": f"⚠️ 季節循環分析時發生錯誤：{e}"}

            seasonality_report = st.session_state.get("seasonality_report")
            if seasonality_report:
                if seasonality_report.get("status") != "ok":
                    st.warning(seasonality_report.get("message", "⚠️ 季節性分析暫時無法使用。"))
                else:
                    st.caption(f"共涵蓋約 {seasonality_report['years_covered']} 個曆年的歷史資料。")
                    st.dataframe(seasonality_report["monthly_table"], use_container_width=True, hide_index=True)
                    for f in seasonality_report["flags"]:
                        st.markdown(f"- {f}")