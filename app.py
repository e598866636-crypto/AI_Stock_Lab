import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as fgo
from engines.data_engine import DataEngine
from engines.indicator_engine import IndicatorEngine
from engines.structure_engine import StructureEngine
from engines.strategy_engine import StrategyEngine
from engines.evidence_engine import EvidenceEngine
from engines.risk_engine import RiskEngine
from engines.backtest_engine import BacktestEngine
from engines.db_engine import DatabaseEngine
from engines.scanner_engine import ScannerEngine
from engines.chip_engine import ChipEngine
from engines.industry_engine import IndustryEngine

# 設置企業級寬螢幕版面
st.set_page_config(layout="wide", page_title="TQAI Pro Enterprise v2.5", page_icon="🏦")

with st.sidebar:
    st.header("⚙️ TQAI 決策中樞")
    mode = st.radio("功能選擇", ["🔍 個股深度分析", "📡 全台股掃描"])
    st.markdown("---")
    
    if mode == "🔍 個股深度分析":
        ticker = st.text_input("輸入股票代碼 (例: 2330)", value="2330")
        run_btn = st.button("🚀 啟動 AI 多智能體分析", use_container_width=True, type="primary")
        use_cache = st.checkbox("🗄️ 使用資料庫快取 (建議開啟)", value=True)
        force_refresh = st.button("🔄 強制重新抓取最新資料", use_container_width=True)
        scan_btn = False
    else:
        st.caption("預設使用台股常見權值股／熱門股觀察名單，也可以自行輸入想掃描的股票代碼。")
        custom_list = st.text_area(
            "自訂股票清單（逗號分隔，留空則使用預設清單）",
            value="", placeholder="例如：2330,2317,2454"
        )
        top_n = st.slider("顯示前 N 名", min_value=5, max_value=30, value=10)
        use_cache_scan = st.checkbox("🗄️ 使用資料庫快取 (建議開啟)", value=True, key="scan_cache")
        scan_btn = st.button("📡 啟動全台股掃描", use_container_width=True, type="primary")
        run_btn = False
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
if mode == "📡 全台股掃描" and scan_btn:
    if custom_list.strip():
        scan_tickers = [t.strip() for t in custom_list.split(",") if t.strip()]
    else:
        scan_tickers = ScannerEngine.DEFAULT_WATCHLIST
    
    st.markdown(f"## 📡 全台股掃描戰情室 (共 {len(scan_tickers)} 檔)")
    
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
    
    if result_df.empty:
        st.error("掃描失敗，所有股票皆無法取得資料，請確認代碼是否正確或網路連線。")
    else:
        st.success(f"✅ 掃描完成，成功 {len(result_df)} 檔，失敗 {len(error_df)} 檔。")
        
        st.markdown(f"### 🏆 Top {top_n} 排行榜（依 AI Score 排序）")
        top_df = ScannerEngine.get_top_n(result_df, top_n)
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 🏭 產業中心 (Industry Center) — 產業強弱排名")
        industry_df = IndustryEngine.rank_industries(result_df)
        if industry_df.empty:
            st.caption("⚠️ 尚無法產生產業排名（掃描結果為空，或股票代碼皆不在預設產業對照表中）。")
        else:
            st.dataframe(industry_df, use_container_width=True, hide_index=True)
            
            industry_options = industry_df["產業"].tolist()
            selected_industry = st.selectbox("🔍 點選查看該產業內的成分股表現", industry_options)
            if selected_industry:
                constituents = IndustryEngine.get_industry_constituents(result_df, selected_industry)
                st.dataframe(constituents, use_container_width=True, hide_index=True)
        
        with st.expander("📋 查看完整掃描結果"):
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        if not error_df.empty:
            with st.expander(f"⚠️ 掃描失敗清單 ({len(error_df)} 檔)"):
                st.dataframe(error_df, use_container_width=True, hide_index=True)

# ==========================================
# 🔍 個股深度分析模式
# ==========================================
if mode == "🔍 個股深度分析" and run_btn:
    with st.spinner("啟動底層特徵引擎與 Agent 辯論中..."):
        # AI Decision Pipeline 流水線執行
        df = DataEngine.get_stock_data(ticker, use_cache=use_cache)
        df = IndicatorEngine.add_indicators(df)
        df = StructureEngine.add_swing_points(df)
        df = RiskEngine.add_risk_metrics(df)
        df = StrategyEngine.generate_signals(df)
        df = EvidenceEngine.add_evidence(df)
        df, report = BacktestEngine.run_backtest(df)
        
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
        
        # ==========================================
        # 終極防禦工程：欄位對齊與安全檢查 (避免 KeyError)
        # ==========================================
        # 當底層引擎尚未升級至包含多智能體欄位時，自動進行安全初始化
        fallback_atr = df['atr_14'] if 'atr_14' in df.columns else df['close'] * 0.02
        
        required_columns_with_defaults = {
            'market_regime': '🔄 盤整或未知狀態 (請更新 strategy_engine.py)',
            'ai_score': 50.0,
            'confidence': 'Medium (中性)',
            'action_guide': '👀 震盪觀望 (多空拉鋸，建議控制倉位)',
            'bull_reason': '⚠️ 多頭辯護因子未完全載入，請確認底層引擎版本。',
            'bear_reason': '⚠️ 空頭辯護因子未完全載入，請確認底層引擎版本。',
            'risk_reason': '⚠️ 風控審查因子未完全載入，請確認底層引擎版本。',
            'stop_loss': df['close'] - (1.5 * fallback_atr),
            'target_1': df['close'] + (2.0 * fallback_atr),
            'target_2': df['close'] + (4.0 * fallback_atr),
            'zigzag': np.nan,
            'confidence_pct': 50.0,
            'data_quality_pct': 100.0,
            'confidence_label': '🟡 中性信心（請更新 evidence_engine.py）',
            'volatility_annualized': np.nan,
            'rolling_mdd_60d': np.nan,
            'var_95_pct': np.nan,
            'var_99_pct': np.nan,
        }
        
        for col, default_val in required_columns_with_defaults.items():
            if col not in df.columns:
                df[col] = default_val
        
        latest = df.iloc[-1]
        
        try:
            evidence_list = EvidenceEngine.get_evidence_list(df, -1)
        except Exception:
            evidence_list = []
        
        # ==========================================
        # Dashboard Layer: 戰情室視覺化
        # ==========================================
        st.markdown(f"## 📊 標的 `{ticker}` 企業級分析戰情室")
        
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