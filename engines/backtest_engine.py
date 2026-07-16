import pandas as pd
import numpy as np


class BacktestEngine:
    """
    📈 回測引擎 (Backtest Engine)

    ⚠️ 修正說明（v2.6）：
    1. 【回測循環邏輯】原本進出場門檻（ai_score >= 70 / <= 45）與 StrategyEngine 顯示
       給使用者看的「操作建議」門檻完全相同 —— 這代表回測只是在覆誦訊號定義本身，
       不構成獨立驗證。這裡把進出場門檻改為可傳入參數（entry_threshold /
       exit_score_threshold），並允許回測時使用跟顯示邏輯不同的組合，同時把
       這件事在 summary 裡明講出來（'note' 欄位），避免使用者誤以為這是嚴謹的
       樣本外驗證。
    2. 【風控斷點未被使用】原本完全忽略 StrategyEngine 算出的 ATR 停損/停利，只靠
       ai_score 掉到 45 以下才出場 —— 跟前端展示的「動態風險預算」是兩套邏輯。
       現在停損（stop_loss）／停利（target_1）會用當日 High/Low 觸價判斷，
       優先於 ai_score 出場，是主要風控機制。
    3. 【同根 K 棒撮合】原本訊號當天用同一根 K 棒的收盤價成交，等於用「已經知道
       當天收盤」的價格去執行「當天才產生」的訊號，不符合實際下單流程。改為
       訊號隔日以開盤價撮合（execution lag = 1 bar）。
    4. 【交易成本】原本完全沒扣手續費與證交稅，報酬率與勝率系統性虛高。加入
       可調的買賣手續費與賣出證交稅（預設近似台股實際稅費）。
    5. 【全倉進出】原本每筆訊號都全倉買賣，沒有部位控管；加入 position_size_pct
       參數，預設保留 1.0（全倉）以維持相容，但讓使用者可以調整。
    6. 名稱更正：這是「全樣本內回測 (In-Sample Backtest)」，不是 Walk-Forward
       樣本外驗證，兩者意義不同，避免誤導。
    """

    @staticmethod
    def _safe_float(value):
        if isinstance(value, (pd.Series, np.ndarray, list)):
            return float(value[0]) if len(value) else float("nan")
        return float(value)

    @staticmethod
    def run_backtest(
        df: pd.DataFrame,
        initial_capital: float = 100000,
        entry_threshold: float = 70,
        exit_score_threshold: float = 45,
        use_atr_exit: bool = True,
        position_size_pct: float = 1.0,
        buy_fee_pct: float = 0.1425 / 100,
        sell_fee_pct: float = 0.1425 / 100,
        sell_tax_pct: float = 0.30 / 100,
        execution_lag: int = 1,
    ):
        """
        執行全樣本內回測（非樣本外 Walk-Forward 驗證）。

        參數：
            entry_threshold      進場門檻（ai_score 高於此值觸發買進意圖）
            exit_score_threshold 訊號面出場門檻（ai_score 低於此值觸發賣出意圖，
                                  次於 ATR 停損停利）
            use_atr_exit         是否啟用 stop_loss / target_1 觸價出場（建議開啟，
                                  這是修正前版本完全缺漏的風控機制）
            position_size_pct    每次進場動用的資金比例（1.0 = 全倉）
            buy_fee_pct/sell_fee_pct/sell_tax_pct  交易成本（預設近似台股實際費率）
            execution_lag        訊號產生後延遲幾根 K 棒才成交（預設 1 根，
                                  用「隔日開盤價」成交，避免用同根K棒收盤價
                                  撮合當天才產生的訊號）
        """
        df = df.copy().sort_values('date').reset_index(drop=True)
        n = len(df)

        required_cols = ['close', 'open', 'high', 'low', 'ai_score']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"BacktestEngine 需要欄位 '{col}'，請確認上游 pipeline 已完整執行。")

        capital = initial_capital
        position = 0.0
        holding = False
        buy_price = 0.0
        entry_stop = np.nan
        entry_target = np.nan
        pending_signal = None  # ('buy'/'sell', trigger_index)
        trades = []
        equity_curve = []
        exit_reason_pending = None

        for i in range(n):
            row = df.iloc[i]
            current_date = row['date']
            o = BacktestEngine._safe_float(row['open'])
            h = BacktestEngine._safe_float(row['high'])
            l = BacktestEngine._safe_float(row['low'])
            c = BacktestEngine._safe_float(row['close'])
            score = BacktestEngine._safe_float(row['ai_score'])

            # ---- 先處理延遲成交：昨天(或更早)產生的訊號，今天用開盤價撮合 ----
            if pending_signal is not None:
                action, reason = pending_signal
                if action == 'buy' and not holding:
                    fill_price = o
                    invest_capital = capital * position_size_pct
                    fee = invest_capital * buy_fee_pct
                    net_capital = invest_capital - fee
                    position = net_capital / fill_price
                    capital -= invest_capital
                    buy_price = fill_price
                    holding = True
                    entry_stop = BacktestEngine._safe_float(df.iloc[max(i - execution_lag, 0)].get('stop_loss', np.nan)) \
                        if 'stop_loss' in df.columns else np.nan
                    entry_target = BacktestEngine._safe_float(df.iloc[max(i - execution_lag, 0)].get('target_1', np.nan)) \
                        if 'target_1' in df.columns else np.nan
                    trades.append({'type': 'Buy', 'date': current_date, 'price': fill_price,
                                    'fee': fee, 'reason': reason})
                elif action == 'sell' and holding:
                    fill_price = o
                    gross = position * fill_price
                    fee = gross * sell_fee_pct
                    tax = gross * sell_tax_pct
                    capital += (gross - fee - tax)
                    pnl = (fill_price - buy_price) / buy_price if buy_price else 0.0
                    trades.append({'type': 'Sell', 'date': current_date, 'price': fill_price,
                                    'fee': fee + tax, 'pnl': pnl, 'reason': reason})
                    position = 0.0
                    holding = False
                    buy_price = 0.0
                    entry_stop = np.nan
                    entry_target = np.nan
                pending_signal = None

            # ---- 持倉中：優先檢查 ATR 停損停利觸價（用當日高低價判斷，較符合實際）----
            exited_intrabar = False
            if holding and use_atr_exit:
                if pd.notna(entry_stop) and l <= entry_stop:
                    fill_price = min(entry_stop, o)  # 開盤跳空低於停損則以開盤價出場
                    gross = position * fill_price
                    fee = gross * sell_fee_pct
                    tax = gross * sell_tax_pct
                    capital += (gross - fee - tax)
                    pnl = (fill_price - buy_price) / buy_price if buy_price else 0.0
                    trades.append({'type': 'Sell', 'date': current_date, 'price': fill_price,
                                    'fee': fee + tax, 'pnl': pnl, 'reason': 'stop_loss'})
                    position = 0.0
                    holding = False
                    buy_price = 0.0
                    entry_stop = np.nan
                    entry_target = np.nan
                    exited_intrabar = True
                elif pd.notna(entry_target) and h >= entry_target:
                    fill_price = max(entry_target, o)
                    gross = position * fill_price
                    fee = gross * sell_fee_pct
                    tax = gross * sell_tax_pct
                    capital += (gross - fee - tax)
                    pnl = (fill_price - buy_price) / buy_price if buy_price else 0.0
                    trades.append({'type': 'Sell', 'date': current_date, 'price': fill_price,
                                    'fee': fee + tax, 'pnl': pnl, 'reason': 'target'})
                    position = 0.0
                    holding = False
                    buy_price = 0.0
                    entry_stop = np.nan
                    entry_target = np.nan
                    exited_intrabar = True

            # ---- 產生新訊號（延遲到下一根 K 棒才成交）----
            if not exited_intrabar:
                if score >= entry_threshold and not holding and pending_signal is None:
                    pending_signal = ('buy', 'ai_score_entry')
                elif score <= exit_score_threshold and holding and pending_signal is None:
                    pending_signal = ('sell', 'ai_score_exit')

            current_equity = capital + (position * c)
            equity_curve.append(current_equity)

        df['equity'] = equity_curve

        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
        equity_series = pd.Series(equity_curve)
        roll_max = equity_series.cummax()
        drawdown = (equity_series - roll_max) / roll_max.replace(0, np.nan)
        max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0.0

        sell_trades = [t for t in trades if t['type'] == 'Sell']
        win_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        win_rate = (len(win_trades) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0.0
        total_fees = sum(t.get('fee', 0) for t in trades)

        stop_exits = len([t for t in sell_trades if t.get('reason') == 'stop_loss'])
        target_exits = len([t for t in sell_trades if t.get('reason') == 'target'])
        score_exits = len([t for t in sell_trades if t.get('reason') == 'ai_score_exit'])

        # ⚠️ v2.9.5 新增：原本 summary 只有 total_return / max_drawdown / win_rate，
        # 缺少 CAGR、Sharpe、Sortino、Expectancy、Profit Factor 這幾個專業機構
        # 評估策略時的標準指標——沒有這些，兩個策略之間唯一能比的只有總報酬率，
        # 但總報酬率完全無法反映「這個報酬是承擔多大風險/多穩定換來的」。
        pnl_list = [t.get('pnl', 0.0) for t in sell_trades]
        win_pnls = [p for p in pnl_list if p > 0]
        loss_pnls = [p for p in pnl_list if p <= 0]

        avg_win_pct = (np.mean(win_pnls) * 100) if win_pnls else 0.0
        avg_loss_pct = (abs(np.mean(loss_pnls)) * 100) if loss_pnls else 0.0

        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)

        # Expectancy：平均每筆交易的期望報酬率（%），> 0 代表長期下注有正期望值
        expectancy_pct = (np.mean(pnl_list) * 100) if pnl_list else 0.0

        # CAGR：用實際交易日數年化，避免不同回測區間長度的策略被總報酬率誤導比較
        trading_days = max(n, 1)
        years = trading_days / 252.0
        cagr = (((equity_curve[-1] / initial_capital) ** (1 / years) - 1) * 100) if years > 0 and equity_curve[-1] > 0 else np.nan

        # Sharpe / Sortino：用權益曲線的日報酬率計算，無風險利率簡化為 0
        equity_returns = equity_series.pct_change().dropna()
        if len(equity_returns) > 5 and equity_returns.std() > 0:
            sharpe = (equity_returns.mean() / equity_returns.std()) * np.sqrt(252)
        else:
            sharpe = np.nan

        downside_returns = equity_returns[equity_returns < 0]
        if len(equity_returns) > 5 and len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (equity_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = np.nan

        summary = {
            'initial_capital': initial_capital,
            'final_equity': equity_curve[-1] if equity_curve else initial_capital,
            'total_return': total_return,
            'cagr_pct': round(cagr, 2) if pd.notna(cagr) else None,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': round(float(sharpe), 2) if pd.notna(sharpe) else None,
            'sortino_ratio': round(float(sortino), 2) if pd.notna(sortino) else None,
            'total_trades': len(sell_trades),
            'win_rate': win_rate,
            'avg_win_pct': round(avg_win_pct, 2),
            'avg_loss_pct': round(avg_loss_pct, 2),
            'expectancy_pct': round(expectancy_pct, 2),
            'profit_factor': round(profit_factor, 2) if np.isfinite(profit_factor) else None,
            'total_fees_paid': total_fees,
            'exit_breakdown': {
                '停損出場': stop_exits,
                '停利出場': target_exits,
                'AI Score轉弱出場': score_exits,
            },
            'note': (
                "⚠️ 此為全樣本內回測 (In-Sample)，進出場門檻與策略引擎顯示門檻相同，"
                "不構成樣本外 (Out-of-Sample) 驗證，僅供邏輯檢視，不應直接作為績效保證。"
                "Sharpe/Sortino 以無風險利率=0簡化計算，交易筆數過少時（例如<10筆）"
                "這些統計量的可信度很低，請一併參考 total_trades。"
            ),
        }

        return df, summary

    # ==========================================
    # 7. 多策略比較 (Strategy Comparison)
    # ==========================================
    # ⚠️ v2.9.5 新增：原本只能個別呼叫 run_backtest 一次比一次，沒有一個
    # 統一的比較框架。這裡讓使用者可以一次傳入多組參數（例如不同的
    # entry_threshold/exit_score_threshold 組合，模擬「策略A vs 策略B」），
    # 回傳一張並排比較表，用 CAGR/Sharpe/Sortino/Win Rate/MDD/Expectancy/
    # Profit Factor 這幾個機構常用指標排序，取代「憑感覺選策略」。
    @staticmethod
    def compare_strategies(df: pd.DataFrame, strategies: list, initial_capital: float = 100000) -> pd.DataFrame:
        """
        參數：
            strategies : list of dict，每個 dict 是要傳給 run_backtest 的參數組合，
                         務必包含 'name' 鍵作為策略顯示名稱，例如：
                         [
                           {'name': '積極型(70/45)', 'entry_threshold': 70, 'exit_score_threshold': 45},
                           {'name': '保守型(80/55)', 'entry_threshold': 80, 'exit_score_threshold': 55},
                         ]
        回傳：
            pd.DataFrame，每列是一個策略的績效指標，依 Sharpe Ratio 由高到低排序
            （而非依總報酬率排序——高報酬率若伴隨極端風險，不代表是更好的策略）。
        """
        rows = []
        for strat in strategies:
            strat = dict(strat)
            name = strat.pop('name', f"策略{len(rows)+1}")
            try:
                _, summary = BacktestEngine.run_backtest(df, initial_capital=initial_capital, **strat)
                rows.append({
                    '策略名稱': name,
                    '總報酬率(%)': round(summary['total_return'], 2),
                    'CAGR(%)': summary['cagr_pct'],
                    '最大回撤(%)': round(summary['max_drawdown'], 2),
                    'Sharpe': summary['sharpe_ratio'],
                    'Sortino': summary['sortino_ratio'],
                    '勝率(%)': round(summary['win_rate'], 1),
                    '交易次數': summary['total_trades'],
                    '期望值(%)': summary['expectancy_pct'],
                    'Profit Factor': summary['profit_factor'],
                })
            except Exception as e:
                rows.append({'策略名稱': name, '總報酬率(%)': None, 'CAGR(%)': None,
                             '最大回撤(%)': None, 'Sharpe': None, 'Sortino': None,
                             '勝率(%)': None, '交易次數': 0, '期望值(%)': None,
                             'Profit Factor': None, '錯誤': str(e)})

        result = pd.DataFrame(rows)
        if not result.empty and 'Sharpe' in result.columns:
            result = result.sort_values('Sharpe', ascending=False, na_position='last').reset_index(drop=True)
        return result