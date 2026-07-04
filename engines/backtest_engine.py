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

        summary = {
            'initial_capital': initial_capital,
            'final_equity': equity_curve[-1] if equity_curve else initial_capital,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(sell_trades),
            'win_rate': win_rate,
            'total_fees_paid': total_fees,
            'exit_breakdown': {
                '停損出場': stop_exits,
                '停利出場': target_exits,
                'AI Score轉弱出場': score_exits,
            },
            'note': (
                "⚠️ 此為全樣本內回測 (In-Sample)，進出場門檻與策略引擎顯示門檻相同，"
                "不構成樣本外 (Out-of-Sample) 驗證，僅供邏輯檢視，不應直接作為績效保證。"
            ),
        }

        return df, summary