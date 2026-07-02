import pandas as pd
import numpy as np

class IndicatorEngine:
    @staticmethod
    def add_indicators(df: pd.DataFrame):
        df = df.copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        
        # 1. 均線系統
        df["ema_8"] = c.ewm(span=8, adjust=False).mean()
        df["ema_21"] = c.ewm(span=21, adjust=False).mean()
        df["sma_20"] = c.rolling(20).mean()
        df["sma_60"] = c.rolling(60).mean()
        df["sma_120"] = c.rolling(120).mean()
        df["sma_200"] = c.rolling(200).mean()
        df["vma_20"] = v.rolling(20).mean()

        # 2. 布林通道與 RSI
        std = c.rolling(20).std()
        df["bb_upper"] = df["sma_20"] + std * 2
        df["bb_lower"] = df["sma_20"] - std * 2
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        df["rsi_14"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

        # 3. KD 與 MACD
        rsv = ((c - l.rolling(9).min()) / (h.rolling(9).max() - l.rolling(9).min() + 1e-9)) * 100
        df["k_9"] = rsv.ewm(alpha=1/3).mean()
        df["d_9"] = df["k_9"].ewm(alpha=1/3).mean()
        
        df["macd_dif"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd_dif"] - df["macd_dea"]

        # 4. ATR (真實波動幅度)
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # 5. OBV (能量潮指標)
        df['obv'] = (np.sign(c.diff()) * v).fillna(0).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()

        # ==========================================
        # 🚀 TQAI Pro 升級：市場特徵與狀態識別因子
        # ==========================================
        # 波動率特徵 (歷史波動率近似)
        df['volatility_ratio'] = (df['atr_14'] / c) * 100 
        
        # 季線斜率 (判斷宏觀趨勢動能)
        df['sma_60_slope'] = (df['sma_60'] - df['sma_60'].shift(5)) / (df['sma_60'].shift(5) + 1e-9) * 100
        
        # 量能特徵 (RVOL - 相對成交量)
        df['rvol'] = v / (df['vma_20'] + 1e-9)

        return df