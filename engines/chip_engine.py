from datetime import datetime, timedelta

import pandas as pd
import requests


class ChipEngine:
    """
    🏦 籌碼中心 (Chip Center)

    抓取台灣證券交易所 (TWSE) 公開資訊：
      - 三大法人買賣超（外資及陸資 / 投信 / 自營商）
      - 融資融券餘額

    限制：
      - 目前僅支援「上市」股票（.TW），上櫃（.TWO）因資料來源不同（TPEx），暫不支援
      - 僅在交易日有資料；若查詢日為假日或盤後資料尚未公布，會自動往前找最近的交易日
      - 依賴 TWSE OpenAPI，若官方介面改版或服務異常，會回傳 status='unavailable'，不影響主程式運作
    """

    TWSE_INSTITUTIONAL_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
    TWSE_MARGIN_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    }

    # ==========================================
    # 工具方法
    # ==========================================
    @staticmethod
    def _to_int(value):
        try:
            return int(str(value).replace(",", "").replace(" ", "").replace("+", ""))
        except Exception:
            return 0

    @staticmethod
    def _find_col(columns, keyword):
        return next((c for c in columns if keyword in c), None)

    # ==========================================
    # 三大法人買賣超
    # ==========================================
    @staticmethod
    def _fetch_institutional_single_day(date_str: str) -> pd.DataFrame:
        try:
            resp = requests.get(
                ChipEngine.TWSE_INSTITUTIONAL_URL,
                params={"date": date_str, "selectType": "ALL", "response": "json"},
                headers=ChipEngine.HEADERS, timeout=10,
            )
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        if data.get("stat") != "OK" or not data.get("data"):
            return pd.DataFrame()

        df = pd.DataFrame(data["data"], columns=data.get("fields", []))
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def get_institutional_snapshot(stock_code: str, lookback_days: int = 10):
        """往前找最近 lookback_days 天內第一個有資料的交易日，回傳該股票的三大法人買賣超明細。"""
        stock_code = str(stock_code).split(".")[0].strip()

        for delta in range(lookback_days):
            d = datetime.now() - timedelta(days=delta)
            df = ChipEngine._fetch_institutional_single_day(d.strftime("%Y%m%d"))
            if df.empty:
                continue

            code_col = ChipEngine._find_col(df.columns, "證券代號")
            if code_col is None:
                continue

            df[code_col] = df[code_col].astype(str).str.strip()
            match = df[df[code_col] == stock_code]
            if match.empty:
                continue

            row = match.iloc[0]

            def g(keyword):
                col = ChipEngine._find_col(df.columns, keyword)
                return ChipEngine._to_int(row[col]) if col else 0

            foreign_net = g("外資及陸資買賣超股數") or g("外資買賣超股數")
            trust_net = g("投信買賣超股數")
            dealer_net = g("自營商買賣超股數(自行買賣)") or g("自營商買賣超股數")
            total_net = g("三大法人買賣超股數") or (foreign_net + trust_net + dealer_net)

            return {
                "date": d.strftime("%Y-%m-%d"),
                "foreign_net": foreign_net,
                "trust_net": trust_net,
                "dealer_net": dealer_net,
                "total_net": total_net,
            }

        return None

    # ==========================================
    # 融資融券
    # ==========================================
    @staticmethod
    def _fetch_margin_single_day(date_str: str) -> pd.DataFrame:
        try:
            resp = requests.get(
                ChipEngine.TWSE_MARGIN_URL,
                params={"date": date_str, "selectType": "ALL", "response": "json"},
                headers=ChipEngine.HEADERS, timeout=10,
            )
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        if data.get("stat") != "OK" or not data.get("data"):
            return pd.DataFrame()

        df = pd.DataFrame(data["data"], columns=data.get("fields", []))
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def get_margin_snapshot(stock_code: str, lookback_days: int = 10):
        stock_code = str(stock_code).split(".")[0].strip()

        for delta in range(lookback_days):
            d = datetime.now() - timedelta(days=delta)
            df = ChipEngine._fetch_margin_single_day(d.strftime("%Y%m%d"))
            if df.empty:
                continue

            code_col = ChipEngine._find_col(df.columns, "代號")
            if code_col is None:
                continue

            df[code_col] = df[code_col].astype(str).str.strip()
            match = df[df[code_col] == stock_code]
            if match.empty:
                continue

            row = match.iloc[0]

            def g(keyword):
                col = ChipEngine._find_col(df.columns, keyword)
                return ChipEngine._to_int(row[col]) if col else 0

            margin_balance = g("融資今日餘額") or g("融資餘額")
            margin_change = g("融資增減")
            short_balance = g("融券今日餘額") or g("融券餘額")
            short_change = g("融券增減")

            return {
                "date": d.strftime("%Y-%m-%d"),
                "margin_balance": margin_balance,
                "margin_change": margin_change,
                "short_balance": short_balance,
                "short_change": short_change,
            }

        return None

    # ==========================================
    # 整合報告（供 Dashboard 顯示）
    # ==========================================
    @staticmethod
    def build_chip_report(stock_code: str):
        stock_code_clean = str(stock_code).split(".")[0].strip()

        try:
            institutional = ChipEngine.get_institutional_snapshot(stock_code_clean)
        except Exception:
            institutional = None

        try:
            margin = ChipEngine.get_margin_snapshot(stock_code_clean)
        except Exception:
            margin = None

        if institutional is None and margin is None:
            return {
                "status": "unavailable",
                "message": "⚠️ 暫時無法取得籌碼資料（可能為上櫃股票、TWSE 服務異常，或近期非交易日）。",
            }

        flags = []
        if institutional:
            f_net, t_net = institutional["foreign_net"], institutional["trust_net"]
            if f_net > 0 and t_net > 0:
                flags.append("🟢 外資與投信同步買超，籌碼面偏多")
            elif f_net < 0 and t_net < 0:
                flags.append("🔴 外資與投信同步賣超，籌碼面偏空")
            elif f_net * t_net < 0:
                flags.append("🟡 外資與投信買賣方向分歧，籌碼面不明確")

        if margin:
            if margin["margin_change"] < 0 and margin["short_change"] > 0:
                flags.append("⚠️ 融資減少、融券增加，散戶轉趨保守／看空")
            elif margin["margin_change"] > 0 and margin["short_change"] < 0:
                flags.append("ℹ️ 融資增加、融券減少，散戶轉趨樂觀")

        if not flags:
            flags.append("ℹ️ 籌碼動向中性，無明顯一致訊號")

        return {
            "status": "ok",
            "institutional": institutional,
            "margin": margin,
            "flags": flags,
        }