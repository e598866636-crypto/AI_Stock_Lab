import io
from datetime import datetime, timedelta

import pandas as pd
import requests


class ChipEngine:
    """
    🏦 籌碼中心 (Chip Center)

    抓取台灣證券交易所 (TWSE) 公開資訊：
      - 三大法人買賣超（外資及陸資 / 投信 / 自營商）
      - 融資融券餘額

    以及台灣集中保管結算所 (TDCC) 開放資料：
      - 集保戶股權分散表（大戶持股／千張大戶集中度，v2.9 新增）

    限制：
      - 三大法人/融資融券目前僅支援「上市」股票（.TW），上櫃（.TWO）因資料來源不同（TPEx），暫不支援
      - 僅在交易日有資料；若查詢日為假日或盤後資料尚未公布，會自動往前找最近的交易日
      - 依賴 TWSE OpenAPI，若官方介面改版或服務異常，會回傳 status='unavailable'，不影響主程式運作
      - 股權分散表（TDCC）涵蓋上市/上櫃/興櫃全市場，見下方 get_shareholding_distribution() 說明的額外限制
    """

    TWSE_INSTITUTIONAL_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
    TWSE_MARGIN_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"

    # TDCC（臺灣集中保管結算所）集保戶股權分散表開放資料，全市場單一CSV，
    # 每週更新。來源：https://data.gov.tw/dataset/11452
    TDCC_SHAREHOLDING_URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"

    # ⚠️ 級距對照表誠實揭露：TDCC 原始CSV的「持股分級」欄位只給數字代碼
    # (1~17)，不含級距說明文字。這裡依業界慣例（Goodinfo/CMoney/神秘金字塔
    # 等第三方籌碼網站都採用同一套）補上人類可讀的級距標籤，這是市場共識
    # 慣例，不是TDCC官方在這份CSV裡直接提供的欄位——若集保結算所未來調整
    # 級距定義，這裡需要同步更新。第16級是罕見的極端值調整列，第17級固定
    # 是全部級距（1~16）的合計列（人數/股數/佔比100%），皆不計入「大戶」
    # 級距判斷。
    SHAREHOLDING_TIER_LABELS = {
        1: "1～999股", 2: "1,000～5,000股", 3: "5,001～10,000股",
        4: "10,001～15,000股", 5: "15,001～20,000股", 6: "20,001～30,000股",
        7: "30,001～40,000股", 8: "40,001～50,000股", 9: "50,001～100,000股",
        10: "100,001～200,000股", 11: "200,001～400,000股", 12: "400,001～600,000股",
        13: "600,001～800,000股", 14: "800,001～1,000,000股", 15: "1,000,001股以上",
        16: "調整列", 17: "合計",
    }
    # 千張大戶／大戶：持股 1,000,001股以上（= 1,000張以上），對應第15級。
    LARGE_HOLDER_TIER = 15

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
    # 全市場三大法人買賣超排行榜（v2.8 新增，對應選股學院「排行榜選股法」）
    # ==========================================
    # ⚠️ 合併說明：這個方法原本被誤放在一個沒有 .py 副檔名、未被任何地方
    # import 的孤兒檔案（Chip_engine___PY）裡，導致 app.py 實際 import 的
    # 這份 chip_engine.py 完全沒有這個方法 —— 「🏆 全市場排行榜」頁面因此
    # 100% 會拋出 AttributeError。現在正式合併進來，並保留原檔的孤兒版本
    # 供刪除。
    @staticmethod
    def get_market_wide_institutional_ranking(lookback_days: int = 10, top_n: int = 20,
                                               use_cache: bool = True, max_age_hours: float = 4,
                                               db_path: str = None):
        """
        🏆 全市場三大法人買賣超排行榜 (Market-wide Institutional Ranking)

        對應選股學院文件「排行榜選股法」：外資買賣超／投信買賣超／自營商買賣超
        排行榜，直接追蹤三大法人的資金流向。

        跟 get_institutional_snapshot()（查「單一股票」的三大法人買賣超）不同，
        這裡是對 _fetch_institutional_single_day() 抓回來的「當日全市場」資料
        做排序，取得買超/賣超前 N 名——複用同一支 TWSE API，不會額外增加對外
        請求次數。

        ⚠️ 資料範圍限制（沿用本類別 docstring 的既有限制）：
          1. TWSE T86 這支 API 只涵蓋「上市」股票，上櫃（TPEx）用的是不同的
             資料源，暫不支援，所以這份排行榜不包含上櫃股票。
          2. 只在交易日有資料；若查詢日為假日或盤後資料尚未公布，會自動
             往前找最近的交易日（跟其他 ChipEngine 方法一致）。
          3. 若 TWSE 官方介面改版或服務異常，會回傳 status='unavailable'，
             不影響呼叫端（app.py）繼續運作。

        ⚠️ 新增（快取）：這份「全市場當日排行」資料只跟日期有關，跟呼叫者
        是誰、要看第幾名都無關，很適合共用快取——原本每次點擊「抓取最新
        排行榜」都會重新對 TWSE 發一次請求，即使同一個交易日內反覆點擊也是
        如此。現在把「原始全市場資料 + 資料日期」快取進 DatabaseEngine 的
        kv_cache（預設新鮮期限 4 小時，盤中/盤後資料本來就不會頻繁更新，
        不需要比這更短的快取窗口），Top-N 排序與切片則每次都重新計算
        （便宜、且允許同一份快取資料配合不同 top_n 使用，不需要因為 top_n
        不同就重抓一次）。use_cache=False 可強制略過快取重新抓取。

        ⚠️ 新增（標的標籤）：加上「標的」欄位（格式 "[代碼] 名稱"），跟
        ScannerEngine/NameEngine 其他頁面的顯示格式一致。這裡直接用 TWSE
        當天回傳的官方名稱組字串，而不是查 NameEngine.NAME_MAP（那份只
        涵蓋本專案內建觀察名單的幾十檔，這份排行榜涵蓋全部上市股票，用
        NAME_MAP 會讓大部分股票顯示成「未知名稱」，反而更不準確）。

        回傳格式：
            {
                "status": "ok",
                "date": "2026-07-08",
                "total_stocks": 950,
                "rankings": {
                    "外資": {"買超前N名": df, "賣超前N名": df},
                    "投信": {"買超前N名": df, "賣超前N名": df},
                    "自營商": {"買超前N名": df, "賣超前N名": df},
                    "三大法人合計": {"買超前N名": df, "賣超前N名": df},
                }
            }
        """
        from engines.db_engine import DatabaseEngine

        cache_key = "chip_market_wide_institutional_raw"
        work = None
        date_str = None

        if use_cache:
            try:
                cached = DatabaseEngine.get_cache(cache_key, max_age_hours=max_age_hours, db_path=db_path)
                if cached:
                    payload = cached["payload"]
                    work = pd.DataFrame(payload["records"])
                    date_str = payload["date"]
            except Exception:
                work = None

        if work is None:
            for delta in range(lookback_days):
                d = datetime.now() - timedelta(days=delta)
                df = ChipEngine._fetch_institutional_single_day(d.strftime("%Y%m%d"))
                if df.empty:
                    continue

                code_col = ChipEngine._find_col(df.columns, "證券代號")
                name_col = ChipEngine._find_col(df.columns, "證券名稱")
                if code_col is None:
                    continue

                foreign_col = ChipEngine._find_col(df.columns, "外資及陸資買賣超股數") \
                    or ChipEngine._find_col(df.columns, "外資買賣超股數")
                trust_col = ChipEngine._find_col(df.columns, "投信買賣超股數")
                dealer_col = ChipEngine._find_col(df.columns, "自營商買賣超股數(自行買賣)") \
                    or ChipEngine._find_col(df.columns, "自營商買賣超股數")
                total_col = ChipEngine._find_col(df.columns, "三大法人買賣超股數")

                if not all([foreign_col, trust_col, dealer_col]):
                    # 當天頁面欄位對不上（例如TWSE偶爾微調格式），換下一個交易日再試，
                    # 不要用錯誤對應的欄位硬算，避免排行榜資料失真。
                    continue

                candidate = pd.DataFrame({
                    "代碼": df[code_col].astype(str).str.strip(),
                    "名稱": df[name_col].astype(str).str.strip() if name_col else "",
                    "外資買賣超": df[foreign_col].apply(ChipEngine._to_int),
                    "投信買賣超": df[trust_col].apply(ChipEngine._to_int),
                    "自營商買賣超": df[dealer_col].apply(ChipEngine._to_int),
                })
                candidate["三大法人合計買賣超"] = (
                    df[total_col].apply(ChipEngine._to_int) if total_col
                    else candidate["外資買賣超"] + candidate["投信買賣超"] + candidate["自營商買賣超"]
                )

                # 過濾掉代碼格式異常的雜項列（權證、受益證券等非普通股/ETF代碼）
                candidate = candidate[candidate["代碼"].str.match(r"^[0-9]{4,6}$")].reset_index(drop=True)
                if candidate.empty:
                    continue

                work = candidate
                date_str = d.strftime("%Y-%m-%d")
                break

            if work is None:
                return {
                    "status": "unavailable",
                    "message": "⚠️ 近期無法取得全市場三大法人買賣超排行榜（可能是連續假日、TWSE服務異常，或近期非交易日）。",
                }

            if use_cache:
                try:
                    DatabaseEngine.set_cache(
                        cache_key,
                        {"date": date_str, "records": work.to_dict(orient="records")},
                        db_path=db_path,
                    )
                except Exception:
                    pass

        work = work.copy()
        work["標的"] = "[" + work["代碼"].astype(str) + "] " + work["名稱"].astype(str)

        rankings = {}
        for label, col in [
            ("外資", "外資買賣超"), ("投信", "投信買賣超"),
            ("自營商", "自營商買賣超"), ("三大法人合計", "三大法人合計買賣超"),
        ]:
            buy_top = work.sort_values(col, ascending=False).head(top_n).reset_index(drop=True)
            sell_top = work.sort_values(col, ascending=True).head(top_n).reset_index(drop=True)
            rankings[label] = {"買超前N名": buy_top, "賣超前N名": sell_top}

        return {
            "status": "ok",
            "date": date_str,
            "total_stocks": len(work),
            "rankings": rankings,
        }

    # ==========================================
    # 大戶持股／千張大戶分析（v2.9 新增，資料源：TDCC 集保戶股權分散表）
    # ==========================================
    @staticmethod
    def _fetch_shareholding_distribution_full(use_cache: bool = True, max_age_hours: float = 20) -> pd.DataFrame:
        """
        抓取 TDCC 集保戶股權分散表「全市場單一CSV」快照（每週更新，涵蓋
        上市/上櫃/興櫃全部證券）。這份CSV涵蓋全市場所有證券、每檔證券
        17個級距列，逐股票查詢的話沒辦法只抓單一股票，每次都要下載全部——
        因此這裡快取「整份原始資料」而不是「單一股票的結果」，同一週期
        (max_age_hours預設20小時，略短於一天，避免跨日快取到舊的一週資料)
        內重複查詢不同股票都只需要下載一次。
        """
        from engines.db_engine import DatabaseEngine

        cache_key = "tdcc_shareholding_distribution_raw"
        if use_cache:
            cached = DatabaseEngine.get_cache(cache_key, max_age_hours=max_age_hours)
            if cached is not None:
                try:
                    return pd.DataFrame(cached["payload"]["records"])
                except Exception:
                    pass

        try:
            resp = requests.get(ChipEngine.TDCC_SHAREHOLDING_URL, headers=ChipEngine.HEADERS, timeout=30)
            resp.encoding = "utf-8-sig"
            df = pd.read_csv(io.StringIO(resp.text), dtype=str)
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        df.columns = [str(c).strip() for c in df.columns]
        expected_cols = {"資料日期", "證券代號", "持股分級", "人數", "股數", "占集保庫存數比例%"}
        if not expected_cols.issubset(set(df.columns)):
            # TDCC 改版欄位名稱或格式跟預期不符，如實回傳空表，不要用錯位的
            # 欄位硬算，避免產生看似正常、實則張冠李戴的數字。
            return pd.DataFrame()

        df["證券代號"] = df["證券代號"].astype(str).str.strip()
        for col in ["持股分級", "人數", "股數"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["占集保庫存數比例%"] = pd.to_numeric(df["占集保庫存數比例%"], errors="coerce")
        df = df.dropna(subset=["持股分級"])
        df["持股分級"] = df["持股分級"].astype(int)

        if use_cache:
            try:
                DatabaseEngine.set_cache(
                    cache_key, {"records": df.to_dict(orient="records")}, db_path=None
                )
            except Exception:
                pass

        return df

    @staticmethod
    def get_shareholding_distribution(stock_code: str, use_cache: bool = True) -> dict:
        """
        🏦 大戶持股／千張大戶分析 (Shareholding Concentration Analysis)

        資料源：TDCC（臺灣集中保管結算所）集保戶股權分散表開放資料
        （https://opendata.tdcc.com.tw/getOD.ashx?id=1-5），涵蓋上市/上櫃/
        興櫃全市場證券，每週更新。

        ⚠️ 誠實揭露：
          1. 「大戶」定義採業界慣例：持股 1,000,001股以上（= 1,000張以上，
             對應第15級），這是市場慣用定義，不是嚴謹的法定分類。
          2. 這裡只反映「集保帳戶」的持股分布，不是「實質受益人」——同一
             個人可能透過信託、代操等方式分散在不同集保帳戶，實際集中度
             可能比表面數字更高或更低，此為 TDCC 資料本身的固有限制，
             不是本引擎的實作問題。
          3. 每週更新一次，不是即時資料；當週資料通常會落後1~2個交易日
             （TDCC官方說明的結算時間差），不適合用來判斷「今天」的籌碼
             變化，是中長期籌碼集中度的參考指標。
          4. 歷史趨勢（見 get_shareholding_trend()）是本系統自己累積的
             快照記錄，不是 TDCC 官方提供的歷史資料——TDCC 這份開放資料
             CSV 本身只給「最新一週」，沒有歷史區間查詢，所以只能從
             「系統開始使用之後」才逐週累積出趨勢，剛啟用時只會有一筆。
        """
        stock_code_clean = str(stock_code).split(".")[0].strip()
        full_df = ChipEngine._fetch_shareholding_distribution_full(use_cache=use_cache)
        if full_df.empty:
            return {"status": "unavailable", "message": "⚠️ 暫時無法取得集保戶股權分散表資料（TDCC開放資料下載失敗或格式異動）。"}

        match = full_df[full_df["證券代號"] == stock_code_clean].copy()
        if match.empty:
            return {"status": "unavailable", "message": f"⚠️ 查無代碼 {stock_code_clean} 的股權分散表資料，請確認代碼是否正確。"}

        match = match.sort_values("持股分級")
        total_row = match[match["持股分級"] == 17]
        detail_tiers = match[match["持股分級"] <= 15].copy()
        detail_tiers["級距"] = detail_tiers["持股分級"].map(ChipEngine.SHAREHOLDING_TIER_LABELS)

        large_holder_row = match[match["持股分級"] == ChipEngine.LARGE_HOLDER_TIER]
        large_holder_pct = float(large_holder_row["占集保庫存數比例%"].iloc[0]) if not large_holder_row.empty else 0.0
        large_holder_count = int(large_holder_row["人數"].iloc[0]) if not large_holder_row.empty else 0

        total_holders = int(total_row["人數"].iloc[0]) if not total_row.empty else int(detail_tiers["人數"].sum())
        total_shares = int(total_row["股數"].iloc[0]) if not total_row.empty else int(detail_tiers["股數"].sum())

        raw_date = str(match["資料日期"].iloc[0])
        date_fmt = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if len(raw_date) == 8 else raw_date

        flags = []
        if large_holder_pct >= 60:
            flags.append(f"🔴 千張大戶(≥1,000張)持股佔比高達 {large_holder_pct:.1f}%，籌碼高度集中在大戶手上")
        elif large_holder_pct >= 40:
            flags.append(f"🟡 千張大戶持股佔比 {large_holder_pct:.1f}%，籌碼集中度中等偏高")
        else:
            flags.append(f"ℹ️ 千張大戶持股佔比 {large_holder_pct:.1f}%，籌碼相對分散")
        flags.append("⚠️ 每週更新一次、反映集保帳戶而非實質受益人，僅供中長期籌碼集中度參考，不適合判斷單日籌碼變化。")

        # 累積這一週的摘要進本地歷史記錄（見上方 class docstring 的趨勢限制說明）
        try:
            from engines.db_engine import DatabaseEngine
            DatabaseEngine.save_shareholding_snapshot(
                date_fmt, stock_code_clean, large_holder_pct, total_holders, total_shares
            )
        except Exception:
            pass

        return {
            "status": "ok",
            "date": date_fmt,
            "total_holders": total_holders,
            "total_shares": total_shares,
            "tiers": detail_tiers[["級距", "人數", "股數", "占集保庫存數比例%"]].reset_index(drop=True),
            "large_holder_pct": round(large_holder_pct, 2),
            "large_holder_count": large_holder_count,
            "flags": flags,
        }

    @staticmethod
    def get_shareholding_trend(stock_code: str, weeks: int = 12) -> pd.DataFrame:
        """
        回傳本系統自己累積的「大戶持股佔比」歷史趨勢（見
        get_shareholding_distribution() docstring 對趨勢資料來源限制的說明）。
        剛開始使用本系統時歷史筆數會很少，屬正常現象，會隨著每週重複
        查詢逐漸累積。
        """
        from engines.db_engine import DatabaseEngine
        stock_code_clean = str(stock_code).split(".")[0].strip()
        try:
            return DatabaseEngine.load_shareholding_history(stock_code_clean, weeks=weeks)
        except Exception:
            return pd.DataFrame()

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