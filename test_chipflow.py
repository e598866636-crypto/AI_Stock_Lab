"""手動驗證 ChipFlowEngine 的即時診斷腳本。

直接執行 `python test_chipflow.py` 才會連線抓取近期資料；被 pytest 收集/匯入時
不應自動發出網路請求，以免測試收集卡住或污染其他測試。
"""

from engines.chip_flow_engine import ChipFlowEngine


def main():
    result = ChipFlowEngine.compute_institutional_streaks(lookback_days=30)
    print("status:", result.get('status'))
    print("days_used:", result.get('days_used'))

    streaks = result.get('streaks', {})
    print(f"共 {len(streaks)} 檔股票有連續天數資料，隨機挑 5 檔看看：")

    sample = list(streaks.items())[:5]
    for code, streak in sample:
        print(code, streak)


if __name__ == "__main__":
    main()
