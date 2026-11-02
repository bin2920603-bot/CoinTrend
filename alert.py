import os
import numpy as np
import requests
import main as m
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SKIP = {("EMA50", "L")}   # 성적표에서 손해였던 조합은 알림에서 뺌
SHOW = 60                 # 차트에 보여줄 일봉 개수
MAX_ALERTS = 20           # 한 번에 보낼 최대 알림 수
SMALL_N = 30              # 신호가 이보다 적으면 '표본 적음' 표시


def is_new(d, ln, i, s):
    """같은 선·방향 신호가 직전 며칠 안에 이미 있었으면 새 신호가 아님"""
    for k in range(1, m.COOLDOWN):
        if m.check_signal(d, ln, i - k) == s:
            return False
    return True


def find_alerts(inst, d, lines, at):
    i = len(d["c"]) - 1
    if np.isnan(at[i]):
        return []
    entry = d["c"][i]
    risk = min(m.ATR_MULT * at[i] / entry, m.MAX_STOP)
    if risk * m.RR < m.MIN_TP:
        return []
    found = {}
    for name in m.LINES:
        s = m.check_signal(d, lines[name], i)
        if not s or (name, s) in SKIP:
            continue
        if not is_new(d, lines[name], i, s):
            continue
        found.setdefault(s, []).append(name)
    out = []
    for s, names in found.items():
        if s == "L":
            sl, tp = entry * (1 - risk), entry * (1 + risk * m.RR)
        else:
            sl, tp = entry * (1 + risk), entry * (1 - risk * m.RR)
        out.append({"inst": inst, "side": s, "names": names, "entry": entry,
                    "sl": sl, "tp": tp, "d": d, "lines": lines, "idx": i})
    return out


def draw_alert(a, path):
    d, i = a["d"], a["idx"]
    lo = max(0, i - SHOW + 1)
    xs = np.arange(lo, i + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for x in xs:
        o, h, l, c = d["o"][x], d["h"][x], d["l"][x], d["c"][x]
        col = "#2e9e5b" if c >= o else "#d64545"
        ax.vlines(x, l, h, color=col, linewidth=1)
        ax.bar(x, max(abs(c - o), c * 0.0005), bottom=min(o, c), width=0.6, color=col)
    for name in a["names"]:
        ax.plot(xs, a["lines"][name][lo:i + 1], label=name, linewidth=1.8)
    ax.axhline(a["entry"], color="#555555", linewidth=1, label="Entry")
    ax.axhline(a["sl"], color="#d64545", linestyle="--", linewidth=1, label="Stop")
    ax.axhline(a["tp"], color="#2e9e5b", linestyle="--", linewidth=1, label="Target")
    side = "LONG" if a["side"] == "L" else "SHORT"
    ax.set_title("%s %s (daily)" % (a["inst"].replace("-USDT-SWAP", ""), side))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(x):
    return "%.6g" % x


def make_caption(a, table):
    side = "롱" if a["side"] == "L" else "숏"
    coin = a["inst"].replace("-USDT-SWAP", "")
    rates = []
    for name in a["names"]:
        n, wr, ar = table[(name, a["side"])]
        tag = "%s 승률 %.0f%%(%d번)" % (name, wr, n)
        if n < SMALL_N:
            tag += " 표본적음"
        rates.append(tag)
    return "%s · %s · 진입 %s 손절 %s 익절 %s · 과거 %s" % (
        side, coin, fmt(a["entry"]), fmt(a["sl"]), fmt(a["tp"]), ", ".join(rates))


def send_text(text):
    tok = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("텔레그램 설정 없음:", text)
        return
    requests.post("https://api.telegram.org/bot%s/sendMessage" % tok,
                  data={"chat_id": chat, "text": text}, timeout=30)


def main():
    coins = m.top_coins()
    print("코인 수:", len(coins))
    with ThreadPoolExecutor(max_workers=4) as ex:
        datas = list(ex.map(m.fetch_candles, coins))
    all_res = []
    alerts = []
    for inst, d in zip(coins, datas):
        if d is None:
            continue
        all_res += m.backtest(d)
        lines = m.make_lines(d)
        at = m.atr(d)
        alerts += find_alerts(inst, d, lines, at)
    table = m.summarize(all_res)
    print("새 신호:", len(alerts))
    if not alerts:
        send_text("추세알림: 오늘 새 신호 없음")
        return
    for k, a in enumerate(alerts[:MAX_ALERTS]):
        path = "alert_%d.png" % k
        draw_alert(a, path)
        m.send_telegram(path, make_caption(a, table))


if __name__ == "__main__":
    main()
