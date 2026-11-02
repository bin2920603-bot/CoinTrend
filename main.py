import os
import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ===== 설정 (여기 숫자만 바꾸면 됨) =====
TOP_N = 60            # 거래대금 상위 몇 개 코인을 볼지
DAYS = 400            # 과거 며칠치 일봉을 볼지
SLOPE_DAYS = 5        # 선의 방향(각도)을 볼 기간(일)
SLOPE_MIN = 0.002     # 그 기간에 선이 이 정도(0.2%) 이상 기울어야 방향 있음
TOUCH = 0.005         # 선에서 0.5% 이내로 닿으면 '닿았다'
MAX_GAP = 0.04        # 종가가 선에서 4% 넘게 벌어지면 '이격 큼' -> 제외
RR = 1.5              # 익절폭 = 손절폭 x 1.5 (손익분기 승률 40%)
ATR_MULT = 1.5        # 손절 = ATR x 1.5
MAX_STOP = 0.05       # 손절은 최대 5%
MIN_TP = 0.01         # 익절 폭이 1% 미만이면 제외
HOLD = 10             # 신호 후 최대 며칠 기다릴지
COOLDOWN = 3          # 같은 코인·선·방향은 3일 안에 또 안 셈
LINES = ["EMA7", "EMA20", "EMA50", "EMA120", "EMA200", "VW100"]
BASE = "https://www.okx.com"


def get_json(url, params=None):
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            d = r.json()
            if d.get("code") == "0":
                return d["data"]
        except Exception:
            pass
        time.sleep(0.5)
    return []


def top_coins():
    data = get_json(BASE + "/api/v5/market/tickers", {"instType": "SWAP"})
    rows = []
    for t in data:
        if not t["instId"].endswith("-USDT-SWAP"):
            continue
        try:
            rows.append((float(t["volCcy24h"]) * float(t["last"]), t["instId"]))
        except Exception:
            pass
    rows.sort(reverse=True)
    return [r[1] for r in rows[:TOP_N]]


def fetch_candles(inst):
    rows = []
    after = None
    while len(rows) < DAYS:
        p = {"instId": inst, "bar": "1D", "limit": "100"}
        if after:
            p["after"] = after
        data = get_json(BASE + "/api/v5/market/history-candles", p)
        time.sleep(0.15)
        if not data:
            break
        rows += data
        after = data[-1][0]
    rows = [r for r in rows if r[8] == "1"]
    rows.sort(key=lambda r: int(r[0]))
    if len(rows) < 60:
        return None
    a = np.array([[float(x) for x in r[1:7]] for r in rows])
    return {"o": a[:, 0], "h": a[:, 1], "l": a[:, 2], "c": a[:, 3], "v": a[:, 5]}


def ema(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    k = 2 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def vwma(c, v, n):
    out = np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        vs = v[i - n + 1:i + 1].sum()
        if vs > 0:
            out[i] = (c[i - n + 1:i + 1] * v[i - n + 1:i + 1]).sum() / vs
    return out


def atr(d, n=14):
    h, l, c = d["h"], d["l"], d["c"]
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    out = np.full(len(c), np.nan)
    for i in range(n, len(c)):
        out[i] = tr[i - n:i].mean()
    return out


def make_lines(d):
    return {
        "EMA7": ema(d["c"], 7),
        "EMA20": ema(d["c"], 20),
        "EMA50": ema(d["c"], 50),
        "EMA120": ema(d["c"], 120),
        "EMA200": ema(d["c"], 200),
        "VW100": vwma(d["c"], d["v"], 100),
    }


def check_signal(d, ln, i):
    """i번째 일봉이 끝난 시점에 롱/숏 신호가 있는지. 반환: 'L' / 'S' / None"""
    o, h, l, c = d["o"][i], d["h"][i], d["l"][i], d["c"][i]
    L = ln[i]
    Lp = ln[i - SLOPE_DAYS]
    if np.isnan(L) or np.isnan(Lp):
        return None
    slope = L / Lp - 1
    gap = c / L - 1
    # 롱: 1) 선이 위로 기울어 있다 2) 종가가 선 위, 이격 작다 3) 선에 닿았다가 양봉으로 튕김
    if slope >= SLOPE_MIN and 0 < gap <= MAX_GAP and l <= L * (1 + TOUCH) and c > o:
        return "L"
    # 숏: 반대
    if slope <= -SLOPE_MIN and -MAX_GAP <= gap < 0 and h >= L * (1 - TOUCH) and c < o:
        return "S"
    return None


def simulate(d, side, i, a):
    """진입(종가) 후 HOLD일 안에 익절/손절 중 뭐가 먼저인지. 결과를 R(손절폭 배수)로 반환."""
    entry = d["c"][i]
    risk = min(ATR_MULT * a / entry, MAX_STOP)
    if risk * RR < MIN_TP:
        return None
    if side == "L":
        sl, tp = entry * (1 - risk), entry * (1 + risk * RR)
    else:
        sl, tp = entry * (1 + risk), entry * (1 - risk * RR)
    for j in range(i + 1, i + HOLD + 1):
        h, l = d["h"][j], d["l"][j]
        if side == "L":
            if l <= sl:
                return -1.0
            if h >= tp:
                return RR
        else:
            if h >= sl:
                return -1.0
            if l <= tp:
                return RR
    last = d["c"][i + HOLD]
    move = (last - entry) / entry if side == "L" else (entry - last) / entry
    return move / risk


def backtest(d):
    results = []  # (선, 방향, R)
    ln = make_lines(d)
    at = atr(d)
    n = len(d["c"])
    for name in LINES:
        last = {"L": -99, "S": -99}
        for i in range(SLOPE_DAYS + 1, n - HOLD):
            if np.isnan(at[i]):
                continue
            s = check_signal(d, ln[name], i)
            if not s or i - last[s] < COOLDOWN:
                continue
            r = simulate(d, s, i, at[i])
            if r is None:
                continue
            last[s] = i
            results.append((name, s, r))
    return results


def summarize(all_res):
    table = {}
    for name in LINES:
        for s in ("L", "S"):
            rs = [r for (n_, s_, r) in all_res if n_ == name and s_ == s]
            if rs:
                table[(name, s)] = (len(rs), 100 * sum(1 for r in rs if r > 0) / len(rs), sum(rs) / len(rs))
            else:
                table[(name, s)] = (0, 0.0, 0.0)
    return table


def draw(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(LINES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    wl = [table[(n, "L")][1] for n in LINES]
    ws = [table[(n, "S")][1] for n in LINES]
    b1 = ax.bar(x - w / 2, wl, w, label="Long", color="#2e9e5b")
    b2 = ax.bar(x + w / 2, ws, w, label="Short", color="#d64545")
    for bars, s in ((b1, "L"), (b2, "S")):
        for b, n in zip(bars, LINES):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, "n=%d" % table[(n, s)][0], ha="center", fontsize=8)
    be = 100 / (1 + RR)
    ax.axhline(be, color="black", linestyle="--")
    ax.text(len(LINES) - 0.5, be + 1, "break-even %.0f%%" % be, ha="right", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(LINES)
    ax.set_ylabel("Win rate %")
    ax.set_ylim(0, 100)
    ax.set_title("Daily line bounce backtest (OKX top %d, %d days)" % (TOP_N, DAYS))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def send_telegram(path, caption):
    tok = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("텔레그램 설정 없음 - 전송 생략")
        return
    with open(path, "rb") as f:
        r = requests.post("https://api.telegram.org/bot%s/sendPhoto" % tok,
                          data={"chat_id": chat, "caption": caption}, files={"photo": f}, timeout=30)
    print("텔레그램 전송:", r.status_code)


def main():
    coins = top_coins()
    print("코인 수:", len(coins))
    with ThreadPoolExecutor(max_workers=4) as ex:
        datas = list(ex.map(fetch_candles, coins))
    all_res = []
    for inst, d in zip(coins, datas):
        if d is not None:
            all_res += backtest(d)
    table = summarize(all_res)
    print("선 / 방향 / 신호수 / 승률% / 평균R")
    best = None
    for (name, s), (n, wr, ar) in table.items():
        print(name, s, n, round(wr, 1), round(ar, 2))
        if n >= 20 and (best is None or ar > best[3]):
            best = (name, s, n, ar, wr)
    draw(table, "result.png")
    if best:
        cap = "손익이 가장 좋은 건 %s %s (신호 %d개, 승률 %.0f%%, 평균 %.2fR). 점선(40%%) 위면 이득" % (
            best[0], "롱" if best[1] == "L" else "숏", best[2], best[4], best[3])
    else:
        cap = "신호가 너무 적어 판단 불가 (점선 40% 위면 이득)"
    send_telegram("result.png", cap)


if __name__ == "__main__":
    main()
