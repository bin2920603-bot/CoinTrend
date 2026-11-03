import time
import numpy as np
import requests
import main as m
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 설정 (여기 숫자만 바꾸면 됨) =====
# 봉마다 '닿았다' 거리(touch), '이격 큼' 기준(gap), 신호 후 기다릴 봉 수(hold)를 따로 둠
CFG = {
    "4H": dict(n=2400, touch=0.004, gap=0.03, hold=12),   # 약 400일치
    "1H": dict(n=1440, touch=0.003, gap=0.02, hold=24),   # 약 60일치
    "15m": dict(n=1440, touch=0.002, gap=0.01, hold=32),  # 약 15일치
}
TOP_N = 60            # 거래대금 상위 몇 개 코인을 볼지
SLOPE_BARS = 5        # 선의 방향을 볼 봉 개수
SLOPE_ATR = 0.3       # 그 사이 선이 ATR의 0.3배 이상 움직여야 '방향 있음'
WORKERS = 3           # OKX 요청 제한 때문에 적게
MIN_N = 30            # 신호가 이보다 적으면 '표본 적음'


def fetch(inst, bar, n):
    rows, after = [], None
    while len(rows) < n:
        p = {"instId": inst, "bar": bar, "limit": "100"}
        if after:
            p["after"] = after
        data = m.get_json(m.BASE + "/api/v5/market/history-candles", p)
        time.sleep(0.25)
        if not data:
            break
        rows += data
        after = data[-1][0]
    rows = [r for r in rows if r[8] == "1"]
    rows.sort(key=lambda r: int(r[0]))
    if len(rows) < 250:
        return None
    a = np.array([[float(x) for x in r[1:7]] for r in rows])
    return {"o": a[:, 0], "h": a[:, 1], "l": a[:, 2], "c": a[:, 3], "v": a[:, 5]}


def fetch_all(inst):
    return {bar: fetch(inst, bar, cfg["n"]) for bar, cfg in CFG.items()}


def signal(d, L, at, i, cfg):
    o, h, l, c = d["o"][i], d["h"][i], d["l"][i], d["c"][i]
    now, before, a = L[i], L[i - SLOPE_BARS], at[i]
    if np.isnan(now) or np.isnan(before) or np.isnan(a):
        return None
    move, gap = now - before, c / now - 1
    if move >= SLOPE_ATR * a and 0 < gap <= cfg["gap"] and l <= now * (1 + cfg["touch"]) and c > o:
        return "L"
    if move <= -SLOPE_ATR * a and -cfg["gap"] <= gap < 0 and h >= now * (1 - cfg["touch"]) and c < o:
        return "S"
    return None


def simulate(d, side, i, a, hold):
    entry = d["c"][i]
    risk = min(m.ATR_MULT * a / entry, m.MAX_STOP)
    if risk * m.RR < m.MIN_TP:
        return None
    if side == "L":
        sl, tp = entry * (1 - risk), entry * (1 + risk * m.RR)
    else:
        sl, tp = entry * (1 + risk), entry * (1 - risk * m.RR)
    for j in range(i + 1, i + hold + 1):
        h, l = d["h"][j], d["l"][j]
        if side == "L":
            if l <= sl:
                return -1.0
            if h >= tp:
                return m.RR
        else:
            if h >= sl:
                return -1.0
            if l <= tp:
                return m.RR
    last = d["c"][i + hold]
    move = (last - entry) / entry if side == "L" else (entry - last) / entry
    return move / risk


def backtest(d, cfg):
    out = []
    ln = m.make_lines(d)
    at = m.atr(d)
    n = len(d["c"])
    for name in m.LINES:
        last = {"L": -99, "S": -99}
        for i in range(SLOPE_BARS + 1, n - cfg["hold"]):
            if np.isnan(at[i]):
                continue
            s = signal(d, ln[name], at, i, cfg)
            if not s or i - last[s] < m.COOLDOWN:
                continue
            r = simulate(d, s, i, at[i], cfg["hold"])
            if r is None:
                continue
            last[s] = i
            out.append((name, s, r))
    return out


def summarize(res):
    t = {}
    for name in m.LINES:
        for s in ("L", "S"):
            rs = [r for (n_, s_, r) in res if n_ == name and s_ == s]
            t[(name, s)] = (len(rs), 100 * sum(1 for r in rs if r > 0) / len(rs) if rs else 0.0,
                            sum(rs) / len(rs) if rs else 0.0)
    return t


def draw(tables, path):
    fig, axes = plt.subplots(len(tables), 1, figsize=(9, 4.2 * len(tables)))
    if len(tables) == 1:
        axes = [axes]
    x = np.arange(len(m.LINES))
    w = 0.38
    be = 100 / (1 + m.RR)
    for ax, (bar, t) in zip(axes, tables.items()):
        bl = ax.bar(x - w / 2, [t[(n, "L")][1] for n in m.LINES], w, label="Long", color="#2e9e5b")
        bs = ax.bar(x + w / 2, [t[(n, "S")][1] for n in m.LINES], w, label="Short", color="#d64545")
        for bars, s in ((bl, "L"), (bs, "S")):
            for b, n in zip(bars, m.LINES):
                cnt = t[(n, s)][0]
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                        "n=%d" % cnt, ha="center", fontsize=8, color="#888888" if cnt < MIN_N else "black")
        ax.axhline(be, color="black", linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(m.LINES)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Win rate %")
        ax.set_title("%s line bounce backtest (break-even %.0f%%, grey n = too few)" % (bar, be))
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    coins = m.top_coins()[:TOP_N]
    print("코인 수:", len(coins))
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        datas = list(ex.map(fetch_all, coins))
    tables, caps = {}, []
    for bar, cfg in CFG.items():
        res = []
        for per in datas:
            d = per.get(bar)
            if d is not None:
                res += backtest(d, cfg)
        t = summarize(res)
        tables[bar] = t
        print(bar, "신호 합계:", len(res))
        best = max(((k, v) for k, v in t.items() if v[0] >= MIN_N), key=lambda kv: kv[1][2], default=None)
        if best:
            (name, s), (cnt, wr, ar) = best
            caps.append("%s 최고 %s %s(%d번, %.0f%%, %.2fR)" % (bar, name, "롱" if s == "L" else "숏", cnt, wr, ar))
        else:
            caps.append("%s 신호 부족" % bar)
    draw(tables, "multi.png")
    m.send_telegram("multi.png", " / ".join(caps) + " · 점선 40% 위면 이득")


if __name__ == "__main__":
    main()
