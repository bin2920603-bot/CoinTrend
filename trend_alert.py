import os
import json
import time
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import main as m   # 일봉 성적표 계산(backtest)과 손절·익절 기준을 가져다 씀

TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT = os.environ.get("TELEGRAM_CHAT_ID")
BASE = "https://www.okx.com"
STATE = "state.json"

# ===== 설정 (여기 숫자만 바꾸면 됨) =====
TOP_N = 60            # 거래대금 상위 몇 개 코인을 볼지
SHOW = 120            # 차트에 보여줄 봉 개수
SLOPE_BARS = 5        # 선의 방향을 볼 봉 개수
SLOPE_ATR = 0.3       # (1시간 이하·4시간봉) 선이 ATR의 0.3배 이상 움직여야 '방향 있음'
COOLDOWN_BARS = 3     # 같은 코인·봉·방향은 3봉 안에 또 알리지 않음
MAX_ALERTS = 60       # 한 번에 보낼 최대 알림 수
SEND_GAP = 3.2        # 알림 사진을 보낼 때마다 쉬는 시간(초). 텔레그램은 방 하나에 1분에 20개쯤이 한도
WORKERS = 4
BREAK_EVEN = 40       # 손익분기 승률(%)
CARD_BARS = 120       # 카드(4칸 사진) 한 칸에 보여줄 봉 개수
TREND_EXT = 20        # 추세선을 오른쪽으로 몇 봉만큼 더 늘려 그릴지
TREND_K = 2           # 추세선 고점/저점 판단: 좌우 몇 봉보다 높아야(낮아야) 고점(저점)으로 볼지
TREND_TOL = 0.03      # 추세선 여유: 차트 높이의 3%까지는 꼬리가 선을 넘어도 봐줌
TREND_BAD = 1         # 선을 넘은 봉이 이 개수 이하면 인정
TREND_MIN_GAP = 8     # 선을 긋는 두 점 사이가 최소 몇 봉 이상 떨어져야 하는지
TREND_COLOR = "#FFC107"   # 추세선 색 (노란색)
BARS = [("1D", "일봉"), ("4H", "4시간봉"), ("1H", "1시간봉"), ("15m", "15분봉")]
PERIOD = {"1D": 86400000, "4H": 14400000, "1H": 3600000, "15m": 900000}
# 봉이 막 끝난 뒤 이 시간 안에 돌았을 때만 그 봉을 봄 (GitHub 실행이 늦어도 놓치지 않게 넉넉히)
WINDOW = {"1D": 4 * 3600000, "4H": 3 * 3600000, "1H": 75 * 60000, "15m": 35 * 60000}
# 봉마다: 선에 닿은 거리(touch), 이격 큼 기준(gap), 신호 후 기다리는 봉 수(hold)
CFG = {
    "1D": dict(touch=0.005, gap=0.04),
    "4H": dict(touch=0.004, gap=0.03),
    "1H": dict(touch=0.003, gap=0.02),
    "15m": dict(touch=0.002, gap=0.01),
}
# 성적표(4H/1H/15분) 승률 표시용: (봉, 선, 방향) -> (신호수, 승률%)  ※ 이제 알림을 거르지는 않음
SCORE = {
    ("4H", "EMA7", "S"): (7183, 44), ("4H", "EMA20", "S"): (1910, 44),
    ("4H", "EMA50", "S"): (106, 45), ("4H", "VW100", "S"): (328, 44),
    ("1H", "EMA7", "S"): (5921, 44), ("1H", "EMA50", "S"): (55, 47),
    ("1H", "VW100", "S"): (205, 42),
    ("15m", "EMA50", "S"): (45, 44), ("15m", "VW100", "L"): (127, 46),
    ("15m", "VW100", "S"): (82, 44),
}
DAILY_SKIP = set()   # 전부 열기
SMALL_N = 30

names = {f.name for f in font_manager.fontManager.ttflist}
for n in ["NanumGothic", "Malgun Gothic", "AppleGothic"]:
    if n in names:
        plt.rcParams["font.family"] = n
        break
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"EMA7": "#d81b60", "EMA20": "#f9a825", "EMA50": "#2e7d32",
          "EMA120": "#757575", "EMA200": "#e53935", "VW100": "#6a1b9a"}
LINE_COL = {"EMA7": "ema7", "EMA20": "ema20", "EMA50": "ema50",
            "EMA120": "ema120", "EMA200": "ema200", "VW100": "vw100"}


# ---------------- 데이터 ----------------
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def get_candles(inst, bar, hist=0):
    """마감된 봉만. hist>0이면 그만큼(봉 수) 과거까지 이어서 받음"""
    rows, after = [], None
    while True:
        if after:
            url, p = "/api/v5/market/history-candles", {"instId": inst, "bar": bar, "limit": "100", "after": after}
        else:
            url, p = "/api/v5/market/candles", {"instId": inst, "bar": bar, "limit": "300"}
        data = []
        for _ in range(3):
            try:
                r = requests.get(BASE + url, params=p, timeout=15).json()
                if r.get("code") == "0" and r["data"]:
                    data = r["data"]
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not data:
            break
        rows += data
        after = data[-1][0]
        if not hist or len(rows) >= hist:
            break
        time.sleep(0.15)
    if not rows:
        return None
    df = pd.DataFrame(rows).iloc[:, :9]
    df.columns = ["ts", "o", "h", "l", "c", "vol", "volc", "turn", "confirm"]
    for k in df.columns:
        df[k] = df[k].astype(float)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df[df["confirm"] == 1].reset_index(drop=True)
    if len(df) < 120:
        return None
    if df["turn"].sum() == 0:
        df["turn"] = df["volc"] * df["c"]
    return df


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def add_ind(df):
    for n in (7, 20, 50, 120, 200):
        df[f"ema{n}"] = ema(df["c"], n)
    vs = df["volc"].rolling(100).sum()
    df["vw100"] = (df["c"] * df["volc"]).rolling(100).sum() / vs.where(vs > 0)
    for n in (6, 12, 24):
        df[f"rsi{n}"] = rsi(df["c"], n)
    df["macd"] = ema(df["c"], 12) - ema(df["c"], 26)
    df["sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"] = df["macd"] - df["sig"]
    pc = df["c"].shift(1)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - pc).abs(), (df["l"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


def trend_of(df):
    if df is None or len(df) < 60:
        return None
    last = df.iloc[-1]
    if last["c"] > last["ema50"] and last["ema20"] > last["ema50"]:
        return "up"
    if last["c"] < last["ema50"] and last["ema20"] < last["ema50"]:
        return "down"
    return None


def arrow(t):
    return {"up": "↑", "down": "↓"}.get(t, "↔")


# ---------------- 신호 ----------------
def signal(df, col, bar):
    """마지막(마감) 봉에 롱/숏 신호가 있는지. 'L' / 'S' / None"""
    i = len(df) - 1
    o, h, l, c = df["o"].iloc[i], df["h"].iloc[i], df["l"].iloc[i], df["c"].iloc[i]
    now, before, a = df[col].iloc[i], df[col].iloc[i - SLOPE_BARS], df["atr"].iloc[i]
    if pd.isna(now) or pd.isna(before) or pd.isna(a):
        return None
    cfg = CFG[bar]
    gap = c / now - 1
    if bar == "1D":      # 일봉은 성적표와 같은 규칙(선이 5일에 0.2% 이상 기울어야 함)
        up = now / before - 1 >= m.SLOPE_MIN
        dn = now / before - 1 <= -m.SLOPE_MIN
    else:
        up = now - before >= SLOPE_ATR * a
        dn = now - before <= -SLOPE_ATR * a
    if up and 0 < gap <= cfg["gap"] and l <= now * (1 + cfg["touch"]) and c > o:
        return "L"
    if dn and -cfg["gap"] <= gap < 0 and h >= now * (1 - cfg["touch"]) and c < o:
        return "S"
    return None


def plan(df, side):
    entry, a = df["c"].iloc[-1], df["atr"].iloc[-1]
    risk = min(m.ATR_MULT * a / entry, m.MAX_STOP)
    if risk * m.RR < m.MIN_TP:
        return None
    if side == "L":
        return entry, entry * (1 - risk), entry * (1 + risk * m.RR)
    return entry, entry * (1 + risk), entry * (1 - risk * m.RR)


def surge_ratio(df):
    base = df["turn"].iloc[-21:-1].mean()
    return df["turn"].iloc[-1] / base if base > 0 else 0


def last_close(bar, now_ms):
    p = PERIOD[bar]
    off = 16 * 3600000 if bar == "1D" else 0     # OKX 일봉은 한국시간 오전 1시에 마감
    return ((now_ms - off) // p) * p + off


# ---------------- 추세선 (노란색) ----------------
def find_swings(vals, k, high=True):
    """좌우 k봉보다 높은(낮은) 봉의 위치 목록 = 고점(저점)"""
    out = []
    for i in range(k, len(vals) - k):
        window = vals[i - k:i + k + 1]
        if (high and vals[i] == max(window)) or ((not high) and vals[i] == min(window)):
            out.append(i)
    return out


def find_trendline(d, kind):
    """kind='down': 고점들을 이은 하락 추세선, kind='up': 저점들을 이은 상승 추세선.
    꼬리가 선을 살짝 넘는 건 봐주고, 넘은 봉이 TREND_BAD개 이하인 선 중 가장 길게 이어진 선을 고름.
    돌려주는 값: (시작 위치, 시작 가격, 끝 위치, 끝 가격) 또는 None"""
    n = len(d)
    highs = list(d["h"].values)
    lows = list(d["l"].values)
    tol = TREND_TOL * (max(highs) - min(lows))
    if kind == "down":
        vals = highs
        swings = find_swings(vals, TREND_K, True)
    else:
        vals = lows
        swings = find_swings(vals, TREND_K, False)
    best = None
    for b in swings[-4:]:
        for a in swings:
            if b - a < TREND_MIN_GAP:
                break
            if kind == "down" and vals[a] <= vals[b]:
                continue
            if kind == "up" and vals[a] >= vals[b]:
                continue
            slope = (vals[b] - vals[a]) / (b - a)
            bad = 0
            for x in range(a, n - 1):
                line = vals[a] + slope * (x - a)
                if kind == "down" and vals[x] > line + tol:
                    bad += 1
                if kind == "up" and vals[x] < line - tol:
                    bad += 1
            if bad > TREND_BAD:
                continue
            span = b - a
            if best is None or span > best[0]:
                best = (span, a, vals[a], n - 1, vals[a] + slope * (n - 1 - a))
    return best[1:] if best else None


def draw_trendlines(ax, d):
    """가격 차트에 노란 추세선(하락선·상승선)을 그림. 선은 오른쪽으로 TREND_EXT봉만큼 더 늘려 그림"""
    first = True
    n = len(d)
    for kind in ("down", "up"):
        t = find_trendline(d, kind)
        if t:
            x0, y0, x1, y1 = t
            slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0
            x_end = n - 1 + TREND_EXT
            y_end = y0 + slope * (x_end - x0)
            ax.plot([x0, x_end], [y0, y_end], color=TREND_COLOR, linewidth=3.0,
                    label="추세선" if first else None)
            first = False


# ---------------- 그림·글 ----------------
def fmt(v):
    if v >= 100:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.4f}"
    return f"{v:.7f}"


def um(v):
    if v >= 1e8:
        return f"{v / 1e8:,.1f}억"
    if v >= 1e4:
        return f"{v / 1e4:,.1f}만"
    return f"{v:,.0f}"


def draw(df, title, path, only, pl, side):
    d = df.iloc[-SHOW:].reset_index(drop=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True,
                           gridspec_kw={"height_ratios": [5, 1.5, 1.5, 1.5]})
    p = ax[0]
    for i, r in d.iterrows():
        col = "#2e9e44" if r["c"] >= r["o"] else "#d62728"
        p.vlines(i, r["l"], r["h"], color=col, lw=0.8)
        p.bar(i, abs(r["c"] - r["o"]), bottom=min(r["o"], r["c"]), color=col, width=0.7)
    for label, c in LINE_COL.items():
        if label in only:
            p.plot(x, d[c], color=COLORS[label], lw=2.2, label=label)
    entry, sl, tp = pl
    for v, name, color in ((tp, "익절", "#2e7d32"), (entry, "진입", "#000000"), (sl, "손절", "#c62828")):
        p.axhline(v, color=color, ls="--", lw=1.2)
        p.annotate(f" {name} {fmt(v)}", xy=(1.0, v), xycoords=("axes fraction", "data"),
                   color=color, fontsize=9, va="center", ha="left")
    p.legend(loc="upper left", fontsize=10)
    p.set_title(title, fontsize=14, color="#2e7d32" if side == "L" else "#c62828")
    p.grid(alpha=0.25)
    for n, color in ((6, "#8e44ad"), (12, "#1e88e5"), (24, "#fb8c00")):
        ax[1].plot(x, d[f"rsi{n}"], color=color, lw=1, label=f"RSI{n}")
    ax[1].axhline(70, color="gray", ls="--", lw=0.6)
    ax[1].axhline(30, color="gray", ls="--", lw=0.6)
    ax[1].legend(loc="upper left", fontsize=7, ncol=3)
    ax[1].grid(alpha=0.25)
    ax[2].bar(x, d["hist"], color=np.where(d["hist"] >= 0, "#2e9e44", "#d62728"), width=0.7)
    ax[2].plot(x, d["macd"], color="#1e88e5", lw=1, label="MACD")
    ax[2].plot(x, d["sig"], color="#fb8c00", lw=1, label="Signal")
    ax[2].legend(loc="upper left", fontsize=7, ncol=2)
    ax[2].grid(alpha=0.25)
    ax[3].bar(x, d["turn"], color=np.where(d["c"] >= d["o"], "#2e9e44", "#d62728"), width=0.7)
    ax[3].set_ylabel("거래대금($)", fontsize=8)
    ax[3].grid(alpha=0.25)
    ticks = list(range(0, len(d), 20))
    t = pd.to_datetime(d["ts"], unit="ms") + pd.Timedelta(hours=9)
    ax[3].set_xticks(ticks)
    ax[3].set_xticklabels([t[i].strftime("%m-%d %H:%M") for i in ticks], fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_day_marks(axp, others, d, bn):
    """오늘 시가·고가·저가·현재가(연한 회색 점선+작은 글자) + 1시간봉·15분봉에는 매일 01시 마감 세로선"""
    grey, txt = "#90a4ae", "#607d8b"
    H = 3600000
    last = int(d["ts"].iloc[-1])
    start = (last + 8 * H) // (24 * H) * (24 * H) - 8 * H
    today = d[d["ts"] >= start]
    if len(today):
        vals = {"시가": float(today["o"].iloc[0]), "고가": float(today["h"].max()),
                "저가": float(today["l"].min()), "현재가": float(d["c"].iloc[-1])}
        items = sorted(vals.items(), key=lambda t: -t[1])
        y0, y1 = axp.get_ylim()
        gap = (y1 - y0) * 0.07
        pos = []
        for _, v in items:
            axp.axhline(v, color=grey, lw=0.9, ls="--", alpha=0.8)
            pos.append(v if not pos else min(v, pos[-1] - gap))
        n = len(d)
        for (name, v), y in zip(items, pos):
            axp.annotate(f"{name} {fmt(v)}", xy=(n + 1, v), xytext=(n + 2, y), xycoords="data", textcoords="data",
                         fontsize=6.5, color=txt, va="center", ha="left", fontweight="bold", annotation_clip=False,
                         arrowprops=dict(arrowstyle="-", color=grey, lw=0.6, shrinkA=0, shrinkB=0))
    if bn in ("1시간봉", "15분봉"):
        kst = pd.to_datetime(d["ts"], unit="ms") + pd.Timedelta(hours=9)
        for i in range(1, len(d)):
            if kst[i].hour == 1 and kst[i].minute == 0:
                for a in [axp] + others:
                    a.axvline(i - 0.5, color=grey, lw=1.0, ls=(0, (2, 2)), alpha=0.9)
                if i >= 14:
                    axp.text(i - 0.5, 1.0, "01시 마감", transform=axp.get_xaxis_transform(), fontsize=7,
                             color=txt, ha="center", va="bottom", fontweight="bold")


def draw_card(dfs, sig, pl, side, primary, title, path):
    """일봉·4시간·1시간·15분을 한 장(2x2)에 그림. 신호난 선은 굵게, 진입·손절·익절은 신호난 봉 칸에만."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(14, 15), dpi=80)
    outer = GridSpec(2, 2, figure=fig, wspace=0.14, hspace=0.12, top=0.94, bottom=0.03, left=0.06, right=0.97)
    fig.suptitle(title, fontsize=16, x=0.06, ha="left", color="#2e7d32" if side == "L" else "#c62828")
    for k, (bar, bn) in enumerate(BARS):
        gs = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[k // 2, k % 2],
                                     height_ratios=[4, 1.2, 1.2, 1.2], hspace=0.08)
        axp = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1], sharex=axp)
        axm = fig.add_subplot(gs[2], sharex=axp)
        axv = fig.add_subplot(gs[3], sharex=axp)
        df = dfs.get(bar)
        if df is None:
            axp.set_title(f"{bn} (데이터 없음)", fontsize=11, loc="left")
            continue
        d = df.iloc[-CARD_BARS:].reset_index(drop=True)
        x = np.arange(len(d))
        up = (d["c"] >= d["o"]).values
        col = np.where(up, "#2e9e44", "#d62728")
        axp.vlines(x, d["l"], d["h"], colors=col, linewidth=0.8)
        axp.bar(x, d["c"] - d["o"], bottom=d["o"], color=col, width=0.7)
        names = sig.get(bar, [])
        for label, c in LINE_COL.items():
            if label in names:
                axp.plot(x, d[c], color=COLORS[label], lw=2.2, label=label)
            else:
                axp.plot(x, d[c], color=COLORS[label], lw=0.7, alpha=0.5)
        # 노란 추세선 (선이 멀리 뻗어도 가격 범위는 그대로 유지, 오른쪽에 빈 공간을 만들어 선을 연장)
        y_lo, y_hi = axp.get_ylim()
        draw_trendlines(axp, d)
        axp.set_ylim(y_lo, y_hi)
        axp.set_xlim(-1, len(d) + TREND_EXT)
        draw_day_marks(axp, [axr, axm, axv], d, bn)
        mark = "  ★ " + "+".join(names) if names else ""
        axp.set_title(f"{bn} ({bar}){mark}", fontsize=11, loc="left", fontweight="bold" if names else "normal")
        if names:
            axp.legend(loc="upper left", fontsize=8)
        if bar == primary and pl:
            entry, sl, tp = pl
            for v, name, color in ((tp, "TP", "#2e7d32"), (entry, "IN", "#000000"), (sl, "SL", "#c62828")):
                axp.axhline(v, color=color, ls="--", lw=1)
                axp.annotate(f" {name} {fmt(v)}", xy=(1.0, v), xycoords=("axes fraction", "data"),
                             color=color, fontsize=7, va="center", ha="left")
        for n, color in ((6, "#8e44ad"), (12, "#1e88e5"), (24, "#fb8c00")):
            axr.plot(x, d[f"rsi{n}"], color=color, lw=0.8)
        axr.axhline(70, color="gray", ls="--", lw=0.5)
        axr.axhline(30, color="gray", ls="--", lw=0.5)
        axr.set_ylim(0, 100)
        axr.set_yticks([30, 70])
        axm.bar(x, d["hist"], color=np.where(d["hist"] >= 0, "#2e9e44", "#d62728"), width=0.7)
        axm.plot(x, d["macd"], color="#1e88e5", lw=0.8)
        axm.plot(x, d["sig"], color="#fb8c00", lw=0.8)
        axv.bar(x, d["turn"], color=col, width=0.7)
        for ax, lab in ((axr, "RSI"), (axm, "MACD"), (axv, "VOL")):
            ax.set_ylabel(lab, fontsize=7)
            ax.tick_params(labelsize=7)
        axp.tick_params(labelsize=8)
        axv.set_yticks([])
        for ax in (axp, axr, axm):
            plt.setp(ax.get_xticklabels(), visible=False)
        axv.set_xticks([])
        for ax in (axp, axr, axm, axv):
            ax.grid(alpha=0.25)
    fig.savefig(path)
    plt.close(fig)


# ---------------- 텔레그램 보내기 ----------------
def send_photo(path, caption):
    """사진을 보내고, 성공하면 True. 텔레그램이 거절하면 이유를 기록에 남기고, '너무 빨라요(429)'면 기다렸다가 다시 보냄"""
    if not TOKEN or not CHAT:
        print("텔레그램 설정 없음:", caption)
        return False
    for _ in range(3):
        try:
            with open(path, "rb") as fh:
                r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                                  data={"chat_id": CHAT, "caption": caption[:1000]},
                                  files={"photo": fh}, timeout=30)
        except Exception as e:
            print("[텔레그램 전송 오류]", e)
            return False
        if r.ok:
            return True
        print("[텔레그램 실패]", r.status_code, r.text[:200])
        if r.status_code == 429:
            wait = 5
            try:
                wait = r.json()["parameters"]["retry_after"]
            except Exception:
                pass
            time.sleep(min(wait, 60) + 1)
            continue
        return False
    return False


def send_text(text):
    if not TOKEN or not CHAT:
        print("텔레그램 설정 없음:", text)
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          data={"chat_id": CHAT, "text": text}, timeout=30)
        if not r.ok:
            print("[텔레그램 실패]", r.status_code, r.text[:200])
        return r.ok
    except Exception as e:
        print("[텔레그램 전송 오류]", e)
        return False


def rate_text(bar, name, side, daily):
    try:
        if bar == "1D":
            n, wr, _ = daily[(name, side)]
        else:
            n, wr = SCORE[(bar, name, side)]
    except Exception:
        return f"{name} 성적표 없음"
    tag = f"{name} {wr:.0f}%({n}번)"
    return tag + (" 표본적음" if n < SMALL_N else "")


# ---------------- 실행 ----------------
def scan_coin(args):
    inst, bar, now_ms = args
    try:
        df = get_candles(inst, bar, hist=400 if bar == "1D" else 0)
        time.sleep(0.1)
        if df is None or df["ts"].iloc[-1] + PERIOD[bar] != last_close(bar, now_ms):
            return inst, bar, None, None     # 아직 마감 확정 전이거나 데이터 부족
        return inst, bar, add_ind(df), df
    except Exception as e:
        print(inst, bar, "error:", e)
        return inst, bar, None, None


def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    now_ms = int(time.time() * 1000)
    state = {k: v for k, v in state.items() if now_ms - v < 3 * 86400000}
    fresh = [b for b, _ in BARS if now_ms - last_close(b, now_ms) <= WINDOW[b]]
    print("지금 볼 봉:", fresh)
    if not fresh:
        json.dump(state, open(STATE, "w"))
        return
    tk = requests.get(BASE + "/api/v5/market/tickers", params={"instType": "SWAP"},
                      timeout=15).json()["data"]
    tk = [t for t in tk if t["instId"].endswith("-USDT-SWAP")]
    vol24 = {t["instId"]: num(t.get("volCcy24h")) * num(t.get("last")) for t in tk}
    top = sorted(vol24, key=vol24.get, reverse=True)[:TOP_N]
    jobs = [(inst, bar, now_ms) for bar in fresh for inst in top]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(scan_coin, jobs))
    # 일봉이면 성적표를 새로 계산
    daily = None
    if "1D" in fresh:
        all_res = []
        for inst, bar, ind, raw in res:
            if bar == "1D" and raw is not None:
                d = {k: raw[k].values for k in ("o", "h", "l", "c")}
                d["v"] = raw["volc"].values
                all_res += m.backtest(d)
        daily = m.summarize(all_res)
    cands = {}      # (코인, 방향) -> 신호 목록
    for inst, bar, ind, raw in res:
        if ind is None:
            continue
        ts = int(ind["ts"].iloc[-1])
        pl_cache = {}
        for name, col in LINE_COL.items():
            s = signal(ind, col, bar)
            if not s:
                continue
            if bar == "1D":
                if (name, s) in DAILY_SKIP:
                    continue
            key = f"{inst}|{bar}|{s}"
            if ts - state.get(key, 0) < COOLDOWN_BARS * PERIOD[bar]:
                continue
            pl_cache.setdefault(s, plan(ind, s))
            if pl_cache[s] is None:
                continue
            cands.setdefault((inst, s), {}).setdefault(bar, {"ind": ind, "ts": ts, "names": [], "key": key})
            cands[(inst, s)][bar]["names"].append(name)
    print("새 신호 코인:", len(cands))
    if not cands:
        json.dump(state, open(STATE, "w"))
        return
    # 시장·큰 흐름
    btc = None
    try:
        btc = trend_of(add_ind(get_candles("BTC-USDT-SWAP", "1D")))
    except Exception as e:
        print("btc error:", e)
    order = {b: i for i, (b, _) in enumerate(BARS)}
    sent = 0
    ok_count = 0
    for (inst, s), by_bar in cands.items():
        bars = sorted(by_bar, key=lambda b: order[b])
        for b in bars:
            state[by_bar[b]["key"]] = by_bar[b]["ts"]
        if sent >= MAX_ALERTS:
            continue
        bar = bars[0]
        c = by_bar[bar]
        ind, names_ = c["ind"], c["names"]
        pl = plan(ind, s)
        coin = inst.replace("-USDT-SWAP", "")
        bn = dict(BARS)[bar]
        side = "롱" if s == "L" else "숏"
        try:
            dfs = {}
            for b2, _ in BARS:
                if b2 in by_bar:
                    dfs[b2] = by_bar[b2]["ind"]
                else:
                    g = get_candles(inst, b2)
                    dfs[b2] = add_ind(g) if g is not None else None
                    time.sleep(0.1)
            t1d = trend_of(dfs.get("1D"))
            t4h = trend_of(dfs.get("4H"))
            title = f"{coin} {side}진입  [{bn}] {'+'.join(names_)} 추세"
            sig = {b2: by_bar[b2]["names"] for b2 in bars}
            draw_card(dfs, sig, pl, s, bar, title, "chart.png")
            entry, sl, tp = pl
            extra = ""
            if len(bars) > 1:
                extra = "겹침: " + " / ".join(f"{dict(BARS)[b]} {'+'.join(by_bar[b]['names'])}" for b in bars) + "\n"
            rates = ", ".join(rate_text(bar, n, s, daily) for n in names_)
            cap = (f"[{bn}] {side}진입 · {coin} · {'+'.join(names_)} 추세\n"
                   f"진입가: {fmt(entry)}\n"
                   f"손절가: {fmt(sl)} ({(sl / entry - 1) * 100:+.2f}%)\n"
                   f"익절가: {fmt(tp)} ({(tp / entry - 1) * 100:+.2f}%)\n"
                   f"과거 승률: {rates} (손익분기 {BREAK_EVEN}%)\n"
                   f"큰 흐름: 일봉{arrow(t1d)} · 4시간{arrow(t4h)}\n"
                   f"시장(비트코인 일봉): {arrow(btc)}\n"
                   f"{extra}"
                   f"거래대금: 평균의 {surge_ratio(ind):.1f}배 ({bn})\n"
                   f"봉 거래대금: ${um(ind['turn'].iloc[-1])}\n"
                   f"24시간 거래대금: ${um(vol24.get(inst, 0))}")
            if send_photo("chart.png", cap):
                ok_count += 1
            sent += 1
            time.sleep(SEND_GAP)      # 텔레그램 '너무 빨라요' 방지
        except Exception as e:
            print(inst, "alert error:", e)
    print(f"알림 보내기 시도 {sent}개, 성공 {ok_count}개")
    if len(cands) > MAX_ALERTS:
        send_text(f"추세알림: 신호가 많아 {MAX_ALERTS}개만 보냈어요 (전체 {len(cands)}개)")
    json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
