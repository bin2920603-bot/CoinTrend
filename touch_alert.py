# touch_alert.py — 차트의 노란 추세선에 가격이 닿으면 추세알림 봇으로 알려 줌 (2종류)
#   ① 닿는 중  : 지금 가격이 선에 닿는 순간 (튕길지 뚫을지는 아직 모름)
#   ② 튕김 확인 : 봉이 선에 닿았다가 반대쪽으로 마감한 뒤 (손절은 추세선 바로 뒤에 둠)
# 기존 trend_alert.py는 건드리지 않고, 거기 있는 함수(추세선 찾기·차트 그리기·텔레그램 보내기)를 그대로 가져다 씀.
#   하락 추세선(고점끼리 이은 선, 저항)에서 → 숏진입
#   상승 추세선(저점끼리 이은 선, 지지)에서 → 롱진입
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import main as m
import trend_alert as t

# ===== 설정 (여기 숫자만 바꾸면 됨) =====
TOP_N = t.TOP_N          # 거래대금 상위 몇 개 코인을 볼지 (추세알림과 같게)
MAX_ALERTS = 20          # 한 번에 보낼 최대 알림 수
# 선에서 이 비율 안으로 들어오면 '닿았다'고 봄 (봉마다 다름: 일봉 0.5%, 4시간 0.4%, 1시간 0.3%, 15분 0.2%)
TOUCH = {b: t.CFG[b]["touch"] for b, _ in t.BARS}
# ① 닿는 중: 같은 코인·같은 봉·같은 방향은 이 시간 안에 또 알리지 않음 (알림이 쏟아지지 않게)
COOLDOWN = {"1D": 24 * 3600000, "4H": 8 * 3600000, "1H": 4 * 3600000, "15m": 2 * 3600000}
STOP_PAD = 0.001         # ② 튕김 확인: 손절은 (봉의 꼬리 끝·추세선 중 더 먼 쪽)에서 0.1% 더 바깥


def line_vals(d, kind):
    """노란 추세선 값 두 개: (마지막 마감 봉 자리, 지금 진행 중인 봉 자리). 선이 없으면 None"""
    tl = t.find_trendline(d, kind)
    if not tl:
        return None
    x0, y0, x1, y1 = tl
    if x1 == x0:
        return None
    slope = (y1 - y0) / (x1 - x0)
    return y1, y1 + slope * 1


def plan_live(df, side, price):
    """① 닿는 중: 지금 가격을 진입가로 한 손절·익절 (기준은 추세알림과 같음)"""
    a = df["atr"].iloc[-1]
    if a != a or price <= 0:       # a != a 는 '숫자 아님(NaN)' 검사
        return None
    risk = min(m.ATR_MULT * a / price, m.MAX_STOP)
    if risk * m.RR < m.MIN_TP:
        return None
    if side == "L":
        return price, price * (1 - risk), price * (1 + risk * m.RR)
    return price, price * (1 + risk), price * (1 - risk * m.RR)


def plan_bounce(side, price, extreme):
    """② 튕김 확인: 손절은 추세선 바로 뒤(꼬리 끝 기준), 익절은 손절 폭의 m.RR배 (최소 m.MIN_TP)"""
    if side == "S":
        sl = extreme * (1 + STOP_PAD)
        risk = sl / price - 1
    else:
        sl = extreme * (1 - STOP_PAD)
        risk = 1 - sl / price
    if risk <= 0.0005 or risk > m.MAX_STOP:
        return None
    reward = max(risk * m.RR, m.MIN_TP)
    tp = price * (1 - reward) if side == "S" else price * (1 + reward)
    return price, sl, tp


def bounce_of(ind, bar, kind, y_conf, price, now_ms):
    """방금 마감한 봉이 선에 닿았다가 반대쪽으로 마감했는지. 맞으면 (방향, 꼬리 끝 가격, 봉 시각)"""
    lc = t.last_close(bar, now_ms)
    ts = int(ind["ts"].iloc[-1])
    if ts + t.PERIOD[bar] != lc or now_ms - lc > t.WINDOW[bar]:
        return None                     # 방금 마감한 봉이 아니면 안 봄
    o, h, l, c = (float(ind[k].iloc[-1]) for k in ("o", "h", "l", "c"))
    tol = TOUCH[bar]
    if kind == "down":
        if h >= y_conf * (1 - tol) and c < y_conf and c < o and price < y_conf:
            return "S", max(h, y_conf), ts
    else:
        if l <= y_conf * (1 + tol) and c > y_conf and c > o and price > y_conf:
            return "L", min(l, y_conf), ts
    return None


def fetch(args):
    inst, bar = args
    try:
        df = t.get_candles(inst, bar)
        time.sleep(0.1)
        return inst, bar, (t.add_ind(df) if df is not None else None)
    except Exception as e:
        print(inst, bar, "error:", e)
        return inst, bar, None


def main():
    state = json.load(open(t.STATE)) if os.path.exists(t.STATE) else {}
    now_ms = int(time.time() * 1000)
    tk = requests.get(t.BASE + "/api/v5/market/tickers", params={"instType": "SWAP"},
                      timeout=15).json()["data"]
    tk = [x for x in tk if x["instId"].endswith("-USDT-SWAP")]
    last = {x["instId"]: t.num(x.get("last")) for x in tk}
    vol24 = {x["instId"]: t.num(x.get("volCcy24h")) * t.num(x.get("last")) for x in tk}
    top = sorted(vol24, key=vol24.get, reverse=True)[:TOP_N]

    jobs = [(inst, bar) for inst in top for bar, _ in t.BARS]
    with ThreadPoolExecutor(max_workers=t.WORKERS) as ex:
        res = list(ex.map(fetch, jobs))
    data = {(inst, bar): ind for inst, bar, ind in res if ind is not None}

    # 알림 후보: (코인, 방향, 종류) -> {봉: 정보}   종류: "touch"(닿는 중) / "bounce"(튕김 확인)
    cands = {}
    for (inst, bar), ind in data.items():
        price = last.get(inst, 0)
        if price <= 0:
            continue
        d = ind.iloc[-t.CARD_BARS:].reset_index(drop=True)     # 차트에 그려지는 구간과 똑같이
        for kind, side in (("down", "S"), ("up", "L")):
            lv = line_vals(d, kind)
            if not lv or lv[0] <= 0 or lv[1] <= 0:
                continue
            y_conf, y_now = lv
            # ② 튕김 확인
            bn = bounce_of(ind, bar, kind, y_conf, price, now_ms)
            if bn:
                bside, extreme, ts = bn
                key = f"{inst}|{bar}|BN{bside}"
                pl = plan_bounce(bside, price, extreme)
                if state.get(key, 0) < ts and pl is not None:
                    cands.setdefault((inst, bside, "bounce"), {})[bar] = dict(
                        kind=kind, line=y_conf, dist=price / y_conf - 1, key=key, val=ts, pl=pl, ind=ind)
            # ① 닿는 중
            dist = price / y_now - 1
            if abs(dist) <= TOUCH[bar]:
                key = f"{inst}|{bar}|TL{side}"
                if now_ms - state.get(key, 0) >= COOLDOWN[bar]:
                    pl = plan_live(ind, side, price)
                    if pl is not None:
                        cands.setdefault((inst, side, "touch"), {})[bar] = dict(
                            kind=kind, line=y_now, dist=dist, key=key, val=now_ms, pl=pl, ind=ind)
    print("추세선 알림 후보:", len(cands))
    if not cands:
        return

    btc = None
    try:
        btc = t.trend_of(t.add_ind(t.get_candles("BTC-USDT-SWAP", "1D")))
    except Exception as e:
        print("btc error:", e)

    order = {b: i for i, (b, _) in enumerate(t.BARS)}
    names = dict(t.BARS)
    sent = ok_count = 0
    # 튕김 확인을 먼저 보냄
    for (inst, side, mode), by_bar in sorted(cands.items(), key=lambda kv: kv[0][2] != "bounce"):
        bars = sorted(by_bar, key=lambda b: order[b])
        for b in bars:
            state[by_bar[b]["key"]] = by_bar[b]["val"]
        if sent >= MAX_ALERTS:
            continue
        bar = bars[0]                    # 겹치면 가장 큰 봉 기준
        c = by_bar[bar]
        entry, sl, tp = c["pl"]
        coin = inst.replace("-USDT-SWAP", "")
        bn_ = names[bar]
        side_txt = "롱" if side == "L" else "숏"
        line_txt = "하락" if c["kind"] == "down" else "상승"
        what = "튕김 확인" if mode == "bounce" else "닿는 중 (튕김 확인 전)"
        try:
            dfs = {b2: data.get((inst, b2)) for b2, _ in t.BARS}
            t1d = t.trend_of(dfs.get("1D"))
            t4h = t.trend_of(dfs.get("4H"))
            title = f"{coin} {side_txt}진입  [{bn_}] {line_txt} 추세선 {what}"
            sig = {b: ["추세선"] for b in bars}
            t.draw_card(dfs, sig, c["pl"], side, bar, title, "chart_touch.png")
            extra = ""
            if len(bars) > 1:
                extra = "겹침: " + " / ".join(f"{names[b]} {'하락' if by_bar[b]['kind'] == 'down' else '상승'}선" for b in bars) + "\n"
            stop_note = " (추세선 바로 뒤)" if mode == "bounce" else ""
            cap = (f"[{bn_}] {side_txt}진입 · {coin} · {line_txt} 추세선 {what}\n"
                   f"추세선 가격: {t.fmt(c['line'])} (현재가 {t.fmt(entry)}, 선에서 {c['dist'] * 100:+.2f}%)\n"
                   f"진입가: {t.fmt(entry)}\n"
                   f"손절가: {t.fmt(sl)} ({(sl / entry - 1) * 100:+.2f}%){stop_note}\n"
                   f"익절가: {t.fmt(tp)} ({(tp / entry - 1) * 100:+.2f}%)\n"
                   f"큰 흐름: 일봉{t.arrow(t1d)} · 4시간{t.arrow(t4h)}\n"
                   f"시장(비트코인 일봉): {t.arrow(btc)}\n"
                   f"{extra}"
                   f"거래대금: 평균의 {t.surge_ratio(c['ind']):.1f}배 ({bn_})\n"
                   f"24시간 거래대금: ${t.um(vol24.get(inst, 0))}")
            if t.send_photo("chart_touch.png", cap):
                ok_count += 1
            sent += 1
            time.sleep(t.SEND_GAP)
        except Exception as e:
            print(inst, "touch alert error:", e)
    print(f"추세선 알림 보내기 시도 {sent}개, 성공 {ok_count}개")
    json.dump(state, open(t.STATE, "w"))


if __name__ == "__main__":
    main()
