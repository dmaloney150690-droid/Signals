# app.py — Signals Pro (single-file, lru_cache only; no Streamlit caching)
import os, re, requests, feedparser, datetime as dt
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote_plus
from functools import lru_cache

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------- App setup -----------------------------
st.set_page_config(page_title="Signals Pro — 1–3 Day Holds", layout="wide")
st.title("Signals Pro — 1–3 Day Holds")
st.caption("Build: lru-cache version — " + dt.datetime.utcnow().isoformat() + "Z")
st.caption("Scanner + Liquidity + FX-aware sizing + News + Regime + Portfolio + Journal + Planner")
st.divider()

# ----------------------------- Defaults -----------------------------
defaults = {
    "equity_gbp": 400.0, "risk_per_trade_pct": 1.0, "max_leverage": 3, "max_open_positions": 3,
    "daily_loss_stop_pct": 3.0, "universe": "All (US+UK+EU)", "timeframe_days": 250, "hold_days_max": 3,
    "allow_shorts": True, "lookback_hours": 72,
    "earnings_window_days": 3, "earnings_action": "Warn",
    "macro_block_days": 1, "macro_action": "Warn",
    "assumed_spread_bps": 5, "min_stop_to_spread": 10,
    "corr_threshold": 0.8, "portfolio_max_positions": 3, "partial_exits": True
}
cfg = defaults.copy()

# ----------------------------- Session -----------------------------
for k, v in [("last_signals", pd.DataFrame()), ("watchlist", []), ("journal", []), ("macro_dates", [])]:
    if k not in st.session_state: st.session_state[k] = v

UA = {"User-Agent": "SignalsPro/1.0"}
analyzer = SentimentIntensityAnalyzer()

def _get_secret(name: str) -> str:
    try:
        if name in st.secrets: return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, "")

# ----------------------------- Universes -----------------------------
SP100 = [
"AAPL","ABBV","ABT","ACN","ADBE","AMD","AMGN","AMT","AMZN","AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","CAT",
"CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX","DE","DHR","DIS","DUK","EMR","EXC","F","FDX","GE","GILD","GM",
"GOOG","GOOGL","GS","HD","HON","IBM","INTC","JNJ","JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ","MDT","META",
"MMM","MO","MS","MSFT","NEE","NFLX","NKE","NOW","NVDA","ORCL","PEP","PFE","PM","PYPL","QCOM","RTX","SBUX","SCHW","SO",
"SPGI","T","TGT","TMO","TMUS","TSLA","TXN","UNH","UNP","UPS","USB","V","VZ","WBA","WFC","WMT","XOM"
]
FTSE100 = ["AZN.L","SHEL.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","DGE.L","BATS.L","BDEV.L","BARC.L","VOD.L","TSCO.L","LLOY.L","RS1.L","IAG.L","WTB.L","CRH.L","REL.L","PRU.L","NG.L","AAL.L","SBRY.L","AUTO.L","SGE.L","JD.L","NXT.L","HLMA.L","SVT.L","SMT.L","BRBY.L","FERG.L","HL.L","KGF.L","IMB.L","III.L","RR.L","RTO.L","EXPN.L","BKG.L","BA.L","CNA.L","AV.L","ITV.L","PHNX.L","PSN.L","SSE.L","STAN.L"]
DAX40 = ["ADS.DE","ALV.DE","BAS.DE","BAYN.DE","BMW.DE","CON.DE","DBK.DE","DTE.DE","DPW.DE","FRE.DE","HEN3.DE","IFX.DE","LIN.DE","MRK.DE","MUV2.DE","PUM.DE","RWE.DE","SAP.DE","SIE.DE","VOW3.DE","VNA.DE","ZAL.DE","DHER.DE","BEI.DE","MTX.DE","HNR1.DE","SY1.DE","1COV.DE","AIR.DE","ENR.DE","HFG.DE","PAH3.DE","HLE.DE","DWNI.DE"]
CAC40 = ["AI.PA","AIR.PA","ALO.PA","BN.PA","BNP.PA","CAP.PA","CA.PA","CS.PA","DG.PA","EL.PA","ENGI.PA","GLE.PA","HO.PA","KER.PA","LR.PA","MC.PA","OR.PA","PUB.PA","RI.PA","SAF.PA","SAN.PA","SGO.PA","STLA.PA","SU.PA","SW.PA","TTE.PA","URW.AS","VIE.PA","VIV.PA","WLN.PA"]
INDICES = ["^GSPC","^NDX","^DJI","^RUT","^VIX","^FTSE","^GDAXI","^FCHI","^STOXX50E"]
FOREX_MAJORS = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X","EURGBP=X"]

@lru_cache(maxsize=1)
def load_sp500_symbols():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        syms = df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper().tolist()
        return sorted(list(set(syms)))
    except Exception:
        return SP100

@lru_cache(maxsize=1)
def load_nasdaq100_symbols():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        cand = None
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if any("ticker" in c or "symbol" in c for c in cols):
                cand = t; break
        if cand is None: return []
        col = [c for c in cand.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()][0]
        return sorted(list(set(cand[col].dropna().astype(str).str.upper().tolist())))
    except Exception:
        return []

@lru_cache(maxsize=1)
def load_ftse350_symbols():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/FTSE_350_Index")
        cand = None
        for t in tables:
            if any(str(c).upper() in ("EPIC","TIDM","TICKER") for c in t.columns):
                cand = t; break
        if cand is None: return FTSE100
        col = next(c for c in cand.columns if str(c).upper() in ("EPIC","TIDM","TICKER"))
        syms = cand[col].dropna().astype(str).str.upper()
        return sorted(list(set((syms + ".L").tolist())))
    except Exception:
        return FTSE100

def pick_universe(uni: str, custom: str, uploaded_tickers=None):
    if uni == "All (US+UK+EU)":      return sorted(list(set(SP100 + FTSE100 + DAX40 + CAC40)))
    if uni == "US (S&P100)":         return SP100
    if uni == "US (S&P500)":         return load_sp500_symbols()
    if uni == "US (Nasdaq-100)":     return load_nasdaq100_symbols() or SP100
    if uni == "UK (FTSE100)":        return FTSE100
    if uni == "UK (FTSE350)":        return load_ftse350_symbols()
    if uni == "EU (DAX40 + CAC40)":  return sorted(list(set(DAX40 + CAC40)))
    if uni == "Indices (US/UK/EU)":  return INDICES
    if uni == "Forex (Majors)":      return FOREX_MAJORS
    if uni == "Custom (upload CSV)": return uploaded_tickers or []
    toks = [t.strip().upper() for t in (custom or "").split(",") if t.strip()]
    return toks if toks else sorted(list(set(SP100 + FTSE100 + DAX40 + CAC40)))

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Risk & Account")
    equity = st.number_input("Equity (£)", min_value=100.0, value=float(cfg["equity_gbp"]), step=50.0, format="%.2f")
    risk_pct = st.slider("Risk per trade (%)", 0.25, 5.0, float(cfg["risk_per_trade_pct"]), 0.25)
    band, color = ("Safe","#16a34a") if risk_pct<=1.0 else (("Moderate","#f59e0b") if risk_pct<=2.0 else ("High","#dc2626"))
    st.markdown(f"<div style='padding:8px;border-radius:10px;background:{color};color:white;font-weight:700;text-align:center;'>Risk Dial: {risk_pct:.2f}% — {band}</div>", unsafe_allow_html=True)

    max_lev = st.select_slider("Max leverage", options=[1,2,3,4,5], value=int(cfg["max_leverage"]))
    max_pos = st.select_slider("Max open positions (per day)", options=[1,2,3,4,5], value=int(cfg["max_open_positions"]))
    dls = st.slider("Daily loss stop (%) (info only)", 1.0, 10.0, float(cfg["daily_loss_stop_pct"]), 0.5)
    hold_days = st.select_slider("Max hold days", options=[1,2,3], value=int(cfg["hold_days_max"]))
    allow_shorts = st.checkbox("Allow SHORT setups", value=bool(cfg["allow_shorts"]))

    st.header("Universe & Scan")
    uni = st.selectbox("Universe", [
        "All (US+UK+EU)", "US (S&P100)", "US (S&P500)", "US (Nasdaq-100)",
        "UK (FTSE100)", "UK (FTSE350)", "EU (DAX40 + CAC40)",
        "Indices (US/UK/EU)", "Forex (Majors)", "Custom (enter tickers)", "Custom (upload CSV)"
    ], index=0)
    custom_tickers = st.text_area("Custom tickers (AAPL, AZN.L, EURUSD=X, GBPUSD=X)").strip()
    uploaded_tickers = []
    if uni == "Custom (upload CSV)":
        uploaded = st.file_uploader("Upload CSV with 'ticker' or 'symbol' column", type=["csv"])
        if uploaded:
            dfu = pd.read_csv(uploaded)
            col = next((c for c in dfu.columns if str(c).lower() in ("ticker","symbol","epic","tidm")), dfu.columns[0])
            uploaded_tickers = (dfu[col].astype(str).str.strip().str.upper().tolist())
            st.caption(f"Loaded {len(uploaded_tickers)} tickers from CSV.")
    timeframe_days = st.slider("History window (days)", 120, 1000, int(cfg["timeframe_days"]), 10)

    st.header("News Intelligence")
    newsapi_key = _get_secret("NEWSAPI_KEY")
    finnhub_key = _get_secret("FINNHUB_KEY")
    te_key      = _get_secret("TE_API_KEY")
    lookback_hours = st.slider("News lookback (hours)", 12, 168, int(cfg.get("lookback_hours",72)), 6)

    st.header("Event Guards")
    earnings_window_days = st.number_input("Earnings window (± days)", 0, 10, int(cfg["earnings_window_days"]))
    earnings_action = st.selectbox("Earnings rule", ["Warn","Skip"], index=0 if cfg["earnings_action"]=="Warn" else 1)
    macro_block_days = st.number_input("Macro window (± days)", 0, 5, int(cfg["macro_block_days"]))
    macro_action = st.selectbox("Macro rule", ["Warn","Skip"], index=0 if cfg["macro_action"]=="Warn" else 1)

    st.divider()
    st.caption("Optional: auto-import high-impact macro for next 14 days (TradingEconomics).")

    if st.button("Auto-import macro dates"):
        dates = []
        try:
            if te_key:
                start = dt.date.today(); end = start + dt.timedelta(days=14)
                url = "https://api.tradingeconomics.com/calendar"
                params = {"d1":start.isoformat(),"d2":end.isoformat(),"importance":3,"format":"json","c": te_key}
                r = requests.get(url, params=params, headers=UA, timeout=15); r.raise_for_status()
                for ev in r.json() or []:
                    ts = ev.get("Date")
                    if ts:
                        try: dates.append(dt.datetime.fromisoformat(ts.replace("Z","")).date())
                        except Exception: pass
        except Exception:
            pass
        st.session_state["macro_dates"] = sorted(list(set(dates)))
        st.success(f"Imported {len(st.session_state['macro_dates'])} dates.") if dates else st.info("No dates imported (no key or API returned nothing).")

    st.caption("Manual macro dates (YYYY-MM-DD)")
    new_macro = st.text_input("Add a macro date").strip()
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Add date"):
            try:
                d = dt.date.fromisoformat(new_macro)
                if d not in st.session_state["macro_dates"]:
                    st.session_state["macro_dates"].append(d)
                    st.session_state["macro_dates"].sort()
            except Exception:
                st.warning("Invalid date format. Use YYYY-MM-DD.")
    with c2:
        if st.button("Clear dates"):
            st.session_state["macro_dates"] = []

    st.header("Execution Filters")
    assumed_spread_bps = st.number_input("Assumed spread+slippage (bps)", 0, 100, int(cfg["assumed_spread_bps"]))
    min_stop_to_spread = st.number_input("Min stop distance ÷ spread (×)", 1, 100, int(cfg["min_stop_to_spread"]))

    st.header("Portfolio Builder")
    corr_threshold = st.slider("Correlation threshold (20d)", 0.0, 1.0, float(cfg["corr_threshold"]), 0.05)
    portfolio_max_positions = st.select_slider("Max positions in portfolio", options=[1,2,3,4,5], value=int(cfg["portfolio_max_positions"]))
    partial_exits = st.checkbox("Plan partial exits (50% @1R, 50% @2R)", value=bool(cfg["partial_exits"]))

    st.divider()
    if st.button("Refresh symbol lists (clear cache)"):
        try:
            load_sp500_symbols.cache_clear()
            load_nasdaq100_symbols.cache_clear()
            load_ftse350_symbols.cache_clear()
            st.success("Symbol caches cleared. Run the scan again.")
        except Exception:
            st.info("Nothing to clear yet.")

# ----------------------------- Regime -----------------------------
def pct_above_sma50(tickers, days=200):
    ok = tot = 0
    for t in tickers[:60]:
        try:
            df = yf.download(t, period=f"{days}d", progress=False, auto_adjust=True)
            if df is None or df.empty: continue
            sma50 = df["Close"].rolling(50).mean().iloc[-1]
            if not np.isnan(sma50):
                tot += 1
                if df["Close"].iloc[-1] > sma50: ok += 1
        except Exception:
            continue
    return (ok / tot * 100.0) if tot>0 else np.nan

def vix_trend(days=200):
    try:
        v = yf.download("^VIX", period=f"{days}d", progress=False, auto_adjust=True)
        if v is None or v.empty: return np.nan, np.nan, False
        sma50 = v["Close"].rolling(50).mean().iloc[-1]
        close = v["Close"].iloc[-1]
        return float(close), float(sma50), (close > sma50)
    except Exception:
        return np.nan, np.nan, False

colA, colB, colC, colD = st.columns(4)
breadth = pct_above_sma50(SP100)
vix_c, vix_sma50, vix_rising = vix_trend()
regime = "Risk-On" if (not np.isnan(breadth) and breadth >= 55 and not vix_rising) else "Neutral"
if not np.isnan(breadth) and breadth <= 45 and vix_rising: regime = "Risk-Off"
with colA: st.metric("Regime", regime)
with colB: st.metric("% SP100 > 50SMA", f"{breadth:.1f}%" if not np.isnan(breadth) else "n/a")
with colC: st.metric("VIX / 50SMA", f"{vix_c:.1f} / {vix_sma50:.1f}" if (not np.isnan(vix_c) and not np.isnan(vix_sma50)) else "n/a")
with colD: st.metric("Risk Budget (worst-case today)", f"{risk_pct * max_pos:.1f}%")
st.divider()

# ----------------------------- Helpers -----------------------------
CCY_MAP = {".L":"GBP",".DE":"EUR",".PA":"EUR",".AS":"EUR",".MI":"EUR",".BR":"EUR",".MC":"EUR",".SW":"CHF",".HK":"HKD",".TO":"CAD",".NE":"CAD"}
def is_fx(symbol: str) -> bool: return symbol.endswith("=X")

@lru_cache(maxsize=8192)
def get_listing_currency(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info or {}
        c = info.get("currency")
        if c: return str(c).upper()
    except Exception:
        pass
    for suf, ccy in CCY_MAP.items():
        if ticker.endswith(suf): return ccy
    return "USD"

@lru_cache(maxsize=4096)
def _last_valid_close(pair: str) -> Optional[float]:
    """Return most recent valid daily close for FX pair in last 60 days (robust to NaNs/junk)."""
    try:
        df = yf.download(pair, period="60d", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        vals = pd.to_numeric(df["Close"], errors="coerce").astype(float).to_numpy()
        for x in reversed(vals):
            if x is not None and np.isfinite(x) and x > 0:
                return float(x)
        return None
    except Exception:
        return None

@lru_cache(maxsize=1024)
def fx_rate(ccy_from: str, ccy_to: str) -> float:
    ccy_from, ccy_to = (ccy_from or "USD").upper(), (ccy_to or "GBP").upper()
    if ccy_from == ccy_to: return 1.0
    direct = _last_valid_close(f"{ccy_from}{ccy_to}=X")
    if direct is not None and np.isfinite(direct): return float(direct)
    inverse = _last_valid_close(f"{ccy_to}{ccy_from}=X")
    if inverse is not None and np.isfinite(inverse) and inverse != 0: return float(1.0 / inverse)
    return 1.0

@lru_cache(maxsize=2048)
def last_close_and_adv_usd(ticker: str):
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty: return float("nan"), float("nan"), None
        df = df.dropna(how="all")
        close_series = pd.to_numeric(df.get("Close", pd.Series([], dtype=float)), errors="coerce").replace(0, np.nan).dropna()
        last_close = float(close_series.iloc[-1]) if not close_series.empty else float("nan")
        ccy = get_listing_currency(ticker)
        tail = df.tail(30).copy()
        closes = pd.to_numeric(tail.get("Close", pd.Series([], dtype=float)), errors="coerce").fillna(method="ffill")
        vols   = pd.to_numeric(tail.get("Volume", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
        adv_ccy = float((closes * vols).mean()) if len(closes) and len(vols) else float("nan")
        adv_usd = adv_ccy * fx_rate(ccy, "USD") if np.isfinite(adv_ccy) else float("nan")
        return last_close, adv_usd, ccy
    except Exception:
        return float("nan"), float("nan"), None

def filter_by_liquidity_and_price(tickers, min_adv_usd=20_000_000, min_price=2.0, max_price=500.0):
    kept, rows = [], []
    for t in tickers:
        if is_fx(t): kept.append(t); continue
        px, adv_usd, ccy = last_close_and_adv_usd(t)
        if np.isfinite(px) and min_price <= px <= max_price and np.isfinite(adv_usd) and adv_usd >= min_adv_usd:
            kept.append(t); rows.append({"Ticker": t, "Price": round(px, 4), "CCY": ccy, "ADV_USD_30D": round(adv_usd, 2)})
    return kept, pd.DataFrame(rows)

def position_size(entry: float, stop: float, equity_gbp: float, risk_pct: float, max_lev: float, ticker: str = None):
    risk_cash_gbp = round(equity_gbp * (risk_pct / 100.0), 2)
    stop_dist = abs(float(entry) - float(stop))
    if entry <= 0 or stop_dist <= 0: return 0, risk_cash_gbp, 0.0
    listing_ccy = get_listing_currency(ticker) if ticker else "GBP"
    fx_to_gbp = fx_rate(listing_ccy, "GBP")
    risk_per_share_gbp = stop_dist * fx_to_gbp
    if risk_per_share_gbp <= 0: return 0, risk_cash_gbp, 0.0
    cash_cap_gbp = 0.9 * equity_gbp * max_lev
    qty_risk = int(risk_cash_gbp // risk_per_share_gbp)
    qty_cash = int(cash_cap_gbp // (entry * fx_to_gbp))
    qty = max(0, min(qty_risk, qty_cash))
    notional_gbp = round(qty * entry * fx_to_gbp, 2)
    return qty, risk_cash_gbp, notional_gbp

def spread_ok(entry, stop, bps, multiple):
    stop_dist = abs(entry - stop)
    spread_px = entry * (bps / 10000.0)
    return stop_dist >= (spread_px * multiple), spread_px, stop_dist

# ----------------------------- Strategies -----------------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def ema(series, period): return series.ewm(span=period, adjust=False).mean()
def sma(series, period): return series.rolling(period).mean()

def scan_signals(df: pd.DataFrame, allow_shorts: bool = False):
    out = []
    if df is None or df.empty or len(df) < 200: return out
    df = df.copy()
    df["20H"]   = df["High"].rolling(20).max()
    df["20L"]   = df["Low"].rolling(20).min()
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"]= sma(df["Close"], 200)
    df["EMA20"] = ema(df["Close"], 20)
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    last = df.iloc[-1]; prev = df.iloc[-2] if len(df)>=2 else None; date = df.index[-1]
    if last["Close"] >= last["20H"] - 1e-9:
        entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
        stop  = float(entry - 2 * atr14); take = float(entry + 2 * (entry - stop))
        if entry > 0 and stop < entry:
            out.append({"strategy":"Breakout Long","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":False})
    bullish_rev = prev is not None and (last["Close"] > last["Open"]) and (prev["Close"] < prev["Open"])
    if last["SMA50"] > last["SMA200"] and abs(last["Close"] - last["EMA20"]) / max(1e-9,last["Close"]) < 0.01 and bullish_rev:
        entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
        stop  = float(min(last["Low"], last["Close"] - 2 * atr14))
        risk  = entry - stop; take  = float(entry + 2 * risk)
        if risk > 0:
            out.append({"strategy":"Pullback Long","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":False})
    if last["Close"] > last["SMA200"] and last["RSI14"] < 30:
        entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
        stop  = float(min(last["Low"], last["Close"] - 2 * atr14))
        risk  = max(entry - stop, 1e-6); take = float(entry + 1.5 * risk)
        out.append({"strategy":"MeanRev Long","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":1.5,"is_short":False})
    if allow_shorts:
        if last["Close"] <= last["20L"] + 1e-9:
            entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
            stop  = float(entry + 2 * atr14); take = float(entry - 2 * (stop - entry))
            out.append({"strategy":"Breakdown Short","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":True})
        if last["Close"] < last["SMA200"] and last["RSI14"] > 70:
            entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
            stop  = float(max(last["High"], last["Close"] + 2 * atr14))
            risk  = max(stop - entry, 1e-6); take = float(entry - 1.5 * risk)
            out.append({"strategy":"MeanRev Short","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":1.5,"is_short":True})
    return out

def scan_signals_fx(df: pd.DataFrame, allow_shorts: bool = True):
    out = []
    if df is None or df.empty or len(df) < 200: return out
    df = df.copy()
    df["20H"] = df["High"].rolling(20).max(); df["20L"] = df["Low"].rolling(20).min()
    df["EMA20"] = ema(df["Close"], 20); df["EMA50"] = ema(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200); df["RSI14"] = rsi(df["Close"], 14); df["ATR14"] = atr(df, 14)
    last = df.iloc[-1]; date = df.index[-1]
    if last["Close"] >= last["20H"] - 1e-9:
        entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
        stop  = float(entry - 2 * atr14); take = float(entry + 2 * (entry - stop))
        out.append({"strategy":"FX Breakout Long","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":False})
    if allow_shorts and last["Close"] <= last["20L"] + 1e-9:
        entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
        stop  = float(entry + 2 * atr14); take = float(entry - 2 * (stop - entry))
        out.append({"strategy":"FX Breakout Short","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":True})
    if last["EMA20"] > last["EMA50"]:
        entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
        stop  = float(entry - 2 * atr14); take = float(entry + 2 * (entry - stop))
        out.append({"strategy":"FX Pullback Long (EMA20>EMA50)","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":False})
    if allow_shorts and last["EMA20"] < last["EMA50"]:
        entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
        stop  = float(entry + 2 * atr14); take = float(entry - 2 * (stop - entry))
        out.append({"strategy":"FX Pullback Short (EMA20<EMA50)","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":True})
    if last["Close"] > last["SMA200"] and last["RSI14"] < 30:
        entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
        stop  = float(entry - 2 * atr14); take = float(entry + 1.5 * (entry - stop))
        out.append({"strategy":"FX MeanRev Long (RSI14<30)","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":1.5,"is_short":False})
    if allow_shorts and last["Close"] < last["SMA200"] and last["RSI14"] > 70:
        entry = float(last["Close"]); atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
        stop  = float(entry + 2 * atr14); take = float(entry - 1.5 * (stop - entry))
        out.append({"strategy":"FX MeanRev Short (RSI14>70)","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":1.5,"is_short":True})
    return out

# ----------------------------- News + Aggregation -----------------------------
def _ts_now_utc() -> dt.datetime: return dt.datetime.utcnow()
def _to_iso_z(ts: dt.datetime) -> str:
    if ts.tzinfo is not None: ts = ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return ts.isoformat(timespec="seconds") + "Z"

TRACKING_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","yclid","gclid","fbclid","mc_cid","mc_eid","cmpid"}
def _normalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k not in TRACKING_PARAMS]
        return urlunparse((p.scheme, p.netloc.lower(), p.path, p.params, urlencode(q, doseq=True), ""))
    except Exception:
        return url or ""
def _domain(url: str) -> str:
    try: return urlparse(url).netloc.lower()
    except Exception: return ""

def _article(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": record.get("source", ""),
        "title": (record.get("title","") or "").replace("\n"," ").strip(),
        "summary": (record.get("summary","") or "").replace("\n"," ").strip(),
        "url": _normalize_url(record.get("url", "")),
        "published": record.get("published", ""),
    }

def news_newsapi(q: str, api_key: str, lookback_hours: int = 72, page_size: int = 20) -> List[Dict[str, Any]]:
    if not api_key: return []
    url = "https://newsapi.org/v2/everything"
    from_dt = _ts_now_utc() - dt.timedelta(hours=lookback_hours)
    params = {"q": q, "from": _to_iso_z(from_dt), "language":"en", "searchIn":"title,description",
              "sortBy":"publishedAt", "pageSize": min(max(1, page_size), 100), "apiKey": api_key}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=12); r.raise_for_status()
        data = r.json() or {}; out = []
        for a in data.get("articles", [])[:params["pageSize"]]:
            ts = a.get("publishedAt") or ""
            out.append(_article({
                "source": (a.get("source") or {}).get("name",""),
                "title": a.get("title",""), "summary": a.get("description",""),
                "url": a.get("url",""), "published": ts if ts.endswith("Z") else (ts.replace("Z","")+"Z" if ts else "")
            }))
        return out
    except Exception:
        return []

def news_finnhub(symbol: str, api_key: str, lookback_hours: int = 72) -> List[Dict[str, Any]]:
    if not api_key: return []
    to_dt = _ts_now_utc().date()
    from_dt = to_dt - dt.timedelta(days=max(1, (lookback_hours+23)//24))
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": from_dt.isoformat(), "to": to_dt.isoformat(), "token": api_key}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=12); r.raise_for_status()
        arr = r.json() or []; out = []
        for a in arr:
            epoch = a.get("datetime"); 
            if not epoch: continue
            ts = dt.datetime.utcfromtimestamp(float(epoch))
            out.append(_article({
                "source": a.get("source",""), "title": a.get("headline",""),
                "summary": a.get("summary",""), "url": a.get("url",""),
                "published": _to_iso_z(ts)
            }))
        return out
    except Exception:
        return []

def news_yahoo(symbol: str, lookback_hours: int = 72) -> List[Dict[str, Any]]:
    try:
        t = yf.Ticker(symbol); items = getattr(t, "news", []) or []; out = []
        cut_ts = _ts_now_utc() - dt.timedelta(hours=lookback_hours)
        for a in items:
            ts_epoch = a.get("providerPublishTime", 0); 
            if not ts_epoch: continue
            ts = dt.datetime.utcfromtimestamp(float(ts_epoch))
            if ts < cut_ts: continue
            out.append(_article({
                "source": a.get("publisher",""), "title": a.get("title",""),
                "summary": a.get("summary",""), "url": a.get("link",""),
                "published": _to_iso_z(ts)
            }))
        return out
    except Exception:
        return []

def news_rss_google(q: str, lookback_hours: int = 72) -> List[Dict[str, Any]]:
    try:
        q_enc = quote_plus(f"{q} when:{lookback_hours}h")
        url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-GB&gl=GB&ceid=GB:en"
        feed = feedparser.parse(url); out = []; cut_ts = _ts_now_utc() - dt.timedelta(hours=lookback_hours)
        for e in feed.entries[:30]:
            ts = None
            if "published_parsed" in e and e.published_parsed: ts = dt.datetime(*e.published_parsed[:6])
            elif "updated_parsed" in e and e.updated_parsed:  ts = dt.datetime(*e.updated_parsed[:6])
            elif "published" in e:
                try: ts = dt.datetime.fromisoformat(e.published.replace("Z",""))
                except Exception: ts = None
            if ts is None or ts < cut_ts: continue
            src = e["source"].get("title","") if isinstance(e.get("source"), dict) else ""
            out.append(_article({"source": src, "title": e.get("title",""), "summary": e.get("summary",""),
                                 "url": e.get("link",""), "published": _to_iso_z(ts)}))
        return out
    except Exception:
        return []

def get_company_name(symbol: str) -> str:
    try:
        t = yf.Ticker(symbol)
        name = ""
        try: name = (getattr(t, "fast_info", {}) or {}).get("shortName") or ""
        except Exception: name = ""
        if not name:
            info = t.info or {}
            name = info.get("longName") or info.get("shortName") or ""
        return name or symbol
    except Exception:
        return symbol

def fetch_all_news(symbol: str, newsapi_key: Optional[str], finnhub_key: Optional[str], lookback_hours: int = 72) -> List[Dict[str, Any]]:
    name = get_company_name(symbol); q = f'"{name}" OR {symbol}'
    out: List[Dict[str, Any]] = []
    out += news_newsapi(q, newsapi_key or "", lookback_hours=lookback_hours)
    out += news_finnhub(symbol, finnhub_key or "", lookback_hours=lookback_hours)
    out += news_yahoo(symbol, lookback_hours=lookback_hours)
    out += news_rss_google(q, lookback_hours=lookback_hours)
    seen_urls = set(); seen_title_dom = set(); dedup = []
    for a in out:
        url = _normalize_url(a.get("url","")); title = (a.get("title","") or "").strip(); dom = _domain(url)
        key_t = (title.lower(), dom)
        if (url and url in seen_urls) or key_t in seen_title_dom: continue
        seen_urls.add(url); seen_title_dom.add(key_t); dedup.append(a)
    final = []
    for a in dedup:
        try:
            ph = (dt.datetime.utcnow() - dt.datetime.fromisoformat(a.get("published","").replace("Z",""))).total_seconds()/3600.0
        except Exception:
            ph = 1e9
        if ph <= lookback_hours: final.append(a)
    return final

# Aggregation / confidence
RUMOUR_TERMS = re.compile(r"(rumou?r|speculat(e|ion)|report(ed|edly)|'?sources?\s+say'?|leak(ed|s?)|unconfirm|whisper|chatter|talks?\s+that)", re.I)
OFFICIAL_DOMAINS = ("sec.gov", "londonstockexchange", "investegate.co.uk", "globenewswire.com","prnewswire.com","businesswire.com")
OFFICIAL_HINTS = re.compile(r"\b(RNS|Regulatory News Service|Form\s?8[- ]?K|SEC filing|trading (update|statement)|press release)\b", re.I)

def is_rumour(title: str, summary: str) -> bool:
    return bool(RUMOUR_TERMS.search((title or "") + " " + (summary or "")))

def source_weight(name: str) -> float:
    SW = {"Bloomberg":1.0,"Reuters":1.0,"Financial Times":1.0,"The Wall Street Journal":1.0,"CNBC":0.8,"MarketWatch":0.8,"Yahoo Finance":0.7,"Seeking Alpha":0.6}
    if not name: return 0.5
    for k,w in SW.items():
        if k.lower() in name.lower(): return w
    return 0.5

def _is_official(source: str, title: str, url: str) -> bool:
    dom = _domain(url)
    if any(d in dom for d in OFFICIAL_DOMAINS): return True
    if dom.startswith("investors.") or dom.startswith("ir.") or ".ir." in dom: return True
    if OFFICIAL_HINTS.search(title or ""): return True
    if OFFICIAL_HINTS.search(source or ""): return True
    return False

def aggregate_news(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not articles:
        return {"count":0,"mean_sent":0.0,"rumour_ratio":0.0,"score":0.0,"fresh_hours":9999.0,"has_official":False,"top":[]}
    now = dt.datetime.utcnow()
    seen = set(); weighted_sum=0.0; weight_total=0.0; rumours=0; has_official=False; most_recent=9999.0; top=[]
    for a in articles:
        title=a.get("title","") or ""; summary=a.get("summary","") or ""; url=a.get("url","") or ""; source=a.get("source","") or ""
        key=(title.strip().lower(), _domain(url)); 
        if key in seen: continue
        seen.add(key)
        vs = analyzer.polarity_scores((title+". "+summary).strip()); s = float(vs.get("compound",0.0)); s = max(-1.0,min(1.0,s))
        w_src = source_weight(source); rum = is_rumour(title, summary); rumours += int(rum)
        ts = a.get("published",""); 
        try: age_h = max(0.0,(now - dt.datetime.fromisoformat(ts.replace("Z",""))).total_seconds()/3600.0)
        except Exception: age_h = 9999.0
        most_recent = min(most_recent, age_h)
        official = _is_official(source, title, url); has_official = has_official or official
        recency = 1.0 if age_h<=6 else (0.75 if age_h<=24 else (0.5 if age_h<=72 else 0.25))
        w = w_src * recency * (1.15 if official else 1.0)
        weighted_sum += s*w; weight_total += w
        if len(top)<5: top.append({"source":source,"title":title,"url":url,"published":ts,"sent":round(s,3)})
    count = len(seen); mean_sent = (weighted_sum/weight_total) if weight_total>0 else 0.0
    rumour_ratio = rumours / max(1,count); score = mean_sent * (1.0 - 0.5 * rumour_ratio)
    if has_official and most_recent < 24: score += 0.05
    score = max(-1.0, min(1.0, score))
    return {"count":int(count),"mean_sent":float(mean_sent),"rumour_ratio":float(rumour_ratio),
            "score":float(score),"fresh_hours":float(most_recent),"has_official":bool(has_official),"top":top}

def fuse(strategy_signal: Dict[str, Any], news_summary: Dict[str, Any]) -> Dict[str, Any]:
    news_score = float(news_summary.get("score", 0.0) or 0.0)
    news_count = int(news_summary.get("count", 0) or 0)
    rumour_ratio = float(news_summary.get("rumour_ratio", 0.0) or 0.0)
    fresh_hours  = float(news_summary.get("fresh_hours", 9999.0) or 9999.0)
    has_official = bool(news_summary.get("has_official", False))
    is_short  = bool(strategy_signal.get("IsShort", False))
    base_conf = 0.60
    aligned = max(0.0, -news_score if is_short else news_score) * 0.30
    opposed = max(0.0,  news_score if is_short else -news_score) * 0.40
    conf = base_conf + aligned - opposed
    conf -= 0.10 * min(1.0, rumour_ratio/0.6) if rumour_ratio<=0.6 else (0.10 + 0.15 * min(1.0,(rumour_ratio-0.6)/0.4))
    if fresh_hours < 48: conf += max(0.0,(48.0-fresh_hours)/48.0)*0.10
    if has_official: conf += 0.05
    conf += min(news_count,10)*0.015
    conf = max(0.0,min(1.0,conf))
    gate = (conf >= 0.35) and (rumour_ratio <= 0.8) and (fresh_hours <= 72.0)
    return {"NewsScore":round(news_score,3),"NewsCount":news_count,"RumourRatio":round(rumour_ratio,3),
            "Confidence":round(conf,3),"Gate":bool(gate),"FreshHours":float(fresh_hours),"HasOfficial":bool(has_official)}

def near_macro(date_list, window_days):
    today = dt.date.today()
    for d in date_list:
        if abs((d - today).days) <= window_days: return True, (d - today).days
    return False, None

# ----------------------------- Tabs -----------------------------
tabs = st.tabs(["🔎 Scanner", "📈 Chart", "⭐ Watchlist", "📒 Journal & 📊 Edge", "📈 Planner", "ℹ️ Help"])

# ---- Scanner ----
with tabs[0]:
    st.subheader("Signal Scanner")
    tickers = pick_universe(uni, custom_tickers, uploaded_tickers=uploaded_tickers)

    lc1, lc2 = st.columns([2,1])
    with lc1: enforce_liq = st.checkbox("Enforce 30D ADV ≥ $20M and price ∈ [2, 500] (equities only)", value=True)
    with lc2: show_liq_table = st.checkbox("Show liquidity table", value=False)

    liq_table = None
    if enforce_liq:
        tickers, liq_table = filter_by_liquidity_and_price(tickers)
        if show_liq_table and liq_table is not None and not liq_table.empty:
            st.dataframe(liq_table, use_container_width=True)

    macro_dates = st.session_state.get("macro_dates", [])
    run = st.button(f"Run scan for {len(tickers)} symbols")

    base_floor = 0.45 if regime=="Risk-On" else (0.55 if regime=="Neutral" else 0.65)
    st.caption("Filters (apply after scan):")
    fcols = st.columns(6)
    with fcols[0]: f_longs  = st.checkbox("Show Longs", value=True)
    with fcols[1]: f_shorts = st.checkbox("Show Shorts", value=True)
    with fcols[2]: f_spread = st.checkbox("SpreadOK only", value=True)
    with fcols[3]: f_conf   = st.slider("Min Confidence", 0.0, 1.0, float(base_floor), 0.05)
    with fcols[4]: f_skipfl = st.checkbox("Hide Filtered", value=True)
    with fcols[5]: confl    = st.checkbox("Confluence mode (strict)", value=False)

    if run:
        st.info("Scanning charts + fetching news + applying guards…")
        rows = []
        for t in tickers:
            try:
                df = yf.download(t, period=f"{timeframe_days}d", progress=False, auto_adjust=True)
                if df is None or df.empty or len(df) < 200: continue
                sigs = scan_signals_fx(df) if is_fx(t) else scan_signals(df, allow_shorts=allow_shorts)
                if not sigs: continue
                arts = fetch_all_news(t, newsapi_key, finnhub_key, lookback_hours=lookback_hours)
                news = aggregate_news(arts)
                near_macro_flag, _ = near_macro(macro_dates, macro_block_days)
                for s in sigs:
                    entry, stop, tp = s["entry"], s["stop"], s["take_profit"]
                    qty, risk_cash, notional = position_size(entry, stop, equity, risk_pct, max_lev, ticker=t)
                    if qty < 1: continue
                    base = {"Ticker":t,"Strategy":s["strategy"],"Date":s["date"].date().isoformat(),
                            "Entry":round(entry,4),"Stop":round(stop,4),"TakeProfit":round(tp,4),
                            "Qty":qty,"Risk_£":round(risk_cash,2),"Notional_£":round(notional,2),
                            "R_multiple":s.get("r_multiple",2.0),"IsShort":s.get("is_short",False)}
                    fused = fuse(base, news)
                    local_multiple = 4 if is_fx(t) else min_stop_to_spread
                    ok_spread, spread_px, stop_dist = spread_ok(entry, stop, assumed_spread_bps, local_multiple)
                    near_earnings = False  # placeholder
                    earnings_block = (earnings_action=="Skip" and near_earnings)
                    macro_block    = (macro_action=="Skip" and near_macro_flag)
                    too_tight = (abs(entry - stop) / max(1e-9, entry)) < 0.0015
                    filtered = (not ok_spread) or earnings_block or macro_block or too_tight
                    if partial_exits:
                        r = abs(entry - stop); pt1 = entry + ( r if not base["IsShort"] else -r); pt2 = entry + (2*r if not base["IsShort"] else -2*r)
                        blended_R = 1.5
                    else:
                        pt1 = np.nan; pt2 = np.nan; blended_R = base["R_multiple"]
                    rows.append({
                        **base, **fused,
                        "StopDist": round(stop_dist,4), "AssumedSpreadPx": round(spread_px,4), "SpreadOK": ok_spread,
                        "NearEarnings": bool(near_earnings), "MacroNear": bool(near_macro_flag),
                        "Partial_TP1": round(pt1,4) if pt1==pt1 else None, "Partial_TP2": round(pt2,4) if pt2==pt2 else None,
                        "Blended_R_Target": blended_R, "Filtered": filtered, "TooTight": too_tight
                    })
            except Exception:
                continue

        if rows:
            df_out = pd.DataFrame(rows)
            mask = pd.Series([True]*len(df_out))
            if not f_longs:  mask &= (df_out["IsShort"] == True)
            if not f_shorts: mask &= (df_out["IsShort"] == False)
            if f_spread:     mask &= (df_out["SpreadOK"] == True)
            if f_skipfl:     mask &= (df_out["Filtered"] == False)
            mask &= (df_out["Confidence"] >= f_conf)
            if confl:
                mask &= (df_out["SpreadOK"] == True) & (df_out["MacroNear"] == False) & (df_out["Confidence"] >= max(f_conf, 0.6))
            df_view = df_out[mask].sort_values(["Confidence","Ticker"], ascending=[False, True]).reset_index(drop=True)
            st.session_state["last_signals"] = df_view.copy()

            st.markdown("### Trade ideas")
            if df_view.empty:
                st.info("No results after filters. Try relaxing them above or scanning during active market hours.")
            else:
                for _, r in df_view.iterrows():
                    conf = float(r["Confidence"])
                    conf_band = "✅" if conf>=0.6 else ("⚠️" if conf>=0.45 else "❌")
                    badge_color = "#16a34a" if conf>=0.6 else ("#f59e0b" if conf>=0.45 else "#dc2626")
                    side = "Short" if r["IsShort"] else "Long"
                    st.markdown(
                        f"""
                        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:12px;margin-bottom:10px;">
                          <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div><b>{r['Ticker']}</b> — {r['Strategy']} ({side})</div>
                            <div style="background:{badge_color};color:white;border-radius:999px;padding:4px 10px;font-weight:700;">
                              Confidence {conf:.2f} {conf_band}
                            </div>
                          </div>
                          <div style="display:flex;gap:16px;margin-top:6px;flex-wrap:wrap;">
                            <div>Entry: <b>{r['Entry']}</b></div>
                            <div>Stop: <b>{r['Stop']}</b></div>
                            <div>TP: <b>{r['TakeProfit']}</b></div>
                            <div>Qty: <b>{int(r['Qty'])}</b></div>
                            <div>Notional £: <b>{r['Notional_£']}</b></div>
                            <div>SpreadOK: <b>{'Yes' if r['SpreadOK'] else 'No'}</b></div>
                            <div>MacroNear: <b>{'Yes' if r['MacroNear'] else 'No'}</b></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True
                    )

            st.markdown("### Suggested Portfolio (top, uncorrelated)")
            pool = df_view.copy()
            if pool.empty:
                st.info("No unfiltered signals for portfolio build.")
            else:
                def build_portfolio(df, corr_threshold=0.8, max_positions=3):
                    syms = df["Ticker"].unique().tolist(); px = {}
                    for s in syms[:40]:
                        try:
                            hist = yf.download(s, period="60d", progress=False, auto_adjust=True)["Close"].pct_change().dropna()
                            px[s] = hist[-20:]
                        except Exception:
                            continue
                    if not px: return df.head(max_positions)
                    ret = pd.DataFrame(px).dropna(axis=1, how="all").fillna(0.0)
                    if ret.shape[1] == 0: return df.head(max_positions)
                    C = ret.corr(); chosen = []
                    for _, row in df.sort_values("Confidence", ascending=False).iterrows():
                        t = row["Ticker"]; ok = True
                        for c in chosen:
                            if t in C.columns and c in C.columns and abs(C.loc[t, c]) >= corr_threshold: ok = False; break
                        if ok: chosen.append(t)
                        if len(chosen) >= max_positions: break
                    return df[df["Ticker"].isin(chosen)].copy()
                port = build_portfolio(pool, corr_threshold=corr_threshold, max_positions=portfolio_max_positions)
                st.dataframe(port, use_container_width=True)
                st.download_button("Download portfolio (CSV)", port.to_csv(index=False), file_name="portfolio.csv", mime="text/csv")
        else:
            st.warning("No qualifying signals right now. Try a broader universe or relax filters.")

# ---- Chart ----
with tabs[1]:
    st.subheader("Chart preview")
    last = st.session_state.get("last_signals", pd.DataFrame())
    default_symbol = last["Ticker"].tolist()[0] if not last.empty else ("EURUSD=X" if uni=="Forex (Majors)" else "AAPL")
    symbol = st.text_input("Symbol", value=default_symbol).strip().upper()
    hist_days = st.slider("Chart history (days)", 120, 1000, int(cfg["timeframe_days"]), 10)
    if st.button("Show chart"):
        try:
            df = yf.download(symbol, period=f"{hist_days}d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                st.warning("No data for this symbol.")
            else:
                df["SMA200"] = df["Close"].rolling(200).mean()
                df["SMA50"]  = df["Close"].rolling(50).mean()
                df["EMA20"]  = df["Close"].ewm(span=20, adjust=False).mean()
                entry = stop = tp = None
                if not last.empty:
                    recs = last[last["Ticker"]==symbol]
                    if not recs.empty:
                        rec = recs.iloc[0]; entry, stop, tp = rec["Entry"], rec["Stop"], rec["TakeProfit"]
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(df.index, df["Close"], label="Close")
                ax.plot(df.index, df["SMA200"], label="SMA200")
                ax.plot(df.index, df["SMA50"],  label="SMA50")
                ax.plot(df.index, df["EMA20"],  label="EMA20")
                if entry: ax.axhline(entry, linestyle="--", label="Entry")
                if stop:  ax.axhline(stop,  linestyle="--", label="Stop")
                if tp:    ax.axhline(tp,    linestyle="--", label="TP")
                ax.set_title(f"{symbol} — price with MAs and levels"); ax.legend()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Chart error: {e}")

# ---- Watchlist ----
with tabs[2]:
    st.subheader("Your Watchlist")
    wl_in = st.text_input("Add symbol (e.g., AAPL, AZN.L, EURUSD=X)").strip().upper()
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Add to Watchlist"):
            if wl_in and wl_in not in st.session_state["watchlist"]:
                st.session_state["watchlist"].append(wl_in); st.success(f"Added {wl_in}")
    with c2:
        if st.button("Clear Watchlist"):
            st.session_state["watchlist"] = []; st.info("Watchlist cleared.")
    if st.session_state["watchlist"]:
        st.write("Current Watchlist:", ", ".join(st.session_state["watchlist"]))
    else:
        st.info("Your Watchlist is empty. Add symbols above.")

# ---- Journal & Edge ----
with tabs[3]:
    st.subheader("Journal (log completed trades)")
    with st.form("add_trade"):
        c1, c2, c3, c4 = st.columns(4)
        with c1: date = st.date_input("Date", value=dt.date.today())
        with c2: ticker = st.text_input("Ticker", value="AAPL").upper().strip()
        with c3: direction = st.selectbox("Direction", ["Long","Short"])
        with c4: qty = st.number_input("Qty", min_value=1, value=1, step=1)
        c5, c6, c7, c8 = st.columns(4)
        with c5: entry = st.number_input("Entry", min_value=0.0, value=0.0, format="%.4f")
        with c6: stop  = st.number_input("Stop",  min_value=0.0, value=0.0, format="%.4f")
        with c7: exitp = st.number_input("Exit",  min_value=0.0, value=0.0, format="%.4f")
        with c8: fees  = st.number_input("Fees (£)", min_value=0.0, value=0.0, format="%.2f")
        note = st.text_input("Notes", value="")
        submitted = st.form_submit_button("Add to journal")
        if submitted:
            if direction == "Long":
                pl = (exitp - entry) * qty; r_denom = max(entry - stop, 1e-9); r_real = (exitp - entry) / r_denom
            else:
                pl = (entry - exitp) * qty; r_denom = max(stop - entry, 1e-9); r_real = (entry - exitp) / r_denom
            net = pl - fees
            st.session_state["journal"].append({
                "Date": date.isoformat(),"Ticker":ticker,"Direction":direction,"Qty":int(qty),
                "Entry":float(entry),"Stop":float(stop),"Exit":float(exitp),
                "Fees_£":float(fees),"P_L_£":float(pl),"Net_£":float(net),"R":float(r_real),"Notes":note
            })
            st.success(f"Added. Net £{net:.2f}, R={r_real:.2f}")
    if st.session_state["journal"]:
        dfj = pd.DataFrame(st.session_state["journal"])
        st.dataframe(dfj, use_container_width=True)
        wins = (dfj["R"] > 0).sum(); win_rate = wins / max(1, len(dfj)) * 100.0
        avg_win_R = dfj[dfj["R"]>0]["R"].mean() if (dfj["R"]>0).any() else 0.0
        avg_loss_R = dfj[dfj["R"]<=0]["R"].mean() if (dfj["R"]<=0).any() else 0.0
        expectancy_R = (win_rate/100.0)*avg_win_R + (1-win_rate/100.0)*avg_loss_R
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Trades", len(dfj))
        with c2: st.metric("Win rate", f"{win_rate:.1f}%")
        with c3: st.metric("Avg Win (R)", f"{avg_win_R:.2f}")
        with c4: st.metric("Avg Loss (R)", f"{avg_loss_R:.2f}")
        st.metric("Expectancy (R/trade)", f"{expectancy_R:.2f}")
        st.caption("Expectancy > 0 means your system makes R on average per trade.")
    else:
        st.info("No trades yet. Log some to see edge metrics.")

# ---- Planner ----
with tabs[4]:
    st.subheader("Planner — Compounding & Monthly Contributions (what-if)")
    c1, c2, c3 = st.columns(3)
    with c1: start_eq = st.number_input("Starting equity (£)", min_value=100.0, value=float(equity), step=50.0, format="%.2f")
    with c2: monthly_contrib = st.number_input("Monthly contribution (£)", min_value=0.0, value=0.0, step=50.0, format="%.2f")
    with c3: months = st.slider("Projection horizon (months)", 3, 24, 12)
    c4, c5, c6, c7 = st.columns(4)
    with c4: risk_pct_pl = st.number_input("Risk per trade (%)", min_value=0.25, value=float(risk_pct), max_value=5.0, step=0.25)
    with c5: trades_per_day = st.slider("Trades per day", 1, 4, 2)
    with c6: days_per_week = st.slider("Days per week", 1, 5, 4)
    with c7: exp_R = st.slider("Expectancy (R/trade, net)",  -0.30, 0.50, 0.15, 0.05)
    monthly_trades = trades_per_day * days_per_week * 4.33
    E = float(start_eq); rows = []
    for m in range(1, months+1):
        risk_cash = E * (risk_pct_pl / 100.0)
        exp_income = monthly_trades * risk_cash * exp_R
        E_next = E + exp_income + monthly_contrib
        rows.append({"Month": m, "Start_£": round(E,2), "Contrib_£": round(monthly_contrib,2),
                     "Exp_Income_£": round(exp_income,2), "End_£": round(E_next,2),
                     "Risk_per_trade_£": round(risk_cash,2), "Trades/mo": int(monthly_trades)})
        E = E_next
    proj = pd.DataFrame(rows)
    st.dataframe(proj, use_container_width=True)
    st.metric("Projected equity after horizon", f"£{proj['End_£'].iloc[-1]:,.2f}")
    st.metric("Projected monthly income at end", f"£{proj['Exp_Income_£'].iloc[-1]:,.2f}")
    st.caption("Simple what-if model (not advice). Assumes constant expectancy & throughput, fixed risk %, ignores tax/fees variance.")
    st.download_button("Download projection (CSV)", proj.to_csv(index=False), "projection.csv", "text/csv")

# ---- Help ----
with tabs[5]:
    st.subheader("Tips")
    st.markdown("""
- Scan near market hours for better signals. Stocks are quiet on Sundays; FX is 24/5.
- Keep Liquidity gate ON (ADV ≥ $20M, price 2–500) to avoid junk.
- Risk Budget shows worst case if all stops hit that day.
- Journal every trade; scale risk only after expectancy is proven.
    """)
    st.subheader("Disclaimer")
    st.markdown("Educational use only. Not financial advice.")
