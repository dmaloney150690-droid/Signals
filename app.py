# app.py — Signals Pro (v10 — stable universes, single confidence slider, batch fetch)
# One-page workflow: Setup ➜ Universe ➜ Scan ➜ Review ➜ Orders
# Uses yfinance + lightweight NLP for news scoring (optional)

import os, re, requests, feedparser, datetime as dt, math, time
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------- Page -----------------------------
st.set_page_config(page_title="Signals Pro — 1–3 Day Holds (v10)", layout="wide")
st.title("Signals Pro — 1–3 Day Holds (v10)")
st.caption("Stable universes • Single confidence slider • FX pivots • Risk-first • Cached loads")
st.divider()

# ----------------------------- Defaults -----------------------------
cfg = {
    "equity_gbp": 500.0,
    "risk_per_trade_pct": 1.0,
    "max_leverage": 3,
    "max_open_positions": 3,
    "daily_loss_stop_pct": 6.0,          # inform-only; you enforce it
    "timeframe_days": 300,
    "hold_days_max": 3,
    "allow_shorts": False,               # longs only by default
    "lookback_hours": 72,
    "assumed_spread_bps": 7,
    "min_stop_to_spread": 8,             # softer than before (was 10+)
    "min_adv_usd": 10_000_000,           # softer than before (was 20M)
    "price_min": 1.0,
    "price_max": 800.0,
    "confidence_floor": 0.50,
    "max_scan_symbols": 400,
    "max_scan_seconds": 50,
}

# ----------------------------- Session -----------------------------
for k, v in [("last_signals", pd.DataFrame()), ("journal", []),
             ("macro_dates", []), ("market_snap", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

UA = {"User-Agent": "SignalsPro/1.0"}
analyzer = SentimentIntensityAnalyzer()

# ----------------------------- Universes (fallbacks) -----------------------------
SP100 = [
    "AAPL","ABBV","ABT","ACN","ADBE","AMD","AMGN","AMT","AMZN","AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","CAT",
    "CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX","DE","DHR","DIS","DUK","EMR","EXC","F","FDX","GE","GILD","GM",
    "GOOG","GOOGL","GS","HD","HON","IBM","INTC","JNJ","JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ","MDT","META",
    "MMM","MO","MS","MSFT","NEE","NFLX","NKE","NOW","NVDA","ORCL","PEP","PFE","PM","PYPL","QCOM","RTX","SBUX","SCHW","SO",
    "SPGI","T","TGT","TMO","TMUS","TSLA","TXN","UNH","UNP","UPS","USB","V","VZ","WBA","WFC","WMT","XOM"
]
# Note: BA.L is correct for BAE Systems on Yahoo Finance.
FTSE100 = [
    "AZN.L","SHEL.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","DGE.L","BATS.L","BDEV.L","BARC.L","VOD.L","TSCO.L","LLOY.L",
    "RS1.L","IAG.L","WTB.L","CRH.L","REL.L","PRU.L","NG.L","AAL.L","SBRY.L","AUTO.L","SGE.L","JD.L","NXT.L","HLMA.L","SVT.L",
    "SMT.L","BRBY.L","FERG.L","HL.L","KGF.L","IMB.L","III.L","RR.L","RTO.L","EXPN.L","BKG.L","BA.L","CNA.L","AV.L","ITV.L",
    "PHNX.L","PSN.L","SSE.L","STAN.L"
]
DAX40 = ["ADS.DE","ALV.DE","BAS.DE","BAYN.DE","BMW.DE","CON.DE","DBK.DE","DTE.DE","DPW.DE","FRE.DE",
         "HEN3.DE","IFX.DE","LIN.DE","MRK.DE","MUV2.DE","PUM.DE","RWE.DE","SAP.DE","SIE.DE","VOW3.DE",
         "VNA.DE","ZAL.DE","DHER.DE","BEI.DE","MTX.DE","HNR1.DE","SY1.DE","1COV.DE","AIR.DE","ENR.DE",
         "HFG.DE","PAH3.DE","HLE.DE","DWNI.DE"]
CAC40 = ["AI.PA","AIR.PA","ALO.PA","BN.PA","BNP.PA","CAP.PA","CA.PA","CS.PA","DG.PA","EL.PA","ENGI.PA",
         "GLE.PA","HO.PA","KER.PA","LR.PA","MC.PA","OR.PA","PUB.PA","RI.PA","SAF.PA","SAN.PA","SGO.PA",
         "STLA.PA","SU.PA","SW.PA","TTE.PA","URW.AS","VIE.PA","VIV.PA","WLN.PA"]
# Clean encoding
FOREX_MAJORS = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X","EURGBP=X"]

# Robust local universes (no scraping)
DATA_DIR = Path(__file__).parent / "data"

def _load_universe_csv(name: str, fallback_list: List[str]) -> List[str]:
    p = DATA_DIR / f"{name}.csv"
    if p.exists():
        df = pd.read_csv(p, encoding="utf-8-sig")
        col = next((c for c in df.columns if str(c).lower() in ("ticker","symbol")), None)
        if col:
            return df[col].dropna().astype(str).str.strip().str.upper().tolist()
    return fallback_list

SP500 = _load_universe_csv("sp500", SP100)
NASDAQ100 = _load_universe_csv("nasdaq100", SP100)
FTSE350 = _load_universe_csv("ftse350", FTSE100)

def pick_universe(uni: str, custom: str, uploaded: Optional[List[str]]):
    if uni == "All (US+UK+EU)":
        return sorted(list(set(SP500 + NASDAQ100 + FTSE350 + DAX40 + CAC40)))
    if uni == "US (S&P500)":
        return SP500
    if uni == "US (Nasdaq-100)":
        return NASDAQ100 or SP100
    if uni == "UK (FTSE350)":
        return FTSE350
    if uni == "EU (DAX40 + CAC40)":
        return sorted(list(set(DAX40 + CAC40)))
    if uni == "Forex (Majors)":
        return FOREX_MAJORS
    if uni == "Custom (upload CSV)":
        return uploaded or []
    if uni == "Custom (enter tickers)":
        toks = [t.strip().upper() for t in (custom or "").split(",") if t.strip()]
        return toks
    return SP100

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Risk & Account")
    equity = st.number_input("Equity (£)", min_value=100.0, value=float(cfg["equity_gbp"]), step=50.0, format="%.2f")
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, float(cfg["risk_per_trade_pct"]), 0.25, key="risk_pct")
    max_lev = st.select_slider("Max leverage (cap)", options=[1, 2, 3], value=int(cfg["max_leverage"]))
    max_pos = st.select_slider("Max open positions", options=[1, 2, 3, 4, 5], value=int(cfg["max_open_positions"]))
    dls = st.slider("Daily loss stop (%) (info only)", 2.0, 10.0, float(cfg["daily_loss_stop_pct"]), 0.5)

    st.header("Universe")
    uni = st.selectbox(
        "Choose universe",
        ["All (US+UK+EU)", "US (S&P500)", "US (Nasdaq-100)", "UK (FTSE350)",
         "EU (DAX40 + CAC40)", "Forex (Majors)", "Custom (enter tickers)", "Custom (upload CSV)"],
        index=0,
    )

    uploaded_tickers = []
    if uni == "Custom (upload CSV)":
        up = st.file_uploader("Upload CSV with 'ticker' column", type=["csv"])
        if up:
            dfu = pd.read_csv(up)
            col = next((c for c in dfu.columns if str(c).lower() in ("ticker","symbol","epic","tidm")), dfu.columns[0])
            uploaded_tickers = dfu[col].dropna().astype(str).str.upper().tolist()
            st.caption(f"Loaded {len(uploaded_tickers)} tickers.")
    custom_tickers = st.text_area("Custom tickers (AAPL, AZN.L, EURUSDÂ=X → EURGBP=X)").strip()

    timeframe_days = st.slider("History window (days)", 150, 1000, int(cfg["timeframe_days"]), 25)

    st.header("Filters & Realism")
    min_adv = st.number_input("Min 30D $ADV (equities)", 1_000_000, 50_000_000, int(cfg["min_adv_usd"]), 1_000_000)
    px_min  = st.number_input("Min price", 0.5, 1000.0, float(cfg["price_min"]))
    px_max  = st.number_input("Max price", 5.0, 1500.0, float(cfg["price_max"]))
    assumed_spread_bps = st.number_input("Assumed spread+slip (bps)", 0, 100, int(cfg["assumed_spread_bps"]))
    min_stop_to_spread = st.number_input("Min stop ÷ spread (×)", 1, 50, int(cfg["min_stop_to_spread"]))
    allow_shorts = st.checkbox("Allow SHORT setups", value=bool(cfg["allow_shorts"]), key="allow_shorts")

    st.header("Signal Sensitivity")
    ema_proximity = st.slider("Pullback: distance to EMA20 (%)", 0.3, 2.5, 1.2, 0.1, key="ema_prox")
    atr_stop_mult = st.slider("ATR stop multiple", 1.0, 3.0, 1.7, 0.1, key="atr_mult")
    breakout_tol  = st.slider("Breakout tolerance (% below 20H)", 0.0, 0.3, 0.1, 0.05, key="breakout_tol")

    st.header("Run Controls")
    max_symbols = st.slider("Max symbols to scan", 50, 800, int(cfg["max_scan_symbols"]), 50, key="max_symbols")
    max_seconds = st.slider("Time cap (seconds)", 20, 90, int(cfg["max_scan_seconds"]), 5, key="max_seconds")

    # Confidence + News (keep both in the sidebar)
    st.slider("Min Confidence", 0.0, 1.0, float(cfg["confidence_floor"]), 0.05, key="min_conf")
    use_news = st.checkbox("Include news scoring (slower)", value=False, key="use_news")


# ----------------------------- FX & Liquidity helpers -----------------------------
CCY_MAP = {".L":"GBP",".DE":"EUR",".PA":"EUR",".AS":"EUR",".MI":"EUR",".BR":"EUR",".MC":"EUR",".SW":"CHF",".HK":"HKD",".TO":"CAD",".NE":"CAD"}
def is_fx(symbol: str) -> bool: return symbol.endswith("=X")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_hist(sym: str, days: int) -> pd.DataFrame:
    return yf.download(sym, period=f"{days}d", progress=False, auto_adjust=True)

@st.cache_data(ttl=900, show_spinner=False)
def _last_close(pair: str) -> Optional[float]:
    try:
        df = yf.download(pair, period="30d", interval="1d", progress=False, auto_adjust=True)
        s = pd.to_numeric(df.get("Close", pd.Series([], dtype=float)), errors="coerce").dropna()
        return float(s.iloc[-1]) if not s.empty else None
    except Exception:
        return None

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

def fx_rate(ccy_from: str, ccy_to: str) -> float:
    ccy_from, ccy_to = (ccy_from or "USD").upper(), (ccy_to or "GBP").upper()
    if ccy_from == ccy_to: return 1.0

    # direct
    d = _last_close(f"{ccy_from}{ccy_to}=X")
    if d and d > 0: return float(d)
    # inverse
    inv = _last_close(f"{ccy_to}{ccy_from}=X")
    if inv and inv > 0: return float(1.0 / inv)
    # USD pivot
    a = _last_close(f"{ccy_from}USD=X")
    b = _last_close(f"{ccy_to}USD=X")
    if a and b and a > 0 and b > 0: return float(a / b)
    # EUR pivot
    a = _last_close(f"{ccy_from}EUR=X")
    b = _last_close(f"{ccy_to}EUR=X")
    if a and b and a > 0 and b > 0: return float(a / b)

    return 1.0  # last resort

def get_ccy_from_suffix(ticker: str) -> str:
    # A simplified, non-network version of get_listing_currency
    for suf, ccy in CCY_MAP.items():
        if ticker.endswith(suf): return ccy
    return "USD"

def filter_by_liquidity_and_price(tickers, min_adv_usd, min_price, max_price):
    kept, rows = [], []
    fx_tickers = [t for t in tickers if is_fx(t)]
    equity_tickers = [t for t in tickers if not is_fx(t) and t]

    kept.extend(fx_tickers)

    if not equity_tickers:
        return kept, pd.DataFrame()

    data = yf.download(
        equity_tickers,
        period="45d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by='ticker',
        threads=True
    )
    
    if data.empty:
        return kept, pd.DataFrame()

    # Pre-fetch all relevant FX rates to avoid calling in a loop
    all_currencies = {get_ccy_from_suffix(t) for t in equity_tickers}
    fx_rates = {ccy: fx_rate(ccy, "USD") for ccy in all_currencies}

    for t in equity_tickers:
        try:
            ticker_data = data[t]
            if ticker_data.empty or ticker_data['Close'].isnull().all():
                continue

            px = ticker_data['Close'].dropna().iloc[-1]
            
            if not (min_price <= px <= max_price):
                continue
            
            tail = ticker_data.tail(30)
            adv_ccy = (tail['Close'] * tail['Volume']).mean()
            
            if not np.isfinite(adv_ccy):
                continue

            ccy = get_ccy_from_suffix(t)
            rate_to_usd = fx_rates.get(ccy, 1.0)
            adv_usd = adv_ccy * rate_to_usd

            if adv_usd >= min_adv_usd:
                kept.append(t)
                rows.append({"Ticker": t, "Price": round(px, 4), "CCY": ccy, "ADV_USD_30D": round(adv_usd, 2)})

        except (KeyError, IndexError):
            continue
        except Exception:
            continue
            
    return kept, pd.DataFrame(rows)

def get_ccy_from_suffix(ticker: str) -> str:
    # A simplified, non-network version of get_listing_currency
    for suf, ccy in CCY_MAP.items():
        if ticker.endswith(suf): return ccy
    return "USD"

def filter_by_liquidity_and_price(tickers, min_adv_usd, min_price, max_price):
    kept, rows = [], []
    fx_tickers = [t for t in tickers if is_fx(t)]
    equity_tickers = [t for t in tickers if not is_fx(t) and t]

    kept.extend(fx_tickers)

    if not equity_tickers:
        return kept, pd.DataFrame()

    data = yf.download(
        equity_tickers,
        period="45d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by='ticker',
        threads=True
    )
    
    if data.empty:
        return kept, pd.DataFrame()

    # Pre-fetch all relevant FX rates to avoid calling in a loop
    all_currencies = {get_ccy_from_suffix(t) for t in equity_tickers}
    fx_rates = {ccy: fx_rate(ccy, "USD") for ccy in all_currencies}

    for t in equity_tickers:
        try:
            ticker_data = data[t]
            if ticker_data.empty or ticker_data['Close'].isnull().all():
                continue

            px = ticker_data['Close'].dropna().iloc[-1]
            
            if not (min_price <= px <= max_price):
                continue
            
            tail = ticker_data.tail(30)
            adv_ccy = (tail['Close'] * tail['Volume']).mean()
            
            if not np.isfinite(adv_ccy):
                continue

            ccy = get_ccy_from_suffix(t)
            rate_to_usd = fx_rates.get(ccy, 1.0)
            adv_usd = adv_ccy * rate_to_usd

            if adv_usd >= min_adv_usd:
                kept.append(t)
                rows.append({"Ticker": t, "Price": round(px, 4), "CCY": ccy, "ADV_USD_30D": round(adv_usd, 2)})

        except (KeyError, IndexError):
            continue
        except Exception:
            continue
            
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

# ----------------------------- Indicators & Setups (looser) -----------------------------
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

def scan_signals(df: pd.DataFrame, allow_shorts: bool, ema_pct=1.2, atr_mult=1.7, breakout_pct=0.1):
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
    last = df.iloc[-1]; date = df.index[-1]

    # Momentum Breakout (tolerance below 20H)
    tol = (1.0 - breakout_pct/100.0)
    if last["Close"] >= float(last["20H"]) * tol and pd.notna(last["ATR14"]) and last["ATR14"]>0:
        entry = float(last["Close"])
        stop  = float(entry - atr_mult * float(last["ATR14"]))
        take  = float(entry + 2.0 * (entry - stop))
        if stop < entry:
            out.append({"strategy":"Breakout Long","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":False})

    # Pullback in Uptrend (near EMA20)
    if (last["SMA50"] > last["SMA200"]) and (abs(last["Close"] - last["EMA20"]) / max(1e-9, last["Close"]) <= ema_pct/100.0) and pd.notna(last["ATR14"]) and last["ATR14"]>0:
        entry = float(last["Close"])
        stop  = float(min(last["Low"], entry - atr_mult * float(last["ATR14"])))
        risk  = entry - stop
        if risk > 0:
            take  = float(entry + 2.0 * risk)
            out.append({"strategy":"Pullback Long","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":False})

    # RSI bounce above 200SMA
    if (last["Close"] > last["SMA200"]) and (rsi(df["Close"],14).iloc[-1] < 35) and pd.notna(last["ATR14"]) and last["ATR14"]>0:
        entry = float(last["Close"])
        stop  = float(min(last["Low"], entry - atr_mult * float(last["ATR14"])))
        risk  = max(entry - stop, 1e-6)
        take  = float(entry + 1.8 * risk)
        out.append({"strategy":"MeanRev Long","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":1.8,"is_short":False})

    # Optional shorts
    if allow_shorts:
        if last["Close"] <= last["20L"] / tol and pd.notna(last["ATR14"]) and last["ATR14"]>0:
            entry = float(last["Close"]); stop = float(entry + atr_mult * float(last["ATR14"]))
            take = float(entry - 2.0 * (stop - entry))
            out.append({"strategy":"Breakdown Short","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":True})
    return out

def scan_signals_fx(df: pd.DataFrame, allow_shorts: bool, atr_mult=1.7):
    out = []
    if df is None or df.empty or len(df) < 200: return out
    df = df.copy()
    df["20H"] = df["High"].rolling(20).max(); df["20L"] = df["Low"].rolling(20).min()
    df["EMA20"] = ema(df["Close"], 20); df["EMA50"] = ema(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200); df["RSI14"] = rsi(df["Close"], 14); df["ATR14"] = atr(df, 14)
    last = df.iloc[-1]; date = df.index[-1]
    if last["Close"] >= last["20H"] * 0.999 and pd.notna(last["ATR14"]) and last["ATR14"]>0:
        entry = float(last["Close"]); stop  = float(entry - atr_mult * float(last["ATR14"]))
        take  = float(entry + 2.0 * (entry - stop))
        out.append({"strategy":"FX Breakout Long","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":False})
    if (last["EMA20"] > last["EMA50"]) and pd.notna(last["ATR14"]) and last["ATR14"]>0:
        entry = float(last["Close"]); stop = float(entry - atr_mult * float(last["ATR14"]))
        take  = float(entry + 2.0 * (entry - stop))
        out.append({"strategy":"FX Pullback Long","date":date,"entry":entry,"stop":stop,"take_profit":take,"r_multiple":2.0,"is_short":False})
    return out

# ----------------------------- News (optional) -----------------------------
def _ts_now_utc(): return dt.datetime.utcnow()
def _to_iso_z(ts: dt.datetime) -> str:
    if ts.tzinfo is not None: ts = ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return ts.isoformat(timespec="seconds") + "Z"
def _domain(url: str) -> str:
    try: from urllib.parse import urlparse; return urlparse(url).netloc.lower()
    except Exception: return ""

RUMOUR_TERMS = re.compile(r"(rumou?r|speculat(e|ion)|report(ed|edly)|'?sources?\s+say'?|leak(ed|s?)|unconfirm|whisper|chatter|talks?\s+that)", re.I)

def is_rumour(title: str, summary: str) -> bool:
    return bool(RUMOUR_TERMS.search((title or "") + " " + (summary or "")))

def source_weight(name: str) -> float:
    SW = {"Bloomberg":1.0,"Reuters":1.0,"Financial Times":1.0,"The Wall Street Journal":1.0,"CNBC":0.8,"MarketWatch":0.8,"Yahoo Finance":0.7,"Seeking Alpha":0.6}
    if not name: return 0.5
    for k,w in SW.items():
        if k.lower() in name.lower(): return w
    return 0.5

def news_yahoo(symbol: str, lookback_hours: int = 72) -> List[Dict[str, Any]]:
    try:
        t = yf.Ticker(symbol); items = getattr(t, "news", []) or []; out = []
        cut_ts = _ts_now_utc() - dt.timedelta(hours=lookback_hours)
        for a in items[:30]:
            ts_epoch = float(a.get("providerPublishTime", 0) or 0)
            if ts_epoch <= 0: continue
            ts = dt.datetime.utcfromtimestamp(ts_epoch)
            if ts < cut_ts: continue
            out.append({"source": a.get("publisher",""), "title": a.get("title",""),
                        "summary": a.get("summary",""), "url": a.get("link",""),
                        "published": _to_iso_z(ts)})
        return out
    except Exception:
        return []

def aggregate_news(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not articles:
        return {"count":0,"score":0.0,"rumour_ratio":0.0}
    weighted_sum=0.0; weight_total=0.0; rumours=0
    for a in articles:
        title=a.get("title","") or ""; summary=a.get("summary","") or ""
        vs = analyzer.polarity_scores((title+". "+summary).strip()); s = float(vs.get("compound",0.0))
        w = source_weight(a.get("source",""))
        rumours += int(is_rumour(title, summary))
        weighted_sum += s*w; weight_total += w
    mean_sent = (weighted_sum/weight_total) if weight_total>0 else 0.0
    rumour_ratio = rumours/max(1,len(articles))
    score = max(-1.0, min(1.0, mean_sent * (1.0 - 0.5*rumour_ratio)))
    return {"count":len(articles),"score":float(score),"rumour_ratio":float(rumour_ratio)}

def fuse(signal: Dict[str, Any], news: Dict[str, Any]) -> Dict[str, Any]:
    score = float(news.get("score",0.0)); cnt = int(news.get("count",0)); rr = float(news.get("rumour_ratio",0.0))
    is_short = bool(signal.get("IsShort", False))
    base = 0.60
    aligned = max(0.0, -score if is_short else score) * 0.30
    opposed = max(0.0,  score if is_short else -score) * 0.35
    conf = max(0.0, min(1.0, base + aligned - opposed - 0.15*rr + min(cnt,8)*0.01))
    return {"Confidence": round(conf,3), "NewsScore": round(score,3), "NewsCount": cnt, "RumourRatio": round(rr,3)}

# ----------------------------- Lazy Regime Snapshot -----------------------------
def pct_above_sma50(tickers, days=200):
    ok = tot = 0
    for t in tickers[:30]:
        try:
            df = fetch_hist(t, days)
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
        v = fetch_hist("^VIX", days)
        if v is None or v.empty: return np.nan, np.nan, False
        sma50 = v["Close"].rolling(50).mean().iloc[-1]
        close = v["Close"].iloc[-1]
        return float(close), float(sma50), (close > sma50)
    except Exception:
        return np.nan, np.nan, False

def refresh_market_snapshot():
    breadth = pct_above_sma50(SP500)
    vix_c, vix_s50, vix_up = vix_trend()
    st.session_state["market_snap"] = {"breadth": breadth, "vix_c": vix_c, "vix_s50": vix_s50, "vix_up": bool(vix_up)}

c0, c1 = st.columns([1,3])
with c0:
    if st.button("Refresh market snapshot"):
        with st.spinner("Loading snapshot…"): refresh_market_snapshot()
snap = st.session_state["market_snap"] or {}
breadth = snap.get("breadth", np.nan); vix_c = snap.get("vix_c", np.nan); vix_s50 = snap.get("vix_s50", np.nan); vix_up = snap.get("vix_up", False)
regime = "Risk-On" if (np.isfinite(breadth) and breadth >= 55 and not vix_up) else ("Risk-Off" if (np.isfinite(breadth) and breadth <= 45 and vix_up) else "Neutral")
colA, colB, colC, colD = st.columns(4)
with colA: st.metric("Regime", regime)
with colB: st.metric("% S&P500 > 50SMA", f"{breadth:.1f}%" if np.isfinite(breadth) else "n/a")
with colC: st.metric("VIX / 50SMA", f"{vix_c:.1f} / {vix_s50:.1f}" if (np.isfinite(vix_c) and np.isfinite(vix_s50)) else "n/a")
with colD: st.metric("Risk Budget (worst-case)", f"{risk_pct * max_pos:.1f}%")
st.divider()

# ----------------------------- Scanner -----------------------------
tabs = st.tabs(["🔎 Scanner", "📈 Chart", "📒 Journal", "📈 Planner", "ℹ️ Help"])

with tabs[0]:
    st.subheader("Signal Scanner")
    tickers = pick_universe(uni, custom_tickers, uploaded_tickers)
    if len(tickers) > max_symbols:
        st.info(f"Capped to first {max_symbols} symbols for speed.")
        tickers = tickers[:max_symbols]

    # Liquidity gate
    tickers, liq_table = filter_by_liquidity_and_price(tickers, min_adv, px_min, px_max)
    with st.expander("Liquidity snapshot"):
        if not liq_table.empty: st.dataframe(liq_table, use_container_width=True)

    run = st.button(f"Run scan ({len(tickers)} symbols)")

    # Display filters (single confidence source)
    st.caption("Display filters")
    dc1, dc2, dc3 = st.columns(3)
    with dc1: show_longs = st.checkbox("Longs", value=True, key="show_longs")
    with dc2: show_shorts = st.checkbox("Shorts", value=allow_shorts, key="show_shorts")
    with dc3: spread_gate = st.checkbox("SpreadOK only", value=True, key="spread_ok_only")
    conf_floor = float(st.session_state.get("min_conf", cfg["confidence_floor"]))
    st.metric("Min Confidence (from sidebar)", f"{conf_floor:.2f}")

    if run:
        t0 = time.perf_counter()
        rows = []
        errors: List[tuple[str, str]] = []
        total = len(tickers)
        prog = st.progress(0)

        # ---------- Batch fetch once ----------
        try:
            hist = yf.download(
                tickers,
                period=f"{timeframe_days}d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            st.error(f"Batch download failed: {e}")
            hist = pd.DataFrame()

        for i, t in enumerate(tickers, 1):
            try:
                # Extract per-ticker frame from batch
                if isinstance(hist.columns, pd.MultiIndex):
                    if t not in hist.columns.levels[0] and t not in hist.columns.get_level_values(0):
                        raise ValueError("No data in batch result")
                    df = hist[t].dropna()
                else:
                    # Single symbol case
                    df = hist.dropna()

                if df is None or df.empty or len(df) < 200:
                    prog.progress(min(i/total, 1.0)); continue

                sigs = scan_signals_fx(df, allow_shorts, atr_mult=atr_stop_mult) if is_fx(t) else \
                       scan_signals(df, allow_shorts, ema_pct=ema_proximity, atr_mult=atr_stop_mult, breakout_pct=breakout_tol)

                if not sigs:
                    prog.progress(min(i/total, 1.0))
                    if time.perf_counter() - t0 > max_seconds: break
                    continue

                news = {"score":0.0,"count":0,"rumour_ratio":0.0}
                if st.session_state.get("use_news") and not is_fx(t):
                    arts = news_yahoo(t, lookback_hours=cfg["lookback_hours"])
                    news = aggregate_news(arts)

                for s in sigs:
                    entry, stop, tp = float(s["entry"]), float(s["stop"]), float(s["take_profit"])
                    qty, risk_cash, notional = position_size(entry, stop, equity, risk_pct, max_lev, ticker=t)
                    if qty < 1: continue
                    base = {"Ticker":t,"Strategy":s["strategy"],"Date":pd.to_datetime(s["date"]).date().isoformat(),
                            "Entry":round(entry,4),"Stop":round(stop,4),"TakeProfit":round(tp,4),
                            "Qty":int(qty),"Risk_£":round(risk_cash,2),"Notional_£":round(notional,2),
                            "R_multiple":s.get("r_multiple",2.0),"IsShort":bool(s.get("is_short",False))}
                    fused = fuse(base, news)
                    local_mult = 4 if is_fx(t) else int(min_stop_to_spread)
                    ok_spread, spread_px, stop_dist = spread_ok(entry, stop, int(assumed_spread_bps), local_mult)
                    rows.append({**base, **fused,
                                 "AssumedSpreadPx":round(spread_px,4),"StopDist":round(stop_dist,4),
                                 "SpreadOK":bool(ok_spread)})
            except Exception as e:
                errors.append((t, str(e)))

            prog.progress(min(i/total, 1.0))
            if time.perf_counter() - t0 > max_seconds:
                st.info("Stopped early (time cap). Increase 'Max symbols' or time cap to scan more.")
                break

        if errors:
            with st.expander(f"Warnings for {len(errors)} symbols"):
                st.dataframe(pd.DataFrame(errors, columns=["Ticker","Error"]), use_container_width=True)

        if not rows:
            st.warning("No signals passed sizing/spread filters. Try lowering confidence, loosening EMA/ATR/breakout tolerances, or ADV/price gates.")
        else:
            df = pd.DataFrame(rows)
            mask = pd.Series(True, index=df.index)
            if not show_longs:  mask &= df["IsShort"] == True
            if not show_shorts: mask &= df["IsShort"] == False
            if spread_gate:     mask &= df["SpreadOK"] == True
            mask &= df["Confidence"] >= conf_floor
            view = df[mask].sort_values(["Confidence","Ticker"], ascending=[False, True]).reset_index(drop=True)
            st.session_state["last_signals"] = view.copy()

            st.subheader("Trade ideas")
            if view.empty:
                st.info("Signals found, but none met your display filters. Try lowering Confidence filter or Spread multiple.")
            else:
                st.dataframe(view[["Ticker","Strategy","Entry","Stop","TakeProfit","Qty","Notional_£","Confidence","AssumedSpreadPx"]],
                             use_container_width=True)
                st.markdown("### Orders to queue (top 3)")
                orders = []
                for _, r in view.head(3).iterrows():
                    side = "SELL" if r["IsShort"] else "BUY"
                    orders.append(f"{r['Ticker']} {side} {int(r['Qty'])} @ Limit {r['Entry']}; TP {r['TakeProfit']}; SL {r['Stop']}")
                if orders: st.code("\n".join([f"{i+1}) {o}" for i,o in enumerate(orders)]))

# ----------------------------- Chart -----------------------------
with tabs[1]:
    st.subheader("Chart")
    last = st.session_state.get("last_signals", pd.DataFrame())
    default_symbol = last["Ticker"].tolist()[0] if not last.empty else "AAPL"
    symbol = st.text_input("Symbol", default_symbol).strip().upper()
    hist_days = st.slider("Chart history (days)", 150, 1000, int(cfg["timeframe_days"]), 25)
    if st.button("Show chart"):
        try:
            df = fetch_hist(symbol, hist_days)
            if df is None or df.empty:
                st.warning("No data.")
            else:
                df["SMA200"] = df["Close"].rolling(200).mean()
                df["SMA50"]  = df["Close"].rolling(50).mean()
                df["EMA20"]  = df["Close"].ewm(span=20, adjust=False).mean()
                entry = stop = tp = None
                if not last.empty and symbol in last["Ticker"].values:
                    rec = last[last["Ticker"]==symbol].iloc[0]
                    entry, stop, tp = rec["Entry"], rec["Stop"], rec["TakeProfit"]
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(df.index, df["Close"], label="Close")
                ax.plot(df.index, df["SMA200"], label="SMA200")
                ax.plot(df.index, df["SMA50"],  label="SMA50")
                ax.plot(df.index, df["EMA20"],  label="EMA20")
                if entry: ax.axhline(entry, linestyle="--", label="Entry")
                if stop:  ax.axhline(stop,  linestyle="--", label="Stop")
                if tp:    ax.axhline(tp,    linestyle="--", label="TP")
                ax.legend(); ax.set_title(symbol)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Chart error: {e}")

# ----------------------------- Journal -----------------------------
with tabs[2]:
    st.subheader("Journal")
    with st.form("add_trade"):
        c1,c2,c3,c4 = st.columns(4)
        with c1: date = st.date_input("Date", value=dt.date.today())
        with c2: ticker = st.text_input("Ticker", value="AAPL").upper().strip()
        with c3: direction = st.selectbox("Direction", ["Long","Short"])
        with c4: qty = st.number_input("Qty", min_value=1, value=1, step=1)
        c5,c6,c7,c8 = st.columns(4)
        with c5: entry = st.number_input("Entry", min_value=0.0, value=0.0, format="%.4f")
        with c6: stop  = st.number_input("Stop",  min_value=0.0, value=0.0, format="%.4f")
        with c7: exitp = st.number_input("Exit",  min_value=0.0, value=0.0, format="%.4f")
        with c8: fees  = st.number_input("Fees (£)", min_value=0.0, value=0.0, format="%.2f")
        note = st.text_input("Notes", value="")
        submitted = st.form_submit_button("Add")
        if submitted:
            if direction == "Long":
                pl = (exitp - entry) * qty; r_denom = max(entry - stop, 1e-9); r_real = (exitp - entry)/r_denom
            else:
                pl = (entry - exitp) * qty; r_denom = max(stop - entry, 1e-9); r_real = (entry - exitp)/r_denom
            net = pl - fees
            st.session_state["journal"].append({"Date":date.isoformat(),"Ticker":ticker,"Direction":direction,"Qty":int(qty),
                                                "Entry":float(entry),"Stop":float(stop),"Exit":float(exitp),
                                                "Fees_£":float(fees),"P_L_£":float(pl),"Net_£":float(net),"R":float(r_real),"Notes":note})
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

        # Equity curve
        dfj_sorted = dfj.sort_values("Date")
        dfj_sorted["CumNet_£"] = dfj_sorted["Net_£"].cumsum()
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(pd.to_datetime(dfj_sorted["Date"]), dfj_sorted["CumNet_£"])
        ax.set_title("Equity Curve (Net £)")
        ax.set_ylabel("£"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("No trades logged yet.")

# ----------------------------- Planner -----------------------------
with tabs[3]:
    st.subheader("Planner — Compounding")
    c1,c2,c3 = st.columns(3)
    with c1: start_eq = st.number_input("Starting equity (£)", min_value=100.0, value=float(equity), step=50.0)
    with c2: monthly_contrib = st.number_input("Monthly contribution (£)", min_value=0.0, value=0.0, step=50.0)
    with c3: months = st.slider("Months", 3, 24, 12)
    c4,c5,c6,c7 = st.columns(4)
    with c4: risk_pct_pl = st.number_input("Risk per trade (%)", min_value=0.25, value=float(risk_pct), step=0.25)
    with c5: trades_per_day = st.slider("Trades/day", 1, 4, 2)
    with c6: days_per_week = st.slider("Days/week", 1, 5, 4)
    with c7: exp_R = st.slider("Expectancy (R)", -0.30, 0.50, 0.15, 0.05)
    monthly_trades = trades_per_day * days_per_week * 4.33
    E = float(start_eq); rows = []
    for m in range(1, months+1):
        risk_cash = E * (risk_pct_pl/100.0)
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

# ----------------------------- Help -----------------------------
with tabs[4]:
    st.subheader("Why you weren’t seeing many trades")
    st.markdown("""
- **Stable universes**: Reads S&P500, Nasdaq-100, FTSE-350 from local CSVs (no fragile scraping).
- **Softer prerequisites**: Looser breakout tolerance, softer EMA proximity, ATR stops default 1.7×.
- **Liquidity/price gate**: Default ADV cut **$10M** and wider price band.
- **Confidence/Spread filters**: Single slider in sidebar drives filtering everywhere.
    """)
    st.subheader("Risk-first reminders")
    st.markdown("""
- Keep risk per trade small (1% default) and cap leverage at **3×** (UK retail CFD limit).
- Respect your daily loss stop and maximum concurrent positions.
- Orders shown here are **queued** instructions—always place/modify with your broker.
    """)
    st.caption("Educational use only. Not financial advice.")
