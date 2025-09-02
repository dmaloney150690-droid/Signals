# app.py — Signals Pro (v11 — Final Performance Fix)
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
st.set_page_config(page_title="Signals Pro — 1–3 Day Holds (v11)", layout="wide")
st.title("Signals Pro — 1–3 Day Holds (v11)")
st.caption("Stable universes • Batch data loading • FX pivots • Risk-first")
st.divider()

# ----------------------------- Defaults -----------------------------
cfg = {
    "equity_gbp": 2000.0,
    "risk_per_trade_pct": 1.0,
    "max_leverage": 3,
    "max_open_positions": 3,
    "daily_loss_stop_pct": 6.0,
    "timeframe_days": 300,
    "hold_days_max": 3,
    "allow_shorts": False,
    "lookback_hours": 72,
    "assumed_spread_bps": 7,
    "min_stop_to_spread": 8,
    "min_adv_usd": 10_000_000,
    "price_min": 1.0,
    "price_max": 800.0,
    "confidence_floor": 0.50,
    "max_scan_symbols": 400,
    "max_scan_seconds": 60,
}

# ----------------------------- Session -----------------------------
for k, v in [("last_signals", pd.DataFrame()), ("journal", []), ("market_snap", None), ("scan_errors", [])]:
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
FTSE100 = [
    "AZN.L","SHEL.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","DGE.L","BATS.L","BDEV.L","BARC.L","VOD.L","TSCO.L","LLOY.L",
    "RS1.L","IAG.L","WTB.L","CRH.L","REL.L","PRU.L","NG.L","AAL.L","SBRY.L","AUTO.L","SGE.L","JD.L","NXT.L","HLMA.L","SVT.L",
    "SMT.L","BRBY.L","FERG.L","HL.L","KGF.L","IMB.L","III.L","RR.L","RTO.L","EXPN.L","BKG.L","BA.L","CNA.L","AV.L","ITV.L",
    "PHNX.L","PSN.L","SSE.L","STAN.L"
]
DAX40 = ["ADS.DE","ALV.DE","BAS.DE","BAYN.DE","BMW.DE","CON.DE","DBK.DE","DTE.DE","DPW.DE","FRE.DE", "HEN3.DE","IFX.DE","LIN.DE","MRK.DE","MUV2.DE","PUM.DE","RWE.DE","SAP.DE","SIE.DE","VOW3.DE", "VNA.DE","ZAL.DE","DHER.DE","BEI.DE","MTX.DE","HNR1.DE","SY1.DE","1COV.DE","AIR.DE","ENR.DE", "HFG.DE","PAH3.DE","HLE.DE","DWNI.DE"]
CAC40 = ["AI.PA","AIR.PA","ALO.PA","BN.PA","BNP.PA","CAP.PA","CA.PA","CS.PA","DG.PA","EL.PA","ENGI.PA", "GLE.PA","HO.PA","KER.PA","LR.PA","MC.PA","OR.PA","PUB.PA","RI.PA","SAF.PA","SAN.PA","SGO.PA", "STLA.PA","SU.PA","SW.PA","TTE.PA","URW.AS","VIE.PA","VIV.PA","WLN.PA"]
FOREX_MAJORS = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X","EURGBP=X"]

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@lru_cache(maxsize=1)
def _load_universe_csv(name: str, fallback_list: List[str]) -> List[str]:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        return fallback_list
    try:
        df = pd.read_csv(p)
        col = next((c for c in df.columns if str(c).lower() in ("ticker", "symbol")), df.columns[0])
        tickers = df[col].dropna().astype(str).str.strip().str.upper().tolist()
        if name == "ftse350": # Add .L suffix for LSE stocks
            return [t if '.' in t else f"{t}.L" for t in tickers]
        return tickers
    except Exception:
        return fallback_list

SP500 = _load_universe_csv("sp500_tickers", SP100)
NASDAQ100 = _load_universe_csv("nasdaq100_tickers", SP100)
FTSE350 = _load_universe_csv("ftse350_tickers", FTSE100)

def pick_universe(uni: str, custom: str, uploaded: Optional[List[str]]):
    if uni == "All (US+UK+EU)": return sorted(list(set(SP500 + NASDAQ100 + FTSE350 + DAX40 + CAC40)))
    if uni == "US (S&P500)": return SP500
    if uni == "US (Nasdaq-100)": return NASDAQ100 or SP100
    if uni == "UK (FTSE350)": return FTSE350
    if uni == "EU (DAX40 + CAC40)": return sorted(list(set(DAX40 + CAC40)))
    if uni == "Forex (Majors)": return FOREX_MAJORS
    if uni == "Custom (upload CSV)": return uploaded or []
    if uni == "Custom (enter tickers)":
        return [t.strip().upper() for t in (custom or "").split(",") if t.strip()]
    return SP100

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Risk & Account")
    equity = st.number_input("Equity (£)", min_value=100.0, value=float(cfg["equity_gbp"]), step=50.0)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, float(cfg["risk_per_trade_pct"]), 0.25)
    max_lev = st.select_slider("Max leverage (cap)", options=[1, 2, 3], value=int(cfg["max_leverage"]))
    max_pos = st.select_slider("Max open positions", options=[1, 2, 3, 4, 5], value=int(cfg["max_open_positions"]))

    st.header("Universe")
    uni = st.selectbox("Choose universe", ["All (US+UK+EU)", "US (S&P500)", "US (Nasdaq-100)", "UK (FTSE350)", "EU (DAX40 + CAC40)", "Forex (Majors)", "Custom (enter tickers)", "Custom (upload CSV)"])
    custom_tickers = st.text_area("Custom tickers (AAPL, AZN.L, EURUSD=X)").strip()
    
    # ... Rest of sidebar remains the same ...
    timeframe_days = st.slider("History window (days)", 150, 1000, int(cfg["timeframe_days"]), 25)

    st.header("Filters & Realism")
    min_adv = st.number_input("Min 30D $ADV (equities)", 1_000_000, 50_000_000, int(cfg["min_adv_usd"]), 1_000_000)
    px_min  = st.number_input("Min price", 0.5, 1000.0, float(cfg["price_min"]))
    px_max  = st.number_input("Max price", 5.0, 1500.0, float(cfg["price_max"]))
    assumed_spread_bps = st.number_input("Assumed spread+slip (bps)", 0, 100, int(cfg["assumed_spread_bps"]))
    min_stop_to_spread = st.number_input("Min stop ÷ spread (×)", 1, 50, int(cfg["min_stop_to_spread"]))
    allow_shorts = st.checkbox("Allow SHORT setups", value=bool(cfg["allow_shorts"]))

    st.header("Signal Sensitivity")
    ema_proximity = st.slider("Pullback: distance to EMA20 (%)", 0.3, 2.5, 1.2, 0.1)
    atr_stop_mult = st.slider("ATR stop multiple", 1.0, 3.0, 1.7, 0.1)
    breakout_tol  = st.slider("Breakout tolerance (% below 20H)", 0.0, 0.3, 0.1, 0.05)

    st.header("Run Controls")
    max_symbols = st.slider("Max symbols to scan", 50, 800, int(cfg["max_scan_symbols"]), 50)
    max_seconds = st.slider("Time cap (seconds)", 20, 90, int(cfg["max_scan_seconds"]), 5)
    
    confidence_floor = st.slider("Min Confidence", 0.0, 1.0, float(cfg["confidence_floor"]), 0.05)
    use_news = st.checkbox("Include news scoring (slower)", value=False)


# ----------------------------- FX & Liquidity helpers (same as your v10) -----------------------------
# ... This entire section of helper functions (is_fx, _last_close, fx_rate, etc.) is correct
# ... No changes needed here. For brevity, it is omitted from this view but should be in your file.
# ----------------------------- Indicators & Setups (same as your v10) --------------------------------
# ... This entire section of indicator functions (rsi, atr, scan_signals, etc.) is correct
# ... No changes needed here. For brevity, it is omitted.
# ----------------------------- News (same as your v10) ---------------------------------------------
# ... This entire section of news functions is correct
# ... No changes needed here. For brevity, it is omitted.
# ----------------------------- Market Snapshot (same as your v10) ----------------------------------
# ... This entire section for the snapshot is correct
# ... No changes needed here. For brevity, it is omitted.


# ----------------------------- Scanner -----------------------------
tabs = st.tabs(["🔎 Scanner", "📈 Chart", "📒 Journal", "📈 Planner", "ℹ️ Help"])

with tabs[0]:
    st.subheader("Signal Scanner")
    st.session_state["scan_errors"] = [] # Clear previous errors
    tickers = pick_universe(uni, custom_tickers, []) # Uploaded CSV handled in sidebar
    
    if len(tickers) > max_symbols:
        st.info(f"Capped to first {max_symbols} symbols for speed.")
        tickers = tickers[:max_symbols]

    # Liquidity gate
    with st.spinner("Filtering by liquidity..."):
        tickers, liq_table = filter_by_liquidity_and_price(tickers, min_adv, px_min, px_max)
    with st.expander("Liquidity Snapshot", expanded=False):
        if not liq_table.empty: st.dataframe(liq_table, use_container_width=True)

    run = st.button(f"Run scan ({len(tickers)} symbols)")

    # Display filters
    st.caption("Display filters")
    dc1, dc2, dc3 = st.columns(3)
    with dc1: show_longs = st.checkbox("Longs", value=True)
    with dc2: show_shorts = st.checkbox("Shorts", value=allow_shorts)
    with dc3: spread_gate = st.checkbox("SpreadOK only", value=True)

    if run:
        t0 = time.perf_counter()
        rows = []
        
        # BATCH DOWNLOAD: Fetch all data at once for maximum speed.
        with st.spinner(f"Downloading historical data for {len(tickers)} symbols..."):
            hist = yf.download(
                tickers,
                period=f"{timeframe_days}d",
                progress=False,
                auto_adjust=True,
                group_by='ticker',
                threads=True
            )

        prog_bar = st.progress(0, text="Scanning for signals...")
        for i, t in enumerate(tickers, 1):
            try:
                # ACCESS DATA: Get the data for the current ticker from the batch download.
                df = hist[t] if isinstance(hist.columns, pd.MultiIndex) else hist
                df.dropna(inplace=True)

                if df.empty or len(df) < 200:
                    continue

                sigs = scan_signals_fx(df, allow_shorts, atr_mult=atr_stop_mult) if is_fx(t) else \
                       scan_signals(df, allow_shorts, ema_pct=ema_proximity, atr_mult=atr_stop_mult, breakout_pct=breakout_tol)

                if not sigs:
                    continue

                news = {"score":0.0,"count":0,"rumour_ratio":0.0}
                if use_news and not is_fx(t):
                    arts = news_yahoo(t, lookback_hours=cfg["lookback_hours"])
                    news = aggregate_news(arts)

                for s in sigs:
                    entry, stop = float(s["entry"]), float(s["stop"])
                    qty, risk_cash, notional = position_size(entry, stop, equity, risk_pct, max_lev, ticker=t)
                    if qty < 1: continue
                    
                    ok_spread, spread_px, stop_dist = spread_ok(entry, stop, int(assumed_spread_bps), int(min_stop_to_spread))
                    
                    base = {"Ticker":t, "Strategy":s["strategy"], "Entry":entry, "Stop":stop, "Qty":qty, "Notional_£":notional}
                    fused = fuse(base, news) # Assumes fuse function exists
                    
                    rows.append({**base, **fused, "SpreadOK": ok_spread})

            except Exception as e:
                # ERROR LOGGING: Keep track of which tickers failed.
                st.session_state["scan_errors"].append(f"{t}: {e}")
            
            prog_bar.progress(i / len(tickers), text=f"Scanning: {t}")
            if time.perf_counter() - t0 > max_seconds:
                st.info("Stopped early (time cap).")
                break

        prog_bar.empty()

        if st.session_state["scan_errors"]:
            with st.expander("Show Scan Errors"):
                st.warning("\n".join(st.session_state["scan_errors"]))

        if not rows:
            st.warning("No signals found. Try adjusting filters in the sidebar.")
        else:
            df = pd.DataFrame(rows)
            mask = (df["Confidence"] >= confidence_floor)
            if not show_longs:  mask &= df["IsShort"]
            if not show_shorts: mask &= ~df["IsShort"]
            if spread_gate:     mask &= df["SpreadOK"]
            
            view = df[mask].sort_values("Confidence", ascending=False).reset_index(drop=True)
            st.session_state["last_signals"] = view.copy()
            
            st.subheader("Trade Ideas")
            st.dataframe(view[["Ticker", "Strategy", "Entry", "Stop", "Qty", "Notional_£", "Confidence"]], use_container_width=True)
            
            # ... Order generation logic remains the same ...


# ----------------------------- Chart, Journal, Planner, Help Tabs -----------------------------
# ... The rest of your code for the other tabs can remain exactly the same.
# ... For brevity, it is omitted here.
