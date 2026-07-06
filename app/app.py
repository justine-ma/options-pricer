import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import sys
import time
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('..')
warnings.filterwarnings('ignore')

from src.black_scholes import bsm_price, put_call_parity_check
from src.greeks import delta, gamma, theta, vega, rho
from src.binomial_tree import binomial_tree
from src.monte_carlo import monte_carlo, monte_carlo_antithetic
from src.implied_vol import implied_vol, implied_vol_newton, implied_vol_bisection

# --- page config ---
st.set_page_config(
    page_title="Options Pricer",
    page_icon="📈",
    layout="wide"
)

st.title("Options pricing dashboard")
st.markdown("Interactive tool covering Black-Scholes-Merton, binomial tree, "
            "Monte Carlo simulation, and implied volatility.")

# --- sidebar inputs ---
st.sidebar.header("Option parameters")

ticker_input = st.sidebar.text_input("Ticker (optional)", value="SPY").upper()

if st.sidebar.button("Fetch spot price"):
    try:
        stock = yf.Ticker(ticker_input)
        spot  = stock.history(period="1d")["Close"].iloc[-1]
        st.session_state["S"] = round(float(spot), 2)
        st.sidebar.success(f"{ticker_input}: ${spot:.2f}")
    except:
        st.sidebar.error("Could not fetch price — check ticker")

S     = st.sidebar.number_input("Spot price (S)",     value=st.session_state.get("S", 100.0),  min_value=0.01, step=1.0)
K     = st.sidebar.number_input("Strike price (K)",   value=100.0,  min_value=0.01, step=1.0)
T     = st.sidebar.number_input("Time to expiry (T, years)", value=1.0, min_value=0.01, max_value=5.0, step=0.05)
r     = st.sidebar.number_input("Risk-free rate (r)", value=0.05,   min_value=0.0,  max_value=0.2,  step=0.005, format="%.3f")
sigma = st.sidebar.slider("Volatility (σ)", min_value=0.01, max_value=2.0, value=0.20, step=0.01)
option_type = st.sidebar.radio("Option type", ["call", "put"])

st.sidebar.markdown("---")
N_tree = st.sidebar.slider("Binomial tree steps (N)", min_value=10, max_value=1000, value=200, step=10)
n_sims = st.sidebar.select_slider("Monte Carlo simulations",
                                    options=[1000, 5000, 10000, 50000, 100000],
                                    value=10000)

# --- compute core outputs ---
bsm    = bsm_price(S, K, T, r, sigma, option_type)
tree   = binomial_tree(S, K, T, r, sigma, N_tree, option_type)
d      = delta(S, K, T, r, sigma, option_type)
g      = gamma(S, K, T, r, sigma)
th     = theta(S, K, T, r, sigma, option_type)
v      = vega(S, K, T, r, sigma)
rh     = rho(S, K, T, r, sigma, option_type)

# --- tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Pricer", "Implied volatility", "Volatility surface", "Model comparison"
])

# ================================================================
# TAB 1 — PRICER
# ================================================================
with tab1:
    st.header("Option pricer")

    # price output
    col1, col2, col3 = st.columns(3)
    col1.metric("BSM price",           f"${bsm:.4f}")
    col2.metric("Binomial tree price", f"${tree:.4f}",
                delta=f"{tree - bsm:+.4f} vs BSM")
    col3.metric("Put-call parity check",
                f"${put_call_parity_check(S, K, T, r, sigma)['diff']:.2e}")

    st.markdown("---")

    # Greeks
    st.subheader("Greeks")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Delta", f"{d:.4f}")
    g2.metric("Gamma", f"{g:.4f}")
    g3.metric("Theta (per day)", f"{th:.4f}")
    g4.metric("Vega (per 1% vol)", f"{v:.4f}")
    g5.metric("Rho (per 1% rate)", f"{rh:.4f}")

    st.markdown("---")

    # Greeks vs spot price plot
    st.subheader("Greeks vs spot price")
    S_range = np.linspace(max(S * 0.5, 1), S * 1.5, 300)

    calls_plot  = [bsm_price(s, K, T, r, sigma, option_type) for s in S_range]
    deltas_plot = [delta(s, K, T, r, sigma, option_type) for s in S_range]
    gammas_plot = [gamma(s, K, T, r, sigma) for s in S_range]
    thetas_plot = [theta(s, K, T, r, sigma, option_type) for s in S_range]
    vegas_plot  = [vega(s, K, T, r, sigma) for s in S_range]

    fig, axes = plt.subplots(1, 5, figsize=(18, 3))
    fig.suptitle(f"{option_type.capitalize()} option — K={K}, T={T}, r={r}, σ={sigma}",
                 fontsize=11)

    for ax, values, label, color in zip(
        axes,
        [calls_plot, deltas_plot, gammas_plot, thetas_plot, vegas_plot],
        ["Price", "Delta", "Gamma", "Theta", "Vega"],
        ["steelblue", "steelblue", "purple", "red", "green"]
    ):
        ax.plot(S_range, values, color=color, linewidth=1.5)
        ax.axvline(S, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(K, color="gray",  linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Spot price", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("Black dashed line = current spot price. Gray dashed line = strike.")

# ================================================================
# TAB 2 — IMPLIED VOLATILITY
# ================================================================
with tab2:
    st.header("Implied volatility calculator")
    st.markdown("Enter an observed market price to back out the implied volatility.")

    col1, col2 = st.columns(2)

    with col1:
        market_price = st.number_input(
            "Market option price ($)",
            value=float(round(bsm, 2)),
            min_value=0.01,
            step=0.01,
            format="%.2f"
        )

        if st.button("Calculate implied vol"):
            # try Newton-Raphson first
            iv_nr = implied_vol_newton(market_price, S, K, T, r, option_type)

            if iv_nr is not None and 1e-4 < iv_nr < 5.0:
                method = "Newton-Raphson"
                iv_result = iv_nr
            else:
                iv_result = implied_vol_bisection(market_price, S, K, T, r, option_type)
                method    = "Bisection (NR fallback)"

            if iv_result is not None:
                st.success(f"Implied volatility: **{iv_result*100:.3f}%**")
                st.info(f"Solver used: {method}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Implied vol",    f"{iv_result*100:.3f}%")
                m2.metric("Input sigma",    f"{sigma*100:.1f}%")
                m3.metric("Difference",     f"{(iv_result - sigma)*100:+.3f}%")

                # verify round-trip
                bsm_check = bsm_price(S, K, T, r, iv_result, option_type)
                st.caption(f"Verification: BSM price at implied vol = "
                           f"${bsm_check:.4f} vs market price ${market_price:.4f} "
                           f"(diff = ${abs(bsm_check - market_price):.2e})")
            else:
                st.error("No implied volatility exists for this price — "
                         "check that the price exceeds the intrinsic value.")

    with col2:
        st.markdown("### Newton-Raphson convergence")
        st.markdown("Iterations taken to converge from initial guess σ=0.20:")

        nr_trace = []
        sig = 0.20
        for i in range(20):
            p    = bsm_price(S, K, T, r, sig, option_type)
            diff = p - market_price
            nr_trace.append({"Iteration": i, "Sigma": round(sig, 6),
                             "Price diff": round(abs(diff), 6)})
            if abs(diff) < 1e-8:
                break
            v_raw = vega(S, K, T, r, sig) / 0.01
            if abs(v_raw) < 1e-10:
                break
            sig = sig - diff / v_raw
            if sig <= 0:
                break

        st.dataframe(pd.DataFrame(nr_trace), use_container_width=True)

# ================================================================
# TAB 3 — VOLATILITY SURFACE
# ================================================================
with tab3:
    st.header("Volatility surface")

    surface_ticker = st.text_input("Ticker for vol surface", value="SPY").upper()
    r_surface      = st.number_input("Risk-free rate", value=0.05,
                                      min_value=0.0, max_value=0.2,
                                      step=0.005, format="%.3f",
                                      key="r_surface")

    if st.button("Build volatility surface"):
        with st.spinner("Fetching option chain data..."):
            try:
                stock    = yf.Ticker(surface_ticker)
                S_surf   = stock.history(period="1d")["Close"].iloc[-1]
                expiries = stock.options
                st.info(f"{surface_ticker} spot: ${S_surf:.2f} — "
                        f"{len(expiries)} expiries available")

                records  = []
                progress = st.progress(0)

                for idx, expiry in enumerate(expiries):
                    T_surf = (datetime.strptime(expiry, "%Y-%m-%d") -
                              datetime.today()).days / 365
                    if T_surf < 7/365:
                        continue

                    chain = stock.option_chain(expiry)
                    for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                        for _, row in df.iterrows():
                            K_surf    = row["strike"]
                            mid       = (row["bid"] + row["ask"]) / 2
                            moneyness = K_surf / S_surf

                            if pd.isna(mid) or mid <= 0:
                                continue
                            if moneyness < 0.7 or moneyness > 1.3:
                                continue
                            if pd.isna(row["volume"]) or row["volume"] < 1:
                                continue

                            iv_val = implied_vol(mid, S_surf, K_surf,
                                                  T_surf, r_surface, opt_type)
                            if iv_val is None or iv_val < 0.01 or iv_val > 5.0:
                                continue

                            records.append({
                                "expiry":    expiry,
                                "T":         T_surf,
                                "K":         K_surf,
                                "moneyness": moneyness,
                                "type":      opt_type,
                                "iv":        iv_val
                            })

                    progress.progress((idx + 1) / len(expiries))

                df_surf = pd.DataFrame(records)
                st.success(f"Computed {len(df_surf)} implied vols "
                           f"across {df_surf['expiry'].nunique()} expiries")

                # --- 3D volatility surface (plotly) ---
                st.subheader("3D volatility surface")

                import plotly.graph_objects as go

                otm = df_surf[
                    ((df_surf["type"] == "call") & (df_surf["moneyness"] >= 1.0)) |
                    ((df_surf["type"] == "put")  & (df_surf["moneyness"] <= 1.0))
                ].copy()

                x = otm["moneyness"].values
                y = otm["T"].values
                z = otm["iv"].values * 100

                # build polynomial fit
                moneyness_grid = np.linspace(x.min(), x.max(), 50)
                T_grid         = np.linspace(y.min(), y.max(), 50)
                M_grid, T_mesh = np.meshgrid(moneyness_grid, T_grid)

                A = np.column_stack([
                    np.ones_like(x),
                    x, y,
                    x**2, x*y, y**2,
                    x**3, x**2*y, x*y**2, y**3
                ])
                coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

                IV_fitted = (coeffs[0] +
                             coeffs[1]*M_grid  + coeffs[2]*T_mesh +
                             coeffs[3]*M_grid**2 + coeffs[4]*M_grid*T_mesh +
                             coeffs[5]*T_mesh**2 +
                             coeffs[6]*M_grid**3 + coeffs[7]*M_grid**2*T_mesh +
                             coeffs[8]*M_grid*T_mesh**2 + coeffs[9]*T_mesh**3)

                fig_3d = go.Figure(data=[
                    # polynomial surface — opaque, primary visual element
                    go.Surface(
                        x=moneyness_grid,
                        y=T_grid,
                        z=IV_fitted,
                        colorscale="RdYlGn",
                        reversescale=True,
                        opacity=0.4,
                        name="Polynomial fit",
                        showscale=True,
                        colorbar=dict(title="Implied vol (%)", x=1.0)
                    ),
                    # raw scatter — transparent, secondary reference points
                    go.Scatter3d(
                        x=x, y=y, z=z,
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=z,
                            colorscale="RdYlGn",
                            reversescale=True,
                            opacity=0.8,
                            showscale=False
                        ),
                        name="Market data"
                    )
                ])

                fig_3d.update_layout(
                    title=f"{surface_ticker} implied volatility surface",
                    scene=dict(
                        xaxis_title="Moneyness (K/S)",
                        yaxis_title="Time to expiry (yr)",
                        zaxis_title="Implied vol (%)",
                        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
                    ),
                    legend=dict(x=0, y=1),
                    width=900,
                    height=600,
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                st.plotly_chart(fig_3d, use_container_width=True)

                # --- smile by expiry (matplotlib — stays the same) ---
                st.subheader("Volatility smile by expiry")
                expiry_list = df_surf["expiry"].unique()[:6]
                fig, axes   = plt.subplots(2, 3, figsize=(15, 8))
                axes        = axes.flatten()

                for i, expiry in enumerate(expiry_list):
                    subset = df_surf[df_surf["expiry"] == expiry]
                    calls  = subset[subset["type"] == "call"].sort_values("moneyness")
                    puts   = subset[subset["type"] == "put"].sort_values("moneyness")
                    T_val  = subset["T"].iloc[0]

                    axes[i].plot(calls["moneyness"], calls["iv"] * 100,
                                  color="steelblue", marker="o", markersize=3,
                                  linewidth=0.8, label="Call IV")
                    axes[i].plot(puts["moneyness"],  puts["iv"]  * 100,
                                  color="coral",     marker="o", markersize=3,
                                  linewidth=0.8, label="Put IV")
                    axes[i].axvline(1.0, color="gray", linestyle="--",
                                     alpha=0.4, label="ATM")
                    axes[i].set_title(f"{expiry} (T={T_val:.2f}y)")
                    axes[i].set_xlabel("Moneyness (K/S)")
                    axes[i].set_ylabel("Implied vol (%)")
                    axes[i].legend(fontsize=8)
                    axes[i].grid(alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.error(f"Error fetching data: {e}")
                
# ================================================================
# TAB 4 — MODEL COMPARISON
# ================================================================
with tab4:
    st.header("Model comparison")
    st.markdown("Compare BSM, binomial tree, and Monte Carlo for current parameters.")

    if st.button("Run comparison"):
        with st.spinner("Running models..."):
            results = []

            # BSM
            start = time.perf_counter()
            bsm_p = bsm_price(S, K, T, r, sigma, option_type)
            bsm_t = (time.perf_counter() - start) * 1000
            results.append({
                "Method": "BSM (exact)",
                "Price": f"${bsm_p:.4f}",
                "Error vs BSM": "—",
                "Time (ms)": f"{bsm_t:.3f}",
                "Notes": "Closed-form, European only"
            })

            # Binomial tree — multiple N
            for N in [50, 200, 500]:
                start  = time.perf_counter()
                tree_p = binomial_tree(S, K, T, r, sigma, N, option_type)
                tree_t = (time.perf_counter() - start) * 1000
                results.append({
                    "Method": f"Binomial tree N={N}",
                    "Price": f"${tree_p:.4f}",
                    "Error vs BSM": f"${abs(tree_p - bsm_p):.6f}",
                    "Time (ms)": f"{tree_t:.3f}",
                    "Notes": "Handles American options"
                })

            # Monte Carlo
            for n in [10000, 100000]:
                start = time.perf_counter()
                mc    = monte_carlo(S, K, T, r, sigma,
                                    n_simulations=n,
                                    option_type=option_type)
                mc_t  = (time.perf_counter() - start) * 1000
                results.append({
                    "Method": f"Monte Carlo {n:,}",
                    "Price": f"${mc['price']:.4f}",
                    "Error vs BSM": f"${abs(mc['price'] - bsm_p):.6f}",
                    "Time (ms)": f"{mc_t:.3f}",
                    "Notes": f"SE=${mc['std_error']:.4f}, "
                             f"95% CI=[{mc['ci_lower']:.4f}, {mc['ci_upper']:.4f}]"
                })

            # Antithetic
            start    = time.perf_counter()
            mc_anti  = monte_carlo_antithetic(S, K, T, r, sigma,
                                               n_simulations=10000,
                                               option_type=option_type)
            anti_t   = (time.perf_counter() - start) * 1000
            results.append({
                "Method": "Antithetic MC 10,000",
                "Price": f"${mc_anti['price']:.4f}",
                "Error vs BSM": f"${abs(mc_anti['price'] - bsm_p):.6f}",
                "Time (ms)": f"{anti_t:.3f}",
                "Notes": f"SE=${mc_anti['std_error']:.4f} (~30% reduction)"
            })

            st.dataframe(pd.DataFrame(results), use_container_width=True)

            # convergence chart
            st.subheader("Binomial tree convergence")
            N_range    = range(10, 310, 10)
            tree_prices = [binomial_tree(S, K, T, r, sigma, N, option_type)
                           for N in N_range]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(N_range, tree_prices, color="steelblue",
                    linewidth=0.8, alpha=0.8, label="Tree price")
            ax.axhline(bsm_p, color="black", linestyle="--",
                       linewidth=1, label=f"BSM price (${bsm_p:.4f})")
            ax.set_xlabel("Number of steps (N)")
            ax.set_ylabel("Option price ($)")
            ax.set_title("Binomial tree convergence to BSM")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()