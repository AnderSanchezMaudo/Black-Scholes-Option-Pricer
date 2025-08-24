import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# -------------------------------
# Black-Scholes pricing function
# -------------------------------
def black_scholes_prices(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(layout="wide")

# Hide the anchor icon that appears on headings
st.markdown("""
    <style>
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
    }
    /* Bigger tab labels */
    div[data-testid="stTabs"] button p {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("European Option Pricing - Black-Scholes Model")

# Author banner
AUTHOR = "Ander Sánchez Maudo"
LINK   = "https://github.com/AnderSanchezMaudo"

st.markdown(f"""
<div style="padding:0.6rem 1rem; border-radius:8px;
            background:linear-gradient(90deg, #e8f0fe, #f8faff);
            font-size:0.95rem; margin-top:0.6rem; margin-bottom:0.6rem;">
  <strong>Author:</strong>
  <a href="{LINK}" target="_blank" style="text-decoration:none; color:#1f77b4; font-weight:bold;">
    {AUTHOR}
  </a>
  · © {pd.Timestamp.today().year} · All rights reserved
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar: parameters (independent scroll)
# -------------------------------
with st.sidebar:
    st.header("Model Inputs")

    S = st.number_input("Current Stock Price (S) [€]", value=100.0, min_value=0.01)
    K = st.number_input("Strike Price (K) [€]", value=100.0, min_value=0.01)
    T = st.number_input("Time to Maturity (T, years)", value=1.0, min_value=0.01)
    r = st.number_input("Risk-Free Interest Rate (r) (e.g., 0.015 for 1.5%)",
                        value=0.015, min_value=0.0, max_value=1.0,
                        step=0.001, format="%.3f")
    sigma = st.number_input("Annual Volatility (σ)", value=0.2,
                            min_value=0.001, max_value=1.0, step=0.01)

    st.header("P&L Inputs")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        premium_call = st.number_input("Premium Paid for Call [€]", value=5.00, min_value=0.0, step=0.01)
    with col_p2:
        premium_put  = st.number_input("Premium Paid for Put [€]",  value=5.00, min_value=0.0, step=0.01)

    st.header("Heatmap Settings")
    spot_min, spot_max = st.slider("Spot Price Range", 50, 150, (80, 120))
    sigma_min, sigma_max = st.slider("Volatility Range (%)", 5, 100, (10, 50))
    n_spot = st.slider("Spot Resolution", 5, 20, 9)
    n_sigma = st.slider("Volatility Resolution", 5, 20, 9)

# -------------------------------
# Main area: results and charts
# -------------------------------
# Point metrics for the current inputs
call_price, put_price = black_scholes_prices(S, K, T, r, sigma)
call_pnl = call_price - premium_call
put_pnl  = put_price  - premium_put

st.subheader("Model Outputs")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Call Price", f"{call_price:.2f}€")
with mcol2:
    st.metric("Put Price", f"{put_price:.2f}€")
with mcol3:
    st.metric("Call P&L (Model)", f"{call_pnl:.2f}€")
with mcol4:
    st.metric("Put P&L (Model)", f"{put_pnl:.2f}€")

# Heatmap grids
spot_prices = np.linspace(spot_min, spot_max, n_spot)
volatilities = np.linspace(sigma_min / 100, sigma_max / 100, n_sigma)
S_grid, sigma_grid = np.meshgrid(spot_prices, volatilities)

# Price matrices
call_matrix_price, put_matrix_price = black_scholes_prices(S_grid, K, T, r, sigma_grid)
call_df_price = pd.DataFrame(
    np.round(call_matrix_price, 2),
    index=[f"{v * 100:.2f}" for v in volatilities],  # show numeric percent without '%'
    columns=[f"{s:.2f}" for s in spot_prices]
)
put_df_price = pd.DataFrame(
    np.round(put_matrix_price, 2),
    index=[f"{v * 100:.2f}" for v in volatilities],
    columns=[f"{s:.2f}" for s in spot_prices]
)

# P&L matrices (Model = model price - premium paid)
call_matrix_pnl = call_matrix_price - premium_call
put_matrix_pnl = put_matrix_price - premium_put
call_df_pnl = pd.DataFrame(
    np.round(call_matrix_pnl, 2),
    index=call_df_price.index, columns=call_df_price.columns
)
put_df_pnl = pd.DataFrame(
    np.round(put_matrix_pnl, 2),
    index=put_df_price.index, columns=put_df_price.columns
)

# Shared symmetric limits for P&L heatmaps (negative=red, zero=yellow, positive=green)
pnl_absmax = np.max(np.abs(np.concatenate([call_df_pnl.values.ravel(),
                                           put_df_pnl.values.ravel()])))
if pnl_absmax == 0:
    pnl_absmax = 1e-9  # avoid zero range
custom_cmap = LinearSegmentedColormap.from_list(
    "rg_custom",
    ["#b30000", "#ffd300", "#008837"],  # rojo, amarillo, verde
    N=256
)
norm = TwoSlopeNorm(vmin=-pnl_absmax, vcenter=0, vmax=pnl_absmax)
pnl_kwargs = dict(cmap=custom_cmap, norm=norm, cbar_kws={'label': '€ P&L'})

# Tabs: Prices / P&L
tab_prices, tab_pnl = st.tabs(["Prices", "P&L"])

with tab_prices:
    st.subheader("Option Price Heatmaps")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("##### Heatmap - Call Price")
        fig_call, ax_call = plt.subplots(figsize=(9, 7))
        sns.heatmap(call_df_price, annot=True, fmt=".2f",
                    cmap="viridis", ax=ax_call, cbar_kws={'label': '€'})
        ax_call.set_xlabel("Spot Price (S)")
        ax_call.set_ylabel("Volatility (%)")
        ax_call.set_yticklabels(ax_call.get_yticklabels(), rotation=0)
        st.pyplot(fig_call)

    with col_h2:
        st.markdown("##### Heatmap - Put Price")
        fig_put, ax_put = plt.subplots(figsize=(9, 7))
        sns.heatmap(put_df_price, annot=True, fmt=".2f",
                    cmap="YlOrRd", ax=ax_put, cbar_kws={'label': '€'})
        ax_put.set_xlabel("Spot Price (S)")
        ax_put.set_ylabel("Volatility (%)")
        ax_put.set_yticklabels(ax_put.get_yticklabels(), rotation=0)
        st.pyplot(fig_put)

    st.markdown("#### Download Tables (Prices)")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download Call Prices (CSV)", call_df_price.to_csv().encode(),
                           file_name="call_prices.csv", mime="text/csv")
    with c2:
        st.download_button("Download Put Prices (CSV)", put_df_price.to_csv().encode(),
                           file_name="put_prices.csv", mime="text/csv")

with tab_pnl:
    st.subheader("Option P&L Heatmaps (Model Price − Premium Paid)")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("##### Heatmap - Call P&L")
        fig_call_pnl, ax_call_pnl = plt.subplots(figsize=(9, 7))
        sns.heatmap(call_df_pnl, annot=True, fmt=".2f",
                    ax=ax_call_pnl, **pnl_kwargs)
        ax_call_pnl.set_xlabel("Spot Price (S)")
        ax_call_pnl.set_ylabel("Volatility (%)")
        ax_call_pnl.set_yticklabels(ax_call_pnl.get_yticklabels(), rotation=0)
        st.pyplot(fig_call_pnl)

    with col_h2:
        st.markdown("##### Heatmap - Put P&L")
        fig_put_pnl, ax_put_pnl = plt.subplots(figsize=(9, 7))
        sns.heatmap(put_df_pnl, annot=True, fmt=".2f",
                    ax=ax_put_pnl, **pnl_kwargs)
        ax_put_pnl.set_xlabel("Spot Price (S)")
        ax_put_pnl.set_ylabel("Volatility (%)")
        ax_put_pnl.set_yticklabels(ax_put_pnl.get_yticklabels(), rotation=0)
        st.pyplot(fig_put_pnl)

    st.markdown("#### Download Tables (P&L)")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download Call P&L (CSV)", call_df_pnl.to_csv().encode(),
                           file_name="call_pnl.csv", mime="text/csv")
    with c2:
        st.download_button("Download Put P&L (CSV)", put_df_pnl.to_csv().encode(),
                           file_name="put_pnl.csv", mime="text/csv")
