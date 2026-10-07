"""Pixel-reveal game: uncover Poisson-sampled pixels and try to find the signal.

Run with:  streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

TEST_CASE_DIR = Path(__file__).parent / "test_cases"

st.set_page_config(page_title="Signal Hunt", layout="wide")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def list_problems():
    return sorted(p.stem.removeprefix("test_case_") for p in TEST_CASE_DIR.glob("test_case_*.json"))


# cache_resource hands back the same object every rerun instead of unpickling a
# copy; the arrays are never mutated.
@st.cache_resource
def load_problem(name):
    with open(TEST_CASE_DIR / f"test_case_{name}.json") as f:
        data = json.load(f)
    func = data["intensity_function"]
    locations = np.asarray(func["locations"], dtype=float)
    raw = np.clip(np.asarray(func["intensities"], dtype=float), 0.0, None)
    # Normalise the shape to 0 <= I(Q, E) <= 1; the true intensity is A * I + BG.
    raw_max = raw.max()
    shape = raw / raw_max if raw_max > 0 else raw
    threshold = data.get("intensity_threshold")

    # The locations form a regular grid; recover each point's (Q, E) grid index
    # so the map can be thinned and drawn as an image.
    qs, iq = np.unique(locations[:, 0], return_inverse=True)
    es, ie = np.unique(locations[:, 1], return_inverse=True)
    grid = np.full((len(es), len(qs)), np.nan)
    grid[ie, iq] = shape

    pad_q = np.ptp(qs) * 0.02
    pad_e = np.ptp(es) * 0.02
    return {
        "name": data.get("name", name),
        "sample": data.get("sample", {}).get("name", ""),
        "axes": data.get("axes"),
        # Signal cut-off, rescaled to the normalised I(Q, E).
        "threshold": threshold / raw_max if threshold is not None and raw_max > 0 else None,
        "locations": locations,
        "shape": shape,
        "iq": iq,
        "ie": ie,
        "qs": qs,
        "es": es,
        "grid": grid,
        "q_range": [qs[0] - pad_q, qs[-1] + pad_q],
        "e_range": [es[0] - pad_e, es[-1] + pad_e],
    }


@st.cache_resource
def playable_pixels(name, thin_q, thin_e):
    """Indices of the locations kept after taking every n-th Q and E value."""
    prob = load_problem(name)
    return np.flatnonzero((prob["iq"] % thin_q == 0) & (prob["ie"] % thin_e == 0))


def axis_label(vec):
    """Turn an axis vector like [1, 1, 0, 0] into a readable label."""
    if vec is None:
        return ""
    if len(vec) == 4 and not any(vec[:3]) and vec[3]:
        return "Energy (meV)"
    hkl = " ".join(f"{v:g}" for v in vec[:3])
    return f"Q along [{hkl}] (r.l.u.)"


# ----------------------------------------------------------------------------
# Game state
# ----------------------------------------------------------------------------
# Signal-to-noise ratios A/BG offered on the slider: 1/8, 1/4, ..., 4, 8.
SNR_OPTIONS = [2.0 ** k for k in range(-3, 4)]


def format_ratio(r):
    return f"1/{1 / r:g}" if r < 1 else f"{r:g}"


# Mantissas per decade for the log-scale vmin/vmax sliders.
LOG_SLIDER_STEPS = (1, 1.5, 2, 3, 5, 7)


def log_values(lo, hi, mantissas):
    """Values m * 10**d within [lo, hi] for each mantissa m, in ascending order."""
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    # Round away float noise so values like 0.3 format cleanly.
    values = [float(f"{m * 10.0 ** d:.6g}") for d in decades for m in mantissas]
    return [v for v in values if lo <= v <= hi]


def amp_bg(snr, max_intensity):
    """Split the peak intensity A + BG into A and BG with A / BG = snr."""
    bg = max_intensity / (1.0 + snr)
    return max_intensity - bg, bg


def new_game(setup):
    """Start a fresh game; `setup` is (problem, (thin_q, thin_e), snr, max_intensity)."""
    st.session_state.setup = setup
    st.session_state.revealed = {}  # location index -> Poisson count
    st.session_state.answer_shown = False
    st.session_state.rng = np.random.default_rng()
    # A fresh game id resets the curve-count answer box.
    st.session_state.game_id = st.session_state.get("game_id", 0) + 1
    # A fresh chart key clears any selection left on the plot.
    st.session_state.chart_key = st.session_state.get("chart_key", 0) + 1


def reveal(indices, prob):
    """Draw Poisson counts for every not-yet-revealed index."""
    revealed = st.session_state.revealed
    new = [i for i in dict.fromkeys(indices) if i not in revealed]
    if not new or st.session_state.answer_shown:
        return False
    amp, bg = amp_bg(*st.session_state.setup[2:])
    counts = st.session_state.rng.poisson(amp * prob["shape"][new] + bg)
    revealed.update(zip(new, counts.tolist()))
    st.session_state.chart_key += 1
    return True


def selected_indices(chart_state):
    """Location indices of the points inside a box selection."""
    if not chart_state:
        return []
    out = []
    for point in chart_state["selection"]["points"]:
        idx = point.get("customdata")
        if isinstance(idx, list):
            idx = idx[0] if idx else None
        if idx is not None:
            out.append(int(idx))
    return out


problems = list_problems()
if "setup" not in st.session_state:
    new_game((problems[0], (2, 2), 1.0, 100))
problem, thinning, snr, max_intensity = st.session_state.setup

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Problem")
    restart_help = "Changing this starts a new game."
    choice = st.selectbox("Select problem", problems, index=problems.index(problem))
    thin_q = st.slider("Q thinning (keep every n-th)", 1, 10, thinning[0], help=restart_help)
    thin_e = st.slider("E thinning (keep every n-th)", 1, 10, thinning[1], help=restart_help)

    st.header("Intensity: A · I(Q, E) + BG")
    new_snr = st.select_slider("Signal-to-noise (A / BG)", SNR_OPTIONS, snr,
                               format_func=format_ratio,
                               help="Peak signal relative to the flat background. " + restart_help)
    new_max = st.slider("Max intensity (A + BG)", 1, 1000, max_intensity,
                        help="Expected counts at the brightest pixel. " + restart_help)

    setup = (choice, (thin_q, thin_e), new_snr, new_max)
    if setup != st.session_state.setup:
        new_game(setup)
    if st.button("Restart", width="stretch"):
        new_game(setup)
    problem, thinning, snr, max_intensity = setup
    amp, bg = amp_bg(snr, max_intensity)
    st.caption(f"A = {amp:.4g}, BG = {bg:.4g}")
    prob = load_problem(problem)

    st.header("Display")
    pixel_size = st.slider("Pixel size", 1, 40, 10)
    log_scale = st.toggle("Log color scale")
    # Poisson counts can overshoot the peak intensity, so leave headroom.
    c_top = max(10.0, float(np.ceil(1.5 * max_intensity)))
    if log_scale:
        # Log-spaced limits; by default the background sits near the bottom of
        # the scale so the signal gets most of the colour range.
        options = log_values(0.1, c_top, LOG_SLIDER_STEPS)
        lo_default = max(v for v in options if v <= max(bg / 2, 0.1))
        hi_default = min(v for v in options if v >= min(max_intensity, options[-1]))
        vmin = st.select_slider("vmin", options, lo_default, format_func=lambda v: f"{v:g}")
        vmax = st.select_slider("vmax", options, hi_default, format_func=lambda v: f"{v:g}")
        if vmax <= vmin:
            st.warning("vmax must be greater than vmin.")
            vmax = options[min(options.index(vmin) + 1, len(options) - 1)]
            vmin = options[options.index(vmax) - 1]
    else:
        c_step = 0.1 if c_top <= 50 else 1.0
        vmin = st.slider("vmin", 0.0, c_top, 0.0, c_step)
        vmax = st.slider("vmax", 0.0, c_top, min(float(max_intensity), c_top), c_step)
        if vmax <= vmin:
            st.warning("vmax must be greater than vmin.")
            vmax = vmin + c_step

playable = playable_pixels(problem, *thinning)

# Apply the box selection from the previous run *before* drawing, so a reveal
# costs one rerun instead of two.
reveal(selected_indices(st.session_state.get(f"meas_{st.session_state.chart_key}")), prob)

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
locations = prob["locations"]
shape = prob["shape"]
n_playable = len(playable)
xlabel = axis_label(prob["axes"][0]) if prob["axes"] else "Q"
ylabel = axis_label(prob["axes"][1]) if prob["axes"] else "E"

st.title("Signal Hunt")
st.caption(
    f"Problem **{prob['name']}**" + (f" · sample {prob['sample']}" if prob["sample"] else "")
    + ". Drag a box over hidden pixels to measure them. Each measurement "
    "is a Poisson draw from the unknown intensity. Find the signal with as few measurements "
    "as you can."
)

st.subheader("How many curves are there?")
curve_guess = st.number_input(
    "Your answer: number of curves in the signal",
    min_value=0, max_value=50, value=None, step=1, placeholder="Enter a number…",
    key=f"curves_{st.session_state.game_id}",
    disabled=st.session_state.answer_shown,
)

revealed_idx = np.fromiter(st.session_state.revealed.keys(), dtype=int)
revealed_counts = np.fromiter(st.session_state.revealed.values(), dtype=float)
hidden_idx = playable[~np.isin(playable, revealed_idx)]

# Colour mapping shared by both plots. Plotly has no log colour axis, so in log
# mode the values are log10-transformed and the colourbar is relabelled.
if log_scale:
    # Tick at 1-2-5 per decade, or finer when the range spans less than a decade.
    nice = log_values(vmin, vmax, (1, 2, 5))
    if len(nice) < 4:
        nice = log_values(vmin, vmax, LOG_SLIDER_STEPS)
    COLORBAR = dict(thickness=15, len=0.9, tickvals=np.log10(nice).tolist(),
                    ticktext=[f"{v:g}" for v in nice])
    C_MIN, C_MAX = np.log10(vmin), np.log10(vmax)

    def to_color(values):
        return np.log10(np.clip(values, vmin, None))
else:
    COLORBAR = dict(thickness=15, len=0.9)
    C_MIN, C_MAX = vmin, vmax

    def to_color(values):
        return values


def base_layout(fig, title):
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=620,
        margin=dict(l=70, r=20, t=50, b=60),
        plot_bgcolor="black",
        dragmode="select",
        # "d" keeps every drag a true box; plotly otherwise turns a nearly flat
        # drag into a selection of the whole row or column.
        selectdirection="d",
        clickmode="none",
        showlegend=False,
    )
    # Fixed ranges: the domain never changes and dragging cannot pan or zoom.
    fig.update_xaxes(range=prob["q_range"], fixedrange=True, showgrid=False)
    fig.update_yaxes(range=prob["e_range"], fixedrange=True, showgrid=False)
    return fig


def revealed_trace(**marker_overrides):
    marker = dict(
        symbol="square",
        size=pixel_size,
        color=to_color(revealed_counts),
        colorscale="Viridis",
        cmin=C_MIN,
        cmax=C_MAX,
        showscale=True,
        colorbar=dict(COLORBAR, title="counts"),
        line=dict(width=0),
    )
    marker.update(marker_overrides)
    return go.Scattergl(
        x=locations[revealed_idx, 0],
        y=locations[revealed_idx, 1],
        mode="markers",
        customdata=revealed_idx,
        marker=marker,
        text=revealed_counts,
        hovertemplate="Q=%{x:.3f}<br>E=%{y:.3f}<br>counts=%{text:.0f}<extra></extra>",
    )


# Measurement map: hidden pixels are faint grey squares you box-select.
fig = go.Figure([
    go.Scattergl(
        x=locations[hidden_idx, 0],
        y=locations[hidden_idx, 1],
        mode="markers",
        customdata=hidden_idx,
        marker=dict(symbol="square", size=pixel_size, color="rgba(160,160,160,0.35)", line=dict(width=0)),
        hovertemplate="Q=%{x:.3f}<br>E=%{y:.3f}<br>drag a box to reveal<extra></extra>",
    ),
    revealed_trace(),
])
base_layout(fig, f"Measurements ({len(revealed_idx)} / {n_playable} pixels revealed)")

stat_cols = st.columns(3)
stat_cols[0].metric("Pixels revealed", len(revealed_idx))
stat_cols[1].metric("Coverage", f"{100 * len(revealed_idx) / n_playable:.1f}%")
stat_cols[2].metric("Total counts", int(revealed_counts.sum()))

if st.session_state.answer_shown:
    st.plotly_chart(fig, width="stretch", key=f"meas_{st.session_state.chart_key}")

    pct_used = 100 * len(revealed_idx) / n_playable
    st.info(
        f"You used **{pct_used:.1f}%** of the pixels ({len(revealed_idx)} of {n_playable} measured)"
        + (f" and answered **{curve_guess}** curve{'s' if curve_guess != 1 else ''}."
           if curve_guess is not None else " and did not give a curve count.")
    )

    # Full-resolution truth as an image, with the measured pixels marked by x.
    true_grid = amp * prob["grid"] + bg
    truth = go.Figure([
        go.Heatmap(
            x=prob["qs"], y=prob["es"], z=to_color(true_grid), customdata=true_grid,
            colorscale="Viridis", zmin=C_MIN, zmax=C_MAX,
            colorbar=dict(COLORBAR, title="intensity"),
            hovertemplate="Q=%{x:.3f}<br>E=%{y:.3f}<br>intensity=%{customdata:.1f}<extra></extra>",
        ),
        revealed_trace(symbol="x", color="white", size=max(3, 0.6 * pixel_size), showscale=False),
    ])
    base_layout(truth, "True intensity (white x = your measurements)")
    st.plotly_chart(truth, width="stretch", key=f"truth_{st.session_state.chart_key}")

    threshold = prob["threshold"]
    if threshold is not None:
        n_signal = int((shape[playable] > threshold).sum())
        hits = int((shape[revealed_idx] > threshold).sum())
        st.subheader("Results")
        res = st.columns(3)
        res[0].metric(f"Signal pixels (I > {threshold:.2f})", n_signal)
        res[1].metric("Signal pixels you measured", hits,
                      f"{100 * hits / max(n_signal, 1):.1f}% of signal")
        res[2].metric("Hit rate", f"{100 * hits / max(len(revealed_idx), 1):.1f}%",
                      help="Fraction of your measurements that landed on signal.")

    if st.button("Play again", type="primary"):
        new_game(st.session_state.setup)
        st.rerun()
else:
    st.plotly_chart(
        fig,
        width="stretch",
        key=f"meas_{st.session_state.chart_key}",
        on_select="rerun",
        selection_mode="box",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )

    st.divider()
    if st.button("Reveal all", type="primary", width="stretch"):
        st.session_state.answer_shown = True
        st.session_state.chart_key += 1
        st.rerun()
