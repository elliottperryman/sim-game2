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
def new_game(setup):
    """Start a fresh game; `setup` is (problem, (thin_q, thin_e), A, BG)."""
    st.session_state.setup = setup
    st.session_state.revealed = {}  # location index -> Poisson count
    st.session_state.answer_shown = False
    st.session_state.rng = np.random.default_rng()
    # A fresh chart key clears any selection left on the plot.
    st.session_state.chart_key = st.session_state.get("chart_key", 0) + 1


def reveal(indices, prob):
    """Draw Poisson counts for every not-yet-revealed index."""
    revealed = st.session_state.revealed
    new = [i for i in dict.fromkeys(indices) if i not in revealed]
    if not new or st.session_state.answer_shown:
        return False
    _, _, amp, bg = st.session_state.setup
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
    new_game((problems[0], (2, 2), 100.0, 1.0))
problem, thinning, amp, bg = st.session_state.setup

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
    new_amp = st.slider("A (signal amplitude)", 0.0, 1000.0, amp, 1.0,
                        help="Peak signal above background. " + restart_help)
    new_bg = st.slider("BG (background)", 0.0, 100.0, bg, 0.1,
                       help="Flat background added everywhere. " + restart_help)

    setup = (choice, (thin_q, thin_e), new_amp, new_bg)
    if setup != st.session_state.setup:
        new_game(setup)
    if st.button("Restart", width="stretch"):
        new_game(setup)
    problem, thinning, amp, bg = setup
    prob = load_problem(problem)

    st.header("Display")
    pixel_size = st.slider("Pixel size", 1, 40, 10)
    # Poisson counts can overshoot the peak intensity, so leave headroom.
    c_top = max(10.0, float(np.ceil(1.5 * (amp + bg))))
    c_step = 0.1 if c_top <= 50 else 1.0
    vmin = st.slider("vmin", 0.0, c_top, 0.0, c_step)
    vmax = st.slider("vmax", 0.0, c_top, min(float(np.ceil(amp + bg)), c_top) or c_top, c_step)
    if vmax <= vmin:
        st.warning("vmax must be greater than vmin.")
        vmax = vmin + c_step
    log_scale = st.toggle("Log color scale")

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

revealed_idx = np.fromiter(st.session_state.revealed.keys(), dtype=int)
revealed_counts = np.fromiter(st.session_state.revealed.values(), dtype=float)
hidden_idx = playable[~np.isin(playable, revealed_idx)]

# Colour mapping shared by both plots. Plotly has no log colour axis, so in log
# mode the values are log10-transformed and the colourbar is relabelled.
if log_scale:
    c_lo = vmin if vmin > 0 else min(0.1, vmax / 10)  # log needs a positive floor
    nice = [m * 10.0 ** d
            for d in range(int(np.floor(np.log10(c_lo))), int(np.ceil(np.log10(vmax))) + 1)
            for m in (1, 2, 5)]
    nice = [v for v in nice if c_lo <= v <= vmax]
    COLORBAR = dict(thickness=15, len=0.9, tickvals=np.log10(nice).tolist(),
                    ticktext=[f"{v:g}" for v in nice])
    C_MIN, C_MAX = np.log10(c_lo), np.log10(vmax)

    def to_color(values):
        return np.log10(np.clip(values, c_lo, None))
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
    col_meas, col_true = st.columns(2)
    with col_meas:
        st.plotly_chart(fig, width="stretch", key=f"meas_{st.session_state.chart_key}")
    with col_true:
        # Full-resolution truth as an image, with the measured pixels outlined.
        true_grid = amp * prob["grid"] + bg
        truth = go.Figure([
            go.Heatmap(
                x=prob["qs"], y=prob["es"], z=to_color(true_grid), customdata=true_grid,
                colorscale="Viridis", zmin=C_MIN, zmax=C_MAX,
                colorbar=dict(COLORBAR, title="intensity"),
                hovertemplate="Q=%{x:.3f}<br>E=%{y:.3f}<br>intensity=%{customdata:.1f}<extra></extra>",
            ),
            revealed_trace(symbol="square-open", color="red", showscale=False, line=dict(width=1)),
        ])
        base_layout(truth, "True intensity (red outlines = your measurements)")
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
    if st.button("Reveal", type="primary", width="stretch"):
        st.session_state.answer_shown = True
        st.session_state.chart_key += 1
        st.rerun()
