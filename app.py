import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="R&R Dashboard", layout="wide")

CONFLUENCE_URL = "https://interstage.atlassian.net/wiki/spaces/TECH/pages/1289682984/L1+Tech+Ops+Team+-+Rewards+Recognition+Program"

st.title("🏆 Rewards & Recognition - L1 Ops")
st.caption(f"📘 Program details (reward categories + eligibility): [Confluence]({CONFLUENCE_URL})")


EXPECTED_FILE = "L1 Ops - Rewards & Recognition - 2025 - Day_Month.csv"

# Fiscal year: FEB -> JAN
MONTH_TO_QUARTER = {
    "FEBRUARY": "Q1",
    "MARCH": "Q1",
    "APRIL": "Q1",
    "MAY": "Q2",
    "JUNE": "Q2",
    "JULY": "Q2",
    "AUGUST": "Q3",
    "SEPTEMBER": "Q3",
    "OCTOBER": "Q3",
    "NOVEMBER": "Q4",
    "DECEMBER": "Q4",
    "JANUARY": "Q4",
}

MONTH_ORDER = [
    "FEBRUARY", "MARCH", "APRIL",
    "MAY", "JUNE", "JULY",
    "AUGUST", "SEPTEMBER", "OCTOBER",
    "NOVEMBER", "DECEMBER", "JANUARY",
]

@st.cache_data
def read_csv(obj):
    df = pd.read_csv(obj)
    df.columns = [c.strip() for c in df.columns]
    return df

def load_data():
    # 1) Try committed CSV
    if os.path.exists(EXPECTED_FILE):
        return read_csv(EXPECTED_FILE)

    # 2) Fallback: uploader
    st.warning("CSV not found in the repo. Upload it below.")
    up = st.file_uploader("Upload the rewards CSV", type=["csv"])
    if up is not None:
        return read_csv(up)

    st.info("Waiting for CSV…")
    st.stop()

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Year", "Month", "Name", "Contribution Category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    out = df.copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Month"] = out["Month"].astype(str).str.strip().str.upper()
    out["Name"] = out["Name"].astype(str).str.strip()
    out["Contribution Category"] = out["Contribution Category"].astype(str).str.strip()

    out["Quarter"] = out["Month"].map(MONTH_TO_QUARTER).fillna("—")
    return out

def build_leaderboard(df: pd.DataFrame, people_order=None) -> pd.DataFrame:
    pivot = pd.pivot_table(
        df,
        index=["Quarter", "Month"],
        columns="Name",
        values="Contribution Category",
        aggfunc="count",
        fill_value=0,
    )

    def month_sort_key(m):
        m = str(m).upper()
        return MONTH_ORDER.index(m) if m in MONTH_ORDER else 999

    if len(pivot.index) > 0:
        ordered_pairs = sorted(
            pivot.index.tolist(),
            key=lambda x: (x[0], month_sort_key(x[1]), x[1]),
        )
        pivot = pivot.loc[ordered_pairs]

    cols = list(pivot.columns)
    if people_order:
        ordered_cols = [p for p in people_order if p in cols] + [c for c in cols if c not in people_order]
        pivot = pivot[ordered_cols]
    else:
        pivot = pivot[sorted(cols, key=lambda s: s.lower())]

    out_rows = []
    out_index = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        if len(pivot.index) == 0 or q not in pivot.index.get_level_values(0):
            continue

        q_rows = pivot.loc[q]
        for m, row in q_rows.iterrows():
            out_index.append((q, m))
            out_rows.append(row)

        total = q_rows.sum(axis=0)
        out_index.append(("", f"{q} TOTAL"))
        out_rows.append(total)

    out = pd.DataFrame(out_rows, index=pd.MultiIndex.from_tuples(out_index, names=["QUARTER", "MONTH"]))
    out = out.reset_index()

    display_quarter = []
    prev_q = None
    for _, r in out.iterrows():
        q = r["QUARTER"]
        if q == "":
            display_quarter.append("")
            prev_q = None
        else:
            if q != prev_q:
                display_quarter.append(q)
                prev_q = q
            else:
                display_quarter.append("")
    out["QUARTER"] = display_quarter

    return out

def highlight_top_scorer(row):
    """
    Highlight top scorer(s) in quarter TOTAL rows by font color only.
    """
    if not (isinstance(row.get("MONTH"), str) and row["MONTH"].endswith("TOTAL")):
        return [""] * len(row)

    styles = [""] * len(row)

    # numeric columns only (exclude QUARTER, MONTH)
    numeric_vals = row.iloc[2:]
    if numeric_vals.empty:
        return styles

    max_val = numeric_vals.max()

    for i, v in enumerate(row):
        # i >= 2 ensures we only style person columns
        if i >= 2 and v == max_val and max_val > 0:
            styles[i] = "color: #f1c40f; font-weight: 700"

    return styles

def highlight_quarter_totals(row):
    if isinstance(row.get("MONTH"), str) and row["MONTH"].endswith("TOTAL"):
        return ["background-color: #4b5d73; color: white; font-weight: 700"] * len(row)
    return [""] * len(row)


def find_slack_col(columns):
    for c in columns:
        if c.strip().lower() in ["slack link", "slack", "slack_url", "slack url"]:
            return c
    return None

# ---------- Load + normalize ----------
df_raw = load_data()
df = normalize(df_raw)

# ---------- Year selector (auto) ----------
years = sorted([int(y) for y in df["Year"].dropna().unique().tolist()])
if not years:
    st.error("No valid Year values found. Please fill Year column with 2025/2026 etc.")
    st.stop()

selected_year = st.selectbox("📅 Select Year", years, index=len(years) - 1)
dfy = df[df["Year"] == selected_year].copy()

# ---------- Tabs (Leaderboard first) ----------
tab1, tab2, tab3 = st.tabs(["🏅 Leaderboard", "📊 Dashboard", "📈 Performance"])

# ---------- Tab 1: Leaderboard ----------
with tab1:
    st.subheader(f"🏅 Leaderboard — {selected_year}")
    st.caption(
        "Counts = number of recognitions for the Month in the selected year. "
        "Quarter totals auto-calculated."
    )

    all_people = sorted(
        dfy["Name"].dropna().unique().tolist(),
        key=lambda s: s.lower()
    )

    selected_people = st.multiselect(
        "Select people (controls columns + order)",
        options=all_people,
        default=all_people,
    )

    lb = build_leaderboard(
        dfy[dfy["Name"].isin(selected_people)],
        people_order=selected_people,
    )

    styled_lb = (
        lb.style
          .apply(highlight_quarter_totals, axis=1)
          .apply(highlight_top_scorer, axis=1)
          .set_table_styles(
              [{"selector": "thead th", "props": [("font-weight", "bold")]}]
          )
    )

    st.dataframe(
        styled_lb,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Leaderboard CSV",
        data=lb.to_csv(index=False).encode("utf-8"),
        file_name=f"leaderboard_{selected_year}.csv",
        mime="text/csv",
    )

# ---------- Tab 2: Dashboard ----------
with tab2:
    st.subheader(f"📊 Dashboard — {selected_year}")

    # ---------------- Sidebar filters (affect Metrics + All Recognitions table) ----------------
    st.sidebar.header("🔎 Filters")

    quarter = st.sidebar.selectbox(
        "Quarter",
        ["All"] + [q for q in ["Q1", "Q2", "Q3", "Q4"] if q in dfy["Quarter"].unique()],
        key="flt_quarter",
    )

    month = st.sidebar.selectbox(
        "Month",
        ["All"]
        + [m for m in MONTH_ORDER if m in dfy["Month"].unique()]
        + sorted([m for m in dfy["Month"].unique() if m not in MONTH_ORDER]),
        key="flt_month",
    )

    category = st.sidebar.selectbox(
        "Category",
        ["All"] + sorted(dfy["Contribution Category"].dropna().unique().tolist()),
        key="flt_category",
    )

    name = st.sidebar.selectbox(
        "Name",
        ["All"] + sorted(dfy["Name"].dropna().unique().tolist()),
        key="flt_name",
    )

    search = st.sidebar.text_input("Search (any column)", "", key="flt_search")

    filtered = dfy.copy()
    if quarter != "All":
        filtered = filtered[filtered["Quarter"] == quarter]
    if month != "All":
        filtered = filtered[filtered["Month"] == month]
    if category != "All":
        filtered = filtered[filtered["Contribution Category"] == category]
    if name != "All":
        filtered = filtered[filtered["Name"] == name]
    if search.strip():
        s = search.strip().lower()
        mask = filtered.astype(str).apply(
            lambda r: r.str.lower().str.contains(s, na=False)
        ).any(axis=1)
        filtered = filtered[mask]

    # ---------------- Summary (uses sidebar-filtered data) ----------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Total recognitions", len(filtered))
    c2.metric("People recognized", filtered["Name"].nunique())
    c3.metric("Categories", filtered["Contribution Category"].nunique())

    st.divider()

    # ---------------- Month dropdown ONLY for the TOP tables ----------------
    st.markdown("### 📅 Month for Top People & Top Categories")
    month_opts = ["All"] + [m for m in MONTH_ORDER if m in dfy["Month"].unique()]
    month_top = st.selectbox(
        "Select Month (Top tables only)",
        month_opts,
        index=0,
        key="top_month",
    )

    top_df = dfy.copy()
    if month_top != "All":
        top_df = top_df[top_df["Month"] == month_top]

    top_label = "All months" if month_top == "All" else month_top

    # ---------------- Build Top People (Category split) ----------------
    keep_cats = ["Sev-2", "Sev-3", "BAU"]
    tmp = top_df.copy()

    tmp["CatBucket"] = tmp["Contribution Category"].astype(str).str.strip()
    tmp["CatBucket"] = tmp["CatBucket"].apply(lambda x: x if x in keep_cats else "Others")

    top_people_split = (
        tmp.pivot_table(
            index="Name",
            columns="CatBucket",
            values="Contribution Category",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    # Ensure all required columns exist
    for c in ["Sev-2", "Sev-3", "BAU", "Others"]:
        if c not in top_people_split.columns:
            top_people_split[c] = 0

    top_people_split["Total Recognitions"] = (
        top_people_split["Sev-2"]
        + top_people_split["Sev-3"]
        + top_people_split["BAU"]
        + top_people_split["Others"]
    )

    top_people_split = top_people_split[
        ["Name", "Sev-2", "Sev-3", "BAU", "Others", "Total Recognitions"]
    ].sort_values("Total Recognitions", ascending=False).head(10)

    # ---------------- Build Top Categories ----------------
    top_categories = (
        top_df["Contribution Category"]
        .value_counts()
        .reset_index()
    )
    top_categories.columns = ["Category", "Recognitions"]
    top_categories = top_categories.head(10)

    # ---------------- Render side-by-side ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Top People ({top_label}) — Category split**")
        st.dataframe(top_people_split, use_container_width=True, hide_index=True)

    with col2:
        st.write(f"**Top Categories ({top_label})**")
        st.dataframe(top_categories, use_container_width=True, hide_index=True)

    st.divider()

    # ---------------- All Recognitions table (uses sidebar filtered + hides Year) ----------------
    st.subheader("🧾 All Recognitions")

    display_df = filtered.drop(columns=["Year"], errors="ignore")
    slack_col = find_slack_col(display_df.columns)

    if slack_col:
        st.data_editor(
            display_df,
            use_container_width=True,
            hide_index=True,
            disabled=True,
            column_config={
                slack_col: st.column_config.LinkColumn(
                    slack_col,
                    help="Open the Slack message",
                    display_text="Open",
                )
            },
        )
    else:
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"recognitions_filtered_{selected_year}.csv",
        mime="text/csv",
    )

# -------------------------------
# Performance CSV names (repo root)
# -------------------------------
PERF_TEAM_FILE = "performance_team.csv"
PERF_PEOPLE_FILE = "performance_people.csv"
PERF_POD_FILE = "performance_pod.csv"

@st.cache_data
def read_csv_cached(path_or_file):
    d = pd.read_csv(path_or_file)
    d.columns = [c.strip() for c in d.columns]
    return d

def load_perf_csv(expected_filename: str, uploader_label: str) -> pd.DataFrame:
    # 1) Try committed file
    if os.path.exists(expected_filename):
        return read_csv_cached(expected_filename)

    # 2) Fallback uploader
    st.warning(f"'{expected_filename}' not found in repo. Upload it below.")
    up = st.file_uploader(uploader_label, type=["csv"], key=f"uploader_{expected_filename}")
    if up is not None:
        return read_csv_cached(up)

    st.info(f"Waiting for {expected_filename}…")
    st.stop()

def _to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def safe_pct(numer: pd.Series, denom: pd.Series) -> pd.Series:
    # returns percent (0..100); 0 when denom = 0
    denom = denom.replace({0: pd.NA})
    return (numer / denom).fillna(0) * 100

def sort_months(df_: pd.DataFrame, month_col="Month") -> pd.DataFrame:
    order_map = {m: i for i, m in enumerate(MONTH_ORDER)}
    return df_.assign(_m=df_[month_col].map(order_map).fillna(999)).sort_values("_m").drop(columns=["_m"])

def normalize_team_perf(df_in: pd.DataFrame) -> pd.DataFrame:
    dfp = df_in.copy()
    dfp.columns = [c.strip() for c in dfp.columns]

    # map variants -> canonical
    colmap = {}
    for c in dfp.columns:
        cl = c.strip().lower()
        if cl == "year":
            colmap[c] = "Year"
        elif cl == "quarter":
            colmap[c] = "Quarter"
        elif cl == "month":
            colmap[c] = "Month"
        elif cl in ["sev2_received", "sev 2 received", "sev2 received"]:
            colmap[c] = "Sev2_Received"
        elif cl in ["sev2_contributed", "sev 2 contributed", "sev2 contributed", "sev2_resolved", "sev 2 resolved"]:
            colmap[c] = "Sev2_Contributed"
        elif cl in ["sev3_received", "sev 3 received", "sev3 received"]:
            colmap[c] = "Sev3_Received"
        elif cl in [
            "sev3_contributed", "sev 3 contributed", "sev3 contributed",
            "sev3_resolved", "sev 3 resolved",
            "sev3 resolved / contributed to rca", "sev3_resolved_rca", "sev3_resolution_rca",
            "sev3_resolved / rca"
        ]:
            colmap[c] = "Sev3_Resolved_RCA"

    dfp = dfp.rename(columns=colmap)

    required = ["Year", "Quarter", "Month", "Sev2_Received", "Sev2_Contributed", "Sev3_Received", "Sev3_Resolved_RCA"]
    missing = [c for c in required if c not in dfp.columns]
    if missing:
        st.error(f"{PERF_TEAM_FILE} is missing columns: {missing}")
        st.stop()

    dfp["Year"] = pd.to_numeric(dfp["Year"], errors="coerce").astype("Int64")
    dfp["Quarter"] = dfp["Quarter"].astype(str).str.strip().str.upper()
    dfp["Month"] = dfp["Month"].astype(str).str.strip().str.upper()

    for c in ["Sev2_Received", "Sev2_Contributed", "Sev3_Received", "Sev3_Resolved_RCA"]:
        dfp[c] = _to_num(dfp[c]).astype(int)

    # Optional: override Quarter using fiscal mapping (safer)
    dfp["Quarter"] = dfp["Month"].map(MONTH_TO_QUARTER).fillna(dfp["Quarter"])

    # % columns (safe divide)
    dfp["Sev2_Contribution_%"] = safe_pct(dfp["Sev2_Contributed"], dfp["Sev2_Received"])
    dfp["Sev3_Resolution_RCA_%"] = safe_pct(dfp["Sev3_Resolved_RCA"], dfp["Sev3_Received"])

    # month ordering
    dfp["Month"] = pd.Categorical(dfp["Month"], categories=MONTH_ORDER, ordered=True)
    dfp = dfp.sort_values(["Quarter", "Month"])

    return dfp

def normalize_people_perf(df_in: pd.DataFrame) -> pd.DataFrame:
    dfp = df_in.copy()
    dfp.columns = [c.strip() for c in dfp.columns]

    colmap = {}
    for c in dfp.columns:
        cl = c.strip().lower()
        if cl == "year":
            colmap[c] = "Year"
        elif cl == "quarter":
            colmap[c] = "Quarter"
        elif cl == "month":
            colmap[c] = "Month"
        elif cl == "name":
            colmap[c] = "Name"
        elif cl in ["sev2_contributed", "sev 2 contributed", "sev2 contributed", "sev2_resolved", "sev 2 resolved"]:
            colmap[c] = "Sev2_Contributed"
        elif cl in [
            "sev3_contributed", "sev 3 contributed", "sev3 contributed",
            "sev3_resolved", "sev 3 resolved",
            "sev3 resolved / contributed to rca", "sev3_resolved_rca", "sev3_resolution_rca",
            "sev3_resolved / rca"
        ]:
            colmap[c] = "Sev3_Resolved_RCA"

    dfp = dfp.rename(columns=colmap)

    required = ["Year", "Quarter", "Month", "Name", "Sev2_Contributed", "Sev3_Resolved_RCA"]
    missing = [c for c in required if c not in dfp.columns]
    if missing:
        st.error(f"{PERF_PEOPLE_FILE} is missing columns: {missing}")
        st.stop()

    dfp["Year"] = pd.to_numeric(dfp["Year"], errors="coerce").astype("Int64")
    dfp["Quarter"] = dfp["Quarter"].astype(str).str.strip().str.upper()
    dfp["Month"] = dfp["Month"].astype(str).str.strip().str.upper()
    dfp["Name"] = dfp["Name"].astype(str).str.strip()

    dfp["Sev2_Contributed"] = _to_num(dfp["Sev2_Contributed"]).astype(int)
    dfp["Sev3_Resolved_RCA"] = _to_num(dfp["Sev3_Resolved_RCA"]).astype(int)

    # Optional: override Quarter using fiscal mapping
    dfp["Quarter"] = dfp["Month"].map(MONTH_TO_QUARTER).fillna(dfp["Quarter"])

    dfp["Month"] = pd.Categorical(dfp["Month"], categories=MONTH_ORDER, ordered=True)
    dfp = dfp.sort_values(["Quarter", "Month", "Name"])

    return dfp

def normalize_pod_perf(df_in: pd.DataFrame) -> pd.DataFrame:
    dfp = df_in.copy()
    dfp.columns = [c.strip() for c in dfp.columns]

    # map variants -> canonical
    colmap = {}
    for c in dfp.columns:
        cl = c.strip().lower()
        if cl == "year":
            colmap[c] = "Year"
        elif cl == "quarter":
            colmap[c] = "Quarter"
        elif cl == "month":
            colmap[c] = "Month"
        elif cl in ["pods", "pod"]:
            colmap[c] = "PODS"
        elif cl in ["sev 2 received", "sev2_received", "sev2 received", "sev2_received "]:
            colmap[c] = "Sev2_Received"
        elif cl in ["sev 2 contributed", "sev2_contributed", "sev2 contributed", "sev2 resolved", "sev 2 resolved"]:
            colmap[c] = "Sev2_Contributed"
        elif cl in ["sev 3 received", "sev3_received", "sev3 received"]:
            colmap[c] = "Sev3_Received"
        elif cl in ["sev 3 resolved / rca", "sev3_resolved_rca", "sev3 resolved / rca", "sev 3 resolved", "sev3 resolved"]:
            colmap[c] = "Sev3_Resolved_RCA"

    dfp = dfp.rename(columns=colmap)

    required = ["Year", "Quarter", "Month", "PODS", "Sev2_Received", "Sev2_Contributed", "Sev3_Received", "Sev3_Resolved_RCA"]
    missing = [c for c in required if c not in dfp.columns]
    if missing:
        st.error(f"{PERF_POD_FILE} is missing columns: {missing}")
        st.stop()

    dfp["Year"] = pd.to_numeric(dfp["Year"], errors="coerce").astype("Int64")
    dfp["Quarter"] = dfp["Quarter"].astype(str).str.strip().str.upper()
    dfp["Month"] = dfp["Month"].astype(str).str.strip().str.upper()
    dfp["PODS"] = dfp["PODS"].astype(str).str.strip().str.upper()

    for c in ["Sev2_Received", "Sev2_Contributed", "Sev3_Received", "Sev3_Resolved_RCA"]:
        dfp[c] = _to_num(dfp[c]).astype(int)

    # Optional: override Quarter using fiscal mapping
    dfp["Quarter"] = dfp["Month"].map(MONTH_TO_QUARTER).fillna(dfp["Quarter"])

    dfp["Sev2_Contribution_%"] = safe_pct(dfp["Sev2_Contributed"], dfp["Sev2_Received"]).round(1)
    dfp["Sev3_Resolution_RCA_%"] = safe_pct(dfp["Sev3_Resolved_RCA"], dfp["Sev3_Received"]).round(1)

    dfp["Month"] = pd.Categorical(dfp["Month"], categories=MONTH_ORDER, ordered=True)
    dfp = dfp.sort_values(["Quarter", "Month", "PODS"])

    return dfp


# ---------- Tab 3: Performance ----------
with tab3:
    import altair as alt

    st.subheader(f"📈 Performance — {selected_year}")
    st.caption(
        "Sev-2 shown as **Contribution %**; Sev-3 shown as **Resolution / RCA %**. "
        "Percentages are computed using Team received volumes."
    )

    # --- Load performance files ---
    team_raw = load_perf_csv(PERF_TEAM_FILE, f"Upload {PERF_TEAM_FILE}")
    people_raw = load_perf_csv(PERF_PEOPLE_FILE, f"Upload {PERF_PEOPLE_FILE}")
    pod_raw = load_perf_csv(PERF_POD_FILE, f"Upload {PERF_POD_FILE}")

    team_perf_all = normalize_team_perf(team_raw)
    people_perf_all = normalize_people_perf(people_raw)
    pod_perf_all = normalize_pod_perf(pod_raw)

    # --- Filter to selected year ---
    team_perf = team_perf_all[team_perf_all["Year"] == selected_year].copy()
    people_perf = people_perf_all[people_perf_all["Year"] == selected_year].copy()
    pod_perf = pod_perf_all[pod_perf_all["Year"] == selected_year].copy()

    if team_perf.empty:
        st.warning(f"No team performance rows found for {selected_year}.")
        st.stop()

    # --- Month filter (same pattern as Tab 2 "Top tables only") ---
    available_months = [m for m in MONTH_ORDER if m in team_perf["Month"].astype(str).unique().tolist()]
    month_filter = st.selectbox(
        "📅 Select Month (Performance tables + charts only)",
        ["All"] + available_months,
        index=0,
        key="perf_month_filter",
    )

    team_view = team_perf.copy()
    people_view = people_perf.copy()
    pod_view = pod_perf.copy()

    if month_filter != "All":
        team_view = team_view[team_view["Month"].astype(str) == month_filter]
        people_view = people_view[people_view["Month"].astype(str) == month_filter]
        pod_view = pod_view[pod_view["Month"].astype(str) == month_filter]

    # =======================
    # 1) TEAM CHARTS (side-by-side) + separator between graphs
    # =======================
    st.subheader("📊 Team Performance — Visual")

    chart_base = team_view.copy()
    chart_base["Month"] = chart_base["Month"].astype(str)

    chart_base["Sev2_Contribution_%"] = pd.to_numeric(chart_base["Sev2_Contribution_%"], errors="coerce").fillna(0)
    chart_base["Sev3_Resolution_RCA_%"] = pd.to_numeric(chart_base["Sev3_Resolution_RCA_%"], errors="coerce").fillna(0)

    left, sep, right = st.columns([1, 0.03, 1])

    with left:
        st.markdown("**Sev-2 Contribution %**")
        sev2_chart = (
            alt.Chart(chart_base)
            .mark_bar()
            .encode(
                x=alt.X("Month:N", sort=MONTH_ORDER, title=None),
                y=alt.Y("Sev2_Contribution_%:Q", title="%", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Month", alt.Tooltip("Sev2_Contribution_%:Q", title="Sev-2 %", format=".1f")],
            )
            .properties(height=320)
        )
        st.altair_chart(sev2_chart, use_container_width=True)

    with sep:
        st.markdown(
            "<div style='height:360px; border-left: 2px solid rgba(255,255,255,0.15); margin: 0 auto;'></div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("**Sev-3 Resolution / RCA %**")
        sev3_chart = (
            alt.Chart(chart_base)
            .mark_bar()
            .encode(
                x=alt.X("Month:N", sort=MONTH_ORDER, title=None),
                y=alt.Y("Sev3_Resolution_RCA_%:Q", title="%", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Month", alt.Tooltip("Sev3_Resolution_RCA_%:Q", title="Sev-3 %", format=".1f")],
            )
            .properties(height=320)
        )
        st.altair_chart(sev3_chart, use_container_width=True)

    st.divider()

    # =======================
    # 2) TEAM TABLE (with received + contributed counts + %)
    # =======================
    st.subheader("🧾 Team Performance — Table")

    team_out = team_view.copy()
    team_out["Month"] = team_out["Month"].astype(str)

    team_out = team_out[
        ["Quarter", "Month",
         "Sev2_Received", "Sev2_Contributed", "Sev2_Contribution_%",
         "Sev3_Received", "Sev3_Resolved_RCA", "Sev3_Resolution_RCA_%"]
    ].copy()

    team_out = team_out.rename(
        columns={
            "Sev2_Received": "Sev-2 Received",
            "Sev2_Contributed": "Sev-2 Contributed",
            "Sev2_Contribution_%": "Sev-2 Contribution %",
            "Sev3_Received": "Sev-3 Received",
            "Sev3_Resolved_RCA": "Sev-3 Resolved / RCA",
            "Sev3_Resolution_RCA_%": "Sev-3 Resolution / RCA %",
        }
    )

    team_out["Sev-2 Contribution %"] = pd.to_numeric(team_out["Sev-2 Contribution %"], errors="coerce").fillna(0).round(1)
    team_out["Sev-3 Resolution / RCA %"] = pd.to_numeric(team_out["Sev-3 Resolution / RCA %"], errors="coerce").fillna(0).round(1)

    percent_cols = ["Sev-2 Contribution %", "Sev-3 Resolution / RCA %"]

    styled_team = team_out.style.format(
        {col: "{:.1f}%" for col in percent_cols if col in team_out.columns}
    )
    
    st.dataframe(styled_team, use_container_width=True, hide_index=True)

    st.divider()

    # =======================
    # 3) POD TABLE (separate)
    # =======================
    st.subheader("🧩 POD Performance — Monthly")

    if pod_view.empty:
        st.info("No POD performance rows for the selected month filter.")
    else:
        pod_out = pod_view.copy()
        pod_out["Month"] = pod_out["Month"].astype(str)

        pod_out = pod_out.rename(
            columns={
                "Sev2_Received": "Sev-2 Received",
                "Sev2_Contributed": "Sev-2 Contributed",
                "Sev2_Contribution_%": "Sev-2 Contribution %",
                "Sev3_Received": "Sev-3 Received",
                "Sev3_Resolved_RCA": "Sev-3 Resolved / RCA",
                "Sev3_Resolution_RCA_%": "Sev-3 Resolution / RCA %",
            }
        )

        # Ensure % are rounded for display
        if "Sev-2 Contribution %" in pod_out.columns:
            pod_out["Sev-2 Contribution %"] = pd.to_numeric(pod_out["Sev-2 Contribution %"], errors="coerce").fillna(0).round(1)
        if "Sev-3 Resolution / RCA %" in pod_out.columns:
            pod_out["Sev-3 Resolution / RCA %"] = pd.to_numeric(pod_out["Sev-3 Resolution / RCA %"], errors="coerce").fillna(0).round(1)

        pod_display_cols = [
            "Quarter", "Month", "PODS",
            "Sev-2 Received", "Sev-2 Contributed", "Sev-2 Contribution %",
            "Sev-3 Received", "Sev-3 Resolved / RCA", "Sev-3 Resolution / RCA %",
        ]
        pod_display_cols = [c for c in pod_display_cols if c in pod_out.columns]

        st.dataframe(
            pod_out[pod_display_cols].sort_values(
                [c for c in ["Quarter", "Month", "PODS"] if c in pod_out.columns]
            ),
            use_container_width=True,
            hide_index=True
        )
        
    st.divider()

    # =======================
    # 4) PEOPLE TABLES
    # =======================
    st.subheader("👥 People Performance — Overall (Year-to-date)")

    team_year_totals = team_perf.copy()  # full-year denominators
    sev2_received_year = float(team_year_totals["Sev2_Received"].sum())
    sev3_received_year = float(team_year_totals["Sev3_Received"].sum())

    people_year = people_perf.copy()
    overall = (
        people_year.groupby("Name", as_index=False)[["Sev2_Contributed", "Sev3_Resolved_RCA"]]
        .sum()
        .rename(columns={"Sev2_Contributed": "Sev-2 Contributed", "Sev3_Resolved_RCA": "Sev-3 Resolved / RCA"})
    )

    overall["Sev-2 Contribution %"] = ((overall["Sev hookup"] if False else overall["Sev-2 Contributed"]) / (sev2_received_year if sev2_received_year else 1) * 100).round(1)
    overall["Sev-3 Resolution / RCA %"] = ((overall["Sev-3 Resolved / RCA"]) / (sev3_received_year if sev3_received_year else 1) * 100).round(1)

    overall["Total Contributed (Sev-2 + Sev-3)"] = (overall["Sev-2 Contributed"] + overall["Sev-3 Resolved / RCA"]).astype(int)
    overall = overall.sort_values("Total Contributed (Sev-2 + Sev-3)", ascending=False)

    percent_cols = ["Sev-2 Contribution %", "Sev-3 Resolution / RCA %"]

    styled_overall = overall.style.format(
        {col: "{:.1f}%" for col in percent_cols if col in overall.columns}
    )
    
    st.dataframe(styled_overall, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("👥 People Performance — Monthly")

    if people_view.empty:
        st.info("No people performance rows for the selected month filter.")
    else:
        team_month_den = team_perf.copy()
        team_month_den["Month"] = team_month_den["Month"].astype(str)
        team_den = team_month_den.groupby("Month", as_index=False)[["Sev2_Received", "Sev3_Received"]].sum()

        people_m = people_view.copy()
        people_m["Month"] = people_m["Month"].astype(str)

        pm = (
            people_m.groupby(["Month", "Name"], as_index=False)[["Sev2_Contributed", "Sev3_Resolved_RCA"]]
            .sum()
            .merge(team_den, on="Month", how="left")
        )

        pm["Sev-2 Contribution %"] = safe_pct(pm["Sev2_Contributed"], pm["Sev2_Received"]).round(1)
        pm["Sev-3 Resolution / RCA %"] = safe_pct(pm["Sev3_Resolved_RCA"], pm["Sev3_Received"]).round(1)

        pm = pm.rename(
            columns={
                "Sev2_Contributed": "Sev-2 Contributed",
                "Sev3_Resolved_RCA": "Sev-3 Resolved / RCA",
                "Sev2_Received": "Sev-2 Received (Team)",
                "Sev3_Received": "Sev-3 Received (Team)",
            }
        )

        pm["Total Contributed (Sev-2 + Sev-3)"] = (pm["Sev-2 Contributed"] + pm["Sev-3 Resolved / RCA"]).astype(int)

        pm["Month"] = pd.Categorical(pm["Month"], categories=MONTH_ORDER, ordered=True)
        pm = pm.sort_values(["Month", "Total Contributed (Sev-2 + Sev-3)"], ascending=[True, False])

        pm_display = pm[
            [
                "Month", "Name",
                "Sev-2 Received (Team)", "Sev-2 Contributed", "Sev-2 Contribution %",
                "Sev-3 Received (Team)", "Sev-3 Resolved / RCA", "Sev-3 Resolution / RCA %",
                "Total Contributed (Sev-2 + Sev-3)",
            ]
        ]

    percent_cols = ["Sev-2 Contribution %", "Sev-3 Resolution / RCA %"]

    styled_pm = pm_display.style.format(
        {col: "{:.1f}%" for col in percent_cols if col in pm_display.columns}
    )
    
    st.dataframe(styled_pm, use_container_width=True, hide_index=True)







