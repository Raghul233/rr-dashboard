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

def normalize_team_perf(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Year", "Quarter", "Month", "Sev2_Received", "Sev2_Contributed", "Sev3_Received", "Sev3_Resolved_RCA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"performance_team.csv is missing columns: {missing}")
        st.stop()

    out = df.copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Quarter"] = out["Quarter"].astype(str).str.strip().str.upper()
    out["Month"] = out["Month"].astype(str).str.strip().str.upper()

    for c in ["Sev2_Received", "Sev2_Contributed", "Sev3_Received", "Sev3_Resolved_RCA"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # Optional: override Quarter using your fiscal mapping (safer)
    out["Quarter"] = out["Month"].map(MONTH_TO_QUARTER).fillna(out["Quarter"])

    return out

def normalize_people_perf(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Year", "Quarter", "Month", "Name", "Sev2_Contributed", "Sev3_Resolved_RCA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"performance_people.csv is missing columns: {missing}")
        st.stop()

    out = df.copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Quarter"] = out["Quarter"].astype(str).str.strip().str.upper()
    out["Month"] = out["Month"].astype(str).str.strip().str.upper()
    out["Name"] = out["Name"].astype(str).str.strip()

    for c in ["Sev2_Contributed", "Sev3_Resolved_RCA"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # Optional: override Quarter using fiscal mapping
    out["Quarter"] = out["Month"].map(MONTH_TO_QUARTER).fillna(out["Quarter"])

    return out

def safe_pct(numer: pd.Series, denom: pd.Series) -> pd.Series:
    # returns percent (0..100), NaN when denom = 0
    denom = denom.replace({0: pd.NA})
    return (numer / denom) * 100

def sort_months(df_: pd.DataFrame, month_col="Month") -> pd.DataFrame:
    order_map = {m: i for i, m in enumerate(MONTH_ORDER)}
    return df_.assign(_m=df_[month_col].map(order_map).fillna(999)).sort_values("_m").drop(columns=["_m"])


# ---------- Tab 3: Performance ----------
with tab3:
    st.subheader(f"📈 Performance — {selected_year}")
    st.caption("Sev-2 shown as **Contribution %**; Sev-3 shown as **Resolution / RCA %**. Percentages are computed from Team received volumes.")

    # Load performance files (repo or uploader)
    team_raw = load_perf_csv(PERF_TEAM_FILE, "Upload performance_team.csv")
    people_raw = load_perf_csv(PERF_PEOPLE_FILE, "Upload performance_people.csv")

    team_perf = normalize_team_perf(team_raw)
    people_perf = normalize_people_perf(people_raw)

    # Scope to selected year
    team_y = team_perf[team_perf["Year"] == selected_year].copy()
    people_y = people_perf[people_perf["Year"] == selected_year].copy()

    # ---------------- Sidebar filters (like Tab 2) ----------------
    st.sidebar.header("📈 Performance Filters")

    perf_quarter = st.sidebar.selectbox(
        "Quarter",
        ["All"] + [q for q in ["Q1", "Q2", "Q3", "Q4"] if q in team_y["Quarter"].unique()],
        key="perf_quarter",
    )

    perf_month = st.sidebar.selectbox(
        "Month",
        ["All"] + [m for m in MONTH_ORDER if m in team_y["Month"].unique()],
        key="perf_month",
    )

    perf_search = st.sidebar.text_input("Search (Name/Month/Quarter)", "", key="perf_search")

    # People-only name filter
    all_perf_people = sorted(people_y["Name"].dropna().unique().tolist(), key=lambda s: s.lower())
    perf_name = st.sidebar.selectbox("Name (People table only)", ["All"] + all_perf_people, key="perf_name")

    # Apply filters
    team_f = team_y.copy()
    people_f = people_y.copy()

    if perf_quarter != "All":
        team_f = team_f[team_f["Quarter"] == perf_quarter]
        people_f = people_f[people_f["Quarter"] == perf_quarter]

    if perf_month != "All":
        team_f = team_f[team_f["Month"] == perf_month]
        people_f = people_f[people_f["Month"] == perf_month]

    if perf_name != "All":
        people_f = people_f[people_f["Name"] == perf_name]

    if perf_search.strip():
        s = perf_search.strip().lower()
        # Team search across Quarter/Month
        team_mask = team_f[["Quarter", "Month"]].astype(str).apply(
            lambda r: r.str.lower().str.contains(s, na=False)
        ).any(axis=1)
        team_f = team_f[team_mask]

        # People search across Quarter/Month/Name
        people_mask = people_f[["Quarter", "Month", "Name"]].astype(str).apply(
            lambda r: r.str.lower().str.contains(s, na=False)
        ).any(axis=1)
        people_f = people_f[people_mask]

    # ---------------- Compute Team % + totals ----------------
    team_f = team_f.copy()
    team_f["Sev2_Contribution_%"] = safe_pct(team_f["Sev2_Contributed"], team_f["Sev2_Received"])
    team_f["Sev3_Resolution_RCA_%"] = safe_pct(team_f["Sev3_Resolved_RCA"], team_f["Sev3_Received"])

    team_f_disp = team_f[[
        "Quarter", "Month",
        "Sev2_Received", "Sev2_Contributed", "Sev2_Contribution_%",
        "Sev3_Received", "Sev3_Resolved_RCA", "Sev3_Resolution_RCA_%"
    ]].copy()

    team_f_disp = sort_months(team_f_disp, "Month")

    # Weighted totals by Quarter (sum numerators / sum denominators)
    if not team_f.empty:
        q_totals = (
            team_f.groupby("Quarter", as_index=False)
            .agg({
                "Sev2_Received": "sum", "Sev2_Contributed": "sum",
                "Sev3_Received": "sum", "Sev3_Resolved_RCA": "sum",
            })
        )
        q_totals["Month"] = q_totals["Quarter"] + " TOTAL"
        q_totals["Sev2_Contribution_%"] = safe_pct(q_totals["Sev2_Contributed"], q_totals["Sev2_Received"])
        q_totals["Sev3_Resolution_RCA_%"] = safe_pct(q_totals["Sev3_Resolved_RCA"], q_totals["Sev3_Received"])

        # Year total
        y_total = pd.DataFrame([{
            "Quarter": "",
            "Month": "YEAR TOTAL",
            "Sev2_Received": int(team_f["Sev2_Received"].sum()),
            "Sev2_Contributed": int(team_f["Sev2_Contributed"].sum()),
            "Sev3_Received": int(team_f["Sev3_Received"].sum()),
            "Sev3_Resolved_RCA": int(team_f["Sev3_Resolved_RCA"].sum()),
        }])
        y_total["Sev2_Contribution_%"] = safe_pct(y_total["Sev2_Contributed"], y_total["Sev2_Received"])
        y_total["Sev3_Resolution_RCA_%"] = safe_pct(y_total["Sev3_Resolved_RCA"], y_total["Sev3_Received"])

        # Append totals underneath (nice read)
        team_f_disp = pd.concat([team_f_disp, q_totals[team_f_disp.columns], y_total[team_f_disp.columns]], ignore_index=True)

    # Pretty formatting
    team_out = team_f_disp.copy()
    team_out["Sev2_Contribution_%"] = team_out["Sev2_Contribution_%"].round(1)
    team_out["Sev3_Resolution_RCA_%"] = team_out["Sev3_Resolution_RCA_%"].round(1)

    # ---------------- Team chart ----------------
    st.markdown("### 🟦 Team Performance")
    if team_y.empty:
        st.info("No team performance data found for this year.")
    else:
        # Chart uses month rows only (exclude totals)
        chart_base = team_f.copy()
        chart_base = chart_base[~chart_base["Month"].str.contains("TOTAL", na=False)]
        chart_base = sort_months(chart_base, "Month")

        chart_df = chart_base.set_index("Month")[["Sev2_Contribution_%", "Sev3_Resolution_RCA_%"]].copy()
        chart_df = chart_df.rename(columns={
            "Sev2_Contribution_%": "Sev-2 Contribution %",
            "Sev3_Resolution_RCA_%": "Sev-3 Resolution/RCA %",
        })

        st.bar_chart(chart_df)

        st.dataframe(
            team_out.rename(columns={
                "Sev2_Contribution_%": "Sev-2 Contribution %",
                "Sev3_Resolution_RCA_%": "Sev-3 Resolution/RCA %",
                "Sev2_Received": "Sev-2 Received",
                "Sev2_Contributed": "Sev-2 Contributed",
                "Sev3_Received": "Sev-3 Received",
                "Sev3_Resolved_RCA": "Sev-3 Resolved/RCA",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ---------------- People table (% computed using TEAM received) ----------------
    st.markdown("### 🟩 People Performance")
    if people_y.empty:
        st.info("No people performance data found for this year.")
    else:
        # Join in Team received denominators by Year+Month (and Quarter for safety)
        denom = team_y[["Year", "Quarter", "Month", "Sev2_Received", "Sev3_Received"]].copy()
        people_join = people_f.merge(denom, on=["Year", "Quarter", "Month"], how="left")

        people_join["Sev2_Contribution_%"] = safe_pct(people_join["Sev2_Contributed"], people_join["Sev2_Received"])
        people_join["Sev3_Resolution_RCA_%"] = safe_pct(people_join["Sev3_Resolved_RCA"], people_join["Sev3_Received"])

        people_out = people_join[[
            "Quarter", "Month", "Name",
            "Sev2_Contributed", "Sev2_Contribution_%",
            "Sev3_Resolved_RCA", "Sev3_Resolution_RCA_%"
        ]].copy()

        people_out = sort_months(people_out, "Month")

        # Round %
        people_out["Sev2_Contribution_%"] = people_out["Sev2_Contribution_%"].round(1)
        people_out["Sev3_Resolution_RCA_%"] = people_out["Sev3_Resolution_RCA_%"].round(1)

        st.dataframe(
            people_out.rename(columns={
                "Sev2_Contributed": "Sev-2 Contributed",
                "Sev2_Contribution_%": "Sev-2 Contribution %",
                "Sev3_Resolved_RCA": "Sev-3 Resolved/RCA",
                "Sev3_Resolution_RCA_%": "Sev-3 Resolution/RCA %",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Optional downloads for performance data
    cdl1, cdl2 = st.columns(2)
    with cdl1:
        st.download_button(
            "⬇️ Download Team Performance (filtered)",
            data=team_out.to_csv(index=False).encode("utf-8"),
            file_name=f"team_performance_{selected_year}.csv",
            mime="text/csv",
        )
    with cdl2:
        st.download_button(
            "⬇️ Download People Performance (filtered)",
            data=people_out.to_csv(index=False).encode("utf-8") if not people_y.empty else "".encode("utf-8"),
            file_name=f"people_performance_{selected_year}.csv",
            mime="text/csv",
        )

