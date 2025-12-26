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

def highlight_quarter_totals(row):
    if isinstance(row.get("MONTH"), str) and row["MONTH"].endswith("TOTAL"):
        return ["background-color: #4b5d73; color: white; font-weight: 700; border-top: 1px solid #999"] * len(row)
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
tab1, tab2 = st.tabs(["🏅 Leaderboard", "📊 Dashboard"])

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

    # ✅ Apply row highlighting for TOTAL rows + bold header
    styled_lb = (
        lb.style
          .apply(highlight_quarter_totals, axis=1)
          .set_table_styles(
              [{"selector": "th", "props": [("font-weight", "bold")]}]
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

    # Filters
    st.sidebar.header("🔎 Filters")
    quarter = st.sidebar.selectbox("Quarter", ["All"] + [q for q in ["Q1", "Q2", "Q3", "Q4"] if q in dfy["Quarter"].unique()])
    month = st.sidebar.selectbox(
        "Month",
        ["All"]
        + [m for m in MONTH_ORDER if m in dfy["Month"].unique()]
        + sorted([m for m in dfy["Month"].unique() if m not in MONTH_ORDER])
    )
    category = st.sidebar.selectbox("Category", ["All"] + sorted(dfy["Contribution Category"].dropna().unique().tolist()))
    name = st.sidebar.selectbox("Name", ["All"] + sorted(dfy["Name"].dropna().unique().tolist()))
    search = st.sidebar.text_input("Search (any column)", "")

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
        mask = filtered.astype(str).apply(lambda row: row.str.lower().str.contains(s, na=False)).any(axis=1)
        filtered = filtered[mask]

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Total recognitions", len(filtered))
    c2.metric("People recognized", filtered["Name"].nunique())
    c3.metric("Categories", filtered["Contribution Category"].nunique())

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top People**")
        top_people = filtered["Name"].value_counts().head(10).reset_index()
        top_people.columns = ["Name", "Recognitions"]
        st.dataframe(top_people, use_container_width=True, hide_index=True)

    with col2:
        st.write("**Top Categories**")
        top_cat = filtered["Contribution Category"].value_counts().head(10).reset_index()
        top_cat.columns = ["Category", "Recognitions"]
        st.dataframe(top_cat, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(f"🧾 All Recognitions — {selected_year}")

    slack_col = find_slack_col(filtered.columns)

    if slack_col:
        st.data_editor(
            filtered,
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
        st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"recognitions_filtered_{selected_year}.csv",
        mime="text/csv",
    )














