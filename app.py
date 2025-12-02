import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="R&R Dashboard", layout="wide")
st.title("🏆 Rewards & Recognition Dashboard")

SHEET_CSV_URL = st.secrets["https://docs.google.com/spreadsheets/d/1Eea9Md0dIFHKP5kM51doAt42lhH0qh7CdHIB5MsvPFI/edit?gid=0#gid=0"]  # put this in Streamlit Cloud secrets

@st.cache_data(ttl=300)  # refresh every 5 minutes
def load_from_sheet():
    df = pd.read_csv(SHEET_CSV_URL)
    df.columns = [c.strip() for c in df.columns]
    return df

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

    # 2) Fallback: uploader (helps if file wasn’t committed or filename differs)
    st.warning("CSV not found in the repo. Upload it below.")
    up = st.file_uploader("Upload the rewards CSV", type=["csv"])
    if up is not None:
        return read_csv(up)

    st.info("Waiting for CSV…")
    st.stop()

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Basic validation
    required = ["Month", "Name", "Contribution Category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    out = df.copy()
    out["Month"] = out["Month"].astype(str).str.strip().str.upper()
    out["Name"] = out["Name"].astype(str).str.strip()
    out["Contribution Category"] = out["Contribution Category"].astype(str).str.strip()

    out["Quarter"] = out["Month"].map(MONTH_TO_QUARTER).fillna("—")
    return out

def make_clickable_links(df: pd.DataFrame) -> pd.DataFrame:
    # If your column name differs, change it here:
    link_col = None
    for c in df.columns:
        if c.strip().lower() in ["slack link", "slack", "slack_url", "slack url"]:
            link_col = c
            break
    if link_col is None:
        return df

    out = df.copy()
    out[link_col] = out[link_col].fillna("").astype(str)
    return out

def build_leaderboard(df: pd.DataFrame, people_order=None) -> pd.DataFrame:
    # Pivot: counts per Name per (Quarter, Month)
    pivot = pd.pivot_table(
        df,
        index=["Quarter", "Month"],
        columns="Name",
        values="Contribution Category",
        aggfunc="count",
        fill_value=0,
    )

    # Order by fiscal month order
    def month_sort_key(m):
        m = str(m).upper()
        return MONTH_ORDER.index(m) if m in MONTH_ORDER else 999

    ordered_pairs = sorted(
        pivot.index.tolist(),
        key=lambda x: (x[0], month_sort_key(x[1]), x[1]),
    )
    pivot = pivot.loc[ordered_pairs]

    # Column order
    cols = list(pivot.columns)
    if people_order:
        ordered_cols = [p for p in people_order if p in cols] + [c for c in cols if c not in people_order]
        pivot = pivot[ordered_cols]
    else:
        pivot = pivot[sorted(cols, key=lambda s: s.lower())]

    # Add quarter total rows
    out_rows = []
    out_index = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        if q not in pivot.index.get_level_values(0):
            continue

        q_rows = pivot.loc[q]  # DataFrame index = Month
        for m, row in q_rows.iterrows():
            out_index.append((q, m))
            out_rows.append(row)

        total = q_rows.sum(axis=0)
        out_index.append(("", f"{q} TOTAL"))
        out_rows.append(total)

    out = pd.DataFrame(out_rows, index=pd.MultiIndex.from_tuples(out_index, names=["QUARTER", "MONTH"]))
    out = out.reset_index()

    # Display quarter only for the first month row of that quarter
    display_quarter = []
    prev_q = None
    for _, r in out.iterrows():
        q = r["QUARTER"]
        m = r["MONTH"]
        if q == "":
            display_quarter.append("")  # total rows blank
            prev_q = None
        else:
            if q != prev_q:
                display_quarter.append(q)
                prev_q = q
            else:
                display_quarter.append("")
    out["QUARTER"] = display_quarter

    return out

# ---------- Load + normalize ----------
df_raw = load_data()
df = normalize(df_raw)

# ---------- Tabs (Leaderboard first) ----------
tab1, tab2 = st.tabs(["🏅 Leaderboard", "📊 Dashboard"])

# ---------- Tab 1: Leaderboard ----------
with tab1:
    st.subheader("🏅 Leaderboard (Quarter → Month → People)")
    st.caption("Counts = number of recognitions in the CSV. Quarter totals are auto-calculated.")

    all_people = sorted(df["Name"].dropna().unique().tolist(), key=lambda s: s.lower())

    selected_people = st.multiselect(
        "Select people (controls columns + order)",
        options=all_people,
        default=all_people,
    )

    lb = build_leaderboard(df[df["Name"].isin(selected_people)], people_order=selected_people)

    st.dataframe(lb, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Leaderboard CSV",
        data=lb.to_csv(index=False).encode("utf-8"),
        file_name="leaderboard.csv",
        mime="text/csv",
    )

# ---------- Tab 2: Dashboard ----------
with tab2:
    # Filters
    st.sidebar.header("🔎 Filters")
    quarter = st.sidebar.selectbox("Quarter", ["All"] + [q for q in ["Q1", "Q2", "Q3", "Q4"] if q in df["Quarter"].unique()])
    month = st.sidebar.selectbox("Month", ["All"] + [m for m in MONTH_ORDER if m in df["Month"].unique()] + sorted([m for m in df["Month"].unique() if m not in MONTH_ORDER]))
    category = st.sidebar.selectbox("Category", ["All"] + sorted(df["Contribution Category"].dropna().unique().tolist()))
    name = st.sidebar.selectbox("Name", ["All"] + sorted(df["Name"].dropna().unique().tolist()))
    search = st.sidebar.text_input("Search (any column)", "")

    filtered = df.copy()
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

    st.subheader("📊 Summary")
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
    st.subheader("🧾 All Entries")

    view_df = make_clickable_links(filtered)

    # Detect Slack Link column name (for LinkColumn display)
    slack_col = None
    for c in view_df.columns:
        if c.strip().lower() in ["slack link", "slack", "slack_url", "slack url"]:
            slack_col = c
            break

    if slack_col:
        st.data_editor(
            view_df,
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
        st.dataframe(view_df, use_container_width=True, hide_index=True)

    # Download filtered
    st.download_button(
        "⬇️ Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="recognitions_filtered.csv",
        mime="text/csv",
    )

