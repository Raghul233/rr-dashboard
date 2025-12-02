import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="R&R Dashboard", layout="wide")
st.title("🏆 Rewards & Recognition Dashboard")

EXPECTED_FILE = "L1 Ops - Rewards & Recognition - 2025 - Day_Month.csv"

# --- Quarter mapping (edit if your fiscal quarters differ) ---
MONTH_TO_QUARTER = {
    "JANUARY": "Q4",
    "FEBRUARY": "Q4",
    "MARCH": "Q4",
    "APRIL": "Q1",
    "MAY": "Q1",
    "JUNE": "Q1",
    "JULY": "Q2",
    "AUGUST": "Q2",
    "SEPTEMBER": "Q3",
    "OCTOBER": "Q3",
    "NOVEMBER": "Q4",
    "DECEMBER": "Q4",
}

MONTH_ORDER = [
    "APRIL","MAY","JUNE",
    "JULY","AUGUST","SEPTEMBER",
    "OCTOBER","NOVEMBER","DECEMBER",
    "JANUARY","FEBRUARY","MARCH",
]

@st.cache_data
def read_csv(obj):
    df = pd.read_csv(obj)
    df.columns = [c.strip() for c in df.columns]
    return df

def load_data():
    if os.path.exists(EXPECTED_FILE):
        return read_csv(EXPECTED_FILE)

    st.warning("CSV not found in the repo. Upload it below.")
    up = st.file_uploader("Upload the rewards CSV", type=["csv"])
    if up is not None:
        return read_csv(up)

    st.info("Waiting for CSV…")
    st.stop()

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure expected columns exist
    needed = ["Month", "Name", "Contribution Category"]
    for c in needed:
        if c not in df.columns:
            st.error(f"Missing required column: {c}")
            st.stop()

    df["Month"] = df["Month"].astype(str).str.strip().str.upper()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Quarter"] = df["Month"].map(MONTH_TO_QUARTER).fillna("—")
    return df

def build_leaderboard(df: pd.DataFrame, people_order=None) -> pd.DataFrame:
    # Count recognitions per person per month
    pivot = pd.pivot_table(
        df,
        index=["Quarter", "Month"],
        columns="Name",
        values="Contribution Category",
        aggfunc="count",
        fill_value=0,
    )

    # Make sure month ordering is consistent
    # Keep only months present, but ordered by MONTH_ORDER if possible
    months_present = pivot.index.get_level_values("Month").unique().tolist()

    def month_sort_key(m):
        m = str(m).upper()
        return MONTH_ORDER.index(m) if m in MONTH_ORDER else 999

    # Rebuild ordered multi-index
    ordered_pairs = sorted(
        [(q, m) for (q, m) in pivot.index.tolist()],
        key=lambda x: (x[0], month_sort_key(x[1]), x[1]),
    )
    pivot = pivot.loc[ordered_pairs]

    # Optional: order people columns
    cols = list(pivot.columns)
    if people_order:
        # keep listed people first, then others
        ordered_cols = [p for p in people_order if p in cols] + [c for c in cols if c not in people_order]
        pivot = pivot[ordered_cols]
    else:
        pivot = pivot[sorted(cols, key=lambda s: s.lower())]

    # Insert quarter total rows
    out_rows = []
    out_index = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        q_rows = pivot.loc[q] if q in pivot.index.get_level_values(0) else None
        if q_rows is None:
            continue

        # q_rows is a DataFrame indexed by Month
        for m, row in q_rows.iterrows():
            out_index.append((q, m))
            out_rows.append(row)

        # total row
        total = q_rows.sum(axis=0)
        out_index.append(("", f"{q} TOTAL"))
        out_rows.append(total)

    out = pd.DataFrame(out_rows, index=pd.MultiIndex.from_tuples(out_index, names=["QUARTER", "MONTH"]))
    out = out.reset_index()

    # Make it look like your sample (quarter shown only at first row of that quarter)
    # We'll blank repeated quarter values except where it changes or total row.
    display_quarter = []
    prev = None
    for _, r in out.iterrows():
        q = r["QUARTER"]
        m = r["MONTH"]
        if q == "":
            display_quarter.append("")  # total row already blank
            prev = None
        else:
            if q != prev:
                display_quarter.append(q)
                prev = q
            else:
                display_quarter.append("")
    out["QUARTER"] = display_quarter

    return out

# ---------------- App ----------------
df_raw = load_data()
df = normalize(df_raw)

tab1, tab2 = st.tabs(["📊 Dashboard", "🏅 Leaderboard"])

with tab1:
    # Sidebar filters (dashboard)
    st.sidebar.header("🔎 Filters")
    month = st.sidebar.selectbox("Month", ["All"] + sorted(df["Month"].dropna().unique().tolist()))
    quarter = st.sidebar.selectbox("Quarter", ["All"] + sorted(df["Quarter"].dropna().unique().tolist()))
    category = st.sidebar.selectbox("Category", ["All"] + sorted(df["Contribution Category"].dropna().unique().tolist()))
    name = st.sidebar.selectbox("Name", ["All"] + sorted(df["Name"].dropna().unique().tolist()))
    search = st.sidebar.text_input("Search (any column)", "")

    filtered = df.copy()
    if month != "All": filtered = filtered[filtered["Month"] == month]
    if quarter != "All": filtered = filtered[filtered["Quarter"] == quarter]
    if category != "All": filtered = filtered[filtered["Contribution Category"] == category]
    if name != "All": filtered = filtered[filtered["Name"] == name]
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
    st.dataframe(filtered, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🏅 Leaderboard (Quarter → Month → People)")
    st.caption("Counts = number of recognitions in the CSV. Quarter totals are auto-calculated.")

    # Optional: choose which people (columns) to show and in what order
    all_people = sorted(df["Name"].dropna().unique()._
