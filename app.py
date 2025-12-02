import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="R&R Dashboard", layout="wide")
st.title("🏆 Rewards & Recognition Dashboard")

EXPECTED_FILE = "L1 Ops - Rewards & Recognition - 2025 - Day_Month.csv"

@st.cache_data
def read_csv(obj):
    df = pd.read_csv(obj)
    df.columns = [c.strip() for c in df.columns]
    return df

# --- Load data ---
df = None
if os.path.exists(EXPECTED_FILE):
    df = read_csv(EXPECTED_FILE)
else:
    st.warning("CSV not found in the repo. Upload it below.")
    up = st.file_uploader("Upload the rewards CSV", type=["csv"])
    if up is not None:
        df = read_csv(up)

if df is None:
    st.info("Waiting for CSV…")
    st.stop()

# --- Filters ---
st.sidebar.header("🔎 Filters")
month = st.sidebar.selectbox("Month", ["All"] + sorted(df["Month"].dropna().unique().tolist()))
week = st.sidebar.selectbox("Week", ["All"] + sorted(df["Week"].dropna().unique().tolist()))
category = st.sidebar.selectbox("Category", ["All"] + sorted(df["Contribution Category"].dropna().unique().tolist()))
name = st.sidebar.selectbox("Name", ["All"] + sorted(df["Name"].dropna().unique().tolist()))
search = st.sidebar.text_input("Search (any column)", "")

filtered = df.copy()
if month != "All": filtered = filtered[filtered["Month"] == month]
if week != "All": filtered = filtered[filtered["Week"] == week]
if category != "All": filtered = filtered[filtered["Contribution Category"] == category]
if name != "All": filtered = filtered[filtered["Name"] == name]
if search.strip():
    s = search.strip().lower()
    mask = filtered.astype(str).apply(lambda row: row.str.lower().str.contains(s, na=False)).any(axis=1)
    filtered = filtered[mask]

# --- Dashboard ---
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
