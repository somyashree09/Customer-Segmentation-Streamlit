# app.py - Advanced Customer Segmentation Dashboard (Sidebar + Tabs)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io

st.set_page_config(page_title="Customer Segmentation (Premium)", layout="wide")

# ---- Helpers ----
@st.cache_data
def load_data():
    for name in ("ecommerce_customer_data_full.csv", "ecommerce_customer_data.csv"):
        try:
            df = pd.read_csv(name)
            st.session_state["_data_file"] = name
            return df
        except Exception:
            continue
    st.warning("No dataset found. Please save your dataset as 'ecommerce_customer_data_full.csv' or 'ecommerce_customer_data.csv' in the project folder.")
    return pd.DataFrame()

def ensure_columns(df):
    # Ensure numeric columns exist
    expected = ['Age','Recency','Frequency','Monetary',
                'Electronics_Spend','Fashion_Spend','Grocery_Spend','Lifestyle_Spend','Region','CustomerID']
    for c in expected:
        if c not in df.columns:
            # create sensible defaults
            if c == 'CustomerID':
                df['CustomerID'] = np.arange(1, len(df)+1)
            elif c == 'Region':
                df['Region'] = 'Unknown'
            else:
                df[c] = 0
    return df

def compute_basic_churn_cltv(df):
    # If churn_proba not present, create a heuristic churn probability from Recency
    if 'churn_proba' not in df.columns:
        df['churn_proba'] = (df['Recency'] / (df['Recency'].max() + 1)).clip(0,1)
    # CLTV fallback: avg_order_value * freq * (1 - churn_proba)
    if 'avg_order_value' not in df.columns:
        df['avg_order_value'] = df.apply(lambda r: r['Monetary'] / r['Frequency'] if r['Frequency']>0 else r['Monetary'], axis=1)
    if 'CLTV_simple' not in df.columns:
        df['retention_prob'] = 1 - df['churn_proba']
        df['exp_purchases_next_yr'] = df['Frequency']
        df['CLTV_simple'] = df['avg_order_value'] * df['exp_purchases_next_yr'] * df['retention_prob']
    return df

@st.cache_data
def train_kmeans(df, n_clusters=5):
    features = ['Age','Recency','Frequency','Monetary']
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, scaler, X_scaled

def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ---- Load & prepare data ----
df = load_data()
if df.empty:
    st.stop()

df = ensure_columns(df)
df = compute_basic_churn_cltv(df)

# Train KMeans automatically (or reuse existing cluster column)
if 'Cluster' not in df.columns:
    kmeans, scaler, X_scaled = train_kmeans(df, n_clusters=5)
    df['Cluster'] = kmeans.labels_
else:
    # if cluster exists, still compute scaler for predictions / PCA
    features = ['Age','Recency','Frequency','Monetary']
    scaler = StandardScaler()
    scaler.fit(df[features].fillna(0))

# PCA for visualization
features = ['Age','Recency','Frequency','Monetary']
pca = PCA(n_components=2, random_state=42)
pca_vals = pca.fit_transform(scaler.transform(df[features].fillna(0)))
df['PCA1'] = pca_vals[:,0]
df['PCA2'] = pca_vals[:,1]

# ---- Sidebar filters ----
st.sidebar.header("Filters & Controls")
clusters_available = sorted(df['Cluster'].unique().tolist())
sel_clusters = st.sidebar.multiselect("Clusters", clusters_available, default=clusters_available)
age_min, age_max = st.sidebar.slider("Age range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
region_opts = df['Region'].unique().tolist()
sel_regions = st.sidebar.multiselect("Regions", region_opts, default=region_opts)
cltv_min = float(df['CLTV_simple'].min())
cltv_max = float(df['CLTV_simple'].max())
sel_cltv = st.sidebar.slider("CLTV range", float(round(cltv_min,0)), float(round(cltv_max,0)), (float(round(cltv_min,0)), float(round(cltv_max,0))))

# Filtered dataframe
df_filtered = df[
    (df['Cluster'].isin(sel_clusters)) &
    (df['Age'] >= age_min) & (df['Age'] <= age_max) &
    (df['Region'].isin(sel_regions)) &
    (df['CLTV_simple'] >= sel_cltv[0]) & (df['CLTV_simple'] <= sel_cltv[1])
].copy()

# ---- Main layout with Tabs ----
tab1, tab2, tab3 = st.tabs(["Overview", "Segments", "Prediction"])

with tab1:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers (filtered)", len(df_filtered))
    c2.metric("Avg Monetary", f"{df_filtered['Monetary'].mean():.0f}")
    c3.metric("Avg Frequency", f"{df_filtered['Frequency'].mean():.2f}")
    c4.metric("Avg CLTV", f"{df_filtered['CLTV_simple'].mean():.0f}")

    st.markdown("**PCA visualization (interactive)**")
    fig = px.scatter(df_filtered, x='PCA1', y='PCA2', color='Cluster',
                     hover_data=['CustomerID','Monetary','Frequency','churn_proba'],
                     color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(template='plotly_dark', height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Monetary vs Frequency**")
    fig2 = px.scatter(df_filtered, x='Frequency', y='Monetary', color='Cluster',
                      size='Monetary', hover_data=['CustomerID','Age'], template='plotly_dark', height=420)
    st.plotly_chart(fig2, use_container_width=True)

# ----------------- REPLACE tab2 (Segments) with this block -----------------
with tab2:
    st.subheader("Segments - Profiles & Drill-down")

    # 1) Friendly cluster names & descriptions
    cluster_names = {
        0: ("Loyal Value Customers", "Regular buyers with steady spend across categories."),
        1: ("Affluent Occasional Buyers", "High spenders but less frequent; reactivation priority."),
        2: ("Premium Young Spenders", "Younger customers with high electronics & lifestyle spend."),
        3: ("Budget-Conscious Regulars", "Frequent buyers focused on essentials and groceries."),
        4: ("New / Low-Engagement", "Younger or new customers with lower spend; nurture with offers.")
    }

    # Show legend with icons
    cols = st.columns(5)
    for i, c in enumerate(sorted(df['Cluster'].unique())):
        name, desc = cluster_names.get(c, (f"Cluster {c}", ""))
        cols[i].markdown(f"**{c} — {name}**")
        cols[i].caption(desc)

    st.markdown("---")

    # 2) Cluster summary (means)
    cluster_summary_all = df.groupby('Cluster')[['Age','Recency','Frequency','Monetary','Electronics_Spend','Fashion_Spend','Grocery_Spend','Lifestyle_Spend']].mean().round(2)
    st.markdown("**Cluster summary (means)**")
    st.dataframe(cluster_summary_all.style.background_gradient(axis=1))

    st.markdown("**Cluster sizes**")
    sizes = df['Cluster'].value_counts().sort_index()
    st.bar_chart(sizes)

    st.markdown("---")

    # 3) Drill-down controls: choose cluster or individual customer
    st.markdown("### Drill-down")
    ccol1, ccol2 = st.columns([2,3])

    with ccol1:
        chosen_cluster = st.selectbox("Select Cluster to inspect", options=sorted(df['Cluster'].unique()))
        cluster_df = df[df['Cluster'] == chosen_cluster].copy()
        st.write(f"Customers in cluster {chosen_cluster}: {len(cluster_df)}")
        if st.button("Download this cluster CSV"):
            st.download_button(label="Download cluster CSV",
                               data=cluster_df.to_csv(index=False).encode('utf-8'),
                               file_name=f"cluster_{chosen_cluster}_customers.csv",
                               mime="text/csv")

    with ccol2:
        # Customer selector inside the chosen cluster
        sample_ids = cluster_df['CustomerID'].tolist()[:500]  # limit dropdown size if huge
        chosen_customer = st.selectbox("Or pick a CustomerID", options=sample_ids)
        cust = df[df['CustomerID'] == chosen_customer].iloc[0]
        st.markdown("**Customer Summary**")
        st.write({
            "CustomerID": cust['CustomerID'],
            "Age": int(cust['Age']),
            "Gender": cust['Gender'],
            "Region": cust['Region'],
            "Cluster": int(cust['Cluster']),
            "Recency (days)": int(cust['Recency']),
            "Frequency (orders)": int(cust['Frequency']),
            "Monetary (annual spend)": float(cust['Monetary'])
        })

        st.markdown("**Category spend breakdown**")
        spend_df = pd.DataFrame({
            "Category": ["Electronics","Fashion","Grocery","Lifestyle"],
            "Amount": [cust['Electronics_Spend'], cust['Fashion_Spend'], cust['Grocery_Spend'], cust['Lifestyle_Spend']]
        })
        fig_spend = px.pie(spend_df, names='Category', values='Amount', title="Category Spend %", template='plotly_dark')
        st.plotly_chart(fig_spend, use_container_width=True)

    st.markdown("---")

    # 4) Show top customers by Monetary in this cluster
    st.markdown(f"**Top 10 customers in Cluster {chosen_cluster} by Monetary**")
    top10 = cluster_df.sort_values(by='Monetary', ascending=False).head(10)[['CustomerID','Age','Gender','Region','Frequency','Monetary']]
    st.table(top10)

# ----------------- end replacement for tab2 -----------------


with tab3:
    st.subheader("Predict new customer's segment + CLTV & churn (quick)")
    st.markdown("Enter new customer values (Age, Recency, Frequency, Monetary). Predictions use the current clustering model and simple CLTV heuristic.")

    colA, colB = st.columns(2)
    with colA:
        age_i = st.number_input("Age", min_value=18, max_value=90, value=30)
        recency_i = st.number_input("Recency (days)", min_value=0, max_value=2000, value=60)
    with colB:
        freq_i = st.number_input("Frequency (per year)", min_value=0, max_value=200, value=6)
        monetary_i = st.number_input("Monetary (annual spend)", min_value=0, max_value=1000000, value=5000)

    if st.button("Predict for this customer"):
        new = np.array([[age_i, recency_i, freq_i, monetary_i]])
        new_scaled = scaler.transform(new)
        # predicted cluster
        try:
            pred_cluster = kmeans.predict(new_scaled)[0]
        except Exception:
            # if kmeans not in session (rare), retrain
            kmeans, scaler2, _ = train_kmeans(df, n_clusters=5)
            pred_cluster = kmeans.predict(new_scaled)[0]

        # Quick CLTV and churn prediction (heuristic)
        churn_proba_new = min(1.0, recency_i / 365.0)
        avg_order_val_new = monetary_i / (freq_i if freq_i>0 else 1)
        exp_purchases = freq_i
        cltv_new = avg_order_val_new * exp_purchases * (1 - churn_proba_new)

        st.success(f"Predicted Cluster: {pred_cluster}")
        st.write("Cluster profile (means):")
        # st.write(cluster_summary.loc[pred_cluster])
        st.markdown("---")
        st.write(f"Estimated churn probability (heuristic): {churn_proba_new:.2f}")
        st.write(f"Estimated CLTV (simple): ₹{cltv_new:,.0f}")

# ---- Footer / Notes ----
st.markdown("---")
st.caption(f"Data file used: {st.session_state.get('_data_file', 'unknown')}  •  App created for project showcase")

