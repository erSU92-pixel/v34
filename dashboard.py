import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.set_page_config(page_title="Apex Dashboard", layout="wide")
st.title("📊 Apex Quant - Institutional Dashboard")

try:
    with open("xauusd_apex_report.json", "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sinyal", len(df))
    col2.metric("Sinyal BUY", len(df[df['signal'] == 'BUY']))
    col3.metric("Sinyal SELL", len(df[df['signal'] == 'SELL']))
    
    st.subheader("Riwayat Sinyal & Harga Entry")
    fig = px.line(df, x='timestamp', y='entry', color='signal', markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Log Transaksi")
    st.dataframe(df.sort_values('timestamp', ascending=False), use_container_width=True)
    
except FileNotFoundError:
    st.warning("File xauusd_apex_report.json belum ada. Bot belum mengirim sinyal.")
