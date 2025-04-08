
import streamlit as st
from employee_dashboard import render_employee_dashboard
from manager_dashboard import render_manager_dashboard

st.set_page_config(page_title="Unified Dashboard", layout="wide")

st.sidebar.title("Dashboard Selector")
dashboard = st.sidebar.radio("Choose view:", ["Manager", "Employee"])

if dashboard == "Manager":
    render_manager_dashboard()
elif dashboard == "Employee":
    render_employee_dashboard()
