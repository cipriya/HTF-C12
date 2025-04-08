import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

from ortools.sat.python import cp_model
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def show():
    st.title("👷 Employee Dashboard")
    st.markdown(f"Welcome, **{st.session_state.username}**!")

    if st.button("🔓 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.experimental_rerun()

    
    st.info("This is the employee dashboard.")

##########################################
# Part 1: Workforce Scheduling Functions #
##########################################

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    X = np.random.rand(num_samples, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] + X[:, 4] + np.random.normal(0, 0.5, num_samples)
    y = np.maximum(y, 1)
    y = np.round(y).astype(int)
    return X, y

def train_demand_model():
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"Demand Model RMSE: {rmse:.4f}")
    return model

def create_schedule(demand_forecast, num_employees, max_shifts_per_employee, min_shifts_per_employee,
                    employee_availability, employee_skills, num_days, num_shifts_per_day):
    model = cp_model.CpModel()
    shifts = {}
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts_per_day):
                shifts[(e, d, s)] = model.NewBoolVar(f'shift_e{e}_d{d}_s{s}')
    
    for e in range(num_employees):
        for d in range(num_days):
            model.Add(sum(shifts[(e, d, s)] for s in range(num_shifts_per_day)) <= 1)
    
    for e in range(num_employees):
        total_shifts = sum(shifts[(e, d, s)] for d in range(num_days) for s in range(num_shifts_per_day))
        model.Add(total_shifts >= min_shifts_per_employee)
        model.Add(total_shifts <= max_shifts_per_employee)
    
    for d in range(num_days):
        for s in range(num_shifts_per_day):
            staffing_level = sum(shifts[(e, d, s)] for e in range(num_employees))
            model.Add(staffing_level >= max(1, demand_forecast[d][s] - 1))
    
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts_per_day):
                if employee_availability[e][d * num_shifts_per_day + s] == 0:
                    model.Add(shifts[(e, d, s)] == 0)
    
    objective_terms = []
    skill_weight = 10
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts_per_day):
                skill_value = employee_skills[e][s]
                objective_terms.append(skill_weight * skill_value * shifts[(e, d, s)])
    
    overstaffing_penalty = 5
    for d in range(num_days):
        for s in range(num_shifts_per_day):
            staffing_level = sum(shifts[(e, d, s)] for e in range(num_employees))
            excess = model.NewIntVar(0, num_employees, f'excess_d{d}_s{s}')
            model.Add(excess >= staffing_level - demand_forecast[d][s])
            objective_terms.append(-overstaffing_penalty * excess)
    
    model.Maximize(sum(objective_terms))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    schedule = np.zeros((num_employees, num_days, num_shifts_per_day), dtype=int)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for e in range(num_employees):
            for d in range(num_days):
                for s in range(num_shifts_per_day):
                    if solver.Value(shifts[(e, d, s)]) == 1:
                        schedule[e, d, s] = 1
        st.write(f"Total objective value: {solver.ObjectiveValue()}")
    else:
        st.error("No solution found. Try relaxing some constraints.")
    
    return schedule, status

def visualize_schedule(schedule, demand_forecast, num_employees, num_days, num_shifts_per_day):
    shift_names = ['Morning', 'Afternoon', 'Night'][:num_shifts_per_day]
    st.write("### Generated Schedule")
    for d in range(num_days):
        st.write(f"Day {d+1}:")
        for s in range(num_shifts_per_day):
            employees_working = [e+1 for e in range(num_employees) if schedule[e, d, s] == 1]
            st.write(f"- {shift_names[s]} shift: Employees {employees_working} (Demand: {demand_forecast[d][s]})")
    total_demand = demand_forecast.sum()
    total_staffing = schedule.sum()
    st.write(f"Total staffing need: {total_demand}")
    st.write(f"Total shifts scheduled: {total_staffing}")
    if total_staffing >= total_demand:
        st.success(f"All demand covered with {total_staffing - total_demand} extra shifts.")
    else:
        st.warning(f"Warning: Understaffed by {total_demand - total_staffing} shifts.")

def run_workforce_planning(num_employees=10, num_days=7, num_shifts_per_day=3):
    st.write("#### Step 1: Training demand prediction model...")
    demand_model = train_demand_model()
    
    st.write("#### Step 2: Generating demand forecast...")
    future_features = np.random.rand(num_days * num_shifts_per_day, 5)
    raw_demand_forecast = demand_model.predict(future_features)
    demand_forecast = raw_demand_forecast.reshape(num_days, num_shifts_per_day)
    demand_forecast = np.round(demand_forecast).astype(int)
    
    st.write("Predicted Demand (employees needed):")
    for d in range(num_days):
        st.write(f"- Day {d+1}: {demand_forecast[d]}")
    
    st.write("#### Step 3: Setting up scheduling constraints...")
    np.random.seed(42)
    availability_prob = 0.8
    employee_availability = np.random.choice(
        [0, 1], 
        size=(num_employees, num_days * num_shifts_per_day),
        p=[1-availability_prob, availability_prob]
    )
    employee_skills = np.random.randint(1, 11, size=(num_employees, num_shifts_per_day))
    
    max_shifts_per_employee = 7
    min_shifts_per_employee = 1
    
    st.write("#### Step 4: Solving scheduling problem...")
    schedule, status = create_schedule(
        demand_forecast, 
        num_employees, 
        max_shifts_per_employee,
        min_shifts_per_employee,
        employee_availability, 
        employee_skills,
        num_days, 
        num_shifts_per_day
    )
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        st.success("Schedule generated!")
        visualize_schedule(schedule, demand_forecast, num_employees, num_days, num_shifts_per_day)
    return schedule, demand_forecast

##########################################
# Part 2: Database Setup for Leave & Tasks
##########################################

conn = sqlite3.connect('leave_system.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS leaves (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee TEXT,
        reason TEXT,
        status TEXT DEFAULT 'Pending'
    )
''')
conn.commit()

##########################################
# Part 3: Streamlit App Navigation       #
##########################################

st.set_page_config(page_title="Workforce Scheduler & Leave Manager", layout="wide")
st.title("🤖 Workforce Scheduling & Leave Management System")

app_mode = st.sidebar.selectbox("Choose App Section", 
                                ["Workforce Scheduling", "Leave Management", "Employee Portal"])

##########################################
# Workforce Scheduling Page              #
##########################################

if app_mode == "Workforce Scheduling":
    st.header("AI-Powered Workforce Scheduling")
    num_employees = st.sidebar.slider("Number of Employees", min_value=5, max_value=50, value=15)
    num_days = st.sidebar.slider("Number of Days", min_value=1, max_value=14, value=7)
    num_shifts = st.sidebar.selectbox("Shifts per Day", options=[1, 2, 3], index=2)
    
    if st.button("Run Workforce Scheduler"):
        run_workforce_planning(num_employees=num_employees, num_days=num_days, num_shifts_per_day=num_shifts)
        st.download_button(
            label="Download Sample Schedule CSV",
            data="This is a placeholder. In a full implementation, convert schedule data to CSV.",
            file_name="schedule.csv",
            mime="text/csv"
        )

##########################################
# Leave Management Page                  #
##########################################

elif app_mode == "Leave Management":
    st.header("Leave Management System")
    user_type = st.sidebar.radio("Select Role", ["Employee", "Manager"])
    
    if user_type == "Employee":
        st.subheader("Employee Leave Request")
        employee_name = st.text_input("Enter your name")
        leave_reason = st.text_area("Enter reason for leave")
        
        if st.button("Submit Leave Request"):
            if employee_name and leave_reason:
                cursor.execute("INSERT INTO leaves (employee, reason) VALUES (?, ?)", 
                               (employee_name, leave_reason))
                conn.commit()
                st.success("Leave request submitted!")
            else:
                st.warning("Please fill in all fields.")
        
        st.subheader("Check Leave Status")
        if st.button("Check My Leave Status"):
            if employee_name:
                cursor.execute("SELECT reason, status FROM leaves WHERE employee=?", (employee_name,))
                results = cursor.fetchall()
                if results:
                    for row in results:
                        st.info(f"Reason: {row[0]} | Status: {row[1]}")
                else:
                    st.warning("No leave requests found.")
            else:
                st.warning("Please enter your name to check status.")
    
    elif user_type == "Manager":
        st.subheader("Manager Panel - Approve/Reject Leaves")
        cursor.execute("SELECT id, employee, reason, status FROM leaves WHERE status='Pending'")
        pending_leaves = cursor.fetchall()
        
        if pending_leaves:
            for leave in pending_leaves:
                with st.expander(f"Request from {leave[1]} (ID: {leave[0]})"):
                    st.write(f"Reason: {leave[2]}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Approve ID {leave[0]}", key=f"app_{leave[0]}"):
                            cursor.execute("UPDATE leaves SET status='Approved' WHERE id=?", (leave[0],))
                            conn.commit()
                            st.success("Leave approved!")
                    with col2:
                        if st.button(f"Reject ID {leave[0]}", key=f"rej_{leave[0]}"):
                            cursor.execute("UPDATE leaves SET status='Rejected' WHERE id=?", (leave[0],))
                            conn.commit()
                            st.error("Leave rejected.")
        else:
            st.info("No pending leave requests.")

##########################################
# Employee Portal Page                   #
##########################################

elif app_mode == "Employee Portal":
    st.header("🧑‍💼 Employee Portal")

    employee_name = st.text_input("Enter your name")

    st.subheader("📅 My Weekly Tasks")

    task_data = {
        "Monday": "Inventory check",
        "Tuesday": "Team meeting",
        "Wednesday": "Client follow-up",
        "Thursday": "Documentation update",
        "Friday": "System testing",
        "Saturday": "Training",
        "Sunday": "Off"
    }

    task_status = {}

    if employee_name:
        for day, task in task_data.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{day}:** {task}")
            with col2:
                task_status[day] = st.checkbox(f"Done", key=f"{day}_done")

        if st.button("Submit Task Updates"):
            st.success("Task status updated and sent to manager.")
    else:
        st.warning("Please enter your name to view and update tasks.")

    st.subheader("📝 Submit a Leave Request")
    leave_reason = st.text_area("Reason for leave")

    if st.button("Submit Leave"):
        if employee_name and leave_reason:
            cursor.execute("INSERT INTO leaves (employee, reason) VALUES (?, ?)", 
                           (employee_name, leave_reason))
            conn.commit()
            st.success("Leave request submitted!")
        else:
            st.warning("Please fill in all fields.")

    st.subheader("📋 My Leave Requests")
    if st.button("Refresh My Leave Requests"):
        if employee_name:
            cursor.execute("SELECT reason, status FROM leaves WHERE employee=?", (employee_name,))
            results = cursor.fetchall()
            if results:
                for row in results:
                    st.info(f"Reason: {row[0]} | Status: {row[1]}")
            else:
                st.warning("No leave requests found.")
        else:
            st.warning("Please enter your name.")