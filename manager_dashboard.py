
import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title="Workforce Optimisation", layout="wide")

import pandas as pd
from backend import run_workforce_planning  # Your AI logic

# ---------- Sidebar Navigation ----------

st.sidebar.markdown("---")
section = st.sidebar.radio("Navigation", ["Home", "Schedules", "Leave Requests", "Employees", "Time Tracking", "Settings"])

st.sidebar.title("Team Members")
team_members = [
    {"name": "Alex Johnson", "role": "Senior Developer", "dept": "Engineering"},
    {"name": "Sarah Williams", "role": "UX Designer", "dept": "Design"},
    {"name": "Michael Chen", "role": "Project Manager", "dept": "Management"},
    {"name": "Emily Rodriguez", "role": "Data Analyst", "dept": "Analytics"},
    {"name": "David Kim", "role": "DevOps Engineer", "dept": "Engineering"}
]
for member in team_members:
    st.sidebar.markdown(f"**{member['name']}** — {member['role']} ({member['dept']})")



# ---------- Session State ----------
if "calendar_events" not in st.session_state:
    st.session_state.calendar_events = []

# ---------- Home ----------
if section == "Home":
    st.title("Workforce Optimisation")
    st.write("Manage leaves, employee schedules, and team efficiency with ease.")

    st.markdown("### Dashboard Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Leave Requests", "7", "+2")
    col2.metric("Pending Approvals", "3", "-1")
    col3.metric("Team Utilization", "78%", "+5.2%")

# ---------- Schedules ----------
elif section == "Schedules":
    st.title("AI-Powered Workforce Schedules")

    schedule_type = st.radio("Select schedule type:", ["Daily", "Weekly", "Monthly"], horizontal=True)

    if st.button("Generate AI Schedule"):
        st.success("Schedule generated! View below")

        num_employees = 15
        num_days = 1 if schedule_type == "Daily" else (7 if schedule_type == "Weekly" else 30)
        num_shifts = 3
        shift_names = ['Morning', 'Afternoon', 'Night']

        schedule, demand = run_workforce_planning(
            num_employees=num_employees,
            num_days=num_days,
            num_shifts_per_day=num_shifts
        )

        events = []
        for e in range(num_employees):
            for d in range(num_days):
                for s in range(num_shifts):
                    if schedule[e, d, s] == 1:
                        start_time = (pd.Timestamp.now().normalize() + pd.Timedelta(days=d) + pd.Timedelta(hours=8 + s * 4)).isoformat()
                        end_time = (pd.Timestamp.now().normalize() + pd.Timedelta(days=d) + pd.Timedelta(hours=12 + s * 4)).isoformat()
                        events.append({
                            "title": f"Employee {e+1} - {shift_names[s]}",
                            "start": start_time,
                            "end": end_time,
                        })

        st.session_state.calendar_events = events

    if st.session_state.calendar_events:
        st.markdown("### AI-Generated Schedule Table")
        schedule_data = []
        for event in st.session_state.calendar_events:
            schedule_data.append({
                "Employee": event["title"].split(" - ")[0],
                "Shift": event["title"].split(" - ")[1],
                "Start": pd.to_datetime(event["start"]).strftime('%Y-%m-%d %H:%M'),
                "End": pd.to_datetime(event["end"]).strftime('%Y-%m-%d %H:%M')
            })
        df_schedule = pd.DataFrame(schedule_data)
        st.dataframe(df_schedule)

# ---------- Leave Requests ----------
elif section == "Leave Requests":
    st.title("Leave Requests")
    leave_requests = [
        {"name": "Alex Johnson", "type": "Vacation", "dates": "Apr 10 - Apr 14", "reason": "Family trip", "status": "Pending"},
        {"name": "Sarah Williams", "type": "Sick Leave", "dates": "Apr 8 - Apr 9", "reason": "Flu recovery", "status": "Pending"},
        {"name": "Emily Rodriguez", "type": "Work from Home", "dates": "Apr 11", "reason": "House maintenance", "status": "Pending"},
    ]
    for i, req in enumerate(leave_requests):
        with st.container():
            st.subheader(f"{req['name']} - {req['type']}")
            st.markdown(f"*Dates:* {req['dates']}")
            st.markdown(f"*Reason:* {req['reason']}")
            st.markdown(f"*Status:* {req['status']}")
            col1, col2 = st.columns(2)
            if col1.button(f"Approve {i}"):
                st.success(f"{req['name']}'s request approved.")
            if col2.button(f"Reject {i}"):
                st.error(f"{req['name']}'s request rejected.")
            st.markdown("---")

# ---------- Employees ----------
elif section == "Employees":
    st.title("Team Members")
    for emp in team_members:
        st.markdown(f"**{emp['name']}** — {emp['role']} ({emp['dept']})")
        st.markdown("---")

# ---------- Time Tracking ----------
elif section == "Time Tracking":
    st.title("Time Tracking")
    st.markdown("#### Weekly Work Hours Summary")
    time_data = [
        {"name": "Alex Johnson", "mon": 8, "tue": 8, "wed": 7, "thu": 8, "fri": 7, "productivity": "95%"},
        {"name": "Sarah Williams", "mon": 7, "tue": 8, "wed": 8, "thu": 8, "fri": 6, "productivity": "89%"},
        {"name": "Michael Chen", "mon": 8, "tue": 7, "wed": 8, "thu": 8, "fri": 8, "productivity": "92%"},
        {"name": "Emily Rodriguez", "mon": 8, "tue": 8, "wed": 6, "thu": 7, "fri": 8, "productivity": "85%"},
    ]
    headers = ["Name", "Mon", "Tue", "Wed", "Thu", "Fri", "Total", "Productivity"]
    rows = []
    for emp in time_data:
        total = emp["mon"] + emp["tue"] + emp["wed"] + emp["thu"] + emp["fri"]
        rows.append([emp["name"], emp["mon"], emp["tue"], emp["wed"], emp["thu"], emp["fri"], total, emp["productivity"]])
    st.table([headers] + rows)

# ---------- Settings ----------
elif section == "Settings":
    st.title("Settings")
    st.subheader("Profile")
    manager_name = st.text_input("Name", value="Manager John Doe")
    email = st.text_input("Email", value="manager@example.com")
    department = st.selectbox("Department", ["HR", "Operations", "Engineering", "Sales"], index=1)

    st.subheader("Preferences")
    notifications = st.checkbox("Enable Email Notifications", value=True)
    dark_mode = st.checkbox("Enable Dark Mode (Experimental)")

    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
    st.caption("Settings are stored locally for now. Backend integration recommended.")
