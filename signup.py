import streamlit as st
import sqlite3
from hashlib import sha256

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    conn.commit()
    conn.close()

def add_user(username, password, role):
    hashed_password = sha256(password.encode()).hexdigest()
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
              (username, hashed_password, role))
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    hashed_password = sha256(password.encode()).hexdigest()
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=? AND password=?", 
              (username, hashed_password))
    result = c.fetchone()
    conn.close()
    return result

# Initialize database
init_db()

# Streamlit app
st.title("Employee and Manager Login System")

menu = st.sidebar.selectbox("Menu", ["Login", "Sign Up"])

if menu == "Sign Up":
    st.subheader("Create an Account")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["Employee", "Manager"])
    
    if st.button("Sign Up"):
        if username and password:
            try:
                add_user(username, password, role)
                st.success(f"Account created successfully for {role}!")
            except sqlite3.IntegrityError:
                st.error("Username already exists. Please choose a different username.")
        else:
            st.error("Please fill all the fields.")

elif menu == "Login":
    st.subheader("Login to Your Account")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        user_role = authenticate_user(username, password)
        if user_role:
            role = user_role[0]
            st.success(f"Welcome {role} {username}!")
            if role == "Manager":
                st.info("Managers can access additional features.")
            else:
                st.info("Employees can access standard features.")
        else:
            st.error("Invalid credentials. Please try again.")