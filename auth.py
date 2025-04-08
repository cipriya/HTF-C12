import sqlite3
from hashlib import sha256

def authenticate_user(username: str, password: str):
    hashed_password = sha256(password.encode()).hexdigest()
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=? AND password=?", 
              (username, hashed_password))
    result = c.fetchone()
    conn.close()

    if result:
        return {"username": username, "role": result[0]}
    return None
