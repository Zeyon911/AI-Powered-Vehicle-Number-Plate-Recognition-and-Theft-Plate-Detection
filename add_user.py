import json
import bcrypt
import os

USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def add_user(username, password):
    users = load_users()
    
    if username in users:
        print(f"❌ User '{username}' already exists.")
        return
    
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = hashed
    save_users(users)
    print(f"✅ User '{username}' added successfully.")

if __name__ == "__main__":
    username = input("Enter new username: ")
    password = input("Enter password: ")
    add_user(username, password)
