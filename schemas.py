from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

class EmployeeCreate(BaseModel):
    name: str
    email: str
    position: str  # or whatever fields your Employee model has
