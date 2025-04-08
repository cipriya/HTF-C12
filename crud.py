from sqlalchemy.orm import Session
from models import Employee
from schemas import EmployeeCreate

def get_all_employees(db: Session):
    return db.query(Employee).all()

def create_employee(db: Session, employee: EmployeeCreate):
    db_employee = Employee(**employee.dict())
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    return db_employee
