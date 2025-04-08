# AI-Powered Workforce Scheduling System
# Using Google OR-Tools for constraint optimization and XGBoost for demand prediction

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Part 1: XGBoost model to predict workload demand
# ------------------------------------------------

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
    print(f"Model RMSE: {rmse:.4f}")
    return model

# Part 2: Google OR-Tools for constraint-based scheduling
# ------------------------------------------------------

def create_schedule(demand_forecast, num_employees, max_shifts_per_employee, min_shifts_per_employee,
                   employee_availability, employee_skills, num_days, num_shifts_per_day):
    
    model = cp_model.CpModel()
    
    # Define variables
    shifts = {}
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts_per_day):
                shifts[(e, d, s)] = model.NewBoolVar(f'shift_e{e}_d{d}_s{s}')
    
    # Constraint 1: Each employee works at most one shift per day
    for e in range(num_employees):
        for d in range(num_days):
            model.Add(sum(shifts[(e, d, s)] for s in range(num_shifts_per_day)) <= 1)
    
    # Constraint 2: Each employee works between min and max shifts
    for e in range(num_employees):
        total_shifts = sum(shifts[(e, d, s)] 
                         for d in range(num_days) 
                         for s in range(num_shifts_per_day))
        model.Add(total_shifts >= min_shifts_per_employee)
        model.Add(total_shifts <= max_shifts_per_employee)
    
    # Constraint 3: Demand constraints with some tolerance
    for d in range(num_days):
        for s in range(num_shifts_per_day):
            staffing_level = sum(shifts[(e, d, s)] for e in range(num_employees))
            model.Add(staffing_level >= max(1, demand_forecast[d][s] - 1))  # Allow 1 person less than forecast
    
    # Constraint 4: Respect employee availability
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts_per_day):
                if employee_availability[e][d * num_shifts_per_day + s] == 0:
                    model.Add(shifts[(e, d, s)] == 0)
    
    # Objective: Maximize skill and minimize overstaffing
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
    
    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Extract solution
    schedule = np.zeros((num_employees, num_days, num_shifts_per_day), dtype=int)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for e in range(num_employees):
            for d in range(num_days):
                for s in range(num_shifts_per_day):
                    if solver.Value(shifts[(e, d, s)]) == 1:
                        schedule[e, d, s] = 1
        print(f"Total objective value: {solver.ObjectiveValue()}")
    else:
        print("No solution found.")
    
    return schedule, status

# Part 3: Integration and workflow
# -------------------------------

def run_workforce_planning(num_employees=10, num_days=7, num_shifts_per_day=3):
    print("Step 1: Training demand prediction model...")
    demand_model = train_demand_model()
    
    print("\nStep 2: Generating demand forecast...")
    future_features = np.random.rand(num_days * num_shifts_per_day, 5)
    raw_demand_forecast = demand_model.predict(future_features)
    demand_forecast = raw_demand_forecast.reshape(num_days, num_shifts_per_day)
    demand_forecast = np.round(demand_forecast).astype(int)
    
    print("Predicted demand (employees needed):")
    for d in range(num_days):
        print(f"Day {d+1}: {demand_forecast[d]}")
    
    print("\nStep 3: Setting up scheduling constraints...")
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
    
    print("\nStep 4: Solving scheduling problem...")
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
    
    if status == cp_model.OPTIMAL:
        print("Found optimal solution!")
    elif status == cp_model.FEASIBLE:
        print("Found a feasible solution, but it may not be optimal.")
    else:
        print("Could not find a solution. Try relaxing some constraints.")
        return np.zeros((num_employees, num_days, num_shifts_per_day), dtype=int), demand_forecast
    
    print("\nStep 5: Visualizing the schedule...")
    visualize_schedule(schedule, demand_forecast, num_employees, num_days, num_shifts_per_day)
    
    return schedule, demand_forecast

# Part 4: Visualizations
# ----------------------

def visualize_schedule(schedule, demand_forecast, num_employees, num_days, num_shifts_per_day):
    shift_names = ['Morning', 'Afternoon', 'Night']
    schedule_df = pd.DataFrame(columns=['Employee', 'Day', 'Shift'])
    
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts_per_day):
                if schedule[e, d, s] == 1:
                    new_row = {'Employee': f'Employee {e+1}', 
                               'Day': f'Day {d+1}',
                               'Shift': shift_names[s]}
                    schedule_df = pd.concat([schedule_df, pd.DataFrame([new_row])], ignore_index=True)
    
    print("\nGenerated Schedule:")
    for d in range(num_days):
        print(f"\nDay {d+1}:")
        for s in range(num_shifts_per_day):
            employees_working = [e+1 for e in range(num_employees) if schedule[e, d, s] == 1]
            print(f"  {shift_names[s]} shift: {employees_working} (Demand: {demand_forecast[d][s]})")
    
    total_demand = demand_forecast.sum()
    total_staffing = schedule.sum()
    
    print(f"\nTotal staffing need: {total_demand}")
    print(f"Total shifts scheduled: {total_staffing}")
    
    if total_staffing >= total_demand:
        print(f"All demand covered with {total_staffing - total_demand} extra shifts.")
    else:
        print(f"Warning: Understaffed by {total_demand - total_staffing} shifts.")
    
    # Heatmap visualization
    plot_schedule_heatmap(schedule)

def plot_schedule_heatmap(schedule):
    flattened = schedule.sum(axis=2)  # Sum across shifts to get total per employee/day
    plt.figure(figsize=(10, 6))
    sns.heatmap(flattened, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Shifts Assigned'})
    plt.xlabel("Day")
    plt.ylabel("Employee")
    plt.title("Employee Shift Assignment Heatmap")
    plt.show()

# Run the entire workforce planning workflow
if __name__ == "__main__":
    schedule, demand = run_workforce_planning(num_employees=15, num_days=7, num_shifts_per_day=3)
