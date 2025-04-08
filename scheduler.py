import numpy as np

def run_workforce_planning(num_employees=5, num_days=7, num_shifts_per_day=3):
    # Example dummy logic for scheduling
    schedule = np.zeros((num_employees, num_days, num_shifts_per_day), dtype=int)
    demand = np.random.randint(1, 4, size=(num_days, num_shifts_per_day))

    for d in range(num_days):
        for s in range(num_shifts_per_day):
            assigned = 0
            for e in range(num_employees):
                if assigned < demand[d][s]:
                    schedule[e][d][s] = 1
                    assigned += 1
    return schedule, demand
