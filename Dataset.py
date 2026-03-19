import pandas as pd
import random

subjects = ["Maths", "Physics", "Chemistry", "CS", "English"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
time_slots = ["9-10", "10-11", "11-12", "2-3", "3-4"]

data = []

for _ in range(500):
    subject = random.choice(subjects)
    day = random.choice(days)
    time = random.choice(time_slots)
    difficulty = random.randint(1, 5)
    past_attendance = random.randint(50, 95)

    # more realistic logic
    attendance = past_attendance - difficulty * random.randint(2, 5)
    attendance += random.randint(-5, 5)
    attendance = max(40, min(attendance, 95))

    # classification label (for logistic regression)
    status = 1 if attendance >= 75 else 0

    data.append([
        subject, day, time, difficulty,
        past_attendance, attendance, status
    ])

df = pd.DataFrame(data, columns=[
    "subject", "day", "time_slot",
    "difficulty", "past_attendance",
    "attendance_percentage", "high_attendance"
])

df.to_csv("attendance_data.csv", index=False)
print("Dataset generated!")