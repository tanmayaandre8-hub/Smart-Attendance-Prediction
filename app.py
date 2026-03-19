import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------- LOAD MODELS ----------------
lin_model = pickle.load(open("linear.pkl", "rb"))
rf_model = pickle.load(open("rf.pkl", "rb"))
log_model = pickle.load(open("logistic.pkl", "rb"))

le_subject = pickle.load(open("le_subject.pkl", "rb"))
le_day = pickle.load(open("le_day.pkl", "rb"))
le_time = pickle.load(open("le_time.pkl", "rb"))

df = pd.read_csv("attendance_data.csv")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Attendance Dashboard", layout="wide")

st.title("🎓 Smart Attendance Prediction System")
st.markdown("### 📊 Analyze & Predict Student Attendance")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest", "Logistic Regression", "Compare All"]
)

st.sidebar.markdown("---")

# ---------------- VISUALIZATION SECTION ----------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

# Subject-wise attendance
with col1:
    st.markdown("#### Subject-wise Average Attendance")
    fig1, ax1 = plt.subplots()
    df.groupby("subject")["attendance_percentage"].mean().plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Attendance %")
    st.pyplot(fig1)

# Difficulty vs attendance
with col2:
    st.markdown("#### Difficulty vs Attendance")
    fig2, ax2 = plt.subplots()
    df.groupby("difficulty")["attendance_percentage"].mean().plot(marker="o", ax=ax2)
    ax2.set_ylabel("Attendance %")
    st.pyplot(fig2)

# -------- EXTRA VISUALIZATION --------
col3, col4 = st.columns(2)

# Day-wise attendance
with col3:
    st.markdown("#### Day-wise Attendance")
    fig3, ax3 = plt.subplots()
    df.groupby("day")["attendance_percentage"].mean().plot(kind="bar", ax=ax3)
    st.pyplot(fig3)

# Time slot analysis
with col4:
    st.markdown("#### Time Slot Impact")
    fig4, ax4 = plt.subplots()
    df.groupby("time_slot")["attendance_percentage"].mean().plot(kind="bar", ax=ax4)
    st.pyplot(fig4)

# ---------------- INPUT SECTION ----------------
st.subheader("🔮 Predict Attendance")

col1, col2, col3 = st.columns(3)

with col1:
    subject = st.selectbox("Subject", le_subject.classes_)
    day = st.selectbox("Day", le_day.classes_)

with col2:
    time = st.selectbox("Time Slot", le_time.classes_)
    difficulty = st.slider("Difficulty", 1, 5)

with col3:
    past = st.slider("Past Attendance (%)", 40, 100)

# Prepare input
input_data = pd.DataFrame([[
    le_subject.transform([subject])[0],
    le_day.transform([day])[0],
    le_time.transform([time])[0],
    difficulty,
    past
]], columns=["subject", "day", "time_slot", "difficulty", "past_attendance"])

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict"):

    st.subheader("📊 Prediction Results")

    # -------- SINGLE MODEL --------
    if model_choice == "Linear Regression":
        pred = lin_model.predict(input_data)[0]
        st.metric("Linear Regression Prediction", f"{pred:.2f}%")

    elif model_choice == "Random Forest":
        pred = rf_model.predict(input_data)[0]
        st.metric("Random Forest Prediction", f"{pred:.2f}%")

    elif model_choice == "Logistic Regression":
        pred = log_model.predict(input_data)[0]
        result = "High Attendance ✅" if pred == 1 else "Low Attendance ❌"
        st.metric("Logistic Result", result)

    # -------- COMPARE ALL --------
    else:
        lin_pred = lin_model.predict(input_data)[0]
        rf_pred = rf_model.predict(input_data)[0]
        log_pred = log_model.predict(input_data)[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Linear", f"{lin_pred:.2f}%")
        col2.metric("Random Forest", f"{rf_pred:.2f}%")
        col3.metric("Logistic", "High" if log_pred==1 else "Low")

        # Comparison chart
        st.subheader("📊 Model Comparison")

        fig5, ax5 = plt.subplots()
        ax5.bar(
            ["Linear", "Random Forest"],
            [lin_pred, rf_pred]
        )
        ax5.set_ylabel("Attendance %")
        st.pyplot(fig5)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("👨‍💻 Developed for University Attendance Analysis Project")