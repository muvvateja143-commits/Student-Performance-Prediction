import streamlit as st
import joblib

# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load model
model = joblib.load("student_model.pkl")

# Title
st.title("🎓 Student Performance Prediction")
st.write("Enter the student's exam scores to predict overall academic performance.")

st.divider()

# Student details
st.subheader("📋 Student Details")

math_score = st.number_input(
    "➗ Mathematics Score",
    min_value=0,
    max_value=100,
    value=50
)

reading_score = st.number_input(
    "📖 Reading Score",
    min_value=0,
    max_value=100,
    value=50
)

writing_score = st.number_input(
    "✍️ Writing Score",
    min_value=0,
    max_value=100,
    value=50
)

st.divider()

# Prediction
if st.button("🔮 Predict Performance", use_container_width=True):

    prediction = model.predict(
        [[math_score, reading_score, writing_score]]
    )

    score = prediction[0]

    if score >= 75:
        category = "🌟 Excellent Performance"
    elif score >= 50:
        category = "👍 Good Performance"
    elif score >= 35:
        category = "📚 Average Performance"
    else:
        category = "⚠️ Needs Improvement"

    st.success(f"🎯 Predicted Overall Score: **{score:.2f} / 100**")
    st.info(f"Performance Level: **{category}**")

    st.progress(min(int(score), 100))

st.divider()

st.caption("Machine Learning Project | Student Performance Prediction")