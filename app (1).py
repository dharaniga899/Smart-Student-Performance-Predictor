# ============================================================
# STUDENT PERFORMANCE PREDICTOR - Hugging Face Spaces version
# This app.py runs permanently on Hugging Face Spaces (free tier).
# No Colab, no expiring links, no runtime disconnects.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
import gradio as gr

# ---------------------------------------------------------
# 1. LOAD DATA
#    student_performance_prediction.csv must be uploaded
#    into the same Space (see instructions below).
# ---------------------------------------------------------
df = pd.read_csv("student_performance_prediction.csv")

# ---------------------------------------------------------
# 2. CLEAN DATA
# ---------------------------------------------------------
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.drop_duplicates(inplace=True)

# Keep human-readable label mappings before encoding
activity_map = {"No": 0, "Yes": 1}
education_map = {"Associate": 0, "Bachelor": 1, "Doctorate": 2,
                  "High School": 3, "Master": 4}

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------
df['Study Attendance Score'] = df['Study Hours per Week'] * df['Attendance Rate']
df['Future Grade'] = (
    df['Previous Grades']
    + (df['Study Hours per Week'] * 0.8)
    + (df['Attendance Rate'] * 0.2)
    + np.random.normal(0, 5, len(df))
)

# ---------------------------------------------------------
# 4. TRAIN: Linear Regression -> Future Grade
# ---------------------------------------------------------
X_reg = df.drop(['Student ID', 'Future Grade', 'Passed'], axis=1)
y_reg = df['Future Grade']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)

lr = LinearRegression()
lr.fit(X_train_reg_scaled, y_train_reg)

# ---------------------------------------------------------
# 5. TRAIN: Logistic Regression -> Pass / Fail
# ---------------------------------------------------------
X_clf = df.drop(['Student ID', 'Passed', 'Future Grade'], axis=1)
y_clf = df['Passed']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)

log_model = LogisticRegression()
log_model.fit(X_train_clf_scaled, y_train_clf)

print("Models trained successfully.")

# ---------------------------------------------------------
# 6. PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_performance(study_hours, attendance, previous_grades,
                         activities, parent_education):

    activities_enc = activity_map[activities]
    education_enc = education_map[parent_education]
    study_attendance_score = study_hours * attendance

    sample = pd.DataFrame({
        'Study Hours per Week': [study_hours],
        'Attendance Rate': [attendance],
        'Previous Grades': [previous_grades],
        'Participation in Extracurricular Activities': [activities_enc],
        'Parent Education Level': [education_enc],
        'Study Attendance Score': [study_attendance_score]
    })

    sample_reg_scaled = scaler_reg.transform(sample)
    sample_clf_scaled = scaler_clf.transform(sample)

    future_grade = lr.predict(sample_reg_scaled)[0]
    pass_prediction = log_model.predict(sample_clf_scaled)[0]
    pass_probability = log_model.predict_proba(sample_clf_scaled)[0][1] * 100

    result = "PASS" if pass_prediction == 1 else "FAIL"

    recommendations = []
    if study_hours < 5:
        recommendations.append("- Increase weekly study hours.")
    if attendance < 75:
        recommendations.append("- Improve class attendance.")
    if previous_grades < 50:
        recommendations.append("- Needs academic support / tutoring.")
    if not recommendations:
        recommendations.append("- Keep up the good work, no major concerns!")

    report = f"""
## Student Performance Report

**Predicted Future Grade:** {future_grade:.2f}
**Predicted Result:** {result}
**Probability of Passing:** {pass_probability:.2f}%

### Recommendations
{chr(10).join(recommendations)}
"""
    return report

# ---------------------------------------------------------
# 7. GRADIO FRONTEND
# ---------------------------------------------------------
with gr.Blocks(title="Smart Student Performance Predictor") as demo:
    gr.Markdown("# Smart Student Performance Predictor")
    gr.Markdown("Fill in the details below to predict a student's future grade and pass/fail outcome.")

    with gr.Row():
        with gr.Column():
            study_hours = gr.Slider(0, 40, value=10, step=0.5, label="Study Hours per Week")
            attendance = gr.Slider(0, 100, value=75, step=0.5, label="Attendance Rate (%)")
            previous_grades = gr.Slider(0, 100, value=65, step=0.5, label="Previous Grades")
            activities = gr.Radio(["Yes", "No"], value="Yes", label="Participation in Extracurricular Activities")
            parent_education = gr.Dropdown(
                ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
                value="Bachelor", label="Parent Education Level"
            )
            predict_btn = gr.Button("Predict", variant="primary")

        with gr.Column():
            output = gr.Markdown(label="Result")

    predict_btn.click(
        fn=predict_performance,
        inputs=[study_hours, attendance, previous_grades, activities, parent_education],
        outputs=output
    )

# NOTE: no share=True needed here - Hugging Face Spaces hosts it permanently.
if __name__ == "__main__":
    demo.launch()
