"""
================================================================================
STUDENT DROPOUT RISK PREDICTOR - STREAMLIT GUI
================================================================================
Applied Machine Learning - EDGE Project
DUET, Gazipur

This application predicts student dropout risk based on academic and
demographic features.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffbb33;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-low {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 5px solid #ffa000;
        padding: 1.5rem;
        margin-top: 2rem;
        border-radius: 8px;
        color: #856404; /* Darker text for better visibility */
    }
    .disclaimer strong {
        color: #533f03;
    }
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_model():
    try:
        with open('student_dropout_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please upload 'student_dropout_model.pkl'")
        return None

# Load the model
model_artifacts = load_model()

if model_artifacts is None:
    st.stop()

model = model_artifacts['model']
scaler = model_artifacts['scaler']
feature_names = model_artifacts['feature_names']
model_name = model_artifacts['model_name']
performance_metrics = model_artifacts['performance_metrics']
feature_importance = pd.DataFrame(model_artifacts['feature_importance'])

# Header
st.markdown('<div class="main-header">🎓 Student Dropout Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Machine Learning | DUET, Gazipur - EDGE Project</div>', unsafe_allow_html=True)

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown(f"**Algorithm:** {model_name}")
    st.markdown(f"**Accuracy:** {performance_metrics['accuracy']:.2%}")
    st.markdown(f"**F1-Score:** {performance_metrics['f1_score']:.4f}")
    st.markdown(f"**AUC-ROC:** {performance_metrics['auc_roc']:.4f}")

    st.markdown("---")
    st.header("ℹ️ About")
    st.info(
        "This application predicts the likelihood of student dropout "
        "based on academic performance, demographic factors, and "
        "socioeconomic indicators."
    )

    st.markdown("---")
    st.header("🎯 Risk Levels")
    st.markdown("🟢 **Low Risk:** < 30%")
    st.markdown("🟡 **Medium Risk:** 30% - 60%")
    st.markdown("🔴 **High Risk:** > 60%")

    st.markdown("---")
    st.header("📖 Feature Guide")
    with st.expander("Show Category Codes"):
        st.markdown("""
        **Marital Status:**
        1: Single, 2: Married, 3: Widower, 4: Divorced, 5: Facto union, 6: Legally separated

        **Course Codes:**
        - 33: Biofuel Production Technologies
        - 171: Animation and Multimedia Design
        - 8014: Social Service
        - 9003: Agronomy
        - 9070: Communication Design
        - 9085: Veterinary Nursing
        - 9119: Informatics Engineering
        - 9130: Equiniculture
        - 9147: Management
        - 9238: Social Service (evening)
        - 9254: Tourism
        - 9500: Nursing
        - 9556: Oral Hygiene
        - 9670: Advertising and Marketing Management
        - 9773: Journalism and Communication
        - 9853: Basic Education
        - 9991: Management (evening)

        **Qualification Codes:**
        1: Secondary education, 19: Higher education (degree), etc.
        """)

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Prediction Form", "📈 Feature Importance", "📋 Model Performance"])

# TAB 1: Prediction Form
with tab1:
    st.header("Enter Student Information")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Personal Information")
            marital_status = st.selectbox("Marital Status", [1, 2, 3, 4, 5, 6],
                                         help="1=Single, 2=Married, 3=Widower, 4=Divorced, 5=Facto union, 6=Legally separated")
            age_enrollment = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=20,
                                            help="Student age when they started the course. Typical range: 18-30.")
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
                                 help="Biological sex of the student.")
            nacionality = st.number_input("Nationality Code", min_value=1, max_value=100, value=1,
                                         help="Country code (1=Portuguese is most common in this dataset).")
            displaced = st.selectbox("Displaced", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                    help="Is the student living away from their home town?")
            international = st.selectbox("International Student", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                        help="Is the student from a different country?")

        with col2:
            st.subheader("📚 Academic Information")
            application_mode = st.number_input("Application Mode", min_value=1, max_value=60, value=1,
                                              help="Method used to apply (1=1st phase, 17=Transfer, etc.).")
            application_order = st.number_input("Application Order", min_value=0, max_value=9, value=1,
                                               help="Preference order of this course (0-9, where 1 is first choice).")
            course = st.number_input("Course Code", min_value=1, max_value=10000, value=9254,
                                    help="See 'Feature Guide' in sidebar for specific course names.")
            daytime_evening = st.selectbox("Attendance", [0, 1], format_func=lambda x: "Evening" if x == 0 else "Daytime",
                                          help="Whether the student attends classes during the day or evening.")
            prev_qualification = st.number_input("Previous Qualification", min_value=1, max_value=50, value=1,
                                                help="Code for highest prior education level (1=Secondary school).")
            prev_qualification_grade = st.number_input("Prev Qualification Grade (0-200)", min_value=0.0, max_value=200.0, value=120.0,
                                                     help="Final grade from previous school level (Scale 0-200).")
            admission_grade = st.number_input("Admission Grade (0-200)", min_value=0.0, max_value=200.0, value=127.0,
                                            help="Entrance exam score (Scale 0-200).")

        with col3:
            st.subheader("🏠 Family Background")
            mothers_qualification = st.number_input("Mother's Qualification", min_value=1, max_value=50, value=1,
                                                   help="Education code for mother.")
            fathers_qualification = st.number_input("Father's Qualification", min_value=1, max_value=50, value=1,
                                                   help="Education code for father.")
            mothers_occupation = st.number_input("Mother's Occupation", min_value=0, max_value=200, value=1,
                                                help="Professional occupation code for mother.")
            fathers_occupation = st.number_input("Father's Occupation", min_value=0, max_value=200, value=1,
                                                help="Professional occupation code for father.")

        st.markdown("---")

        col4, col5 = st.columns(2)

        with col4:
            st.subheader("📅 1st Semester Performance")
            units_1st_credited = st.number_input("Units Credited (1st Sem)", min_value=0, max_value=30, value=0,
                                                help="Number of credits from previous work/study.")
            units_1st_enrolled = st.number_input("Units Enrolled (1st Sem)", min_value=0, max_value=30, value=6,
                                                help="Number of subjects student registered for.")
            units_1st_evaluations = st.number_input("Evaluations (1st Sem)", min_value=0, max_value=50, value=6,
                                                   help="Number of assessments (exams, tests) completed.")
            units_1st_approved = st.number_input("Units Approved (1st Sem)", min_value=0, max_value=30, value=6,
                                                help="Number of subjects successfully passed.")
            units_1st_grade = st.number_input("Semester Grade (1st Sem) (0-20)", min_value=0.0, max_value=20.0, value=14.0,
                                             help="Average grade score (Scale 0-20).")
            units_1st_without_eval = st.number_input("Units without Eval (1st Sem)", min_value=0, max_value=30, value=0,
                                                    help="Subjects that didn't have a final assessment.")

        with col5:
            st.subheader("📅 2nd Semester Performance")
            units_2nd_credited = st.number_input("Units Credited (2nd Sem)", min_value=0, max_value=30, value=0)
            units_2nd_enrolled = st.number_input("Units Enrolled (2nd Sem)", min_value=0, max_value=30, value=6)
            units_2nd_evaluations = st.number_input("Evaluations (2nd Sem)", min_value=0, max_value=50, value=6)
            units_2nd_approved = st.number_input("Units Approved (2nd Sem)", min_value=0, max_value=30, value=6)
            units_2nd_grade = st.number_input("Semester Grade (2nd Sem) (0-20)", min_value=0.0, max_value=20.0, value=14.0)
            units_2nd_without_eval = st.number_input("Units without Eval (2nd Sem)", min_value=0, max_value=30, value=0)

        st.markdown("---")

        col6, col7, col8 = st.columns(3)

        with col6:
            st.subheader("💰 Financial Status")
            scholarship_holder = st.selectbox("Scholarship Holder", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                             help="Does the student receive financial aid?")
            tuition_fees_updated = st.selectbox("Tuition Fees Up to Date", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                               help="Are all university fees paid up to date?")
            debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                 help="Does the student have outstanding debts?")

        with col7:
            st.subheader("🤝 Special Needs")
            educational_special_needs = st.selectbox("Educational Special Needs", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                                    help="Does the student require specific educational support?")

        with col8:
            st.subheader("📉 Economic Indicators")
            unemployment_rate = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=30.0, value=10.8,
                                               help="National unemployment rate at time of data collection.")
            inflation_rate = st.number_input("Inflation Rate (%)", min_value=-5.0, max_value=10.0, value=1.4,
                                            help="National inflation rate.")
            gdp = st.number_input("GDP Growth", min_value=-10.0, max_value=10.0, value=1.74,
                                 help="Annual GDP growth rate.")

        submitted = st.form_submit_button("🔍 Predict Dropout Risk", use_container_width=True, type="primary")

    if submitted:
        # Create input dictionary matching the exact feature names from training
        input_dict = {}

        # Map all features - use the exact column names from training
        # This is a simplified mapping - you'll need to match ALL features from your training data
        feature_mapping = {
            'Marital status': marital_status,
            'Application mode': application_mode,
            'Application order': application_order,
            'Course': course,
            'Daytime/evening attendance\t': daytime_evening,
            'Previous qualification': prev_qualification,
            'Previous qualification (grade)': prev_qualification_grade,
            'Nacionality': nacionality,
            "Mother's qualification": mothers_qualification,
            "Father's qualification": fathers_qualification,
            "Mother's occupation": mothers_occupation,
            "Father's occupation": fathers_occupation,
            'Admission grade': admission_grade,
            'Displaced': displaced,
            'Educational special needs': educational_special_needs,
            'Debtor': debtor,
            'Tuition fees up to date': tuition_fees_updated,
            'Gender': gender,
            'Scholarship holder': scholarship_holder,
            'Age at enrollment': age_enrollment,
            'International': international,
            'Curricular units 1st sem (credited)': units_1st_credited,
            'Curricular units 1st sem (enrolled)': units_1st_enrolled,
            'Curricular units 1st sem (evaluations)': units_1st_evaluations,
            'Curricular units 1st sem (approved)': units_1st_approved,
            'Curricular units 1st sem (grade)': units_1st_grade,
            'Curricular units 1st sem (without evaluations)': units_1st_without_eval,
            'Curricular units 2nd sem (credited)': units_2nd_credited,
            'Curricular units 2nd sem (enrolled)': units_2nd_enrolled,
            'Curricular units 2nd sem (evaluations)': units_2nd_evaluations,
            'Curricular units 2nd sem (approved)': units_2nd_approved,
            'Curricular units 2nd sem (grade)': units_2nd_grade,
            'Curricular units 2nd sem (without evaluations)': units_2nd_without_eval,
            'Unemployment rate': unemployment_rate,
            'Inflation rate': inflation_rate,
            'GDP': gdp
        }

        # Create DataFrame with all features
        for feature in feature_names:
            if feature in feature_mapping:
                input_dict[feature] = feature_mapping[feature]
            else:
                # Default value for encoded categorical features
                input_dict[feature] = 0

        input_df = pd.DataFrame([input_dict])

        # Scale the features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        dropout_probability = prediction_proba[1] * 100

        # Determine risk level
        if dropout_probability < 30:
            risk_level = "LOW"
            risk_color = "risk-low"
            risk_emoji = "🟢"
        elif dropout_probability < 60:
            risk_level = "MEDIUM"
            risk_color = "risk-medium"
            risk_emoji = "🟡"
        else:
            risk_level = "HIGH"
            risk_color = "risk-high"
            risk_emoji = "🔴"

        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")

        col_a, col_b, col_c = st.columns([1, 2, 1])

        with col_b:
            st.markdown(f'<div class="{risk_color}">{risk_emoji} {risk_level} RISK: {dropout_probability:.1f}%</div>',
                       unsafe_allow_html=True)

        st.markdown("")

        # Gauge chart for risk
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=dropout_probability,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Dropout Risk Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#00C851'},
                    {'range': [30, 60], 'color': '#ffbb33'},
                    {'range': [60, 100], 'color': '#ff4444'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': dropout_probability
                }
            }
        ))

        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Top contributing factors
        st.subheader("🔍 Top 3 Contributing Risk Factors")

        # Get top features by importance
        top_features = feature_importance.head(3)

        cols = st.columns(3)
        for idx, (_, row) in enumerate(top_features.iterrows()):
            with cols[idx]:
                st.metric(
                    label=f"#{idx+1} {row['Feature'][:30]}...",
                    value=f"{row['Importance']:.4f}",
                    help=f"Feature importance score: {row['Importance']:.4f}"
                )

        # Recommendation
        st.markdown("---")
        st.subheader("💡 Recommendations")

        if risk_level == "HIGH":
            st.error("""
            **Immediate Intervention Required:**
            - Schedule urgent meeting with academic advisor
            - Assess financial aid options if applicable
            - Provide tutoring and mentoring support
            - Review course load and difficulty
            - Connect with counseling services
            """)
        elif risk_level == "MEDIUM":
            st.warning("""
            **Proactive Support Recommended:**
            - Regular check-ins with academic advisor
            - Monitor academic performance closely
            - Offer study skills workshops
            - Provide information on campus resources
            - Early alert system monitoring
            """)
        else:
            st.success("""
            **Continue Current Support:**
            - Maintain regular progress monitoring
            - Celebrate academic achievements
            - Provide growth opportunities
            - Keep communication channels open
            - Encourage peer mentoring roles
            """)

# TAB 2: Feature Importance
with tab2:
    st.header("📈 Feature Importance Analysis")
    st.write("Understanding which factors most influence dropout prediction:")

    # Top 15 features chart
    top_n = st.slider("Number of top features to display", 5, 30, 15)

    top_features_plot = feature_importance.head(top_n).sort_values('Importance')

    fig_importance = px.bar(
        top_features_plot,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'Importance': 'Feature Importance Score', 'Feature': 'Feature Name'},
        color='Importance',
        color_continuous_scale='Viridis'
    )

    fig_importance.update_layout(height=max(400, top_n * 25), showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)

    # Feature importance table
    with st.expander("📋 View Full Feature Importance Table"):
        st.dataframe(feature_importance, use_container_width=True, height=400)

# TAB 3: Model Performance
with tab3:
    st.header("📋 Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{performance_metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{performance_metrics['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{performance_metrics['recall']:.2%}")
    with col4:
        st.metric("F1-Score", f"{performance_metrics['f1_score']:.4f}")

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Model Details")
        st.info(f"""
        **Model Type:** {model_name}

        **Training Strategy:**
        - 5-Fold Stratified Cross-Validation
        - SMOTE for class balancing
        - GridSearchCV for hyperparameter tuning

        **Features:** {len(feature_names)} input features
        """)

    with col6:
        st.subheader("Performance Interpretation")
        st.success(f"""
        **AUC-ROC Score:** {performance_metrics['auc_roc']:.4f}

        This indicates {'excellent' if performance_metrics['auc_roc'] > 0.9 else 'good' if performance_metrics['auc_roc'] > 0.8 else 'moderate'}
        discriminative ability between dropout and non-dropout students.

        The model has been validated using rigorous cross-validation
        techniques to ensure reliable predictions.
        """)

# Disclaimer
st.markdown("---")
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Disclaimer & Limitations:</strong>
    <ul>
        <li>This tool is designed to assist educators and advisors, not replace human judgment</li>
        <li>Predictions are based on historical data and statistical patterns</li>
        <li>Individual circumstances may vary significantly from model predictions</li>
        <li>Model accuracy: ~{:.1f}% - not all predictions will be correct</li>
        <li>Should be used as one factor among many in student support decisions</li>
        <li>Regular model updates with new data are recommended for best performance</li>
    </ul>
</div>
""".format(performance_metrics['accuracy'] * 100), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Student Dropout Risk Predictor</strong></p>
    <p>Developed by <strong>Naimur</strong> | Applied Machine Learning - EDGE Series</p>
    <p>Dhaka University of Engineering & Technology, Gazipur</p>
    <p>Instructors: Prof. Dr. Fazlul Hasan Siddiqui & Md. Rahad Khan</p>
</div>
""", unsafe_allow_html=True)