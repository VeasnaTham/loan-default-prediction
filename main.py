import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model('final_model.h5')
encoders = joblib.load('label_encoders.joblib')
scaler = joblib.load('scaler.pkl')

# Helper function to preprocess data exactly like in training
def preprocess_data(input_data):
    """
    Preprocess the input data to match the training pipeline
    """
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Define datetime columns
    datetime_columns = ['issue_date', 'last_payment_date', 'next_payment_date', 
                       'last_credit_pull_d', 'earliest_cr_line']
    
    # Convert datetime columns to datetime
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    for col, le in encoders.items():
        if col in df.columns:
        # Transform using the pre-fitted encoder
            df[col] = le.transform(df[col])
    
    # Reduce memory for int columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Reduce memory for float columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert datetime to numerical value
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for model information
with st.sidebar:
    st.title("📊 Model Information")
    st.markdown("**Features:** 21 input parameters")
    st.markdown("**Model Type:** Neural Network")
    st.info("**Accuracy:** 98.34%")
    st.info("**Precision:** 96%")
    st.info("**Recall:** 97%")
    st.info("**F1-Score:** 96.5%")
    st.info("**AUC Score:** 0.9957")


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.title("📝 Loan Application Details")
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Loan Details", "👤 Borrower Info", "📅 Dates & Timeline", "📊 Credit History"])
    
    with tab1:
        st.subheader("Loan Information")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=0.0,
                value=10000.0,
                step=500.0,
                help="Total loan amount requested"
            )
            
            funded_amnt = st.number_input(
                "Funded Amount ($)",
                min_value=0.0,
                value=10000.0,
                step=500.0,
                help="Amount actually funded"
            )
            
            installment = st.number_input(
                "Monthly Installment ($)",
                min_value=0.0,
                value=300.0,
                step=10.0,
                help="Monthly payment amount"
            )
            
            int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=12.0,
                step=0.1,
                help="Annual interest rate"
            )
        
        with col1_2:
            term = st.selectbox(
                "Loan Term in months",
                options=[36, 60],
                help="Loan repayment period"
            )
            
            purpose = st.selectbox(
                "Loan Purpose",
                options=[
                "debt_consolidation",
                "credit_card", 
                "home_improvement",
                "personal",
                "healthcare",
                "vehicle",
                "small_sized_enterprise",
                "house",
                "education",
                "others"

            ])
            
            application_type = st.selectbox(
                "Application Type",
                options=["personal", "joint"],
                help="Type of loan application"
            )
            
            restructured_loan = st.selectbox(
                "Restructured Loan",
                options=["No", "Yes"],
                help="Has this loan been restructured?"
            )
    
    with tab2:
        st.subheader("Borrower Information")
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            annual_inc = st.number_input(
                "Annual Income ($)",
                min_value=0.0,
                max_value=500000.0,
                value=50000.0,
                step=1000.0,
                help="Borrower's annual income"
            )
            
            total_acc = st.number_input(
                "Total Accounts",
                min_value=0,
                max_value=100,
                value=10,
                step=1,
                help="Total number of credit accounts"
            )
            
            all_enquiries = st.number_input(
                "Credit Enquiries",
                min_value=0,
                max_value=50,
                value=2,
                step=1,
                help="Number of credit enquiries in last 6 months"
            )
        
        with col2_2:
            credit_score_low = st.number_input(
                "Credit Score (Low)",
                min_value=300,
                max_value=850,
                value=650,
                step=5,
                help="Lower bound of credit score range"
            )
            
            credit_score_high = st.number_input(
                "Credit Score (High)",
                min_value=300,
                max_value=850,
                value=700,
                step=5,
                help="Upper bound of credit score range"
            )
            
            outstanding_balance = st.number_input(
                "Outstanding Balance ($)",
                min_value=0.0,
                max_value=100000.0,
                value=5000.0,
                step=100.0,
                help="Current outstanding loan balance"
            )
    
    with tab3:
        st.subheader("Important Dates")
        
        col3_1, col3_2 = st.columns(2)
        with col3_1:
            issue_date = st.date_input(
                "Issue Date", 
                min_value=datetime(1900, 1, 1), 
                max_value=datetime(2100, 12, 31),
                value=date.today(),
                help="Date when loan was issued"
            )
            
            last_payment_date = st.date_input(
                "Last Payment Date",
                min_value=datetime(1900, 1, 1), 
                max_value=datetime(2100, 12, 31),
                value=date.today(),
                help="Date of last payment received"
            )
            
            next_payment_date = st.date_input(
                "Next Payment Date",
                min_value=datetime(1900, 1, 1), 
                max_value=datetime(2100, 12, 31),
                value=date.today(),
                help="Due date for next payment"
            )
        
        with col3_2:
            last_credit_pull_d = st.date_input(
                "Last Credit Pull Date",
                min_value=datetime(1900, 1, 1), 
                max_value=datetime(2100, 12, 31),
                value=date.today(),
                help="Date when credit was last pulled"
            )
            
            earliest_cr_line = st.date_input(
                "Earliest Credit Line",
                min_value=datetime(1900, 1, 1), 
                max_value=datetime(2100, 12, 31),
                value=date.today(),
                help="Date of borrower's earliest credit line"
            )
    
    with tab4:
        st.subheader("Payment History")
        
        col4_1, col4_2 = st.columns(2)
        with col4_1:
            last_amnt_paid = st.number_input(
                "Last Amount Paid ($)",
                min_value=0.0,
                max_value=10000.0,
                value=300.0,
                step=10.0,
                help="Amount of last payment received"
            )
        
        with col4_2:
            past_due_fee = st.number_input(
                "Past Due Fee ($)",
                min_value=0.0,
                max_value=1000.0,
                value=0.0,
                step=5.0,
                help="Any past due fees accumulated"
            )

# Prediction section
with col2:
    st.header("🎯 Prediction Results")
    
    # Prediction button
    if st.button("🔮 Predict Default Risk", type="primary", use_container_width=True):
        
        if model is None:
            st.error("❌ Model not loaded. Please check file path.")
        else:
            try:
                # Prepare input data
                input_data = {
                    'loan_amnt': loan_amnt,
                    'funded_amnt': funded_amnt,
                    'restructured_loan': "N" if restructured_loan == "No" else "Y",
                    'tenure': term,
                    'int_rate': int_rate/100,
                    'installment': installment,
                    'loan_type': purpose,
                    'account_type': application_type,
                    'annual_inc': annual_inc,
                    'outstanding_balance': outstanding_balance,
                    'issue_date': issue_date,
                    'last_amount_paid': last_amnt_paid,
                    'last_payment_date': last_payment_date,
                    'past_due_fee': past_due_fee,
                    'next_payment_date': next_payment_date,
                    'last_credit_pull_d': last_credit_pull_d,
                    'earliest_cr_line': earliest_cr_line,
                    'all_enquiries': all_enquiries,
                    'total_acc': total_acc,
                    'fico_range_low': credit_score_low,
                    'fico_range_high': credit_score_high
                }
                
                # Preprocess data to match training pipeline
                processed_df = preprocess_data(input_data)
                
                # Debug: Show the processed data
                # st.write(f"Processed data shape: {processed_df.shape}")
                # st.write("Processed data sample:")
                # st.dataframe(processed_df)
                
                test = scaler.transform(processed_df)
                
                # Make prediction directly without additional encoder
                prediction_prob = model.predict(test, verbose=0)[0][0]
                prediction_class = 1 if prediction_prob > 0.5 else 0
                
                # Display results
                st.subheader("📊 Risk Assessment")
                
                # Risk level determination
                if prediction_prob < 0.3:
                    risk_level = "Low Risk"
                    color = "green"
                    recommendation = "✅ **APPROVE** - Low default risk detected"
                elif prediction_prob < 0.6:
                    risk_level = "Medium Risk"
                    color = "orange"
                    recommendation = "⚠️ **REVIEW** - Moderate risk, consider additional verification"
                else:
                    risk_level = "High Risk"
                    color = "red"
                    recommendation = "❌ **DECLINE** - High default risk detected"
                
                st.markdown(f"**Risk Level:** :{color}[{risk_level}]")
                
                # Probability metrics
                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric(
                        "Model Confidence",
                        f"{prediction_prob:.2%}" if prediction_class == 1 else f"{1 - prediction_prob:.2%}"
                    )
                with col_metrics2:
                    st.metric(
                        "Classification",
                        "Default" if prediction_class == 1 else "Non-Default"
                    )
                
                
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.write("**Error details:**")
                st.exception(e)
                st.write("**Input data for debugging:**")
                st.json(input_data)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 14px; padding: 20px 0;'>
        © 2025 Loan Default Prediction<br>
        RUPP - Department of Data Science and Engineering<br>
        Developed by Tham Veasna
    </div>
    """, 
    unsafe_allow_html=True
)