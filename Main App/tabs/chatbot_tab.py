import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from openai import OpenAI
import os

# To securely store your OpenRouter API key, create a file named .streamlit/secrets.toml
# in the root directory of your project (same level as your main .py app file or requirements.txt)
# Add the following content to secrets.toml:
#
# OPENROUTER_API_KEY = "your_actual_api_key_here"
#
# Make sure this file is added to your .gitignore if you're using version control.

# Constants for OpenRouter
# HARDCODED_OPENROUTER_API_KEY = "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" # Commented out/removed
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = "meta-llama/llama-3.3-8b-instruct:free" # or your preferred model

def analyze_dataset_for_chatbot(current_df): # Renamed df to current_df
    suggestions = {
        "Missing Values": [],
        "Duplicate Rows": [],
        "Low Variance Features": [],
        "Categorical Features": [],
        "Outliers": [],
        "Correlation and Visualization": [],
        "Model Suggestions": []
    }
    if current_df is None or current_df.empty:
        suggestions["Missing Values"].append("Dataset is empty or not loaded.")
        return suggestions

    # Missing Values
    missing = current_df.isnull().sum()
    if missing.any() and missing.sum() > 0 : # check if there are any missing values at all
        missing_cols_summary = missing[missing > 0]
        suggestions["Missing Values"].append(f"❗ Found missing values in: {', '.join(missing_cols_summary.index.tolist())}. Consider 'Handle Missing Values'.")
        suggestions["Missing Values"].extend([
            "🔧 Options:",
            "- Delete rows (if small % missing).",
            "- Impute with mean/median (numerical).",
            "- Impute with mode (categorical).",
            "- Advanced imputation (e.g., KNN Imputer)."
        ])

    # Duplicate Rows
    if current_df.duplicated().sum() > 0:
        suggestions["Duplicate Rows"].append(f"🔁 {current_df.duplicated().sum()} duplicate rows found. Consider 'Remove Duplicates'.")

    # Low Variance Features
    try:
        numeric_df = current_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            # Check for columns with only one unique value (excluding NaN)
            constant_cols = [col for col in numeric_df.columns if numeric_df[col].dropna().nunique() <= 1]
            if constant_cols:
                 suggestions["Low Variance Features"].append(
                    f"⚠️ Constant/Near-Constant: {', '.join(constant_cols)}. These might not be useful. Consider using 'Constant or Low-Variance' removal."
                )
            # Variance Threshold check for more subtle cases (can be slow on very wide data)
            # For performance, this could be conditional or simplified
            elif numeric_df.shape[1] > 0: # Only if there are numeric columns
                selector = VarianceThreshold(threshold=0.01) # Example threshold, adjust as needed
                try:
                    selector.fit(numeric_df) # This can fail if all values are NaN in a column after dropna
                    low_var_cols = [col for i, col in enumerate(numeric_df.columns) if not selector.get_support()[i] and col not in constant_cols]
                    if low_var_cols:
                        suggestions["Low Variance Features"].append(
                            f"📉 Low Variance (threshold 0.01): {', '.join(low_var_cols)}. Consider 'Constant or Low-Variance' removal."
                        )
                except ValueError as e:
                     suggestions["Low Variance Features"].append(f"Could not perform detailed variance check: {e}")
    except Exception as e:
        suggestions["Low Variance Features"].append(f"Error in low variance check: {e}")

    # Categorical Features
    cat_cols = current_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        suggestions["Categorical Features"].append(
            f"🧩 Categorical columns: {', '.join(cat_cols)}. May need 'Encode Features' for some models."
        )

    # Outliers (simplified check for brevity in chatbot summary)
    numeric_cols_outliers = current_df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_info = []
    for col in numeric_cols_outliers:
        if current_df[col].empty or current_df[col].isnull().all():
            continue
        try:
            if current_df[col].dropna().nunique() > 1: # Ensure there's variance
                q1 = current_df[col].quantile(0.25)
                q3 = current_df[col].quantile(0.75)
                iqr = q3 - q1
                outlier_threshold_upper = q3 + 1.5 * iqr
                outlier_threshold_lower = q1 - 1.5 * iqr
                num_outliers = current_df[(current_df[col] < outlier_threshold_lower) | (current_df[col] > outlier_threshold_upper)].shape[0]
                if num_outliers > 0:
                    outlier_info.append(f"{col} (~{num_outliers} outliers)")
        except Exception:
            pass
    if outlier_info:
        suggestions["Outliers"].append(f"🚨 Potential outliers in: {', '.join(outlier_info)}. Consider 'Handle Outliers'.")

    # Correlation and Visualization - Keep it high level for chatbot
    suggestions["Correlation and Visualization"].append("📊 Explore relationships: Use Scatter, Heatmap (numeric), or Boxplots (num-cat).")
    if len(numeric_cols_outliers) >=2:
         suggestions["Correlation and Visualization"].append("🔍 Check 'Correlation Analysis' for highly correlated numeric features.")

    # Model Suggestions
    target_col = st.session_state.get("target_col")
    if target_col and target_col in current_df.columns:
        target_type = current_df[target_col].dtype
        if pd.api.types.is_numeric_dtype(target_type) and current_df[target_col].nunique() > 10 :
            suggestions["Model Suggestions"].append(
                f"🤖 Target (`{target_col}`) seems numeric/continuous. Try Regression models (e.g., Linear Regression, XGBoost Regressor)."
            )
        elif pd.api.types.is_numeric_dtype(target_type) and current_df[target_col].nunique() <=10 :
             suggestions["Model Suggestions"].append(
                f"🤖 Target (`{target_col}`) is numeric but has few unique values. Could be Classification (e.g., Logistic Regression, Random Forest). Verify task type."
            )
        else:
            suggestions["Model Suggestions"].append(
                f"🤖 Target (`{target_col}`) seems categorical. Try Classification models (e.g., Logistic Regression, Random Forest)."
            )
    else:
        suggestions["Model Suggestions"].append(
            "🎯 No target variable set in 'Preprocessing'. Select one for tailored model advice."
        )
    return suggestions

def show_chatbot_tab(df_passed):
    st.header("🦾 YourAnalyst Assistant")
    st.subheader("🔍 Analyze Dataset and Suggest Steps")

    # Attempt to get API key from Streamlit secrets first
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")

    # If not in secrets, allow user to input it
    if not OPENROUTER_API_KEY:
        st.warning("OpenRouter API key not found in Streamlit secrets. Please enter it below to enable the ChatBot.")
        OPENROUTER_API_KEY = st.text_input("Enter your OpenRouter API Key:", type="password", key="chatbot_api_key_input")
        if OPENROUTER_API_KEY:
            st.success("API Key entered. ChatBot might be available if the key is valid.")
        else:
            st.info("Without an API key, the ChatBot functionality will be disabled.")

    client = None
    if OPENROUTER_API_KEY:
        try:
            client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL  # Use the variable defined at the top
            )
        except Exception as e:
            st.error(f"❌ Error initializing OpenRouter client: {e}. Please ensure the API key is correct and valid.")
            client = None # Ensure client is None if initialization fails
    # Only show this specific warning if the key was initially missing from secrets and also not entered by user.
    elif not st.secrets.get("OPENROUTER_API_KEY"):
        st.warning("ChatBot functionality is disabled as no API key was provided via secrets or manual input.")


    current_df_for_chatbot = st.session_state.get("df")
    if current_df_for_chatbot is None or current_df_for_chatbot.empty:
        st.warning("No dataset loaded or dataset is empty. Please upload data in the main section.")
        return

    st.markdown("""
        Welcome! 🎉 I’m your smart assistant. I can provide suggestions for data cleaning, visualizations, and model types based on your current dataset.
    """)

    col1_chat, col2_chat = st.columns(2)
    with col1_chat:
        with st.expander("📊 Display first 5 rows of data", expanded=True):
            st.dataframe(current_df_for_chatbot.head())

    with col2_chat:
        st.subheader("💡 Assistant's Suggestions (Automatic Analysis):")
        with st.spinner("🔍 Analyzing dataset for suggestions..."):
            suggestions = analyze_dataset_for_chatbot(current_df_for_chatbot)

        if any(s_list for s_list in suggestions.values() if s_list):
            for section, items in suggestions.items():
                if items:
                    with st.expander(f"📌 {section}", expanded=(section == "Model Suggestions" or section == "Missing Values")):
                        for s_item in items:
                            st.markdown(f"- {s_item}")
        else:
            st.markdown("✅ No specific issues flagged by the basic automatic analysis. You can still ask questions below.")

    st.subheader(f"🤔 Ask the Assistant (via OpenRouter - Model: {OPENROUTER_MODEL_NAME}):")
    user_question = st.text_input("💬 Ask any question about the data (e.g., 'What is the average of column X?', 'Suggest a plot for columns Y and Z'):", key="chatbot_question")

    if user_question: # Only proceed if there's a question
        if client:
            with st.spinner("⏳ Thinking..."):
                df_preview_for_llm_rows = current_df_for_chatbot.head(20).to_string()
                df_description_for_llm = f"The dataset has {current_df_for_chatbot.shape[0]} rows and {current_df_for_chatbot.shape[1]} columns. Column names are: {', '.join(current_df_for_chatbot.columns)}. \nData types:\n{current_df_for_chatbot.dtypes.to_string()}\n\nFirst 20 rows preview:\n{df_preview_for_llm_rows}"

                MAX_PREVIEW_LENGTH = 3000
                if len(df_description_for_llm) > MAX_PREVIEW_LENGTH:
                    df_description_for_llm = df_description_for_llm[:MAX_PREVIEW_LENGTH] + "\n[Preview truncated]"

                prompt = f"""You are an expert data analysis assistant.
                The user is working with a pandas DataFrame in Python.
                Here's a description of their dataset:
                {df_description_for_llm}

                The user's question is: "{user_question}"

                Please provide a concise and helpful answer based *only* on the provided dataset description and preview.
                If the question is about specific values not in the preview, state that.
                If it's a general data analysis question, answer it generally.
                If you suggest Python/pandas code, keep it simple.
                """
                try:
                    response = client.chat.completions.create(
                        model=OPENROUTER_MODEL_NAME, # Use the variable
                        messages=[
                            {"role": "system", "content": "You are an intelligent data analysis assistant. Answer questions based on the provided table data. Respond in English."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    answer = response.choices[0].message.content
                    st.markdown(f"🤖 {answer}")
                except Exception as e:
                    st.error(f"❌ An error occurred while communicating with OpenRouter: {e}")
        else: # if client is None (meaning API key issue)
            st.error("⚠️ ChatBot service is not available. Please ensure your OpenRouter API key is correctly configured and valid, either via Streamlit secrets or by entering it above.")

# This is the ask_llm logic, now integrated directly within the button press,
# which is a common Streamlit pattern. No separate ask_llm function needed here anymore.
# def ask_llm(question, df_preview_text, current_client): # Added client as argument
#     if not current_client: # Check the passed client
#         return "Sorry, the ChatBot service is not available. Please ensure your OpenRouter API key is correctly configured via Streamlit secrets or manual input."
#     prompt = f"""This is a preview of the first 100 rows of the dataset:\n{df_preview_text}\n\nMy question is: {question}"""
#     try:
#         response = current_client.chat.completions.create( # Use current_client
#             model=OPENROUTER_MODEL_NAME, # Use the variable
#             messages=[
#                 {"role": "system", "content": "You are an intelligent data analysis assistant. Answer questions based on the provided table data. Respond in English."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"❌ An error occurred while communicating with OpenRouter: {e}"
