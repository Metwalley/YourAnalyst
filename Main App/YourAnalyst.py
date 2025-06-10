import streamlit as st
import pandas as pd
# Update imports:
# import all_function as af # Old
# import visualize as v # Old
from Preprocessing_Functions import all_function as af
from visualization_Functions import visualize as v
# from io import BytesIO # Moved to visualization_tab.py where save_plot is
# Sklearn imports are moved to models_tab.py
# import pickle # Moved to models_tab.py
import os
# import io # Moved to models_tab.py
# import numpy as np # Moved to relevant tabs
# from sklearn.feature_selection import VarianceThreshold # Moved to chatbot_tab.py
# from scipy import stats # Moved to relevant tabs
# from openai import OpenAI # Moved to chatbot_tab.py

# Import tab functions
from tabs.preprocessing_tab import show_preprocessing_tab
from tabs.visualization_tab import show_visualization_tab
from tabs.models_tab import show_models_tab
from tabs.chatbot_tab import show_chatbot_tab

st.set_page_config(page_title="YourAnalyst Assistant", page_icon="🦾", layout="wide")

# Load CSS from external file
css_file_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
try:
    with open(css_file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"Error: Could not load stylesheet. Expected at: {css_file_path}")

# ========== العنوان الرئيسي ==========
st.title("🌟 YourAnalyst ")

# ========== تحميل الداتا ==========
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "excel", "json", "parquet"])

if uploaded_file is not None:
    # Only reload df if new file uploaded or df not in session
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name or "df" not in st.session_state:
        try:
            df_loaded = af.load_data(uploaded_file) # df_loaded instead of df
            if df_loaded is not None:
                st.session_state.df = df_loaded
                st.session_state.uploaded_file_name = uploaded_file.name # Store filename to detect changes
                # Clear potentially outdated session state from previous datasets
                keys_to_clear = ['X_train', 'X_test', 'y_train', 'y_test', 'target_col',
                                 'trained_model', 'y_pred', 'selected_preprocessing', 'prev_df']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ Dataset uploaded and processed successfully.")
                st.info("Previous analysis states (like selected target, trained models) have been reset.")

        except Exception as e:
            st.error(f"Error loading data: {e}")
            if "df" in st.session_state: # remove potentially corrupted df
                 del st.session_state.df
            if "uploaded_file_name" in st.session_state:
                 del st.session_state.uploaded_file_name


# ========= Show Tabs If Data Exists =========
if "df" in st.session_state and st.session_state.df is not None:
    df_to_pass = st.session_state.df # Use a consistent variable name for clarity

    tab_labels = ["🛠️ Preprocessing", "📊 Visualization", "🤖 Models", "🦾 ChatBot"]

    # Get current active tab index from session state, default to 0 if not set
    active_tab_index = st.session_state.get("active_tab_index", 0)

    selected_tab_label = st.radio(
        "Choose a tab",
        tab_labels,
        index=active_tab_index,
        key="main_tab_selector" # Add a key for robust state management
    )

    # Update active_tab_index in session state if selection changes
    if tab_labels.index(selected_tab_label) != active_tab_index:
        st.session_state.active_tab_index = tab_labels.index(selected_tab_label)
        # No need to rerun here, Streamlit handles it.

    # Call the appropriate tab function
    if selected_tab_label == "🛠️ Preprocessing":
        show_preprocessing_tab(df_to_pass)
    elif selected_tab_label == "📊 Visualization":
        show_visualization_tab(df_to_pass)
    elif selected_tab_label == "🤖 Models":
        show_models_tab(df_to_pass)
    elif selected_tab_label == "🦾 ChatBot":
        show_chatbot_tab(df_to_pass)
else:
    if not uploaded_file : # only show if no file has been uploaded yet in this session
        st.info("👋 Welcome to YourAnalyst! Please upload a dataset to begin analysis.")
    # If df is None due to an error during upload, the error message from load_data should be visible.
    # If file was uploaded but df is None, it means loading failed.
    elif uploaded_file and ("df" not in st.session_state or st.session_state.df is None):
        st.warning("Dataset could not be loaded. Please check the file format and content.")

# Optional: Add a footer or some persistent UI element if desired
# st.markdown("---")
# st.markdown("YourAnalyst © 2024")
