import streamlit as st
import pandas as pd
import all_function as af
import visualize as v
from io import BytesIO
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import io
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from openai import OpenAI
st.set_page_config(page_title="YourAnalyst Assistant", page_icon="🦾", layout="wide")



st.markdown(
    """
    <style>
    /* ===== Original styles (unchanged) ===== */

    /* App background */
    .reportview-container {
        background-color: #212121;
        color: #e0e0fd;
    }

    /* Sidebar background */
    .sidebar .sidebar-content {
        background-color: #e0e0fd;
        color: #212121;
    }

    /* Buttons */
    .stButton>button {
        cursor: pointer;
        font-family: "system-ui", sans-serif;
        font-size: 13px;
        color: #ffffff;
        padding: 6px 16px;
        width: 100%;
        border-radius: 30px;
        border: none;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #0093e9 0%, #80d0c7 100%);
        color: #ffffff;
        transform: scale(1.02);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    }

    /* Tab titles */
    .css-1c6jbr4 {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 26px;
        font-weight: bold;
        color: #e0e0fd;
    }

    /* General font */
    body {
        font-family: 'Roboto', sans-serif;
    }

    /* DataFrames */
    .stDataFrame {
        background-color: #424242;
        color: #e0e0fd;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Title styling */
    .stTitle {
        font-family: 'Arial', sans-serif;
        font-size: 34px;
        color: #e0e0fd;
        font-weight: bold;
    }

    /* Tab section styling */
    .stTab {
        border: 2px solid #e0e0fd;
        border-radius: 10px;
        background-color: #333333;
        padding: 10px;
    }

    .stHeader {
        color: #e0e0fd;
        font-size: 22px;
        font-weight: 600;
    }

    .stSubheader {
        color: #e0e0fd;
        font-size: 18px;
    }

    /* ===== New: Purple horizontal radio tabs only ===== */

    /* Container: horizontal flex row for the radio buttons */
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;  /* ADD THIS */
        flex-wrap: nowrap !important;
        gap: 1rem;
        padding: 10px 0;
        justify-content: flex-start;
    }

    /* Each radio label styled as purple pill */
    div[role="radiogroup"] > label {
        position: relative;
        display: inline-block;
        padding: 12px 28px;
        border-radius: 30px;
        font-size: 18px;
        font-weight: 600;
        background-color: #4b367c; /* dark purple */
        color: #e0e0fd;
        border: 2px solid transparent;
        transition: all 0.3s ease-in-out;
        cursor: pointer;
        display: inline-flex;      /* use inline-flex to align items inside */
        align-items: center;       /* vertical center of circle and text */
        gap: 8px;   
        user-select: none;
        min-width: 130px;
        text-align: center;
    }

    /* Hover effect on each tab */
    div[role="radiogroup"] > label:hover {
        background-color: #6f53c9; /* lighter purple */
        border-color: #b4a7e7;
    }

    /* Selected (checked) tab styling */
    div[role="radiogroup"] > label:has(input[type="radio"]:checked) {
        background-color: #9b59b6; /* bright purple */
        border: 2px solid #b4a7e7;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(155, 89, 182, 0.5);
    }

    /* Hide the default radio input */
    div[role="radiogroup"] input[type="radio"] {
        display: none;
    }

    </style>
    """,
    unsafe_allow_html=True,
)







# ========== العنوان الرئيسي ==========
st.title("🌟 YourAnalyst ")

# ========== تحميل الداتا ==========
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "excel", "json", "parquet"])
if uploaded_file is not None:
    # Only reload df if new file uploaded
    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file != uploaded_file:
        df = af.load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.uploaded_file = uploaded_file
            st.success("✅ Dataset uploaded and stored successfully.")
else:
    if "df" not in st.session_state:
        st.info("Please upload a dataset to continue.")

# ========= Show Tabs If Data Exists =========

if "df" in st.session_state:
    df = st.session_state.df
    tab_labels = ["🛠️ Preprocessing", "📊 Visualization", "🤖 Models", "🦾 ChatBot"]
    selected_tab = st.radio("Choose a tab", tab_labels, index=st.session_state.get("active_tab", 0))
    
    
    # ==============================
    # 1. تبويب Preprocessing
    # ==============================
    if selected_tab == "🛠️ Preprocessing":
        st.header("🛠️ Data Preprocessing")

        # Define all preprocessing options
        preprocessing_options = [
            "📄 View Dataset", "📌 List All Columns", "📈 Column Info", "🧾 Check Data Integrity",
            "📊 Dataset Summary", "🧹 Remove Duplicates", "🔍 Detect Missing Values", "🔧 Handle Missing Values",
            "💡 Encode Features", "🗑️ Remove Columns", "✏️ Rename Columns", "🧠 Handle Outliers",
            "🔄 Replace Values", "🔁 Change Column Data Types", "🕓 Extract Datetime Features",
            "🔢 Scale/Normalize Features", "🧬 Feature Interaction or Generation",
            "🧽 Constant or Low-Variance", "📉 Correlation Analysis",
            "📊 Feature Importance", "🔍 Check Class Imbalance", "🎯 Set Target Variable",
            "📦 Split Dataset (Train/Test)", "📥 Download Cleaned Dataset"
        ]

        # Initialize selected option
        if "selected_preprocessing" not in st.session_state:
            st.session_state.selected_preprocessing = preprocessing_options[0]


        # Display buttons in a grid (4 columns per row)
        cols = st.columns(4)
        for idx, label in enumerate(preprocessing_options):
            if cols[idx % 4].button(label):
                st.session_state.selected_preprocessing = label

        # Read selected option from session state
        option = st.session_state.selected_preprocessing


        if option == "📄 View Dataset":
            st.subheader("📄 Dataset Preview")

            show_all = st.checkbox("🔁 Show full dataset", value=False)
            df = st.session_state.get("df", df)
            if show_all:
                st.dataframe(df)
            else:
                st.write("### 🔝 First 10 Rows")
                st.dataframe(df.head(10))

                st.write("### 🔚 Last 10 Rows")
                st.dataframe(df.tail(10))


        elif option == "📌 List All Columns":
            df = st.session_state.get("df", df)
            af.list_all_columns(df)

        elif option == "📈 Column Info":
            col = st.selectbox("Choose column:", df.columns)
            df = st.session_state.get("df", df)
            af.view_column_info(df, col)
        
        elif option == "📊 Feature Importance":
            st.subheader("📊 Feature Importance (via Random Forest)")
            df = st.session_state.get("df", df)
            target = st.selectbox("Select the target column:", df.columns)
            if target:
                importance = af.get_feature_importance(df, target)
                st.bar_chart(importance.head(20))
                st.dataframe(importance.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}))

        elif option == "🔍 Check Class Imbalance":
            st.subheader("🔍 Search for Class Imbalance")
            df = st.session_state.get("df", df)
            target_col = st.selectbox("Select the target column:", df.columns)
            class_counts = af.check_class_imbalance(df, target_col)

            st.bar_chart(class_counts)
            st.dataframe(class_counts.reset_index().rename(columns={'index': 'Class', target_col: 'Count'}))

            imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
            if imbalance_ratio > 1.5:
                st.warning(f"⚠️ Class imbalance detected (max/min ratio = {imbalance_ratio:.2f}). Consider balancing techniques like SMOTE.")
            else:
                st.success("✅ Class distribution looks reasonably balanced.")

        elif option == "🎯 Set Target Variable":
            st.subheader("🎯 Select Target (Prediction) Column")
            df = st.session_state.get("df", df)
            target_col = st.selectbox("Select your target variable (what you're predicting):", df.columns)

            if st.button("Confirm Target Column"):
                st.session_state.target_col = target_col
                st.success(f"🎯 Target variable set to: `{target_col}`")


        elif option == "📦 Split Dataset (Train/Test)":
            st.subheader("📦 Split Dataset into Train/Test")
            df = st.session_state.get("df", df)
            if "target_col" not in st.session_state:
                st.warning("⚠️ Please set a target column first in the 🎯 section.")
            else:
                test_size = st.slider("Test Size (%)", min_value=5, max_value=50, value=20, step=5)
                train_size = st.slider("Train Size (%)", min_value=10, max_value=90, value=80, step=5)
                stratify = st.checkbox("Use Stratified Split (for classification)?")

                if st.button("Split Dataset"):
                    X_train, X_test, y_train, y_test = af.split_dataset(
                    st.session_state.df,
                    st.session_state.target_col,
                    test_size=test_size / 100,
                    train_size=train_size / 100,
                    stratify=stratify
                )
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test

                    st.success(f"✅ Dataset split successfully: {len(X_train)} train rows, {len(X_test)} test rows.")
                    st.write("📘 **Train Set Preview**")
                    st.dataframe(X_train.head())

            
        elif option == "🕓 Extract Datetime Features":
            st.subheader("🕓 Extract Datetime Features")
            df = st.session_state.get("df", df)
            datetime_cols = df.select_dtypes(include=["datetime64[ns]", "object"]).columns
            datetime_col = st.selectbox("Select a datetime column:", datetime_cols)

            if st.button("Extract Features"):
                df = af.extract_datetime_features(df, datetime_col)
                st.session_state.df = df
                st.dataframe(df)
            st.session_state.df = df    
            
        elif option == "🔁 Change Column Data Types":
            st.subheader("🔁 Change Column Data Types")
            df = st.session_state.get("df", df)
            if st.checkbox("Select all columns"):
                columns = st.multiselect("Select columns:", df.columns, default=list(df.columns))
            else:
                columns = st.multiselect("Select columns:", df.columns)

            dtype_options = ["int", "float", "object", "datetime", "category"]
            new_dtype = st.selectbox("Convert to data type:", dtype_options)

            if st.button("Apply Type Conversion"):
                df = af.change_columns_dtype(df, columns, new_dtype)
                st.session_state.df = df
                st.success("✅ Type conversion applied.")
                st.dataframe(df)
        

        elif option == "📊 Dataset Summary":
            st.subheader("📊 Dataset Summary & Columns Info")
            af.view_all_columns_summary(df)



        elif option == "🧹 Remove Duplicates":
            df = st.session_state.get("df", df)
            st.subheader("🧹 Remove Duplicate Rows")
            num_duplicates = af.check_duplicates(df)

            if num_duplicates == 0:
                st.info("✅ No duplicate rows found in the dataset.")
            else:
                st.warning(f"⚠️ Found {num_duplicates} duplicate rows in your dataset.")
                if st.button("🧹 Remove Duplicates Now"):
                    df, removed = af.remove_duplicates(df)
                    st.session_state.df = df
                    st.success(f"✅ Removed {removed} duplicate rows.")
                    st.dataframe(df)

        elif option == "🔍 Detect Missing Values":
            st.subheader("🔍 Missing Values")
            df = st.session_state.get("df", df)
            missing_cols = af.get_missing_columns(df)
            if not missing_cols:
                st.info("✅ No missing values found in the dataset.")
            else:
                st.warning("⚠️ Missing values found in the following columns:")
                for col in missing_cols:
                    dtype = df[col].dtype
                    missing_count = df[col].isnull().sum()
                    st.markdown(f"- **{col}** *(dtype: `{dtype}`, missing: `{missing_count}`)*")

        elif option == "📉 Correlation Analysis":
            st.subheader("📉 Correlation Analysis")
            df = st.session_state.get("df", df)
            threshold = st.slider("Correlation Threshold (absolute)", min_value=0.5, max_value=1.0, value=0.9, step=0.01)
            to_drop, corr_matrix = af.detect_highly_correlated(df, threshold)

            st.write("Correlation Matrix:")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

            if not to_drop:
                st.info("✅ No highly correlated features found.")
            else:
                st.warning(f"⚠️ {len(to_drop)} features exceed the correlation threshold.")
                st.write(to_drop)

                if st.button("Remove Correlated Features"):
                    df = af.remove_correlated_features(df, to_drop)
                    st.session_state.df = df
                    st.dataframe(df)



        elif option == "🧬 Feature Interaction or Generation":
            st.subheader("🧬 Feature Interaction / Generation")
            df = st.session_state.get("df", df)
            st.markdown("Example: `(df['Feature1'] + df['Feature2']) / 2`")
            cols = df.columns.tolist()

            with st.expander("➕ Generate New Feature", expanded=True):
                col1 = st.selectbox("Select first column:", cols, key="gen_col1")
                col2 = st.selectbox("Select second column:", cols, key="gen_col2")

                operation = st.radio("Select operation:", ["add", "subtract", "multiply", "divide"], horizontal=True)
                new_col_name = st.text_input("Enter name for the new feature:", key="new_feature_name")

                col1_, col2_ = st.columns(2)
                with col1_:
                    if st.button("✅ Generate Feature"):
                        if new_col_name.strip() == "":
                            st.error("⚠️ Please enter a valid name for the new feature.")
                        elif new_col_name in df.columns:
                            st.error("⚠️ A column with this name already exists.")
                        else:
                            # Save previous version before change
                            st.session_state.prev_df = df.copy()

                            # Apply feature creation
                            df = af.generate_feature(df, col1, col2, operation, new_col_name)
                            st.session_state.df = df
                            st.dataframe(df.head())

                with col2_:
                    if st.button("↩️ Undo Last Change"):
                        if "prev_df" in st.session_state:
                            st.session_state.df = st.session_state.prev_df.copy()
                            del st.session_state.prev_df
                            st.success("✅ Reverted to previous version.")
                            st.dataframe(st.session_state.df.head())
                        else:
                            st.warning("⚠️ No previous version to revert to.")

        elif option == "🧽 Constant or Low-Variance":
            st.subheader("🧽 Detect & Handle Constant or Low-Variance Features")
            df = st.session_state.get("df", df)
            threshold = st.slider("Variance Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            low_var_cols = af.detect_low_variance_features(df, threshold)

            if not low_var_cols:
                st.success("✅ No low-variance features detected.")
            else:
                st.warning(f"⚠️ Found {len(low_var_cols)} low-variance feature(s):")
                
                # Show columns in a nice dataframe
                st.dataframe(pd.DataFrame(low_var_cols, columns=["Low-Variance Features"]))

                if st.button("🗑️ Remove Low-Variance Features"):
                    df = af.remove_low_variance_features(df, low_var_cols)
                    st.session_state.df = df
                    st.success(f"Removed {len(low_var_cols)} low-variance feature(s). Updated dataframe:")
                    st.dataframe(df)   




        elif option == "🔧 Handle Missing Values":
            st.subheader("🔧 Handle Missing Values")
            missing_cols = af.get_missing_columns(df)
            df = st.session_state.get("df", df)
            if not missing_cols:
                st.info("✅ No missing values found in the dataset.")
            else:
                selected_cols = st.multiselect(
                    "Select columns with missing values:", missing_cols
                )

                methods_dict = {}
                custom_values = {}

                for col in selected_cols:
                    col_type = df[col].dtype
                    st.markdown(f"**Column: `{col}` ({col_type})**")

                    # 1) Offer the same “Custom” option for both numeric & non-numeric
                    if pd.api.types.is_numeric_dtype(df[col]):
                        options = ["Drop", "Mean", "Median", "Custom"]
                    else:
                        options = ["Drop", "Mode", "Custom"]

                    method = st.selectbox(
                        f"Choose method for `{col}`:", options, key=f"method_{col}"
                    )
                    methods_dict[col] = method

                    # 2) If user picks Custom, ask for the value
                    if method == "Custom":
                        custom_value = st.text_input(
                            f"Enter custom fill for `{col}`:", key=f"custom_{col}"
                        )
                        custom_values[col] = custom_value

                if st.button("Apply Missing Value Handling"):
                    for col in selected_cols:
                        df = af.handle_missing_column(
                            df,
                            col,
                            methods_dict[col],
                            custom_values.get(col)
                        )

                    st.session_state.df = df
                    st.success("✅ Missing values handled successfully.")
                    st.dataframe(df)


        elif option == "💡 Encode Features":
            st.subheader("💡 Encode Features")
            df = st.session_state.get("df", df)
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if st.checkbox("Select all categorical columns"):
                cols = st.multiselect("Select columns to encode", cat_cols, default=cat_cols)
            else:
                cols = st.multiselect("Select columns to encode", cat_cols)

            method = st.selectbox("Encoding method", ['label', 'onehot', 'ordinal'])

            if st.button("Apply Encoding"):
                df = af.encode_features(df, cols, method)
                st.session_state.df = df
                st.dataframe(df)

        elif option == "🗑️ Remove Columns":
            st.subheader("🗑️ Remove Columns")
            df = st.session_state.get("df", df)
            cols = st.multiselect("Select columns to remove", df.columns)
            if st.button("Remove Columns"):
                df = af.remove_columns(df, cols)
                st.session_state.df = df
                st.success("✅ Columns removed.")
                st.dataframe(df)

        elif option == "✏️ Rename Columns":
            st.subheader("✏️ Rename Columns")
            df = st.session_state.get("df", df)
            # اختيار العمود الذي سيتم تغييره
            col_to_rename = st.selectbox("Choose column to rename:", df.columns)

            # إدخال الاسم الجديد للعمود
            new_name = st.text_input(f"Enter new name for `{col_to_rename}`:")

            if st.button("Rename Column"):
                if new_name:
                    # تطبيق التغيير على العمود في الـ DataFrame
                    df = af.rename_columns(df, {col_to_rename: new_name})
                    st.session_state.df = df
                    st.success(f"✅ Column `{col_to_rename}` renamed to `{new_name}`.")
                    st.dataframe(df)  # عرض الـ DataFrame بعد التغيير
                else:
                    st.error("⚠️ Please provide a new name for the column.")

        elif option == "🧠 Handle Outliers":
            st.subheader("🧠 Handle Outliers")
            df = st.session_state.get("df", df)
            # اختيار العمود الذي سيتم تطبيقه عليه
            col_to_check = st.selectbox("Choose column to check for outliers:", df.columns)

            # اختيار طريقة الكشف عن الـ outliers
            method = st.selectbox("Outlier detection method", ['IQR', 'zscore', 'isolation_forest'])

            # وضع زر لتطبيق الـ outliers
            if st.button("Handle Outliers"):
                if col_to_check and method:
                    # تطبيق الكشف والتعامل مع الـ outliers
                    df = af.handle_outliers(df, col_to_check, method)
                    st.session_state.df = df
                    st.success(f"✅ Outliers handled in column `{col_to_check}` using `{method}` method.")
                    st.dataframe(df)  # عرض الـ DataFrame بعد التغيير
                else:
                    st.error("⚠️ Please select a column and a method for outlier handling.")

        elif option == "🧾 Check Data Integrity":
            st.subheader("🧾 Check Data Integrity")
            df = st.session_state.get("df", df)
            result = af.check_data_integrity(df)
            st.subheader("📊 Data Integrity Report")
            for key, value in result.items():
                st.write(f"**{key}:** {value}")


        elif option == "📥 Download Cleaned Dataset":
            df = st.session_state.get("df", df)
            st.subheader("📥 Download Cleaned Dataset")
            format = st.selectbox("Choose format", ['csv', 'excel'])
        
            if format == "csv":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    mime='text/csv'
                )
            elif format == "excel":
                excel_buffer = af.convert_df_to_excel(df)
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_buffer,
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
        elif option == "🔄 Replace Values":
            st.subheader("🔄 Replace Values in Columns")
            df = st.session_state.get("df", df)
            # 1. Let user pick one or more columns
            cols = st.multiselect("Select column(s) to modify:", df.columns.tolist())

            # 2. Ask for the exact value to replace, and its replacement
            to_replace = st.text_input("Value to replace (exact match):")
            replacement = st.text_input("Replacement value:")

            # 3. Only run when user clicks the button
            if st.button("Apply Replacement"):
                df = af.replace_values_in_columns(df, cols, to_replace, replacement)
                st.success(f"Replaced **{to_replace}** with **{replacement}** in {len(cols)} column(s).")
                st.dataframe(df.head())
        elif option == "🔢 Scale/Normalize Features":
            st.subheader("🔢 Scale or Normalize Features")
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if st.checkbox("Select all numerical columns"):
                columns = st.multiselect("Select columns to scale/normalize", num_cols, default=num_cols)
            else:
                columns = st.multiselect("Select columns to scale/normalize", num_cols)

            method = st.selectbox("Scaling method:", ["minmax", "standard", "robust"])

            if st.button("Apply Scaling/Normalization"):
                df = af.scale_features(df, columns, method)
                st.session_state.df = df
                st.dataframe(df)



    # ==============================
    # 2. تبويب Visualization
    # ==============================
    if selected_tab == "📊 Visualization":
        st.header("📊 Visualization")
        # Select the type of plot
        plot_type = st.selectbox("Choose the type of plot", [
            "Scatter Plot", "Line Plot (Time Series)", 
            "Correlation Heatmap", "Pairplot", "Histogram", "Density Plot", 
            "Boxplot", "Violin Plot", "Bar Plot", "Pie Chart", "Missing Heatmap", 
            "Missing Barplot", "Word Cloud"
        ])
        # Function to show appropriate columns for each plot
        def get_column_options(plot_type, df):
            if plot_type in ["Scatter Plot", "Line Plot (Time Series)"]:
                return df.columns
            else:
                return df.select_dtypes(include=['float64', 'int64', 'object']).columns

        columns_for_plot = get_column_options(plot_type, df)

        # Handle the cases where multiple columns need to be selected (for Scatter, Line)
        if plot_type in ["Scatter Plot", "Line Plot (Time Series)"]:
            x_column = st.selectbox(f"Choose the X-axis column for {plot_type}", columns_for_plot)
            y_column = st.selectbox(f"Choose the Y-axis column for {plot_type}", columns_for_plot)

        # Handle the plots that need just one column
        elif plot_type in ["Histogram", "Density Plot", "Boxplot", "Violin Plot", "Bar Plot", "Pie Chart"]:
            selected_column = st.selectbox(f"Choose a column for the {plot_type}", columns_for_plot)

        # Function to save the plot
        def save_plot(fig, plot_type):
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            return buf

        # Handling different plot types
        if st.button(f"Generate {plot_type}"):
            st.session_state.active_tab = 1 
            try:
                if plot_type == "Histogram":
                    fig = v.plot_histogram(df, selected_column)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Density Plot":
                    fig = v.plot_density(df, selected_column)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Boxplot":
                    fig = v.plot_boxplot(df, selected_column)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Violin Plot":
                    fig = v.plot_violin(df, selected_column)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Bar Plot":
                    fig = v.plot_bar(df, selected_column)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Pie Chart":
                    fig = v.plot_pie(df, selected_column)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Scatter Plot":
                    fig = v.plot_scatter(df, x_column, y_column)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Correlation Heatmap":
                    # Identify non-numeric columns
                    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
                    
                    if non_numeric_cols:
                        st.error(f"⚠️ All features must be numeric for the Correlation Heatmap. Please encode or remove these columns: {', '.join(non_numeric_cols)}")
                    else:
                        fig = v.plot_correlation_heatmap(df)
                        st.pyplot(fig, use_container_width=True)
                        plot_buffer = save_plot(fig, plot_type)
                        st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Pairplot":
                    fig = v.plot_pairplot(df)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Line Plot (Time Series)":
                    fig = v.plot_line(df, x_column, y_column)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")
                elif plot_type == "Missing Heatmap":
                    fig = v.plot_missing_heatmap(df)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Missing Barplot":
                    fig = v.plot_missing_barplot(df)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Word Cloud":
                    text = ' '.join(df.select_dtypes(include=['object']).fillna('').apply(lambda x: ' '.join(x), axis=1))
                    fig = v.plot_wordcloud(text)
                    st.pyplot(fig, use_container_width=True)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")



    # ==============================
    # 3. تبويب Models
    # ==============================

    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_absolute_error

    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor

    
# ========= Model Selection & Training =========
    if selected_tab == "🤖 Models":
        st.header("🤖 Model Selection & Training")

        # Check if preprocessing is done
        if not all(k in st.session_state for k in ['X_train', 'X_test', 'y_train', 'y_test', 'target_col']):
            st.warning("⚠️ Please complete preprocessing steps first: Set target variable and split dataset.")
        else:
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test

            # ========== Load Pre-trained Model ==========
            st.subheader("📂 Load a Pre-trained Model")
            uploaded_model = st.file_uploader("Upload a .pkl model file", type=["pkl"])

            if uploaded_model is not None:
                try:
                    loaded_model = pickle.load(uploaded_model)
                    st.session_state.trained_model = loaded_model
                    st.success("✅ Model loaded successfully!")

                    if st.button("🔮 Predict with Loaded Model"):
                        try:
                            y_pred = loaded_model.predict(X_test)
                            st.session_state.y_pred = y_pred

                            st.write("### 🧾 Predictions")
                            st.write(pd.DataFrame({
                                "True": y_test,
                                "Predicted": y_pred
                            }))

                            st.subheader("📊 Evaluation Metrics")
                            from sklearn.base import ClassifierMixin, RegressorMixin

                            if isinstance(loaded_model, ClassifierMixin):
                                st.write("Accuracy:", accuracy_score(y_test, y_pred))
                                st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
                                st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
                                st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
                            elif isinstance(loaded_model, RegressorMixin):
                                st.write("R² Score:", r2_score(y_test, y_pred))
                                st.write("MAE:", mean_absolute_error(y_test, y_pred))
                            else:
                                st.warning("⚠️ Unsupported model type for evaluation.")

                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")
                except Exception as e:
                    st.error(f"❌ Failed to load model: {e}")

            st.divider()

            # ========== Train New Model ==========
            st.subheader("🧠 Train a New Model")

            task_type = st.radio("Select task type", ["Classification", "Regression"], horizontal=True)

            classification_models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Support Vector Machine": SVC(probability=True),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            }

            regression_models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
                "Support Vector Regressor": SVR(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor()
            }

            model_options = list(classification_models.keys()) if task_type == "Classification" else list(regression_models.keys())
            model_name = st.selectbox("Select a model", model_options)

            st.markdown("#### 🚦 Training")
            if st.button("🚀 Train Model"):
                try:
                    # Validate feature types
                    non_numeric_cols = [col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])]
                    if non_numeric_cols:
                        st.warning(f"⚠️ All features must be numeric. Please encode: {', '.join(non_numeric_cols)}")
                    else:
                        model = classification_models[model_name] if task_type == "Classification" else regression_models[model_name]
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        st.session_state.trained_model = model
                        st.session_state.y_pred = y_pred

                        st.success(f"✅ {model_name} trained successfully!")

                        st.write("### 🧾 True vs Predicted")
                        st.write(pd.DataFrame({
                            "True": y_test,
                            "Predicted": y_pred
                        }))

                        st.subheader("📊 Evaluation Metrics")
                        if task_type == "Classification":
                            st.write("Accuracy:", accuracy_score(y_test, y_pred))
                            st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
                            st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
                            st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
                        else:
                            st.write("R² Score:", r2_score(y_test, y_pred))
                            st.write("MAE:", mean_absolute_error(y_test, y_pred))

                        # Save and offer download
                        model_filename = f"{model_name.replace(' ', '_')}.pkl"
                        os.makedirs("models", exist_ok=True)
                        model_path = os.path.join("models", model_filename)
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)

                        buffer = io.BytesIO()
                        pickle.dump(model, buffer)
                        buffer.seek(0)
                        st.download_button(
                            label="📥 Download Trained Model",
                            data=buffer,
                            file_name=model_filename,
                            mime="application/octet-stream"
                        )

                except Exception as e:
                    st.error(f"❌ Model training failed: {e}")

            # ========== Manual Input Prediction ==========
            if "trained_model" in st.session_state:
                st.subheader("🧪 Predict on Custom Input")
                feature_names = st.session_state.X_train.columns

                user_input = {}
                for feature in feature_names:
                    dtype = st.session_state.X_train[feature].dtype
                    if dtype == 'object' or dtype.name == 'category':
                        unique_vals = st.session_state.X_train[feature].unique()
                        user_input[feature] = st.selectbox(f"{feature} (categorical)", unique_vals)
                    else:
                        user_input[feature] = st.number_input(f"{feature} (numeric)", value=0.0)

                if st.button("🔍 Predict on Input"):
                    input_df = pd.DataFrame([user_input])
                    try:
                        prediction = st.session_state.trained_model.predict(input_df)
                        st.success(f"✅ Predicted Value: {prediction[0]}")
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")

# ==============================
# 3. تبويب chatbot
# ==============================

    if selected_tab == "🦾 ChatBot":
        st.header("🦾 YourAnalyst Assistant")
        st.subheader("🔍 Analyze Dataset and Suggest Steps")

        HARDCODED_OPENROUTER_API_KEY = "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"
        HARDCODED_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
        HARDCODED_OPENROUTER_MODEL_NAME = "meta-llama/llama-3.3-8b-instruct:free"

        df = st.session_state.get("df", df)
        def analyze_dataset(df):
            suggestions = {
                "Missing Values": [],
                "Duplicate Rows": [],
                "Low Variance Features": [],
                "Categorical Features": [],
                "Outliers": [],
                "Correlation and Visualization": [],
                "Model Suggestions": []
            }

            # Missing Values
            missing = df.isnull().sum()
            if missing.any():
                suggestions["Missing Values"].append("❗ Some columns contain missing values. Consider using the 'Handle Missing Values' section in your dashboard.")
                suggestions["Missing Values"].extend([
                    "🔧 Missing values can be handled in several ways:",
                    "- Delete rows if the percentage of missing values is small.",
                    "- Impute with the mean or median for numerical columns.",
                    "- Impute with the mode for categorical columns.",
                    "- Use advanced techniques like KNN Imputer for large, complex datasets."
                ])

            # Duplicate Rows
            if df.duplicated().sum() > 0:
                suggestions["Duplicate Rows"].append("🔁 Duplicate rows found. Consider using the 'Remove Duplicates' section in your dashboard.")

            # Low Variance Features
            try:
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    selector = VarianceThreshold(threshold=0.0)
                    selector.fit(numeric_df)
                    low_var = [col for i, col in enumerate(numeric_df.columns) if selector.variances_[i] == 0.0]
                    if low_var:
                        suggestions["Low Variance Features"].append(
                            f"⚠️ The following columns have almost constant values (zero variance): {', '.join(low_var)}. They might not be useful for analysis."
                        )
            except Exception as e:
                print(f"Error in low variance check: {e}")

            # Categorical Features
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                suggestions["Categorical Features"].append(
                    f"🧩 The following columns are categorical: {', '.join(cat_cols)}. You might need to 'Encode Features' before using some models."
                )

            # Outliers
            numeric_cols_outliers = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols_outliers:
                if df[col].empty or df[col].isnull().all():
                    continue
                try:
                    if df[col].dropna().nunique() > 1:
                        z_scores = np.abs(stats.zscore(df[col].dropna()))
                        outliers_count = (z_scores > 3).sum()
                        if outliers_count > 0:
                            suggestions["Outliers"].append(
                                f"🚨 The column '{col}' contains approximately {outliers_count} outliers (using Z-score > 3). Consider using the 'Handle Outliers' section."
                            )
                    elif df[col].dropna().nunique() == 1 and len(df[col].dropna()) > 1:
                        suggestions["Outliers"].append(
                            f"ℹ️ The column '{col}' contains a single unique value (after removing NaN). Z-score cannot be effectively calculated."
                        )
                except Exception as e:
                    print(f"Error processing outliers for column {col}: {e}")

            # Correlation and Visualization
            try:
                cols_for_viz = df.select_dtypes(include=[np.number, "object", "category"]).columns.tolist()
                viz_suggestions_count = 0
                max_viz_suggestions = 5
                for i, col1 in enumerate(cols_for_viz):
                    if viz_suggestions_count >= max_viz_suggestions:
                        break
                    for j in range(i + 1, len(cols_for_viz)):
                        if viz_suggestions_count >= max_viz_suggestions:
                            break
                        col2 = cols_for_viz[j]
                        if col1 not in df.columns or col2 not in df.columns:
                            continue
                        col1_type = df[col1].dtype
                        col2_type = df[col2].dtype

                        if pd.api.types.is_numeric_dtype(col1_type) and pd.api.types.is_numeric_dtype(col2_type):
                            try:
                                if df[col1].nunique() > 1 and df[col2].nunique() > 1:
                                    corr = df[[col1, col2]].corr().iloc[0, 1]
                                    if abs(corr) > 0.7:
                                        suggestions["Correlation and Visualization"].append(
                                            f"📈 Strong correlation (correlation: {corr:.2f}) between '{col1}' and '{col2}'. Try a Scatter Plot or Line Plot."
                                        )
                                        viz_suggestions_count += 1
                            except Exception as e:
                                print(f"Error calculating correlation for {col1} and {col2}: {e}")
                        elif (pd.api.types.is_numeric_dtype(col1_type) and pd.api.types.is_object_dtype(col2_type)) or \
                            (pd.api.types.is_object_dtype(col1_type) and pd.api.types.is_numeric_dtype(col2_type)):
                            num_col, cat_col = (col1, col2) if pd.api.types.is_numeric_dtype(col1_type) else (col2, col1)
                            if df[cat_col].nunique() < 20:
                                suggestions["Correlation and Visualization"].append(
                                    f"📊 '{cat_col}' is categorical and '{num_col}' is numerical. Try a Boxplot or Bar Plot (with aggregation)."
                                )
                                viz_suggestions_count += 1
                        elif pd.api.types.is_object_dtype(col1_type) and pd.api.types.is_object_dtype(col2_type):
                            if df[col1].nunique() < 20 and df[col2].nunique() < 20:
                                suggestions["Correlation and Visualization"].append(
                                    f"🧩 '{col1}' and '{col2}' are categorical. Try a Heatmap (crosstab) or a stacked Count Plot."
                                )
                                viz_suggestions_count += 1
            except Exception as e:
                print(f"Error in visual analysis suggestions: {e}")

            # Model Suggestions — Updated to use st.session_state
            target_col = st.session_state.get("target_col")
            if target_col and target_col in df.columns:
                target_type = df[target_col].dtype
                if pd.api.types.is_numeric_dtype(target_type):
                    suggestions["Model Suggestions"].append(
                        f"🤖 The target (`{target_col}`) appears to be numerical. Try Regression models like Linear Regression or XGBoost Regressor."
                    )
                else:
                    suggestions["Model Suggestions"].append(
                        f"🤖 The target (`{target_col}`) appears to be categorical. Try Classification models like Random Forest or XGBoost Classifier."
                    )
            else:
                suggestions["Model Suggestions"].append(
                    "🎯 No target variable set. Please select one from the '🎯 Set Target Variable' option."
                )

            return suggestions


        client = None
        if HARDCODED_OPENROUTER_API_KEY and HARDCODED_OPENROUTER_API_KEY != "sk-or-v1-PUT_YOUR_NEW_SECURE_OPENROUTER_KEY_HERE":
            try:
                client = OpenAI(
                    api_key=HARDCODED_OPENROUTER_API_KEY,
                    base_url=HARDCODED_OPENROUTER_BASE_URL
                )
            except Exception as e:
                st.error(f"❌ Error initializing OpenRouter client: {e}")
        else:
            st.error("⚠️ A valid OpenRouter API key has not been entered in the script variables (HARDCODED_OPENROUTER_API_KEY). The ChatBot will not work.")

        def ask_llm(question, df_preview_text):
            if not client:
                return "Sorry, the ChatBot service is not properly initialized. Please check the hardcoded OpenRouter values in the script and ensure they are correct."
            prompt = f"""This is a preview of the first 100 rows of the dataset:\n{df_preview_text}\n\nMy question is: {question}"""
            try:
                response = client.chat.completions.create(
                    model=HARDCODED_OPENROUTER_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an intelligent data analysis assistant. Answer questions based on the provided table data. Respond in English."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"❌ An error occurred while communicating with OpenRouter: {e}"

        st.markdown("""
            Welcome! 🎉 I’m your smart assistant. Your dataset is ready for analysis. I’ll help you with suggestions for data cleaning, visualizations, and possible model types.

            If you'd like model recommendations, make sure your dataset includes a column named 'target' for supervised learning tasks.
        """)

        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📊 Display first 5 rows of data", expanded=True):
                st.dataframe(df.head())

        with col2:
            st.subheader("💡 Assistant's Suggestions (Automatic Analysis):")
            suggestions = analyze_dataset(df)
            if any(s for s_list in suggestions.values() for s in s_list):
                for section, items in suggestions.items():
                    if items:
                        with st.expander(f"📌 {section}"):
                            for s_item in items:
                                st.markdown(f"- {s_item}")
            else:
                st.markdown("✅ No obvious issues found in the data based on automatic analysis.")

        st.subheader(f"🤔 Ask the Assistant (via OpenRouter - Model: {HARDCODED_OPENROUTER_MODEL_NAME}):")
        user_question = st.text_input("💬 Ask any question about the data:")

        if user_question and client:
            with st.spinner("⏳ Thinking..."):
                df_preview_for_llm = df.head(100).to_csv(index=False)
                answer = ask_llm(user_question, df_preview_for_llm)
                st.markdown(f"🤖 {answer}")
        elif user_question and not client:
            st.warning("⚠️ ChatBot not initialized. Please check the hardcoded OpenRouter API values in the script and ensure they are correct.")
