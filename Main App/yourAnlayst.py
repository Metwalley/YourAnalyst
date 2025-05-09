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



st.markdown(
    """
    <style>
    /* تخصيص الخلفية العامة */
    .reportview-container {
        background-color: #212121;  /* خلفية داكنة */
        color: #e0e0fd;  /* لون النص الفاتح */
    }

    /* تخصيص الخلفية الجانبية */
    .sidebar .sidebar-content {
        background-color: #e0e0fd;  /* خلفية داكنة للشريط الجانبي */
        color: #e0e0fd;  /* لون النص الفاتح */
    }

    /* تخصيص الأزرار */
    .stButton>button {
        cursor: pointer;
        border: solid rgb(187, 204, 0);
        font-family: "system-ui";
        font-size: 14px;
        color: rgb(255, 255, 255);
        padding: 10px 30px;
        transition: 2s;
        width: 335px;
        box-shadow: rgb(0, 0, 0) 0px 0px 0px 0px;
        border-radius: 50px;
        background: linear-gradient(90deg, rgb(0, 102, 204) 0%, rgb(197, 0, 204) 100%);
    }

    .stButton>button:hover {
        color: rgb(255, 255, 255);
        width: 337px;
        background: rgb(0, 102, 204) none repeat scroll 0% 0% / auto padding-box border-box;
        border-color: rgb(204, 0, 105);
        border-width: 1px;
        border-style: solid;
    }

    /* تخصيص عنوان التبويب */
    .css-1c6jbr4 {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 26px;
        font-weight: bold;
        color: #e0e0fd;  /* لون أبيض للعناوين */
    }

    /* تخصيص الفونت العام */
    body {
        font-family: 'Roboto', sans-serif;
    }

    /* تخصيص جداول البيانات */
    .stDataFrame {
        background-color: #424242;  /* خلفية داكنة للجداول */
        color: #e0e0fd;  /* لون النص الفاتح في الجداول */
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* تخصيص الترويسة الرئيسية */
    .stTitle {
        font-family: 'Arial', sans-serif;
        font-size: 34px;
        color: #e0e0fd;  /* لون الأزرق للألقاب الرئيسية */
        font-weight: bold;
    }

    /* تخصيص الفاصل بين التبويبات */
    .stTab {
        border: 2px solid #e0e0fd;
        border-radius: 10px;
        background-color: #333333; 
        padding: 10px;
    }

    /* تخصيص الترويسات داخل التبويبات */
    .stHeader {
        color: #e0e0fd;  
        font-size: 22px;
        font-weight: 600;
    }

    /* تخصيص العناوين داخل التبويبات */
    .stSubheader {
        color: #e0e0fd;  /* لون فاتح للعناوين الفرعية */
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)





# ========== العنوان الرئيسي ==========
st.title("🌟 YourAnalyst ")

# ========== تحميل الداتا ==========
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "excel", "json", "parquet"])

if uploaded_file is not None:
    file_type = st.selectbox("Select file type", ["csv", "excel", "json", "parquet"])

    if "df" not in st.session_state:
        st.session_state.df = af.load_data(uploaded_file, file_type)

    df = st.session_state.df

    st.success("✅ Dataset uploaded successfully!")

    # ========== التابز ==========
    tabs = st.tabs(["🛠️ Preprocessing", "📊 Visualization", "🤖 Models"])

    # ==============================
    # 1. تبويب Preprocessing
    # ==============================
    with tabs[0]:
        st.header("🛠️ Data Preprocessing")

        option = st.radio(
            "Select a preprocessing task:",
            (
                "📄 View Dataset",
                "📌 List All Columns",
                "📈 Column Info",
                "🧾 Check Data Integrity",
                "📊 Dataset Summary",
                "🧹 Remove Duplicates",
                "🔍 Detect Missing Values",
                "🔧 Handle Missing Values",
                "💡 Encode Features",
                "🗑️ Remove Columns",
                "✏️ Rename Columns",
                "🧠 Handle Outliers",
                "🔄 Replace Values",        # ← NEW
                "🔁 Change Column Data Types", # ← NEW
                "🕓 Extract Datetime Features", # ← NEW
                "🔢 Scale/Normalize Features", # ← NEW prepFor ML
                "🧬 Feature Interaction or Generation", # ← NEW prepFor ML
                "🧽 Detect & Handle Low-Variance Features", # ← NEW prepFor ML
                "📉 Correlation Analysis / Multicollinearity Detection", # ← NEW prepFor ML
                "📊 Feature Importance", # ← NEW prepFor ML
                "🔍 Check Class Imbalance", # ← NEW prepFor ML
                "🎯 Set Target Variable", # ← NEW prepFor ML
                "📦 Split Dataset (Train/Test)", # ← NEW prepFor ML
                "📥 Download Cleaned Dataset"
            )
        )

        if option == "📄 View Dataset":
            st.subheader("📄 Dataset Preview")

            show_all = st.checkbox("🔁 Show full dataset", value=False)

            if show_all:
                st.dataframe(df)
            else:
                st.write("### 🔝 First 10 Rows")
                st.dataframe(df.head(10))

                st.write("### 🔚 Last 10 Rows")
                st.dataframe(df.tail(10))


        elif option == "📌 List All Columns":
            af.list_all_columns(df)

        elif option == "📈 Column Info":
            col = st.selectbox("Choose column:", df.columns)
            af.view_column_info(df, col)
        
        elif option == "📊 Feature Importance":
            st.subheader("📊 Feature Importance (via Random Forest)")

            target = st.selectbox("Select the target column:", df.columns)
            if target:
                importance = af.get_feature_importance(df, target)
                st.bar_chart(importance.head(20))
                st.dataframe(importance.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}))

        elif option == "🔍 Check Class Imbalance":
            st.subheader("🔍 Search for Class Imbalance")

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

            target_col = st.selectbox("Select your target variable (what you're predicting):", df.columns)

            if st.button("Confirm Target Column"):
                st.session_state.target_col = target_col
                st.success(f"🎯 Target variable set to: `{target_col}`")

        elif option == "📦 Split Dataset (Train/Test)":
            st.subheader("📦 Split Dataset into Train/Test")

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
            datetime_cols = df.select_dtypes(include=["datetime64[ns]", "object"]).columns
            datetime_col = st.selectbox("Select a datetime column:", datetime_cols)

            if st.button("Extract Features"):
                df = af.extract_datetime_features(df, datetime_col)
                st.session_state.df = df
                st.dataframe(df)
    
            
        elif option == "🔁 Change Column Data Types":
            st.subheader("🔁 Change Column Data Types")

            if st.checkbox("Select all columns"):
                columns = st.multiselect("Select columns:", df.columns, default=list(df.columns))
            else:
                columns = st.multiselect("Select columns:", df.columns)

            dtype_options = ["int", "float", "object", "datetime", "category"]
            new_dtype = st.selectbox("Convert to data type:", dtype_options)

            if st.button("Apply Type Conversion"):
                df = af.change_columns_dtype(df, columns, new_dtype)
                st.session_state.df = df
                st.dataframe(df)



        elif option == "📊 Dataset Summary":
            af.view_all_columns_summary(df)

        elif option == "🧹 Remove Duplicates":
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
            missing_cols = af.get_missing_columns(df)
            if not missing_cols:
                st.info("✅ No missing values found in the dataset.")
            else:
                st.warning("⚠️ Missing values found in the following columns:")
                for col in missing_cols:
                    dtype = df[col].dtype
                    missing_count = df[col].isnull().sum()
                    st.markdown(f"- **{col}** *(dtype: `{dtype}`, missing: `{missing_count}`)*")

        elif option == "📉 Correlation Analysis / Multicollinearity Detection":
            st.subheader("📉 Correlation Analysis / Multicollinearity Detection")

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
                            st.success("🎉 Feature created successfully!")
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

        elif option == "🧽 Detect & Handle Low-Variance Features":
            st.subheader("🧽 Detect & Handle Constant or Low-Variance Features")

            threshold = st.slider("Variance Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            low_var_cols = af.detect_low_variance_features(df, threshold)

            if not low_var_cols:
                st.info("✅ No low-variance features detected.")
            else:
                st.warning(f"⚠️ Found {len(low_var_cols)} low-variance columns.")
                st.write(low_var_cols)

                if st.button("Remove Low-Variance Features"):
                    df = af.remove_low_variance_features(df, low_var_cols)
                    st.session_state.df = df
                    st.dataframe(df)



        elif option == "🔧 Handle Missing Values":
            st.subheader("🔧 Handle Missing Values")
            missing_cols = af.get_missing_columns(df)

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
            cols = st.multiselect("Select columns to remove", df.columns)
            if st.button("Remove Columns"):
                df = af.remove_columns(df, cols)
                st.session_state.df = df
                st.success("✅ Columns removed.")
                st.dataframe(df)

        elif option == "✏️ Rename Columns":
            st.subheader("✏️ Rename Columns")

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
            result = af.check_data_integrity(df)
            st.subheader("📊 Data Integrity Report")
            for key, value in result.items():
                st.write(f"**{key}:** {value}")


        elif option == "📥 Download Cleaned Dataset":
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
    with tabs[1]:
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
            try:
                if plot_type == "Histogram":
                    fig = v.plot_histogram(df, selected_column)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Density Plot":
                    fig = v.plot_density(df, selected_column)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Boxplot":
                    fig = v.plot_boxplot(df, selected_column)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Violin Plot":
                    fig = v.plot_violin(df, selected_column)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Bar Plot":
                    fig = v.plot_bar(df, selected_column)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Pie Chart":
                    fig = v.plot_pie(df, selected_column)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Scatter Plot":
                    fig = v.plot_scatter(df, x_column, y_column)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Correlation Heatmap":
                    fig = v.plot_correlation_heatmap(df)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Pairplot":
                    fig = v.plot_pairplot(df)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Line Plot (Time Series)":
                    fig = v.plot_line(df, x_column, y_column)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")
                elif plot_type == "Missing Heatmap":
                    fig = v.plot_missing_heatmap(df)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Missing Barplot":
                    fig = v.plot_missing_barplot(df)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")

                elif plot_type == "Word Cloud":
                    text = ' '.join(df.select_dtypes(include=['object']).fillna('').apply(lambda x: ' '.join(x), axis=1))
                    fig = v.plot_wordcloud(text)
                    st.pyplot(fig)
                    plot_buffer = save_plot(fig, plot_type)
                    st.download_button("Download Plot", plot_buffer, file_name=f"{plot_type}.png", mime="image/png")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # ==============================
    # 3. تبويب Models
    # ==============================import streamlit as st

    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_absolute_error

    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor

    # ========= Model Selection & Training =========
    with tabs[2]:
        st.header("🤖 Model Selection & Training")

        if not all(k in st.session_state for k in ['X_train', 'X_test', 'y_train', 'y_test', 'target_col']):
            st.warning("⚠️ Please complete preprocessing steps first: Set target variable and split dataset.")
        else:
            # ========== Load Models ==========
            st.subheader("📂 Load Pre-trained Models")
            uploaded_model = st.file_uploader("Upload a pre-trained model", type=["pkl"])

            if uploaded_model is not None:
                try:
                    model_bytes = uploaded_model.read()
                    loaded_model = pickle.loads(model_bytes)
                    st.session_state.trained_model = loaded_model
                    st.success("✅ Model loaded successfully!")
                    
                    if st.button("🔮 Predict with Loaded Model"):
                        y_pred = loaded_model.predict(st.session_state.X_test)
                        st.session_state.y_pred = y_pred
                        st.write("**Predictions**", pd.DataFrame({
                            "True Values": st.session_state.y_test,
                            "Predictions": y_pred
                        }))

                        st.subheader("📊 Evaluation Metrics")
                        if isinstance(loaded_model, (LogisticRegression, RandomForestClassifier, KNeighborsClassifier, SVC, DecisionTreeClassifier, GradientBoostingClassifier, XGBClassifier)):
                            if isinstance(st.session_state.y_test.iloc[0], (int, str)):
                                st.write("**Accuracy:**", accuracy_score(st.session_state.y_test, y_pred))
                                st.write("**F1 Score:**", f1_score(st.session_state.y_test, y_pred, average='weighted'))
                                st.write("**Precision:**", precision_score(st.session_state.y_test, y_pred, average='weighted'))
                                st.write("**Recall:**", recall_score(st.session_state.y_test, y_pred, average='weighted'))
                            else:
                                st.error("❌ Error: Target variable is not categorical. Please ensure you're using a classification model with categorical targets.")
                        
                        elif isinstance(loaded_model, (LinearRegression, RandomForestRegressor, KNeighborsRegressor, SVR, GradientBoostingRegressor, XGBRegressor)):
                            st.write("**R² Score:**", r2_score(st.session_state.y_test, y_pred))
                            st.write("**MAE:**", mean_absolute_error(st.session_state.y_test, y_pred))
                        else:
                            st.error("❌ Unsupported model type for evaluation.")
                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")

            # ========== Train New Model ========== 
            st.subheader("🔄 Train New Model")

            task_type = st.radio("Select ML Task Type:", ["Classification", "Regression"])

            if task_type == "Classification":
                model_name = st.selectbox("Choose a classification model:", [
                    "Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Support Vector Machine", 
                    "Decision Tree", "Gradient Boosting", "XGBoost"
                ])
            else:
                model_name = st.selectbox("Choose a regression model:", [
                    "Linear Regression", "Random Forest Regressor", "K-Nearest Neighbors Regressor", 
                    "Support Vector Regressor", "Gradient Boosting Regressor", "XGBoost Regressor"
                ])

            if st.button("🚀 Train Model"):
                if task_type == "Classification":
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier()
                    elif model_name == "K-Nearest Neighbors":
                        model = KNeighborsClassifier()
                    elif model_name == "Support Vector Machine":
                        model = SVC(probability=True)
                    elif model_name == "Decision Tree":
                        model = DecisionTreeClassifier()
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingClassifier()
                    elif model_name == "XGBoost":
                        model = XGBClassifier()
                else:
                    if model_name == "Linear Regression":
                        model = LinearRegression()
                    elif model_name == "Random Forest Regressor":
                        model = RandomForestRegressor()
                    elif model_name == "K-Nearest Neighbors Regressor":
                        model = KNeighborsRegressor()
                    elif model_name == "Support Vector Regressor":
                        model = SVR()
                    elif model_name == "Gradient Boosting Regressor":
                        model = GradientBoostingRegressor()
                    elif model_name == "XGBoost Regressor":
                        model = XGBRegressor()

                model.fit(st.session_state.X_train, st.session_state.y_train)
                y_pred = model.predict(st.session_state.X_test)

                st.session_state.trained_model = model
                st.session_state.y_pred = y_pred
                st.success(f"✅ {model_name} trained successfully!")

                # ========== عرض القيم الحقيقية والمتوقعة ==========
                st.write("### 🔍 True vs Predicted Values")
                st.write(pd.DataFrame({
                    "True Values": st.session_state.y_test,
                    "Predicted Values": y_pred
                }))

                # ========== Evaluation ==========
                st.subheader("📊 Evaluation Metrics")
                if task_type == "Classification":
                    st.write("**Accuracy:**", accuracy_score(st.session_state.y_test, y_pred))
                    st.write("**F1 Score:**", f1_score(st.session_state.y_test, y_pred, average='weighted'))
                    st.write("**Precision:**", precision_score(st.session_state.y_test, y_pred, average='weighted'))
                    st.write("**Recall:**", recall_score(st.session_state.y_test, y_pred, average='weighted'))
                else:
                    st.write("**R² Score:**", r2_score(st.session_state.y_test, y_pred))
                    st.write("**MAE:**", mean_absolute_error(st.session_state.y_test, y_pred))

                # ========== Save and Download ==========
                model_filename = f"{model_name}.pkl"
                model_path = os.path.join("models", model_filename)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)

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
                
