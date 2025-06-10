import streamlit as st
import pandas as pd
from Preprocessing_Functions import all_function as af
import numpy as np # Added based on usage in YourAnalyst.py
from scipy import stats # Added based on usage in YourAnalyst.py

def show_preprocessing_tab(df):
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
        df_display = st.session_state.get("df", df) # Use df passed to function or from session
        if show_all:
            st.dataframe(df_display)
        else:
            st.write("### 🔝 First 10 Rows")
            st.dataframe(df_display.head(10))
            st.write("### 🔚 Last 10 Rows")
            st.dataframe(df_display.tail(10))

    elif option == "📌 List All Columns":
        df_display = st.session_state.get("df", df)
        af.list_all_columns(df_display)

    elif option == "📈 Column Info":
        df_display = st.session_state.get("df", df)
        col = st.selectbox("Choose column:", df_display.columns)
        af.view_column_info(df_display, col)

    elif option == "📊 Feature Importance":
        st.subheader("📊 Feature Importance (via Random Forest)")
        df_display = st.session_state.get("df", df)
        target = st.selectbox("Select the target column:", df_display.columns)
        if target:
            importance = af.get_feature_importance(df_display, target)
            st.bar_chart(importance.head(20))
            st.dataframe(importance.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}))

    elif option == "🔍 Check Class Imbalance":
        st.subheader("🔍 Search for Class Imbalance")
        df_display = st.session_state.get("df", df)
        target_col = st.selectbox("Select the target column:", df_display.columns)
        if target_col: # Ensure target_col is selected
            class_counts = af.check_class_imbalance(df_display, target_col)
            st.bar_chart(class_counts)
            st.dataframe(class_counts.reset_index().rename(columns={'index': 'Class', target_col: 'Count'}))
            imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
            if imbalance_ratio > 1.5:
                st.warning(f"⚠️ Class imbalance detected (max/min ratio = {imbalance_ratio:.2f}). Consider balancing techniques like SMOTE.")
            else:
                st.success("✅ Class distribution looks reasonably balanced.")

    elif option == "🎯 Set Target Variable":
        st.subheader("🎯 Select Target (Prediction) Column")
        df_display = st.session_state.get("df", df)
        target_col = st.selectbox("Select your target variable (what you're predicting):", df_display.columns)
        if st.button("Confirm Target Column"):
            st.session_state.target_col = target_col
            st.success(f"🎯 Target variable set to: `{target_col}`")

    elif option == "📦 Split Dataset (Train/Test)":
        st.subheader("📦 Split Dataset into Train/Test")
        df_display = st.session_state.get("df", df)
        if "target_col" not in st.session_state:
            st.warning("⚠️ Please set a target column first in the 🎯 section.")
        else:
            test_size = st.slider("Test Size (%)", min_value=5, max_value=50, value=20, step=5)
            # train_size = st.slider("Train Size (%)", min_value=10, max_value=90, value=80, step=5) # train_size is usually 1 - test_size
            train_size = 100 - test_size
            st.info(f"Train size will be {train_size}%")
            stratify = st.checkbox("Use Stratified Split (for classification)?")

            if st.button("Split Dataset"):
                X_train, X_test, y_train, y_test = af.split_dataset(
                st.session_state.df, # use the df from session_state as it might have been modified
                st.session_state.target_col,
                test_size=test_size / 100,
                train_size=train_size / 100, # Corrected
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
        df_current = st.session_state.get("df", df).copy() # Operate on a copy
        datetime_cols = df_current.select_dtypes(include=["datetime64[ns]", "object"]).columns
        datetime_col = st.selectbox("Select a datetime column:", datetime_cols)

        if st.button("Extract Features"):
            df_updated = af.extract_datetime_features(df_current, datetime_col)
            st.session_state.df = df_updated # Update session state
            st.dataframe(df_updated)
        # st.session_state.df = df_current # This was potentially overwriting changes if button not pressed

    elif option == "🔁 Change Column Data Types":
        st.subheader("🔁 Change Column Data Types")
        df_current = st.session_state.get("df", df).copy()
        if st.checkbox("Select all columns"):
            columns = st.multiselect("Select columns:", df_current.columns, default=list(df_current.columns))
        else:
            columns = st.multiselect("Select columns:", df_current.columns)

        dtype_options = ["int", "float", "object", "datetime", "category"]
        new_dtype = st.selectbox("Convert to data type:", dtype_options)

        if st.button("Apply Type Conversion"):
            df_updated = af.change_columns_dtype(df_current, columns, new_dtype)
            st.session_state.df = df_updated
            st.success("✅ Type conversion applied.")
            st.dataframe(df_updated)

    elif option == "📊 Dataset Summary":
        st.subheader("📊 Dataset Summary & Columns Info")
        df_display = st.session_state.get("df", df)
        af.view_all_columns_summary(df_display)

    elif option == "🧹 Remove Duplicates":
        st.subheader("🧹 Remove Duplicate Rows")
        df_current = st.session_state.get("df", df).copy()
        num_duplicates = af.check_duplicates(df_current)

        if num_duplicates == 0:
            st.info("✅ No duplicate rows found in the dataset.")
        else:
            st.warning(f"⚠️ Found {num_duplicates} duplicate rows in your dataset.")
            if st.button("🧹 Remove Duplicates Now"):
                df_updated, removed = af.remove_duplicates(df_current)
                st.session_state.df = df_updated
                st.success(f"✅ Removed {removed} duplicate rows.")
                st.dataframe(df_updated)

    elif option == "🔍 Detect Missing Values":
        st.subheader("🔍 Missing Values")
        df_display = st.session_state.get("df", df)
        missing_cols = af.get_missing_columns(df_display)
        if not missing_cols:
            st.info("✅ No missing values found in the dataset.")
        else:
            st.warning("⚠️ Missing values found in the following columns:")
            for col_name in missing_cols: # Renamed col to col_name to avoid conflict
                dtype = df_display[col_name].dtype
                missing_count = df_display[col_name].isnull().sum()
                st.markdown(f"- **{col_name}** *(dtype: `{dtype}`, missing: `{missing_count}`)*")

    elif option == "📉 Correlation Analysis":
        st.subheader("📉 Correlation Analysis")
        df_current = st.session_state.get("df", df).copy()
        threshold = st.slider("Correlation Threshold (absolute)", min_value=0.5, max_value=1.0, value=0.9, step=0.01)
        to_drop, corr_matrix = af.detect_highly_correlated(df_current, threshold)

        st.write("Correlation Matrix:")
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

        if not to_drop:
            st.info("✅ No highly correlated features found.")
        else:
            st.warning(f"⚠️ {len(to_drop)} features exceed the correlation threshold.")
            st.write(to_drop)
            if st.button("Remove Correlated Features"):
                df_updated = af.remove_correlated_features(df_current, to_drop)
                st.session_state.df = df_updated
                st.dataframe(df_updated)

    elif option == "🧬 Feature Interaction or Generation":
        st.subheader("🧬 Feature Interaction / Generation")
        df_current = st.session_state.get("df", df).copy()
        st.markdown("Example: `(df['Feature1'] + df['Feature2']) / 2`")
        cols_list = df_current.columns.tolist() # Renamed cols to cols_list

        with st.expander("➕ Generate New Feature", expanded=True):
            col1 = st.selectbox("Select first column:", cols_list, key="gen_col1")
            col2 = st.selectbox("Select second column:", cols_list, key="gen_col2")
            operation = st.radio("Select operation:", ["add", "subtract", "multiply", "divide"], horizontal=True)
            new_col_name = st.text_input("Enter name for the new feature:", key="new_feature_name")

            col1_btn, col2_btn = st.columns(2) # Renamed cols to col1_btn, col2_btn
            with col1_btn:
                if st.button("✅ Generate Feature"):
                    if new_col_name.strip() == "":
                        st.error("⚠️ Please enter a valid name for the new feature.")
                    elif new_col_name in df_current.columns:
                        st.error("⚠️ A column with this name already exists.")
                    else:
                        st.session_state.prev_df = df_current.copy()
                        df_updated = af.generate_feature(df_current, col1, col2, operation, new_col_name)
                        st.session_state.df = df_updated
                        st.dataframe(df_updated.head())
            with col2_btn:
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
        df_current = st.session_state.get("df", df).copy()
        threshold = st.slider("Variance Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        low_var_cols = af.detect_low_variance_features(df_current, threshold)

        if not low_var_cols:
            st.success("✅ No low-variance features detected.")
        else:
            st.warning(f"⚠️ Found {len(low_var_cols)} low-variance feature(s):")
            st.dataframe(pd.DataFrame(low_var_cols, columns=["Low-Variance Features"]))
            if st.button("🗑️ Remove Low-Variance Features"):
                df_updated = af.remove_low_variance_features(df_current, low_var_cols)
                st.session_state.df = df_updated
                st.success(f"Removed {len(low_var_cols)} low-variance feature(s). Updated dataframe:")
                st.dataframe(df_updated)

    elif option == "🔧 Handle Missing Values":
        st.subheader("🔧 Handle Missing Values")
        df_current = st.session_state.get("df", df).copy()
        missing_cols = af.get_missing_columns(df_current)
        if not missing_cols:
            st.info("✅ No missing values found in the dataset.")
        else:
            selected_cols = st.multiselect("Select columns with missing values:", missing_cols)
            methods_dict = {}
            custom_values = {}
            for col_name in selected_cols: # Renamed col to col_name
                col_type = df_current[col_name].dtype
                st.markdown(f"**Column: `{col_name}` ({col_type})**")
                if pd.api.types.is_numeric_dtype(df_current[col_name]):
                    options = ["Drop", "Mean", "Median", "Custom"]
                else:
                    options = ["Drop", "Mode", "Custom"]
                method = st.selectbox(f"Choose method for `{col_name}`:", options, key=f"method_{col_name}")
                methods_dict[col_name] = method
                if method == "Custom":
                    custom_value = st.text_input(f"Enter custom fill for `{col_name}`:", key=f"custom_{col_name}")
                    custom_values[col_name] = custom_value
            if st.button("Apply Missing Value Handling"):
                df_updated = df_current.copy() # operate on a copy for multiple changes
                for col_name in selected_cols: # Renamed col to col_name
                    df_updated = af.handle_missing_column(df_updated, col_name, methods_dict[col_name], custom_values.get(col_name))
                st.session_state.df = df_updated
                st.success("✅ Missing values handled successfully.")
                st.dataframe(df_updated)

    elif option == "💡 Encode Features":
        st.subheader("💡 Encode Features")
        df_current = st.session_state.get("df", df).copy()
        cat_cols = df_current.select_dtypes(include=['object', 'category']).columns.tolist()
        if st.checkbox("Select all categorical columns"):
            cols_to_encode = st.multiselect("Select columns to encode", cat_cols, default=cat_cols) # Renamed cols
        else:
            cols_to_encode = st.multiselect("Select columns to encode", cat_cols) # Renamed cols
        method = st.selectbox("Encoding method", ['label', 'onehot', 'ordinal'])
        if st.button("Apply Encoding"):
            df_updated = af.encode_features(df_current, cols_to_encode, method)
            st.session_state.df = df_updated
            st.dataframe(df_updated)

    elif option == "🗑️ Remove Columns":
        st.subheader("🗑️ Remove Columns")
        df_current = st.session_state.get("df", df).copy()
        cols_to_remove = st.multiselect("Select columns to remove", df_current.columns) # Renamed cols
        if st.button("Remove Columns"):
            df_updated = af.remove_columns(df_current, cols_to_remove)
            st.session_state.df = df_updated
            st.success("✅ Columns removed.")
            st.dataframe(df_updated)

    elif option == "✏️ Rename Columns":
        st.subheader("✏️ Rename Columns")
        df_current = st.session_state.get("df", df).copy()
        col_to_rename = st.selectbox("Choose column to rename:", df_current.columns)
        new_name = st.text_input(f"Enter new name for `{col_to_rename}`:")
        if st.button("Rename Column"):
            if new_name:
                df_updated = af.rename_columns(df_current, {col_to_rename: new_name})
                st.session_state.df = df_updated
                st.success(f"✅ Column `{col_to_rename}` renamed to `{new_name}`.")
                st.dataframe(df_updated)
            else:
                st.error("⚠️ Please provide a new name for the column.")

    elif option == "🧠 Handle Outliers":
        st.subheader("🧠 Handle Outliers")
        df_current = st.session_state.get("df", df).copy()
        col_to_check = st.selectbox("Choose column to check for outliers:", df_current.columns)
        method = st.selectbox("Outlier detection method", ['IQR', 'zscore', 'isolation_forest'])
        if st.button("Handle Outliers"):
            if col_to_check and method:
                df_updated = af.handle_outliers(df_current, col_to_check, method)
                st.session_state.df = df_updated
                st.success(f"✅ Outliers handled in column `{col_to_check}` using `{method}` method.")
                st.dataframe(df_updated)
            else:
                st.error("⚠️ Please select a column and a method for outlier handling.")

    elif option == "🧾 Check Data Integrity":
        st.subheader("🧾 Check Data Integrity")
        df_display = st.session_state.get("df", df)
        result = af.check_data_integrity(df_display)
        st.subheader("📊 Data Integrity Report")
        for key, value in result.items():
            st.write(f"**{key}:** {value}")

    elif option == "📥 Download Cleaned Dataset":
        df_display = st.session_state.get("df", df)
        st.subheader("📥 Download Cleaned Dataset")
        file_format = st.selectbox("Choose format", ['csv', 'excel']) # Renamed format to file_format
        file_name = st.text_input("Enter filename (without extension)", "cleaned_dataset")

        if file_format == "csv":
            csv_data = df_display.to_csv(index=False).encode('utf-8') # Renamed csv to csv_data
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"{file_name}.csv" if file_name else "cleaned_dataset.csv",
                mime='text/csv'
            )
        elif file_format == "excel":
            excel_buffer = af.convert_df_to_excel(df_display)
            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer,
                file_name=f"{file_name}.xlsx" if file_name else "cleaned_dataset.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

    elif option == "🔄 Replace Values":
        st.subheader("🔄 Replace Values in Columns")
        df_current = st.session_state.get("df", df).copy()
        cols_to_modify = st.multiselect("Select column(s) to modify:", df_current.columns.tolist()) # Renamed cols
        to_replace = st.text_input("Value to replace (exact match):")
        replacement = st.text_input("Replacement value:")
        if st.button("Apply Replacement"):
            if not cols_to_modify:
                st.warning("Please select at least one column.")
            else:
                df_updated = af.replace_values_in_columns(df_current, cols_to_modify, to_replace, replacement)
                st.session_state.df = df_updated
                st.success(f"Replaced **{to_replace}** with **{replacement}** in {len(cols_to_modify)} column(s).")
                st.dataframe(df_updated.head())

    elif option == "🔢 Scale/Normalize Features":
        st.subheader("🔢 Scale or Normalize Features")
        df_current = st.session_state.get("df", df).copy()
        num_cols = df_current.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if st.checkbox("Select all numerical columns"):
            columns_to_scale = st.multiselect("Select columns to scale/normalize", num_cols, default=num_cols) # Renamed columns
        else:
            columns_to_scale = st.multiselect("Select columns to scale/normalize", num_cols) # Renamed columns
        method = st.selectbox("Scaling method:", ["minmax", "standard", "robust"])
        if st.button("Apply Scaling/Normalization"):
            if not columns_to_scale:
                st.warning("Please select at least one column to scale.")
            else:
                df_updated = af.scale_features(df_current, columns_to_scale, method)
                st.session_state.df = df_updated
                st.dataframe(df_updated)
