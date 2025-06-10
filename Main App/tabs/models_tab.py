import streamlit as st
import pandas as pd
import numpy as np # Added
import pickle
import os
import io
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             r2_score, mean_absolute_error, mean_squared_error) # Added mean_squared_error
from sklearn.base import ClassifierMixin, RegressorMixin
# import xgboost as xgb # Ensure this is how xgb is used if needed directly
from xgboost import XGBClassifier, XGBRegressor # More common import style

from visualization_Functions.visualize import plot_confusion_matrix, plot_roc_curve # New imports

# Helper function for model evaluation
def display_model_evaluation(model, X_test_data, y_test_data, y_pred_data, task_type, model_name="Model"):
    st.write(f"### 📊 Evaluation Metrics for {model_name}")

    if task_type == "Classification":
        st.write(f"Accuracy: {accuracy_score(y_test_data, y_pred_data):.4f}")
        st.write(f"F1 Score (weighted): {f1_score(y_test_data, y_pred_data, average='weighted', zero_division=0):.4f}")
        st.write(f"Precision (weighted): {precision_score(y_test_data, y_pred_data, average='weighted', zero_division=0):.4f}")
        st.write(f"Recall (weighted): {recall_score(y_test_data, y_pred_data, average='weighted', zero_division=0):.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        if hasattr(model, 'classes_'):
            class_names = model.classes_
        else:
            # Create class_names from unique values in y_test_data and y_pred_data
            # Convert to list and sort for consistent ordering
            class_names = sorted(list(pd.unique(np.concatenate((y_test_data, y_pred_data)))))

        cm_fig = plot_confusion_matrix(y_test_data, y_pred_data, class_names=class_names)
        if cm_fig:
            st.pyplot(cm_fig)
        else:
            st.warning("Could not generate Confusion Matrix.")

        # ROC Curve & AUC
        st.subheader("ROC Curve & AUC")
        y_scores_for_roc = None
        # Check if the model can provide probability scores
        if hasattr(model, "predict_proba"):
            try:
                y_pred_proba = model.predict_proba(X_test_data)
                # For binary classification, use probabilities of the positive class
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    y_scores_for_roc = y_pred_proba[:, 1]
                # For multi-class, inform user or implement advanced ROC (e.g. OvR)
                elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 2:
                    st.info("ROC/AUC for multi-class classification is typically calculated per class (One-vs-Rest). This plot is primarily for binary tasks. Displaying ROC for the first class vs rest as an example if possible, or averaged if supported by your `plot_roc_curve` function for multi-class scores.")
                    # Defaulting to scores for the first class if multi-class probabilities are directly passed
                    # This might need adjustment based on how plot_roc_curve handles multi-class
                    # For simplicity, let's assume plot_roc_curve expects 1D scores, so this might be an issue for multi-class.
                    # y_scores_for_roc = y_pred_proba[:, 0] # Example: scores for class 0
                    # A better approach for multi-class might be to not plot or ask user to select a class.
                    st.warning("Multi-class ROC plotting is not fully supported with single curve. Consider specific class probabilities if needed.")

                else:
                    st.warning("`predict_proba` output has an unexpected shape. Cannot prepare scores for ROC curve.")
            except Exception as e_proba:
                st.warning(f"Could not get probability scores from model: {e_proba}")
        # Fallback to decision_function if predict_proba is not available
        elif hasattr(model, "decision_function"):
            try:
                y_decision_scores = model.decision_function(X_test_data)
                if y_decision_scores.ndim == 1: # Binary or already OvR
                    y_scores_for_roc = y_decision_scores
                else: # Multi-class decision_function with multiple columns
                    st.info("Multi-column `decision_function` output. ROC curve typically requires one score per sample. This plot is best for binary tasks or if decision scores are 1D.")
            except Exception as e_decision:
                st.warning(f"Could not get decision scores from model: {e_decision}")
        else:
            st.warning(f"Model {model.__class__.__name__} does not have `predict_proba` or `decision_function`. ROC curve cannot be plotted.")

        if y_scores_for_roc is not None:
            # Ensure y_scores_for_roc is 1D numpy array
            if isinstance(y_scores_for_roc, pd.Series):
                y_scores_for_roc = y_scores_for_roc.values
            if isinstance(y_scores_for_roc, list):
                y_scores_for_roc = np.array(y_scores_for_roc)

            if y_scores_for_roc.ndim == 1:
                roc_fig = plot_roc_curve(y_test_data, y_scores_for_roc)
                if roc_fig:
                    st.pyplot(roc_fig)
                else:
                    st.warning("Could not generate ROC Curve. Ensure model provides valid 1D scores (for positive class in binary, or for a specific class in multi-class) and data is appropriate.")
            else:
                st.warning("ROC scores are not 1-dimensional. Cannot plot standard ROC curve.")

    elif task_type == "Regression":
        st.write(f"R² Score: {r2_score(y_test_data, y_pred_data):.4f}")
        st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_data, y_pred_data):.4f}")
        mse = mean_squared_error(y_test_data, y_pred_data)
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        rmse = np.sqrt(mse) # np.sqrt needs numpy import
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    else:
        st.warning("Unsupported model type for detailed evaluation.")


def show_models_tab(df): # df is passed but models primarily use X_train, y_train etc. from session_state
    st.header("🤖 Model Selection & Training")

    required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'target_col']
    if not all(k in st.session_state for k in required_keys):
        st.warning("⚠️ Please complete preprocessing steps first: Set target variable and split dataset.")
        return

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    # ========== Load Pre-trained Model ==========
    st.subheader("📂 Load a Pre-trained Model")
    uploaded_model_file = st.file_uploader("Upload a .pkl model file", type=["pkl"], key="model_uploader")

    if uploaded_model_file is not None:
        try:
            loaded_model = pickle.load(uploaded_model_file)
            st.session_state.trained_model = loaded_model
            st.success("✅ Model loaded successfully!")

            if st.button("🔮 Predict with Loaded Model"):
                try:
                    y_pred_loaded = loaded_model.predict(X_test)
                    st.session_state.y_pred = y_pred_loaded

                    st.write("### 🧾 Predictions (Loaded Model)")
                    st.dataframe(pd.DataFrame({"True": y_test, "Predicted": y_pred_loaded}))

                    loaded_model_task_type = None
                    if isinstance(loaded_model, ClassifierMixin):
                        loaded_model_task_type = "Classification"
                    elif isinstance(loaded_model, RegressorMixin):
                        loaded_model_task_type = "Regression"

                    if loaded_model_task_type:
                        display_model_evaluation(loaded_model, X_test, y_test, y_pred_loaded, loaded_model_task_type, model_name="Loaded Model")
                    else:
                        st.warning("⚠️ Could not determine if the loaded model is for Classification or Regression. Full evaluation metrics might not be available.")
                        st.write("Raw Predictions from loaded model:", y_pred_loaded)

                except Exception as e:
                    st.error(f"❌ Prediction failed with loaded model: {e}")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")

    st.divider()

    # ========== Train New Model ==========
    st.subheader("🧠 Train a New Model")
    task_type = st.radio("Select task type", ["Classification", "Regression"], horizontal=True, key="model_task_type")

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
        "XGBoost Regressor": XGBRegressor(eval_metric='rmse')
    }

    model_options = list(classification_models.keys()) if task_type == "Classification" else list(regression_models.keys())
    selected_model_name = st.selectbox("Select a model", model_options, key="model_select")

    st.markdown("#### 🚦 Training")
    if st.button("🚀 Train Model", key="train_new_model_button"):
        try:
            non_numeric_cols = [col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])]
            if non_numeric_cols:
                st.warning(f"⚠️ All features for model training must be numeric. Please encode or remove these columns from X_train: {', '.join(non_numeric_cols)}")
            else:
                current_model = classification_models[selected_model_name] if task_type == "Classification" else regression_models[selected_model_name]

                if selected_model_name == "XGBoost" and task_type == "Classification":
                    y_train_xgb = y_train.copy()
                    if y_train.nunique() > 2 and y_train.min() != 0 :
                         # XGBoost expects labels in [0, num_class-1] for multiclass
                         from sklearn.preprocessing import LabelEncoder
                         le = LabelEncoder()
                         y_train_xgb = le.fit_transform(y_train)
                    current_model.fit(X_train, y_train_xgb)
                else:
                    current_model.fit(X_train, y_train)

                y_pred_new = current_model.predict(X_test)
                st.session_state.trained_model = current_model
                st.session_state.y_pred = y_pred_new

                st.success(f"✅ {selected_model_name} trained successfully!")
                st.write("### 🧾 True vs Predicted (New Model)")
                st.dataframe(pd.DataFrame({"True": y_test, "Predicted": y_pred_new}))

                # Call the new evaluation display function
                display_model_evaluation(current_model, X_test, y_test, y_pred_new, task_type, model_name=selected_model_name)

                model_filename = f"{selected_model_name.replace(' ', '_')}_trained.pkl"
                os.makedirs("models", exist_ok=True)
                model_path = os.path.join("models", model_filename)
                with open(model_path, 'wb') as f_new_model:
                    pickle.dump(current_model, f_new_model)

                with open(model_path, 'rb') as f_new_model_rb:
                    buffer = io.BytesIO(f_new_model_rb.read())
                st.download_button(
                    label="📥 Download Trained Model",
                    data=buffer,
                    file_name=model_filename,
                    mime="application/octet-stream"
                )
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")

    # ========== Manual Input Prediction (if a model is trained/loaded) ==========
    if "trained_model" in st.session_state:
        st.subheader("🧪 Predict on Custom Input")

        if 'X_train' in st.session_state and not st.session_state.X_train.empty:
            feature_names = st.session_state.X_train.columns
            user_input_dict = {}

            for feature in feature_names:
                dtype = st.session_state.X_train[feature].dtype
                if dtype == 'object' or dtype.name == 'category':
                    unique_vals = list(st.session_state.X_train[feature].unique())
                    try:
                        unique_vals.sort()
                    except TypeError:
                        pass
                    user_input_dict[feature] = st.selectbox(f"{feature} (categorical)", unique_vals, key=f"manual_{feature}")
                else:
                    default_val = float(st.session_state.X_train[feature].mean())
                    user_input_dict[feature] = st.number_input(f"{feature} (numeric)", value=default_val, key=f"manual_{feature}")

            if st.button("🔍 Predict on Input", key="predict_manual_button"):
                input_df = pd.DataFrame([user_input_dict])
                input_df = input_df[feature_names]

                try:
                    prediction = st.session_state.trained_model.predict(input_df)
                    st.success(f"✅ Predicted Value: {prediction[0]}")

                    if hasattr(st.session_state.trained_model, "predict_proba"):
                        try:
                            prediction_proba = st.session_state.trained_model.predict_proba(input_df)
                            st.write("Confidence Probabilities:")
                            # For binary, show prob of positive class, for multi-class, show all
                            if prediction_proba.shape[1] == 2:
                                st.write(f"Prob (Positive Class): {prediction_proba[0][1]:.4f}")
                            else:
                                st.write(prediction_proba[0])
                        except Exception as e_proba:
                            st.warning(f"Could not get probabilities: {e_proba}")

                except Exception as e:
                    st.error(f"❌ Prediction failed on manual input: {e}")
        else:
            st.info("Train or load a model and ensure X_train is available in session state to use manual prediction.")

```
