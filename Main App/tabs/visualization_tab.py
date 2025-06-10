import streamlit as st
import pandas as pd
from visualization_Functions import visualize as v
from io import BytesIO # Added based on usage in YourAnalyst.py
import numpy as np # Added based on usage

def show_visualization_tab(df):
    st.header("📊 Visualization")
    # Select the type of plot
    plot_type = st.selectbox("Choose the type of plot", [
        "Scatter Plot", "Line Plot (Time Series)",
        "Correlation Heatmap", "Pairplot", "Histogram", "Density Plot",
        "Boxplot", "Violin Plot", "Bar Plot", "Pie Chart", "Missing Heatmap",
        "Missing Barplot", "Word Cloud"
    ])

    df_display = st.session_state.get("df", df) # Use df passed or from session state

    # Function to show appropriate columns for each plot
    def get_column_options(plot_type, current_df): # Renamed df to current_df
        if plot_type in ["Scatter Plot", "Line Plot (Time Series)"]:
            return current_df.columns
        # For heatmap and pairplot, it's better to use only numeric, but visualize.py might handle it.
        # For simplicity, allow all, but note that visualize.py might raise errors for non-numeric.
        elif plot_type in ["Correlation Heatmap", "Pairplot"]:
            return current_df.columns # Or current_df.select_dtypes(include=np.number).columns
        else: # Histogram, Density, Box, Violin, Bar, Pie
            return current_df.select_dtypes(include=['float64', 'int64', 'object', 'category']).columns


    columns_for_plot = get_column_options(plot_type, df_display)

    # Handle the cases where multiple columns need to be selected (for Scatter, Line)
    if plot_type in ["Scatter Plot", "Line Plot (Time Series)"]:
        x_column = st.selectbox(f"Choose the X-axis column for {plot_type}", columns_for_plot, key="vis_x_col")
        y_column = st.selectbox(f"Choose the Y-axis column for {plot_type}", columns_for_plot, key="vis_y_col")

    # Handle the plots that need just one column
    elif plot_type in ["Histogram", "Density Plot", "Boxplot", "Violin Plot", "Bar Plot", "Pie Chart"]:
        selected_column = st.selectbox(f"Choose a column for the {plot_type}", columns_for_plot, key="vis_sel_col")

    # For plots that use the whole dataframe or only numeric parts handled by the function
    elif plot_type in ["Correlation Heatmap", "Pairplot", "Missing Heatmap", "Missing Barplot", "Word Cloud"]:
        pass # No specific column selection needed here at this stage

    # Function to save the plot
    def save_plot(fig, p_type): # Renamed plot_type to p_type
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    if st.button(f"Generate {plot_type}"):
        # st.session_state.active_tab = 1 # This might not be needed or should be handled in YourAnalyst.py
        try:
            fig = None # Initialize fig
            if plot_type == "Histogram":
                if selected_column:
                    fig = v.plot_histogram(df_display, selected_column)
            elif plot_type == "Density Plot":
                if selected_column:
                    fig = v.plot_density(df_display, selected_column)
            elif plot_type == "Boxplot":
                if selected_column:
                    fig = v.plot_boxplot(df_display, selected_column)
            elif plot_type == "Violin Plot":
                if selected_column:
                    fig = v.plot_violin(df_display, selected_column)
            elif plot_type == "Bar Plot":
                if selected_column:
                    fig = v.plot_bar(df_display, selected_column)
            elif plot_type == "Pie Chart":
                if selected_column:
                    fig = v.plot_pie(df_display, selected_column)
            elif plot_type == "Scatter Plot":
                if x_column and y_column:
                    fig = v.plot_scatter(df_display, x_column, y_column)
            elif plot_type == "Correlation Heatmap":
                # Identify non-numeric columns
                numeric_df_display = df_display.select_dtypes(include=np.number)
                if numeric_df_display.shape[1] < df_display.shape[1]:
                    non_numeric_cols = df_display.select_dtypes(exclude=np.number).columns.tolist()
                    st.warning(f"⚠️ Correlation Heatmap typically uses numeric features. Non-numeric columns found and excluded: {', '.join(non_numeric_cols)}. If these should be included, please encode them first.")
                if not numeric_df_display.empty and numeric_df_display.shape[1] > 1 :
                    fig = v.plot_correlation_heatmap(numeric_df_display)
                else:
                    st.error("⚠️ Not enough numeric columns to generate a correlation heatmap.")
            elif plot_type == "Pairplot":
                # Pairplot can be slow on large datasets, consider a warning or sampling if df is large
                numeric_df_display = df_display.select_dtypes(include=np.number)
                if numeric_df_display.shape[1] < df_display.shape[1]:
                     non_numeric_cols = df_display.select_dtypes(exclude=np.number).columns.tolist()
                     st.warning(f"⚠️ Pairplot typically uses numeric features. Non-numeric columns found and excluded: {', '.join(non_numeric_cols)}. If these should be included, please encode them first.")
                if not numeric_df_display.empty and numeric_df_display.shape[1] > 1:
                    st.info("Generating Pairplot. This might take a moment for larger datasets...")
                    fig = v.plot_pairplot(numeric_df_display) # Use only numeric part or all df if visualize handles it
                else:
                    st.error("⚠️ Not enough numeric columns to generate a pairplot.")

            elif plot_type == "Line Plot (Time Series)":
                if x_column and y_column:
                    fig = v.plot_line(df_display, x_column, y_column)
            elif plot_type == "Missing Heatmap":
                fig = v.plot_missing_heatmap(df_display)
            elif plot_type == "Missing Barplot":
                fig = v.plot_missing_barplot(df_display)
            elif plot_type == "Word Cloud":
                # Ensure there's text data
                text_df = df_display.select_dtypes(include=['object'])
                if not text_df.empty:
                    text_data = ' '.join(text_df.fillna('').apply(lambda x: ' '.join(x), axis=1))
                    if text_data.strip():
                        fig = v.plot_wordcloud(text_data)
                    else:
                        st.warning("⚠️ No text data found to generate a word cloud.")
                else:
                    st.warning("⚠️ No object type columns found for word cloud.")

            if fig:
                st.pyplot(fig, use_container_width=True)
                plot_buffer = save_plot(fig, plot_type)
                st.download_button(f"Download {plot_type}", plot_buffer, file_name=f"{plot_type.lower().replace(' ', '_')}.png", mime="image/png")
            # else:
            #     st.info(f"Please select appropriate options to generate the {plot_type}.")

        except Exception as e:
            st.error(f"An error occurred while generating {plot_type}: {e}")
            # import traceback
            # st.error(traceback.format_exc()) # For more detailed error during development
