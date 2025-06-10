import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, roc_curve, auc # Added
import numpy as np # Added

# دالة لتحميل البيانات
def load_data(file_path):
    return pd.read_csv(file_path)


# دالة لعمل Histogram
def plot_histogram(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Histogram of {column}")
    return fig

# دالة لعمل Density Plot
def plot_density(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df[column], ax=ax)
    ax.set_title(f"Density Plot of {column}")
    return fig

# دالة لعمل Boxplot
def plot_boxplot(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Boxplot of {column}")
    return fig

# دالة لعمل Violin Plot
def plot_violin(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x=df[column], ax=ax)
    ax.set_title(f"Violin Plot of {column}")
    return fig

# دالة لعمل Bar Plot
def plot_bar(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=df[column].value_counts().index, y=df[column].value_counts().values, ax=ax)
    ax.set_title(f"Bar Plot of {column}")
    return fig

# دالة لعمل Pie Chart
def plot_pie(df, column):
    fig, ax = plt.subplots(figsize=(8, 8))
    df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, cmap="Set3")
    ax.set_title(f"Pie Chart of {column}")
    return fig

# دالة لعمل Scatter Plot
def plot_scatter(df, x_column, y_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x_column], df[y_column])
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"Scatter Plot: {x_column} vs {y_column}")
    return fig

# دالة لعمل Correlation Heatmap
def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

# دالة لعمل Pairplot
def plot_pairplot(df):
    fig = sns.pairplot(df)
    fig.fig.suptitle("Pairplot", y=1.02)
    return fig

# دالة لعمل Line Plot (Time Series)
def plot_line(df, x_column, y_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=df[x_column], y=df[y_column], ax=ax)
    ax.set_title(f"Line Plot: {x_column} vs {y_column}")
    return fig

# دالة لعمل Missing Data Heatmap
def plot_missing_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title("Missing Data Heatmap")
    return fig

# دالة لعمل Missing Data Barplot
def plot_missing_barplot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.isnull().sum().plot.bar(ax=ax)
    ax.set_title("Missing Data Barplot")
    return fig

# دالة لعمل Word Cloud
def plot_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud")
    return fig

# New function for Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    # Ensure class_names are unique and sorted if not from model.classes_ directly
    # The calling function should handle providing appropriate class_names
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)*0.75), max(6, len(class_names)*0.6))) # Adjust size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

# New function for ROC Curve
def plot_roc_curve(y_true, y_pred_scores): # y_pred_scores are 1D array: probabilities of positive class or decision scores
    if y_pred_scores is None:
        # print("Debug: y_pred_scores is None in plot_roc_curve.") # For worker's debugging
        return None # Silently return None, calling function handles messages
    if not isinstance(y_pred_scores, np.ndarray) or y_pred_scores.ndim != 1:
        # print(f"Debug: y_pred_scores type {type(y_pred_scores)}, ndim {y_pred_scores.ndim if hasattr(y_pred_scores, 'ndim') else 'N/A'}") # For worker's debugging
        # This case should be handled by the caller, but as a safeguard:
        return None # Silently return None

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_scores)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        plt.tight_layout()
    except Exception as e:
        # print(f"Error in plot_roc_curve: {e}") # For worker's debugging
        # Optionally, could draw error on fig, but returning None is cleaner for caller
        plt.close(fig) # Close the figure if an error occurred during plotting
        return None
    return fig
