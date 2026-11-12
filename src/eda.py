
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_data_overview(data):

    overview = {
        "shape": data.shape,
        "columns": list(data.columns),
        "data_types": data.dtypes,
        "missing_values": data.isnull().sum(),
        "duplicates": data.duplicated().sum()
    }

    return overview


def plot_distributions(data):

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) == 0:
        return None

    fig, axes = plt.subplots(
        nrows=len(numeric_columns),
        ncols=1,
        figsize=(8, 4 * len(numeric_columns))
    )

    if len(numeric_columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, numeric_columns):

        sns.histplot(
            data=data,
            x=column,
            kde=True,
            ax=ax
        )

        ax.set_title(f"Distribution of {column}")

    fig.tight_layout()

    return fig


def plot_correlation_matrix(data):

    numeric_data = data.select_dtypes(
        include="number"
    )

    if numeric_data.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    correlation = numeric_data.corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Matrix")

    fig.tight_layout()

    return fig
