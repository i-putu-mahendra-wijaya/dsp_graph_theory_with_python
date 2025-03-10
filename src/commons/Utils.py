from typing import List, Union, Dict, Optional
import datetime

from google.cloud.bigquery.enums import SqlTypeNames
from whittaker_eilers import WhittakerSmoother
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.dao.GoogleBigquery import GoogleBigQuery


def smoothen_data(
        smoother: WhittakerSmoother,
        data_to_smooth: pd.Series
) -> pd.Series:
    """
    :param smoother: An instance of the WhittakerSmoother class used to smooth the data.
    :param data_to_smooth: A pd.Series containing the data to be smoothed.
    :return: A pd.Series containing the smoothed data.
    """
    list_data_to_smooth: List[Union[float, int]] = data_to_smooth.to_list()
    index: pd.Index = data_to_smooth.index

    smoothed_data = smoother.smooth(list_data_to_smooth)

    pd_smoothed_data: pd.Series = pd.Series(
        index=index,
        data=smoothed_data
    )

    return pd_smoothed_data


def write_to_bigquery(
        data_df: pd.DataFrame,
        mygbq: GoogleBigQuery,
        schema_name: str,
        table_name: str,
) -> None:
    """

    Writes a pandas DataFrame to Google BigQuery.

    :param data_df: The pandas DataFrame to be written to BigQuery.
    :type data_df: pd.DataFrame
    :param mygbq: An instance of GoogleBigQuery representing the connection to BigQuery.
    :type mygbq: GoogleBigQuery
    :param schema_name: The name of the BigQuery schema.
    :type schema_name: str
    :param table_name: The name of the BigQuery table.
    :type table_name: str
    :return: None
    :rtype: None

    This method writes the given pandas DataFrame to a specified BigQuery table in the specified schema. It appends the data to the table if it already exists, otherwise creates a new table.

    Example usage:
    mygbq = GoogleBigQuery()
    data_df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})
    write_to_bigquery(data_df, mygbq, 'my_schema', 'my_table')

    """

    # record current time stamp
    timestamp_now: datetime = datetime.datetime.now().date()

    data_df["prc_dt"] = timestamp_now
    data_df["snapshot_dt"] = timestamp_now

    data_df = data_df.astype(
        {
            "prc_dt": "datetime64[ns]",
            "snapshot_dt": "datetime64[ns]",
        }
    )

    column_names: List[str] = data_df.columns.tolist()
    list_datatypes: List[str] = [str(each_emt) for each_emt in data_df.dtypes.tolist()]

    conversion_dict: Dict = {
        "object": SqlTypeNames.STRING,
        "geography": SqlTypeNames.GEOGRAPHY,
        "int64": SqlTypeNames.INT64,
        "float64": SqlTypeNames.FLOAT,
        "bool": SqlTypeNames.BOOL,
        "datetime64[ns]": SqlTypeNames.DATETIME,
        "timedelta[ns]": SqlTypeNames.INTERVAL,
        "category": SqlTypeNames.STRING
    }

    list_datatype_gbq: List[SqlTypeNames] = [
        conversion_dict[each_type] for each_type in list_datatypes
    ]

    mygbq.gbq_write(
        dataframe=data_df,
        bq_cols=column_names,
        bq_types=list_datatype_gbq,
        bq_dst_table=f"{schema_name}.{table_name}",
        bq_write_disposition="WRITE_APPEND",
        bq_partition_key="snapshot_dt",
        bq_partition_type="MONTH"
    )


# define plot_correlation_matrx_function
def plot_correlation_matrix(
        df: pd.DataFrame,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        show_colorbar: bool = True,
        make_triangular: bool = False,
        *args
) -> pd.DataFrame:
    """
    Plot a correlation matrix heatmap for a given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing numerical data.
        fig (Optional[plt.Figure], optional): Figure to plot the heatmap on. Defaults to None.
        ax (Optional[plt.Axes], optional): Axes to plot the heatmap on. Defaults to None.
        show_colorbar (bool, optional): Flag to show colorbar. Defaults to True.

    Returns:
        pd.DataFrame: The correlation matrix.
    """

    def check_fig_and_ax_presence(
            fig: plt.Figure
            , ax: plt.Axes
    ) -> bool:
        """
        check if both fig and ax exists, or fig and ax both None
        """
        if (fig is None and ax is None) or (fig is not None and ax is not None):
            return True
        else:
            return False

    is_valid_fig_ax: bool = check_fig_and_ax_presence(fig=fig, ax=ax)

    if not is_valid_fig_ax:
        raise Exception(
            f"""
        Figure and Axes must be both set, or both None
        """
        )

    # remove all string columns before we calculate the correlation matrix
    df: pd.DataFrame = df.select_dtypes(
        exclude=["object", "datetime64[ns]"]
    )

    # calculate the correlation matrix
    corr_matrix: pd.DataFrame = df.corr()

    if make_triangular:
        # clear up the upper triangle portion of the corr_matrix
        mask: np.ndarray = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        corr_matrix[mask] = np.nan

    # plot the matrix
    cax = ax.matshow(corr_matrix, cmap='coolwarm') if ax else plt.matshow(corr_matrix, cmap="coolwarm")

    ticks: List[int] = range(len(corr_matrix.columns))

    if ax is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels(corr_matrix.columns, rotation=90)

        ax.set_yticks(ticks)
        ax.set_yticklabels(corr_matrix.columns)
    else:
        plt.xticks(ticks, corr_matrix.columns, rotation=90)
        plt.yticks(ticks, corr_matrix.columns)

    if show_colorbar:
        plt.colorbar(cax) if ax is None else fig.colorbar(cax, ax=ax)

    if ax is not None:
        ax.grid(False)
    else:
        plt.grid(False)
        plt.show()

    return corr_matrix


def log_transform_pandas(
        df: pd.DataFrame,
        columns_to_transform: Optional[List[str]] = None
) -> pd.DataFrame:
    """Applies a logarithmic transformation to specified numerical columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        columns_to_transform (List[str], optional): List of column names to apply the log transformation.
            If None, applies the transformation to all numerical columns.

    Returns:
        pd.DataFrame: A DataFrame in which the specified columns have been log transformed.

    Raises:
        ValueError: If any of the columns to transform are non-numerical.

    Note:
        The function handles negative values and zero values in the DataFrame by shifting the values before
        applying the log transformation. Also renames the transformed columns by prepending 'log_' to the column name.

    """

    # First, we separate categorical, string, and datetime columns to another dataframe
    cat_time_df: pd.DataFrame = df.select_dtypes(include=["object", "datetime64"]).copy()
    cat_time_columns: List[str] = cat_time_df.columns.to_list()

    # Second, we separate all numerical column into temporary num_df
    num_df: pd.DataFrame = df.select_dtypes(include="number").copy()
    num_columns: List[str] = num_df.columns.to_list()

    # Third, we assign which columns to transform. If None, we will assume that all columns will be transformed
    if columns_to_transform is not None:
        col2tr: List[str] = columns_to_transform
    else:
        col2tr: List[str] = num_df.columns.to_list()

    # Fourth, check if there are any of the columns to transform are non-numerical
    if set(col2tr).intersection(cat_time_columns):
        raise ValueError("One or more columns to transform are non-numerical.")

    # Fifth, create a stash_df to temporary park numerical columns that does not want to be transformed
    non_transformed_cols: List[str] = [each_col for each_col in num_columns if each_col not in col2tr]
    stash_df: pd.DataFrame = num_df[non_transformed_cols]
    to_transform_df: pd.DataFrame = num_df[col2tr]

    # Now, we are ready to log-transform the pandas at specified columns
    # Sixth, get the minimum values at each column
    min_values: pd.Series = to_transform_df.min()

    # Seventh, create a mask for negative values in each column
    negative_value_mask: bool = to_transform_df < 0

    # Now, we use the mask to find rows with negative values in each column, and add corresponding minimum for that column to these rows
    # Eighth, add minimum columns to negative value rows in each column
    for each_col in col2tr:
        to_transform_df.loc[negative_value_mask[each_col], each_col] += abs(min_values[each_col])

    # Ninth, log transform all columns in `to_transform_df`
    to_transform_df = to_transform_df.apply(np.log1p)  # --> log1p() can handle `0` better than log
    after_transform_columns: List[str] = [f"log_{each_col}" for each_col in col2tr]
    to_transform_df.columns = after_transform_columns

    # Tenth, merge back the stash_df (if any) to to_transform_df
    if len(stash_df) > 0:
        first_merge_df: pd.DataFrame = pd.merge(
            left=stash_df,
            right=to_transform_df,
            left_index=True,
            right_index=True,
        )
    else:
        first_merge_df: pd.DataFrame = to_transform_df.copy()

    # Elevnth, merge back `cat_time_df`
    second_merge_df: pd.DataFrame = pd.merge(
        left=cat_time_df,
        right=first_merge_df,
        left_index=True,
        right_index=True,
    )

    return second_merge_df


def find_next_perfect_square(
        n: int
) -> int:
    """
    Parameters
    ----------
    n : int
        The input number for which to find the next perfect square.

    Returns
    -------
    int
        The next perfect square after `n`.
    """

    root = math.sqrt(n)

    if root.is_integer():
        # if root is already integer, then return `n` itself
        return n
    else:
        # if `n` is a float, find the next integer
        return int(math.ceil(root) ** 2)


def plot_many_scatter_plots(
        df: pd.DataFrame,
        column_to_plot: str
) -> None:
    """
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data for plotting.

    column_to_plot : str
        The name of the column from the DataFrame to be plotted against other columns.

    Returns
    -------
    None

    Notes
    -----
    This method plots scatter plots between the specified column (`column_to_plot`) and all other columns of the DataFrame (`df`).

    The scatter plots are arranged in a grid format with the number of rows and columns determined by the square root of the number of columns in the DataFrame (excluding the `column_to_plot`). The size of the grid is adjusted to accommodate all the plots.

    Each scatter plot represents the relationship between the `column_to_plot` and a specific column from the DataFrame. The x-axis of each plot represents the values of the `column_to_plot` and the y-axis represents the values of the specific column being plotted.

    The titles of each plot are generated using a format: "{column_to_plot} vs {each_column}". The titles help in identifying the variables being compared in each plot.

    This method uses the matplotlib library for plotting the scatter plots.
    """

    # Get all the column from the dataframe
    df_columns: List[str] = df.columns.to_list()

    # Remove the `column_to_plot` from this list
    df_columns.pop(
        df_columns.index(column_to_plot)
    )

    # Calculate the num_plot_grid to contain these scatter plots
    num_plot_grid: float = math.sqrt(
        find_next_perfect_square(n=len(df_columns))
    )

    # Define a figure and axs based on `num_plot_grid`
    fig, axs = plt.subplots(
        nrows=int(num_plot_grid),
        ncols=int(num_plot_grid),
        figsize=(10 * num_plot_grid, 10 * num_plot_grid)
    )

    # Flatten axs to 1D array
    axs = axs.ravel()

    # Create the titles of each plot
    titles: List[str] = [f"{column_to_plot} vs {each_col}" for each_col in df_columns]

    # Create the plot
    for each_idx, (each_col, each_title, each_ax) in enumerate(zip(df_columns, titles, axs)):
        each_ax.scatter(
            x=df[column_to_plot],
            y=df[each_col],
        )

        each_ax.set_title(each_title)
        each_ax.set_xlabel(column_to_plot)
        each_ax.set_ylabel(each_col)

    plt.tight_layout()
    plt.show()