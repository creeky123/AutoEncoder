import plotly.express as px
import pandas as pd


def plot_results(df, column_to_plot, index, color, fig_size):

    df_ = df[[column_to_plot, index, color]]
    df_ = pd.melt(df, )
