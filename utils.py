import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_results(df, column_to_plot, index, color):
    fig = px.line(data_frame=df, x=index, y=column_to_plot, color=color)
    fig.update_layout(title=column_to_plot)
    fig.show()

