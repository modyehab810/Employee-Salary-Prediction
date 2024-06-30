# Importing ToolKits
import pandas as pd
import numpy as np
import plotly.express as px


import streamlit as st
import warnings

# pd.set_option('future.no_silent_downcasting', True)
# pd.options.mode.copy_on_write = "warn"


def creat_matrix_score_cards(card_image="", card_title="Card Title", card_value=None, percent=False):
    st.image(card_image,
             caption="", width=70)

    st.subheader(
        card_title)

    if percent:
        st.subheader(
            f"{card_value}%")

    else:
        st.subheader(
            f"{card_value}")


def create_comparison_df(y_actual, y_pred):
    predected_df = pd.DataFrame()
    predected_df["Actual Spent Values"] = y_actual
    predected_df.reset_index(
        drop=True, inplace=True)
    predected_df["Predicted Spent Value"] = y_pred

    return predected_df


def create_residules_scatter(predected_df):
    fig = px.scatter(
        predected_df,
        x=predected_df.iloc[:, 0],
        y=predected_df.iloc[:, 1],
        color=predected_df.iloc[:, 1] - predected_df.iloc[:, 0],
        opacity=0.8,
        title="Predicted Vs. Actual",
        template="plotly_dark",
        trendline="ols",
        height=650,
        labels={"x": "Actual Value", "y": "Predicted Value"}
    )

    fig.update_layout(
        title={
            "font": {
                "size": 28,
                "family": "tahoma"
            }
        }
    )
    return fig
