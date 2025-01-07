
""" 
This script contains standard user plotting settings to create 
exploratory and result oriented visualations throughout the project. 
"""

## Import Libraries 
import plotly.express as px
import plotly.io as pio

# ----------------------------------------------------------
def plotly_user_standard_settings(pio, px):
    """
    This function enforces the standard settings for plotly plots 
    created throughout at various stages in the project
    """
    pio.templates.default = 'simple_white'
    px.defaults.width = 800
    px.defaults.height = 500

    excel_style_template = dict(
        layout=dict(
            font=dict(family="Trebuchet MS, sans-serif", size=11, color="black"),
            title=dict(font=dict(size=18, color="black")),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
                showline=True,
                linecolor="black",
                linewidth=1,
                ticks="outside",
                tickcolor="black",
                mirror=True
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
                showline=True,
                linecolor="black",
                linewidth=1,
                ticks="outside",
                tickcolor="black",
                mirror=True
            ),
            margin=dict(
                l=60,
                r=30,
                t=30,
                b=60
            ),
            legend=dict(
                font=dict(size=12, color="black")
            )
        )
    )

    # Register the template and set it as default
    pio.templates["excel_style"] = excel_style_template
    pio.templates.default = "excel_style"
    
    return pio









