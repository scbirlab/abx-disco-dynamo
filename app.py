
from typing import Callable, Iterable, List, Optional, Tuple
from functools import partial
import os

import altair as alt
from carabiner import print_err, colorblind_palette
from carabiner.mpl import grid
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import approx_fprime, minimize
from scipy.stats import  expon, gamma, poisson, probplot

DATA_PATH = f'{os.getcwd()}/stennett2022-table1.xlsx'
CLASS_COL = 'class'
YEAR_COL = 'year'
FIG_PANEL_SIZE = 3.5

NUMBER_DISCOVERED = "Discovered"
NUMBER_WO_R = "Without resistance"
NUMBER_W_R = "With resistance"

def load_data(
    path: str = DATA_PATH, 
    class_col: str = CLASS_COL, 
    year_col_prefix: str = YEAR_COL
) -> pd.DataFrame:
    df = pd.read_excel(path)
    summaries = [df.groupby(col)[[class_col]]
                        .agg('count')
                        .rename(columns={class_col: f'{col}_count'}) 
                for col in df if col.startswith(year_col_prefix)]

    df = (
        pd.concat(summaries, axis=1)
            .sort_index()
            .fillna(0.)
            .cumsum()
            .reset_index()
            .rename(columns={'index': year_col_prefix})
            .assign(**{
                NUMBER_WO_R: lambda x: x[f'{year_col_prefix}_discovered_count'] - x[f'{year_col_prefix}_resistance_count'],
                "time": lambda x: x[year_col_prefix] - x[year_col_prefix].min(),
            })
            .rename(columns={
                YEAR_COL: "Year",
                f'{year_col_prefix}_discovered_count': NUMBER_DISCOVERED,
                f'{year_col_prefix}_resistance_count': NUMBER_W_R,
            })
        )
    return df


def dg_dt(params: ArrayLike) -> Callable[[ArrayLike, ArrayLike], List[np.ndarray]]:
    k, n, tlag, half_life = params
    slope = 1.
    
    def f(y: ArrayLike, t: ArrayLike):
        m, D, R, g = y
        dm = (n / 2.) * (1. + np.tanh(((t - tlag) ** slope))) 
        dD = ((k - D + 1.) / k) * dm 
        dR = (D - R) / (half_life / np.log(2.))  # Scale to half-life
        dg = dD - dR
        return [dm, dD, dR, dg]
    
    return f
    
    
def dynamic_model(
    t: float, 
    params: ArrayLike, 
    y0: Optional[ArrayLike] = None
) -> np.ndarray:
    if y0 is None:
        y0 = np.ones((4,))
        
    o = odeint(
        dg_dt(params), 
        y0=y0, 
        t=[0., t],
    )
    return o[-1,:]  # -1 to take the endpoint only


def nloglik_poisson(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    return -np.sum(poisson.logpmf(y_true, mu=y_pred))


def dobj_fun(
    model: Callable[[float, ArrayLike, Optional[ArrayLike]], np.ndarray], 
    df: pd.DataFrame
) -> Callable[[ArrayLike], float]:
    
    def _dobj_fun(params: ArrayLike) -> float:
        return nloglik_poisson(
            [model(t, params)[1:-1] for t in df['time']], 
            y_true=df[[NUMBER_DISCOVERED, NUMBER_W_R]].values,
        )
    
    return _dobj_fun

def fit_to_data(
    df: pd.DataFrame, 
    init_params: ArrayLike
) -> Tuple[float]:
    init_params = np.asarray([float(p) for p in init_params])
    print_err(f"Fitting with init params = {init_params}")
    function_to_minimize = dobj_fun(dynamic_model, df)
    jacobian = partial(approx_fprime, f=function_to_minimize)
    print_err(f"Initial objective: {function_to_minimize(init_params)}, initial gradients:\n{jacobian(init_params)}")
    do = minimize(
        function_to_minimize, 
        x0=init_params,
        jac=jacobian,
    )
    print(do)
    return tuple(do.x.flatten())


def plot_prediction(
    predicted_values: ArrayLike, 
    columns: ArrayLike, 
    index: ArrayLike, 
    year_col: str = "Year",
    y_col: str = "Number of classes",
    **kwargs
) -> alt.Chart:
    df_pred = (
        pd.DataFrame(
            predicted_values, 
            columns=columns, 
            index=index,
        )
        .reset_index(
            names=year_col
        )
        .melt(
            id_vars=year_col, 
            value_vars=columns, 
            var_name='count_type',
            value_name=y_col,
        )
    )
    return alt.Chart(df_pred).mark_line().encode(**kwargs)


def plot_data_altair(
    df: pd.DataFrame, 
    year_col: str = "Year",
    y_col: str = "Number of classes",
    params: Optional[ArrayLike] = None,
    add_config: bool = True,
):  

    cols_to_plot = [NUMBER_DISCOVERED, NUMBER_W_R, NUMBER_WO_R]

    df_m = df.melt(
        id_vars=year_col, 
        value_vars=cols_to_plot, 
        var_name='count_type', 
        value_name=y_col,
    )
    print(df_m)
    encoding = dict(
        x=alt.X(year_col).scale(zero=False),
        y=alt.Y(y_col),
        color=alt.Color('count_type').title("").scale(range=colorblind_palette()),
    )
    figure = alt.Chart(df_m).mark_circle().encode(**encoding, tooltip=[year_col, y_col])
    if params is not None:
        params = np.asarray([float(p) for p in params])
        print_err(f"Plotting with params = {params}")
        predicted_values = np.asarray([dynamic_model(t, params)[1:] for t in df['time']])
        figure += plot_prediction(
            predicted_values=predicted_values, 
            columns=cols_to_plot, 
            index=df[year_col], 
            **encoding
        )

    if add_config:
        return figure.configure_axis(
            grid=False
        ).interactive()
    else:
        return figure


def plot_data_forecast_altair(
    df: pd.DataFrame, 
    year_col: str = "Year",
    y_col: str = "Number of classes",
    params: Optional[ArrayLike] = None
):  
    params = np.asarray([float(p) for p in params])
    old_params, fold_changes, forecast_time = params[:4], params[4:-1], params[-1]
    new_params = [old_params[0] * fold_changes[0], old_params[1] * fold_changes[1], 0., old_params[-1] * fold_changes[-1]]
    print_err(f"Plotting with params = {params}, forcasting for {forecast_time} years")
    cols_to_plot = [NUMBER_DISCOVERED, NUMBER_W_R, NUMBER_WO_R]

    figure = plot_data_altair(df, year_col, y_col, old_params, add_config=False)
    encoding = dict(
        x=alt.X(year_col).scale(zero=False),
        y=alt.Y(y_col),
        color=alt.Color('count_type').title(""),
    )
    figure += alt.Chart(pd.DataFrame(dict(
        Year=[df["Year"].max()], 
        color=["lightgrey"]
    ))).mark_rule().encode(
        x=alt.X('Year'),
        color=alt.Color('Year:N', scale=None)
    )
    future_times = np.linspace(0., forecast_time, num=20)
    y0 = dynamic_model(df['time'].values[-1], old_params)
    new_values = np.array([dynamic_model(t, new_params, y0=y0)[1:] for t in future_times])
    figure += plot_prediction(
            predicted_values=new_values, 
            columns=cols_to_plot, 
            index=df["Year"].max() + future_times, 
            **encoding
        )
    return figure.configure_axis(
        grid=False
    ).interactive()


pool_size_title = "**Effective pool size** | _effective number of antibiotic classes being sampled by drug discovery, from the [Coupon Collector problem](https://en.wikipedia.org/wiki/Coupon_collector%27s_problem)_"
discovery_rate_title = " **Maximal rate of new discoveries** | _effective number of samples from the pool of antibiotic classes per year_"
half_life_title = "**Resistance-free half-life** | _exponential decay from start of clinical use_"


def parameter_msg(*params) -> str:

    params = np.asarray([float(p) for p in params])
    pool_size, discovery_rate, discovery_lag, half_life = params
    return f"""
    {pool_size_title} | **{pool_size:.1f} classes**

    {discovery_rate_title} | **{discovery_rate:.1f} / year**

    **Discovery lag** | _time to maximal discovery rate_ | **{discovery_lag:.1f} years**

    {half_life_title} | **{half_life:.1f} years**

    """


def forecast_msg(*params) -> str:

    params = np.asarray([float(p) for p in params])
    pool_size, discovery_rate, discovery_lag, half_life = params[:4]
    x_pool_size, x_discovery_rate, x_half_life, _ = params[4:]

    return f"""
    {pool_size_title} | {pool_size:.1f} classes ⨉ {x_pool_size} = **{pool_size * x_pool_size:.1f} classes**

    {discovery_rate_title} | {discovery_rate:.1f} / year ⨉ {x_discovery_rate} = **{x_discovery_rate * discovery_rate:.1f} / year**

    {half_life_title} | {half_life:.1f} years ⨉ {x_half_life} = **{x_half_life * half_life:.1f} years**

    """


with gr.Blocks() as demo:

    data = load_data()
      
    gr.Markdown(
        """
        # Dynamics of antibiotic discovery and resistance
        [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scbirlab/2024-Parkhill-BiochemJ/main?labpath=modelling-abx-discovery.ipynb)

        Interface to demonstrate Figure 1 from [Parkhill SL, Johnson EO, Integrating bacterial molecular genetics with chemical biology for renewed antibacterial 
        drug discovery, _Biochemical Journal_ (2024) 481 (13): 839–864](https://doi.org/10.1042/BCJ20220062).

        Access the notebook explaining the models [here](https://github.com/scbirlab/2024-Parkhill-BiochemJ/blob/main/modelling-abx-discovery.ipynb). Run 
        the notebook interactively in Binder [here](https://mybinder.org/v2/gh/scbirlab/2024-Parkhill-BiochemJ/main?labpath=modelling-abx-discovery.ipynb).

        """
    )

    with gr.Tab("Fitting parameters"):
        gr.Markdown(
            """
            # Finding the dynamic parameters

            **Adjust the sliders** to alter the parameters underlying the rate of antibiotic discovery and resistance.

            **Click update plot** to see what the dynamics would look like with your parameters.
            
            **Click "Fit parameters!"** to automatically find the best fitting parameters.

            """
        )
        with gr.Row():
            param_sliders = [
                gr.Slider(label="Pool size", info="Effective number of antibiotic classes being sampled by drug discovery", 
                          value=20., minimum=0., maximum=100., scale=10),
                gr.Slider(label="Maximal discovery rate", info="Effective number of samples from the pool per year", 
                          value=1., minimum=0., maximum=10., scale=10),
                gr.Slider(label="Discovery lag", info="Time to maximum discovery rate", 
                          value=25., minimum=0., maximum=100., scale=10),
                gr.Slider(label="Resistance-free half-life", info="Relative to start of clinical use", 
                          value=30., minimum=0., maximum=50., scale=10),
            ]
            refresh_button = gr.Button("Update plot", scale=6)
            fit_button = gr.Button("Fit parameters!", scale=6)
        
        # with gr.Row():
        fit_message = gr.Markdown(parameter_msg, inputs=param_sliders)
        plot = gr.Plot(
            label="Model fit", 
            scale=4,
        )
        gr.on(
            triggers=[s.release for s in param_sliders] + [refresh_button.click], 
            fn=lambda *x: plot_data_altair(df=data, params=x),
            inputs=param_sliders,
            outputs=plot,
            trigger_mode="once",
        )

    with gr.Tab("Forecasting the future!"):
        gr.Markdown(
            """
            # Forecasting future discovery and resistance!

            **Adjust the sliders** to see how changes in these parameters would change the future.

            **Click update plot** to see what the dynamics would look like with your parameters.

            **Click "Fit parameters!"** on the previous tab to set the parameters to fit historical data, 
            then come back to this tab to check the forecast.

            """
        )

        with gr.Row():
            forecast_sliders = [
                gr.Slider(label="⨉ pool size", info="Increase in accessible antibiotic classes", 
                          value=1., minimum=0., maximum=10., step=.2, scale=10),
                gr.Slider(label="⨉ discovery rate", info="Increase in rate of discovery", 
                          value=1., minimum=0., maximum=10., step=.2, scale=10),
                gr.Slider(label="⨉ half-life", info="Increase in resistance-free half-life", 
                          value=1., minimum=0., maximum=10., step=.2, scale=10),
                gr.Slider(label="🔮", info="In years", 
                          value=100., minimum=0., maximum=200., step=.5, scale=10),
            ]
            refresh_button_forecast = gr.Button("Update plot", scale=6)
        
        param_and_forecast_sliders = param_sliders + forecast_sliders
        fit_message = gr.Markdown(forecast_msg, inputs=param_and_forecast_sliders)    
        forecast = gr.Plot(
            label="Forecast", 
            scale=4,
        )
        gr.on(
            triggers=[s.release for s in param_and_forecast_sliders] + [refresh_button_forecast.click, refresh_button.click], 
            fn=lambda *x: plot_data_forecast_altair(df=data, params=x),
            inputs=param_and_forecast_sliders,
            outputs=forecast,
            trigger_mode="once",
        )

        (fit_button
         .click(lambda *x: fit_to_data(data, init_params=x), inputs=param_sliders, outputs=param_sliders)
         .then(lambda *x: plot_data_altair(df=data, params=x),inputs=param_sliders, outputs=plot)
         .then(lambda *x: plot_data_forecast_altair(df=data, params=x),inputs=param_and_forecast_sliders, outputs=forecast))

demo.launch(share=True)