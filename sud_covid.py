# analysis current conditions
# forcat case

from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import add_changepoints_to_plot
from fbprophet import Prophet
import seaborn as sns
import warnings
import folium
from folium import plugins
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
# plotly.io.renders.default = 'colab'


# mainpulate the defult plot size
plt.rcParams['figure.figsize'] = 10, 12
# disable wornings
warnings.filterwarnings('ignore')

# Read data into dataframe datasets

df_sud_covid = pd.read_excel(r'sud_cov.xlsx')
print(df_sud_covid.dtypes)
print(df_sud_covid)

# total number of confirmed cases
total_active = df_sud_covid['confirmed'].sum()
print('Total number of confirmed cases since (10/31/20) is:  {}'.format(total_active))

total_cases = df_sud_covid.groupby(
    'state')['confirmed'].sum().sort_values(ascending=False).to_frame()
print(total_cases)

# Geogrphical distribution
map = folium.Map(location=[12.8628, 30.2176],
                 zoom_start=6, tiles='Stamen Terrain')
tooltip = 'Click me!'

# generate the popup message that is shown on click.
for index, row in df_sud_covid.iterrows():

    # generate the popup message that is shown on click.
    popup_text = "{}<br> Confirmed: {}<br> State: {}"
    popup_text = popup_text.format(
        index,
        row["confirmed"],
        row["state"]
    )

    folium.CircleMarker(location=(row["lat"],
                                  row["lng"]),
                        radius=row['confirmed']*0.1,
                        color="RED",
                        popup="State",
                        fill=False).add_to(map)

map.save('index.html')

# Trend

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_sud_covid['date'], y=df_sud_covid['confirmed'], mode='markers',  marker_color=df_sud_covid['confirmed'], name='total_casces'))
fig.update_layout(
    title_text='Trend os coronaVirus cases in Sudan (cumultive casses)', plot_bgcolor='rgb(230,230,230)')
fig.show()


fig = px.bar(df_sud_covid, x="date", y="confirmed",
             barmode='group', height=400)
fig.update_layout(
    title_text='Trend os coronaVirus cases in Sudan (cumultive casses)', plot_bgcolor='rgb(230,230,230)')
fig.show()

fig = px.scatter(x=df_sud_covid['date'], y=df_sud_covid['confirmed'])
fig.show()

# forcasting  total number of cases with prophet time series data.

confirmed = df_sud_covid.groupby('date').sum()['confirmed'].reset_index()
# Prophet requires time series data to have a minimum of two columns:
# ds which is the time stamp and y which is the values.
confirmed.columns = ['ds', 'y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
print(confirmed.tail())

# forecast for week ahead

m = Prophet(interval_width=(0.95), growth='linear', daily_seasonality=False,
            weekly_seasonality=True, yearly_seasonality=False)  # decimal confendance
print(confirmed.tail())
m.fit(confirmed)
future = m.make_future_dataframe(periods=7, freq='d')
print(future.tail())

confirmed_forecast = m.predict(future)
print(confirmed_forecast[[
      'ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
# The output of cross_validation is a dataframe with the true values y and
# the out-of-sample forecast values yhat, at each simulated forecast date and for
# each cutoff date. In particular, a forecast is made for every observed point between
# cutoff and cutoff + horizon. This dataframe can then be used to compute
# error measures of yhat vs. y.
# cv_result = cross_validation(
#    m, initial='3 days', period='1 day', horizon='5 days')

# print(cv_result)
# df_p = performance_metrics(cv_result)
# print(df_p)

# fig = plot_cross_validation_metric(cv_result, metric='mape')
# plt.show()

# plot forecast
confirmed_forecast_plot = m.plot(confirmed_forecast)
a = add_changepoints_to_plot(
    confirmed_forecast_plot.gca(), m, confirmed_forecast)
plt.show()
confirmed_forecast_plot2 = m.plot_components(confirmed_forecast)
plt.show()

# forcasting Death with prophet time series data.
death = df_sud_covid.groupby('date').sum()['death'].reset_index()
death.columns = ['ds', 'y']
print(death.tail())

# forecast for week ahead
m = Prophet(interval_width=(0.95))  # decimal confendance
print(death.tail())
m.fit(death)
future = m.make_future_dataframe(periods=7, freq='d')
print(future.tail())

death_forecast = m.predict(future)
print(death_forecast[[
      'ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# plot forecast
death_forecast_plot = m.plot(death_forecast)
plt.show()
death_forecast_plot = m.plot_components(death_forecast)
plt.show()
