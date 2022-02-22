# Import needed packages
import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from copy import deepcopy

# Load data and add it to cache
@st.cache
def load_dataframe(path):
    df = pd.read_csv(path)
    return df

@st.cache
def load_jsonfile(path):
    with open(path) as response:
        regions = json.load(response)
    return regions

dfdog_raw = load_dataframe(path="data/raw/20200306_hundehalter.csv")
dfdog = deepcopy(dfdog_raw)

regions = load_jsonfile("data/raw/stzh.adm_stadtkreise_a.json")

# Helper functions
st.markdown("""
<style>
.text-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

# Add title and header
st.title("The Quizzical Chihuahua")
st.header("Exploring Canine Preferences in Zürich")

# Add option to see raw data (Widgets: checkbox)
if st.checkbox("Show data"):
    st.subheader("Dogs registered in the city of Zürich")
    st.markdown('<p class="text-font">Data source: https://data.stadt-zuerich.ch/dataset/sid_stapo_hundebestand</p>',
                unsafe_allow_html=True)
    st.dataframe(data=dfdog)

st.subheader("Breed distribution per Kreis in Zürich")

# Setting up columns
left_column, middle_column, right_column = st.columns([3, 1, 1])

# Enable selection of dog breed for map (Widgets: selectbox)
breeds = sorted(pd.unique(dfdog['RASSE1']))
breed = left_column.selectbox("Choose a Breed", breeds)

# Enable selection of total count or percentage (Widgets: radio button)
plot_types = ["Absolute Number", "Percentage"]
plot_type = right_column.radio("Choose Plot Type", plot_types)

# Process dog breed data
breed_df = dfdog[dfdog.RASSE1==breed]
breeds_per_region = breed_df.groupby('STADTKREIS').RASSE1.count()
total_dogs_per_region = dfdog.groupby('STADTKREIS').RASSE1.count()
percent_per_region = breeds_per_region / total_dogs_per_region * 100

# Plot map of selected dog breed (Choropleth mapbox using Plotly GO)
def plot_dog_map(breed_df, z_values, fig_title, hover_temp):

    c_fig = go.Figure(go.Choroplethmapbox(geojson=regions,
                                          locations=breed_df.STADTKREIS.sort_values().unique(),
                                          z=z_values,
                                          featureidkey='properties.name',
                                          colorscale="Plasma",
                                          marker_opacity=0.5,
                                          marker_line_width=0,
                                          hovertemplate=hover_temp))
    c_fig.update_layout(mapbox_style="stamen-toner",
                        mapbox_zoom=10,
                        mapbox_center={"lat": 47.36667, "lon": 8.55},
                        margin={"r": 150, "t": 40, "l": 80, "b": 0},
                        title=fig_title,
                        title_x=0.5)
    return c_fig, z_values


if plot_type == "Absolute Number":
    fig, z_values = plot_dog_map(breed_df,
                                 breeds_per_region,
                                 fig_title="Total Number of " + breed + "s in Zürich",
                                 hover_temp="Kreis: %{z:.0f}<br>" + "Number " + breed
                                            + "s: %{z:.0f}<br>" + "<extra></extra>")
elif plot_type == "Percentage":
    fig, z_values = plot_dog_map(breed_df,
                                 percent_per_region,
                                 fig_title="Percentage of " + breed + "s in Zürich",
                                 hover_temp="Kreis: %{z:.0f}<br>" + "Percent " + breed
                                            + "s: %{z:.2f}%<br>" + "<extra></extra>")

st.plotly_chart(fig)

st.header(" ")
st.subheader("Does age influence breed preference?")

# Process dog owner age data
dfdog['mean_age'] = dfdog.apply(lambda row: (int(row['ALTER'][0:2])+int(row['ALTER'][-2:]))/2, axis=1)
owner_age_per_region = dfdog.groupby('STADTKREIS').mean_age.mean()
age_hover_temp = hover_temp="Kreis: %{z:.0f}<br>" + "Average age: %{z:.0f}<br>" + "<extra></extra>"

# Plot dog owner mean age data
age_fig = go.Figure(go.Choroplethmapbox(geojson=regions,
                                        locations=breed_df.STADTKREIS.sort_values().unique(),
                                        z=owner_age_per_region,
                                        featureidkey='properties.name',
                                        colorscale="Plasma",
                                        marker_opacity=0.5,
                                        marker_line_width=0,
                                        hovertemplate=age_hover_temp))
age_fig.update_layout(mapbox_style="stamen-toner",
                      mapbox_zoom=10,
                      mapbox_center={"lat": 47.36667, "lon": 8.55},
                      margin={"r": 150, "t": 40, "l": 80, "b": 0},
                      title="Average Dog Owner Age in Zürich",
                      title_x=0.5)

st.plotly_chart(age_fig)

st.subheader(" ")

# Compute correlation between percentage of a certain breed and mean owner age
df_for_fit = pd.DataFrame([owner_age_per_region, z_values]).T
lm = smf.ols(formula="RASSE1 ~ mean_age", data=df_for_fit).fit()
rsquared = lm.rsquared

age_fit_fig, ax = plt.subplots()
regline = sns.regplot(x=owner_age_per_region, y=z_values, color='purple', ax=ax)
ax.set_xlabel('Age [years]', fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
if plot_type == "Absolute Number":
    ax.set_ylabel(breed + 's [total number]', fontsize=8)
    ax.set_title('Mean Dog Owner Age vs. Number ' + breed + 's Owned', fontsize=9)
elif plot_type == "Percentage":
    ax.set_ylabel(breed+'s [%]', fontsize=8)
    ax.set_title('Mean Dog Owner Age vs. Percentage ' + breed + 's Owned', fontsize=9)
ax.text(np.mean(owner_age_per_region)+.2, np.mean(z_values)+.2,
        'r\u00b2 = ' + str(rsquared.round(2)), fontsize=8, fontweight='bold')

st.pyplot(age_fit_fig)
