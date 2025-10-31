import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="COVID-19 Tracker", layout="wide")
st.title("🦠 COVID-19 Tracker (CSSE Data)")
st.markdown("Select a country to view COVID-19 statistics using Johns Hopkins CSSE data.")

# Load data with caching
@st.cache_data
def load_covid_data():
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

    # Load dataset
    df = pd.read_csv(url)

    # Convert wide format to long format (melting date columns)
    df_melted = df.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"], 
        var_name="Date", 
        value_name="Confirmed Cases"
    )

    # Convert 'Date' column to datetime format
    df_melted["Date"] = pd.to_datetime(df_melted["Date"], format="%m/%d/%y", errors="coerce")

    # Drop NaN values in Cases
    df_melted = df_melted.dropna(subset=["Confirmed Cases"])

    # Convert Cases column to numeric
    df_melted["Confirmed Cases"] = pd.to_numeric(df_melted["Confirmed Cases"], errors="coerce")

    # Sort data by Country/Region and Date to ensure correct order for calculations
    df_melted = df_melted.sort_values(by=["Country/Region", "Date"])

    # Aggregate by country and date, summing the confirmed cases (to handle multiple entries per date)
    df_aggregated = df_melted.groupby(["Country/Region", "Date"]).agg({"Confirmed Cases": "sum"}).reset_index()

    # Compute cumulative cases (running total of Confirmed Cases for each country)
    df_aggregated["Cumulative Cases"] = df_aggregated.groupby(["Country/Region"])["Confirmed Cases"].cumsum()

    # Daily cases is simply the Confirmed Cases for that day (already in the dataset)
    df_aggregated["Daily Cases"] = df_aggregated["Confirmed Cases"]

    return df_aggregated, df  # Return original df as well for map

# Load data
df_covid, df_raw = load_covid_data()

# Display Global Choropleth Map (Always at the top of the page)
# Get the latest data by country
latest_data = df_covid[df_covid["Date"] == df_covid["Date"].max()]

# Aggregate by country
country_cases = latest_data.groupby("Country/Region").agg({"Confirmed Cases": "sum"}).reset_index()

# Plot choropleth map
choropleth_fig = px.choropleth(
    country_cases,
    locations="Country/Region",
    locationmode="country names",  # Ensure countries are matched by name
    color="Confirmed Cases",
    hover_name="Country/Region",
    color_continuous_scale="Viridis",  # You can change this to any other color scale
    title="Global Distribution of COVID-19 Confirmed Cases",
    labels={"Confirmed Cases": "Confirmed Cases"},
)

choropleth_fig.update_geos(showcoastlines=True, coastlinecolor="Black", projection_type="mercator")

# Layout: Side-by-Side with equal column sizes
col1, col2 = st.columns([1, 1])  # Equal width columns

# Sidebar for user input
st.sidebar.header("User Input")
selected_countries = st.sidebar.multiselect(
    "Select countries:",
    df_covid["Country/Region"].unique().tolist(),
    default=["US", "India"]
)

display_option = st.sidebar.radio("Select data to display:", ["Daily Cases", "Cumulative Cases"])

# Filter data for selected countries
filtered_data = df_covid[df_covid["Country/Region"].isin(selected_countries)]

if len(selected_countries) > 0:
    # Ensure all countries have the same date range (union of dates)
    all_dates = pd.date_range(start=filtered_data["Date"].min(), end=filtered_data["Date"].max(), freq="D")

    # Reindex each country's data to include all dates
    aligned_data = []
    for country in selected_countries:
        country_data = filtered_data[filtered_data["Country/Region"] == country]
        country_data = country_data.set_index("Date").reindex(all_dates).reset_index()
        country_data["Date"] = country_data["index"]
        country_data = country_data.drop(columns=["index"])

        # Fill missing values (fill NaN with 0 for daily cases, or forward-fill for cumulative cases)
        if display_option == "Daily Cases":
            country_data["Daily Cases"] = country_data["Daily Cases"].fillna(0)
        else:
            country_data["Cumulative Cases"] = country_data["Cumulative Cases"].fillna(method="ffill")

        aligned_data.append(country_data)

    # Concatenate the aligned data for all countries
    df_aligned = pd.concat(aligned_data)

    # Plot data (Line chart for Daily/Cumulative cases)
    fig = px.line(title=f"COVID-19 {display_option} Over Time")

    for country in selected_countries:
        country_data = df_aligned[df_aligned["Country/Region"] == country]

        if display_option == "Daily Cases":
            fig.add_scatter(
                x=country_data["Date"], y=country_data["Daily Cases"], mode="lines", name=f"{country} - Daily Cases"
            )
        else:
            fig.add_scatter(
                x=country_data["Date"], y=country_data["Cumulative Cases"], mode="lines", name=f"{country} - Cumulative Cases"
            )

    fig.update_layout(xaxis_title="Date", yaxis_title="Cases", template="plotly_dark")

    # Choropleth Map on the right
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    # Choropleth map on the right column
    with col2:
        st.plotly_chart(choropleth_fig, use_container_width=True)

else:
    st.write("Please select at least one country to view the data.")

