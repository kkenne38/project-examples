import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # As a simpler baseline
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

st.set_page_config(
    page_title="What's in Your Water?",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.epa.gov/ground-water-and-drinking-water',
        'Report a bug': None,
        'About': "This app analyzes California water quality data."
    },
)

# Custom CSS for a water-themed look
st.markdown(
    """
    <style>
    :root {
        --primary-color: #1E88E5; /* A nice blue */
        --secondary-color: #81D4FA; /* A lighter blue */
        --background-color: #E3F2FD; /* Very light blue background */
        --text-color: #212121; /* Dark gray text */
        --font-family: sans-serif;
    }
    body {
        color: var(--text-color);
        background-color: var(--background-color);
        font-family: var(--font-family);
    }
    .streamlit-header {
        background-color: var(--primary-color);
        color: white;
    }
    .st-sidebar {
        background-color: var(--secondary-color);
        color: var(--text-color);
    }
    .st-tabs>ul>li>a {
        color: var(--primary-color);
    }
    .st-tabs>ul>li.active>a {
        background-color: var(--primary-color);
        color: white;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Data Loading and Preparation ---
@st.cache_data
def load_data():
    data = pd.read_csv("C:\\Users\\kbk10\\Desktop\\Thesis_Analysis\\DataFrames\\cleaned_median_weighted.csv")
    return data

def filter_data(df, selected_zips, selected_years):
    filtered_df = df[(df['zip_code'].isin(selected_zips)) & (df['year'].isin(selected_years))]
    return filtered_df

def get_contaminant_columns(df):
    contaminant_cols = [
        col for col in df.columns 
        if col not in ['zip_code', 'year', 'cause', 'count', 'population'] and col in epa_limits
    ]
    return contaminant_cols

@st.cache_data
def get_zip_code_coordinates(zip_codes):
    geolocator = Nominatim(user_agent="water_quality_app", timeout=3)  # Added timeout for robustness
    coordinates = {}
    for zip_code in zip_codes:
        try:
            location = geolocator.geocode(f"{zip_code}, CA, USA")
            if location:
                coordinates[zip_code] = (location.latitude, location.longitude)
            else:
                coordinates[zip_code] = None
        except GeocoderTimedOut:
            st.warning(f"Geocoding timed out for zip code: {zip_code}. Retrying in 1 second...")
            time.sleep(1)
            try:
                location = geolocator.geocode(f"{zip_code}, CA, USA")
                if location:
                    coordinates[zip_code] = (location.latitude, location.longitude)
                else:
                    coordinates[zip_code] = None
            except GeocoderTimedOut:
                st.error(f"Geocoding failed for zip code: {zip_code} after retry.")
                coordinates[zip_code] = None
            except Exception as e:
                st.error(f"An error occurred while geocoding {zip_code} (retry): {e}")
                coordinates[zip_code] = None
        except Exception as e:
            st.error(f"An error occurred while geocoding {zip_code}: {e}")
            coordinates[zip_code] = None
    return coordinates

# --- Function to train XGBoost model ---
def train_xgboost_model(df, contaminant):
    features = ['year', 'population'] # Using these as basic features
    if contaminant not in df.columns:
        st.error(f"Contaminant '{contaminant}' not found in the data.")
        return None

    df_model = df[['zip_code', 'year', 'population', contaminant]].dropna()
    if df_model.empty:
        st.warning(f"No data available to train XGBoost for '{contaminant}' in the selected zip code.")
        return None

    X = df_model[features]
    y = df_model[contaminant]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Simple ANN Model ---
class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.output_relu = nn.ReLU()  # ReLU for the output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.output_relu(out)  # Apply ReLU to the final output
        return out

def train_ann_model(df, contaminant, hidden_size=10, learning_rate=0.01, epochs=100):
    features = ['year', 'population']
    if contaminant not in df.columns:
        st.error(f"Contaminant '{contaminant}' not found in the data.")
        return None

    df_model = df[['zip_code', 'year', 'population', contaminant]].dropna()
    if df_model.empty:
        st.warning(f"No data available to train ANN for '{contaminant}' in the selected zip code.")
        return None

    X = df_model[features].values.astype(np.float32)
    y = df_model[contaminant].values.astype(np.float32).reshape(-1, 1)

    # Scale features
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    # Scale target
    scaler_y = MinMaxScaler()  # Or StandardScaler
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_scaled))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_size = X_train.shape[1]
    output_size = 1
    model = SimpleANN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    st.info(f"Training ANN for '{contaminant}'...")
    progress_bar = st.progress(0)

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)

    st.success(f"ANN training for '{contaminant}' complete.")
    return model, scaler_x, scaler_y # Return the scalers

# EPA Safe Limits for various contaminants
epa_limits = {
    'total_coliform': 'Treatment Technique',
    'e_coli': 'Treatment Technique',
    'giardia_lamblia': 'Treatment Technique',
    'cryptosporidium': 'Treatment Technique',
    'viruses': 'Treatment Technique',
    'chlorine': 4.0,  # MRDL mg/L
    'chloramines': 4.0,  # MRDL mg/L
    'chlorine_dioxide': 0.8,  # MCL mg/L
    'chlorite': 1.0,  # MCL mg/L
    'haloacetic_acids_haa5': 0.060,  # MCL mg/L
    'total_trihalomethanes_tthms': 0.080,  # MCL mg/L
    'bromate': 0.010,  # MCL mg/L
    'antimony': 0.006,  # MCL mg/L
    'arsenic': 0.010,  # MCL mg/L
    'asbestos': 7000000,  # MCL fibers/liter
    'barium': 2.0,  # MCL mg/L
    'beryllium': 0.004,  # MCL mg/L
    'cadmium': 0.005,  # MCL mg/L
    'chromium': 0.1,  # MCL mg/L
    'copper': 1.3,  # Action Level mg/L (Treatment Technique Trigger)
    'cyanide': 0.2,  # MCL mg/L
    'fluoride': 4.0,  # MCL mg/L
    'lead': 0.015,  # Action Level mg/L (Treatment Technique Trigger)
    'mercury': 0.002,  # MCL mg/L
    'nitrate_as_n': 10.0,  # MCL mg/L
    'nitrite_as_n': 1.0,  # MCL mg/L
    'selenium': 0.05,  # MCL mg/L
    'thallium': 0.002,  # MCL mg/L
    'benzene': 0.005,  # MCL mg/L
    'carbon_tetrachloride': 0.005,  # MCL mg/L
    'chlorobenzene': 0.1,  # MCL mg/L
    'dichlorobenzene_1_2': 0.6,  # MCL mg/L
    'dichlorobenzene_1_4': 0.075,  # MCL mg/L
    'dichloroethane_1_2': 0.005,  # MCL mg/L
    'dichloroethylene_cis_1_2': 0.07,  # MCL mg/L
    'dichloroethylene_trans_1_2': 0.1,  # MCL mg/L
    'dichloromethane': 0.005,  # MCL mg/L
    'dichloropropane_1_2': 0.005,  # MCL mg/L
    'ethylbenzene': 0.7,  # MCL mg/L
    'styrene': 0.1,  # MCL mg/L
    'tetrachloroethylene': 0.005,  # MCL mg/L
    'toluene': 1.0,  # MCL mg/L
    'trichlorobenzene_1_2_4': 0.07,  # MCL mg/L
    'trichloroethane_1_1_1': 0.2,  # MCL mg/L
    'trichloroethane_1_1_2': 0.005,  # MCL mg/L
    'trichloroethylene': 0.005,  # MCL mg/L
    'vinyl_chloride': 0.002,  # MCL mg/L
    'xylenes_total': 10.0,  # MCL mg/L
    'd_2_4': 0.07,  # MCL mg/L
    'tp_2_4_5_silvex': 0.05,  # MCL mg/L
    'alachlor': 0.002,  # MCL mg/L
    'atrazine': 0.003,  # MCL mg/L
    'benzo_a_pyrene': 0.0002,  # MCL mg/L
    'carbofuran': 0.04,  # MCL mg/L
    'chlordane': 0.002,  # MCL mg/L
    'dalapon': 0.2,  # MCL mg/L
    'di_2_ethylhexyl_adipate': 0.4,  # MCL mg/L
    'di_2_ethylhexyl_phthalate': 0.006,  # MCL mg/L
    'dibromochloropropane_dbcp': 0.0002,  # MCL mg/L
    'dinoseb': 0.007,  # MCL mg/L
    'diquat': 0.02,  # MCL mg/L
    'endothall': 0.1,  # MCL mg/L
    'endrin': 0.002,  # MCL mg/L
    'ethylene_dibromide_edb': 0.00005,  # MCL mg/L
    'glyphosate': 0.7,  # MCL mg/L
    'heptachlor': 0.0004,  # MCL mg/L
    'heptachlor_epoxide': 0.0002,  # MCL mg/L
    'hexachlorobenzene': 0.001,  # MCL mg/L
    'hexachlorocyclopentadiene': 0.05,  # MCL mg/L
    'lindane': 0.0002,  # MCL mg/L
    'methoxychlor': 0.04,  # MCL mg/L
    'metolachlor': 0.15,  # MCL mg/L
    'metribuzin': 0.04,  # MCL mg/L
    'molinate': 0.07,  # MCL mg/L
    'oxamyl': 0.2,  # MCL mg/L
    'pcbs': 0.0005,  # MCL mg/L
    'pentachlorophenol': 0.001,  # MCL mg/L
    'picloram': 0.5,  # MCL mg/L
    'simazine': 0.004,  # MCL mg/L
    'toxaphene': 0.003,  # MCL mg/L
    'gross_alpha_particle_activity': 15,  # MCL pCi/L
    'combined_radium_226_228': 5,  # MCL pCi/L
    'uranium': 0.030,  # MCL mg/L
    'beta_particle_photon_emitters': 'Treatment Technique',  # MCL mrem/year
    'pfoa': 4.0 / 1000000,  # MCL mg/L (converted from ng/L)
    'pfos': 4.0 / 1000000,  # MCL mg/L (converted from ng/L)
    'pfhxs': 10.0 / 1000000,  # MCL mg/L (converted from ng/L)
    'pfna': 10.0 / 1000000,  # MCL mg/L (converted from ng/L)
    'hfpo_da': 10.0 / 1000000,  # MCL mg/L (converted from ng/L)
    'pfas_mixtures_hazard_index': 1.0  # MCL (unitless)
}

# --- Main App ---
def main():
    st.title("What's in Your Water? 💧")
    st.header("California Water Quality Data (2012-2022)")

    data = load_data()

    tab1, tab2 = st.tabs(["Exploratory Data Analysis", "Predictive Analysis"])

    # --- Exploratory Data Analysis Tab ---
    with tab1:
        st.header("Exploratory Data Analysis")

        st.sidebar.header("Filter Options")
        zip_codes = sorted(data['zip_code'].unique())
        selected_zips = st.sidebar.multiselect("Select Zip Codes:", zip_codes)

        years = sorted(data['year'].unique())
        selected_years = st.sidebar.multiselect("Select Years:", years, default=years)

        if selected_zips:
            filtered_data = filter_data(data, selected_zips, selected_years)

            st.subheader(f"Water Quality Data for Zip Codes: {', '.join(map(str, selected_zips))} ({', '.join(map(str, selected_years))})")

            contaminant_cols = get_contaminant_columns(data)
            default_contaminants = ["arsenic", "lead", "nitrate", "fluoride", "bromate", "radium_228"]
            pre_selected = [contaminant for contaminant in default_contaminants if contaminant in contaminant_cols]
            selected_contaminants = st.multiselect("Select Contaminants to Visualize:", contaminant_cols, default=pre_selected)

            if selected_contaminants:
                st.subheader("Contaminant Levels Over Time")
                plot_data = filtered_data[['year', 'zip_code'] + selected_contaminants].melt(
                    id_vars=['year', 'zip_code'],
                    var_name='Contaminant',
                    value_name='Level'
                )

                fig = px.line(
                    plot_data,
                    x='year',
                    y='Level',
                    color='Contaminant',
                    facet_col='zip_code',
                    facet_col_wrap=2,
                    title="Contaminant Levels in Selected Zip Codes Over Selected Years",
                    labels={
                        'Level': 'Contaminant Level',
                        'year': 'Year',
                        'zip_code': 'Zip Code'
                    },
                    color_discrete_sequence=px.colors.sequential.Viridis
                )

                fig.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.6)',
                    paper_bgcolor='rgba(255,255,255,0.6)',
                    font=dict(color='black'),
                    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
                    yaxis=dict(showgrid=True, gridcolor='lightgrey')
                )    

                st.plotly_chart(fig)
            else:
                st.info("Please select at least one contaminant to visualize.")
            if st.checkbox("Show Raw Data"):
                st.subheader("Raw Data")
                st.dataframe(filtered_data)
            # --- Map Visualization ---
            st.subheader("Geographic Visualization of Selected Zip Codes")
            zip_coordinates = get_zip_code_coordinates(selected_zips)

            if any(zip_coordinates.values()):
                m = folium.Map(location=[37.0902, -119.5575], zoom_start=6)

                for zip_code, coords in zip_coordinates.items():
                    if coords:
                        folium.Circle(
                            location=coords,
                            radius=800,
                            color='blue',
                            fill=True,
                            fill_color='lightblue',
                            fill_opacity=0.3,
                            tooltip=f"Zip Code: {zip_code} Area"
                        ).add_to(m)

                        folium.Marker(
                            location=coords,
                            popup=f"Zip Code: {zip_code}",
                            tooltip=zip_code
                        ).add_to(m)
                    else:
                        st.warning(f"Could not retrieve coordinates for zip code: {zip_code}")

                st_folium(m, width=700, height=500)
            else:
                st.info("Could not retrieve coordinates for any of the selected zip codes.")
        else:
            st.info("Please select at least one zip code to visualize.")

    # --- Predictive Analysis Tab ---
    with tab2:
        st.header("Predictive Analysis")
        st.write("Predict future contaminant levels based on historical data.")

        zip_codes = sorted(data['zip_code'].unique())
        predict_zip = st.selectbox("Select Zip Code for Prediction:", zip_codes, key="predict_zip")
        contaminant_cols = get_contaminant_columns(data)
        predict_contaminant = st.selectbox("Select Contaminant to Predict:", contaminant_cols, key="predict_contaminant")
        predict_year = st.slider("Select Year for Prediction:", min_value=data['year'].max() + 1, max_value=2030, value=data['year'].max() + 1, key="predict_year")
        model_type = st.selectbox("Select Prediction Model:", ["XGBoost", "Simple ANN"], key="predict_model")
        predict_button = st.button("Predict", key="predict_button")

        if predict_button:
            # Prepare prediction input for the selected zip code
            predict_data_zip = data[data['zip_code'] == predict_zip][['year', 'population']].dropna().mean().to_frame().T
            predict_data = pd.DataFrame({
                'year': [predict_year],
                'population': [predict_data_zip['population'].iloc[0] if not predict_data_zip.empty and 'population' in predict_data_zip.columns else data['population'].mean()]
            })

            if model_type == "XGBoost":
                model = train_xgboost_model(data[data['zip_code'] == predict_zip], predict_contaminant)
                if model:
                    try:
                        prediction = model.predict(predict_data[['year', 'population']])
                        st.subheader("Prediction Result (XGBoost)")
                        st.write(f"Predicted level of {predict_contaminant} in {predict_zip} for {predict_year}: {prediction[0]:.4f} mg/L")
                        safe_limit = epa_limits.get(predict_contaminant.lower(), None)
                        if safe_limit:
                            st.write(f"EPA Safe Limit for {predict_contaminant}: {safe_limit} mg/L")
                            if prediction[0] > safe_limit:
                                st.error(f"⚠️ Predicted level ({prediction[0]:.4f} mg/L) exceeds the EPA safe limit for {predict_contaminant}!")
                            else:
                                st.success(f"✅ Predicted level ({prediction[0]:.4f} mg/L) is within the EPA safe limit for {predict_contaminant}.")
                        else:
                            st.warning(f"No EPA safe limit found for {predict_contaminant}.")
                    except Exception as e:
                        st.error(f"XGBoost Prediction failed: {e}")
                else:
                    st.warning(f"Could not train XGBoost model for {predict_contaminant} in the selected zip code.")

            elif model_type == "Simple ANN":
                model_tuple = train_ann_model(data[data['zip_code'] == predict_zip], predict_contaminant)
                if model_tuple:
                    model, scaler_x, scaler_y = model_tuple
                    try:
                        predict_data_ann = pd.DataFrame({
                            'year': [predict_year],
                            'population': [predict_data_zip['population'].iloc[0] if not predict_data_zip.empty and 'population' in predict_data_zip.columns else data['population'].mean()]
                        }).values.astype(np.float32)

                        # Scale the prediction input using the fitted feature scaler
                        predict_data_ann_scaled = scaler_x.transform(predict_data_ann)

                        with torch.no_grad():
                            prediction_scaled = model(torch.tensor(predict_data_ann_scaled)).numpy()

                        # Inverse transform the prediction using the fitted target scaler
                        prediction = scaler_y.inverse_transform(prediction_scaled)

                        st.subheader("Prediction Result (Simple ANN)")
                        st.write(f"Predicted level of {predict_contaminant} in {predict_zip} for {predict_year}: {prediction[0][0]:.4f} mg/L")
                        safe_limit = epa_limits.get(predict_contaminant.lower(), None)
                        if safe_limit:
                            st.write(f"EPA Safe Limit for {predict_contaminant}: {safe_limit} mg/L")
                            if prediction[0][0] > safe_limit:
                                st.error(f"⚠️ Predicted level ({prediction[0][0]:.4f} mg/L) exceeds the EPA safe limit for {predict_contaminant}!")
                            else:
                                st.success(f"✅ Predicted level ({prediction[0][0]:.4f} mg/L) is within the EPA safe limit for {predict_contaminant}.")
                        else:
                            st.warning(f"No EPA safe limit found for {predict_contaminant}.")

                    except Exception as e:
                        st.error(f"ANN Prediction failed: {e}")
                else:
                    st.warning(f"Could not train Simple ANN model for {predict_contaminant} in the selected zip code.")

if __name__ == "__main__":
    main()