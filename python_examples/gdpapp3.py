import streamlit as st
import pandas as pd
import plotly.express as px
import requests as rq
import bs4
import re
from io import StringIO

country_region_map = {
'World' : 'World',
'Puerto Rico' : 'North America',
'Palestine' : 'Asia',
'Algeria' : 'Africa',
'Angola' : 'Africa' ,
'Benin' : 'Africa' ,
'Botswana' : 'Africa' ,
'Burkina' : 'Africa' ,
'Burkina Faso' : 'Africa' ,
'Burundi' : 'Africa' ,
'Cameroon' : 'Africa' ,
'Cape Verde' : 'Africa' ,
'Central African Republic' : 'Africa' ,
'Chad' : 'Africa' ,
'Comoros' : 'Africa' ,
'Congo' : 'Africa' ,
'Democratic Republic of Congo' : 'Africa' ,
'Djibouti' : 'Africa' ,
'Egypt' : 'Africa' ,
'Equatorial Guinea' : 'Africa' ,
'Eritrea' : 'Africa' ,
'Ethiopia' : 'Africa' ,
'Gabon' : 'Africa' ,
'Gambia' : 'Africa' ,
'Ghana' : 'Africa' ,
'Guinea' : 'Africa' ,
'Guinea-Bissau' : 'Africa' ,
'Ivory Coast' : 'Africa' ,
'Kenya' : 'Africa' ,
'Lesotho' : 'Africa' ,
'Liberia' : 'Africa' ,
'Libya' : 'Africa' ,
'Madagascar' : 'Africa' ,
'Malawi' : 'Africa' ,
'Mali' : 'Africa' ,
'Mauritania' : 'Africa' ,
'Mauritius' : 'Africa' ,
'Morocco' : 'Africa' ,
'Mozambique' : 'Africa' ,
'Namibia' : 'Africa' ,
'Niger' : 'Africa' ,
'Nigeria' : 'Africa' ,
'Rwanda' : 'Africa' ,
'São Tomé and Príncipe' : 'Africa' ,
'Senegal' : 'Africa' ,
'Seychelles' : 'Africa' ,
'DR Congo' : 'Africa' ,
'Eswatini' : 'Africa' ,
'Sierra Leone' : 'Africa' ,
'Somalia' : 'Africa' ,
'South Africa' : 'Africa' ,
'South Sudan' : 'Africa' ,
'Sudan' : 'Africa' ,
'Swaziland' : 'Africa' ,
'Tanzania' : 'Africa' ,
'Togo' : 'Africa' ,
'Tunisia' : 'Africa' ,
'Uganda' : 'Africa' ,
'Zambia' : 'Africa' ,
'Zimbabwe' : 'Africa' ,
'Afghanistan' : 'Asia' ,
'Bahrain' : 'Asia' ,
'Bangladesh' : 'Asia' ,
'Bhutan' : 'Asia' ,
'Brunei' : 'Asia' ,
'Burma (Myanmar)' : 'Asia' ,
'Cambodia' : 'Asia' ,
'China' : 'Asia' ,
'East Timor' : 'Asia' ,
'Hong Kong' : 'Asia' ,
'India' : 'Asia' ,
'Indonesia' : 'Asia' ,
'Iran' : 'Asia' ,
'Iraq' : 'Asia' ,
'Israel' : 'Asia' ,
'Japan' : 'Asia' ,
'Jordan' : 'Asia' ,
'Kazakhstan' : 'Asia' ,
'North Korea' : 'Asia' ,
'South Korea' : 'Asia' ,
'Kuwait' : 'Asia' ,
'Kyrgyzstan' : 'Asia' ,
'Laos' : 'Asia' ,
'Lebanon' : 'Asia' ,
'Malaysia' : 'Asia' ,
'Maldives' : 'Asia' ,
'Macau' : 'Asia' ,
'Mongolia' : 'Asia' ,
'Myanmar' : 'Asia' ,
'Nepal' : 'Asia' ,
'Oman' : 'Asia' ,
'Pakistan' : 'Asia' ,
'Philippines' : 'Asia' ,
'Qatar' : 'Asia' ,
'Russia' : 'Asia' ,
'Saudi Arabia' : 'Asia' ,
'Singapore' : 'Asia' ,
'Sri Lanka' : 'Asia' ,
'Syria' : 'Asia' ,
'Taiwan' : 'Asia' ,
'Tajikistan' : 'Asia' ,
'Timor-Leste' : 'Asia' ,
'Thailand' : 'Asia' ,
'Turkey' : 'Europe' ,
'Turkmenistan' : 'Asia' ,
'United Arab Emirates' : 'Asia' ,
'Uzbekistan' : 'Asia' ,
'Vietnam' : 'Asia' ,
'Yemen' : 'Asia' ,
'Albania' : 'Europe' ,
'Andorra' : 'Europe' ,
'Armenia' : 'Europe' ,
'Austria' : 'Europe' ,
'Azerbaijan' : 'Europe' ,
'Belarus' : 'Europe' ,
'Belgium' : 'Europe' ,
'Bosnia and Herzegovina' : 'Europe' ,
'Bulgaria' : 'Europe' ,
'Croatia' : 'Europe' ,
'Cyprus' : 'Europe' ,
'Czechia' : 'Europe' ,
'Denmark' : 'Europe' ,
'Estonia' : 'Europe' ,
'Finland' : 'Europe' ,
'France' : 'Europe' ,
'Georgia' : 'Europe' ,
'Germany' : 'Europe' ,
'Greece' : 'Europe' ,
'Hungary' : 'Europe' ,
'Iceland' : 'Europe' ,
'Ireland' : 'Europe' ,
'Italy' : 'Europe' ,
'Kosovo' : 'Europe' ,
'Czech Republic' : 'Europe' ,
'Latvia' : 'Europe' ,
'Liechtenstein' : 'Europe' ,
'Lithuania' : 'Europe' ,
'Luxembourg' : 'Europe' ,
'Macedonia' : 'Europe' ,
'Malta' : 'Europe' ,
'Moldova' : 'Europe' ,
'Monaco' : 'Europe' ,
'Montenegro' : 'Europe' ,
'North Macedonia' : 'Europe' ,
'Netherlands' : 'Europe' ,
'Norway' : 'Europe' ,
'Poland' : 'Europe' ,
'Portugal' : 'Europe' ,
'Romania' : 'Europe' ,
'San Marino' : 'Europe' ,
'Serbia' : 'Europe' ,
'Slovakia' : 'Europe' ,
'Slovenia' : 'Europe' ,
'Spain' : 'Europe' ,
'Sweden' : 'Europe' ,
'Switzerland' : 'Europe' ,
'Ukraine' : 'Europe' ,
'United Kingdom' : 'Europe' ,
'Vatican City' : 'Europe' ,
'Antigua and Barbuda' : 'North America' ,
'Aruba' : 'North America' ,
'Bahamas' : 'North America' ,
'Barbados' : 'North America' ,
'Belize' : 'North America' ,
'Canada' : 'North America' ,
'Costa Rica' : 'North America' ,
'Cuba' : 'North America' ,
'Dominica' : 'North America' ,
'Dominican Republic' : 'North America' ,
'El Salvador' : 'North America' ,
'Grenada' : 'North America' ,
'Guatemala' : 'North America' ,
'Haiti' : 'North America' ,
'Honduras' : 'North America' ,
'Jamaica' : 'North America' ,
'Mexico' : 'North America' ,
'Nicaragua' : 'North America' ,
'Panama' : 'North America' ,
'Saint Kitts and Nevis' : 'North America' ,
'Saint Lucia' : 'North America' ,
'Saint Vincent and the Grenadines' : 'North America' ,
'Trinidad and Tobago' : 'North America' ,
'United States' : 'North America' ,
'Australia' : 'Oceania' ,
'Fiji' : 'Oceania' ,
'Kiribati' : 'Oceania' ,
'Marshall Islands' : 'Oceania' ,
'Micronesia' : 'Oceania' ,
'Nauru' : 'Oceania' ,
'New Zealand' : 'Oceania' ,
'Palau' : 'Oceania' ,
'Papua New Guinea' : 'Oceania' ,
'Samoa' : 'Oceania' ,
'Solomon Islands' : 'Oceania' ,
'Tonga' : 'Oceania' ,
'Tuvalu' : 'Oceania' ,
'Vanuatu' : 'Oceania' ,
'Argentina' : 'South America' ,
'Bolivia' : 'South America' ,
'Brazil' : 'South America' ,
'Chile' : 'South America' ,
'Colombia' : 'South America' ,
'Ecuador' : 'South America' ,
'Guyana' : 'South America' ,
'Paraguay' : 'South America' ,
'Peru' : 'South America' ,
'Suriname' : 'South America' ,
'Uruguay' : 'South America' ,
'Venezuela' : 'South America' ,
}

# Streamlit Page Configuration
st.set_page_config(page_title="Global GDP Explorer", page_icon="🌍", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .description {
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<h1 class='main-header'>🌍 Global GDP Explorer</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='description'>
Welcome to the Global GDP Explorer! This interactive visualization shows the International Monetary Fund's (IMF) 
GDP data for countries around the world, organized by region.
</div>
""", unsafe_allow_html=True)

# Sidebar with Information
with st.sidebar:
    st.header("📊 About the Data")
    st.markdown("""
    This visualization uses the latest IMF GDP estimates, sourced from Wikipedia.
    The data shows nominal GDP in millions of US dollars.
    """)

# Function to scrape and clean GDP data
def scrape_and_clean_gdp_data():
    url = 'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)'
    page = rq.get(url)
    
    # Parse the HTML with BeautifulSoup
    bs4page = bs4.BeautifulSoup(page.text, 'html.parser')
    tables = bs4page.find_all('table', {'class': "wikitable"})
    GDP = pd.read_html(StringIO(str(tables[0])))[0]
    
    # Clean the DataFrame
    GDP = GDP.dropna()
    GDP.columns = GDP.columns.get_level_values(-1)
    GDP = GDP.rename(columns={'Forecast': 'IMF', 'Country/Territory': 'Country'})
    
    # Find the 'Year' column and clean it
    year_column_index = next((i for i, col in enumerate(GDP.columns) if 'Year' in col), None)
    if year_column_index is not None:
        GDP = GDP.iloc[:, :year_column_index+1]
    
    # Clean the 'Year' column
    def clean_year(year_str):
        return re.sub(r'[n \d+]', '', year_str)
    
    GDP['Year'] = GDP['Year'].astype(str).apply(clean_year)
    
    # Remove invalid rows
    GDP = GDP[GDP['IMF'] != '__']
    GDP = GDP[GDP['Country'] != 'World']
    GDP['IMF'] = pd.to_numeric(GDP['IMF'], errors='coerce')
    
    # Map continents to countries
    GDP['Continent'] = GDP['Country'].map(country_region_map)
    GDP['Continent'].fillna('Unknown', inplace=True)
    GDP = GDP[GDP['Continent'] != 'Unknown']
    
    return GDP

# Fetch the GDP data and show a loading spinner
with st.spinner("📊 Fetching the latest GDP data..."):
    gdp_data = scrape_and_clean_gdp_data()

# Check if data was fetched successfully
if gdp_data is None or gdp_data.empty:
    st.error("❌ Error fetching GDP data. Please try again later.")
else:
    # Create a Plotly bar chart
    fig = px.bar(
        gdp_data,
        x="Continent",
        y="IMF",
        color="Country",
        title="Global GDP Distribution by Region",
        labels={"Continent": "World Region", "IMF": "GDP (Millions USD)", "Country": "Country"},
        height=600
    )

    # Update layout for the chart
    fig.update_layout(
        barmode='stack',
        showlegend=True,
        legend_title_text="Countries",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        title=dict(font=dict(size=24), y=0.95),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Display the Plotly chart
    st.plotly_chart(fig, use_container_width=True)

    # Display the raw data in an interactive table
    st.subheader("🔍 Explore the Raw Data")
    st.markdown("Search and sort through the GDP data using the interactive table below:")

    # Format GDP values for better readability
    gdp_data_display = gdp_data.copy()
    gdp_data_display['GDP (Millions USD)'] = gdp_data_display['IMF'] / 1000000
    gdp_data_display = gdp_data_display[['Country', 'Continent', 'GDP (Millions USD)']]

    st.dataframe(gdp_data_display)

    # Add footer with information
    st.markdown("""
    ---
    ### 📚 Data Sources and Information
    - Data source: [Wikipedia - List of countries by GDP (nominal)](https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal))
    - Data provider: International Monetary Fund
    - Last updated: {}
    Made with ❤️ using Streamlit and Plotly
    """.format(gdp_data['Year'].iloc[0] if 'Year' in gdp_data.columns else "N/A"))
