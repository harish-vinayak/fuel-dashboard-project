import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- 1. Load and Clean Data ---
@st.cache_data
def load_data():
    # Load datasets
    df1 = pd.read_csv("MY2022 Fuel Consumption Ratings.csv", encoding='latin1')
    df2 = pd.read_csv("Fuel Consumption Ratings 2023.csv", encoding='latin1')
    df3 = pd.read_csv("Fuel Consumption Ratings 2024.csv", encoding='latin1')

    # Clean column names
    def clean_cols(df):
        df.columns = (df.columns
                      .str.strip()
                      .str.lower()
                      .str.replace(' ', '_')
                      .str.replace(r'\(', '', regex=True)
                      .str.replace(r'\)', '', regex=True))
        return df

    df1, df2, df3 = clean_cols(df1), clean_cols(df2), clean_cols(df3)

    # Rename to standard format
    df1.rename(columns={
        'engine_sizel': 'engine_size_l',
        'fuel_consumption_city_l/100_km': 'city_l_100km',
        'fuel_consumptionhwy_l/100_km': 'highway_l_100km',
        'fuel_consumptioncomb_l/100_km': 'combined_l_100km',
        'fuel_consumptioncomb_mpg': 'combined_mpg',
        'co2_emissionsg/km': 'co2_emissions_g/km'
    }, inplace=True)

    df2.rename(columns={
        'year': 'model_year',
        'fuel_consumption_l/100km': 'city_l_100km',
        'hwy_l/100_km': 'highway_l_100km',
        'comb_l/100_km': 'combined_l_100km',
        'comb_mpg': 'combined_mpg'
    }, inplace=True)

    df3.rename(columns={
        'city_l/100_km': 'city_l_100km',
        'highway_l/100_km': 'highway_l_100km',
        'combined_l/100_km': 'combined_l_100km'
    }, inplace=True)

    for df, year in zip([df1, df2, df3], [2022, 2023, 2024]):
        df['source_year'] = year

    common_cols = ['model_year', 'make', 'model', 'vehicle_class', 'engine_size_l',
                   'cylinders', 'transmission', 'fuel_type', 'city_l_100km',
                   'highway_l_100km', 'combined_l_100km', 'combined_mpg',
                   'co2_emissions_g/km', 'co2_rating', 'smog_rating', 'source_year']

    df_all = pd.concat([df1[common_cols], df2[common_cols], df3[common_cols]], ignore_index=True)
    return df_all.dropna()

df_all = load_data()

# --- 2. Train the Model ---
@st.cache_resource
def train_model(data):
    encoders = {}
    for col in ['transmission', 'fuel_type']:
        le = LabelEncoder()
        data.loc[:, col] = le.fit_transform(data[col])
        encoders[col] = le

    X = data[['engine_size_l', 'cylinders', 'transmission', 'fuel_type', 'combined_l_100km']]
    y = data['co2_emissions_g/km']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    return model, encoders

model, encoders = train_model(df_all.copy())

# --- 3. Streamlit Dashboard UI ---
st.title("🚗 CO₂ Emissions Predictor for 2025 Vehicles")
st.write("Enter the specifications for a hypothetical vehicle to predict its CO₂ emissions.")

with st.sidebar:
    st.header("🛠 Vehicle Specs")
    engine_size = st.slider("Engine Size (L)", 1.0, 5.0, 2.0)
    cylinders = st.slider("Number of Cylinders", 2, 12, 4)
    transmission = st.selectbox("Transmission", encoders['transmission'].classes_)
    fuel_type = st.selectbox("Fuel Type", encoders['fuel_type'].classes_)
    combined_fuel = st.slider("Combined Fuel Consumption (L/100 km)", 4.0, 15.0, 6.5)

# --- 4. Make Prediction ---
trans_enc = encoders['transmission'].transform([transmission])[0]
fuel_enc = encoders['fuel_type'].transform([fuel_type])[0]

input_data = pd.DataFrame(
    [[engine_size, cylinders, trans_enc, fuel_enc, combined_fuel]],
    columns=['engine_size_l', 'cylinders', 'transmission', 'fuel_type', 'combined_l_100km']
)

prediction = model.predict(input_data)[0]

# --- 5. Display Result ---
st.subheader("📊 Predicted CO₂ Emissions")
st.metric(label="g/km", value=f"{prediction:.2f}")

# --- 6. Horizontal Emission Level Bar Chart ---
import seaborn as sns

# Define categories and ranges (customize as needed)
emission_ranges = [0, 150, 200, 250, 350]
emission_labels = ["Low", "Moderate", "High", "Very High"]
emission_colors = ["green", "yellow", "orange", "red"]

# Find the category the prediction falls into
for i in range(len(emission_ranges) - 1):
    if emission_ranges[i] <= prediction < emission_ranges[i + 1]:
        category = emission_labels[i]
        color = emission_colors[i]
        break
else:
    category = "Extremely High"
    color = "darkred"

# Draw horizontal bar
fig, ax = plt.subplots(figsize=(6, 1.5))
ax.barh(0, prediction, color=color, height=0.3)
ax.set_xlim(0, emission_ranges[-1])
ax.set_yticks([])
ax.set_xlabel("CO₂ Emissions (g/km)", fontsize=12)
ax.set_title(f"Predicted CO₂ Emissions: {prediction:.2f} g/km ({category})", fontsize=12, pad=10)

# Draw range lines and labels
for val, label in zip(emission_ranges[1:], emission_labels):
    ax.axvline(val, color='grey', linestyle='--', linewidth=0.8)
    ax.text(val - 20, 0.35, label, ha='center', fontsize=10)

fig.tight_layout()
st.pyplot(fig)











