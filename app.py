import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load merged data
df = pd.read_csv("merged_aquaimpact_dataset.csv")

# Drop 'Unnamed' columns and 'Year' if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
if 'Year' in df.columns:
    df.drop(columns=['Year'], inplace=True)

# Keep only numeric columns
df = df.select_dtypes(include=[np.number])

st.set_page_config(page_title="AquaImpact - Water Stress Predictor", layout="wide")
st.title("💧 AquaImpact: Water Stress Prediction Dashboard")

# Feature engineering
df['AbstractionPerCapita'] = df['WaterAbstracted'] / df['PopSuppliedPct']
df['EmissionsPerAbstraction'] = df['CO2Emissions'] / (df['WaterAbstracted'] + 1)

# Drop NA and select features
features = ['RenewableWater', 'WaterAbstracted', 'WastewaterGenerated', 'PopSuppliedPct', 'CO2Emissions',
            'AbstractionPerCapita', 'EmissionsPerAbstraction']
df.dropna(subset=features, inplace=True)
X = df[features]

# Manually set weights
weights = np.array([0.15, 0.20, 0.15, 0.12, 0.15, 0.12, 0.11])

# Compute raw weighted index
raw_stress = np.dot(X, weights)

# Normalize to 0–100 range
min_val, max_val = raw_stress.min(), raw_stress.max()
normalized_stress = (raw_stress - min_val) / (max_val - min_val) * 100

# Assign to dataframe
df['WaterStressIndex'] = normalized_stress


# Model training
st.header("1️⃣ Model Training and Evaluation")
X = df.drop('WaterStressIndex', axis=1)
y = df['WaterStressIndex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
percent_error = np.abs((y_test - preds) / y_test) * 100

st.success(f"✅ Model trained. Test Percent Error Range: {percent_error.min():.2f}% - {percent_error.max():.2f}%")

# Feature importance (manually overridden)
st.subheader("🔍 Feature Importance")
fig, ax = plt.subplots()
ax.barh(features, weights)
ax.set_xlabel("Importance")
ax.set_title("XGBoost Feature Importance (Manually Set)")
st.pyplot(fig)

# Future predictions
st.header("3️⃣ Predict Future Water Stress")
st.write("Use the sliders below to simulate a country's environmental profile in 2030.")

# Definitions for each metric
definitions = {
    "RenewableWater": "Total renewable freshwater resources (million m³/year)",
    "WaterAbstracted": "Total freshwater withdrawn (million m³/year)",
    "WastewaterGenerated": "Total wastewater produced (1,000 m³/day)",
    "PopSuppliedPct": "% of population with water supply access",
    "CO2Emissions": "Annual CO₂ emissions (kilotonnes)",
    "AbstractionPerCapita": "Water abstracted per % of population supplied",
    "EmissionsPerAbstraction": "CO₂ emissions per unit of water abstracted"
}

# Generate sliders only for selected features
input_data = {}
for feature in features:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    default_val = float(X[feature].mean())
    label = f"{feature} ({definitions.get(feature, '')})"
    input_data[feature] = st.slider(label, min_val, max_val, default_val)

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

# Categorize stress level
def categorize_stress(score):
    if score < 33:
        return "🟢 Low"
    elif score < 66:
        return "🟡 Medium"
    else:
        return "🔴 High"

stress_category = categorize_stress(prediction)
st.metric("🌐 Predicted Water Stress Index (2030)", f"{prediction:.2f} — {stress_category}")

st.caption(
    "The Water Stress Index is a composite score (0–100) that reflects how much pressure a country places on its freshwater resources. "
    "It accounts for factors like water withdrawal, renewable availability, wastewater generation, and emissions intensity. "
    "Higher scores indicate greater stress and unsustainable usage. "
    "This model was trained on UN environment and emissions datasets, and feature importance is based on manually set weights to ensure transparency and domain relevance."
)
