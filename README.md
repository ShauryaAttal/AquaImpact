# AquaImpact

**AquaImpact** is an interactive machine-learning dashboard that predicts a country's water stress based on freshwater availability, water withdrawal, wastewater generation, population access, and carbon emissions.

Built with **Python, Streamlit, and XGBoost**, AquaImpact transforms environmental data into a transparent **Water Stress Index from 0 to 100**, helping users explore how different environmental conditions may affect future water sustainability.

## Overview

Freshwater stress is influenced by more than the amount of water a country has available. Population access, withdrawal rates, wastewater production, and emissions intensity can all contribute to pressure on water resources.

AquaImpact combines these factors into a single, interpretable score and uses an XGBoost regression model to estimate future water stress.

The dashboard allows users to:

* Train and evaluate a water-stress prediction model
* Explore the relative importance of environmental indicators
* Adjust environmental variables using interactive sliders
* Simulate a country's environmental profile for 2030
* Receive a predicted Water Stress Index and risk category

## Water Stress Categories

| Index      | Category  | Meaning                                                      |
| ---------- | --------- | ------------------------------------------------------------ |
| `0–32.99`  | 🟢 Low    | Relatively limited pressure on freshwater resources          |
| `33–65.99` | 🟡 Medium | Moderate pressure and possible sustainability concerns       |
| `66–100`   | 🔴 High   | Significant pressure and potentially unsustainable water use |

Higher scores indicate greater stress on freshwater resources.

## Features

### Interactive Prediction Dashboard

Users can adjust environmental variables and immediately generate a predicted water-stress score.

### Composite Water Stress Index

AquaImpact creates a normalized index between 0 and 100 using weighted environmental indicators.

### Machine-Learning Prediction

An `XGBRegressor` model learns patterns from the environmental dataset and predicts the Water Stress Index for new scenarios.

### Feature Engineering

The application creates additional indicators that better represent the relationship between water use, population access, and emissions:

```text
AbstractionPerCapita = WaterAbstracted / PopSuppliedPct
```

```text
EmissionsPerAbstraction = CO2Emissions / (WaterAbstracted + 1)
```

### Transparent Feature Weights

The dashboard displays the weights used to construct the Water Stress Index, making the scoring process easier to understand and evaluate.

## Environmental Indicators

| Feature                   | Description                                                            |
| ------------------------- | ---------------------------------------------------------------------- |
| `RenewableWater`          | Total renewable freshwater resources in million cubic meters per year  |
| `WaterAbstracted`         | Total freshwater withdrawn in million cubic meters per year            |
| `WastewaterGenerated`     | Total wastewater produced in thousands of cubic meters per day         |
| `PopSuppliedPct`          | Percentage of the population with access to a water supply             |
| `CO2Emissions`            | Annual carbon dioxide emissions in kilotonnes                          |
| `AbstractionPerCapita`    | Water abstracted relative to the percentage of the population supplied |
| `EmissionsPerAbstraction` | Carbon dioxide emissions per unit of water abstracted                  |

## Technology Stack

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **XGBoost**
* **scikit-learn**
* **Matplotlib**

## Project Structure

```text
AquaImpact/
├── app.py
├── merged_aquaimpact_dataset.csv
├── requirements.txt
└── README.md
```

## How It Works

### 1. Data Preparation

The application loads the merged environmental dataset, removes unnecessary columns, and retains numerical data.

### 2. Feature Engineering

Two derived variables are calculated to capture relationships that are not represented by the original metrics alone:

* Water abstraction relative to population access
* Carbon emissions relative to water abstraction

### 3. Index Construction

A weighted combination of the selected environmental indicators is calculated:

```python
weights = np.array([0.15, 0.20, 0.15, 0.12, 0.15, 0.12, 0.11])
```

The resulting values are normalized to a scale from 0 to 100 to create the Water Stress Index.

### 4. Model Training

The dataset is split into training and testing sets using an 80/20 split. An XGBoost regression model is then trained to predict the Water Stress Index.

```python
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1
)
```

### 5. Future Scenario Simulation

Users adjust sliders representing a hypothetical country's environmental profile. The trained model processes those values and predicts a Water Stress Index for 2030.

## Example Use Cases

AquaImpact can be used to:

* Explore relationships between freshwater resources and water withdrawal
* Demonstrate environmental machine-learning applications
* Compare hypothetical sustainability scenarios
* Study the effects of wastewater and emissions on water stress
* Support classroom discussions about environmental data science
* Prototype more advanced climate-risk and resource-planning tools

## Model Limitations

AquaImpact is an educational and exploratory project rather than an official water-risk assessment system.

Important limitations include:

* The target Water Stress Index is constructed using manually selected weights
* The model learns to reproduce this constructed index rather than a directly observed outcome
* Predictions depend on the quality and coverage of the underlying dataset
* Country-specific geographic, political, economic, and infrastructure conditions are not fully represented
* Slider values are limited to the ranges found in the training data
* Predictions should not be used as the sole basis for public policy or resource-management decisions

## Potential Improvements

Future versions of AquaImpact could include:

* Geographic visualizations and interactive world maps
* Country selection and comparison
* Historical water-stress trends
* Additional climate and socioeconomic indicators
* Cross-validation and expanded model evaluation
* Explainable-AI tools such as SHAP values
* Data validation and outlier handling
* Model persistence instead of retraining at application startup
* Integration with regularly updated public environmental datasets
* Scenario comparisons across multiple years
* Downloadable prediction reports

## Data

The application uses a merged dataset containing water-resource, population-access, wastewater, and emissions indicators.

The dashboard describes the underlying information as originating from United Nations environmental and emissions datasets. Before using the project for research or publication, the specific source organizations, dataset names, years, licenses, and preprocessing steps should be documented in this section.

## Disclaimer

AquaImpact is intended for educational, research, and demonstration purposes. Its predictions are estimates produced by a simplified machine-learning model and should not be interpreted as official environmental assessments.

## Author

**Shaurya Attal**

* GitHub: [@ShauryaAttal](https://github.com/ShauryaAttal)
