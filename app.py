import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CO2 Emissions Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Define the CO2Predictor model class
class CO2Predictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.network(x)

# Load or create mock data for demonstration
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib # To save/load scaler parameters

# Define the neural network model (must be the same as in final.py)
class CO2Predictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.network(x)

# --- Data Loading and Preprocessing (from final.py) ---
@st.cache_data
def load_data_and_preprocess(file_path="emissions.csv"):
    df = pd.read_csv(file_path)

    categorical_features = ['CompanyID', 'CompanyName', 'Sector', 'Location', 'Year']
    numerical_features = ['CH4', 'N2O', 'Electricity', 'FossilFuels', 'Renewables', 
                         'RecycledWaste', 'LandfillWaste', 'CompostedWaste']

    # Categorical encoding
    for feat in categorical_features:
        df[f'{feat}_encoded'] = df[feat].astype('category').cat.codes.astype(np.float32)

    # Create interaction features
    df['Energy_Mix'] = (df['Electricity'] / (df['FossilFuels'] + df['Renewables'] + 1e-8)).astype(np.float32)
    df['Waste_Ratio'] = (df['RecycledWaste'] / (df['LandfillWaste'] + df['CompostedWaste'] + 1e-8)).astype(np.float32)
    df['GHG_Intensity'] = ((df['CH4'] * 25 + df['N2O'] * 298) / (df['CO2'] + 1e-8)).astype(np.float32)

    features = ([f'{feat}_encoded' for feat in categorical_features] + 
               numerical_features + ['Energy_Mix', 'Waste_Ratio', 'GHG_Intensity'])
    target = 'CO2'

    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)

    # Calculate normalization parameters
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0) + 1e-8

    return df, features, X_mean, X_std, y_mean, y_std, categorical_features, numerical_features

# Load data and get preprocessing parameters
df, features, X_mean, X_std, y_mean, y_std, categorical_features, numerical_features = load_data_and_preprocess()

# Save the normalization parameters for later use in prediction
# This is crucial because the model was trained on normalized data
normalization_params = {
    'X_mean': X_mean,
    'X_std': X_std,
    'y_mean': y_mean,
    'y_std': y_std
}
joblib.dump(normalization_params, 'normalization_params.pkl')


# --- Model Loading ---
@st.cache_resource # Use st.cache_resource for models to avoid reloading
def load_model(input_size, model_path="co2_emission_best.pt"):
    model = CO2Predictor(input_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the trained model
input_size = len(features)
model = load_model(input_size)

# Load normalization parameters (if not already loaded)
try:
    normalization_params = joblib.load('normalization_params.pkl')
    X_mean = normalization_params['X_mean']
    X_std = normalization_params['X_std']
    y_mean = normalization_params['y_mean']
    y_std = normalization_params['y_std']
except FileNotFoundError:
    st.error("Normalization parameters file 'normalization_params.pkl' not found. Please ensure it's in the same directory.")
    st.stop()


# --- Streamlit App Interface ---
st.set_page_config(page_title="Carbon Footprint Predictor", layout="centered")

st.title("🌍 Carbon Footprint Prediction")
st.markdown("Enter the parameters below to predict the CO2 emissions.")

# Input fields for user
st.header("Input Parameters")

# Create unique lists for categorical features from the loaded DataFrame
company_ids = sorted(df['CompanyID'].unique().tolist())
company_names = sorted(df['CompanyName'].unique().tolist())
sectors = sorted(df['Sector'].unique().tolist())
locations = sorted(df['Location'].unique().tolist())
years = sorted(df['Year'].unique().tolist())

col1, col2 = st.columns(2)

with col1:
    company_id = st.selectbox("Company ID", company_ids)
    company_name = st.selectbox("Company Name", company_names)
    sector = st.selectbox("Sector", sectors)
    location = st.selectbox("Location", locations)
    year = st.selectbox("Year", years)

with col2:
    ch4 = st.number_input("CH4 (Methane Emissions)", min_value=0.0, value=20.0, step=0.1)
    n2o = st.number_input("N2O (Nitrous Oxide Emissions)", min_value=0.0, value=15.0, step=0.1)
    electricity = st.number_input("Electricity Consumption", min_value=0.0, value=1500.0, step=10.0)
    fossil_fuels = st.number_input("Fossil Fuels Consumption", min_value=0.0, value=3000.0, step=10.0)
    renewables = st.number_input("Renewables Consumption", min_value=0.0, value=700.0, step=10.0)
    recycled_waste = st.number_input("Recycled Waste", min_value=0.0, value=250.0, step=1.0)
    landfill_waste = st.number_input("Landfill Waste", min_value=0.0, value=600.0, step=1.0)
    composted_waste = st.number_input("Composted Waste", min_value=0.0, value=200.0, step=1.0)

# Prediction Button
if st.button("Predict CO2 Emission"):
    # Create a DataFrame for the single input
    input_data = pd.DataFrame([[
        company_id, company_name, sector, location, year,
        ch4, n2o, electricity, fossil_fuels, renewables,
        recycled_waste, landfill_waste, composted_waste
    ]], columns=['CompanyID', 'CompanyName', 'Sector', 'Location', 'Year',
                 'CH4', 'N2O', 'Electricity', 'FossilFuels', 'Renewables',
                 'RecycledWaste', 'LandfillWaste', 'CompostedWaste'])

    # Apply the same feature engineering and encoding as in training
    temp_df = df.copy() # Use a copy of the original df for categorical encoding mapping
    
    for feat in categorical_features:
        # Ensure the input categorical values are mapped to the same codes as in the training data
        # Handle unseen categories gracefully by assigning -1 or a default if necessary
        # For simplicity, we assume input categories exist in the training data
        input_data[f'{feat}_encoded'] = input_data[feat].map(
            temp_df.set_index(feat)[f'{feat}_encoded'].to_dict()
        ).astype(np.float32)

    # Interaction features for the input
    input_data['Energy_Mix'] = (input_data['Electricity'] / (input_data['FossilFuels'] + input_data['Renewables'] + 1e-8)).astype(np.float32)
    input_data['Waste_Ratio'] = (input_data['RecycledWaste'] / (input_data['LandfillWaste'] + input_data['CompostedWaste'] + 1e-8)).astype(np.float32)
    # For GHG_Intensity, we need a placeholder for CO2, as it's the target.
    # We'll use a small constant or a mean value if CO2 is not available in input_data.
    # For prediction, we might not have CO2, so we need to adjust this.
    # A safer approach is to not use the target variable in input features for prediction.
    # If 'GHG_Intensity' was truly an input feature, it would need to be provided.
    # Given its definition, it's a derived feature *from* CO2, so it shouldn't be an input for predicting CO2.
    # Let's remove 'GHG_Intensity' from the `features` list for prediction if it relies on the target.
    # Re-evaluating `final.py`: `GHG_Intensity = ((CH4 * 25 + N2O * 298) / (CO2 + 1e-8))`
    # This means GHG_Intensity *depends* on CO2. So it cannot be an input feature for predicting CO2.
    # We need to adjust the `features` list used for prediction.

    # Re-define features for prediction (excluding GHG_Intensity as it depends on CO2)
    prediction_features = ([f'{feat}_encoded' for feat in categorical_features] + 
                           numerical_features + ['Energy_Mix', 'Waste_Ratio'])
    
    # Ensure the order of columns matches the training features
    X_input = input_data[prediction_features].values.astype(np.float32)

    # Normalize the input features using the saved normalization parameters
    X_input_normalized = (X_input - X_mean[np.array([features.index(f) for f in prediction_features])]) / \
                         (X_std[np.array([features.index(f) for f in prediction_features])])

    # Convert to PyTorch tensor
    X_input_tensor = torch.tensor(X_input_normalized, dtype=torch.float32)

    # Make prediction
    model.eval()
    with torch.no_grad():
        predicted_normalized = model(X_input_tensor).numpy()
    
    # Denormalize the prediction
    predicted_co2 = predicted_normalized * y_std + y_mean

    st.subheader("Prediction Result")
    st.success(f"Predicted CO2 Emission: **{predicted_co2[0][0]:.2f}**")

st.markdown("""
---
**Note:** This application uses a pre-trained PyTorch model to predict CO2 emissions based on the provided inputs.
Ensure `emissions.csv` and `co2_emission_best.pt` are in the same directory as this script.
""")


@st.cache_data
def prepare_features(df):
    """Prepare features similar to the original model"""
    categorical_features = ['CompanyID', 'CompanyName', 'Sector', 'Location', 'Year']
    numerical_features = ['CH4', 'N2O', 'Electricity', 'FossilFuels', 'Renewables', 
                         'RecycledWaste', 'LandfillWaste', 'CompostedWaste']
    
    # Categorical encoding
    for feat in categorical_features:
        df[f'{feat}_encoded'] = df[feat].astype('category').cat.codes.astype(np.float32)
    
    # Create interaction features
    df['Energy_Mix'] = (df['Electricity'] / (df['FossilFuels'] + df['Renewables'] + 1e-8)).astype(np.float32)
    df['Waste_Ratio'] = (df['RecycledWaste'] / (df['LandfillWaste'] + df['CompostedWaste'] + 1e-8)).astype(np.float32)
    df['GHG_Intensity'] = ((df['CH4'] * 25 + df['N2O'] * 298) / (df['CO2'] + 1e-8)).astype(np.float32)
    
    features = ([f'{feat}_encoded' for feat in categorical_features] + 
               numerical_features + ['Energy_Mix', 'Waste_Ratio', 'GHG_Intensity'])
    
    return features, df[features].values.astype(np.float32)

def create_mock_model(input_size):
    """Create a mock trained model for demonstration"""
    model = CO2Predictor(input_size)
    model.eval()
    return model

def normalize_input(data, mean, std):
    """Normalize input data"""
    return (data - mean) / (std + 1e-8)

# Main app
def main():
    st.markdown('<h1 class="main-header">🌍 CO2 Emissions Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict CO2 emissions based on company environmental data using Neural Networks")
    
    # Load sample data
    df = load_sample_data()
    features, X = prepare_features(df)
    
    # Create mock normalization parameters
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    y_mean = np.mean(df['CO2'].values)
    y_std = np.std(df['CO2'].values) + 1e-8
    
    # Create mock model
    model = create_mock_model(len(features))
    
    # Sidebar for input
    st.sidebar.header("🔧 Input Parameters")
    
    # Company Information
    st.sidebar.subheader("Company Information")
    company_id = st.sidebar.selectbox("Company ID", range(1, 21), index=0)
    company_name = st.sidebar.selectbox("Company Name", [f"Company_{i}" for i in range(1, 21)])
    sector = st.sidebar.selectbox("Sector", ['Energy', 'Manufacturing', 'Technology', 'Transportation', 'Agriculture'])
    location = st.sidebar.selectbox("Location", ['USA', 'Europe', 'Asia', 'South America', 'Africa'])
    year = st.sidebar.selectbox("Year", [2018, 2019, 2020, 2021, 2022, 2023, 2024])
    
    # Environmental Data
    st.sidebar.subheader("Environmental Data")
    ch4 = st.sidebar.slider("CH4 Emissions (tons)", 0.0, 200.0, 50.0, 1.0)
    n2o = st.sidebar.slider("N2O Emissions (tons)", 0.0, 100.0, 30.0, 1.0)
    electricity = st.sidebar.slider("Electricity Usage (MWh)", 0.0, 3000.0, 1000.0, 10.0)
    fossil_fuels = st.sidebar.slider("Fossil Fuels Usage (units)", 0.0, 2000.0, 800.0, 10.0)
    renewables = st.sidebar.slider("Renewables Usage (units)", 0.0, 1000.0, 200.0, 10.0)
    
    # Waste Management
    st.sidebar.subheader("Waste Management")
    recycled_waste = st.sidebar.slider("Recycled Waste (tons)", 0.0, 500.0, 150.0, 5.0)
    landfill_waste = st.sidebar.slider("Landfill Waste (tons)", 0.0, 800.0, 300.0, 5.0)
    composted_waste = st.sidebar.slider("Composted Waste (tons)", 0.0, 300.0, 100.0, 5.0)
    
    # Create input data
    input_data = {
        'CompanyID': company_id,
        'CompanyName': company_name,
        'Sector': sector,
        'Location': location,
        'Year': year,
        'CH4': ch4,
        'N2O': n2o,
        'Electricity': electricity,
        'FossilFuels': fossil_fuels,
        'Renewables': renewables,
        'RecycledWaste': recycled_waste,
        'LandfillWaste': landfill_waste,
        'CompostedWaste': composted_waste,
        'CO2': 0  # Placeholder
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 Predict CO2 Emissions", key="predict_button"):
            # Create DataFrame for input
            input_df = pd.DataFrame([input_data])
            
            # Prepare features
            _, input_features = prepare_features(input_df)
            
            # Normalize input
            normalized_input = normalize_input(input_features, X_mean, X_std)
            
            # Convert to tensor
            input_tensor = torch.tensor(normalized_input, dtype=torch.float32)
            
            # Make prediction
            with torch.no_grad():
                model.eval()
                normalized_prediction = model(input_tensor).numpy()
                
                # Denormalize prediction
                prediction = normalized_prediction * y_std + y_mean
                predicted_co2 = float(prediction[0][0])
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-card">
                Predicted CO2 Emissions: {predicted_co2:.2f} tons
            </div>
            """, unsafe_allow_html=True)
            
            # Additional metrics
            st.subheader("📊 Environmental Impact Analysis")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                ghg_intensity = (ch4 * 25 + n2o * 298) / (predicted_co2 + 1e-8)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>GHG Intensity</h3>
                    <h2>{ghg_intensity:.2f}</h2>
                    <p>CO2 equivalent ratio</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                energy_mix = electricity / (fossil_fuels + renewables + 1e-8)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Energy Mix</h3>
                    <h2>{energy_mix:.2f}</h2>
                    <p>Electricity/Total Energy</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                waste_ratio = recycled_waste / (landfill_waste + composted_waste + 1e-8)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Waste Ratio</h3>
                    <h2>{waste_ratio:.2f}</h2>
                    <p>Recycled/Total Waste</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization
            st.subheader("📈 Emission Breakdown")
            
            # Create pie chart for emissions breakdown
            emissions_data = {
                'Source': ['Direct CO2', 'CH4 (CO2 eq)', 'N2O (CO2 eq)'],
                'Emissions': [predicted_co2, ch4 * 25, n2o * 298]
            }
            
            fig_pie = px.pie(
                values=emissions_data['Emissions'],
                names=emissions_data['Source'],
                title="Greenhouse Gas Emissions Breakdown",
                color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Energy usage chart
            energy_data = {
                'Source': ['Electricity', 'Fossil Fuels', 'Renewables'],
                'Usage': [electricity, fossil_fuels, renewables]
            }
            
            fig_bar = px.bar(
                x=energy_data['Source'],
                y=energy_data['Usage'],
                title="Energy Usage by Source",
                color=energy_data['Source'],
                color_discrete_sequence=['#feca57', '#ff9ff3', '#54a0ff']
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Model Information")
        st.info("""
        **Model Architecture:**
        - Neural Network with 4 layers
        - BatchNorm + Dropout regularization
        - Trained on company emissions data
        
        **Features Used:**
        - Company metadata
        - Greenhouse gas emissions (CH4, N2O)
        - Energy consumption patterns
        - Waste management data
        - Engineered interaction features
        """)
        
        st.markdown("### 📋 Emission Categories")
        st.markdown("""
        **Low Impact:** < 1000 tons CO2
        
        **Medium Impact:** 1000-3000 tons CO2
        
        **High Impact:** > 3000 tons CO2
        """)
        
        st.markdown("### 🌱 Sustainability Tips")
        st.success("""
        💡 **Reduce Emissions:**
        - Increase renewable energy usage
        - Improve waste recycling rates
        - Optimize energy efficiency
        - Monitor CH4 and N2O emissions
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("### 📊 Sample Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Companies", len(df['CompanyName'].unique()))
    with col2:
        st.metric("Avg CO2 Emissions", f"{df['CO2'].mean():.1f} tons")
    with col3:
        st.metric("Data Points", len(df))
    with col4:
        st.metric("Sectors Covered", len(df['Sector'].unique()))
    
    # Show sample data
    with st.expander("View Sample Training Data"):
        st.dataframe(df.head(10))

if __name__ == "__main__":
    main()