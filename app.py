import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib # To save/load scaler parameters
import plotly.express as px # For visualizations
import warnings
warnings.filterwarnings('ignore')

# --- Page configuration and Custom CSS ---
st.set_page_config(
    page_title="CO2 Emissions Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #333; /* Darker text for better readability */
    }

    .main-header {
        font-size: 3.2rem; /* Slightly larger */
        font-weight: 700; /* Bolder */
        text-align: center;
        color: #1E88E5; /* A fresh blue */
        margin-bottom: 2.5rem;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.15); /* More pronounced shadow */
    }
    
    .stApp {
        background-color: #f0f2f6; /* Light grey background for the app */
    }

    .metric-card {
        background: linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%); /* Blue gradient */
        padding: 1.2rem;
        border-radius: 12px; /* Slightly less rounded */
        color: white;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15); /* Softer, larger shadow */
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px); /* Lift effect on hover */
    }
    .metric-card h3 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .metric-card p {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    .prediction-card {
        background: linear-gradient(135deg, #81c784 0%, #4caf50 100%); /* Green gradient */
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.8rem; /* Larger font for prediction */
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(0,0,0,0.25); /* More prominent shadow */
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #2E7D32); /* Darker green gradient */
        color: white;
        border-radius: 30px; /* More rounded */
        border: none;
        padding: 0.8rem 2.5rem; /* More padding */
        font-weight: 600;
        font-size: 1.2rem; /* Slightly larger font */
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-3px); /* More pronounced lift */
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        background: linear-gradient(90deg, #45a049, #28602A); /* Slightly darker on hover */
    }

    .sidebar .stSelectbox, .sidebar .stNumberInput, .sidebar .stSlider {
        margin-bottom: 1.2rem; /* More spacing in sidebar */
        background-color: #ffffff; /* White background for inputs */
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    h1, h2, h3, h4, h5, h6 {
        color: #424242; /* Dark grey for headers */
        font-weight: 600;
    }
    .stInfo, .stSuccess {
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.95rem;
    }
    .stInfo {
        background-color: #e3f2fd; /* Light blue for info */
        color: #1e88e5;
        border-left: 5px solid #1e88e5;
    }
    .stSuccess {
        background-color: #e8f5e9; /* Light green for success */
        color: #4caf50;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# --- Define the Neural Network Model ---
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

# --- Data Loading and Preprocessing for categorical mapping and unique values ---
@st.cache_data
def load_base_data(file_path="emissions.csv"):
    """Loads the raw emissions data for categorical feature mapping."""
    df = pd.read_csv(file_path)
    return df

# Define feature lists (MUST match Predictor.ipynb's final feature list)
categorical_features = ['CompanyID', 'CompanyName', 'Sector', 'Location', 'Year']
numerical_features = ['CH4', 'N2O', 'Electricity', 'FossilFuels', 'Renewables', 
                     'RecycledWaste', 'LandfillWaste', 'CompostedWaste']

# The 'features' list used for training and prediction input (NO GHG_Intensity here)
MODEL_FEATURES = ([f'{feat}_encoded' for feat in categorical_features] + 
                  numerical_features + ['Energy_Mix', 'Waste_Ratio'])

# --- Model and Normalization Parameters Loading ---
@st.cache_resource
def load_model_and_params(model_path="co2_emission_best.pt", params_path="normalization_params.pkl"):
    """Loads the trained model and normalization parameters."""
    try:
        # Load normalization parameters
        normalization_params = joblib.load(params_path)
        X_mean = normalization_params['X_mean']
        X_std = normalization_params['X_std']
        y_mean = normalization_params['y_mean']
        y_std = normalization_params['y_std']
        
        # Ensure the features list from params matches the app's expectation
        # This is a good sanity check
        if 'features' in normalization_params and normalization_params['features'] != MODEL_FEATURES:
            st.error("Feature list mismatch between saved parameters and app's model features. Please retrain your model with the correct features.")
            st.stop()

        # Load the trained model
        # Input size for the model must match the number of features used during training
        input_size = len(MODEL_FEATURES) 
        model = CO2Predictor(input_size)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model, X_mean, X_std, y_mean, y_std

    except FileNotFoundError:
        st.error(f"Required files not found: '{model_path}' and/or '{params_path}'. "
                 "Please ensure you have run 'Predictor.ipynb' (or .py) locally to train the model "
                 "and save these files, then commit them to your GitHub repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or normalization parameters: {e}")
        st.stop()

# --- Main Application Logic ---
def run_app():
    # Load base data for categorical feature mapping
    df_base = load_base_data()

    # Load model and parameters globally
    model, X_mean, X_std, y_mean, y_std = load_model_and_params()

    st.markdown('<h1 class="main-header">🌍 Carbon Footprint Prediction</h1>', unsafe_allow_html=True)
    st.markdown("### Predict CO2 emissions based on company environmental data using Neural Networks")

    st.header("Input Parameters")

    # Create unique lists for categorical features from the loaded DataFrame
    company_ids = sorted(df_base['CompanyID'].unique().tolist())
    company_names = sorted(df_base['CompanyName'].unique().tolist())
    
    # --- Dependent Dropdowns Logic ---
    col1, col2 = st.columns(2)

    with col1:
        company_id = st.selectbox("Company ID", company_ids, key="company_id_select")
        
        # Filter company names based on selected Company ID
        filtered_company_names = sorted(df_base[df_base['CompanyID'] == company_id]['CompanyName'].unique().tolist())
        company_name = st.selectbox("Company Name", filtered_company_names, key="company_name_select")
        
        # Filter sectors based on selected Company Name
        filtered_sectors = sorted(df_base[df_base['CompanyName'] == company_name]['Sector'].unique().tolist())
        sector = st.selectbox("Sector", filtered_sectors, key="sector_select")
        
        # Filter locations based on selected Company Name and Sector
        filtered_locations = sorted(df_base[(df_base['CompanyName'] == company_name) & 
                                            (df_base['Sector'] == sector)]['Location'].unique().tolist())
        location = st.selectbox("Location", filtered_locations, key="location_select")
        
        # Filter years based on selected Company Name, Sector, and Location
        filtered_years = sorted(df_base[(df_base['CompanyName'] == company_name) & 
                                        (df_base['Sector'] == sector) &
                                        (df_base['Location'] == location)]['Year'].unique().tolist())
        year = st.selectbox("Year", filtered_years, key="year_select")

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
    if st.button("🔮 Predict CO2 Emission", key="predict_button"):
        # Create a DataFrame for the single input
        input_data = pd.DataFrame([[
            company_id, company_name, sector, location, year,
            ch4, n2o, electricity, fossil_fuels, renewables,
            recycled_waste, landfill_waste, composted_waste
        ]], columns=['CompanyID', 'CompanyName', 'Sector', 'Location', 'Year',
                     'CH4', 'N2O', 'Electricity', 'FossilFuels', 'Renewables',
                     'RecycledWaste', 'LandfillWaste', 'CompostedWaste'])

        # Apply the same feature engineering and encoding as in training
        # Create a temporary DataFrame to map categorical features using the original df_base's categories
        temp_df_for_encoding = pd.concat([df_base[categorical_features], input_data[categorical_features]], ignore_index=True)
        for feat in categorical_features:
            temp_df_for_encoding[f'{feat}_encoded'] = temp_df_for_encoding[feat].astype('category').cat.codes
        
        # Extract the encoded values for the input_data row
        for feat in categorical_features:
            input_data[f'{feat}_encoded'] = temp_df_for_encoding.iloc[-1][f'{feat}_encoded']

        # Interaction features for the input
        input_data['Energy_Mix'] = (input_data['Electricity'] / (input_data['FossilFuels'] + input_data['Renewables'] + 1e-8)).astype(np.float32)
        input_data['Waste_Ratio'] = (input_data['RecycledWaste'] / (input_data['LandfillWaste'] + input_data['CompostedWaste'] + 1e-8)).astype(np.float32)
        
        # Ensure the order of columns matches the MODEL_FEATURES used during training
        X_input = input_data[MODEL_FEATURES].values.astype(np.float32)

        # Normalize the input features using the saved normalization parameters
        X_input_normalized = (X_input - X_mean) / X_std

        # Convert to PyTorch tensor
        X_input_tensor = torch.tensor(X_input_normalized, dtype=torch.float32)

        # Make prediction
        model.eval()
        with torch.no_grad():
            predicted_normalized = model(X_input_tensor).numpy()
        
        # Denormalize the prediction
        predicted_co2 = predicted_normalized * y_std + y_mean

        st.subheader("Prediction Result")
        st.markdown(f"""
        <div class="prediction-card">
            Predicted CO2 Emissions: {predicted_co2[0][0]:.2f} tons
        </div>
        """, unsafe_allow_html=True)

        # Additional metrics and visualizations
        st.subheader("📊 Environmental Impact Analysis")
                
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            # GHG_Intensity calculation for display (not used as model input)
            ghg_intensity = (ch4 * 25 + n2o * 298) / (predicted_co2[0][0] + 1e-8)
            st.markdown(f"""
            <div class="metric-card">
                <h3>GHG Intensity</h3>
                <h2>{ghg_intensity:.2f}</h2>
                <p>CO2 equivalent ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            energy_mix_display = electricity / (fossil_fuels + renewables + 1e-8)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Energy Mix</h3>
                <h2>{energy_mix_display:.2f}</h2>
                <p>Electricity/Total Energy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            waste_ratio_display = recycled_waste / (landfill_waste + composted_waste + 1e-8)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Waste Ratio</h3>
                <h2>{waste_ratio_display:.2f}</h2>
                <p>Recycled/Total Waste</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization
        st.subheader("📈 Emission Breakdown")
        
        # Create pie chart for emissions breakdown
        emissions_data = {
            'Source': ['Direct CO2', 'CH4 (CO2 eq)', 'N2O (CO2 eq)'],
            'Emissions': [predicted_co2[0][0], ch4 * 25, n2o * 298]
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


    # --- Information Section (Right Column) ---
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
    - Engineered interaction features (Energy_Mix, Waste_Ratio)
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

    col_footer1, col_footer2, col_footer3, col_footer4 = st.columns(4)
    with col_footer1:
        st.metric("Total Companies", len(df_base['CompanyName'].unique()))
    with col_footer2:
        st.metric("Avg CO2 Emissions", f"{df_base['CO2'].mean():.1f} tons")
    with col_footer3:
        st.metric("Data Points", len(df_base))
    with col_footer4:
        st.metric("Sectors Covered", len(df_base['Sector'].unique()))

    # Show sample data
    with st.expander("View Sample Training Data"):
        st.dataframe(df_base.head(10))

if __name__ == "__main__":
    run_app()
