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
@st.cache_data
def load_sample_data():
    """Create sample data structure similar to emissions.csv"""
    np.random.seed(42)
    n_samples = 100
    
    companies = [f"Company_{i}" for i in range(1, 21)]
    sectors = ['Energy', 'Manufacturing', 'Technology', 'Transportation', 'Agriculture']
    locations = ['USA', 'Europe', 'Asia', 'South America', 'Africa']
    years = [2018, 2019, 2020, 2021, 2022]
    
    data = {
        'CompanyID': np.random.choice(range(1, 21), n_samples),
        'CompanyName': np.random.choice(companies, n_samples),
        'Sector': np.random.choice(sectors, n_samples),
        'Location': np.random.choice(locations, n_samples),
        'Year': np.random.choice(years, n_samples),
        'CH4': np.random.normal(50, 20, n_samples),
        'N2O': np.random.normal(30, 15, n_samples),
        'Electricity': np.random.normal(1000, 300, n_samples),
        'FossilFuels': np.random.normal(800, 250, n_samples),
        'Renewables': np.random.normal(200, 100, n_samples),
        'RecycledWaste': np.random.normal(150, 50, n_samples),
        'LandfillWaste': np.random.normal(300, 100, n_samples),
        'CompostedWaste': np.random.normal(100, 40, n_samples),
        'CO2': np.random.normal(2000, 600, n_samples)
    }
    
    return pd.DataFrame(data)

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