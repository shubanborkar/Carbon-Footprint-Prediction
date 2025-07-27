# 🌍 Carbon Footprint Prediction

This repository hosts a machine learning project aimed at predicting CO2 emissions (carbon footprints) based on various company-specific environmental factors. The project utilizes a deep learning model built with PyTorch and features an interactive web application developed with Streamlit for easy prediction and visualization.

## ✨ Features

* **CO2 Emission Prediction:** Predicts CO2 emissions based on a range of input parameters including company details, greenhouse gas emissions (CH4, N2O), energy consumption (Electricity, Fossil Fuels, Renewables), and waste management data.
* **Interactive Streamlit UI:** A user-friendly web interface for inputting parameters and viewing real-time predictions.
* **Cascading Input Filters:** Smart dropdowns for `Company ID`, `Company Name`, `Sector`, `Location`, and `Year` that dynamically filter options based on previous selections, ensuring coherent inputs.
* **Future Year Prediction:** Ability to select years beyond the historical dataset (up to 5 years into the future) to facilitate forward-looking scenario analysis.
* **Dynamic Emission Categories:** The predicted CO2 emission result tile is color-coded (Low, Medium, High Impact) based on the statistical distribution (percentiles) of CO2 values in your training data, providing immediate context.
* **Environmental Impact Analysis:** Displays calculated metrics like GHG Intensity, Energy Mix, and Waste Ratio based on inputs and predicted CO2.
* **Emission Breakdown Visualizations:** Interactive charts (using Plotly) to visualize the breakdown of greenhouse gas emissions (CO2, CH4, N2O equivalents) and energy usage.
* **Dark Theme UI:** A sleek and modern dark user interface with enhanced contrast for comfortable viewing.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.8+
* Git (for cloning the repository)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/shubanborkar/Carbon-Footprint-Prediction.git](https://github.com/shubanborkar/Carbon-Footprint-Prediction.git)
    cd Carbon-Footprint-Prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    (If `pip` command is not found, try `pip3`. If `requirements.txt` is missing, please create it with the content provided in the "Requirements File" section below.)

### Dataset

The project uses `emissions.csv`, which should be placed in the root directory of the cloned repository. This dataset contains historical environmental data for various companies.

### Training the Model

Before running the Streamlit application, you **must** train the PyTorch model and save its weights along with the normalization parameters.

1.  **Run the training script:**
    ```bash
    python Predictor.py
    # or if `python` points to Python 2:
    python3 Predictor.py
    ```
    This script will:
    * Load and preprocess `emissions.csv`.
    * Train the `CO2Predictor` neural network.
    * Save the trained model as `co2_emission_best.pt`.
    * Save the normalization parameters as `normalization_params.pkl` (essential for making correct predictions in the app).

2.  **Verify generated files:**
    After running `Predictor.py`, ensure that `co2_emission_best.pt` and `normalization_params.pkl` files are created in the root directory of your project.

### Running the Streamlit App

Once the model and parameters are generated, you can launch the Streamlit app:

1.  **Ensure your virtual environment is active.**
2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This command will open the Streamlit app in your web browser.


## 📄 Requirements File (`requirements.txt`)

Ensure your `requirements.txt` file contains the following libraries:
streamlit
pandas
numpy
torch
joblib
plotly

## 💡 Usage

* Select various company and environmental parameters using the dropdowns and number inputs.
* The dropdowns for `Company ID`, `Sector`, `Location`, and `Year` will dynamically update based on your `Company Name` selection and historical data, including future years for prediction.
* Click the "🔮 Predict CO2 Emission" button to get the predicted CO2 value and see detailed analysis and visualizations.
* The prediction result tile will change color based on whether the emission is "Low," "Medium," or "High" impact, relative to your dataset's distribution.

## 🤝 Contributing

Feel free to fork this repository, open issues, or submit pull requests for any improvements or bug fixes.

## Contact

**Shuban Borkar**  
Email: [shubanborkar@gmail.com](mailto:shubanborkar@gmail.com)  
LinkedIn: [shuban-borkar](https://www.linkedin.com/in/shuban-borkar)  
GitHub: [shubanborkar](https://github.com/shubanborkar)

---