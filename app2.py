import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
from datetime import timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def compute_geometry(Mfuel, Ffuel, Fstruct, Porosity, Rflat, Alpha):
    Pi = 4.0 * np.arctan(1.0)
    Mcorium = Mfuel * Ffuel * (1 + Fstruct)
    Vdebris = Mcorium / 7300.0 / (1.0 - Porosity)
    Rcav, Hcav, Hwater = 4.4, 5.0, 5.0
    Rlow, Hlow = 3.35, 1.0
    alpha = Alpha * Pi / 180.0
    tana = np.tan(alpha)

    V1 = Pi / 3.0 * tana * (Rlow**3 - Rflat**3)
    V2 = V1 + Pi * Hlow * Rlow**2
    V3 = V2 - V1 + Pi / 3.0 * tana * (Rcav**3 - Rflat**3)

    if Vdebris < V1:
        R = (Rflat**3 + 3.0 * Vdebris / (Pi * tana))**(1.0/3.0)
        H = tana * (R - Rflat)
        rDebris = [0.0, R, Rflat, 0.0]
        zDebris = [0.0, 0.0, H, H]
    elif Vdebris < V2:
        Vcyl = Vdebris - V1
        Hcyl = Vcyl / (Pi * Rlow**2)
        Hcon = tana * (Rlow - Rflat)
        H = Hcyl + Hcon
        rDebris = [0.0, Rlow, Rlow, Rflat, 0.0]
        zDebris = [0.0, 0.0, Hcyl, H, H]
    elif Vdebris < V3:
        Vcon = Vdebris - Pi * Hlow * Rlow**2
        R = (Rflat**3 + 3.0 * Vcon / (Pi * tana))**(1.0/3.0)
        Hcon = tana * (R - Rflat)
        H = Hlow + Hcon
        rDebris = [0.0, Rlow, Rlow, R, Rflat, 0.0]
        zDebris = [0.0, 0.0, Hlow, Hlow, H, H]
    else:
        Vcyl = Vdebris - V3
        Hcyl = Vcyl / (Pi * Rcav**2) + Hlow
        Hcon = tana * (Rcav - Rflat)
        H = Hcyl + Hcon
        rDebris = [0.0, Rlow, Rlow, Rcav, Rcav, Rflat, 0.0]
        zDebris = [0.0, 0.0, Hlow, Hlow, Hcyl, H, H]

    return rDebris, zDebris

def plot_geometry(rDebris, zDebris):
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Plot the debris shape
    ax.fill(rDebris, zDebris, color='red', alpha=0.5, edgecolor='black', label='Schüttbett')
    
    # Define the block region (example: between r = 1.0 and r = 2.0)
    block_start = 3.35 #['3.35', ' 4.4', ' 4.4', ' 3.35']
    block_end = 4.4
    ax.fill_betweenx(
        [0, 1], block_start, block_end, color='grey', alpha=0.3, label='Block'
    )

    # Set plot limits
    ax.set_xlim(0, 4.4)
    ax.set_ylim(0, 5)
    
    # Labels and grid
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True)

    # Adding a legend
    ax.legend()

    return fig


# Suppress the feature name warning
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

@st.cache_resource
def load_models():
    return [tf.keras.models.load_model(f"Model_{i}.keras") for i in range(6) if os.path.exists(f"Model_{i}.keras")]

def load_classifier():
    # Load the saved model
    return joblib.load("Voting Classifier.pkl")


@st.cache_resource
def load_scaler_X():
    return joblib.load("MinMaxScaler_X.pkl")

@st.cache_resource
def load_scaler_y():
    return joblib.load("MinMaxScaler_y.pkl")

models = load_models()
scaler_X = load_scaler_X()
scaler_y = load_scaler_y()

# Use feature names directly from the scaler
param_names = scaler_X.feature_names_in_.tolist()

# Manually define descriptions and limits
descriptions = [
    "Systemdruck in MPa", "Verhältnis der Masse des aus dem RDB verlagerten Brennstoffs zur Gesamtmasse des Brennstoffs",
    "Porosität der Schüttung", "Mittlerer Partikeldurchmesser in mm", "Böschungswinkel der Schüttung in Grad", 
    "Oberer Radius des Kegelstumpfs in m", "Anfangstemperatur der trockenen Schüttung in K", 
    "Verhältnis der Nachzerfallsleistung zur thermischen Leistung im Nennalbetrieb in %", 
    "Verhältnis der Masse verlagerten Hüllrohr- und Strukturmaterials zur Masse des aus dem RDB verlagerten Brennstoffs"
]

min_vals = [0.11, 0.5, 0.25, 1.0, 15.0, 0.0, 400.0, 0.3, 0.25]
max_vals = [0.5, 1.0, 0.5, 5.0, 45.0, 2.0, 1700.0, 1.0, 2.0]

# Setup layout with two columns
col1, col2 = st.columns([1, 2])  # Left column (1 part) and right column (2 parts)

# Left column for parameter sliders
with col1:
    # add picture
    st.image("Logo_long.png", use_container_width=True)
    
    st.title("Schüttbett-KI: Vorhersage des Quenchverhaltens")

    # Build sliders with manually defined parameters
    user_inputs_scaled = []
    for i in range(len(param_names)):
        # Show slider in real-world units
        val = st.slider(
            label=f"{descriptions[i]} (`{param_names[i]}`)",
            min_value=float(min_vals[i]),
            max_value=float(max_vals[i]),
            value=(min_vals[i] + max_vals[i]) / 2,
            key=f"slider_{i}"
        )

        # Manually scale to [0, 1]
        scaled_val = (val - min_vals[i]) / (max_vals[i] - min_vals[i])
        user_inputs_scaled.append(scaled_val)

        # Optionally show scaled value
        st.caption(f"Skalierter Wert: {scaled_val:.2f}")

# Right column for geometry plot and prediction output
with col2:
    # Compute geometry based on user inputs
    Ffuel = st.session_state["slider_1"]
    Porosity = st.session_state["slider_2"]
    Alpha = st.session_state["slider_4"]
    Rflat = st.session_state["slider_5"]
    Fstruct = st.session_state["slider_8"]

    Mfuel = 136000.0  # Fixed value for fuel mass, or add a slider if needed
    rDebris, zDebris = compute_geometry(Mfuel, Ffuel, Fstruct, Porosity, Rflat, Alpha)

    # Plot geometry
    fig = plot_geometry(rDebris, zDebris)
    st.markdown("### Geometrie der Schüttung")
    st.pyplot(fig)
    
    # ------------------------------------------------------------------------------
    # Load classifier model
    classifier = load_classifier()
    print(classifier.predict([user_inputs_scaled]))
    if classifier.predict([user_inputs_scaled]) == 1:
        st.markdown("### ✅ Kühlung möglich:")

        # Scale input and make predictions
        input_array = np.array(user_inputs_scaled).reshape(1, -1)
        predictions = [model.predict(input_array)[0][0] for model in models]
        avg = np.mean(predictions)
        std = np.std(predictions)
        
        # Inverse transform the predictions
        scaled_avg = scaler_y.inverse_transform([[avg]])[0][0]
        scaled_std = scaler_y.inverse_transform([[std]])[0][0]

        # Check for valid values before using timedelta
        if isinstance(scaled_avg, (int, float)) and scaled_avg >= 0:
            predicted_duration = timedelta(seconds=scaled_avg)
        else:
            st.markdown("### ⚠️ Invalid prediction for average quench time.")
            predicted_duration = timedelta(seconds=0)  # Default value for error case

        if isinstance(scaled_std, (int, float)) and scaled_std >= 0:
            uncertainty_duration = timedelta(seconds=scaled_std)
        else:
            st.markdown("### ⚠️ Invalid prediction for uncertainty in quench time.")
            uncertainty_duration = timedelta(seconds=0)  # Default value for error case

        # Format timedelta for display
        def format_timedelta(td):
            total_seconds = int(td.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours}h {minutes}m {seconds}s"

        st.markdown(f"## Vorhersage der Quenchzeit: {format_timedelta(predicted_duration)}")
        st.caption(f"Vorhersage der Quenchzeit: {scaled_avg:.2f} Sekunden")
        st.markdown(f"### Unsicherheit: {format_timedelta(uncertainty_duration)}")

        # Plot ensemble predictions
        predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=predictions, mode='lines+markers', name='Vorhersage'))
        fig.update_layout(title="Ensemble Predictions [s]",
                        xaxis_title="Modelle",
                        yaxis_title="Vorhersage der Quenchzeit [s]",)

        st.plotly_chart(fig)
    else:
        st.markdown("### 🚫 Kühlung nicht möglich, Schüttbett schmilzt wieder")
