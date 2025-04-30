import streamlit as st
import numpy as np
import tensorflow as tf
tf.compat.v1.reset_default_graph() 
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
    
    # Set the background color to blue
    ax.set_facecolor('lightblue')
    

    # Adding the "Wasser" background as a label in the legend
    ax.fill_betweenx([0, 5], 0, 4.4, color='lightblue', alpha=0.5, label='Wasser')
        # Plot the debris shape
    ax.fill(rDebris, zDebris, color='red', alpha=0.5, edgecolor='black', label='Schüttbett')
    
    # Define the block region (example: between r = 1.0 and r = 2.0)
    block_start = 3.35  # ['3.35', ' 4.4', ' 4.4', ' 3.35']
    block_end = 4.4
    ax.fill_betweenx(
        [0, 1], block_start, block_end, color='grey', alpha=0.9, label='Block'
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
    return [tf.keras.models.load_model(f"model_{i}.keras") for i in range(6) if os.path.exists(f"model_{i}.keras")]

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
    "Systemdruck [MPa]",
    "Verhältnis der Masse des aus dem RDB verlagerten Brennstoffs zur Gesamtmasse des Brennstoffs [-]",
    "Porosität der Schüttung [-]",
    "Mittlerer Partikeldurchmesser [mm]",
    "Böschungswinkel der Schüttung [°]",
    "Oberer Radius des Kegelstumpfs [m]",
    "Anfangstemperatur der trockenen Schüttung [K]",
    "Verhältnis der Nachzerfallsleistung zur thermischen Leistung im Nennalbetrieb [%]",
    "Verhältnis der Masse verlagerten Hüllrohr- und Strukturmaterials zur Masse des aus dem RDB verlagerten Brennstoffs [-]"
]

min_vals = [0.11, 0.5, 0.25, 1.0, 15.0, 0.0, 400.0, 0.3, 0.25]
max_vals = [0.5, 1.0, 0.5, 5.0, 45.0, 2.0, 1700.0, 1.0, 2.0]

#Hedder 
st.markdown("---")
# add picture
st.image("Logo_long.png", width=300) 

st.title("Schüttbett-KI: Vorhersage des Quenchverhaltens")
st.markdown("""
### ℹ️ Anleitung

Verwenden Sie die Schieberegler auf der linken Seite, um die Parameter der Schüttung anzupassen.  
Die Geometrie wird automatisch aktualisiert und rechts angezeigt.  
Wenn eine Kühlung möglich ist, berechnet die Anwendung live eine Vorhersage der Quenchzeit mithilfe von KI-Modellen.
""")

with st.expander("🛠️ Weitere Hilfe & Hintergrundinformationen"):
    st.markdown("""
- Die Vorhersagen basieren auf einem Ensemble aus mehreren neuronalen Netzen.
- Die Eingabewerte werden automatisch normalisiert und in das KI-Modell eingespeist.
- Ist eine Kühlung laut Klassifikator möglich, wird die zu erwartende Quenchzeit berechnet.
- Die Unsicherheit der Vorhersage ergibt sich aus der Streuung der Ergebnisse aller Modelle.
- Die Visualisierung zeigt sowohl die geometrische Anordnung der Schüttung als auch die Vorhersagegrafik.
- Die App wurde für Forschungs- und Analysezwecke entwickelt und liefert keine sicherheitsrelevanten Bewertungen.

Bei Fragen oder Feedback wenden Sie sich bitte an das Entwicklerteam.
""")
st.markdown("---")


# Setup layout with two columns
col1, col2 = st.columns([2, 3])  # Left column (1 part) and right column (2 parts)

# Left column for parameter sliders
with col1:
   

    # Build sliders with manually defined parameters
    user_inputs_scaled = []
    st.markdown("### Startparameter des Schüttbetts:")
    st.markdown("---")
    for i in range(len(param_names)):

        # Show slider in real-world units
        val = st.slider(
            label=f"{descriptions[i]}",  # (`{param_names[i]}`)",
            min_value=float(min_vals[i]),
            max_value=float(max_vals[i]),
            value=(min_vals[i] + max_vals[i]) / 2,
            key=f"slider_{i}",
            format="%.2g"  # 2 significant figures
        )

        # Manually scale to [0, 1]
        scaled_val = (val - min_vals[i]) / (max_vals[i] - min_vals[i])
        user_inputs_scaled.append(scaled_val)

        # Optionally show scaled value
        
        #st.caption(f"Skalierter Wert: {scaled_val:.2f}")
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
    st.markdown("---")
    # ------------------------------------------------------------------------------
    # Load classifier model
    classifier = load_classifier()
    print(classifier.predict([user_inputs_scaled]))
    if classifier.predict([user_inputs_scaled]) == 1:
        st.markdown("### ✅ Kühlung möglich:")

        # Input vorbereiten und Vorhersagen durchführen
        input_array = np.array(user_inputs_scaled).reshape(1, -1)
        predictions = [model.predict(input_array)[0][0] for model in models]
        print("Raw Predictions:", predictions)
        if predictions and len(predictions) > 0:
            avg = np.mean(predictions)
            std = np.std(predictions)
            scaled_avg = scaler_y.inverse_transform([[avg]])[0][0]
            scaled_std = scaler_y.inverse_transform([[std]])[0][0]

            # Umrechnung in Zeitformat
            try:
                predicted_duration = timedelta(seconds=scaled_avg)
                uncertainty_duration = timedelta(seconds=scaled_std)

                def format_timedelta(td):
                    total_seconds = int(td.total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    return f"{hours}h {minutes}m {seconds}s"

                st.markdown(f"## Quenchzeit: {format_timedelta(predicted_duration)}")
                #st.caption(f"Vorhersage der Quenchzeit: {scaled_avg:.2f} Sekunden")
                st.markdown(f"### Unsicherheit: {format_timedelta(uncertainty_duration)}")
                with st.expander("Was ist ein Unsicherheit?"):
                    st.markdown("""
                    Die Unsicherheit in den Vorhersagen stammt aus verschiedenen Quellen, darunter:
                    - **Modellvariationen**: Unterschiedliche Modelle können auf die gleichen Eingabedaten auf verschiedene Weise reagieren.
                    - **Datenunsicherheit**: Variationen und Unschärfen in den Eingangsdaten, wie Messfehler oder unvollständige Daten.
                    - **Stochastische Prozesse**: Zufällige Schwankungen, die durch die intrinsische Natur der Systemdynamik verursacht werden.
                    
                    Die Unsicherheit wird durch die **Standardabweichung** der Vorhersagen der Modelle berechnet. Dies gibt an, wie stark die Modelle in ihren Vorhersagen variieren, und liefert so eine Schätzung für die Unsicherheit der Ensemble-Vorhersage.
                    """)

                # Plot erstellen
                predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                # Define x labels and numeric positions
                model_names = [f"Model {i+1}" for i in range(len(predictions))] 
                def plot_ensemble_non_interactive():
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    # Plot the predictions
                    ax.plot(predictions, marker='o', label='Vorhersage', color='b')
                    
                    # Add labels and title
                    ax.set_title("Ensemble-Vorhersagen [s]", fontsize=14)
                    #ax.set_xlabel("Modelle")
                    ax.set_ylabel("Vorhersage der Quenchzeit [s]")
                    ax.set_xticks(np.arange(len(model_names)))
                    ax.set_xticklabels(model_names)
                    
                    # Add a legend
                    ax.legend(loc='best')
                    
                    # Display a grid
                    ax.grid(True)

                    return fig

                # Show the static plot in Streamlit
                fig = plot_ensemble_non_interactive()
                st.pyplot(fig)

                # Help box with explanation
                # Dropdown (Expander) for explanation with uncertainty and standard deviation
                with st.expander("Was ist ein Ensemble?"):
                    st.markdown("""
                    Ein Ensemble kombiniert die Vorhersagen mehrerer Modelle, um die Vorhersage zu stabilisieren und zu verbessern.  
                    Dies hilft, Fehler von Einzelmodellen zu reduzieren und robustere, verlässlichere Ergebnisse zu liefern.
                    
                    Ein Ensemble wird auch verwendet, um zu berechnen, wie sicher das Modell in seiner Vorhersage ist.  
                    Dies geschieht durch die Analyse der Variation der Vorhersagen zwischen den einzelnen Modellen. Eine größere Variation deutet auf eine höhere Unsicherheit hin, während eine geringere Variation auf eine größere Zuversicht des Modells hinweist.
                    """)
            except Exception as e:
                st.error(f"Fehler bei der Umrechnung der Vorhersage: {e}")
        else:
            st.warning("⚠️ Keine gültigen Vorhersagen verfügbar. Bitte Eingaben überprüfen.")

    else:
        st.markdown("### 🚫 Kühlung nicht möglich, Schüttbett schmilzt wieder")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em;'>
    <strong>Entwickelt von:</strong> Jasmin Joshi-Thompson, Universität Stuttgart, Institut für Kernenergetik und Energiesysteme (IKE), 2025 <br>
    <strong>Kontakt:</strong> Jasmin.Joshi-Thompson@ike.uni-stuttgart.de <br>
    <strong>Version:</strong> 1.0 – Stand: April 2025<br><br>
    <em>Diese App nutzt KI-Modelle zur Live-Vorhersage von Quenchzeiten.</em><br>
</div>
""", unsafe_allow_html=True)

# Add the expander for the Hinweis information
with st.expander("Hinweis"):
    st.markdown("""
    Dieses KI-Modell wurde mit Simulationsdaten aus **COCOMO** (Corium Coolability Model) vortrainiert,  
    basierend auf früheren Arbeiten, die auf der **NENE-Konferenz** veröffentlicht wurden [1].  
    Die Simulationsdaten wurden anhand experimenteller Daten aus der **FLOAT-Versuchsanlage** [2] validiert.  
    COCOMO wurde am **Institut für Kernenergetik und Energiesysteme (IKE)** der **Universität Stuttgart** entwickelt [3].
    
    **Referenzen:**  
    [1] Joshi-Thompson, J., Buck, M., und Starflinger, J., „Application of AI Methods for Describing the Coolability of Debris Beds Formed in the Late Accident Phase of Nuclear Reactors“,  
    *Proceedings of the 33rd International Conference Nuclear Energy for New Europe (NENE 2024)*, Portorož, Slowenien, 9.–12. September 2024.

    [2] M. Petroff, R. Kulenovic, und J. Starflinger, „Experimental investigation on debris bed quenching with additional non-condensable gas injection“,  
    *Journal of Nuclear Engineering and Radiation Science*, NERS-21-1028, 2022.

    [3] Buck, M., und Pohlner, G., „Ex-Vessel Debris Bed Formation and Coolability – Challenges and Chances for Severe Accident Mitigation“,  
    *Proceedings of the International Congress on Advances in Nuclear Power Plants (ICAPP 2016)*, San Francisco, USA, 17.–20. April 2016.
    """, unsafe_allow_html=True)

