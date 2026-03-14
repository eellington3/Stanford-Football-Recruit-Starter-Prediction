# starter_heatmap_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Starter Probability Heatmap", layout="wide")
st.title("Recruit Starter Probability Heatmap (Random Forest)")

# -------------------------
# Load model and feature list
# -------------------------
rf_model = joblib.load("rf_starter_predictor.pkl")
feature_columns = joblib.load("rf_model_features.pkl")

# -------------------------
# Load original training data
# -------------------------
data = pd.read_csv("all_data.csv", encoding="latin1")

positions = ["QB","RB","WR","TE","OT","OG","C","DE","DT","LB","CB","S","LS","EDGE","IOL"]

# -------------------------
# Create position-specific limits
# -------------------------
position_limits = {}

for pos in positions:
    pos_data = data[data["position"] == pos]

    if len(pos_data) > 0:
        position_limits[pos] = {
            "height_min": int(pos_data["height"].min()),
            "height_max": int(pos_data["height"].max()),
            "weight_min": int(pos_data["weight"].min()),
            "weight_max": int(pos_data["weight"].max()),
        }

# -------------------------
# Sidebar Form (GO BUTTON)
# -------------------------
with st.sidebar.form("input_form"):

    st.header("Player Attributes")

    position = st.selectbox("Position", positions)

    conference_options = ["All"] + sorted(
        [c.replace("conference_", "") for c in feature_columns if c.startswith("conference_")]
    )

    conference = st.selectbox("Conference", conference_options)

    rating = st.slider("Rating", 0.0, 1.0, 0.95)

    limits = position_limits[position]

    st.subheader("Player Measurement")

    player_height = st.number_input(
        "Player Height (inches)",
        limits["height_min"],
        limits["height_max"],
        int((limits["height_min"] + limits["height_max"]) / 2)
    )

    player_weight = st.number_input(
        "Player Weight (lbs)",
        limits["weight_min"],
        limits["weight_max"],
        int((limits["weight_min"] + limits["weight_max"]) / 2)
    )

    grid_step = st.number_input(
        "Grid step size",
        min_value=1,
        max_value=5,
        value=1
    )

    run_model = st.form_submit_button("Generate Heatmap")

# -------------------------
# Only run model if button pressed
# -------------------------
if run_model:

    height_min = limits["height_min"]
    height_max = limits["height_max"]
    weight_min = limits["weight_min"]
    weight_max = limits["weight_max"]

    heights = np.arange(height_min, height_max + 1, grid_step)
    weights = np.arange(weight_min, weight_max + 1, grid_step)

    heatmap = np.zeros((len(weights), len(heights)))

    # -------------------------
    # Predict grid
    # -------------------------
    for i, w in enumerate(weights):
        for j, h in enumerate(heights):

            input_dict = {
                "rating": rating,
                "height": h,
                "weight": w,
            }

            input_dict[f"position_{position}"] = 1

            if conference != "All":
                input_dict[f"conference_{conference}"] = 1

            for col in feature_columns:
                if col not in input_dict:
                    input_dict[col] = 0

            df = pd.DataFrame([input_dict])[feature_columns]

            heatmap[i, j] = rf_model.predict_proba(df)[0,1]

    # -------------------------
    # Player prediction
    # -------------------------
    player_input = {
        "rating": rating,
        "height": player_height,
        "weight": player_weight,
    }

    player_input[f"position_{position}"] = 1

    if conference != "All":
        player_input[f"conference_{conference}"] = 1

    for col in feature_columns:
        if col not in player_input:
            player_input[col] = 0

    player_df = pd.DataFrame([player_input])[feature_columns]

    player_prob = rf_model.predict_proba(player_df)[0,1]

    # -------------------------
    # Plot heatmap
    # -------------------------
    fig, ax = plt.subplots(figsize=(10,6))

    c = ax.imshow(
        heatmap,
        origin='lower',
        aspect='auto',
        extent=[height_min, height_max, weight_min, weight_max],
        cmap="YlGnBu",
        vmin=0,
        vmax=1
    )

    ax.set_xlabel("Height (inches)")
    ax.set_ylabel("Weight (lbs)")
    ax.set_title(f"Predicted Starter Probability (2 yrs) - {position}, {conference}")

    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Probability")

    # Player point
    ax.scatter(
        player_height,
        player_weight,
        color="red",
        s=200,
        edgecolor="black",
        linewidth=2,
        zorder=10,
        label="Prospective Player"
    )

    ax.text(
        player_height + 0.3,
        player_weight + 0.3,
        f"{player_prob:.2%}",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    ax.legend()

    st.metric("Starter Probability (2 yrs)", f"{player_prob:.2%}")

    st.pyplot(fig)

    # -------------------------
    # Optional table
    # -------------------------
    st.subheader("Starter Probability Table")

    table_df = pd.DataFrame(heatmap, index=weights, columns=heights)

    table_df.index.name = "Weight"
    table_df.columns.name = "Height"

    def color_prob(val):

        if val >= 0.6:
            color = 'background-color: green; color: white'
        elif val >= 0.4:
            color = 'background-color: gold; color: black'
        else:
            color = 'background-color: red; color: white'

        return color

    st.dataframe(table_df.style.applymap(color_prob))

else:
    st.info("Adjust inputs in the sidebar and click **Generate Heatmap**.")