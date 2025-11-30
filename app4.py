import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import io
import plotly.graph_objects as go
# =======================
# Page Setup
# =======================
st.set_page_config(
    layout="wide",
    page_title="Multi-Model, Multi-Field RHO/U/V/P Dashboard"
)

# =======================
# Safe Model Loader
# =======================
def load_model_safe(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError()
        }
    )

st.sidebar.title("Model Loader & Settings")

# Load models
model_simple = load_model_safe("model_simple.h5")
model_bi     = load_model_safe("model_bi.h5")
model_cnn    = load_model_safe("model_cnn.h5")
model_conv   = load_model_safe("model_conv.h5")
model_lst    = load_model_safe("model_lst.h5")

# Load scaler
scaler = joblib.load("rho_scaler.pkl")

models = {
    "Simple LSTM": model_simple,
    "BiLSTM":      model_bi,
    "CNN-LSTM":    model_cnn,
    "ConvLSTM":    model_conv,
    "LST-Net":     model_lst
}

st.sidebar.success("✅ All models loaded successfully!")

# Field selection
field_options = ["RHO", "U", "V", "P"]
field_to_analyze = st.sidebar.selectbox("Field to Analyze", field_options, index=0)

# Map field → index inside [RHO, U, V, P]
field_idx = {"RHO": 0, "U": 1, "V": 2, "P": 3}

# =======================
# Data Helpers
# =======================
def load_gads_file_from_bytes(uploaded_file):
    """Read Gads_*.dat file with 6 columns: X Y RHO U V P"""
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, skiprows=2)
    df.columns = ["X", "Y", "RHO", "U", "V", "P"]
    return df

def create_sequences(data_scaled, n_past, n_future, target_cols_slice=slice(2, 6)):
    """
    Create sequences:
    X: (n_past, 6)    full input
    Y: (4,)           [RHO, U, V, P] at target step (scaled)
    """
    X, Y, idx = [], [], []
    for i in range(n_past, len(data_scaled) - n_future + 1):
        X.append(data_scaled[i-n_past:i, :])
        Y.append(data_scaled[i+n_future-1, target_cols_slice])
        idx.append(i + n_future - 1)
    return np.array(X), np.array(Y), np.array(idx)

def inverse_fields_from_scaled(scaled_Y, scaler, n_features=6):
    """
    scaled_Y: (N, 4) in scaled space: [RHO, U, V, P]
    scaler: StandardScaler fitted on all 6 features [X, Y, RHO, U, V, P]
    Returns 4 arrays of length N: RHO, U, V, P in original units.
    """
    N = scaled_Y.shape[0]
    dummy = np.zeros((N, n_features))
    dummy[:, 2:6] = scaled_Y
    inv = scaler.inverse_transform(dummy)
    return inv[:, 2], inv[:, 3], inv[:, 4], inv[:, 5]  # RHO, U, V, P

def contour_plot(ax, Xi, Yi, Z, title, cmap="jet"):
    c = ax.contourf(Xi, Yi, Z, levels=50, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return c

# =======================
# MAIN UI
# =======================
st.title("📌 Multi-Model, Multi-Field Prediction Dashboard")
st.write(
    "Upload a **Gads_*.dat** file. All 5 models will predict "
    "**RHO, U, V, P**, and you can compare them for any one field at a time."
)

uploaded = st.file_uploader("📤 Upload Gads DAT File", type=["dat"])

if uploaded:
    # -----------------------
    # Load & Scale
    # -----------------------
    df_file = load_gads_file_from_bytes(uploaded)
    data = df_file.values
    data_scaled = scaler.transform(data)

    n_past = 10
    n_future = 1
    n_features = data.shape[1]

    X_seq, Y_scaled, idx_seq = create_sequences(data_scaled, n_past, n_future)

    # shapes
    n_steps = n_past
    n_seq = 1

    X_simple = X_seq.reshape((X_seq.shape[0], n_steps, n_features))
    X_bi     = X_simple
    X_lst    = X_simple
    X_cnn    = X_seq.reshape((X_seq.shape[0], n_seq, n_steps, n_features))
    X_conv   = X_seq.reshape((X_seq.shape[0], n_seq, 1, n_steps, n_features))

    X_inputs = {
        "Simple LSTM": X_simple,
        "BiLSTM":      X_bi,
        "CNN-LSTM":    X_cnn,
        "ConvLSTM":    X_conv,
        "LST-Net":     X_lst
    }

    # Inverse all fields for ground truth
    rho_true_all, u_true_all, v_true_all, p_true_all = inverse_fields_from_scaled(
        Y_scaled, scaler
    )

    field_true_map = {
        "RHO": rho_true_all,
        "U":   u_true_all,
        "V":   v_true_all,
        "P":   p_true_all
    }
    y_true = field_true_map[field_to_analyze]

    # Coordinates for plotting
    coords = df_file.iloc[idx_seq][["X", "Y"]].copy()
    coords["True_" + field_to_analyze] = y_true

    xi = np.sort(coords["X"].unique())
    yi = np.sort(coords["Y"].unique())
    Xi, Yi = np.meshgrid(xi, yi)
    Z_true = coords.pivot(index="Y", columns="X", values="True_" + field_to_analyze).values

    # -----------------------
    # FULL GRID
    # -----------------------
    st.subheader(f"📊 Model Comparison Grid for {field_to_analyze} (True | Pred | Error)")

    fig, axes = plt.subplots(5, 3, figsize=(24, 28))

    metrics = {"Model": [], "MAE": [], "MSE": [], "RMSE": [], "R2": []}

    for row, (name, model) in enumerate(models.items()):
        # Predict (N,4) scaled [RHO, U, V, P]
        Y_pred_scaled = model.predict(X_inputs[name])
        rho_p, u_p, v_p, p_p = inverse_fields_from_scaled(Y_pred_scaled, scaler)

        field_pred_map = {
            "RHO": rho_p,
            "U":   u_p,
            "V":   v_p,
            "P":   p_p
        }
        y_pred = field_pred_map[field_to_analyze]
        y_err = y_true - y_pred

        coords[f"{name}_pred_{field_to_analyze}"] = y_pred
        coords[f"{name}_err_{field_to_analyze}"]  = y_err

        Z_pred = coords.pivot(index="Y", columns="X",
                              values=f"{name}_pred_{field_to_analyze}").values
        Z_err  = coords.pivot(index="Y", columns="X",
                              values=f"{name}_err_{field_to_analyze}").values

        contour_plot(
            axes[row, 0], Xi, Yi, Z_true,
            f"{name} – TRUE {field_to_analyze}", cmap="jet"
        )
        contour_plot(
            axes[row, 1], Xi, Yi, Z_pred,
            f"{name} – PRED {field_to_analyze}", cmap="jet"
        )
        contour_plot(
            axes[row, 2], Xi, Yi, Z_err,
            f"{name} – ERROR {field_to_analyze}", cmap="coolwarm"
        )

        # Metrics for this model & field
        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_true, y_pred)

        metrics["Model"].append(name)
        metrics["MAE"].append(mae)
        metrics["MSE"].append(mse)
        metrics["RMSE"].append(rmse)
        metrics["R2"].append(r2)

    plt.tight_layout()
    st.pyplot(fig)

    metrics_df = pd.DataFrame(metrics)

    # -----------------------
    # METRICS BAR CHART
    # -----------------------
    st.subheader(f"📈 Metrics Comparison for {field_to_analyze}")

    fig2, ax2 = plt.subplots(figsize=(16, 8))
    x = np.arange(len(metrics_df["Model"]))

    ax2.bar(x - 0.3, metrics_df["MAE"],  width=0.15, label="MAE")
    ax2.bar(x - 0.15, metrics_df["MSE"], width=0.15, label="MSE")
    ax2.bar(x + 0.0,  metrics_df["RMSE"],width=0.15, label="RMSE")
    ax2.bar(x + 0.15, metrics_df["R2"],  width=0.15, label="R²")

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_df["Model"], rotation=45, ha="right")
    ax2.set_title(f"Model Performance for {field_to_analyze}")
    ax2.legend()
    st.pyplot(fig2)

    # -----------------------
    # RADAR / SPIDER PLOT
    # -----------------------
    st.subheader(f"🕸️ Radar Plot (Normalized Metrics) – {field_to_analyze}")

    norm_df = metrics_df.copy()
    for col in ["MAE", "MSE", "RMSE"]:
        denom = (norm_df[col].max() - norm_df[col].min())
        denom = denom if denom != 0 else 1.0
        norm_df[col] = (norm_df[col] - norm_df[col].min()) / denom

    denom_r2 = (norm_df["R2"].max() - norm_df["R2"].min())
    denom_r2 = denom_r2 if denom_r2 != 0 else 1.0
    norm_df["R2"] = (norm_df["R2"] - norm_df["R2"].min()) / denom_r2

    categories = ["MAE", "MSE", "RMSE", "R2"]
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig3 = plt.figure(figsize=(10, 10))
    ax3 = plt.subplot(111, polar=True)

    for i, row in norm_df.iterrows():
        values = row[categories].tolist()
        values += values[:1]
        ax3.plot(angles, values, linewidth=2, label=row["Model"])
        ax3.fill(angles, values, alpha=0.15)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_title(
        f"Normalized Performance Radar Chart – {field_to_analyze}",
        size=16
    )
    ax3.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
    st.pyplot(fig3)

    # -----------------------
    # RANKING TABLE
    # -----------------------
    st.subheader(f"🏆 Model Ranking (by RMSE) – {field_to_analyze}")

    ranking_df = metrics_df.sort_values(by="RMSE", ascending=True)
    ranking_df.index = range(1, len(ranking_df) + 1)
    st.dataframe(ranking_df)

    # -----------------------
    # CSV DOWNLOAD
    # -----------------------
    st.subheader("💾 Download All Predictions (All Models & This Field)")

    csv = coords.to_csv(index=False).encode()
    st.download_button(
        "Download Full Prediction CSV",
        csv,
        f"All_Models_{field_to_analyze}_Predictions.csv",
        "text/csv"
    )

# ----------------------------------------------------
# ERROR HISTOGRAMS
# ----------------------------------------------------
st.subheader(f"📉 Error Histogram – {field_to_analyze}")

fig_h, ax_h = plt.subplots(1, 2, figsize=(16, 6))

# Absolute Errors
ax_h[0].hist((y_true - y_pred), bins=50, color="steelblue")
ax_h[0].set_title(f"Absolute Error Histogram ({field_to_analyze})")
ax_h[0].set_xlabel("Absolute Error")
ax_h[0].set_ylabel("Frequency")

# Relative Errors
rel_error = ((y_true - y_pred) / (y_true + 1e-6))
ax_h[1].hist(rel_error, bins=50, color="tomato")
ax_h[1].set_title(f"Relative Error Histogram ({field_to_analyze})")
ax_h[1].set_xlabel("Relative Error")
ax_h[1].set_ylabel("Frequency")

st.pyplot(fig_h)

# ----------------------------------------------------
# PARITY PLOT
# ----------------------------------------------------
st.subheader(f"📌 Parity Plot – True vs Predicted ({field_to_analyze})")

fig_p, ax_p = plt.subplots(figsize=(7, 7))

ax_p.scatter(y_true, y_pred, s=8, alpha=0.5, color="purple")
ax_p.plot([y_true.min(), y_true.max()],
          [y_true.min(), y_true.max()],
          'r--', linewidth=2)

ax_p.set_xlabel("True Values")
ax_p.set_ylabel("Predicted Values")
ax_p.set_title(f"Parity Plot – {field_to_analyze}")

st.pyplot(fig_p)


# ----------------------------------------------------
# STREAMLINES (U-V Velocity Field)
# ----------------------------------------------------
if field_to_analyze in ["U", "V"]:
    st.subheader("🌪 Streamlines – Velocity Field Comparison")

    # Build 2D fields
    ZU_true = coords.pivot(index="Y", columns="X", values="True_U").values
    ZV_true = coords.pivot(index="Y", columns="X", values="True_V").values

    ZU_pred = coords.pivot(index="Y", columns="X", values=f"{name}_pred_U").values
    ZV_pred = coords.pivot(index="Y", columns="X", values=f"{name}_pred_V").values

    fig_s, axes_s = plt.subplots(1, 2, figsize=(18, 7))

    # True streamlines
    axes_s[0].streamplot(Xi, Yi, ZU_true, ZV_true, color="black", density=1.5)
    axes_s[0].set_title("True Streamlines")

    # Pred streamlines
    axes_s[1].streamplot(Xi, Yi, ZU_pred, ZV_pred, color="blue", density=1.5)
    axes_s[1].set_title("Predicted Streamlines")

    st.pyplot(fig_s)


