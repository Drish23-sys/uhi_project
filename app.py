import streamlit as st
import numpy as np
import rasterio
from rasterio.enums import Resampling
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import gdown
import os
import tempfile
import pandas as pd
from scipy.ndimage import uniform_filter

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UHI Predictor | Delhi NCT",
    page_icon="🌡️",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    h1 { font-size: 2.4rem !important; font-weight: 700 !important; }
    h2, h3 { font-weight: 600 !important; }
    .uhi-card {
        background: #1a1d27; border: 1px solid #2e3147;
        border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    }
    .cool-badge  { background:#1e3a4a; color:#5ecfff; padding:4px 12px; border-radius:20px; font-weight:600; font-size:1.1rem; }
    .mod-badge   { background:#3a2e10; color:#ffd166; padding:4px 12px; border-radius:20px; font-weight:600; font-size:1.1rem; }
    .hot-badge   { background:#3a1a1a; color:#ff6b6b; padding:4px 12px; border-radius:20px; font-weight:600; font-size:1.1rem; }
    .metric-box {
        background:#1a1d27; border:1px solid #2e3147;
        border-radius:10px; padding:0.9rem 1.2rem; text-align:center;
    }
    .metric-box .val { font-size:1.6rem; font-weight:700; }
    .metric-box .lbl { font-size:0.78rem; color:#8888aa; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("## Urban Heat Island Predictor")
st.markdown("**Delhi NCT · Landsat 9 · XGBoost (94.01% accuracy)**")
st.markdown("---")

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
MODEL_PATH = "uhi_xgboost_model.pkl"

@st.cache_resource(show_spinner="Loading XGBoost model…")
def load_model():
    """Load model from local path or Google Drive."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    # Fallback: download from Google Drive using secret
    gdrive_id = st.secrets.get("GDRIVE_MODEL_ID", "")
    if gdrive_id:
        gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", MODEL_PATH, quiet=False)
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Window sizes in pixels (Landsat 9 = 30m/px → 3px=90m, 7=210m, 11=330m, 15=450m, 21=630m, 31=930m)
WINDOWS = [3, 7, 11, 15, 21, 31]

def safe_divide(a, b, fill=0.0):
    out = np.where(b != 0, a / b, fill)
    return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)

def compute_indices(bands: dict) -> dict:
    B2, B3, B4 = bands["B2"], bands["B3"], bands["B4"]
    B5, B6, B7 = bands["B5"], bands["B6"], bands["B7"]
    B10 = bands["B10"]

    ndvi  = safe_divide(B5 - B4, B5 + B4)
    ndbi  = safe_divide(B6 - B5, B6 + B5)
    mndwi = safe_divide(B3 - B6, B3 + B6)
    savi  = safe_divide(1.5 * (B5 - B4), B5 + B4 + 0.5)
    nbai  = safe_divide(B6 - B7, B6 + B7)
    albedo = np.clip(0.356*B2 + 0.130*B4 + 0.373*B5 + 0.085*B6 + 0.072*B7 - 0.0018, 0, 1)
    built  = np.clip((ndbi + (1 - ndvi)) / 2, 0, 1)

    return {
        "ndvi": ndvi, "ndbi": ndbi, "mndwi": mndwi,
        "savi": savi, "nbai": nbai, "albedo": albedo,
        "built": built, "lst": B10,
    }

def build_feature_matrix(bands: dict) -> np.ndarray:
    """
    Builds 53-feature matrix per pixel matching training pipeline:
      8 pixel-level indices + 6 window sizes × 8 indices = 56 → drop 3 → 53
    """
    indices = compute_indices(bands)

    pixel_feats = np.stack([
        indices["ndvi"], indices["ndbi"], indices["mndwi"],
        indices["savi"], indices["nbai"], indices["albedo"],
        indices["built"], indices["lst"],
    ], axis=-1)  # (H, W, 8)

    spatial_feats = []
    for name in ["ndvi", "ndbi", "albedo", "built", "lst", "mndwi", "savi", "nbai"]:
        for w in WINDOWS:
            smoothed = uniform_filter(indices[name].astype(np.float32), size=w, mode="reflect")
            spatial_feats.append(smoothed)

    spatial_stack = np.stack(spatial_feats, axis=-1)  # (H, W, 48)
    all_feats = np.concatenate([pixel_feats, spatial_stack], axis=-1)  # (H, W, 56)

    # Drop 3 lowest-importance features to reach 53 (nbai_w3, savi_w3, mndwi_w3)
    keep_cols = [i for i in range(56) if i not in [38, 44, 50]]
    return all_feats[:, :, keep_cols].reshape(-1, 53)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
CLASS_LABELS = {0: "Cool", 1: "Moderate", 2: "Hot"}
CMAP = mcolors.ListedColormap(["#5ecfff", "#ffd166", "#ff6b6b"])

def predict_uhi(tif_path: str, model) -> tuple:
    with rasterio.open(tif_path) as src:
        profile = src.profile
        if src.count < 7:
            raise ValueError(
                f"Expected 7 bands (B2,B3,B4,B5,B6,B7,B10), found {src.count}. "
                "Please stack your bands before uploading."
            )

        def read_norm(idx):
            arr = src.read(idx).astype(np.float32)
            # If DN values (large numbers), apply Landsat 9 L2 scale factor
            if np.nanpercentile(arr, 95) > 10:
                arr = arr * 2.75e-5 - 0.2
            return np.clip(arr, 0, 1)

        bands = {
            "B2": read_norm(1), "B3": read_norm(2), "B4": read_norm(3),
            "B5": read_norm(4), "B6": read_norm(5), "B7": read_norm(6),
            "B10": src.read(7).astype(np.float32),
        }
        H, W = bands["B2"].shape

    feat_matrix = build_feature_matrix(bands)

    # Chunked prediction (avoids memory spikes on large scenes)
    CHUNK = 100_000
    preds = np.empty(H * W, dtype=np.uint8)
    for start in range(0, H * W, CHUNK):
        end = min(start + CHUNK, H * W)
        preds[start:end] = model.predict(feat_matrix[start:end])

    pred_map = preds.reshape(H, W)
    total = H * W
    stats = {CLASS_LABELS[c]: round(int(np.sum(pred_map == c)) / total * 100, 2) for c in range(3)}
    return pred_map, profile, stats, (H, W)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Input Requirements")
    st.markdown("""
**GeoTIFF must have 7 bands in this order:**
1. B2 (Blue)
2. B3 (Green)
3. B4 (Red)
4. B5 (NIR)
5. B6 (SWIR1)
6. B7 (SWIR2)
7. B10 (Thermal)
    """)
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("""
| | |
|---|---|
| Algorithm | XGBoost |
| Accuracy | **94.01%** |
| F1 (weighted) | **94.0%** |
| Features | **53** |
| Classes | Cool / Moderate / Hot |
    """)
    st.markdown("---")
    st.markdown("###  Legend")
    st.markdown("""
<span style='color:#5ecfff'>■</span> **Cool** – Vegetation / water bodies<br>
<span style='color:#ffd166'>■</span> **Moderate** – Mixed / residential<br>
<span style='color:#ff6b6b'>■</span> **Hot** – Dense built-up / industrial
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
col_upload, col_info = st.columns([1.2, 1])

with col_upload:
    st.markdown("### 📤 Upload Landsat 9 GeoTIFF")
    uploaded = st.file_uploader(
        "Drop your 7-band stacked GeoTIFF here",
        type=["tif", "tiff"],
        help="Stack bands B2,B3,B4,B5,B6,B7,B10 using QGIS or GDAL before uploading.",
    )

with col_info:
    st.markdown("### 🛠 How to prepare your GeoTIFF")
    st.markdown("""
**GDAL (recommended):**
```bash
gdal_merge.py -separate \\
  B2.TIF B3.TIF B4.TIF \\
  B5.TIF B6.TIF B7.TIF B10.TIF \\
  -o landsat9_stacked.tif
```

**QGIS:**
`Raster → Miscellaneous → Merge`
→ Enable "Place each input file into a separate band"
→ Select B2→B7→B10 in order
    """)

# ─────────────────────────────────────────────
# PREDICTION FLOW
# ─────────────────────────────────────────────
if uploaded is not None:
    if model is None:
        st.error(
            "⚠️ Model file not found. "
            "Place `uhi_xgboost_model.pkl` next to `app.py`, "
            "or add `GDRIVE_MODEL_ID` to `.streamlit/secrets.toml`."
        )
        st.stop()

    st.markdown("---")
    progress = st.progress(0, text="Saving uploaded file…")

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    progress.progress(20, text="Loading satellite bands…")

    try:
        progress.progress(40, text="Engineering 53 spatial features…")
        pred_map, profile, stats, (H, W) = predict_uhi(tmp_path, model)
        progress.progress(80, text="Rendering UHI map…")

        # ── STATS ─────────────────────────────────────────────
        st.markdown("### 📊 Zone Coverage")
        m1, m2, m3 = st.columns(3)
        badges = {"Cool": "cool-badge", "Moderate": "mod-badge", "Hot": "hot-badge"}
        for col, label in zip([m1, m2, m3], ["Cool", "Moderate", "Hot"]):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="val"><span class="{badges[label]}">{stats[label]}%</span></div>
                    <div class="lbl">{label} zone coverage</div>
                </div>
                """, unsafe_allow_html=True)

        # ── MAP ───────────────────────────────────────────────
        st.markdown("### 🗺️ Predicted UHI Intensity Map")
        fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        ax.imshow(pred_map, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            "Urban Heat Island Intensity Zones — Delhi NCT",
            color="white", fontsize=14, fontweight="bold", pad=12
        )
        legend_elements = [
            Patch(facecolor="#5ecfff", label="Cool"),
            Patch(facecolor="#ffd166", label="Moderate"),
            Patch(facecolor="#ff6b6b", label="Hot"),
        ]
        ax.legend(
            handles=legend_elements, loc="lower right",
            facecolor="#1a1d27", edgecolor="#2e3147",
            labelcolor="white", fontsize=11
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── DOWNLOAD PREDICTION GeoTIFF ───────────────────────
        out_path = tmp_path.replace(".tif", "_uhi_out.tif")
        with rasterio.open(
            out_path, "w",
            driver="GTiff", height=H, width=W,
            count=1, dtype=rasterio.uint8,
            crs=profile.get("crs"),
            transform=profile.get("transform"),
        ) as dst:
            dst.write(pred_map.astype(np.uint8), 1)

        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️  Download Prediction GeoTIFF",
                data=f,
                file_name="uhi_prediction_delhi.tif",
                mime="image/tiff",
            )

        progress.progress(100, text="Done ✅")

        # ── INTERPRETATION ────────────────────────────────────
        hottest = max(stats, key=stats.get)
        st.markdown(f"""
        <div class="uhi-card">
        <b>🔍 Interpretation</b><br><br>
        The scene is dominated by <b>{hottest}</b> zones ({stats[hottest]}% coverage).
        Hot zones correspond to dense commercial/industrial areas with high built-up density
        and low vegetation. Cool zones align with urban green infrastructure — parks, forests,
        and water bodies like the Yamuna floodplain.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("Ensure your GeoTIFF has exactly 7 bands stacked as B2, B3, B4, B5, B6, B7, B10.")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

else:
    st.markdown("---")
    st.markdown("""
    <div class="uhi-card" style="text-align:center; padding: 3rem;">
        <div style="font-size:3.5rem">🛰️</div>
        <div style="font-size:1.2rem; color:#8888aa; margin-top:0.8rem;">
            Upload a Landsat 9 GeoTIFF above to generate your UHI intensity map
        </div>
    </div>
    """, unsafe_allow_html=True)
