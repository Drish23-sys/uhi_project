# UHI Predictor — Deployment Guide

**Urban Heat Island prediction for Delhi NCT using Landsat 9 + XGBoost (94.01% accuracy)**

---

## File Structure

```
your-repo/
├── app.py                    ← Main Streamlit app
├── requirements.txt          ← Python dependencies
├── uhi_xgboost_model.pkl     ← Your trained model (option A: local)
└── .streamlit/
    └── secrets.toml          ← Google Drive model ID (option B: cloud)
```

---

## Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add app.py requirements.txt
git commit -m "Initial UHI predictor"
git remote add origin https://github.com/YOUR_USERNAME/uhi-predictor.git
git push -u origin main
```

### Step 2 — Connect your model (choose one)

**Option A — Include model in repo (if < 100MB):**
```bash
git add uhi_xgboost_model.pkl
git commit -m "Add model"
git push
```

**Option B — Google Drive (recommended for large models):**
1. Upload `uhi_xgboost_model.pkl` to Google Drive
2. Right-click → Share → Anyone with link → Copy link
3. Extract the file ID from the link
4. In Streamlit Cloud dashboard → App settings → Secrets → paste:
   ```toml
   GDRIVE_MODEL_ID = "1AbCdEfGhIjKlMnOpQ..."
   ```

### Step 3 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy** 

---

## 🛰️ How to prepare your Landsat 9 input

Your GeoTIFF must have **7 bands stacked in this order**:
`B2 → B3 → B4 → B5 → B6 → B7 → B10`

**Using GDAL:**
```bash
gdal_merge.py -separate \
  LC09_B2.TIF LC09_B3.TIF LC09_B4.TIF \
  LC09_B5.TIF LC09_B6.TIF LC09_B7.TIF LC09_B10.TIF \
  -o landsat9_stacked.tif
```

**Using QGIS:**
`Raster → Miscellaneous → Merge → Place each input into a separate band`

---

## Feature Engineering Summary

The app replicates your paper's 53-feature pipeline:

| Feature Type | Count | Description |
|---|---|---|
| Pixel-level indices | 8 | NDVI, NDBI, MNDWI, SAVI, NBAI, Albedo, Built fraction, LST |
| Multi-scale spatial (×6 windows) | 45 | Mean of each index at 90m–930m neighbourhood |
| **Total** | **53** | Matching training pipeline |

Window sizes: 3px (90m), 7px (210m), 11px (330m), 15px (450m), 21px (630m), 31px (930m)

---

## Run Locally

```bash
pip install -r requirements.txt
# Place uhi_xgboost_model.pkl in the same folder
streamlit run app.py
```

---

## ⚠️ Important Notes

- The model was trained on **Delhi NCT summer 2023** data — accuracy may vary for other cities/seasons
- Input bands should be **Landsat 9 Collection 2 Level-2** (surface reflectance) for best results
- Large scenes (full Landsat tile ~7500×7500px) may take 2–5 minutes to process on Streamlit Cloud free tier
