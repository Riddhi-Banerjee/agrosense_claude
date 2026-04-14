import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    load_model = None
    TF_AVAILABLE = False
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="AgroSense — Smart Farm Anomaly Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main { background-color: #0d1117; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #0f1e12 50%, #0d1117 100%); }
section[data-testid="stSidebar"] { background: #0f1a12; border-right: 1px solid #1e3a28; }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #e8f5e3 !important; }
section[data-testid="stSidebar"] h3 { color: #4caf72 !important; }
section[data-testid="stSidebar"] .stNumberInput label { color: #e8f5e3 !important; }
section[data-testid="stSidebar"] .stNumberInput input {
    background: #1a2e1f !important; color: #e8f5e3 !important;
    border: 1px solid #2e5a3a !important; border-radius: 6px !important; }
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio > div > label { color: #e8f5e3 !important; }
.stSlider > div > div > div > div { background: #4caf72 !important; }
section[data-testid="stSidebar"] .stButton > button {
    color: #000000 !important; font-weight: 700 !important;
    background: #4caf72 !important; border: none !important; }
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #66bb6a !important; color: #000000 !important; }
.hero-title { font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800;
    color:#e8f5e3; letter-spacing:-0.02em; line-height:1.1; margin-bottom:0.2rem; }
.hero-sub { font-family:'DM Mono',monospace; font-size:0.8rem; color:#4caf72;
    letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2rem; }
.result-anomaly { background:linear-gradient(135deg,#2d0f0f,#1a0808);
    border:1.5px solid #e53935; border-radius:14px; padding:1.5rem 2rem; margin:1rem 0; }
.result-normal  { background:linear-gradient(135deg,#0a2615,#061a0e);
    border:1.5px solid #4caf72; border-radius:14px; padding:1.5rem 2rem; margin:1rem 0; }
.result-borderline { background:linear-gradient(135deg,#2d2200,#1a1500);
    border:1.5px solid #f5a623; border-radius:14px; padding:1.5rem 2rem; margin:1rem 0; }
.result-tendency { background:linear-gradient(135deg,#1e1530,#120e20);
    border:1.5px solid #9c6fde; border-radius:14px; padding:1.5rem 2rem; margin:1rem 0; }
.result-title { font-size:1.5rem; font-weight:700; margin-bottom:0.3rem; }
.section-head { font-family:'DM Mono',monospace; font-size:0.72rem;
    letter-spacing:0.15em; text-transform:uppercase; color:#4caf72;
    border-bottom:1px solid #1e3a28; padding-bottom:0.5rem; margin:1.5rem 0 1rem 0; }
.sensor-card { background:#161d1a; border-radius:10px; padding:1rem 1.2rem;
    margin-bottom:0.75rem; border-left:4px solid #1e3a28; }
.sensor-card.anomaly { border-left-color:#ef5350; }
.sensor-card.warning { border-left-color:#f5a623; }
.sensor-card.ok      { border-left-color:#4caf72; }
.sensor-card.purple  { border-left-color:#9c6fde; }
.sc-feature { font-family:'DM Mono',monospace; font-size:0.7rem;
    text-transform:uppercase; letter-spacing:0.1em; color:#6b8f73; margin-bottom:4px; }
.sc-value  { font-size:1.3rem; font-weight:700; color:#e8f5e3; margin-bottom:2px; }
.sc-status { font-size:0.82rem; margin-bottom:4px; }
.sc-desc   { font-size:0.78rem; color:#9ab5a0; line-height:1.55; }
.rec-card { background:#0f1e12; border:1px solid #1e3a28;
    border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.75rem; }
.rec-card.urgent  { border-color:#ef5350; background:#1a0808; }
.rec-card.caution { border-color:#f5a623; background:#1a1200; }
.rec-card.info    { border-color:#378ADD; background:#0a1520; }
.rc-num   { font-family:'DM Mono',monospace; font-size:0.7rem; color:#4caf72; margin-bottom:4px; }
.rc-title { font-size:0.95rem; font-weight:600; color:#e8f5e3; margin-bottom:4px; }
.rc-desc  { font-size:0.8rem; color:#9ab5a0; line-height:1.55; }
.prob-bar-wrap { background:#1e3a28; border-radius:4px; height:8px; margin-top:2px; }
.prob-bar-fill { height:8px; border-radius:4px; background:#4caf72; }
.stRadio > div { flex-direction:row !important; gap:1.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_bundle():
    with open("models (1)/model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    autoencoders = {}
    for fname in os.listdir("models"):
        if fname.startswith("ae_") and fname.endswith(".keras"):
            parts  = fname.replace(".keras", "").split("_")
            crop   = int(parts[1])
            region = int(parts[2])
            if TF_AVAILABLE:
                autoencoders[(crop, region)] = load_model(
                    f"models/{fname}", compile=False)
            else:
                autoencoders[(crop, region)] = None
    bundle['autoencoders'] = autoencoders
    return bundle

bundle        = load_bundle()
if_models     = bundle['if_models']
lof_models    = bundle['lof_models']
autoencoders  = bundle['autoencoders']
scalers       = bundle['scalers']
fusion_models = bundle['fusion_models']
norm_if       = bundle['norm_if']
norm_lof      = bundle['norm_lof']
norm_ae       = bundle['norm_ae']
norm_fus      = bundle['norm_fus']
weights       = bundle['weights']
best_thresh   = 0.25
ae_thresh_raw = bundle['ae_thresh_raw']
encoders      = bundle['encoders']
segment_stats = bundle['segment_stats']
FEATURES      = bundle['features']
perf          = bundle['performance_metrics']
type_clf      = bundle.get('type_classifier', None)
clf_feat_names= bundle.get('classifier_feature_names', None)
cluster_summary = bundle.get('cluster_summary', None)

if 'crop_base_map' in bundle:
    crop_base_map = bundle['crop_base_map']
else:
    _base_yields = {'Cotton':3800,'Maize':5500,'Rice':4500,'Soybean':3200,'Wheat':4800}
    crop_base_map = {}
    for name in encoders['crop_type'].classes_:
        enc = int(encoders['crop_type'].transform([name])[0])
        crop_base_map[enc] = _base_yields.get(name, 4000)

CROP_LIST   = list(encoders['crop_type'].classes_)
REGION_LIST = list(encoders['region'].classes_)

_base_yields = {'Cotton':3800,'Maize':5500,'Rice':4500,'Soybean':3200,'Wheat':4800}

# ============================================================
# YIELD FORMULA
# ============================================================
def formula_yield(user_data, crop_enc):
    base = crop_base_map.get(int(crop_enc), 4000)
    sm   = user_data.get('soil_moisture_%', 25)
    tmp  = user_data.get('temperature_C', 24)
    ph   = user_data.get('soil_pH', 6.5)
    ndvi = user_data.get('NDVI_index', 0.6)
    rain = user_data.get('rainfall_mm', 180)
    sun  = user_data.get('sunlight_hours', 7)
    pest = user_data.get('pesticide_usage_ml', 25)
    sm_f  = 1.0 if 20<=sm<=35 else (max(0.40,0.60+0.02*sm) if sm<20 else max(0.65,1.0-0.012*(sm-35)))
    tmp_f = 1.0 if 20<=tmp<=30 else (max(0.65,0.65+0.017*(tmp-10)) if tmp<20 else max(0.45,1.0-0.028*(tmp-30)))
    ph_f  = 1.0 if 6.0<=ph<=7.0 else max(0.70,1.0-0.12*abs(ph-6.5))
    ndvi_f= max(0.30,min(1.30,0.50+0.80*ndvi))
    rain_f= 1.0 if 100<=rain<=250 else (max(0.50,0.50+0.005*rain) if rain<100 else max(0.70,1.0-0.0012*(rain-250)))
    sun_f = min(1.15,max(0.70,0.70+0.045*sun))
    pest_f= 1.0 if 15<=pest<=35 else max(0.80,1.0-0.006*abs(pest-25))
    return max(300.0, round(base*sm_f*tmp_f*ph_f*ndvi_f*rain_f*sun_f*pest_f, 0))

# ============================================================
# SCORE HELPERS
# ============================================================
def encode_val(val, encoder):
    if val in encoder.classes_:
        return int(encoder.transform([val])[0])
    return None

def get_if_score(model, X):
    return float(-model.decision_function(X)[0])

def get_lof_score(model, X):
    return float(-model.decision_function(X)[0])

def get_ae_score(ae, scaler, X):
    if ae is None: return 0.0
    Xs = scaler.transform(X); recon = ae.predict(Xs, verbose=0)
    return float(np.mean((Xs - recon) ** 2))

def get_ae_feature_scores(ae, scaler, X):
    if ae is None: return {f: 0.0 for f in FEATURES}
    Xs = scaler.transform(X); recon = ae.predict(Xs, verbose=0)
    return dict(zip(FEATURES, (Xs - recon)[0] ** 2))

def get_fusion_score(row_dict, key):
    if key not in fusion_models: return 0.0, {}
    seg = fusion_models[key]; total_z = 0.0; detail = {}
    for (xf, yf), info in seg.items():
        expected = info['model'].predict([[row_dict[xf]]])[0]
        residual = abs(row_dict[yf] - expected)
        z = residual / info['std']; total_z += z
        detail[f"{xf} → {yf}"] = round(z, 3)
    return total_z / max(len(seg), 1), detail

def predict(user_data):
    crop_enc   = encode_val(user_data['crop_type'], encoders['crop_type'])
    region_enc = encode_val(user_data['region'],    encoders['region'])
    if crop_enc is None or region_enc is None: return None
    key = (crop_enc, region_enc)
    if key not in if_models: return None
    X = np.array([[user_data[f] for f in FEATURES]])
    if_r  = get_if_score(if_models[key], X)
    lof_r = get_lof_score(lof_models[key], X)
    ae_r  = get_ae_score(autoencoders[key], scalers[key], X)
    fs_r, fus_detail = get_fusion_score(user_data, key)
    if_s  = float(norm_if.transform([[if_r]])[0][0])
    lof_s = float(norm_lof.transform([[lof_r]])[0][0])
    ae_s  = float(norm_ae.transform([[ae_r]])[0][0])
    fus_s = float(norm_fus.transform([[fs_r]])[0][0])
    final_score = (weights['if']*if_s + weights['lof']*lof_s +
                   weights['ae']*ae_s + weights['fusion']*fus_s)
    seg_s = segment_stats.get(key, {})
    param_issues = []
    for f in FEATURES:
        if f not in seg_s: continue
        val = user_data[f]; low = seg_s[f]['low']; high = seg_s[f]['high']
        if val < low:
            param_issues.append({'feature':f,'value':val,'status':'LOW','low':low,'high':high})
        elif val > high:
            param_issues.append({'feature':f,'value':val,'status':'HIGH','low':low,'high':high})

    # Anomaly type classification
    type_pred, type_proba, type_classes = None, None, None
    if type_clf is not None and final_score > best_thresh:
        cls_input = np.array([[if_s, lof_s, ae_s, fus_s] +
                               [user_data[f] for f in FEATURES]])
        type_pred    = type_clf.predict(cls_input)[0]
        type_proba   = type_clf.predict_proba(cls_input)[0]
        type_classes = type_clf.classes_

    return {
        'final_score': final_score,
        'if_s': if_s, 'lof_s': lof_s, 'ae_s': ae_s, 'fus_s': fus_s,
        'fusion_detail': fus_detail,
        'ae_feat': get_ae_feature_scores(autoencoders[key], scalers[key], X),
        'param_issues': param_issues,
        'predicted_yield': formula_yield(user_data, crop_enc),
        'seg_stats': seg_s, 'key': key,
        'crop_enc': crop_enc, 'region_enc': region_enc,
        'type_pred': type_pred, 'type_proba': type_proba, 'type_classes': type_classes,
    }

# ============================================================
# PLAIN ENGLISH
# ============================================================
FEATURE_LABELS = {
    'soil_moisture_%':'Soil Moisture','soil_pH':'Soil pH',
    'temperature_C':'Temperature','rainfall_mm':'Rainfall',
    'humidity_%':'Humidity','NDVI_index':'NDVI (Crop Health Index)',
}
FEATURE_UNITS = {
    'soil_moisture_%':'%','soil_pH':'','temperature_C':'°C',
    'rainfall_mm':' mm','humidity_%':'%','NDVI_index':'',
}

def feature_plain_english(f, val, status, low, high, crop, region):
    unit = FEATURE_UNITS.get(f,'')
    expl = {
        'soil_moisture_%':{
            'LOW': f"Soil moisture is critically low at {val:.1f}%. The healthy range for {crop} in {region} is {low:.1f}–{high:.1f}%. Dry soil causes root stress, wilting and reduced nutrient uptake.",
            'HIGH':f"Soil moisture is too high at {val:.1f}% (normal: {low:.1f}–{high:.1f}%). Excess water drowns roots and promotes fungal disease such as root rot."},
        'soil_pH':{
            'LOW': f"Soil pH is {val:.2f} — too acidic for {crop} (normal: {low:.2f}–{high:.2f}). Acidic soil blocks phosphorus and calcium absorption.",
            'HIGH':f"Soil pH is {val:.2f} — too alkaline (normal: {low:.2f}–{high:.2f}). Alkaline conditions lock out iron and manganese, causing yellowing leaves."},
        'temperature_C':{
            'LOW': f"Temperature is {val:.1f}°C — below the safe threshold for {crop} in {region} (normal: {low:.1f}–{high:.1f}°C). Cold slows photosynthesis and delays flowering.",
            'HIGH':f"Temperature is {val:.1f}°C — critically above the safe range of {low:.1f}–{high:.1f}°C for {crop}. Extreme heat damages pollen and accelerates moisture loss."},
        'rainfall_mm':{
            'LOW': f"Rainfall is only {val:.0f} mm against the expected {low:.0f}–{high:.0f} mm for {crop} in {region}. This moisture deficit requires immediate irrigation.",
            'HIGH':f"Rainfall of {val:.0f} mm greatly exceeds the {low:.0f}–{high:.0f} mm normal range. Flooding risk is high — excess water leaches nutrients."},
        'humidity_%':{
            'LOW': f"Humidity is {val:.1f}% — lower than the {low:.1f}–{high:.1f}% range expected for {crop}. Low humidity increases leaf transpiration and water stress.",
            'HIGH':f"Humidity is {val:.1f}% — above the safe {low:.1f}–{high:.1f}% range. High humidity creates ideal conditions for blight and mildew in {crop}."},
        'NDVI_index':{
            'LOW': f"NDVI is {val:.2f} — well below the healthy {low:.2f}–{high:.2f} range. A low NDVI signals poor canopy coverage — the crop may be stressed, diseased, or pest-damaged.",
            'HIGH':f"NDVI is {val:.2f} — above the expected {low:.2f}–{high:.2f}. Unusually high NDVI can indicate sensor miscalibration or weed overgrowth."},
    }
    default = f"{FEATURE_LABELS.get(f,f)} reading of {val:.2f}{unit} is outside expected range of {low:.2f}–{high:.2f}{unit} for {crop} in {region}."
    return expl.get(f,{}).get(status, default)

def fusion_plain_english(pair, z, crop, region):
    sev = "critically" if z>3.0 else "significantly" if z>2.0 else "slightly"
    expl = {
        'rainfall_mm → soil_moisture_%':
            f"Given the rainfall reading, the expected soil moisture is {sev} different from what the soil sensor reports (deviation: {z:.1f}σ). This level of rain should produce higher soil moisture. Likely causes: sensor fault, rapid drainage, or rainfall data error.",
        'temperature_C → humidity_%':
            f"Temperature and humidity readings are {sev} inconsistent (deviation: {z:.1f}σ). A faulty humidity sensor or unusual microclimate event may explain this gap.",
        'NDVI_index → soil_moisture_%':
            f"Crop health (NDVI) and soil moisture are {sev} inconsistent (deviation: {z:.1f}σ). A crop with this NDVI typically requires more soil moisture than reported. Possible soil sensor fault.",
    }
    return expl.get(pair, f"Sensor pair '{pair}' shows a {sev} inconsistency (z={z:.1f}σ). Cross-check both sensors manually.")

TYPE_DESCRIPTIONS = {
    'drought_stress':      ('Drought Stress', '#EF9F27',
        'Soil moisture, rainfall and NDVI are all critically low. The crop is experiencing water deficit stress. Immediate irrigation is required.'),
    'heat_stress':         ('Heat Stress', '#E24B4A',
        'Temperature is abnormally high and humidity is elevated. The crop is under thermal stress which damages pollen and accelerates water loss.'),
    'sensor_inconsistency':('Sensor Inconsistency', '#9c6fde',
        'High rainfall but very low soil moisture — this combination is physically unlikely. A sensor fault is the probable cause rather than a real field condition.'),
    'crop_failure':        ('Crop Failure', '#D85A30',
        'NDVI, soil moisture and temperature are all anomalous simultaneously. Multiple stress factors are combining to cause significant crop health decline.'),
}

def generate_recommendations(param_issues, fusion_detail, score, crop,
                              region, predicted_yield, type_pred):
    recs = []
    if score > best_thresh:
        recs.append({'type':'urgent','title':'Immediate field inspection required',
            'desc':f"The anomaly score exceeds the detection threshold. Multiple sensors are reporting abnormal conditions for {crop} in {region}. Visit the field within 24 hours to verify readings and check crop health."})
    elif score >= best_thresh - 0.05:
        recs.append({'type':'caution','title':'Schedule a precautionary field check',
            'desc':f"Conditions are approaching the anomaly threshold for {crop} in {region}. A precautionary inspection within 48 hours is recommended."})

    feature_recs = {
        ('soil_moisture_%','LOW'): ('urgent','Activate emergency irrigation now',
            "Soil moisture is critically low. Start drip or sprinkler irrigation within 6–12 hours. Target the 20–35% range and monitor every 12 hours."),
        ('soil_moisture_%','HIGH'):('caution','Improve field drainage immediately',
            "Pause all irrigation. Clear drainage channels and create furrows to redirect excess water. Watch for root rot symptoms."),
        ('temperature_C','HIGH'):  ('urgent','Apply heat stress mitigation',
            "Apply reflective mulch, increase irrigation frequency, and avoid field work during peak heat hours."),
        ('temperature_C','LOW'):   ('caution','Protect crops from cold stress',
            "Use frost protection cloth overnight. Delay transplanting until temperatures return to normal."),
        ('soil_pH','LOW'):         ('caution','Apply lime to correct soil acidity',
            "Apply agricultural lime at 1–2 tonnes/hectare. Retest pH after 4–6 weeks."),
        ('soil_pH','HIGH'):        ('caution','Apply sulphur to reduce alkalinity',
            "Apply elemental sulphur or ammonium sulphate. Add compost to gradually lower pH."),
        ('rainfall_mm','LOW'):     ('urgent','Switch to supplemental irrigation — drought risk',
            f"Rainfall is far below requirements for {crop} in {region}. Begin scheduled irrigation and apply mulch."),
        ('rainfall_mm','HIGH'):    ('caution','Monitor for flood damage and disease',
            "Inspect drainage systems. Apply preventive fungicide within 48 hours."),
        ('humidity_%','HIGH'):     ('caution','Apply preventive fungicide — disease risk elevated',
            "High humidity creates conditions ideal for fungal disease. Spray broad-spectrum fungicide."),
        ('NDVI_index','LOW'):      ('urgent','Investigate crop health decline urgently',
            "NDVI is critically low — inspect for disease, pest damage, or nutrient deficiency."),
    }
    for iss in param_issues:
        k = (iss['feature'], iss['status'])
        if k in feature_recs:
            t, title, desc = feature_recs[k]
            recs.append({'type':t,'title':title,'desc':desc})

    for pair, z in fusion_detail.items():
        if z > 2.0:
            sensor_name = pair.split('→')[0].strip().replace('_',' ')
            recs.append({'type':'info','title':f'Verify sensor calibration — {sensor_name}',
                'desc':f"The {pair} relationship is statistically inconsistent (z={z:.1f}σ). This may indicate a faulty sensor. Take a manual measurement before acting on this alert."})

    crop_avg = _base_yields.get(crop, 4033)
    if predicted_yield < crop_avg * 0.70:
        recs.append({'type':'urgent','title':'Predicted yield critically low',
            'desc':f"Model predicts {predicted_yield:.0f} kg/ha, more than 30% below the expected average of ~{crop_avg:,} kg/ha for {crop}. Address the urgent conditions above immediately."})
    elif predicted_yield > crop_avg * 1.10:
        recs.append({'type':'info','title':'Yield outlook is positive',
            'desc':f"Predicted yield of {predicted_yield:.0f} kg/ha is above average for {crop}. Maintain current practices and address any flagged issues promptly."})

    if not recs:
        recs.append({'type':'info','title':'All conditions normal — continue routine monitoring',
            'desc':f"No anomalies detected for {crop} in {region}. Next recommended check: 48–72 hours."})
    return recs

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem;color:#e8f5e3;">🌾 AgroSense</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Anomaly Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Farm Context")
    crop_sel   = st.selectbox("Crop Type", CROP_LIST)
    region_sel = st.selectbox("Region",    REGION_LIST)
    st.markdown("### Input Mode")
    input_mode = st.radio("", ["🎚 Sliders", "🔢 Manual Entry"], horizontal=True)
    st.markdown("### Sensor Readings")
    if input_mode == "🎚 Sliders":
        soil_moist  = st.slider("Soil Moisture (%)",  0.0, 60.0, 25.0, 0.1)
        soil_ph     = st.slider("Soil pH",            4.0,  9.0,  6.5, 0.01)
        temperature = st.slider("Temperature (°C)",  10.0, 55.0, 24.0, 0.1)
        rainfall    = st.slider("Rainfall (mm)",       0.0,400.0,180.0, 1.0)
        humidity    = st.slider("Humidity (%)",        0.0,100.0, 65.0, 0.1)
        ndvi        = st.slider("NDVI Index",          0.0,  1.0,  0.6, 0.01)
    else:
        soil_moist  = st.number_input("Soil Moisture (%)",  0.0, 60.0,  25.0, 0.1,  format="%.1f")
        soil_ph     = st.number_input("Soil pH",            4.0,  9.0,   6.5, 0.01, format="%.2f")
        temperature = st.number_input("Temperature (°C)",  10.0, 55.0,  24.0, 0.1,  format="%.1f")
        rainfall    = st.number_input("Rainfall (mm)",      0.0,400.0, 180.0, 1.0,  format="%.0f")
        humidity    = st.number_input("Humidity (%)",       0.0,100.0,  65.0, 0.1,  format="%.1f")
        ndvi        = st.number_input("NDVI Index",         0.0,  1.0,   0.6, 0.01, format="%.2f")
    st.markdown("### Farm Details")
    if input_mode == "🎚 Sliders":
        sunlight   = st.slider("Sunlight Hours/day", 2.0, 14.0,  7.0, 0.1)
        pesticide  = st.slider("Pesticide (ml)",      0.0, 60.0, 25.0, 0.5)
        total_days = st.slider("Growing Days",        60,  180,  120,   1)
    else:
        sunlight   = st.number_input("Sunlight Hours/day", 2.0, 14.0,  7.0, 0.1, format="%.1f")
        pesticide  = st.number_input("Pesticide (ml)",     0.0, 60.0, 25.0, 0.5, format="%.1f")
        total_days = st.number_input("Growing Days",       60,  180,  120,   1)
    sow_month = st.selectbox("Sowing Month",
                             ['Jan','Feb','Mar','Apr','May','Jun',
                              'Jul','Aug','Sep','Oct','Nov','Dec'])
    month_num = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec'].index(sow_month) + 1
    st.markdown("### Farm Location (optional)")
    lat_input = st.number_input("Latitude",  -90.0, 90.0,  None, 0.01, format="%.4f")
    lon_input = st.number_input("Longitude",-180.0,180.0,  None, 0.01, format="%.4f")
    run_btn = st.button("🔍 Analyse Farm Conditions", use_container_width=True)

# ============================================================
# MAIN LAYOUT
# ============================================================
st.markdown('<div class="hero-title">AgroSense Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Hybrid Contextual Anomaly Detection · Precision Agriculture</div>',
            unsafe_allow_html=True)

if run_btn:
    user_data = {
        'crop_type': crop_sel, 'region': region_sel,
        'soil_moisture_%': float(soil_moist), 'soil_pH': float(soil_ph),
        'temperature_C': float(temperature), 'rainfall_mm': float(rainfall),
        'humidity_%': float(humidity), 'NDVI_index': float(ndvi),
        'sunlight_hours': float(sunlight), 'pesticide_usage_ml': float(pesticide),
        'total_days': int(total_days),
        'season_sin': np.sin(2*np.pi*month_num/12),
        'season_cos': np.cos(2*np.pi*month_num/12),
    }
    if lat_input is not None: user_data['latitude']  = float(lat_input)
    if lon_input is not None: user_data['longitude'] = float(lon_input)

    result = predict(user_data)

    if result is None:
        st.error("No trained model found for this crop-region combination.")
    else:
        score         = result['final_score']
        tendency_low  = best_thresh - 0.05

        if score > best_thresh:
            st.markdown(f"""
            <div class="result-anomaly">
                <div class="result-title" style="color:#ef5350;">⚠ Anomaly Detected</div>
                <div style="font-size:0.85rem;color:#ef9a9a;margin-top:0.5rem;">
                    Abnormal farming conditions identified for <strong>{crop_sel}</strong>
                    in <strong>{region_sel}</strong>. Review the sensor analysis and
                    recommended actions below.
                </div>
            </div>""", unsafe_allow_html=True)
        elif tendency_low <= score <= best_thresh:
            st.markdown(f"""
            <div class="result-tendency">
                <div class="result-title" style="color:#9c6fde;">🔮 Tendency Towards Anomaly</div>
                <div style="font-size:0.85rem;color:#c9a8f5;margin-top:0.5rem;">
                    Conditions for <strong>{crop_sel}</strong> in <strong>{region_sel}</strong>
                    are shifting in an unfavourable direction. Monitor closely over 24–48 hours.
                </div>
            </div>""", unsafe_allow_html=True)
        elif score >= 0.5:
            st.markdown(f"""
            <div class="result-borderline">
                <div class="result-title" style="color:#f5a623;">⚡ Borderline Conditions</div>
                <div style="font-size:0.85rem;color:#ffe082;margin-top:0.5rem;">
                    Readings for <strong>{crop_sel}</strong> in <strong>{region_sel}</strong>
                    are approaching the anomalous range. Monitor closely.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-normal">
                <div class="result-title" style="color:#66bb6a;">✓ Conditions Normal</div>
                <div style="font-size:0.85rem;color:#a5d6a7;margin-top:0.5rem;">
                    All sensor readings are within the expected range for
                    <strong>{crop_sel}</strong> in <strong>{region_sel}</strong>.
                </div>
            </div>""", unsafe_allow_html=True)

        # [CLS] Anomaly type diagnosis banner
        if result['type_pred'] is not None:
            tname, tcolor, tdesc = TYPE_DESCRIPTIONS.get(
                result['type_pred'],
                (result['type_pred'], '#9ab5a0', ''))
            st.markdown(f"""
            <div class="sensor-card purple" style="margin-bottom:1rem;">
                <div class="sc-feature">Diagnosis — probable anomaly type</div>
                <div class="sc-value" style="color:{tcolor};">{tname}</div>
                <div class="sc-desc">{tdesc}</div>
            </div>""", unsafe_allow_html=True)

            # Probability breakdown
            st.markdown('<div class="section-head">Type Probability Breakdown</div>',
                        unsafe_allow_html=True)
            if result['type_classes'] is not None:
                proba_pairs = sorted(zip(result['type_classes'], result['type_proba']),
                                     key=lambda x: x[1], reverse=True)
                for cls, prob in proba_pairs:
                    lbl, col, _ = TYPE_DESCRIPTIONS.get(cls, (cls, '#9ab5a0',''))
                    pct = int(prob * 100)
                    st.markdown(f"""
                    <div style="margin-bottom:8px;">
                        <div style="display:flex;justify-content:space-between;
                                    font-size:12px;color:#9ab5a0;margin-bottom:3px;">
                            <span style="color:{col};font-weight:500;">{lbl}</span>
                            <span>{pct}%</span>
                        </div>
                        <div class="prob-bar-wrap">
                            <div class="prob-bar-fill"
                                 style="width:{pct}%;background:{col};"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown('<div class="section-head">Sensor-by-Sensor Analysis</div>',
                        unsafe_allow_html=True)
            seg_s      = result['seg_stats']
            issues_map = {iss['feature']: iss for iss in result['param_issues']}
            for f in FEATURES:
                if f not in seg_s: continue
                val  = user_data[f]; low = seg_s[f]['low']; high = seg_s[f]['high']
                unit = FEATURE_UNITS.get(f,''); label = FEATURE_LABELS.get(f,f)
                if f in issues_map:
                    iss    = issues_map[f]; status = iss['status']
                    icon   = '🔴' if status=='HIGH' else '🔵'
                    shtml  = (f'<span style="color:#ef5350;font-weight:600;">'
                              f'{icon} {status} — outside normal range '
                              f'({low:.2f}–{high:.2f}{unit})</span>')
                    desc   = feature_plain_english(f,val,status,low,high,crop_sel,region_sel)
                    cls    = 'anomaly'
                else:
                    shtml  = (f'<span style="color:#4caf72;font-weight:600;">'
                              f'✓ Normal — within range ({low:.2f}–{high:.2f}{unit})</span>')
                    desc   = f"Reading of {val:.2f}{unit} is within the expected range for {crop_sel} in {region_sel}. No action needed."
                    cls    = 'ok'
                st.markdown(f"""
                <div class="sensor-card {cls}">
                    <div class="sc-feature">{label}</div>
                    <div class="sc-value">{val:.2f}{unit}</div>
                    <div class="sc-status">{shtml}</div>
                    <div class="sc-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-head">Sensor Consistency Analysis</div>',
                        unsafe_allow_html=True)
            fd = result['fusion_detail']
            if fd:
                for pair, z in fd.items():
                    if   z > 2.0: card_cls='anomaly'; zh=f'<span style="color:#ef5350;font-weight:600;">⚠ Inconsistent — {z:.1f}σ above expected</span>'
                    elif z > 1.0: card_cls='warning'; zh=f'<span style="color:#f5a623;font-weight:600;">⚡ Slight deviation — {z:.1f}σ</span>'
                    else:         card_cls='ok';      zh=f'<span style="color:#4caf72;font-weight:600;">✓ Consistent — {z:.1f}σ (normal)</span>'
                    desc = fusion_plain_english(pair, z, crop_sel, region_sel)
                    st.markdown(f"""
                    <div class="sensor-card {card_cls}">
                        <div class="sc-feature">{pair.replace('_',' ').replace('%','')}</div>
                        <div class="sc-status" style="margin-top:4px;">{zh}</div>
                        <div class="sc-desc" style="margin-top:6px;">{desc}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Sensor fusion model not available for this segment.")

            # [GEO] Geospatial context
            if 'latitude' in user_data and cluster_summary is not None:
                st.markdown('<div class="section-head">Geospatial Context</div>',
                            unsafe_allow_html=True)
                lat, lon = user_data['latitude'], user_data['longitude']
                st.markdown(f"""
                <div class="sensor-card ok">
                    <div class="sc-feature">Farm location</div>
                    <div class="sc-value" style="font-size:1rem;">
                        {lat:.4f}°N, {lon:.4f}°E
                    </div>
                    <div class="sc-desc">
                        Location recorded. Geospatial cluster analysis below shows
                        regional anomaly patterns from training data.
                    </div>
                </div>""", unsafe_allow_html=True)

                # Show cluster summary table
                st.markdown("Regional anomaly cluster summary from training data:")
                st.dataframe(
                    cluster_summary[['Cluster','Farms','Anomaly %','Lat centre','Lon centre']],
                    use_container_width=True, hide_index=True)

        with right_col:
            # Yield gauge
            st.markdown('<div class="section-head">Predicted Crop Yield</div>',
                        unsafe_allow_html=True)
            pred_yield = result['predicted_yield']
            crop_enc   = result['crop_enc']
            crop_ref   = _base_yields.get(crop_sel, 4033)
            yield_min  = int(crop_ref * 0.40); yield_max = int(crop_ref * 1.35)

            fig_yield = go.Figure(go.Indicator(
                mode="number+delta+gauge",
                value=round(pred_yield, 0),
                number={'suffix':' kg/ha','font':{'color':'#4caf72','size':28,'family':'DM Mono'}},
                delta={'reference':crop_ref,'increasing':{'color':'#4caf72'},
                       'decreasing':{'color':'#ef5350'},'font':{'size':14}},
                gauge={'axis':{'range':[yield_min,yield_max],'tickcolor':'#4a6650',
                               'tickfont':{'color':'#6b8f73','size':9}},
                       'bar':{'color':'#4caf72','thickness':0.3},
                       'bgcolor':'#161d1a','bordercolor':'#1e3a28',
                       'steps':[
                           {'range':[yield_min,int(crop_ref*0.70)],'color':'#2d0f0f'},
                           {'range':[int(crop_ref*0.70),int(crop_ref*1.05)],'color':'#0a2615'},
                           {'range':[int(crop_ref*1.05),yield_max],'color':'#0d3320'}],
                       'threshold':{'line':{'color':'#ffffff','width':2},
                                    'thickness':0.8,'value':crop_ref}}))
            fig_yield.update_layout(height=240,margin=dict(l=20,r=20,t=10,b=10),
                                    paper_bgcolor='rgba(0,0,0,0)',font={'color':'#e8f5e3'})
            st.plotly_chart(fig_yield, use_container_width=True)

            diff = pred_yield - crop_ref
            diff_pct = abs(diff)/crop_ref*100
            diff_txt = f"{'above' if diff>=0 else 'below'} the {crop_sel} average of {crop_ref:,} kg/ha"
            yld_col  = '#4caf72' if diff >= 0 else '#ef5350'
            yld_card = 'ok' if diff >= 0 else 'anomaly'
            st.markdown(f"""
            <div class="sensor-card {yld_card}" style="margin-top:0;">
                <div class="sc-desc">
                    Predicted yield is <strong style="color:{yld_col};">{pred_yield:.0f} kg/ha</strong>
                    — {diff_pct:.0f}% ({abs(diff):.0f} kg/ha) {diff_txt}.
                    {'Good conditions support a strong harvest.' if diff>=0
                     else 'The detected anomalies are likely contributing to this yield reduction.'}
                </div>
            </div>""", unsafe_allow_html=True)

            # Recommendations
            st.markdown('<div class="section-head">Recommended Actions</div>',
                        unsafe_allow_html=True)
            recs = generate_recommendations(
                result['param_issues'], result['fusion_detail'],
                score, crop_sel, region_sel, pred_yield, result['type_pred'])
            badge_map = {
                'urgent': ('rec-card urgent','🚨 Urgent Action'),
                'caution':('rec-card caution','⚡ Caution'),
                'info':   ('rec-card info','ℹ Info'),
            }
            for i, rec in enumerate(recs, 1):
                cls, badge = badge_map.get(rec['type'],('rec-card info','ℹ Info'))
                st.markdown(f"""
                <div class="{cls}">
                    <div class="rc-num">{badge} &nbsp;·&nbsp; Action {i} of {len(recs)}</div>
                    <div class="rc-title">{rec['title']}</div>
                    <div class="rc-desc">{rec['desc']}</div>
                </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 1rem;">
        <div style="font-size:5rem;">🌾</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.3rem;
                    color:#6b8f73; margin-top:1rem; line-height:1.8;">
            Select your crop type and region,<br>
            enter sensor readings in the sidebar,<br>
            then click<br>
            <strong style="color:#4caf72; font-size:1.1rem;">
                🔍 Analyse Farm Conditions
            </strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
