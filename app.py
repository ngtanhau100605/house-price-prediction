import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load model ────────────────────────────────
@st.cache_resource
def load_model():
    with open('models.pkl', 'rb') as f:
        return pickle.load(f)

data         = load_model()
best_xgb     = data['best_xgb']
FEATURES     = data['FEATURES']
district_map = data['district_map']
df_stats     = data['df_stats']

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title = 'Dự đoán giá căn hộ TPHCM',
    page_icon  = '🏠',
    layout     = 'wide'
)

# ── Header ────────────────────────────────────
st.title('🏠 Dự đoán giá căn hộ TP.HCM')
st.caption('Model: XGBoost | Dataset: 11,107 căn hộ')

# ── Sidebar ───────────────────────────────────
st.sidebar.header('Nhập thông tin căn hộ')

district_name = st.sidebar.selectbox(
    '📍 Quận/Huyện',
    options=sorted(district_map.values()),
    index=5
)

area_m2  = st.sidebar.slider('📐 Diện tích (m²)', 20, 300, 70, step=5)
bedroom  = st.sidebar.selectbox('🛏️ Số phòng ngủ', [1,2,3,4,5], index=1)
toilet   = st.sidebar.selectbox('🚿 Số WC', [1,2,3,4], index=1)

has_furniture = st.sidebar.checkbox('🛋️ Nội thất đầy đủ')
is_corner     = st.sidebar.checkbox('📐 Căn góc')
has_view      = st.sidebar.checkbox('🌊 View đẹp')
is_vinhomes   = st.sidebar.checkbox('🏙️ Vinhomes')
is_masteri    = st.sidebar.checkbox('🏢 Masteri')

# ── Predict ───────────────────────────────────
def predict_price(area_m2, bedroom, toilet, district_name,
                  has_furniture, is_corner, has_view,
                  is_vinhomes, is_masteri):

    dist_id = {v:k for k,v in district_map.items()}.get(district_name, 61)

    d_med = df_stats['median'].get(dist_id, 5e9)
    d_mean = df_stats['mean'].get(dist_id, 6e9)
    d_std = df_stats['std'].get(dist_id, 2e9)
    d_cnt = df_stats['count'].get(dist_id, 100)

    sample = pd.DataFrame([{f: 0 for f in FEATURES}])

    sample['area_m2']       = area_m2
    sample['area_log']      = np.log1p(area_m2)
    sample['bedroom']       = bedroom
    sample['toilet']        = toilet
    sample['area_per_room'] = area_m2 / max(bedroom, 1)
    sample['area_x_bed']    = area_m2 * bedroom
    sample['bed_x_toilet']  = bedroom * toilet
    sample['districtId']    = dist_id

    sample['dist_median']   = d_med
    sample['dist_mean']     = d_mean
    sample['dist_std']      = d_std
    sample['dist_count']    = d_cnt

    sample['ward_median']   = d_med
    sample['ward_count']    = 50

    sample['verified']      = 1
    sample['days_posted']   = 1
    sample['is_new_post']   = 1

    sample['has_furniture'] = int(has_furniture)
    sample['is_corner']     = int(is_corner)
    sample['has_view']      = int(has_view)
    sample['is_vinhomes']   = int(is_vinhomes)
    sample['is_masteri']    = int(is_masteri)

    # 👉 CHỈ dùng XGBoost
    pred_log = best_xgb.predict(sample[FEATURES])[0]
    return np.expm1(pred_log)

# ── Run prediction ────────────────────────────
price = predict_price(area_m2, bedroom, toilet, district_name,
                      has_furniture, is_corner, has_view,
                      is_vinhomes, is_masteri)

low  = price * 0.86
high = price * 1.14

# ── Output ───────────────────────────────────
st.subheader('💰 Giá dự đoán')

st.metric('Giá căn hộ', f'{price/1e9:.2f} tỷ đồng')
st.metric('Giá/m²', f'{price/area_m2/1e6:.1f} triệu/m²')
st.write(f'Khoảng giá: {low/1e9:.2f} — {high/1e9:.2f} tỷ')

# ── So sánh thị trường ───────────────────────
dist_id = {v:k for k,v in district_map.items()}.get(district_name, 61)
d_med   = df_stats['median'].get(dist_id, 5e9)

pct = (price / d_med - 1) * 100

if pct > 10:
    st.error(f'🔴 Cao hơn thị trường ~{pct:.1f}%')
elif pct < -10:
    st.success(f'🟢 Thấp hơn thị trường ~{abs(pct):.1f}%')
else:
    st.warning(f'🟡 Gần mức thị trường ({pct:.1f}%)')

# ── Footer ───────────────────────────────────
st.caption('Demo phục vụ mục đích học tập & nghiên cứu')
