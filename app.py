import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Load model ────────────────────────────────
@st.cache_resource
def load_model():
    with open('models.pkl', 'rb') as f:
        return pickle.load(f)

data         = load_model()
best_xgb     = data['best_xgb']
best_lgb     = data['best_lgb']
best_cat     = data['best_cat']
FEATURES     = data['FEATURES']
district_map = data['district_map']
df_stats     = data['df_stats']

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title = 'Dự đoán giá căn hộ TPHCM',
    page_icon  = '🏠',
    layout     = 'wide'
)

# ── CSS custom ────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 20px; border-radius: 10px;
    color: white; text-align: center; margin-bottom: 20px;
}
.metric-card {
    background: #f8f9fa; border-radius: 10px;
    padding: 15px; border-left: 4px solid #2a5298;
}
.predict-box {
    background: linear-gradient(135deg, #2a5298, #1e3c72);
    border-radius: 15px; padding: 25px;
    color: white; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏠 Dự đoán giá căn hộ TP.HCM</h1>
    <p>XGBoost + LightGBM + CatBoost Ensemble | R² = 0.8963 | MAPE = 13.76%</p>
    <p>Dữ liệu: 11,107 căn hộ từ batdongsan.com.vn</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────
st.sidebar.image('https://img.icons8.com/color/96/000000/real-estate.png', width=80)
st.sidebar.header('Nhập thông tin căn hộ')

district_name = st.sidebar.selectbox(
    '📍 Quận/Huyện',
    options=sorted(district_map.values()),
    index=5
)
area_m2  = st.sidebar.slider('📐 Diện tích (m²)', 20, 300, 70, step=5)
bedroom  = st.sidebar.selectbox('🛏️ Số phòng ngủ', [1,2,3,4,5], index=1)
toilet   = st.sidebar.selectbox('🚿 Số WC', [1,2,3,4], index=1)

st.sidebar.subheader('✨ Đặc điểm khác')
has_furniture = st.sidebar.checkbox('🛋️ Nội thất đầy đủ')
is_corner     = st.sidebar.checkbox('📐 Căn góc')
has_view      = st.sidebar.checkbox('🌊 View đẹp (sông/hồ/CV)')
is_vinhomes   = st.sidebar.checkbox('🏙️ Dự án Vinhomes')
is_masteri    = st.sidebar.checkbox('🏢 Dự án Masteri')

# ── Predict function ──────────────────────────
def predict_price(area_m2, bedroom, toilet, district_name,
                  has_furniture, is_corner, has_view,
                  is_vinhomes, is_masteri):

    dist_id = {v:k for k,v in district_map.items()}.get(district_name, 61)
    d_med   = df_stats['median'].get(dist_id, 5e9)
    d_mean  = df_stats['mean'].get(dist_id, 6e9)
    d_std   = df_stats['std'].get(dist_id, 2e9)
    d_cnt   = df_stats['count'].get(dist_id, 100)

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

    p_xgb = best_xgb.predict(sample[FEATURES])[0]
    p_lgb = best_lgb.predict(sample[FEATURES])[0]
    p_cat = best_cat.predict(sample[FEATURES])[0]
    return np.expm1(0.4*p_xgb + 0.4*p_lgb + 0.2*p_cat)

# ── Main predict ──────────────────────────────
price = predict_price(area_m2, bedroom, toilet, district_name,
                      has_furniture, is_corner, has_view,
                      is_vinhomes, is_masteri)

low  = price * 0.86
high = price * 1.14

# ── Kết quả chính ─────────────────────────────
st.markdown(f"""
<div class="predict-box">
    <h2>💰 Giá dự đoán</h2>
    <h1>{price/1e9:.2f} tỷ đồng</h1>
    <p>Khoảng: {low/1e9:.2f} — {high/1e9:.2f} tỷ (±14%)</p>
    <p>Giá/m²: {price/area_m2/1e6:.1f} triệu/m²</p>
</div>
""", unsafe_allow_html=True)

st.write('')

# ── 3 metrics ─────────────────────────────────
col1, col2, col3 = st.columns(3)
dist_id = {v:k for k,v in district_map.items()}.get(district_name, 61)
d_med   = df_stats['median'].get(dist_id, 5e9)
pct     = (price / d_med - 1) * 100
delta_label = f'{pct:+.1f}% so với median {district_name}'

with col1:
    st.metric('💰 Giá dự đoán',
              f'{price/1e9:.2f} tỷ',
              delta_label)
with col2:
    st.metric('📐 Giá/m²',
              f'{price/area_m2/1e6:.1f} triệu/m²')
with col3:
    rank = '🔴 Cao hơn TB' if pct > 10 else ('🟢 Thấp hơn TB' if pct < -10 else '🟡 Bằng TB')
    st.metric('📊 So với thị trường', rank)

st.divider()

# ── 2 columns charts ──────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader('📊 Khoảng giá dự đoán')
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.barh([''], [high-low], left=[low],
            color='#2a5298', alpha=0.5, height=0.4)
    ax.axvline(price, color='red', lw=2.5,
               label=f'Dự đoán: {price/1e9:.2f} tỷ')
    ax.axvline(d_med, color='green', lw=1.5, linestyle='--',
               label=f'Median quận: {d_med/1e9:.2f} tỷ')
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x,_: f'{x/1e9:.1f}tỷ'))
    ax.legend(fontsize=9)
    ax.set_title(f'{low/1e9:.2f} — {high/1e9:.2f} tỷ')
    ax.set_yticks([])
    st.pyplot(fig)
    plt.close()

with col_r:
    st.subheader('🗺️ Giá median các quận')
    dist_prices = {
        district_map[k]: v/1e9
        for k,v in df_stats['median'].items()
        if k in district_map
    }
    dist_df = pd.Series(dist_prices).sort_values()
    colors  = ['#e74c3c' if d == district_name
               else '#2a5298' for d in dist_df.index]
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    dist_df.plot(kind='barh', ax=ax2, color=colors)
    ax2.set_xlabel('Tỷ đồng')
    ax2.set_title('Đỏ = quận đang chọn')
    st.pyplot(fig2)
    plt.close()

st.divider()

# ── Model info ────────────────────────────────
st.subheader('📈 Thông tin mô hình')
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric('R² Score',  '0.8963')
col_b.metric('MAPE',      '13.76%')
col_c.metric('MAE',       '0.985 tỷ')
col_d.metric('Dataset',   '11,107 căn hộ')

st.caption('📌 Nguồn: batdongsan.com.vn | Model: XGBoost+LightGBM+CatBoost Ensemble')
