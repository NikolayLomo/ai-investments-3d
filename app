import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import io
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI-Инвестиции РФ: 3D", layout="wide")

@st.cache_data
def load_data():
    csv_data = """Year,AI_Invest_Total_Bln_RUB,AI_Salary_Share_Pct,IT_Grads_K,Grants_Volume_Bln,Avg_Salary_RU_K,Deposits_Volume_Bln,Stock_Private_Inv_Bln,M2_Growth_Pct,AI_ROI_Pct,Fear_Index,Oil_Price_USD,Compute_PetaFLOPS,IT_Migration_Netto_K,RTS_Index
2018,45.0,68.5,32.5,12.0,42.5,24500,350,10.5,18.5,22,65.0,120,-15,1100
2019,58.0,70.2,35.1,15.5,47.0,26800,410,12.1,21.0,25,64.0,180,-12,1180
2020,75.0,72.5,38.4,22.0,51.5,29500,480,25.4,15.0,45,42.0,250,-25,1450
2021,95.0,74.0,42.0,28.5,58.0,31200,650,18.2,24.5,28,71.0,350,-30,1650
2022,110.0,76.5,45.5,35.0,65.0,38500,210,25.0,12.0,65,80.0,380,-45,1150
2023,135.0,73.0,51.2,42.0,74.5,42000,380,20.5,19.5,35,75.0,450,-20,1050
2024,165.0,71.5,56.8,51.5,85.0,46500,520,15.0,22.0,30,82.0,580,-10,1120
2025,195.0,69.0,62.5,60.0,96.0,51000,680,12.5,25.5,26,78.0,750,-5,1210
2026,230.0,67.5,68.0,72.0,108.0,55500,850,10.0,28.0,24,75.0,950,0,1280"""
    return pd.read_csv(io.StringIO(csv_data))

df = load_data()
FEATURE_NAMES = ['AI_Salary_Share_Pct', 'IT_Grads_K', 'Grants_Volume_Bln', 'Avg_Salary_RU_K', 'M2_Growth_Pct', 'Fear_Index', 'Oil_Price_USD', 'Compute_PetaFLOPS', 'IT_Migration_Netto_K', 'Deposits_Volume_Bln', 'Stock_Private_Inv_Bln']

@st.cache_resource
def train_model():
    X = df[FEATURE_NAMES].values
    return RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42).fit(X, df['RTS_Index'].values)

model = train_model()

st.title("🎯 AI-Инвестиции РФ: 3D Генеративный дизайн")

col1, col2, col3 = st.columns(3)
with col1:
    target_rts = st.number_input("Целевой РТС:", min_value=800, max_value=2000, value=1400, step=50)
with col2:
    max_salary = st.slider("Макс. доля зарплат (%):", 50, 90, 70)
with col3:
    num_samples = st.slider("Вариантов:", 500, 2000, 1000, step=100)

if st.button("🚀 Сгенерировать 3D-модель", type="primary"):
    with st.spinner(f"Генерация {num_samples} сценариев..."):
        solutions = []
        for _ in range(num_samples):
            cand = {
                'AI_Salary_Share_Pct': np.random.uniform(50.0, 90.0),
                'IT_Grads_K': np.random.uniform(30.0, 120.0),
                'Grants_Volume_Bln': np.random.uniform(10.0, 150.0),
                'Avg_Salary_RU_K': np.random.uniform(40.0, 200.0),
                'M2_Growth_Pct': np.random.uniform(5.0, 35.0),
                'Fear_Index': np.random.uniform(10.0, 80.0),
                'Oil_Price_USD': np.random.uniform(40.0, 120.0),
                'Compute_PetaFLOPS': np.random.uniform(100.0, 2000.0),
                'IT_Migration_Netto_K': np.random.uniform(-60.0, 20.0),
                'Deposits_Volume_Bln': np.random.uniform(20000.0, 80000.0),
                'Stock_Private_Inv_Bln': np.random.uniform(100.0, 1500.0)
            }
            if cand['AI_Salary_Share_Pct'] > max_salary:
                continue
            
            X_cand = np.array([[cand[f] for f in FEATURE_NAMES]])
            pred = model.predict(X_cand)[0]
            solutions.append({'pred': pred, 'dev': abs(pred - target_rts), 'params': cand})
        
        solutions.sort(key=lambda x: x['dev'])
        
        if not solutions:
            st.error("Не найдено решений. Увеличьте макс. долю зарплат.")
        else:
            best = solutions[0]
            st.success(f"Найдено {len(solutions)} решений! Лучшее отклонение: {best['dev']:.1f}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Цель", f"{target_rts}")
            c2.metric("Лучший прогноз", f"{best['pred']:.0f}", f"{best['pred']-target_rts:+.0f}")
            c3.metric("Найдено", f"{len(solutions)}")
            
            st.subheader("🌐 3D Визуализация пространства решений")
            st.caption("Оси: Доля зарплат (X) | Выпускники IT (Y) | Прогноз РТС (Z). Цвет: отклонение (зеленый=близко, красный=далеко)")
            
            x_vals = [s['params']['AI_Salary_Share_Pct'] for s in solutions]
            y_vals = [s['params']['IT_Grads_K'] for s in solutions]
            z_vals = [s['pred'] for s in solutions]
            dev_vals = [s['dev'] for s in solutions]
            
            fig = go.Figure()
            
            # Облако всех решений
            fig.add_trace(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals, mode='markers',
                marker=dict(size=4, color=dev_vals, colorscale='RdYlGn', opacity=0.6, colorbar=dict(title='Отклонение'))
            ))
            
            # Топ-3 решения (крупные золотые точки)
            top3 = solutions[:3]
            fig.add_trace(go.Scatter3d(
                x=[s['params']['AI_Salary_Share_Pct'] for s in top3],
                y=[s['params']['IT_Grads_K'] for s in top3],
                z=[s['pred'] for s in top3],
                mode='markers+text',
                marker=dict(size=10, color='gold', symbol='diamond'),
                text=['ТОП-1', 'ТОП-2', 'ТОП-3'],
                textposition='top center',
                textfont=dict(color='black', size=10)
            ))
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='Доля зарплат (%)',
                    yaxis_title='Выпускники IT (тыс.)',
                    zaxis_title='Индекс РТС'
                ),
                height=700,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🏆 Топ-3 решений")
            for idx, sol in enumerate(top3, 1):
                st.write(f"**#{idx}:** РТС = {sol['pred']:.0f}, Отклонение = {sol['dev']:.1f}, Доля зарплат = {sol['params']['AI_Salary_Share_Pct']:.1f}%")
