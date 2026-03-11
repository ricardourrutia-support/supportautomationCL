import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Cabify Support Dashboard", layout="wide")
st.title("🚕 Cabify Support Dashboard - Weekly Metrics")

@st.cache_data
def load_and_clean_data(filepath):
    # 1. Cargar datos
    df = pd.read_csv(filepath, delimiter=';')
    
    # 2. Mapear audiencias
    df['Audience'] = df['Audience'].replace({'Private': 'Rider', 'C4B': 'B2B', 'Driver': 'Driver'})
    df = df[df['Audience'].isin(['Rider', 'B2B', 'Driver'])]
    
    # 3. Fechas y Semanas
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y', errors='coerce')
    df['Week'] = df['Date_Time'].dt.isocalendar().week
    
    # 4. Limpieza de números (comas por puntos)
    def parse_num(s):
        if pd.isna(s): return np.nan
        s = str(s).strip().replace('%', '')
        if '.' in s and ',' in s:
            s = s.replace('.', '').replace(',', '.')
        elif ',' in s:
            s = s.replace(',', '.')
        try:
            return float(s)
        except:
            return np.nan

    cols_to_clean = ['User # DO (Total)', '% CSAT', '# First Reply Time (Hours) ', 
                     '# Full Resolution Time (Hours)', '#\xa0Tickets con reopen', 'NPS Score']
    for col in cols_to_clean:
        df[col] = df[col].apply(parse_num)

    # 5. Agrupación por métricas clave
    def aggs(grp):
        res = {}
        res['Contactos Recibidos'] = len(grp)
        res['Contactos Ticket'] = len(grp[grp['Contact Type'] == 'Ticket'])
        res['NPS'] = grp['NPS Score'].mean()
        
        csat = grp['% CSAT'].mean()
        res['CSAT (%)'] = csat * 100 if pd.notna(csat) and csat <= 1.0 else csat
            
        valid_frt = grp['# First Reply Time (Hours) '].dropna()
        res['SLA 1ra Respuesta <24h (%)'] = (valid_frt < 24).mean() * 100 if len(valid_frt) > 0 else np.nan
        
        valid_res = grp['# Full Resolution Time (Hours)'].dropna()
        res['SLA Resolucion <36h (%)'] = (valid_res < 36).mean() * 100 if len(valid_res) > 0 else np.nan
        
        return pd.Series(res)

    weekly_stats = df.groupby(['Week', 'Audience']).apply(aggs).reset_index()
    return weekly_stats

# --- Interfaz de Usuario (UI) ---

# Sube el archivo CSV
uploaded_file = st.file_uploader("Sube el archivo 'Key Audience_Datos completos_data.csv'", type=['csv'])

if uploaded_file is not None:
    # Procesar datos
    df_metrics = load_and_clean_data(uploaded_file)
    
    # Filtro de Audiencia
    audiences = df_metrics['Audience'].unique()
    selected_audience = st.selectbox("Selecciona la Audiencia", audiences)
    
    # Filtrar DF
    df_filtered = df_metrics[df_metrics['Audience'] == selected_audience]
    
    # Mostrar KPIs de la última semana disponible
    latest_week = df_filtered['Week'].max()
    latest_data = df_filtered[df_filtered['Week'] == latest_week].iloc[0]
    
    st.subheader(f"Métricas Resumen - Semana {latest_week}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Contactos Recibidos", f"{latest_data['Contactos Recibidos']:,.0f}")
    col2.metric("NPS Score", f"{latest_data['NPS']:.1f}")
    col3.metric("CSAT", f"{latest_data['CSAT (%)']:.1f}%")
    col4.metric("SLA 1ra Respuesta", f"{latest_data['SLA 1ra Respuesta <24h (%)']:.1f}%")
    
    st.divider()
    
    # Gráficos de Tendencia
    st.subheader("Tendencia Semanal")
    
    fig_vol = px.line(df_filtered, x='Week', y='Contactos Recibidos', markers=True, title="Volumen de Contactos")
    st.plotly_chart(fig_vol, use_container_width=True)
    
    fig_nps = px.line(df_filtered, x='Week', y='NPS', markers=True, title="Evolución del NPS")
    st.plotly_chart(fig_nps, use_container_width=True)
    
    # Tabla de datos crudos (opcional)
    with st.expander("Ver tabla de datos detallada"):
        st.dataframe(df_filtered)
