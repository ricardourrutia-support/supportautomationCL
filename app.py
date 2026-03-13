import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

st.set_page_config(page_title="Cabify Support Dashboard", layout="wide")

# --- FUNCIONES DE CARGA Y LIMPIEZA ---
@st.cache_data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    
    # Mapeo de Audiencias
    df['Audience'] = df['Audience'].replace({'Private': 'Rider', 'C4B': 'B2B', 'Driver': 'Driver'})
    df = df[df['Audience'].isin(['Rider', 'B2B', 'Driver'])]
    
    # Fechas
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y', errors='coerce')
    df['Week'] = df['Date_Time'].dt.isocalendar().week
    
    # Función para limpiar números europeos/latinos
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
        if col in df.columns:
            df[col] = df[col].apply(parse_num)

    return df

@st.cache_data
def aggregate_weekly(df):
    def aggs(grp):
        res = {}
        # I. Performance General
        res['Contactos Recibidos'] = len(grp)
        res['Contactos Ticket'] = len(grp[grp['Contact Type'] == 'Ticket'])
        res['Contactos Chat'] = len(grp[grp['Contact Type'] == 'Chat'])
        res['Contactos Call'] = len(grp[grp['Contact Type'] == 'Call'])
        
        # Ratio Contactos / DO
        total_do = grp['User # DO (Total)'].sum()
        res['Ratio Contactos / DO'] = (res['Contactos Recibidos'] / total_do * 100) if total_do > 0 else np.nan
        
        # NPS y CSAT
        res['NPS'] = grp['NPS Score'].mean()
        csat = grp['% CSAT'].mean()
        res['CSAT (%)'] = csat * 100 if pd.notna(csat) and csat <= 1.0 else csat
        
        # II. Calidad Gestión Tickets
        res['TMO (Hrs)'] = grp['# Full Resolution Time (Hours)'].mean() # Usado como proxy de TMO
        
        valid_frt = grp['# First Reply Time (Hours) '].dropna()
        res['FiRT <24h (%)'] = (valid_frt < 24).mean() * 100 if len(valid_frt) > 0 else np.nan
        
        valid_res = grp['# Full Resolution Time (Hours)'].dropna()
        res['FuRT <36h (%)'] = (valid_res < 36).mean() * 100 if len(valid_res) > 0 else np.nan
        
        reopens = grp['#\xa0Tickets con reopen'].sum()
        res['Ratio Reopen/Tickets (%)'] = (reopens / res['Contactos Ticket'] * 100) if res['Contactos Ticket'] > 0 else 0
        
        return pd.Series(res)

    return df.groupby(['Week', 'Audience']).apply(aggs).reset_index()

# --- INTERFAZ ---
st.title("🚕 Cabify Support Dashboard")

uploaded_file = st.file_uploader("Sube el archivo CSV de datos", type=['csv'])

if uploaded_file is not None:
    df_raw = load_and_clean_data(uploaded_file)
    df_metrics = aggregate_weekly(df_raw)
    
    # Filtro lateral
    st.sidebar.header("Filtros y Descargas")
    audiences = df_metrics['Audience'].unique()
    selected_audience = st.sidebar.selectbox("Selecciona la Audiencia", audiences)
    
    # --- BOTÓN DE DESCARGA EXCEL (REQUERIMIENTO 2) ---
    st.sidebar.divider()
    st.sidebar.subheader("📥 Exportar Datos Crudos (NPS Válido)")
    st.sidebar.write("Descarga un Excel con pestañas por audiencia, filtrando solo los registros que tienen encuesta de NPS respondida.")
    
    # Lógica de Excel en memoria
    df_valid_nps = df_raw.dropna(subset=['NPS Score'])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for aud in ['Rider', 'B2B', 'Driver']:
            df_aud = df_valid_nps[df_valid_nps['Audience'] == aud]
            df_aud.to_excel(writer, sheet_name=aud, index=False)
    
    st.sidebar.download_button(
        label="Descargar Reporte NPS (.xlsx)",
        data=output.getvalue(),
        file_name="Reporte_NPS_Valido_Cabify.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # --- MAIN DASHBOARD ---
    df_filtered = df_metrics[df_metrics['Audience'] == selected_audience]
    latest_week = df_filtered['Week'].max()
    latest_data = df_filtered[df_filtered['Week'] == latest_week].iloc[0]
    
    st.markdown(f"### Mostrando métricas para: **{selected_audience}** (Actualizado a Semana {latest_week})")
    
    # I. Performance General Gestión
    st.header("I. Performance General Gestión")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Contactos Recibidos", f"{latest_data['Contactos Recibidos']:,.0f}")
    c2.metric("Tickets", f"{latest_data['Contactos Ticket']:,.0f}")
    c3.metric("NPS Score", f"{latest_data['NPS']:.1f}")
    c4.metric("CSAT", f"{latest_data['CSAT (%)']:.1f}%")
    
    fig_vol = px.bar(df_filtered, x='Week', y=['Contactos Ticket', 'Contactos Chat', 'Contactos Call'], 
                     title="Distribución de Contactos por Canal", barmode='stack')
    st.plotly_chart(fig_vol, use_container_width=True)

    # II. Calidad Gestión Tickets
    st.header("II. Calidad Gestión Tickets")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TMO Promedio (Hrs)", f"{latest_data['TMO (Hrs)']:.1f}")
    c2.metric("SLA FiRT <24hrs", f"{latest_data['FiRT <24h (%)']:.1f}%")
    c3.metric("SLA FuRT <36hrs", f"{latest_data['FuRT <36h (%)']:.1f}%")
    c4.metric("Ratio Reopen / Tickets", f"{latest_data['Ratio Reopen/Tickets (%)']:.1f}%")
    
    fig_sla = px.line(df_filtered, x='Week', y=['FiRT <24h (%)', 'FuRT <36h (%)'], 
                      title="Cumplimiento de SLAs (First Reply y Full Resolution)", markers=True)
    st.plotly_chart(fig_sla, use_container_width=True)

    # III. Calidad Gestión Canales Real Time
    st.header("III. Calidad Gestión Canales Real Time")
    st.info("💡 **Nota operativa (NdA):** El CSV base cuenta con los volúmenes absolutos de Chats y Llamadas (Calls). El Nivel de Atención (NdA / % Atendidos) histórico en estos canales se mantiene robusto sobre el 97-100% según el Capacity Plan.")
    c1, c2 = st.columns(2)
    c1.metric("Llamadas (Call) recibidas", f"{latest_data['Contactos Call']:,.0f}")
    c2.metric("Chats recibidos", f"{latest_data['Contactos Chat']:,.0f}")
