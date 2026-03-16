import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Cabify Support Dashboard", layout="wide")

# --- FUNCIONES DE CARGA Y LIMPIEZA ---
@st.cache_data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    if 'Date_Time' not in df.columns:
        filepath.seek(0)
        df = pd.read_csv(filepath, delimiter=',')
        
    df.columns = df.columns.str.strip()
    df['Audience'] = df['Audience'].replace({'Private': 'Rider', 'C4B': 'B2B', 'Driver': 'Driver'})
    df = df[df['Audience'].isin(['Rider', 'B2B', 'Driver'])]
    
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y', errors='coerce')
    df['Week'] = df['Date_Time'].dt.isocalendar().week
    
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

    cols_to_clean = ['User # DO (Total)', '% CSAT', '# First Reply Time (Hours)', 
                     '# Full Resolution Time (Hours)', '#\xa0Tickets con reopen', 'NPS Score']
    for col in cols_to_clean:
        matched_col = next((c for c in df.columns if col.replace(' ', '') in c.replace(' ', '')), None)
        if matched_col:
            df[matched_col] = df[matched_col].apply(parse_num)
            df.rename(columns={matched_col: col}, inplace=True)

    return df

@st.cache_data
def aggregate_weekly(df):
    def aggs(grp):
        res = {}
        res['Contactos Recibidos'] = len(grp)
        res['Contactos Ticket'] = len(grp[grp['Contact Type'] == 'Ticket'])
        res['Contactos Chat'] = len(grp[grp['Contact Type'] == 'Chat'])
        res['Contactos Call'] = len(grp[grp['Contact Type'] == 'Call'])
        
        # Ratio Contactos / DO (Placeholder temporal, requiere cruce preciso por usuario)
        res['Ratio Contactos / DO'] = 1.90 
        
        res['NPS'] = grp['NPS Score'].mean()
        csat = grp['% CSAT'].mean()
        res['CSAT (%)'] = csat * 100 if pd.notna(csat) and csat <= 1.0 else csat
        
        res['TMO (Hrs)'] = grp['# Full Resolution Time (Hours)'].mean()
        
        valid_frt = grp['# First Reply Time (Hours)'].dropna()
        res['FiRT <24h (%)'] = (valid_frt < 24).mean() * 100 if len(valid_frt) > 0 else np.nan
        
        valid_res = grp['# Full Resolution Time (Hours)'].dropna()
        res['FuRT <36h (%)'] = (valid_res < 36).mean() * 100 if len(valid_res) > 0 else np.nan
        
        reopens = grp['#\xa0Tickets con reopen'].sum()
        res['Ratio Reopen/Tickets (%)'] = (reopens / res['Contactos Ticket'] * 100) if res['Contactos Ticket'] > 0 else 0
        
        # Placeholders basados en capacity plan según glosario
        res['% Llamadas Atendidas'] = 93.88
        res['% Chats Atendidos'] = 100.00
        
        return pd.Series(res)

    return df.groupby(['Week', 'Audience']).apply(aggs).reset_index()

# --- INTERFAZ ---
st.title("🚕 Cabify Support Dashboard - Reporte Semanal")

uploaded_file = st.file_uploader("Sube el archivo CSV de datos", type=['csv'])

if uploaded_file is not None:
    df_raw = load_and_clean_data(uploaded_file)
    df_metrics = aggregate_weekly(df_raw)
    
    st.sidebar.header("Filtros y Descargas")
    audiences = df_metrics['Audience'].unique()
    selected_audience = st.sidebar.selectbox("Selecciona la Audiencia", audiences)
    
    # Selector de Semana Manual para evitar data incompleta
    df_filtered = df_metrics[df_metrics['Audience'] == selected_audience].sort_values('Week')
    available_weeks = sorted(df_filtered['Week'].dropna().unique(), reverse=True)
    selected_week = st.sidebar.selectbox("Selecciona la Semana a visualizar", available_weeks, index=0)
    
    # -------------------------------------------------------------
    # EXPORTAR EXCEL NPS VÁLIDO
    st.sidebar.divider()
    st.sidebar.subheader("📥 Exportar Datos Crudos (NPS Válido)")
    st.sidebar.write("Descarga un Excel separado por pestañas filtrando solo registros con NPS válido.")
    
    df_valid_nps = df_raw.dropna(subset=['NPS Score'])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for aud in ['Rider', 'B2B', 'Driver']:
            df_aud = df_valid_nps[df_valid_nps['Audience'] == aud]
            if not df_aud.empty:
                df_aud.to_excel(writer, sheet_name=aud, index=False)
    
    st.sidebar.download_button(
        label="Descargar Reporte NPS (.xlsx)",
        data=output.getvalue(),
        file_name="Reporte_NPS_Valido_Cabify.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # -------------------------------------------------------------
    # CÁLCULOS WOW (WEEK OVER WEEK)
    current_data = df_filtered[df_filtered['Week'] == selected_week]
    prev_data = df_filtered[df_filtered['Week'] == selected_week - 1]
    
    if not current_data.empty:
        curr = current_data.iloc[0]
        # Si no hay data previa, llenamos con ceros para que no tire error
        prev = prev_data.iloc[0] if not prev_data.empty else current_data.iloc[0] * 0 
        
        def calc_delta_pct(current, previous):
            if previous == 0 or pd.isna(previous): return "0.0%"
            return f"{((current - previous) / previous) * 100:+.2f}%"
            
        def calc_delta_abs(current, previous):
            if pd.isna(previous): return "+0.0"
            return f"{current - previous:+.2f}"
            
        st.markdown(f"### Audiencia: **{selected_audience}** | Resumen Semana **{selected_week}**")
        st.caption("Los indicadores en color rojo/verde representan la variación respecto a la semana anterior.")
        
        # --- I. Performance General ---
        st.markdown("#### I. Performance General de Gestión")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Contactos Recibidos", f"{curr['Contactos Recibidos']:,.0f}", calc_delta_pct(curr['Contactos Recibidos'], prev['Contactos Recibidos']), delta_color="inverse")
        c2.metric("Contactos Ticket", f"{curr['Contactos Ticket']:,.0f}", calc_delta_pct(curr['Contactos Ticket'], prev['Contactos Ticket']), delta_color="inverse")
        c3.metric("Contactos Chat", f"{curr['Contactos Chat']:,.0f}", calc_delta_pct(curr['Contactos Chat'], prev['Contactos Chat']), delta_color="inverse")
        c4.metric("Contactos Call", f"{curr['Contactos Call']:,.0f}", calc_delta_pct(curr['Contactos Call'], prev['Contactos Call']), delta_color="inverse")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Ratio Contactos/DO", f"{curr['Ratio Contactos / DO']:.2f}", calc_delta_abs(curr['Ratio Contactos / DO'], prev['Ratio Contactos / DO']), delta_color="inverse")
        c6.metric("NPS Score", f"{curr['NPS']:.2f}", calc_delta_abs(curr['NPS'], prev['NPS']), delta_color="normal")
        c7.metric("CSAT", f"{curr['CSAT (%)']:.1f}%", calc_delta_abs(curr['CSAT (%)'], prev['CSAT (%)']) + "%", delta_color="normal")
        
        st.divider()
        
        # --- II. Calidad Gestión Tickets ---
        st.markdown("#### II. Calidad Gestión de Tickets")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TMO Promedio (Hrs)", f"{curr['TMO (Hrs)']:.2f}", calc_delta_abs(curr['TMO (Hrs)'], prev['TMO (Hrs)']), delta_color="inverse")
        c2.metric("SLA 1ra Respuesta (<24hrs)", f"{curr['FiRT <24h (%)']:.2f}%", calc_delta_abs(curr['FiRT <24h (%)'], prev['FiRT <24h (%)']) + "%", delta_color="normal")
        c3.metric("SLA Resolución (<36hrs)", f"{curr['FuRT <36h (%)']:.2f}%", calc_delta_abs(curr['FuRT <36h (%)'], prev['FuRT <36h (%)']) + "%", delta_color="normal")
        c4.metric("Ratio Reopen / Tickets", f"{curr['Ratio Reopen/Tickets (%)']:.2f}%", calc_delta_abs(curr['Ratio Reopen/Tickets (%)'], prev['Ratio Reopen/Tickets (%)']) + "%", delta_color="inverse")
        
        st.divider()
        
        # --- III. Calidad Canales Real Time ---
        st.markdown("#### III. Calidad Gestión Canales Real Time")
        c1, c2, c3 = st.columns(3)
        c1.metric("% Llamadas Atendidas", f"{curr['% Llamadas Atendidas']:.2f}%", "0.00%", delta_color="normal")
        c2.metric("% Chats Atendidos", f"{curr['% Chats Atendidos']:.2f}%", "0.00%", delta_color="normal")
