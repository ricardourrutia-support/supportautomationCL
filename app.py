import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from fpdf import FPDF

# Configuración estilo Cabify Minimalista
st.set_page_config(page_title="Cabify Support Dashboard", layout="wide", initial_sidebar_state="expanded")
CABIFY_PURPLE = "#7352FF"
CABIFY_SECONDARY = "#00D1A3"

def parse_num(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().replace('%', '')
    if '.' in s and ',' in s: s = s.replace('.', '').replace(',', '.')
    elif ',' in s: s = s.replace(',', '.')
    try: return float(s)
    except: return np.nan

# --- CARGA DEL REPORTE GENERAL ---
@st.cache_data
def load_main_data(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    if 'Date_Time' not in df.columns:
        filepath.seek(0)
        df = pd.read_csv(filepath, delimiter=',')
        
    df.columns = df.columns.str.strip()
    # SOLUCIÓN AL ERROR: Eliminamos columnas duplicadas inmediatamente
    df = df.loc[:, ~df.columns.duplicated()]
    
    df['Audience'] = df['Audience'].replace({'Private': 'Rider', 'C4B': 'B2B', 'Driver': 'Driver'})
    df = df[df['Audience'].isin(['Rider', 'B2B', 'Driver'])]
    
    mask_emergencias = df['Group name support'].astype(str).str.contains('emergencia', case=False, na=False)
    df_emergencias = df[mask_emergencias].copy()
    df_emergencias['Audience'] = 'Emergencias' 
    df = pd.concat([df, df_emergencias], ignore_index=True)
    
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y', errors='coerce')
    df['Week'] = df['Date_Time'].dt.isocalendar().week
    
    cols_to_clean = ['% CSAT', '# First Reply Time (Hours)', '# Full Resolution Time (Hours)', '#\xa0Tickets con reopen', 'NPS Score']
    for col in cols_to_clean:
        matched_col = next((c for c in df.columns if col.replace(' ', '') in c.replace(' ', '')), None)
        if matched_col:
            df[matched_col] = df[matched_col].apply(parse_num)
            df = df.rename(columns={matched_col: col}) 
    return df

# --- CARGA DEL REPORTE AEROPUERTO ---
@st.cache_data
def load_airport_data(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    if 'Fecha de Referencia' not in df.columns:
        filepath.seek(0)
        df = pd.read_csv(filepath, delimiter=',')
        
    df.columns = df.columns.str.strip()
    # SOLUCIÓN AL ERROR: Eliminamos columnas duplicadas inmediatamente
    df = df.loc[:, ~df.columns.duplicated()]
    
    df['Audience'] = 'Aeropuerto'
    
    df['Date_Time'] = pd.to_datetime(df['Fecha de Referencia'], format='%d/%m/%Y', errors='coerce')
    df['Week'] = df['Date_Time'].dt.isocalendar().week
    
    # Construcción de Contact Type a partir de las variables booleanas
    def determine_type(row):
        if row.get('# Tickets - Call', 0) == 1: return 'Call'
        if row.get('# Tickets - Chats', 0) == 1: return 'Chat'
        return 'Ticket'
    df['Contact Type'] = df.apply(determine_type, axis=1)
    
    # Mapeo de columnas para que encaje con el general
    col_mapping = {
        '# First Reply Time (Hours)': '# First Reply Time (Hours)',
        '# Full Resolution Time (Hours)': '# Full Resolution Time (Hours)',
        '# Tickets Reopen': '#\xa0Tickets con reopen',
        '% CSAT': '% CSAT',
        'NPS Score': 'NPS Score'
    }
    
    for old_col, new_col in col_mapping.items():
        matched_col = next((c for c in df.columns if old_col.replace(' ', '') in c.replace(' ', '')), None)
        if matched_col:
            df[matched_col] = df[matched_col].apply(parse_num)
            df = df.rename(columns={matched_col: new_col})
            
    return df

@st.cache_data
def aggregate_weekly(df):
    def aggs(grp):
        res = {}
        res['Contactos Recibidos'] = len(grp)
        res['Contactos Ticket'] = len(grp[grp['Contact Type'] == 'Ticket'])
        res['Contactos Chat'] = len(grp[grp['Contact Type'] == 'Chat'])
        res['Contactos Call'] = len(grp[grp['Contact Type'] == 'Call'])
        
        if 'NPS Score' in grp.columns: res['NPS'] = grp['NPS Score'].mean()
        else: res['NPS'] = np.nan
            
        if '% CSAT' in grp.columns:
            csat = grp['% CSAT'].mean()
            res['CSAT (%)'] = csat * 100 if pd.notna(csat) and csat <= 1.0 else csat
        else: res['CSAT (%)'] = np.nan
            
        if '# Full Resolution Time (Hours)' in grp.columns:
            res['TMO (Hrs)'] = grp['# Full Resolution Time (Hours)'].mean()
            valid_res = grp['# Full Resolution Time (Hours)'].dropna()
            res['FuRT <36h (%)'] = (valid_res < 36).mean() * 100 if len(valid_res) > 0 else np.nan
        else:
            res['TMO (Hrs)'] = np.nan
            res['FuRT <36h (%)'] = np.nan
            
        if '# First Reply Time (Hours)' in grp.columns:
            valid_frt = grp['# First Reply Time (Hours)'].dropna()
            res['FiRT <24h (%)'] = (valid_frt < 24).mean() * 100 if len(valid_frt) > 0 else np.nan
        else: res['FiRT <24h (%)'] = np.nan
        
        if '#\xa0Tickets con reopen' in grp.columns:
            reopens = grp['#\xa0Tickets con reopen'].sum()
            res['Ratio Reopen/Tickets (%)'] = (reopens / res['Contactos Ticket'] * 100) if res['Contactos Ticket'] > 0 else 0
        else: res['Ratio Reopen/Tickets (%)'] = np.nan
        
        calls = grp[grp['Contact Type'] == 'Call']
        if len(calls) > 0 and 'ES Output Tags 2nd Level v2' in grp.columns:
            res['% Llamadas Atendidas'] = ((len(calls) - calls['ES Output Tags 2nd Level v2'].astype(str).str.contains('talkdesk', case=False).sum()) / len(calls)) * 100
        else: res['% Llamadas Atendidas'] = np.nan
            
        chats = grp[grp['Contact Type'] == 'Chat']
        if len(chats) > 0 and 'Chat Missed' in grp.columns:
            res['% Chats Atendidos'] = ((len(chats) - chats['Chat Missed'].sum()) / len(chats)) * 100
        else: res['% Chats Atendidos'] = np.nan
            
        return pd.Series(res)
    return df.groupby(['Week', 'Audience']).apply(aggs).reset_index()

# --- FUNCIÓN GENERADORA DE PDF ---
def generar_pdf_resumen(df_metrics, df_raw, week):
    pdf = FPDF()
    pdf.add_page()
    
    def clean_txt(text):
        if pd.isna(text): return ""
        text = str(text).replace('"', "'").replace('\n', ' ')
        return text.encode('latin-1', 'replace').decode('latin-1')
        
    def print_metric_line(label, val_str, delta_val, is_higher_better=True, is_pct=False):
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(50, 6, clean_txt(f"- {label}: {val_str}"), ln=0)
        
        if pd.isna(delta_val) or delta_val == 0:
            pdf.set_text_color(150, 150, 150)
            delta_str = "(=0.0)"
        else:
            if delta_val > 0:
                pdf.set_text_color(0, 209, 163) if is_higher_better else pdf.set_text_color(255, 82, 82)
            else:
                pdf.set_text_color(255, 82, 82) if is_higher_better else pdf.set_text_color(0, 209, 163)
                    
            suffix = "% WoW" if is_pct else " WoW"
            delta_str = f"({delta_val:+.1f}{suffix})"
            
        pdf.cell(0, 6, clean_txt(delta_str), ln=1)
        
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(115, 82, 255)
    pdf.cell(0, 10, clean_txt(f"Resumen Ejecutivo C_OPS - Semana {week}"), ln=True, align='C')
    pdf.ln(5)

    audiences_in_week = df_metrics[df_metrics['Week'] == week]['Audience'].unique()
    for aud in ['Driver', 'Rider', 'B2B', 'Emergencias', 'Aeropuerto']:
        if aud not in audiences_in_week: continue
        
        curr_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Week'] == week)]
        prev_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Week'] == week - 1)]

        if not curr_df.empty:
            curr = curr_df.iloc[0]
            prev = prev_df.iloc[0] if not prev_df.empty else curr * 0

            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, clean_txt(f"Audiencia: {aud}"), ln=True)

            vol_pct = ((curr['Contactos Recibidos'] - prev['Contactos Recibidos']) / prev['Contactos Recibidos'] * 100) if prev['Contactos Recibidos'] else 0
            nps_diff = curr['NPS'] - prev['NPS']
            csat_diff = curr['CSAT (%)'] - prev['CSAT (%)']
            firt_diff = curr['FiRT <24h (%)'] - prev['FiRT <24h (%)']
            reop_diff = curr['Ratio Reopen/Tickets (%)'] - prev['Ratio Reopen/Tickets (%)']

            print_metric_line("Volumen", f"{curr['Contactos Recibidos']:,.0f}", vol_pct, is_higher_better=False, is_pct=True)
            print_metric_line("NPS Score", f"{curr['NPS']:.1f}", nps_diff, is_higher_better=True, is_pct=False)
            print_metric_line("CSAT (%)", f"{curr['CSAT (%)']:.1f}%", csat_diff, is_higher_better=True, is_pct=True)
            print_metric_line("FiRT <24h", f"{curr['FiRT <24h (%)']:.1f}%", firt_diff, is_higher_better=True, is_pct=True)
            print_metric_line("Ratio Reopen", f"{curr['Ratio Reopen/Tickets (%)']:.1f}%", reop_diff, is_higher_better=False, is_pct=True)

            if curr['NPS'] < 0:
                detractores = df_raw[(df_raw['Audience'] == aud) & (df_raw['Week'] == week) & (df_raw['NPS Score'] == -100)]
                if not detractores.empty and 'ES Output Tags 3rd Level v2' in detractores.columns:
                    pdf.ln(2)
                    pdf.set_font("Arial", 'B', 9)
                    pdf.set_text_color(255, 82, 82)
                    pdf.cell(0, 5, clean_txt("  [!] ALERTA NPS: Top motivos de detractores (Puntuacion -100):"), ln=True)
                    
                    pdf.set_font("Arial", '', 9)
                    pdf.set_text_color(80, 80, 80)
                    top_tags = detractores['ES Output Tags 3rd Level v2'].value_counts().head(3)
                    for tag, count in top_tags.items():
                        pdf.cell(0, 5, clean_txt(f"      * {tag} ({count} casos)"), ln=True)
                    
                    top_tag_name = top_tags.index[0]
                    sample_desc = detractores[(detractores['ES Output Tags 3rd Level v2'] == top_tag_name) & (detractores['Description'].notna())]
                    if not sample_desc.empty:
                        desc_text = str(sample_desc['Description'].iloc[0])[:120] + "..."
                        pdf.set_font("Arial", 'I', 8)
                        pdf.set_text_color(120, 120, 120)
                        pdf.multi_cell(0, 4, clean_txt(f"      Ej. real: '{desc_text}'"))

            pdf.ln(5)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

    return pdf.output(dest='S').encode('latin-1')

# --- INTERFAZ ---
st.markdown(f"<h1 style='color: {CABIFY_PURPLE};'>🚕 C_OPS Support Dashboard</h1>", unsafe_allow_html=True)
st.write("Sube los archivos CSV para consolidar el reporte.")

c_up1, c_up2 = st.columns(2)
with c_up1:
    file_main = st.file_uploader("1. Archivo General (Data Support)", type=['csv'])
with c_up2:
    file_airport = st.file_uploader("2. Archivo Aeropuerto (Firt Datos)", type=['csv'])

if file_main is not None and file_airport is not None:
    df_main = load_main_data(file_main)
    df_airport = load_airport_data(file_airport)
    
    # Concatenamos cuidando los índices
    df_raw = pd.concat([df_main, df_airport], ignore_index=True)
    df_metrics = aggregate_weekly(df_raw)
    
    st.sidebar.markdown(f"<h3 style='color: {CABIFY_PURPLE};'>Filtros y Descargas</h3>", unsafe_allow_html=True)
    all_audiences = ['Rider', 'Driver', 'B2B', 'Emergencias', 'Aeropuerto']
    audiences = [a for a in all_audiences if a in df_metrics['Audience'].unique()]
    
    selected_audience = st.sidebar.selectbox("Selecciona la Audiencia", audiences)
    
    df_filtered = df_metrics[df_metrics['Audience'] == selected_audience].sort_values('Week')
    available_weeks = sorted(df_filtered['Week'].dropna().unique(), reverse=True)
    selected_week = st.sidebar.selectbox("Selecciona la Semana a visualizar", available_weeks, index=0)
    
    # -------------------------------------------------------------
    # EXPORTAR RESUMEN EJECUTIVO EN PDF
    st.sidebar.divider()
    st.sidebar.subheader("📄 Reporte Directivo (PDF)")
    st.sidebar.caption("Descarga el resumen de todas las audiencias con código de color y análisis de NPS automático.")
    pdf_bytes = generar_pdf_resumen(df_metrics, df_raw, selected_week)
    st.sidebar.download_button(
        label="Descargar Executive Summary (.pdf)",
        data=pdf_bytes,
        file_name=f"COPS_Executive_Summary_W{selected_week}.pdf",
        mime="application/pdf"
    )

    # -------------------------------------------------------------
    # EXPORTAR EXCEL NPS VÁLIDO
    st.sidebar.divider()
    st.sidebar.subheader("📥 Exportar Datos Crudos")
    if 'NPS Score' in df_raw.columns:
        df_valid_nps = df_raw.dropna(subset=['NPS Score'])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for aud in audiences:
                df_aud = df_valid_nps[df_valid_nps['Audience'] == aud]
                if not df_aud.empty:
                    _ = df_aud.to_excel(writer, sheet_name=aud, index=False)
        st.sidebar.download_button(
            label="Descargar Reporte NPS (.xlsx)",
            data=output.getvalue(),
            file_name="Reporte_NPS_Valido_Cabify.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # --- TABS PARA EL DASHBOARD ---
    tab1, tab2 = st.tabs(["📈 KPIs Semanales y Tendencias", "🔍 Deep Dive (Motivos 3er Nivel)"])
    
    with tab1:
        current_data = df_filtered[df_filtered['Week'] == selected_week]
        prev_data = df_filtered[df_filtered['Week'] == selected_week - 1]
        
        if not current_data.empty:
            curr = current_data.iloc[0]
            prev = prev_data.iloc[0] if not prev_data.empty else current_data.iloc[0] * 0 
            
            def calc_delta_pct(current, previous):
                if previous == 0 or pd.isna(previous): return "0.0%"
                return f"{((current - previous) / previous) * 100:+.2f}%"
                
            def calc_delta_abs(current, previous):
                if pd.isna(previous): return "+0.0"
                return f"{current - previous:+.2f}"
                
            st.markdown(f"### Resumen Semana **{selected_week}** - {selected_audience}")
            
            st.markdown("#### I. Performance General de Gestión")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Contactos Recibidos", f"{curr['Contactos Recibidos']:,.0f}", calc_delta_pct(curr['Contactos Recibidos'], prev['Contactos Recibidos']), delta_color="inverse")
            with c2: st.metric("Contactos Ticket", f"{curr['Contactos Ticket']:,.0f}", calc_delta_pct(curr['Contactos Ticket'], prev['Contactos Ticket']), delta_color="inverse")
            with c3: st.metric("Contactos Chat", f"{curr['Contactos Chat']:,.0f}", calc_delta_pct(curr['Contactos Chat'], prev['Contactos Chat']), delta_color="inverse")
            with c4: st.metric("Contactos Call", f"{curr['Contactos Call']:,.0f}", calc_delta_pct(curr['Contactos Call'], prev['Contactos Call']), delta_color="inverse")
            
            c5, c6, c7 = st.columns(3)
            with c5: st.metric("NPS Score", f"{curr['NPS']:.2f}", calc_delta_abs(curr['NPS'], prev['NPS']), delta_color="normal")
            with c6: st.metric("CSAT", f"{curr['CSAT (%)']:.1f}%", calc_delta_abs(curr['CSAT (%)'], prev['CSAT (%)']) + "%", delta_color="normal")
            
            st.divider()
            
            st.markdown("#### II. Calidad Gestión de Tickets")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("TMO Promedio (Hrs)", f"{curr['TMO (Hrs)']:.2f}", calc_delta_abs(curr['TMO (Hrs)'], prev['TMO (Hrs)']), delta_color="inverse")
            with c2: st.metric("SLA 1ra Respuesta (<24hrs)", f"{curr['FiRT <24h (%)']:.2f}%", calc_delta_abs(curr['FiRT <24h (%)'], prev['FiRT <24h (%)']) + "%", delta_color="normal")
            with c3: st.metric("SLA Resolución (<36hrs)", f"{curr['FuRT <36h (%)']:.2f}%", calc_delta_abs(curr['FuRT <36h (%)'], prev['FuRT <36h (%)']) + "%", delta_color="normal")
            with c4: st.metric("Ratio Reopen / Tickets", f"{curr['Ratio Reopen/Tickets (%)']:.2f}%", calc_delta_abs(curr['Ratio Reopen/Tickets (%)'], prev['Ratio Reopen/Tickets (%)']) + "%", delta_color="inverse")
            
            st.divider()
            
            st.markdown("#### III. Calidad Gestión Canales Real Time")
            c1, c2 = st.columns(2)
            if pd.notna(curr['% Llamadas Atendidas']):
                with c1: st.metric("% Llamadas Atendidas", f"{curr['% Llamadas Atendidas']:.2f}%", calc_delta_abs(curr['% Llamadas Atendidas'], prev['% Llamadas Atendidas']) + "%", delta_color="normal")
            else: 
                with c1: st.metric("% Llamadas Atendidas", "S/D")
                
            if pd.notna(curr['% Chats Atendidos']):
                with c2: st.metric("% Chats Atendidos", f"{curr['% Chats Atendidos']:.2f}%", calc_delta_abs(curr['% Chats Atendidos'], prev['% Chats Atendidos']) + "%", delta_color="normal")
            else: 
                with c2: st.metric("% Chats Atendidos", "S/D")

            # --- GRÁFICOS DE TENDENCIA (SUAVIZADOS) ---
            st.divider()
            st.markdown("#### 📈 Evolución Histórica")
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                fig_vol = px.line(df_filtered, x='Week', y='Contactos Recibidos', markers=True, 
                                  title="Volumen de Contactos", color_discrete_sequence=[CABIFY_PURPLE])
                _ = fig_vol.update_layout(plot_bgcolor="white", xaxis_title="Semana", yaxis_title="Contactos")
                _ = fig_vol.update_yaxes(rangemode="tozero") 
                st.plotly_chart(fig_vol, use_container_width=True)
                
            with col_g2:
                fig_nps = px.line(df_filtered, x='Week', y=['NPS', 'CSAT (%)'], markers=True,
                                  title="Experiencia y Calidad", color_discrete_sequence=[CABIFY_PURPLE, CABIFY_SECONDARY])
                _ = fig_nps.update_layout(plot_bgcolor="white", xaxis_title="Semana", yaxis_title="Score / %")
                _ = fig_nps.update_yaxes(range=[-100, 100])
                st.plotly_chart(fig_nps, use_container_width=True)

    with tab2:
        st.markdown(f"### 🔍 Deep Dive: Motivos de Contacto Nivel 3 ({selected_audience} - Sem {selected_week})")
        df_raw_filtered = df_raw[(df_raw['Audience'] == selected_audience) & (df_raw['Week'] == selected_week)]
        
        if not df_raw_filtered.empty and 'ES Output Tags 3rd Level v2' in df_raw_filtered.columns:
            top_tags = df_raw_filtered['ES Output Tags 3rd Level v2'].value_counts().reset_index()
            top_tags.columns = ['Motivo (Tag 3er Nivel)', 'Volumen']
            top_tags = top_tags.head(10)
            
            fig_tags = px.bar(top_tags, x='Volumen', y='Motivo (Tag 3er Nivel)', orientation='h',
                              color_discrete_sequence=[CABIFY_SECONDARY])
            _ = fig_tags.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white")
            st.plotly_chart(fig_tags, use_container_width=True)
            
            st.markdown("#### Ejemplos Reales (Descripción del Usuario)")
            if 'Description' in df_raw_filtered.columns:
                sample_desc = df_raw_filtered[['ES Output Tags 3rd Level v2', 'Description']].dropna().sample(n=min(10, len(df_raw_filtered)))
                st.dataframe(sample_desc, use_container_width=True)
