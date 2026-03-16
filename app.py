import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuración estilo Cabify Minimalista
st.set_page_config(page_title="Cabify Support Dashboard", layout="wide", initial_sidebar_state="expanded")
CABIFY_PURPLE = "#7352FF"
CABIFY_SECONDARY = "#00D1A3"

# Columnas Core
CORE_COLUMNS = [
    'Date_Time', 'Audience', 'Contact Type', 'NPS_Score', 'CSAT_Pct', 
    'FRT_Hours', 'FuRT_Hours', 'Reopen_Count', 'Tag_1', 'Tag_2', 
    'Tag_3', 'Chat_Missed', 'Description', 'Group_Name'
]

# --- LECTOR ROBUSTO DE CSV ---
def read_csv_robust(filepath):
    encodings_to_try = ['utf-8-sig', 'utf-8', 'latin-1']
    delimiters_to_try = [';', ',']
    
    for enc in encodings_to_try:
        for sep in delimiters_to_try:
            try:
                filepath.seek(0)
                df = pd.read_csv(filepath, delimiter=sep, low_memory=False, encoding=enc, on_bad_lines='skip')
                if len(df.columns) > 1:
                    df.columns = df.columns.str.strip().str.replace('\xa0', ' ')
                    return df
            except Exception:
                continue
                
    filepath.seek(0)
    df = pd.read_csv(filepath, delimiter=';', low_memory=False, encoding='latin-1', on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.replace('\xa0', ' ')
    return df

def parse_num(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().replace('%', '')
    if '.' in s and ',' in s: s = s.replace('.', '').replace(',', '.')
    elif ',' in s: s = s.replace(',', '.')
    try: return float(s)
    except: return np.nan

def standard_clean(df, mapping):
    existing_mapping = {k: v for k, v in mapping.items() if k in df.columns}
    df = df[list(existing_mapping.keys())].rename(columns=existing_mapping)
    for c in ['NPS_Score', 'CSAT_Pct', 'FRT_Hours', 'FuRT_Hours', 'Reopen_Count']:
        if c in df.columns: df[c] = df[c].apply(parse_num)
    return df

# --- CARGA DEL REPORTE GENERAL ---
@st.cache_data
def load_main_data(filepath):
    df = read_csv_robust(filepath)
    df = df.loc[:, ~df.columns.duplicated()] 
    
    # --- NUEVO FILTRO: Include Contacts == 'Rest' ---
    if 'Include Contacts' in df.columns:
        df = df[df['Include Contacts'].astype(str).str.strip().str.lower() == 'rest']
    
    mapping = {
        'Date_Time': 'Date_Time', 'Audience': 'Audience', 'Contact Type': 'Contact Type',
        'NPS Score': 'NPS_Score', '% CSAT': 'CSAT_Pct', '# First Reply Time (Hours)': 'FRT_Hours',
        '# Full Resolution Time (Hours)': 'FuRT_Hours', '# Tickets con reopen': 'Reopen_Count',
        'ES Output Tags 1st Level v2': 'Tag_1', 'ES Output Tags 2nd Level v2': 'Tag_2', 
        'ES Output Tags 3rd Level v2': 'Tag_3', 'Chat Missed': 'Chat_Missed', 
        'Description': 'Description', 'Group name support': 'Group_Name'
    }
    
    df = standard_clean(df, mapping)
    
    if 'Audience' in df.columns:
        df['Audience'] = df['Audience'].replace({'Private': 'Rider', 'C4B': 'B2B', 'Driver': 'Driver'})
        df = df[df['Audience'].isin(['Rider', 'B2B', 'Driver'])]
        
    if 'Group_Name' in df.columns:
        mask = df['Group_Name'].astype(str).str.contains('emergencia', case=False, na=False)
        df_em = df[mask].copy()
        df_em['Audience'] = 'Emergencias'
        df = pd.concat([df, df_em], ignore_index=True)
        
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y', errors='coerce')
        
    df = df.loc[:, ~df.columns.duplicated()]
    final_cols = [c for c in CORE_COLUMNS if c in df.columns]
    return df[final_cols].copy()

# --- CARGA DEL REPORTE AEROPUERTO ---
@st.cache_data
def load_airport_data(filepath):
    df = read_csv_robust(filepath)
    df = df.loc[:, ~df.columns.duplicated()] 
    
    def determine_type(row):
        if row.get('# Tickets - Call', 0) == 1: return 'Call'
        if row.get('# Tickets - Chats', 0) == 1: return 'Chat'
        return 'Ticket'
    df['Contact Type'] = df.apply(determine_type, axis=1)
    df['Audience'] = 'Aeropuerto'
    
    mapping = {
        'Fecha de Referencia': 'Date_Time', 'Audience': 'Audience', 'Contact Type': 'Contact Type',
        'NPS Score': 'NPS_Score', '% CSAT': 'CSAT_Pct', '# First Reply Time (Hours)': 'FRT_Hours',
        '# Full Resolution Time (Hours)': 'FuRT_Hours', '# Tickets Reopen': 'Reopen_Count',
        'ES Output Tags 1st Level v2': 'Tag_1', 'ES Output Tags 2nd Level v2': 'Tag_2', 
        'ES Output Tags 3rd Level v2': 'Tag_3', 'Chat Missed': 'Chat_Missed', 
        'Description': 'Description', 'Group name support': 'Group_Name'
    }
    
    df = standard_clean(df, mapping)
    
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y', errors='coerce')
        
    df = df.loc[:, ~df.columns.duplicated()]
    final_cols = [c for c in CORE_COLUMNS if c in df.columns]
    return df[final_cols].copy()

# --- AGREGACIÓN SEMANAL ---
@st.cache_data
def aggregate_weekly(df):
    df['Week'] = df['Date_Time'].dt.isocalendar().week
    def aggs(grp):
        res = {}
        res['Contactos Recibidos'] = len(grp)
        res['Contactos Ticket'] = len(grp[grp['Contact Type'] == 'Ticket'])
        res['Contactos Chat'] = len(grp[grp['Contact Type'] == 'Chat'])
        res['Contactos Call'] = len(grp[grp['Contact Type'] == 'Call'])
        res['NPS'] = grp['NPS_Score'].mean() if 'NPS_Score' in grp.columns else np.nan
        
        csat = grp['CSAT_Pct'].mean() if 'CSAT_Pct' in grp.columns else np.nan
        res['CSAT (%)'] = csat * 100 if pd.notna(csat) and csat <= 1.0 else csat
        res['TMO (Hrs)'] = grp['FuRT_Hours'].mean() if 'FuRT_Hours' in grp.columns else np.nan
        
        valid_res = grp['FuRT_Hours'].dropna() if 'FuRT_Hours' in grp.columns else pd.Series()
        res['FuRT <36h (%)'] = (valid_res < 36).mean() * 100 if len(valid_res) > 0 else np.nan
        
        valid_frt = grp['FRT_Hours'].dropna() if 'FRT_Hours' in grp.columns else pd.Series()
        res['FiRT <24h (%)'] = (valid_frt < 24).mean() * 100 if len(valid_frt) > 0 else np.nan
        
        reopens = grp['Reopen_Count'].sum() if 'Reopen_Count' in grp.columns else 0
        res['Ratio Reopen/Tickets (%)'] = (reopens / res['Contactos Ticket'] * 100) if res['Contactos Ticket'] > 0 else 0
        
        calls = grp[grp['Contact Type'] == 'Call']
        if len(calls) > 0 and 'Tag_2' in grp.columns:
            res['% Llamadas Atendidas'] = ((len(calls) - calls['Tag_2'].astype(str).str.contains('talkdesk', case=False).sum()) / len(calls)) * 100
        else: res['% Llamadas Atendidas'] = np.nan
            
        chats = grp[grp['Contact Type'] == 'Chat']
        if len(chats) > 0 and 'Chat_Missed' in grp.columns:
            res['% Chats Atendidos'] = ((len(chats) - chats['Chat_Missed'].sum()) / len(chats)) * 100
        else: res['% Chats Atendidos'] = np.nan
        return pd.Series(res)
    return df.groupby(['Week', 'Audience']).apply(aggs).reset_index()

# --- FUNCIÓN DE ANÁLISIS DE DETRACTORES ---
def analizar_detractores(df_raw, aud, week):
    if 'Week' not in df_raw.columns:
        df_raw['Week'] = df_raw['Date_Time'].dt.isocalendar().week
    
    detractores = df_raw[(df_raw['Audience'] == aud) & (df_raw['Week'] == week) & (df_raw['NPS_Score'] == -100)]
    
    if detractores.empty:
        return "Excelente: No hay registros de encuestas NPS -100 para esta audiencia en la semana seleccionada."
    
    total = len(detractores)
    t1 = detractores['Tag_1'].value_counts().index[0] if 'Tag_1' in detractores.columns and not detractores['Tag_1'].dropna().empty else "No Definido"
    t2 = detractores['Tag_2'].value_counts().index[0] if 'Tag_2' in detractores.columns and not detractores['Tag_2'].dropna().empty else "No Definido"
    t3 = detractores['Tag_3'].value_counts().index[0] if 'Tag_3' in detractores.columns and not detractores['Tag_3'].dropna().empty else "No Definido"
    
    desc_sample = ""
    if 'Description' in detractores.columns:
        sample_df = detractores[(detractores['Tag_3'] == t3) & (detractores['Description'].notna())]
        if not sample_df.empty:
            desc_sample = str(sample_df['Description'].iloc[0]).replace('\n', ' ')[:150] + "..."
            
    resumen = f"Tuvimos {total} casos evaluados con NPS -100. "
    resumen += f"A nivel macro (Tag 1), la friccion principal esta en '{t1}'. "
    resumen += f"Al profundizar, los usuarios reportan mayormente problemas de '{t2}' (Tag 2), y de forma muy especifica se quejan de '{t3}' (Tag 3). "
    if desc_sample:
        resumen += f"Un ejemplo literal de la voz del cliente indica: \"{desc_sample}\""
        
    return resumen

# --- FUNCIÓN GENERADORA DE TEXTO SLACK ---
def generar_texto_slack(df_metrics, week):
    lines = []
    lines.append(f"📣 *C_OPS Weekly Update - Support - Semana {week}* 📣\n")
    lines.append("📄 *Resumen Ejecutivo (PDF):* `[Pega el link a tu Drive aquí]`")
    lines.append("📊 *Datos Crudos por Audiencias (Excel):* `[Pega el link a tu Drive aquí]`\n")
    lines.append("*--- RESUMEN DE INDICADORES ---*\n")

    audiences_in_week = df_metrics[df_metrics['Week'] == week]['Audience'].unique()
    iconos = {'Rider': '🚶', 'Driver': '🚘', 'B2B': '🏢', 'Emergencias': '🚑', 'Aeropuerto': '✈️'}
    
    for aud in ['Rider', 'Driver', 'B2B', 'Emergencias', 'Aeropuerto']:
        if aud not in audiences_in_week: continue
        
        curr_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Week'] == week)]
        prev_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Week'] == week - 1)]

        curr = curr_df.iloc[0]
        prev = prev_df.iloc[0] if not prev_df.empty else curr * 0

        vol_curr = curr['Contactos Recibidos']
        vol_prev = prev['Contactos Recibidos']
        vol_wow = ((vol_curr - vol_prev) / vol_prev * 100) if vol_prev else 0

        nps_curr = curr['NPS']
        nps_prev = prev['NPS']
        nps_wow = nps_curr - nps_prev

        csat_curr = curr['CSAT (%)']
        csat_wow = csat_curr - prev['CSAT (%)']

        firt_curr = curr['FiRT <24h (%)']
        reop_curr = curr['Ratio Reopen/Tickets (%)']

        icon = iconos.get(aud, '📊')
        lines.append(f"*{icon} {aud.upper()}*")
        lines.append(f"• *Volumen:* {vol_curr:,.0f} ({vol_wow:+.1f}% WoW)")
        
        nps_str = f"{nps_curr:.1f} ({nps_wow:+.1f})" if pd.notna(nps_curr) else "S/D"
        csat_str = f"{csat_curr:.1f}% ({csat_wow:+.1f}%)" if pd.notna(csat_curr) else "S/D"
        lines.append(f"• *Calidad:* NPS {nps_str} | CSAT {csat_str}")
        
        firt_str = f"{firt_curr:.1f}%" if pd.notna(firt_curr) else "S/D"
        reop_str = f"{reop_curr:.1f}%" if pd.notna(reop_curr) else "S/D"
        lines.append(f"• *Eficiencia:* SLA 1ra Rsp: {firt_str} | Reopen: {reop_str}\n")

    return "\n".join(lines)

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
    pdf.cell(0, 10, clean_txt(f"Resumen Ejecutivo C_OPS - Support - Semana {week}"), ln=True, align='C')
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

            df_trend = df_metrics[df_metrics['Audience'] == aud].sort_values('Week')
            if len(df_trend) > 1:
                img_path = f"trend_{aud}.png"
                try:
                    fig, ax1 = plt.subplots(figsize=(7, 2.5))
                    fig.patch.set_facecolor('white')
                    ax1.set_facecolor('white')
                    ax1.bar(df_trend['Week'].astype(str), df_trend['Contactos Recibidos'], color='#E2D9FF', label='Volumen')
                    ax1.set_ylabel('Contactos', color='#7352FF', fontweight='bold')
                    ax1.tick_params(axis='y', labelcolor='#7352FF')
                    ax2 = ax1.twinx()
                    ax2.plot(df_trend['Week'].astype(str), df_trend['NPS'], color='#00D1A3', marker='o', linewidth=2, label='NPS')
                    ax2.set_ylabel('NPS', color='#00D1A3', fontweight='bold')
                    ax2.tick_params(axis='y', labelcolor='#00D1A3')
                    ax2.set_ylim([-100, 100])
                    plt.title(f"Evolucion {aud} (Volumen vs NPS)", color='#333333', fontweight='bold')
                    ax1.spines['top'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    plt.tight_layout()
                    plt.savefig(img_path, dpi=150)
                    plt.close(fig)
                    
                    if pdf.get_y() > 200: pdf.add_page()
                    pdf.ln(2)
                    pdf.image(img_path, x=20, w=170)
                    os.remove(img_path)
                except Exception: pass

            insight_nps = analizar_detractores(df_raw, aud, week)
            pdf.ln(2)
            
            if "Excelente" in insight_nps:
                pdf.set_font("Arial", 'I', 9)
                pdf.set_text_color(0, 209, 163) # Verde
                pdf.multi_cell(0, 5, clean_txt(insight_nps))
            else:
                pdf.set_font("Arial", 'B', 9)
                pdf.set_text_color(255, 82, 82)
                pdf.cell(0, 5, clean_txt("  [!] ALERTA NPS:"), ln=True)
                pdf.set_font("Arial", '', 9)
                pdf.set_text_color(80, 80, 80)
                pdf.multi_cell(0, 5, clean_txt("  " + insight_nps))

            pdf.ln(5)
            pdf.set_draw_color(220, 220, 220)
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
    with st.spinner('Procesando, filtrando y consolidando archivos...'):
        df_main = load_main_data(file_main)
        df_airport = load_airport_data(file_airport)
        
        df_raw = pd.concat([df_main, df_airport], ignore_index=True)
        df_raw['Week'] = df_raw['Date_Time'].dt.isocalendar().week
        df_metrics = aggregate_weekly(df_raw)
    
    st.sidebar.markdown(f"<h3 style='color: {CABIFY_PURPLE};'>Filtros y Descargas</h3>", unsafe_allow_html=True)
    all_audiences = ['Rider', 'Driver', 'B2B', 'Emergencias', 'Aeropuerto']
    audiences = [a for a in all_audiences if a in df_metrics['Audience'].unique()]
    
    selected_audience = st.sidebar.selectbox("Selecciona la Audiencia", audiences)
    
    df_filtered = df_metrics[df_metrics['Audience'] == selected_audience].sort_values('Week')
    available_weeks = sorted(df_filtered['Week'].dropna().unique(), reverse=True)
    selected_week = st.sidebar.selectbox("Selecciona la Semana a visualizar", available_weeks, index=0)
    
    # EXPORTAR RESUMEN EJECUTIVO EN PDF
    st.sidebar.divider()
    st.sidebar.subheader("📄 Reporte Directivo (PDF)")
    st.sidebar.caption("Descarga el resumen con gráficos evolutivos y análisis de detractores.")
    
    _pdf_bytes = generar_pdf_resumen(df_metrics, df_raw, selected_week)
    _ = st.sidebar.download_button(
        label="Descargar Executive Summary (.pdf)",
        data=_pdf_bytes,
        file_name=f"COPS_Executive_Summary_W{selected_week}.pdf",
        mime="application/pdf"
    )

    # EXPORTAR EXCEL NPS VÁLIDO
    st.sidebar.divider()
    st.sidebar.subheader("📥 Exportar Datos Crudos")
    
    if 'NPS_Score' in df_raw.columns:
        df_valid_nps = df_raw.dropna(subset=['NPS_Score'])
        if not df_valid_nps.empty:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for aud in audiences:
                    df_aud = df_valid_nps[df_valid_nps['Audience'] == aud]
                    if not df_aud.empty:
                        _ = df_aud.to_excel(writer, sheet_name=aud, index=False)
            _ = st.sidebar.download_button(
                label="Descargar Reporte NPS (.xlsx)",
                data=output.getvalue(),
                file_name="Reporte_NPS_Valido_Cabify.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    tab1, tab2 = st.tabs(["📈 KPIs Semanales y Tendencias", "🔍 Deep Dive (Análisis de Detractores)"])
    
    with tab1:
        st.markdown("### 💬 Copiar Resumen para Slack")
        st.info("Pasa el mouse sobre la caja de abajo y haz clic en el ícono de copiar para pegarlo en Slack.")
        slack_msg = generar_texto_slack(df_metrics, selected_week)
        st.code(slack_msg, language="markdown")
        st.divider()

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

            st.divider()
            st.markdown("#### 📈 Evolución Histórica")
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                fig_vol = px.line(df_filtered, x='Week', y='Contactos Recibidos', markers=True, 
                                  title="Volumen de Contactos", color_discrete_sequence=[CABIFY_PURPLE])
                _ = fig_vol.update_layout(plot_bgcolor="white", xaxis_title="Semana", yaxis_title="Contactos")
                _ = fig_vol.update_yaxes(rangemode="tozero") 
                _ = st.plotly_chart(fig_vol, use_container_width=True)
                
            with col_g2:
                fig_nps = px.line(df_filtered, x='Week', y=['NPS', 'CSAT (%)'], markers=True,
                                  title="Experiencia y Calidad", color_discrete_sequence=[CABIFY_PURPLE, CABIFY_SECONDARY])
                _ = fig_nps.update_layout(plot_bgcolor="white", xaxis_title="Semana", yaxis_title="Score / %")
                _ = fig_nps.update_yaxes(range=[-100, 100])
                _ = st.plotly_chart(fig_nps, use_container_width=True)

    with tab2:
        st.markdown(f"### 🔍 Deep Dive: ¿Qué dicen nuestros detractores? ({selected_audience} - Sem {selected_week})")
        
        resumen_app = analizar_detractores(df_raw, selected_audience, selected_week)
        if "Excelente" in resumen_app:
            st.success("✅ " + resumen_app)
        else:
            st.error("🚨 " + resumen_app)
            
        st.divider()
        
        st.markdown("#### Distribución General de Motivos (Tag Nivel 3)")
        df_raw_filtered = df_raw[(df_raw['Audience'] == selected_audience) & (df_raw['Week'] == selected_week)]
        
        if not df_raw_filtered.empty and 'Tag_3' in df_raw_filtered.columns:
            top_tags = df_raw_filtered['Tag_3'].value_counts().reset_index()
            top_tags.columns = ['Motivo (Tag 3er Nivel)', 'Volumen']
            top_tags = top_tags.head(10)
            
            fig_tags = px.bar(top_tags, x='Volumen', y='Motivo (Tag 3er Nivel)', orientation='h',
                              color_discrete_sequence=[CABIFY_SECONDARY])
            _ = fig_tags.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white")
            _ = st.plotly_chart(fig_tags, use_container_width=True)
