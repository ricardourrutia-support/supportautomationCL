import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import calendar

# Configuración estilo Cabify Minimalista
st.set_page_config(page_title="Cabify Support Dashboard - Mensual", layout="wide", initial_sidebar_state="expanded")
CABIFY_PURPLE = "#7352FF"
CABIFY_SECONDARY = "#00D1A3"

# Columnas Core (Agregamos los nuevos atributos para auditoría)
CORE_COLUMNS = [
    'Date_Time', 'Audience', 'Contact Type', 'NPS_Score', 'CSAT_Pct', 
    'FRT_Hours', 'FuRT_Hours', 'Reopen_Count', 'Tag_1', 'Tag_2', 
    'Tag_3', 'Chat_Missed', 'Description', 'Group_Name', 'Include_Contacts', 'Service_Type',
    'Assignee_Email', 'Assignee_FullName', 'Ticket_Number', 'Automated'
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

# --- CARGA DEL REPORTE MAESTRO ---
@st.cache_data
def load_main_data(filepath):
    df = read_csv_robust(filepath)
    df = df.loc[:, ~df.columns.duplicated()] 
    
    mapping = {
        'Date_Time': 'Date_Time', 'Audience': 'Audience', 'Contact Type': 'Contact Type',
        'NPS Score': 'NPS_Score', '% CSAT': 'CSAT_Pct', '# First Reply Time (Hours)': 'FRT_Hours',
        '# Full Resolution Time (Hours)': 'FuRT_Hours', '# Tickets con reopen': 'Reopen_Count',
        'ES Output Tags 1st Level v2': 'Tag_1', 'ES Output Tags 2nd Level v2': 'Tag_2', 
        'ES Output Tags 3rd Level v2': 'Tag_3', 'Chat Missed': 'Chat_Missed', 
        'Description': 'Description', 'Group name support': 'Group_Name',
        'Include Contacts': 'Include_Contacts', 'Service Type': 'Service_Type',
        'Assignee Email': 'Assignee_Email', 'Assignee FullName': 'Assignee_FullName', 
        'Ticket Number': 'Ticket_Number', 'Automated': 'Automated'
    }
    df = standard_clean(df, mapping)
    
    # FILTRO 1: Include Contacts = 'Rest'
    if 'Include_Contacts' in df.columns:
        df = df[df['Include_Contacts'].astype(str).str.strip().str.lower() == 'rest']
    
    # FILTRO 2: Excluir Delivery
    if 'Service_Type' in df.columns:
        df = df[~df['Service_Type'].astype(str).str.lower().str.contains('delivery', na=False)]
    
    # FILTRO 3 (NUEVO): Solo tickets con Automated = 'Agent' (coincide con Tableau)
    if 'Automated' in df.columns:
        df = df[df['Automated'].astype(str).str.strip() == 'Agent']
        
    if 'Audience' in df.columns and 'Group_Name' in df.columns:
        df['Audience'] = df['Audience'].replace({'Private': 'Rider', 'C4B': 'B2B', 'Driver': 'Driver'})
        gn = df['Group_Name'].astype(str).str.strip().str.lower()
        
        valid_b2b = ['cl b2b atencion', 'cl b2b atención', 'tn b2b atencion', 'tn b2b atención', 'auto answer', 'autoanswer']
        mask_b2b = (df['Audience'] == 'B2B') & gn.isin(valid_b2b)
        
        invalid_rd = gn.str.contains('admin|fraude|applicants support', regex=True, na=False)
        invalid_starts = gn.str.startswith(('global', 'co ', 'pe ', 'uy ', 'ar ', 'es ', 'cex '))
        invalid_null = gn.isin(['null', 'nan', '', 'none'])
        mask_rd = df['Audience'].isin(['Rider', 'Driver']) & ~invalid_rd & ~invalid_starts & ~invalid_null
        
        valid_em = ['tn emergencias drivers', 'tn energencias drivers', 
                    'tn emergencias rider', 'tn energencias rider',
                    'tn emergencias', 'tn energencias']
        mask_em = gn.isin(valid_em)
        
        mask_aero = (gn == 'cl aeropuerto local')
        
        df['Final_Audience'] = np.nan
        df.loc[mask_rd, 'Final_Audience'] = df.loc[mask_rd, 'Audience']
        df.loc[mask_b2b, 'Final_Audience'] = 'B2B'
        df.loc[mask_em, 'Final_Audience'] = 'Emergencias'
        df.loc[mask_aero, 'Final_Audience'] = 'Aeropuerto'
        
        df = df.dropna(subset=['Final_Audience'])
        df['Audience'] = df['Final_Audience']
        
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y', errors='coerce')
        
    df = df.loc[:, ~df.columns.duplicated()]
    final_cols = [c for c in CORE_COLUMNS if c in df.columns]
    return df[final_cols].copy()

# --- AGREGACIÓN MENSUAL (ANTES ERA SEMANAL) ---
@st.cache_data
def aggregate_monthly(df):
    df['Year'] = df['Date_Time'].dt.year
    df['Month'] = df['Date_Time'].dt.month
    df['YearMonth'] = df['Date_Time'].dt.to_period('M')
    
    def aggs(grp):
        res = {}
        res['Contactos Recibidos'] = len(grp)
        res['Contactos Ticket'] = len(grp[grp['Contact Type'] == 'Ticket'])
        res['Contactos Chat'] = len(grp[grp['Contact Type'] == 'Chat'])
        res['Contactos Call'] = len(grp[grp['Contact Type'] == 'Call'])
        
        if 'NPS_Score' in grp.columns:
            res['NPS'] = grp['NPS_Score'].mean()
            res['NPS_Count'] = grp['NPS_Score'].count()
        else: 
            res['NPS'] = np.nan
            res['NPS_Count'] = 0
            
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
    
    return df.groupby(['YearMonth', 'Audience']).apply(aggs).reset_index()

# --- FUNCIÓN DE ANÁLISIS DE DETRACTORES ---
def analizar_detractores(df_raw, aud, year_month):
    if 'YearMonth' not in df_raw.columns:
        df_raw['YearMonth'] = df_raw['Date_Time'].dt.to_period('M')
    
    detractores = df_raw[(df_raw['Audience'] == aud) & (df_raw['YearMonth'] == year_month) & (df_raw['NPS_Score'] == -100)]
    
    if detractores.empty:
        return "Excelente: No hay registros de encuestas NPS -100 para esta audiencia en el mes seleccionado."
    
    total = len(detractores)
    t1 = detractores['Tag_1'].value_counts().index[0] if 'Tag_1' in detractores.columns and not detractores['Tag_1'].dropna().empty else "No Definido"
    t2 = detractores['Tag_2'].value_counts().index[0] if 'Tag_2' in detractores.columns and not detractores['Tag_2'].dropna().empty else "No Definido"
    t3 = detractores['Tag_3'].value_counts().index[0] if 'Tag_3' in detractores.columns and not detractores['Tag_3'].dropna().empty else "No Definido"
    
    desc_sample = ""
    if 'Description' in detractores.columns:
        sample_df = detractores[(detractores['Tag_3'] == t3) & (detractores['Description'].notna())]
        if not sample_df.empty:
            valid_samples = sample_df[~sample_df['Description'].str.contains('Conversation with', case=False, na=False)]
            
            if not valid_samples.empty:
                best_desc = str(valid_samples['Description'].iloc[0])
            else:
                best_desc = str(sample_df['Description'].iloc[0])
                
            best_desc = best_desc.replace('\n', ' ').replace('\r', '').strip()
            
            if len(best_desc) > 600:
                desc_sample = best_desc[:600] + "..."
            else:
                desc_sample = best_desc
            
    resumen = f"Tuvimos {total} casos evaluados con NPS -100. "
    resumen += f"A nivel macro (Tag 1), la friccion principal esta en '{t1}'. "
    resumen += f"Al profundizar, los usuarios reportan mayormente problemas de '{t2}' (Tag 2), y de forma muy especifica se quejan de '{t3}' (Tag 3). "
    if desc_sample:
        resumen += f"Un ejemplo literal de la voz del cliente indica: \"{desc_sample}\""
        
    return resumen

# --- TEXTO SLACK (MENSUAL) ---
def generar_texto_slack(df_metrics, year_month):
    month_name = calendar.month_name[year_month.month]
    lines = []
    lines.append(f"📣 C_OPS Monthly Update - Support - {month_name} {year_month.year} 📣\n")
    lines.append("📄 Resumen Ejecutivo (PDF): [Pega el link a tu Drive aquí]")
    lines.append("📊 Datos Crudos por Audiencias (Excel): [Pega el link a tu Drive aquí]\n")
    lines.append("--- RESUMEN DE INDICADORES ---\n")

    audiences_in_month = df_metrics[df_metrics['YearMonth'] == year_month]['Audience'].unique()
    iconos = {'Rider': '🚶', 'Driver': '🚘', 'B2B': '🏢', 'Emergencias': '🚑', 'Aeropuerto': '✈️'}
    
    # Calcular mes anterior
    prev_month = year_month - 1
    
    for aud in ['Rider', 'Driver', 'B2B', 'Emergencias', 'Aeropuerto']:
        if aud not in audiences_in_month: continue
        
        curr_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['YearMonth'] == year_month)]
        prev_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['YearMonth'] == prev_month)]

        curr = curr_df.iloc[0]
        prev = prev_df.iloc[0] if not prev_df.empty else curr * 0

        vol_curr = curr['Contactos Recibidos']
        vol_prev = prev['Contactos Recibidos']
        vol_mom = ((vol_curr - vol_prev) / vol_prev * 100) if vol_prev else 0

        nps_curr = curr['NPS']
        nps_prev = prev['NPS']
        nps_mom = nps_curr - nps_prev
        nps_count = curr.get('NPS_Count', 0)

        csat_curr = curr['CSAT (%)']
        csat_mom = csat_curr - prev['CSAT (%)']

        firt_curr = curr['FiRT <24h (%)']
        reop_curr = curr['Ratio Reopen/Tickets (%)']

        icon = iconos.get(aud, '📊')
        lines.append(f"{icon} {aud.upper()}")
        lines.append(f"• Volumen: {vol_curr:,.0f} ({vol_mom:+.1f}% MoM)")
        
        nps_str = f"{nps_curr:.1f} ({nps_mom:+.1f}) | {int(nps_count)} encuestas" if pd.notna(nps_curr) else "S/D"
        csat_str = f"{csat_curr:.1f}% ({csat_mom:+.1f}%)" if pd.notna(csat_curr) else "S/D"
        lines.append(f"• Calidad: NPS {nps_str} | CSAT {csat_str}")
        
        firt_str = f"{firt_curr:.1f}%" if pd.notna(firt_curr) else "S/D"
        reop_str = f"{reop_curr:.1f}%" if pd.notna(reop_curr) else "S/D"
        lines.append(f"• Eficiencia: SLA 1ra Rsp: {firt_str} | Reopen: {reop_str}\n")

    return "\n".join(lines)

# --- PDF 1: REPORTE VERTICAL CLÁSICO (MENSUAL) ---
def generar_pdf_resumen(df_metrics, df_raw, year_month):
    pdf = FPDF()
    pdf.add_page()
    
    month_name = calendar.month_name[year_month.month]
    prev_month = year_month - 1
    
    def clean_txt(text):
        if pd.isna(text): return ""
        return str(text).replace('"', "'").replace('\n', ' ').encode('latin-1', 'replace').decode('latin-1')
        
    def print_metric_line(label, val_str, delta_val, is_higher_better=True, is_pct=False):
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(55, 6, clean_txt(f"- {label}: {val_str}"), ln=0)
        
        if pd.isna(delta_val) or delta_val == 0:
            pdf.set_text_color(150, 150, 150)
            delta_str = "(=0.0)"
        else:
            if delta_val > 0:
                pdf.set_text_color(0, 209, 163) if is_higher_better else pdf.set_text_color(255, 82, 82)
            else:
                pdf.set_text_color(255, 82, 82) if is_higher_better else pdf.set_text_color(0, 209, 163)
            suffix = "% MoM" if is_pct else " MoM"
            delta_str = f"({delta_val:+.1f}{suffix})"
        pdf.cell(0, 6, clean_txt(delta_str), ln=1)
        
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(115, 82, 255)
    pdf.cell(0, 10, clean_txt(f"Resumen Ejecutivo C_OPS - Support - {month_name} {year_month.year}"), ln=True, align='C')
    pdf.ln(5)

    audiences_in_month = df_metrics[df_metrics['YearMonth'] == year_month]['Audience'].unique()
    for aud in ['Driver', 'Rider', 'B2B', 'Emergencias', 'Aeropuerto']:
        if aud not in audiences_in_month: continue
        
        curr_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['YearMonth'] == year_month)]
        prev_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['YearMonth'] == prev_month)]

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
            
            nps_count = curr.get('NPS_Count', 0)
            nps_val_str = f"{curr['NPS']:.1f} ({int(nps_count)} encuestas)" if pd.notna(curr['NPS']) else "S/D"

            print_metric_line("Volumen", f"{curr['Contactos Recibidos']:,.0f}", vol_pct, is_higher_better=False, is_pct=True)
            print_metric_line("NPS Score", nps_val_str, nps_diff, is_higher_better=True, is_pct=False)
            print_metric_line("CSAT (%)", f"{curr['CSAT (%)']:.1f}%", csat_diff, is_higher_better=True, is_pct=True)
            print_metric_line("FiRT <24h", f"{curr['FiRT <24h (%)']:.1f}%", firt_diff, is_higher_better=True, is_pct=True)
            print_metric_line("Ratio Reopen", f"{curr['Ratio Reopen/Tickets (%)']:.1f}%", reop_diff, is_higher_better=False, is_pct=True)

            insight_nps = analizar_detractores(df_raw, aud, year_month)
            pdf.ln(2)
            
            if "Excelente" in insight_nps:
                pdf.set_font("Arial", 'I', 9)
                pdf.set_text_color(0, 209, 163)
                pdf.multi_cell(0, 5, clean_txt(insight_nps))
            else:
                pdf.set_font("Arial", 'B', 9)
                pdf.set_text_color(255, 82, 82)
                pdf.cell(0, 5, clean_txt("  [!] ALERTA NPS:"), ln=True)
                pdf.set_font("Arial", '', 9)
                pdf.set_text_color(80, 80, 80)
                pdf.set_x(15)
                pdf.multi_cell(185, 5, clean_txt(insight_nps))

            pdf.ln(5)
            pdf.set_draw_color(220, 220, 220)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

    return pdf.output(dest='S').encode('latin-1')


# --- PDF 2: PRESENTACIÓN HORIZONTAL (Look Cabify) ---
def generar_pdf_presentacion(df_metrics, df_raw, year_month):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    
    month_name = calendar.month_name[year_month.month]
    prev_month = year_month - 1
    
    def clean_txt(text):
        if pd.isna(text): return ""
        return str(text).replace('"', "'").replace('\n', ' ').encode('latin-1', 'replace').decode('latin-1')
    
    def add_header(title):
        pdf.set_fill_color(115, 82, 255)
        pdf.rect(0, 0, 297, 22, 'F')
        pdf.set_font("Arial", 'B', 16)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(15, 6)
        pdf.cell(0, 10, clean_txt(title), ln=True)
    
    pdf.add_page()
    pdf.set_fill_color(115, 82, 255)
    pdf.rect(0, 0, 297, 210, 'F')
    pdf.set_text_color(255, 255, 255)
    
    pdf.set_y(85)
    pdf.set_font("Arial", 'B', 40)
    pdf.cell(0, 15, clean_txt("C_OPS Support Dashboard"), align='C', ln=True)
    pdf.set_font("Arial", '', 22)
    pdf.set_text_color(0, 209, 163)
    pdf.cell(0, 15, clean_txt(f"Resumen Directivo - {month_name} {year_month.year}"), align='C', ln=True)
    
    audiences_in_month = df_metrics[df_metrics['YearMonth'] == year_month]['Audience'].unique()
    
    for aud in ['Driver', 'Rider', 'B2B', 'Emergencias', 'Aeropuerto']:
        if aud not in audiences_in_month: continue
        
        curr_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['YearMonth'] == year_month)]
        prev_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['YearMonth'] == prev_month)]

        if not curr_df.empty:
            curr = curr_df.iloc[0]
            prev = prev_df.iloc[0] if not prev_df.empty else curr * 0
            
            pdf.add_page()
            add_header(f"Performance Operativo: {aud.upper()} | {month_name} {year_month.year}")
            
            vol_pct = ((curr['Contactos Recibidos'] - prev['Contactos Recibidos']) / prev['Contactos Recibidos'] * 100) if prev['Contactos Recibidos'] else 0
            nps_diff = curr['NPS'] - prev['NPS']
            csat_diff = curr['CSAT (%)'] - prev['CSAT (%)']
            firt_diff = curr['FiRT <24h (%)'] - prev['FiRT <24h (%)']
            reop_diff = curr['Ratio Reopen/Tickets (%)'] - prev['Ratio Reopen/Tickets (%)']
            nps_count = curr.get('NPS_Count', 0)
            
            pdf.set_xy(15, 35)
            pdf.set_fill_color(245, 245, 245)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(115, 82, 255)
            pdf.cell(85, 10, "  Indicador Principal", 0, 1, 'L', 1)
            
            metrics = [
                ("Volumen", f"{curr['Contactos Recibidos']:,.0f}", vol_pct, False, True),
                ("NPS Score", f"{curr['NPS']:.1f}", nps_diff, True, False),
                ("CSAT", f"{curr['CSAT (%)']:.1f}%", csat_diff, True, True),
                ("SLA 1ra Respuesta", f"{curr['FiRT <24h (%)']:.1f}%", firt_diff, True, True),
                ("Ratio Reopen", f"{curr['Ratio Reopen/Tickets (%)']:.1f}%", reop_diff, False, True)
            ]
            
            y_start = 50
            for name, val_str, mom_val, is_higher_better, is_pct in metrics:
                pdf.set_xy(15, y_start)
                pdf.set_font("Arial", 'B', 11)
                pdf.set_text_color(50, 50, 50)
                pdf.cell(40, 8, clean_txt(name), 0, 0, 'L')
                
                pdf.set_font("Arial", '', 11)
                pdf.cell(20, 8, clean_txt(val_str), 0, 0, 'L')
                
                if pd.isna(mom_val) or mom_val == 0:
                    pdf.set_text_color(150, 150, 150)
                    mom_str = "(-)"
                else:
                    if mom_val > 0:
                        pdf.set_text_color(0, 209, 163) if is_higher_better else pdf.set_text_color(255, 82, 82)
                    else:
                        pdf.set_text_color(255, 82, 82) if is_higher_better else pdf.set_text_color(0, 209, 163)
                    suffix = "%" if is_pct else ""
                    mom_str = f"({mom_val:+.1f}{suffix})"
                
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(25, 8, clean_txt(mom_str), 0, 1, 'R')
                
                pdf.set_draw_color(230, 230, 230)
                pdf.line(15, y_start + 8, 100, y_start + 8)
                y_start += 12
                
            pdf.set_xy(15, y_start)
            pdf.set_font("Arial", 'I', 9)
            pdf.set_text_color(150, 150, 150)
            pdf.cell(85, 6, clean_txt(f"*NPS basado en {int(nps_count)} encuestas validas."), 0, 1, 'L')

            df_trend = df_metrics[df_metrics['Audience'] == aud].sort_values('YearMonth')
            if len(df_trend) > 1:
                img_path = f"slide_trend_{aud}.png"
                try:
                    fig, ax1 = plt.subplots(figsize=(9, 4.5))
                    fig.patch.set_facecolor('white')
                    ax1.set_facecolor('white')
                    
                    # Convertir YearMonth a string para el eje X
                    x_labels = df_trend['YearMonth'].astype(str)
                    
                    ax1.bar(x_labels, df_trend['Contactos Recibidos'], color='#E2D9FF', label='Volumen')
                    ax1.set_ylabel('Contactos (Volumen)', color='#7352FF', fontweight='bold')
                    ax1.tick_params(axis='y', labelcolor='#7352FF')
                    
                    ax2 = ax1.twinx()
                    ax2.plot(x_labels, df_trend['NPS'], color='#00D1A3', marker='o', markersize=8, linewidth=3, label='NPS')
                    ax2.set_ylabel('NPS Score', color='#00D1A3', fontweight='bold')
                    ax2.tick_params(axis='y', labelcolor='#00D1A3')
                    ax2.set_ylim([-100, 100])
                    
                    plt.title("Evolucion Volumen vs NPS", color='#333333', fontweight='bold', fontsize=14, pad=15)
                    ax1.spines['top'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    plt.tight_layout()
                    plt.savefig(img_path, dpi=200)
                    plt.close(fig)
                    
                    pdf.image(img_path, x=115, y=30, w=170)
                    os.remove(img_path)
                except Exception: pass

            insight_nps = analizar_detractores(df_raw, aud, year_month)
            
            pdf.set_xy(15, 146)
            pdf.set_fill_color(250, 240, 255)
            pdf.rect(15, 144, 267, 56, 'F')
            
            if "Excelente" in insight_nps:
                pdf.set_font("Arial", 'B', 11)
                pdf.set_text_color(0, 209, 163)
                pdf.cell(0, 8, clean_txt("  [Voz del Cliente] Excelente rendimiento"), ln=True)
                pdf.set_x(17)
                pdf.set_font("Arial", 'I', 11)
                pdf.set_text_color(80, 80, 80)
                pdf.multi_cell(263, 5.5, clean_txt(insight_nps))
            else:
                pdf.set_font("Arial", 'B', 11)
                pdf.set_text_color(255, 82, 82)
                pdf.cell(0, 8, clean_txt("  [!] Focos de Friccion (Voz del Cliente)"), ln=True)
                pdf.set_x(17)
                pdf.set_font("Arial", '', 10)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(263, 5.5, clean_txt(insight_nps))

    return pdf.output(dest='S').encode('latin-1')


# --- INTERFAZ PRINCIPAL ---
st.markdown(f"<h1 style='color: {CABIFY_PURPLE};'>🚕 C_OPS Support Dashboard - Mensual</h1>", unsafe_allow_html=True)
st.write("Sube el archivo general para procesar todas las audiencias. **Versión Mensual con filtro Automated=Agent** (coincide con Tableau).")

file_main = st.file_uploader("Archivo Maestro (CSV)", type=['csv'])

if file_main is not None:
    with st.spinner('Aplicando reglas de negocio, limpiando y analizando datos...'):
        df_raw = load_main_data(file_main)
        df_raw['YearMonth'] = df_raw['Date_Time'].dt.to_period('M')
        df_metrics = aggregate_monthly(df_raw)
    
    st.sidebar.markdown(f"<h3 style='color: {CABIFY_PURPLE};'>Filtros y Descargas</h3>", unsafe_allow_html=True)
    all_audiences = ['Rider', 'Driver', 'B2B', 'Emergencias', 'Aeropuerto']
    audiences = [a for a in all_audiences if a in df_metrics['Audience'].unique()]
    
    selected_audience = st.sidebar.selectbox("Selecciona la Audiencia", audiences)
    
    df_filtered = df_metrics[df_metrics['Audience'] == selected_audience].sort_values('YearMonth')
    available_months = sorted(df_filtered['YearMonth'].dropna().unique(), reverse=True)
    
    # Formatear meses para display
    month_options = {str(ym): f"{calendar.month_name[ym.month]} {ym.year}" for ym in available_months}
    selected_month_str = st.sidebar.selectbox(
        "Selecciona el Mes a visualizar", 
        options=list(month_options.keys()),
        format_func=lambda x: month_options[x],
        index=0
    )
    selected_month = pd.Period(selected_month_str)
    
    # REPORTE VERTICAL
    st.sidebar.divider()
    st.sidebar.subheader("📄 Reportes Ejecutivos (PDF)")
    st.sidebar.caption("Formato documento clásico.")
    _pdf_vertical = generar_pdf_resumen(df_metrics, df_raw, selected_month)
    _ = st.sidebar.download_button(
        label=f"📄 Descargar Informe (Vertical)",
        data=_pdf_vertical,
        file_name=f"COPS_Informe_{selected_month}.pdf",
        mime="application/pdf"
    )

    # PRESENTACIÓN HORIZONTAL
    st.sidebar.caption("Formato diapositivas visuales.")
    _pdf_horizontal = generar_pdf_presentacion(df_metrics, df_raw, selected_month)
    _ = st.sidebar.download_button(
        label=f"📊 Descargar Presentacion (Horizontal)",
        data=_pdf_horizontal,
        file_name=f"COPS_Presentacion_{selected_month}.pdf",
        mime="application/pdf"
    )

    # EXCEL DATOS CRUDOS
    st.sidebar.divider()
    st.sidebar.subheader("📥 Exportar Datos Crudos")
    
    if 'NPS_Score' in df_raw.columns:
        df_valid_nps = df_raw[(df_raw['NPS_Score'].notna()) & (df_raw['YearMonth'] == selected_month)]
        if not df_valid_nps.empty:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for aud in audiences:
                    df_aud = df_valid_nps[df_valid_nps['Audience'] == aud]
                    if not df_aud.empty:
                        _ = df_aud.to_excel(writer, sheet_name=aud, index=False)
            _ = st.sidebar.download_button(
                label=f"Descargar Reporte NPS {selected_month} (.xlsx)",
                data=output.getvalue(),
                file_name=f"Reporte_NPS_Valido_Cabify_{selected_month}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.sidebar.warning(f"No hay encuestas de NPS para exportar en {month_options[selected_month_str]}.")
    
    # === TABS PRINCIPALES ===
    tab1, tab2, tab3 = st.tabs(["📈 KPIs Mensuales", "📊 Evolución Histórica", "🔍 Deep Dive (Análisis de Detractores)"])
    
    with tab1:
        st.markdown("### 💬 Copiar Resumen para Slack")
        st.info("Pasa el mouse sobre la caja de abajo y haz clic en el ícono de copiar para pegarlo directamente en Slack.")
        slack_msg = generar_texto_slack(df_metrics, selected_month)
        st.code(slack_msg, language="markdown")
        st.divider()

        current_data = df_filtered[df_filtered['YearMonth'] == selected_month]
        prev_month = selected_month - 1
        prev_data = df_filtered[df_filtered['YearMonth'] == prev_month]
        
        if not current_data.empty:
            curr = current_data.iloc[0]
            prev = prev_data.iloc[0] if not prev_data.empty else current_data.iloc[0] * 0 
            
            def calc_delta_pct(current, previous):
                if previous == 0 or pd.isna(previous): return "0.0%"
                return f"{((current - previous) / previous) * 100:+.2f}%"
                
            def calc_delta_abs(current, previous):
                if pd.isna(previous): return "+0.0"
                return f"{current - previous:+.2f}"
                
            month_display = month_options[selected_month_str]
            st.markdown(f"### Resumen **{month_display}** - {selected_audience}")
            
            st.markdown("#### I. Performance General de Gestión")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Contactos Recibidos", f"{curr['Contactos Recibidos']:,.0f}", calc_delta_pct(curr['Contactos Recibidos'], prev['Contactos Recibidos']), delta_color="inverse")
            with c2: st.metric("Contactos Ticket", f"{curr['Contactos Ticket']:,.0f}", calc_delta_pct(curr['Contactos Ticket'], prev['Contactos Ticket']), delta_color="inverse")
            with c3: st.metric("Contactos Chat", f"{curr['Contactos Chat']:,.0f}", calc_delta_pct(curr['Contactos Chat'], prev['Contactos Chat']), delta_color="inverse")
            with c4: st.metric("Contactos Call", f"{curr['Contactos Call']:,.0f}", calc_delta_pct(curr['Contactos Call'], prev['Contactos Call']), delta_color="inverse")
            
            c5, c6, c7 = st.columns(3)
            with c5: st.metric("NPS Score", f"{curr['NPS']:.2f}", calc_delta_abs(curr['NPS'], prev['NPS']), delta_color="normal", help=f"Basado en {int(curr.get('NPS_Count', 0))} encuestas este mes")
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

    # === TAB 2: EVOLUCIÓN HISTÓRICA (NUEVO) ===
    with tab2:
        st.markdown(f"### 📊 Evolución Mensual de Indicadores - {selected_audience}")
        st.info("Visualiza la tendencia histórica de todos los indicadores clave por mes.")
        
        df_trend = df_filtered.sort_values('YearMonth').copy()
        df_trend['Mes'] = df_trend['YearMonth'].astype(str)
        
        if len(df_trend) > 0:
            # Gráfico 1: Volumen
            st.markdown("#### 📦 Volumen de Contactos")
            fig_vol = px.bar(df_trend, x='Mes', y='Contactos Recibidos', 
                           color_discrete_sequence=[CABIFY_PURPLE],
                           text='Contactos Recibidos')
            fig_vol.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_vol.update_layout(plot_bgcolor="white", xaxis_title="Mes", yaxis_title="Contactos")
            fig_vol.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Distribución por tipo de contacto
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Por Tipo de Contacto")
                fig_tipo = px.bar(df_trend, x='Mes', 
                                 y=['Contactos Ticket', 'Contactos Chat', 'Contactos Call'],
                                 barmode='stack',
                                 color_discrete_sequence=[CABIFY_PURPLE, CABIFY_SECONDARY, '#FFB347'])
                fig_tipo.update_layout(plot_bgcolor="white", yaxis_title="Contactos", legend_title="Tipo")
                st.plotly_chart(fig_tipo, use_container_width=True)
            
            with col2:
                # Mostrar tabla de composición
                st.markdown("##### Composición Mensual")
                df_comp = df_trend[['Mes', 'Contactos Ticket', 'Contactos Chat', 'Contactos Call', 'Contactos Recibidos']].copy()
                df_comp['% Ticket'] = (df_comp['Contactos Ticket'] / df_comp['Contactos Recibidos'] * 100).round(1)
                df_comp['% Chat'] = (df_comp['Contactos Chat'] / df_comp['Contactos Recibidos'] * 100).round(1)
                df_comp['% Call'] = (df_comp['Contactos Call'] / df_comp['Contactos Recibidos'] * 100).round(1)
                st.dataframe(df_comp[['Mes', '% Ticket', '% Chat', '% Call']], use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Gráfico 2: NPS y CSAT
            st.markdown("#### ⭐ Experiencia del Cliente (NPS y CSAT)")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_nps = px.line(df_trend, x='Mes', y='NPS', markers=True,
                                color_discrete_sequence=[CABIFY_PURPLE])
                fig_nps.update_traces(line=dict(width=3), marker=dict(size=10))
                fig_nps.update_layout(plot_bgcolor="white", title="NPS Score", yaxis_range=[-100, 100])
                # Agregar línea de referencia en 0
                fig_nps.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig_nps, use_container_width=True)
            
            with col2:
                fig_csat = px.line(df_trend, x='Mes', y='CSAT (%)', markers=True,
                                 color_discrete_sequence=[CABIFY_SECONDARY])
                fig_csat.update_traces(line=dict(width=3), marker=dict(size=10))
                fig_csat.update_layout(plot_bgcolor="white", title="CSAT (%)", yaxis_range=[0, 100])
                st.plotly_chart(fig_csat, use_container_width=True)
            
            st.divider()
            
            # Gráfico 3: SLAs
            st.markdown("#### ⏱️ Cumplimiento de SLAs")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_firt = px.area(df_trend, x='Mes', y='FiRT <24h (%)', 
                                  color_discrete_sequence=[CABIFY_PURPLE])
                fig_firt.update_layout(plot_bgcolor="white", title="SLA 1ra Respuesta (<24h)", yaxis_range=[0, 100])
                fig_firt.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7, 
                                  annotation_text="Meta 80%", annotation_position="top right")
                st.plotly_chart(fig_firt, use_container_width=True)
            
            with col2:
                fig_furt = px.area(df_trend, x='Mes', y='FuRT <36h (%)',
                                  color_discrete_sequence=[CABIFY_SECONDARY])
                fig_furt.update_layout(plot_bgcolor="white", title="SLA Resolución (<36h)", yaxis_range=[0, 100])
                fig_furt.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7,
                                  annotation_text="Meta 80%", annotation_position="top right")
                st.plotly_chart(fig_furt, use_container_width=True)
            
            st.divider()
            
            # Gráfico 4: TMO y Reopen
            st.markdown("#### 🔄 Eficiencia Operativa")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_tmo = px.line(df_trend, x='Mes', y='TMO (Hrs)', markers=True,
                                color_discrete_sequence=['#FF6B6B'])
                fig_tmo.update_traces(line=dict(width=3), marker=dict(size=10))
                fig_tmo.update_layout(plot_bgcolor="white", title="TMO Promedio (Horas)")
                fig_tmo.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_tmo, use_container_width=True)
            
            with col2:
                fig_reopen = px.bar(df_trend, x='Mes', y='Ratio Reopen/Tickets (%)',
                                   color_discrete_sequence=['#FFB347'],
                                   text='Ratio Reopen/Tickets (%)')
                fig_reopen.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_reopen.update_layout(plot_bgcolor="white", title="Ratio Reopen/Tickets (%)")
                fig_reopen.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_reopen, use_container_width=True)
            
            st.divider()
            
            # Tabla resumen completa
            st.markdown("#### 📋 Tabla Resumen de Indicadores")
            
            df_table = df_trend[['Mes', 'Contactos Recibidos', 'NPS', 'CSAT (%)', 
                                'FiRT <24h (%)', 'FuRT <36h (%)', 'TMO (Hrs)', 
                                'Ratio Reopen/Tickets (%)']].copy()
            df_table = df_table.round(2)
            
            # Calcular variaciones MoM
            for col in ['Contactos Recibidos', 'NPS', 'CSAT (%)', 'FiRT <24h (%)', 'FuRT <36h (%)', 'TMO (Hrs)', 'Ratio Reopen/Tickets (%)']:
                df_table[f'{col} MoM'] = df_table[col].pct_change() * 100
            
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            
            # Exportar evolución a Excel
            st.markdown("##### 📥 Exportar Evolución")
            output_evol = io.BytesIO()
            with pd.ExcelWriter(output_evol, engine='xlsxwriter') as writer:
                df_table.to_excel(writer, sheet_name='Evolución', index=False)
            st.download_button(
                label="📊 Descargar Evolución Mensual (.xlsx)",
                data=output_evol.getvalue(),
                file_name=f"Evolucion_Mensual_{selected_audience}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No hay suficientes datos para mostrar la evolución histórica.")

    # === TAB 3: DEEP DIVE ===
    with tab3:
        month_display = month_options[selected_month_str]
        st.markdown(f"### 🔍 Deep Dive: ¿Qué dicen nuestros detractores? ({selected_audience} - {month_display})")
        
        resumen_app = analizar_detractores(df_raw, selected_audience, selected_month)
        if "Excelente" in resumen_app:
            st.success("✅ " + resumen_app)
        else:
            st.error("🚨 " + resumen_app)
            
        st.divider()
        
        st.markdown("#### Distribución General de Motivos (Tag Nivel 3)")
        df_raw_filtered = df_raw[(df_raw['Audience'] == selected_audience) & (df_raw['YearMonth'] == selected_month)]
        
        if not df_raw_filtered.empty and 'Tag_3' in df_raw_filtered.columns:
            top_tags = df_raw_filtered['Tag_3'].value_counts().reset_index()
            top_tags.columns = ['Motivo (Tag 3er Nivel)', 'Volumen']
            top_tags = top_tags.head(10)
            
            fig_tags = px.bar(top_tags, x='Volumen', y='Motivo (Tag 3er Nivel)', orientation='h',
                              color_discrete_sequence=[CABIFY_SECONDARY])
            fig_tags.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white")
            st.plotly_chart(fig_tags, use_container_width=True)

    # === INFORMACIÓN DE FILTROS APLICADOS ===
    st.sidebar.divider()
    st.sidebar.subheader("ℹ️ Información de Filtros")
    st.sidebar.caption(f"""
    **Filtros aplicados:**
    - Include Contacts = 'Rest'
    - Service Type ≠ 'Delivery'
    - **Automated = 'Agent'** (nuevo)
    
    **Registros procesados:** {len(df_raw):,}
    """)
