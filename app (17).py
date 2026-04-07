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
from datetime import datetime

# Configuración estilo Cabify
st.set_page_config(page_title="Cabify Support Dashboard", layout="wide", initial_sidebar_state="expanded")
CABIFY_PURPLE = "#7352FF"
CABIFY_SECONDARY = "#00D1A3"

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    div[data-testid="stMetricValue"] { color: #7352FF; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

CORE_COLUMNS = [
    'Date_Time', 'Audience', 'Contact Type', 'NPS_Score', 'CSAT_Pct', 
    'FRT_Hours', 'FuRT_Hours', 'Reopen_Count', 'Tag_1', 'Tag_2', 
    'Tag_3', 'Chat_Missed', 'Description', 'Group_Name', 'Include_Contacts', 
    'Service_Type', 'Assignee_Email', 'Assignee_FullName', 'Ticket_Number', 'Automated'
]

def read_csv_robust(filepath):
    for enc in ['utf-8-sig', 'utf-8', 'latin-1']:
        for sep in [';', ',']:
            try:
                filepath.seek(0)
                df = pd.read_csv(filepath, delimiter=sep, low_memory=False, encoding=enc, on_bad_lines='skip')
                if len(df.columns) > 1:
                    df.columns = df.columns.str.strip().str.replace('\xa0', ' ')
                    return df
            except: continue
    filepath.seek(0)
    return pd.read_csv(filepath, delimiter=';', low_memory=False, encoding='latin-1', on_bad_lines='skip')

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

@st.cache_data
def load_main_data(filepath, include_abibot=False):
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
    
    if 'Include_Contacts' in df.columns:
        df = df[df['Include_Contacts'].astype(str).str.strip().str.lower() == 'rest']
    if 'Service_Type' in df.columns:
        df = df[~df['Service_Type'].astype(str).str.lower().str.contains('delivery', na=False)]
    if 'Automated' in df.columns and not include_abibot:
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
        valid_em = ['tn emergencias drivers', 'tn energencias drivers', 'tn emergencias rider', 'tn energencias rider', 'tn emergencias', 'tn energencias']
        mask_em = gn.isin(valid_em)
        mask_aero = (gn == 'cl aeropuerto local')
        df['Final_Audience'] = pd.Series(dtype='object', index=df.index)
        df.loc[mask_rd, 'Final_Audience'] = df.loc[mask_rd, 'Audience'].values
        df.loc[mask_b2b, 'Final_Audience'] = 'B2B'
        df.loc[mask_em, 'Final_Audience'] = 'Emergencias'
        df.loc[mask_aero, 'Final_Audience'] = 'Aeropuerto'
        df = df[df['Final_Audience'].notna()]
        df['Audience'] = df['Final_Audience']
        
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y', errors='coerce')
    df = df.loc[:, ~df.columns.duplicated()]
    final_cols = [c for c in CORE_COLUMNS if c in df.columns]
    return df[final_cols].copy()

@st.cache_data
def load_whatsapp_data(filepath):
    df = read_csv_robust(filepath)
    df = df.loc[:, ~df.columns.duplicated()]
    mapping = {
        'Created At Local Dt': 'Date_Time', 'Ticket Number': 'Ticket_Number',
        '# First Reply Time (Min)': 'FRT_Min', '# Full Resolution Time (Hours)': 'FuRT_Hours',
        '% CSAT': 'CSAT_Pct', 'NPS Score': 'NPS_Score',
        'ES Output Tags 1st Level v2': 'Tag_1', 'ES Output Tags 2nd Level v2': 'Tag_2', 
        'ES Output Tags 3rd Level v2': 'Tag_3',
    }
    existing_mapping = {k: v for k, v in mapping.items() if k in df.columns}
    df = df[list(existing_mapping.keys())].rename(columns=existing_mapping)
    for c in ['FRT_Min', 'FuRT_Hours', 'CSAT_Pct', 'NPS_Score']:
        if c in df.columns: df[c] = df[c].apply(parse_num)
    if 'FRT_Min' in df.columns: df['FRT_Hours'] = df['FRT_Min'] / 60
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y', errors='coerce')
    df['Audience'] = 'Aeropuerto WhatsApp'
    df['Contact Type'] = 'WhatsApp'
    df['Reopen_Count'] = 0
    df['Chat_Missed'] = 0
    return df

def clasificar_mencion(texto):
    if not isinstance(texto, str): return "Desconocido"
    t = texto.lower()
    if any(p in t for p in ["ley uber", "bencina", "gobierno", "ministro", "noticia", "rt @"]): return "Ruido Mediático"
    if any(p in t for p in ["cobro", "tarifa", "cobraron", "estafa", "robo", "promoción", "código", "precio", "caro"]): return "Cobros y Tarifas"
    if any(p in t for p in ["aire", "calor", "conductor", "rasca", "pésimo", "grosero", "sucio", "olor"]): return "Calidad de Servicio"
    if any(p in t for p in ["espera", "toman", "cancel", "demora", "no llega", "app", "acepta", "disponib"]): return "Disponibilidad / App"
    if any(p in t for p in ["penca", "callampa", "ctm", "wea", "qlo", "mierda", "asco", "basura", "nunca más"]): return "Frustración Crítica"
    if any(p in t for p in ["segur", "miedo", "acoso", "peligr", "ruta", "desvío"]): return "Seguridad"
    return "Otros / Neutro"

@st.cache_data
def load_brandwatch_data(filepath):
    try:
        filepath.seek(0)
        content = filepath.read()
        text = content.decode('utf-8')
        lines = text.split('\n')
        skip = 0
        for i, line in enumerate(lines[:20]):
            if 'Snippet' in line or 'Date' in line:
                skip = i
                break
        filepath.seek(0)
        df = pd.read_csv(filepath, skiprows=skip, encoding='utf-8', low_memory=False)
        df.columns = [str(c).replace('"', '').strip() for c in df.columns]
        txt_col = next((c for c in df.columns if c.lower() in ['snippet', 'full text', 'text']), None)
        if txt_col:
            df['Texto'] = df[txt_col]
            df['Categoría'] = df[txt_col].apply(clasificar_mencion)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error Brandwatch: {e}")
        return None

def aggregate_data(df, period_type='monthly'):
    if period_type == 'weekly':
        df['Period'] = df['Date_Time'].dt.isocalendar().week
        df['Year'] = df['Date_Time'].dt.isocalendar().year
        df['PeriodLabel'] = 'S' + df['Period'].astype(str)
    else:
        df['Year'] = df['Date_Time'].dt.year
        df['Month'] = df['Date_Time'].dt.month
        df['Period'] = df['Date_Time'].dt.to_period('M')
        df['PeriodLabel'] = df['Period'].astype(str)
    
    def aggs(grp):
        res = {}
        res['Contactos Recibidos'] = len(grp)
        res['Contactos Ticket'] = len(grp[grp['Contact Type'] == 'Ticket'])
        res['Contactos Chat'] = len(grp[grp['Contact Type'] == 'Chat'])
        res['Contactos Call'] = len(grp[grp['Contact Type'] == 'Call'])
        if 'NPS_Score' in grp.columns:
            res['NPS'] = grp['NPS_Score'].mean()
            res['NPS_Count'] = grp['NPS_Score'].count()
        else: res['NPS'], res['NPS_Count'] = np.nan, 0
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
    
    if period_type == 'weekly':
        return df.groupby(['Year', 'Period', 'PeriodLabel', 'Audience']).apply(aggs).reset_index()
    return df.groupby(['Period', 'PeriodLabel', 'Audience']).apply(aggs).reset_index()

def analizar_detractores(df_raw, aud, period_value, period_type='monthly'):
    if period_type == 'weekly':
        if 'Week' not in df_raw.columns: df_raw['Week'] = df_raw['Date_Time'].dt.isocalendar().week
        detractores = df_raw[(df_raw['Audience'] == aud) & (df_raw['Week'] == period_value) & (df_raw['NPS_Score'] == -100)]
    else:
        if 'YearMonth' not in df_raw.columns: df_raw['YearMonth'] = df_raw['Date_Time'].dt.to_period('M')
        detractores = df_raw[(df_raw['Audience'] == aud) & (df_raw['YearMonth'] == period_value) & (df_raw['NPS_Score'] == -100)]
    if detractores.empty: return "Excelente: No hay registros de encuestas NPS -100 para esta audiencia en el período seleccionado."
    total = len(detractores)
    t1 = detractores['Tag_1'].value_counts().index[0] if 'Tag_1' in detractores.columns and not detractores['Tag_1'].dropna().empty else "No Definido"
    t2 = detractores['Tag_2'].value_counts().index[0] if 'Tag_2' in detractores.columns and not detractores['Tag_2'].dropna().empty else "No Definido"
    t3 = detractores['Tag_3'].value_counts().index[0] if 'Tag_3' in detractores.columns and not detractores['Tag_3'].dropna().empty else "No Definido"
    desc_sample = ""
    if 'Description' in detractores.columns:
        sample_df = detractores[(detractores['Tag_3'] == t3) & (detractores['Description'].notna())]
        if not sample_df.empty:
            valid_samples = sample_df[~sample_df['Description'].str.contains('Conversation with', case=False, na=False)]
            best_desc = str(valid_samples['Description'].iloc[0]) if not valid_samples.empty else str(sample_df['Description'].iloc[0])
            best_desc = best_desc.replace('\n', ' ').replace('\r', '').strip()
            desc_sample = best_desc[:600] + "..." if len(best_desc) > 600 else best_desc
    resumen = f"Tuvimos {total} casos evaluados con NPS -100. A nivel macro (Tag 1), la friccion principal esta en '{t1}'. "
    resumen += f"Al profundizar, los usuarios reportan mayormente problemas de '{t2}' (Tag 2), y de forma muy especifica se quejan de '{t3}' (Tag 3). "
    if desc_sample: resumen += f"Un ejemplo literal de la voz del cliente indica: \"{desc_sample}\""
    return resumen

def generar_texto_slack(df_metrics, period_value, period_type='monthly'):
    if period_type == 'weekly':
        title = f"📣 C_OPS Weekly Update - Support - Semana {period_value} 📣\n"
        period_suffix = "WoW"
        prev_period = period_value - 1
    else:
        month_name = calendar.month_name[period_value.month]
        title = f"📣 C_OPS Monthly Update - Support - {month_name} {period_value.year} 📣\n"
        period_suffix = "MoM"
        prev_period = period_value - 1
    lines = [title, "📄 Resumen Ejecutivo (PDF): [Pega el link aquí]", "📊 Datos Crudos (Excel): [Pega el link aquí]\n", "--- RESUMEN DE INDICADORES ---\n"]
    audiences_in_period = df_metrics[df_metrics['Period'] == period_value]['Audience'].unique()
    iconos = {'Rider': '🚶', 'Driver': '🚘', 'B2B': '🏢', 'Emergencias': '🚑', 'Aeropuerto': '✈️', 'Aeropuerto WhatsApp': '📱'}
    for aud in ['Rider', 'Driver', 'B2B', 'Emergencias', 'Aeropuerto', 'Aeropuerto WhatsApp']:
        if aud not in audiences_in_period: continue
        curr_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Period'] == period_value)]
        prev_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Period'] == prev_period)]
        if curr_df.empty: continue
        curr = curr_df.iloc[0]
        prev = prev_df.iloc[0] if not prev_df.empty else curr * 0
        vol_curr, vol_prev = curr['Contactos Recibidos'], prev['Contactos Recibidos']
        vol_change = ((vol_curr - vol_prev) / vol_prev * 100) if vol_prev else 0
        nps_curr = curr['NPS']
        nps_change = nps_curr - prev['NPS'] if pd.notna(prev['NPS']) else 0
        nps_count = curr.get('NPS_Count', 0)
        csat_curr = curr['CSAT (%)']
        csat_change = csat_curr - prev['CSAT (%)'] if pd.notna(prev['CSAT (%)']) else 0
        firt_curr, reop_curr = curr['FiRT <24h (%)'], curr['Ratio Reopen/Tickets (%)']
        icon = iconos.get(aud, '📊')
        lines.append(f"{icon} {aud.upper()}")
        lines.append(f"• Volumen: {vol_curr:,.0f} ({vol_change:+.1f}% {period_suffix})")
        nps_str = f"{nps_curr:.1f} ({nps_change:+.1f}) | {int(nps_count)} enc." if pd.notna(nps_curr) else "S/D"
        csat_str = f"{csat_curr:.1f}% ({csat_change:+.1f}%)" if pd.notna(csat_curr) else "S/D"
        lines.append(f"• Calidad: NPS {nps_str} | CSAT {csat_str}")
        firt_str = f"{firt_curr:.1f}%" if pd.notna(firt_curr) else "S/D"
        reop_str = f"{reop_curr:.1f}%" if pd.notna(reop_curr) else "S/D"
        lines.append(f"• Eficiencia: SLA 1ra Rsp: {firt_str} | Reopen: {reop_str}\n")
    return "\n".join(lines)

def generar_excel_cabify(df_raw, audiences, period_type, selected_period, period_display):
    output = io.BytesIO()
    if period_type == 'weekly':
        df_period = df_raw[df_raw['Week'] == selected_period].copy()
    else:
        df_period = df_raw[df_raw['YearMonth'] == selected_period].copy()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#7352FF', 'border': 1, 'align': 'center', 'valign': 'vcenter', 'font_size': 11})
        alt_fmt = workbook.add_format({'border': 1, 'bg_color': '#F5F3FF', 'font_size': 10})
        normal_fmt = workbook.add_format({'border': 1, 'font_size': 10})
        for aud in audiences:
            df_aud = df_period[df_period['Audience'] == aud].copy()
            if df_aud.empty: continue
            cols = df_aud.columns.tolist()
            if 'Ticket_Number' in cols:
                cols.remove('Ticket_Number')
                cols = ['Ticket_Number'] + cols
                df_aud = df_aud[cols]
            sheet_name = aud[:31].replace('/', '-')
            df_aud.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
            ws = writer.sheets[sheet_name]
            for col_num, value in enumerate(df_aud.columns.values):
                ws.write(0, col_num, value, header_fmt)
            for i, col in enumerate(df_aud.columns):
                max_len = max(df_aud[col].astype(str).map(len).max(), len(str(col))) + 2
                ws.set_column(i, i, min(max_len, 40))
            for row_num in range(len(df_aud)):
                fmt = alt_fmt if row_num % 2 == 0 else normal_fmt
                for col_num in range(len(df_aud.columns)):
                    ws.write(row_num + 1, col_num, df_aud.iloc[row_num, col_num], fmt)
            ws.freeze_panes(1, 1)
    return output.getvalue()

def generar_excel_brandwatch(df_risk):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_risk.to_excel(writer, sheet_name='Riesgo_Reputacional', index=False, startrow=0)
        workbook = writer.book
        ws = writer.sheets['Riesgo_Reputacional']
        header_fmt = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#7352FF', 'border': 1, 'align': 'center'})
        for col_num, value in enumerate(df_risk.columns.values):
            ws.write(0, col_num, value, header_fmt)
        for i, col in enumerate(df_risk.columns):
            max_len = max(df_risk[col].astype(str).map(len).max(), len(str(col))) + 2
            ws.set_column(i, i, min(max_len, 60))
    return output.getvalue()

def generar_pdf_resumen(df_metrics, df_raw, period_value, period_type='monthly'):
    pdf = FPDF()
    pdf.add_page()
    if period_type == 'weekly':
        period_title = f"Semana {period_value}"
        prev_period = period_value - 1
        period_suffix = "WoW"
        df_raw['Week'] = df_raw['Date_Time'].dt.isocalendar().week
    else:
        month_name = calendar.month_name[period_value.month]
        period_title = f"{month_name} {period_value.year}"
        prev_period = period_value - 1
        period_suffix = "MoM"
        df_raw['YearMonth'] = df_raw['Date_Time'].dt.to_period('M')
    def clean_txt(text):
        if pd.isna(text): return ""
        return str(text).replace('"', "'").replace('\n', ' ').encode('latin-1', 'replace').decode('latin-1')
    def print_metric(label, val_str, delta_val, is_higher_better=True, is_pct=False):
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(55, 6, clean_txt(f"- {label}: {val_str}"), ln=0)
        if pd.isna(delta_val) or delta_val == 0:
            pdf.set_text_color(150, 150, 150)
            delta_str = "(=0.0)"
        else:
            if delta_val > 0: pdf.set_text_color(0, 209, 163) if is_higher_better else pdf.set_text_color(255, 82, 82)
            else: pdf.set_text_color(255, 82, 82) if is_higher_better else pdf.set_text_color(0, 209, 163)
            suffix = f"% {period_suffix}" if is_pct else f" {period_suffix}"
            delta_str = f"({delta_val:+.1f}{suffix})"
        pdf.cell(0, 6, clean_txt(delta_str), ln=1)
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(115, 82, 255)
    pdf.cell(0, 10, clean_txt(f"Resumen Ejecutivo C_OPS - Support - {period_title}"), ln=True, align='C')
    pdf.ln(5)
    audiences_in_period = df_metrics[df_metrics['Period'] == period_value]['Audience'].unique()
    for aud in ['Driver', 'Rider', 'B2B', 'Emergencias', 'Aeropuerto', 'Aeropuerto WhatsApp']:
        if aud not in audiences_in_period: continue
        curr_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Period'] == period_value)]
        prev_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Period'] == prev_period)]
        if not curr_df.empty:
            curr = curr_df.iloc[0]
            prev = prev_df.iloc[0] if not prev_df.empty else curr * 0
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, clean_txt(f"Audiencia: {aud}"), ln=True)
            vol_pct = ((curr['Contactos Recibidos'] - prev['Contactos Recibidos']) / prev['Contactos Recibidos'] * 100) if prev['Contactos Recibidos'] else 0
            nps_diff = curr['NPS'] - prev['NPS'] if pd.notna(prev['NPS']) else 0
            csat_diff = curr['CSAT (%)'] - prev['CSAT (%)'] if pd.notna(prev['CSAT (%)']) else 0
            firt_diff = curr['FiRT <24h (%)'] - prev['FiRT <24h (%)'] if pd.notna(prev['FiRT <24h (%)']) else 0
            reop_diff = curr['Ratio Reopen/Tickets (%)'] - prev['Ratio Reopen/Tickets (%)'] if pd.notna(prev['Ratio Reopen/Tickets (%)']) else 0
            nps_count = curr.get('NPS_Count', 0)
            nps_val_str = f"{curr['NPS']:.1f} ({int(nps_count)} encuestas)" if pd.notna(curr['NPS']) else "S/D"
            print_metric("Volumen", f"{curr['Contactos Recibidos']:,.0f}", vol_pct, False, True)
            print_metric("NPS Score", nps_val_str, nps_diff, True, False)
            print_metric("CSAT (%)", f"{curr['CSAT (%)']:.1f}%", csat_diff, True, True)
            print_metric("FiRT <24h", f"{curr['FiRT <24h (%)']:.1f}%", firt_diff, True, True)
            print_metric("Ratio Reopen", f"{curr['Ratio Reopen/Tickets (%)']:.1f}%", reop_diff, False, True)
            insight_nps = analizar_detractores(df_raw, aud, period_value, period_type)
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

# ============================================
# === INTERFAZ PRINCIPAL ===
# ============================================
st.markdown(f"<h1 style='color: {CABIFY_PURPLE};'>🚕 C_OPS Support Dashboard</h1>", unsafe_allow_html=True)
st.write("Dashboard de indicadores con análisis de Riesgo Reputacional.")

st.sidebar.markdown(f"<h3 style='color: {CABIFY_PURPLE};'>⚙️ Configuración</h3>", unsafe_allow_html=True)
period_type = st.sidebar.radio("📅 Tipo de Análisis", options=['monthly', 'weekly'], format_func=lambda x: 'Mensual' if x == 'monthly' else 'Semanal', index=0)
include_abibot = st.sidebar.checkbox("🤖 Incluir tickets Abi/Bot", value=False)

st.sidebar.divider()
st.sidebar.markdown("### 📁 Archivos")
file_main = st.file_uploader("📊 Archivo Maestro (CSV)", type=['csv'])
file_whatsapp = st.file_uploader("📱 Aeropuerto WhatsApp - Opcional", type=['csv'])
file_brandwatch = st.file_uploader("🔍 Brandwatch - Opcional", type=['csv', 'xlsx'])

if file_main is not None:
    with st.spinner('Procesando...'):
        df_raw = load_main_data(file_main, include_abibot=include_abibot)
        
        if file_whatsapp is not None:
            df_whatsapp = load_whatsapp_data(file_whatsapp)
            common_cols = ['Date_Time', 'Audience', 'Contact Type', 'NPS_Score', 'CSAT_Pct', 'FRT_Hours', 'FuRT_Hours', 'Reopen_Count', 'Tag_1', 'Tag_2', 'Tag_3', 'Ticket_Number']
            for col in common_cols:
                if col not in df_raw.columns: df_raw[col] = np.nan
                if col not in df_whatsapp.columns: df_whatsapp[col] = np.nan
            df_raw = pd.concat([df_raw[common_cols], df_whatsapp[common_cols]], ignore_index=True)
            st.sidebar.success("✅ WhatsApp cargado")
        
        df_brandwatch = None
        if file_brandwatch is not None:
            df_brandwatch = load_brandwatch_data(file_brandwatch)
            if df_brandwatch is not None: st.sidebar.success("✅ Brandwatch cargado")
        
        df_metrics = aggregate_data(df_raw, period_type=period_type)
        if period_type == 'weekly':
            df_raw['Week'] = df_raw['Date_Time'].dt.isocalendar().week
        else:
            df_raw['YearMonth'] = df_raw['Date_Time'].dt.to_period('M')
    
    st.sidebar.markdown(f"<h3 style='color: {CABIFY_PURPLE};'>📊 Filtros</h3>", unsafe_allow_html=True)
    all_audiences = ['Rider', 'Driver', 'B2B', 'Emergencias', 'Aeropuerto', 'Aeropuerto WhatsApp']
    audiences = [a for a in all_audiences if a in df_metrics['Audience'].unique()]
    selected_audience = st.sidebar.selectbox("Audiencia", audiences)
    
    df_filtered = df_metrics[df_metrics['Audience'] == selected_audience].sort_values('Period')
    available_periods = sorted(df_filtered['Period'].dropna().unique(), reverse=True)
    
    if period_type == 'monthly':
        today = datetime.now()
        current_period = pd.Period(f"{today.year}-{today.month:02d}", freq='M')
        available_periods = [p for p in available_periods if p < current_period]
    
    if len(available_periods) == 0:
        st.warning("No hay períodos completos disponibles.")
        st.stop()
    
    period_options = {str(p): f"Semana {p}" if period_type == 'weekly' else f"{calendar.month_name[p.month]} {p.year}" for p in available_periods}
    selected_period_str = st.sidebar.selectbox("Período", options=list(period_options.keys()), format_func=lambda x: period_options[x])
    selected_period = int(selected_period_str) if period_type == 'weekly' else pd.Period(selected_period_str)
    period_display = period_options[selected_period_str]
    
    # Descargas
    st.sidebar.divider()
    st.sidebar.subheader("📥 Descargas")
    _pdf = generar_pdf_resumen(df_metrics, df_raw.copy(), selected_period, period_type)
    st.sidebar.download_button("📄 PDF Resumen", data=_pdf, file_name=f"COPS_Informe_{selected_period_str}.pdf", mime="application/pdf")
    
    _excel = generar_excel_cabify(df_raw.copy(), audiences, period_type, selected_period, period_display)
    st.sidebar.download_button("📊 Excel Datos Crudos", data=_excel, file_name=f"Datos_{selected_period_str}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # Tabs principales
    tabs_list = ["📈 KPIs", "📊 Evolución", "🔍 Deep Dive"]
    if df_brandwatch is not None: tabs_list.append("⚠️ Riesgo Reputacional")
    tabs = st.tabs(tabs_list)
    
    with tabs[0]:
        st.markdown("### 💬 Resumen para Slack")
        slack_msg = generar_texto_slack(df_metrics, selected_period, period_type)
        st.code(slack_msg, language="markdown")
        st.divider()
        
        current_data = df_filtered[df_filtered['Period'] == selected_period]
        prev_period = selected_period - 1
        prev_data = df_filtered[df_filtered['Period'] == prev_period]
        
        if not current_data.empty:
            curr = current_data.iloc[0]
            prev = prev_data.iloc[0] if not prev_data.empty else current_data.iloc[0] * 0
            
            def calc_delta_pct(c, p): return f"{((c - p) / p * 100):+.2f}%" if p != 0 and pd.notna(p) else "0.0%"
            def calc_delta_abs(c, p): return f"{c - p:+.2f}" if pd.notna(p) else "+0.0"
            
            st.markdown(f"### {period_display} - {selected_audience}")
            
            st.markdown("#### I. Performance General")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Contactos", f"{curr['Contactos Recibidos']:,.0f}", calc_delta_pct(curr['Contactos Recibidos'], prev['Contactos Recibidos']), delta_color="inverse")
            c2.metric("Tickets", f"{curr['Contactos Ticket']:,.0f}", calc_delta_pct(curr['Contactos Ticket'], prev['Contactos Ticket']), delta_color="inverse")
            c3.metric("Chats", f"{curr['Contactos Chat']:,.0f}", calc_delta_pct(curr['Contactos Chat'], prev['Contactos Chat']), delta_color="inverse")
            c4.metric("Calls", f"{curr['Contactos Call']:,.0f}", calc_delta_pct(curr['Contactos Call'], prev['Contactos Call']), delta_color="inverse")
            
            c5, c6 = st.columns(2)
            c5.metric("NPS", f"{curr['NPS']:.2f}", calc_delta_abs(curr['NPS'], prev['NPS']), help=f"{int(curr.get('NPS_Count', 0))} encuestas")
            c6.metric("CSAT", f"{curr['CSAT (%)']:.1f}%", calc_delta_abs(curr['CSAT (%)'], prev['CSAT (%)']) + "%")
            
            st.divider()
            st.markdown("#### II. Calidad de Gestión")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("TMO (Hrs)", f"{curr['TMO (Hrs)']:.2f}", calc_delta_abs(curr['TMO (Hrs)'], prev['TMO (Hrs)']), delta_color="inverse")
            c2.metric("FiRT <24h", f"{curr['FiRT <24h (%)']:.2f}%", calc_delta_abs(curr['FiRT <24h (%)'], prev['FiRT <24h (%)']) + "%")
            c3.metric("FuRT <36h", f"{curr['FuRT <36h (%)']:.2f}%", calc_delta_abs(curr['FuRT <36h (%)'], prev['FuRT <36h (%)']) + "%")
            c4.metric("Reopen", f"{curr['Ratio Reopen/Tickets (%)']:.2f}%", calc_delta_abs(curr['Ratio Reopen/Tickets (%)'], prev['Ratio Reopen/Tickets (%)']) + "%", delta_color="inverse")
    
    with tabs[1]:
        st.markdown(f"### 📊 Evolución - {selected_audience}")
        df_trend = df_filtered[df_filtered['Period'] <= selected_period].sort_values('Period')
        
        if len(df_trend) > 0:
            fig_vol = px.bar(df_trend, x='PeriodLabel', y='Contactos Recibidos', color_discrete_sequence=[CABIFY_PURPLE], text='Contactos Recibidos')
            fig_vol.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_vol.update_layout(plot_bgcolor="white", xaxis_title="Período", yaxis_title="Contactos")
            st.plotly_chart(fig_vol, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_nps = px.line(df_trend, x='PeriodLabel', y='NPS', markers=True, color_discrete_sequence=[CABIFY_PURPLE])
                fig_nps.update_layout(plot_bgcolor="white", title="NPS", yaxis_range=[-100, 100])
                fig_nps.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_nps, use_container_width=True)
            with col2:
                fig_csat = px.line(df_trend, x='PeriodLabel', y='CSAT (%)', markers=True, color_discrete_sequence=[CABIFY_SECONDARY])
                fig_csat.update_layout(plot_bgcolor="white", title="CSAT (%)", yaxis_range=[0, 100])
                st.plotly_chart(fig_csat, use_container_width=True)
    
    with tabs[2]:
        st.markdown(f"### 🔍 Deep Dive: Detractores ({selected_audience} - {period_display})")
        resumen = analizar_detractores(df_raw, selected_audience, selected_period, period_type)
        if "Excelente" in resumen: st.success("✅ " + resumen)
        else: st.error("🚨 " + resumen)
        
        st.divider()
        if period_type == 'weekly':
            df_raw_f = df_raw[(df_raw['Audience'] == selected_audience) & (df_raw['Week'] == selected_period)]
        else:
            df_raw_f = df_raw[(df_raw['Audience'] == selected_audience) & (df_raw['YearMonth'] == selected_period)]
        
        if not df_raw_f.empty and 'Tag_3' in df_raw_f.columns:
            top_tags = df_raw_f['Tag_3'].value_counts().head(10).reset_index()
            top_tags.columns = ['Motivo', 'Volumen']
            fig = px.bar(top_tags, x='Volumen', y='Motivo', orientation='h', color_discrete_sequence=[CABIFY_SECONDARY])
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab Riesgo Reputacional
    if df_brandwatch is not None and len(tabs) > 3:
        with tabs[3]:
            st.markdown("### ⚠️ Riesgo Reputacional (Brandwatch)")
            st.info("Análisis del período actual - Solo menciones negativas clasificadas.")
            
            df_risk = df_brandwatch[~df_brandwatch['Categoría'].isin(['Ruido Mediático', 'Otros / Neutro', 'Desconocido'])].copy()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Menciones", len(df_brandwatch))
            col2.metric("Menciones de Riesgo", len(df_risk), delta=f"{len(df_risk)/len(df_brandwatch)*100:.1f}%", delta_color="inverse")
            col3.metric("Frustración Crítica", len(df_risk[df_risk['Categoría'] == 'Frustración Crítica']), delta_color="inverse")
            col4.metric("Seguridad", len(df_risk[df_risk['Categoría'] == 'Seguridad']), delta_color="inverse")
            
            st.divider()
            st.markdown("#### Distribución por Categoría de Riesgo")
            
            cat_counts = df_risk['Categoría'].value_counts().reset_index()
            cat_counts.columns = ['Categoría', 'Volumen']
            
            fig = px.bar(cat_counts, x='Volumen', y='Categoría', orientation='h', 
                        color='Categoría', color_discrete_map={
                            'Frustración Crítica': '#FF5252',
                            'Seguridad': '#FF8A80',
                            'Cobros y Tarifas': '#FFB347',
                            'Calidad de Servicio': '#FFD54F',
                            'Disponibilidad / App': '#81D4FA'
                        })
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.markdown("#### Ejemplos de Menciones Críticas")
            if 'Texto' in df_risk.columns:
                for cat in ['Frustración Crítica', 'Seguridad', 'Cobros y Tarifas']:
                    df_cat = df_risk[df_risk['Categoría'] == cat]
                    if not df_cat.empty:
                        st.markdown(f"**{cat}** ({len(df_cat)} menciones)")
                        sample = df_cat['Texto'].dropna().head(2).tolist()
                        for s in sample:
                            st.caption(f"• {s[:200]}...")
            
            st.divider()
            st.markdown("#### 📥 Exportar Riesgo Reputacional")
            _excel_bw = generar_excel_brandwatch(df_risk)
            st.download_button("📊 Excel Casos Críticos", data=_excel_bw, file_name="Riesgo_Reputacional.xlsx")
    
    # Info sidebar
    st.sidebar.divider()
    abibot_status = "✅ Incluidos" if include_abibot else "❌ Excluidos"
    st.sidebar.caption(f"**Config:** {period_type.title()} | Abi/Bot: {abibot_status}\n**Registros:** {len(df_raw):,}")
