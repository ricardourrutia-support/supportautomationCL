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

# Configuración estilo Cabify Minimalista
st.set_page_config(page_title="Cabify Support Dashboard", layout="wide", initial_sidebar_state="expanded")
CABIFY_PURPLE = "#7352FF"
CABIFY_SECONDARY = "#00D1A3"
CABIFY_LIGHT_PURPLE = "#E2D9FF"

# Columnas Core
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
    
    # FILTRO 1: Include Contacts = 'Rest'
    if 'Include_Contacts' in df.columns:
        df = df[df['Include_Contacts'].astype(str).str.strip().str.lower() == 'rest']
    
    # FILTRO 2: Excluir Delivery
    if 'Service_Type' in df.columns:
        df = df[~df['Service_Type'].astype(str).str.lower().str.contains('delivery', na=False)]
    
    # FILTRO 3: Automated (configurable)
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
        
        valid_em = ['tn emergencias drivers', 'tn energencias drivers', 
                    'tn emergencias rider', 'tn energencias rider',
                    'tn emergencias', 'tn energencias']
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

# --- CARGA DEL REPORTE AEROPUERTO WHATSAPP ---
@st.cache_data
def load_whatsapp_data(filepath):
    df = read_csv_robust(filepath)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Mapeo específico para este CSV
    mapping = {
        'Created At Local Dt': 'Date_Time',
        'Ticket Number': 'Ticket_Number',
        '# First Reply Time (Min)': 'FRT_Min',
        '# Full Resolution Time (Hours)': 'FuRT_Hours',
        '% CSAT': 'CSAT_Pct',
        'NPS Score': 'NPS_Score',
        'ES Output Tags 1st Level v2': 'Tag_1',
        'ES Output Tags 2nd Level v2': 'Tag_2',
        'ES Output Tags 3rd Level v2': 'Tag_3',
        '# Tickets': 'Ticket_Count'
    }
    
    existing_mapping = {k: v for k, v in mapping.items() if k in df.columns}
    df = df[list(existing_mapping.keys())].rename(columns=existing_mapping)
    
    # Parsear números
    for c in ['FRT_Min', 'FuRT_Hours', 'CSAT_Pct', 'NPS_Score']:
        if c in df.columns: df[c] = df[c].apply(parse_num)
    
    # Convertir FRT de minutos a horas
    if 'FRT_Min' in df.columns:
        df['FRT_Hours'] = df['FRT_Min'] / 60
    
    # Parsear fecha
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y', errors='coerce')
    
    # Agregar audiencia fija
    df['Audience'] = 'Aeropuerto WhatsApp'
    df['Contact Type'] = 'WhatsApp'
    df['Reopen_Count'] = 0  # No tenemos este dato
    df['Chat_Missed'] = 0
    
    return df

# --- CLASIFICADOR DE MENCIONES BRANDWATCH ---
def clasificar_mencion(texto):
    """Clasifica una mención de redes sociales en categorías de riesgo."""
    if not isinstance(texto, str): return "Otros / Neutro"
    t = texto.lower()
    
    # Excluir menciones claramente positivas o neutras
    if any(p in t for p in ["gracias cabify", "buen servicio", "excelente", "recomiendo", "genial", "bacan", "buena onda", "felicidades"]): 
        return "Otros / Neutro"
    
    # Ruido mediático (noticias, política, no es queja)
    if any(p in t for p in ["ley uber", "gobierno", "ministro", "noticia", "congreso", "senado", "proyecto de ley"]): 
        return "Ruido Mediático"
    
    # Cobros y Tarifas - quejas sobre precios o cobros
    if any(p in t for p in ["cobro", "cobraron", "estafa", "robo", "carísimo", "caro", "precio", "tarifa", "doble cobro", "cobro indebido", "me robaron"]):
        return "Cobros y Tarifas"
    
    # Calidad de Servicio - quejas sobre conductores o vehículos
    if any(p in t for p in ["conductor", "chofer", "pésimo", "grosero", "sucio", "olor", "manejo", "mal servicio", "pésimo servicio"]):
        return "Calidad de Servicio"
    
    # Disponibilidad / App - problemas con la app o disponibilidad
    if any(p in t for p in ["no hay", "nadie acepta", "canceló", "cancelaron", "demora", "nunca llegó", "esperando", "no llega", "app no funciona", "no funciona"]):
        return "Disponibilidad / App"
    
    # Seguridad - situaciones de riesgo
    if any(p in t for p in ["miedo", "acoso", "peligro", "ruta extraña", "desvió", "inseguro", "asaltaron"]):
        return "Seguridad"
    
    # Frustración general - insultos o quejas fuertes DIRIGIDAS a Cabify
    if any(p in t for p in ["mierda", "horrible", "basura", "nunca más", "peor servicio", "penca", "malo", "odio"]):
        return "Frustración Crítica"
    
    return "Otros / Neutro"

# --- CARGA DE BRANDWATCH ---
@st.cache_data
def load_brandwatch_data(filepath):
    """Carga y procesa archivo de Brandwatch"""
    try:
        filepath.seek(0)
        content = filepath.read()
        text = content.decode('utf-8')
        lines = text.split('\n')
        
        # Encontrar la fila con los headers
        skip = 0
        for i, line in enumerate(lines[:20]):
            if 'Snippet' in line or 'Full Text' in line:
                skip = i
                break
        
        filepath.seek(0)
        df = pd.read_csv(filepath, skiprows=skip, encoding='utf-8', low_memory=False)
        
        # Limpiar nombres de columnas
        df.columns = [str(c).replace('"', '').strip() for c in df.columns]
        
        # Encontrar columna de texto
        txt_col = next((c for c in df.columns if c.lower() in ['snippet', 'full text', 'text']), None)
        
        if txt_col:
            df['Texto'] = df[txt_col]
            df['Categoría'] = df[txt_col].apply(clasificar_mencion)
        
        # Parsear fecha
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Obtener Sentiment si existe
        if 'Sentiment' in df.columns:
            df['Sentimiento'] = df['Sentiment']
        
        return df
    except Exception as e:
        st.error(f"Error al cargar Brandwatch: {e}")
        return None

# --- EXCEL CON FORMATO CABIFY ---
def generar_excel_cabify(df_raw, audiences, period_type, selected_period, period_display):
    """Genera Excel de datos crudos con formato Cabify (colores, headers estilizados)"""
    output = io.BytesIO()
    
    # Filtrar por período
    if period_type == 'weekly':
        df_period = df_raw[df_raw['Week'] == selected_period].copy()
    else:
        df_period = df_raw[df_raw['YearMonth'] == selected_period].copy()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formatos Cabify
        header_format = workbook.add_format({
            'bold': True,
            'font_color': 'white',
            'bg_color': '#7352FF',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'font_size': 11,
            'font_name': 'Arial'
        })
        
        for aud in audiences:
            df_aud = df_period[df_period['Audience'] == aud].copy()
            if df_aud.empty: continue
            
            # Reordenar: Ticket_Number primero
            cols = df_aud.columns.tolist()
            if 'Ticket_Number' in cols:
                cols.remove('Ticket_Number')
                cols = ['Ticket_Number'] + cols
                df_aud = df_aud[cols]
            
            # Limpiar nombre de hoja (max 31 chars, sin caracteres especiales)
            sheet_name = aud[:31].replace('/', '-').replace('\\', '-')
            
            # Escribir datos usando pandas (más seguro)
            df_aud.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1, header=False)
            
            worksheet = writer.sheets[sheet_name]
            
            # Aplicar formato solo a headers
            for col_num, value in enumerate(df_aud.columns.values):
                worksheet.write(0, col_num, str(value), header_format)
            
            # Ajustar anchos de columna
            for i, col in enumerate(df_aud.columns):
                worksheet.set_column(i, i, 18)
            
            # Congelar primera fila y primera columna
            worksheet.freeze_panes(1, 1)
    
    return output.getvalue()

# --- EXCEL DE BRANDWATCH ---
def generar_excel_brandwatch(df_risk):
    """Genera Excel de casos críticos de Brandwatch con formato Cabify"""
    output = io.BytesIO()
    
    # Seleccionar columnas relevantes
    cols_export = ['Date', 'Texto', 'Categoría', 'Sentimiento', 'Page Type', 'Url']
    cols_exist = [c for c in cols_export if c in df_risk.columns]
    df_export = df_risk[cols_exist].copy()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, sheet_name='Riesgo_Reputacional', index=False, startrow=0)
        
        workbook = writer.book
        worksheet = writer.sheets['Riesgo_Reputacional']
        
        header_format = workbook.add_format({
            'bold': True,
            'font_color': 'white',
            'bg_color': '#7352FF',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'font_size': 11
        })
        
        # Aplicar formato a headers
        for col_num, value in enumerate(df_export.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Ajustar anchos
        for i, col in enumerate(df_export.columns):
            try:
                col_data = df_export[col].astype(str)
                max_data_len = col_data.str.len().max()
                if pd.isna(max_data_len):
                    max_data_len = 10
                max_len = max(int(max_data_len), len(str(col))) + 2
                worksheet.set_column(i, i, min(max_len, 60))
            except:
                worksheet.set_column(i, i, 20)
        
        worksheet.freeze_panes(1, 0)
    
    return output.getvalue()

# --- AGREGACIÓN GENÉRICA CON SLAs POR AUDIENCIA ---
# SLAs por audiencia:
# - Rider, Driver, B2B, Emergencias: FiRT 24h, FuRT 36h
# - Aeropuerto (ticket): FiRT 24h, FuRT 120h
# - Aeropuerto WhatsApp: FiRT 1h, FuRT 1h

SLA_CONFIG = {
    'Rider': {'firt': 24, 'furt': 36},
    'Driver': {'firt': 24, 'furt': 36},
    'B2B': {'firt': 24, 'furt': 36},
    'Emergencias': {'firt': 24, 'furt': 36},
    'Aeropuerto': {'firt': 24, 'furt': 120},
    'Aeropuerto WhatsApp': {'firt': 1, 'furt': 1},
}

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
        # Obtener la audiencia del grupo
        aud = grp['Audience'].iloc[0] if 'Audience' in grp.columns and len(grp) > 0 else 'Rider'
        sla = SLA_CONFIG.get(aud, {'firt': 24, 'furt': 36})
        
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
        
        # SLAs dinámicos según audiencia
        valid_res = grp['FuRT_Hours'].dropna() if 'FuRT_Hours' in grp.columns else pd.Series()
        res['FuRT_SLA'] = sla['furt']  # Guardar el SLA usado
        res[f'FuRT SLA (%)'] = (valid_res < sla['furt']).mean() * 100 if len(valid_res) > 0 else np.nan
        
        valid_frt = grp['FRT_Hours'].dropna() if 'FRT_Hours' in grp.columns else pd.Series()
        res['FiRT_SLA'] = sla['firt']  # Guardar el SLA usado
        res[f'FiRT SLA (%)'] = (valid_frt < sla['firt']).mean() * 100 if len(valid_frt) > 0 else np.nan
        
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
        result = df.groupby(['Year', 'Period', 'PeriodLabel', 'Audience']).apply(aggs).reset_index()
    else:
        result = df.groupby(['Period', 'PeriodLabel', 'Audience']).apply(aggs).reset_index()
    
    return result

# --- FUNCIÓN DE ANÁLISIS DE DETRACTORES ---
def analizar_detractores(df_raw, aud, period_value, period_type='monthly'):
    if period_type == 'weekly':
        if 'Week' not in df_raw.columns:
            df_raw['Week'] = df_raw['Date_Time'].dt.isocalendar().week
        detractores = df_raw[(df_raw['Audience'] == aud) & (df_raw['Week'] == period_value) & (df_raw['NPS_Score'] == -100)]
    else:
        if 'YearMonth' not in df_raw.columns:
            df_raw['YearMonth'] = df_raw['Date_Time'].dt.to_period('M')
        detractores = df_raw[(df_raw['Audience'] == aud) & (df_raw['YearMonth'] == period_value) & (df_raw['NPS_Score'] == -100)]
    
    if detractores.empty:
        return "Excelente: No hay registros de encuestas NPS -100 para esta audiencia en el período seleccionado."
    
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

# --- FUNCIÓN AUXILIAR: TOP MOTIVOS ---
def obtener_top_motivos(df_raw, audience, period_value, period_type='monthly', n=3, solo_detractores=True):
    """Obtiene los top N motivos (Tag_3) para una audiencia en un período"""
    if period_type == 'weekly':
        if 'Week' not in df_raw.columns:
            df_raw['Week'] = df_raw['Date_Time'].dt.isocalendar().week
        df_filtered = df_raw[(df_raw['Audience'] == audience) & (df_raw['Week'] == period_value)]
    else:
        if 'YearMonth' not in df_raw.columns:
            df_raw['YearMonth'] = df_raw['Date_Time'].dt.to_period('M')
        df_filtered = df_raw[(df_raw['Audience'] == audience) & (df_raw['YearMonth'] == period_value)]
    
    # Para audiencias con NPS, filtrar solo detractores
    if solo_detractores and audience != 'Aeropuerto WhatsApp':
        df_filtered = df_filtered[df_filtered['NPS_Score'] == -100]
    
    if df_filtered.empty or 'Tag_3' not in df_filtered.columns:
        return []
    
    top = df_filtered['Tag_3'].dropna().value_counts().head(n)
    return [(motivo, int(vol)) for motivo, vol in top.items()]

# --- TEXTO SLACK ---
def generar_texto_slack(df_metrics, df_raw, period_value, period_type='monthly', df_brandwatch=None):
    if period_type == 'weekly':
        title = f"📣 C_OPS Weekly Update - Support - Semana {period_value} 📣\n"
        period_suffix = "WoW"
    else:
        month_name = calendar.month_name[period_value.month]
        title = f"📣 C_OPS Monthly Update - Support - {month_name} {period_value.year} 📣\n"
        period_suffix = "MoM"
    
    lines = []
    lines.append(title)
    lines.append("📄 Resumen Ejecutivo (PDF): [Pega el link a tu Drive aquí]")
    lines.append("📊 Datos Crudos por Audiencias (Excel): [Pega el link a tu Drive aquí]\n")
    
    # === SECCIÓN 1: MÉTRICAS POR AUDIENCIA ===
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("📊 MÉTRICAS POR AUDIENCIA")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━\n")

    if period_type == 'weekly':
        audiences_in_period = df_metrics[df_metrics['Period'] == period_value]['Audience'].unique()
        prev_period = period_value - 1
    else:
        audiences_in_period = df_metrics[df_metrics['Period'] == period_value]['Audience'].unique()
        prev_period = period_value - 1
    
    iconos = {'Rider': '🚶', 'Driver': '🚘', 'B2B': '🏢', 'Emergencias': '🚑', 'Aeropuerto': '✈️', 'Aeropuerto WhatsApp': '📱'}
    
    for aud in ['Rider', 'Driver', 'B2B', 'Emergencias', 'Aeropuerto', 'Aeropuerto WhatsApp']:
        if aud not in audiences_in_period: continue
        
        curr_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Period'] == period_value)]
        prev_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Period'] == prev_period)]

        if curr_df.empty: continue
        curr = curr_df.iloc[0]
        prev = prev_df.iloc[0] if not prev_df.empty else curr * 0

        vol_curr = curr['Contactos Recibidos']
        vol_prev = prev['Contactos Recibidos']
        vol_change = ((vol_curr - vol_prev) / vol_prev * 100) if vol_prev else 0

        nps_curr = curr['NPS']
        nps_prev = prev['NPS']
        nps_change = nps_curr - nps_prev if pd.notna(nps_prev) else 0
        nps_count = curr.get('NPS_Count', 0)

        csat_curr = curr['CSAT (%)']
        csat_change = csat_curr - prev['CSAT (%)'] if pd.notna(prev['CSAT (%)']) else 0

        firt_curr = curr['FiRT SLA (%)']
        reop_curr = curr['Ratio Reopen/Tickets (%)']

        icon = iconos.get(aud, '📊')
        lines.append(f"{icon} {aud.upper()}")
        lines.append(f"• Volumen: {vol_curr:,.0f} ({vol_change:+.1f}% {period_suffix})")
        
        nps_str = f"{nps_curr:.1f} ({nps_change:+.1f}) | {int(nps_count)} enc." if pd.notna(nps_curr) else "S/D"
        csat_str = f"{csat_curr:.1f}% ({csat_change:+.1f}%)" if pd.notna(csat_curr) else "S/D"
        lines.append(f"• Calidad: NPS {nps_str} | CSAT {csat_str}")
        
        firt_str = f"{firt_curr:.1f}%" if pd.notna(firt_curr) else "S/D"
        reop_str = f"{reop_curr:.1f}%" if pd.notna(reop_curr) else "S/D"
        lines.append(f"• Eficiencia: SLA 1ra Rsp: {firt_str} | Reopen: {reop_str}\n")
    
    # === SECCIÓN 2: MOTIVOS DE CONTACTO ===
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("🔍 MOTIVOS PRINCIPALES")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    for aud in ['Rider', 'Driver', 'B2B', 'Emergencias', 'Aeropuerto', 'Aeropuerto WhatsApp']:
        if aud not in audiences_in_period: continue
        
        icon = iconos.get(aud, '📊')
        
        if aud == 'Aeropuerto WhatsApp':
            # Para WhatsApp: motivos más recurrentes (sin NPS)
            motivos = obtener_top_motivos(df_raw, aud, period_value, period_type, n=3, solo_detractores=False)
            if motivos:
                lines.append(f"{icon} {aud} - Motivos más recurrentes:")
                for motivo, vol in motivos:
                    lines.append(f"   • {motivo}: {vol} casos")
                lines.append("")
        else:
            # Para otras audiencias: motivos en detractores NPS -100
            motivos = obtener_top_motivos(df_raw, aud, period_value, period_type, n=3, solo_detractores=True)
            if motivos:
                lines.append(f"{icon} {aud} - Top fricciones NPS -100:")
                for motivo, vol in motivos:
                    lines.append(f"   • {motivo}: {vol} casos")
                lines.append("")
    
    # === SECCIÓN 3: RIESGO REPUTACIONAL ===
    if df_brandwatch is not None:
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("⚠️ RIESGO REPUTACIONAL")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        total_menciones = len(df_brandwatch)
        
        # Filtrar por Sentiment = 'negative'
        if 'Sentimiento' in df_brandwatch.columns:
            df_negativas = df_brandwatch[df_brandwatch['Sentimiento'].astype(str).str.lower() == 'negative']
        elif 'Sentiment' in df_brandwatch.columns:
            df_negativas = df_brandwatch[df_brandwatch['Sentiment'].astype(str).str.lower() == 'negative']
        else:
            df_negativas = pd.DataFrame()
        
        menciones_negativas = len(df_negativas)
        pct_negativas = (menciones_negativas / total_menciones * 100) if total_menciones > 0 else 0
        
        lines.append(f"📊 Total menciones analizadas: {total_menciones:,}")
        lines.append(f"⚠️ Menciones negativas: {menciones_negativas:,} ({pct_negativas:.1f}%)\n")
        
        if menciones_negativas > 0 and 'Categoría' in df_negativas.columns:
            lines.append("Top tipos de reclamos:")
            cat_counts = df_negativas['Categoría'].value_counts().head(5)
            for cat, vol in cat_counts.items():
                lines.append(f"   • {cat}: {vol} menciones")
    
    return "\n".join(lines)

# --- GENERAR GRÁFICO DE EVOLUCIÓN PARA PDF ---
def crear_grafico_evolucion(df_trend, metric_name, color, title, y_range=None, add_zero_line=False):
    """Crea un gráfico de evolución para incluir en PDFs"""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    x_labels = df_trend['PeriodLabel'].values
    y_values = df_trend[metric_name].values
    
    ax.plot(x_labels, y_values, color=color, marker='o', markersize=6, linewidth=2)
    ax.fill_between(x_labels, y_values, alpha=0.2, color=color)
    
    if add_zero_line:
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    if y_range:
        ax.set_ylim(y_range)
    
    ax.set_title(title, fontsize=10, fontweight='bold', color='#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, v in enumerate(y_values):
        if pd.notna(v):
            ax.annotate(f'{v:.1f}', (x_labels[i], v), textcoords="offset points", 
                       xytext=(0, 8), ha='center', fontsize=8, color=color)
    
    plt.tight_layout()
    return fig

# --- PDF 1: REPORTE VERTICAL CLÁSICO ---
def generar_pdf_resumen(df_metrics, df_raw, period_value, period_type='monthly', df_brandwatch=None):
    pdf = FPDF()
    pdf.add_page()
    
    if period_type == 'weekly':
        period_title = f"Semana {period_value}"
        prev_period = period_value - 1
        period_suffix = "WoW"
        # Para filtrar df_raw
        df_raw['Week'] = df_raw['Date_Time'].dt.isocalendar().week
        period_col = 'Week'
    else:
        month_name = calendar.month_name[period_value.month]
        period_title = f"{month_name} {period_value.year}"
        prev_period = period_value - 1
        period_suffix = "MoM"
        df_raw['YearMonth'] = df_raw['Date_Time'].dt.to_period('M')
        period_col = 'YearMonth'
    
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
            firt_diff = curr['FiRT SLA (%)'] - prev['FiRT SLA (%)'] if pd.notna(prev['FiRT SLA (%)']) else 0
            reop_diff = curr['Ratio Reopen/Tickets (%)'] - prev['Ratio Reopen/Tickets (%)'] if pd.notna(prev['Ratio Reopen/Tickets (%)']) else 0
            
            nps_count = curr.get('NPS_Count', 0)
            nps_val_str = f"{curr['NPS']:.1f} ({int(nps_count)} encuestas)" if pd.notna(curr['NPS']) else "S/D"

            print_metric_line("Volumen", f"{curr['Contactos Recibidos']:,.0f}", vol_pct, is_higher_better=False, is_pct=True)
            print_metric_line("NPS Score", nps_val_str, nps_diff, is_higher_better=True, is_pct=False)
            print_metric_line("CSAT (%)", f"{curr['CSAT (%)']:.1f}%", csat_diff, is_higher_better=True, is_pct=True)
            print_metric_line("FiRT SLA", f"{curr['FiRT SLA (%)']:.1f}%", firt_diff, is_higher_better=True, is_pct=True)
            print_metric_line("Ratio Reopen", f"{curr['Ratio Reopen/Tickets (%)']:.1f}%", reop_diff, is_higher_better=False, is_pct=True)

            # Análisis de detractores usando el período correcto
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
    
    # === PÁGINA DE EVOLUCIÓN ===
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(115, 82, 255)
    pdf.cell(0, 10, clean_txt(f"Evolucion de Indicadores - {period_title}"), ln=True, align='C')
    pdf.ln(5)
    
    for aud in ['Driver', 'Rider', 'B2B', 'Emergencias', 'Aeropuerto', 'Aeropuerto WhatsApp']:
        if aud not in audiences_in_period: continue
        
        df_trend = df_metrics[df_metrics['Audience'] == aud].sort_values('Period')
        
        # Filtrar solo períodos completos (excluir el período actual si es incompleto)
        if period_type == 'monthly':
            today = datetime.now()
            current_period = pd.Period(f"{today.year}-{today.month:02d}", freq='M')
            df_trend = df_trend[df_trend['Period'] <= period_value]
        
        if len(df_trend) > 1:
            pdf.set_font("Arial", 'B', 11)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 6, clean_txt(f"Audiencia: {aud}"), ln=True)
            
            try:
                # Crear mini gráficos de evolución
                fig_vol = crear_grafico_evolucion(df_trend, 'Contactos Recibidos', CABIFY_PURPLE, 'Volumen')
                img_path_vol = f"evol_vol_{aud}.png"
                fig_vol.savefig(img_path_vol, dpi=150, bbox_inches='tight')
                plt.close(fig_vol)
                
                fig_nps = crear_grafico_evolucion(df_trend, 'NPS', CABIFY_SECONDARY, 'NPS', y_range=[-100, 100], add_zero_line=True)
                img_path_nps = f"evol_nps_{aud}.png"
                fig_nps.savefig(img_path_nps, dpi=150, bbox_inches='tight')
                plt.close(fig_nps)
                
                # Insertar en PDF
                pdf.image(img_path_vol, x=10, y=pdf.get_y(), w=90)
                pdf.image(img_path_nps, x=105, y=pdf.get_y(), w=90)
                pdf.ln(35)
                
                fig_csat = crear_grafico_evolucion(df_trend, 'CSAT (%)', '#FFB347', 'CSAT (%)', y_range=[0, 100])
                img_path_csat = f"evol_csat_{aud}.png"
                fig_csat.savefig(img_path_csat, dpi=150, bbox_inches='tight')
                plt.close(fig_csat)
                
                fig_firt = crear_grafico_evolucion(df_trend, 'FiRT SLA (%)', '#FF6B6B', 'FiRT SLA (%)', y_range=[0, 100])
                img_path_firt = f"evol_firt_{aud}.png"
                fig_firt.savefig(img_path_firt, dpi=150, bbox_inches='tight')
                plt.close(fig_firt)
                
                pdf.image(img_path_csat, x=10, y=pdf.get_y(), w=90)
                pdf.image(img_path_firt, x=105, y=pdf.get_y(), w=90)
                pdf.ln(40)
                
                # Limpiar archivos temporales
                for f in [img_path_vol, img_path_nps, img_path_csat, img_path_firt]:
                    if os.path.exists(f): os.remove(f)
                    
            except Exception as e:
                pdf.set_font("Arial", 'I', 9)
                pdf.cell(0, 5, clean_txt(f"No se pudo generar graficos de evolucion."), ln=True)
            
            pdf.ln(5)

    # === PÁGINA DE MOTIVOS PRINCIPALES ===
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(115, 82, 255)
    pdf.cell(0, 10, clean_txt(f"Deep Dive: Motivos de Contacto - {period_title}"), ln=True, align='C')
    pdf.ln(5)
    
    def clean_txt_local(text):
        if pd.isna(text): return ""
        return str(text).replace('"', "'").replace('\n', ' ').encode('latin-1', 'replace').decode('latin-1')
    
    iconos_txt = {'Rider': '[Rider]', 'Driver': '[Driver]', 'B2B': '[B2B]', 'Emergencias': '[Emerg]', 'Aeropuerto': '[Aero]', 'Aeropuerto WhatsApp': '[AeroWA]'}
    
    for aud in ['Driver', 'Rider', 'B2B', 'Emergencias', 'Aeropuerto', 'Aeropuerto WhatsApp']:
        if aud not in audiences_in_period: continue
        
        # Filtrar datos del período para esta audiencia
        if period_type == 'weekly':
            df_aud_period = df_raw[(df_raw['Audience'] == aud) & (df_raw['Week'] == period_value)]
        else:
            df_aud_period = df_raw[(df_raw['Audience'] == aud) & (df_raw['YearMonth'] == period_value)]
        
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(115, 82, 255)
        pdf.cell(0, 7, clean_txt_local(f"{iconos_txt.get(aud, '')} {aud}"), ln=True)
        
        if aud == 'Aeropuerto WhatsApp':
            # Para WhatsApp: solo motivos recurrentes (sin NPS)
            total_contactos = len(df_aud_period)
            pdf.set_font("Arial", 'I', 9)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 5, f"Total contactos: {total_contactos} | Sin encuestas NPS disponibles", ln=True)
            
            pdf.set_font("Arial", 'B', 9)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 5, "Top 5 motivos de contacto (Tag Nivel 3):", ln=True)
            
            motivos = obtener_top_motivos(df_raw, aud, period_value, period_type, n=5, solo_detractores=False)
            if motivos:
                pdf.set_font("Arial", '', 9)
                for motivo, vol in motivos:
                    motivo_clean = clean_txt_local(str(motivo)[:45])
                    pdf.cell(0, 5, f"  - {motivo_clean}: {vol} casos", ln=True)
            else:
                pdf.set_font("Arial", 'I', 9)
                pdf.cell(0, 5, "  Sin datos disponibles.", ln=True)
        else:
            # Para otras audiencias: detractores NPS -100 + distribución general
            df_detractores = df_aud_period[df_aud_period['NPS_Score'] == -100]
            universo_nps100 = len(df_detractores)
            total_contactos = len(df_aud_period)
            
            pdf.set_font("Arial", 'I', 9)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 5, f"Total contactos: {total_contactos} | Universo NPS -100: {universo_nps100} casos", ln=True)
            
            # Top fricciones en detractores
            pdf.set_font("Arial", 'B', 9)
            pdf.set_text_color(255, 82, 82)
            pdf.cell(0, 5, f"Top fricciones en detractores NPS -100:", ln=True)
            
            motivos_nps = obtener_top_motivos(df_raw, aud, period_value, period_type, n=5, solo_detractores=True)
            if motivos_nps:
                pdf.set_font("Arial", '', 9)
                pdf.set_text_color(0, 0, 0)
                for motivo, vol in motivos_nps:
                    motivo_clean = clean_txt_local(str(motivo)[:45])
                    pdf.cell(0, 5, f"  - {motivo_clean}: {vol} casos", ln=True)
            else:
                pdf.set_font("Arial", 'I', 9)
                pdf.set_text_color(0, 209, 163)
                pdf.cell(0, 5, "  Sin detractores NPS -100 en este periodo.", ln=True)
            
            # Distribución general de motivos
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 9)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 5, "Distribucion general de motivos (Tag Nivel 3):", ln=True)
            
            motivos_general = obtener_top_motivos(df_raw, aud, period_value, period_type, n=5, solo_detractores=False)
            if motivos_general:
                pdf.set_font("Arial", '', 9)
                for motivo, vol in motivos_general:
                    motivo_clean = clean_txt_local(str(motivo)[:45])
                    pdf.cell(0, 5, f"  - {motivo_clean}: {vol} casos", ln=True)
        
        pdf.ln(4)
    
    # === SECCIÓN DE RIESGO REPUTACIONAL ===
    if df_brandwatch is not None:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(115, 82, 255)
        pdf.cell(0, 10, clean_txt_local("Riesgo Reputacional - Menciones en RRSS"), ln=True, align='C')
        pdf.ln(5)
        
        total_menciones = len(df_brandwatch)
        
        # Filtrar por Sentiment = 'negative' (campo de Brandwatch)
        if 'Sentimiento' in df_brandwatch.columns:
            df_negativas = df_brandwatch[df_brandwatch['Sentimiento'].astype(str).str.lower() == 'negative'].copy()
        elif 'Sentiment' in df_brandwatch.columns:
            df_negativas = df_brandwatch[df_brandwatch['Sentiment'].astype(str).str.lower() == 'negative'].copy()
        else:
            df_negativas = pd.DataFrame()
        
        menciones_negativas = len(df_negativas)
        pct_negativas = (menciones_negativas / total_menciones * 100) if total_menciones > 0 else 0
        
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, "Resumen General:", ln=True)
        
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 6, f"  - Total menciones analizadas: {total_menciones:,}", ln=True)
        pdf.cell(0, 6, f"  - Menciones negativas: {menciones_negativas:,} ({pct_negativas:.1f}%)", ln=True)
        pdf.ln(5)
        
        if menciones_negativas > 0 and 'Categoría' in df_negativas.columns:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, "Top 5 Tipos de Reclamos (en menciones negativas):", ln=True)
            
            cat_counts = df_negativas['Categoría'].value_counts().head(5)
            pdf.set_font("Arial", '', 10)
            for cat, vol in cat_counts.items():
                pct_cat = vol / menciones_negativas * 100
                pdf.cell(0, 6, f"  - {clean_txt_local(cat)}: {vol} menciones ({pct_cat:.1f}%)", ln=True)
        
        pdf.ln(5)
        
        # Distribución general de sentimiento
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 7, "Distribucion por Sentimiento:", ln=True)
        
        if 'Sentimiento' in df_brandwatch.columns:
            sent_counts = df_brandwatch['Sentimiento'].value_counts()
        elif 'Sentiment' in df_brandwatch.columns:
            sent_counts = df_brandwatch['Sentiment'].value_counts()
        else:
            sent_counts = pd.Series()
        
        pdf.set_font("Arial", '', 10)
        for sent, vol in sent_counts.items():
            pct = vol / total_menciones * 100
            pdf.cell(0, 6, f"  - {clean_txt_local(str(sent).capitalize())}: {vol} ({pct:.1f}%)", ln=True)

    return pdf.output(dest='S').encode('latin-1')


# --- PDF 2: PRESENTACIÓN HORIZONTAL ---
def generar_pdf_presentacion(df_metrics, df_raw, period_value, period_type='monthly', df_brandwatch=None):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    
    if period_type == 'weekly':
        period_title = f"Semana {period_value}"
        prev_period = period_value - 1
        df_raw['Week'] = df_raw['Date_Time'].dt.isocalendar().week
    else:
        month_name = calendar.month_name[period_value.month]
        period_title = f"{month_name} {period_value.year}"
        prev_period = period_value - 1
        df_raw['YearMonth'] = df_raw['Date_Time'].dt.to_period('M')
    
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
    
    # Portada
    pdf.add_page()
    pdf.set_fill_color(115, 82, 255)
    pdf.rect(0, 0, 297, 210, 'F')
    pdf.set_text_color(255, 255, 255)
    
    pdf.set_y(85)
    pdf.set_font("Arial", 'B', 40)
    pdf.cell(0, 15, clean_txt("C_OPS Support Dashboard"), align='C', ln=True)
    pdf.set_font("Arial", '', 22)
    pdf.set_text_color(0, 209, 163)
    pdf.cell(0, 15, clean_txt(f"Resumen Directivo - {period_title}"), align='C', ln=True)
    
    audiences_in_period = df_metrics[df_metrics['Period'] == period_value]['Audience'].unique()
    
    for aud in ['Driver', 'Rider', 'B2B', 'Emergencias', 'Aeropuerto', 'Aeropuerto WhatsApp']:
        if aud not in audiences_in_period: continue
        
        curr_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Period'] == period_value)]
        prev_df = df_metrics[(df_metrics['Audience'] == aud) & (df_metrics['Period'] == prev_period)]

        if not curr_df.empty:
            curr = curr_df.iloc[0]
            prev = prev_df.iloc[0] if not prev_df.empty else curr * 0
            
            pdf.add_page()
            add_header(f"Performance Operativo: {aud.upper()} | {period_title}")
            
            vol_pct = ((curr['Contactos Recibidos'] - prev['Contactos Recibidos']) / prev['Contactos Recibidos'] * 100) if prev['Contactos Recibidos'] else 0
            nps_diff = curr['NPS'] - prev['NPS'] if pd.notna(prev['NPS']) else 0
            csat_diff = curr['CSAT (%)'] - prev['CSAT (%)'] if pd.notna(prev['CSAT (%)']) else 0
            firt_diff = curr['FiRT SLA (%)'] - prev['FiRT SLA (%)'] if pd.notna(prev['FiRT SLA (%)']) else 0
            reop_diff = curr['Ratio Reopen/Tickets (%)'] - prev['Ratio Reopen/Tickets (%)'] if pd.notna(prev['Ratio Reopen/Tickets (%)']) else 0
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
                ("SLA 1ra Respuesta", f"{curr['FiRT SLA (%)']:.1f}%", firt_diff, True, True),
                ("Ratio Reopen", f"{curr['Ratio Reopen/Tickets (%)']:.1f}%", reop_diff, False, True)
            ]
            
            y_start = 50
            for name, val_str, delta_val, is_higher_better, is_pct in metrics:
                pdf.set_xy(15, y_start)
                pdf.set_font("Arial", 'B', 11)
                pdf.set_text_color(50, 50, 50)
                pdf.cell(40, 8, clean_txt(name), 0, 0, 'L')
                
                pdf.set_font("Arial", '', 11)
                pdf.cell(20, 8, clean_txt(val_str), 0, 0, 'L')
                
                if pd.isna(delta_val) or delta_val == 0:
                    pdf.set_text_color(150, 150, 150)
                    delta_str = "(-)"
                else:
                    if delta_val > 0:
                        pdf.set_text_color(0, 209, 163) if is_higher_better else pdf.set_text_color(255, 82, 82)
                    else:
                        pdf.set_text_color(255, 82, 82) if is_higher_better else pdf.set_text_color(0, 209, 163)
                    suffix = "%" if is_pct else ""
                    delta_str = f"({delta_val:+.1f}{suffix})"
                
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(25, 8, clean_txt(delta_str), 0, 1, 'R')
                
                pdf.set_draw_color(230, 230, 230)
                pdf.line(15, y_start + 8, 100, y_start + 8)
                y_start += 12
                
            pdf.set_xy(15, y_start)
            pdf.set_font("Arial", 'I', 9)
            pdf.set_text_color(150, 150, 150)
            pdf.cell(85, 6, clean_txt(f"*NPS basado en {int(nps_count)} encuestas validas."), 0, 1, 'L')

            # Gráfico de evolución
            df_trend = df_metrics[df_metrics['Audience'] == aud].sort_values('Period')
            
            # Filtrar solo hasta el período seleccionado
            df_trend = df_trend[df_trend['Period'] <= period_value]
            
            if len(df_trend) > 1:
                img_path = f"slide_trend_{aud}.png"
                try:
                    fig, ax1 = plt.subplots(figsize=(9, 4.5))
                    fig.patch.set_facecolor('white')
                    ax1.set_facecolor('white')
                    
                    x_labels = df_trend['PeriodLabel'].values
                    
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

            insight_nps = analizar_detractores(df_raw, aud, period_value, period_type)
            
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
    
    # === PÁGINA DE RIESGO REPUTACIONAL ===
    if df_brandwatch is not None:
        pdf.add_page()
        add_header("Riesgo Reputacional - Menciones en RRSS")
        
        total_menciones = len(df_brandwatch)
        
        # Filtrar por Sentiment = 'negative'
        if 'Sentimiento' in df_brandwatch.columns:
            df_negativas = df_brandwatch[df_brandwatch['Sentimiento'].astype(str).str.lower() == 'negative']
        elif 'Sentiment' in df_brandwatch.columns:
            df_negativas = df_brandwatch[df_brandwatch['Sentiment'].astype(str).str.lower() == 'negative']
        else:
            df_negativas = pd.DataFrame()
        
        menciones_negativas = len(df_negativas)
        pct_negativas = (menciones_negativas / total_menciones * 100) if total_menciones > 0 else 0
        
        # Resumen en tarjetas
        pdf.set_xy(15, 35)
        pdf.set_fill_color(245, 245, 245)
        
        # Tarjeta 1: Total
        pdf.set_font("Arial", 'B', 24)
        pdf.set_text_color(115, 82, 255)
        pdf.cell(65, 20, f"{total_menciones:,}", 0, 0, 'C', 1)
        
        # Tarjeta 2: Negativas
        pdf.set_font("Arial", 'B', 24)
        pdf.set_text_color(255, 82, 82)
        pdf.cell(65, 20, f"{menciones_negativas:,}", 0, 0, 'C', 1)
        
        # Tarjeta 3: Porcentaje
        pdf.set_font("Arial", 'B', 24)
        pdf.set_text_color(255, 179, 71)
        pdf.cell(65, 20, f"{pct_negativas:.1f}%", 0, 0, 'C', 1)
        
        pdf.ln(22)
        pdf.set_x(15)
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(65, 6, "Total Menciones", 0, 0, 'C')
        pdf.cell(65, 6, "Menciones Negativas", 0, 0, 'C')
        pdf.cell(65, 6, "% Negativas", 0, 0, 'C')
        
        # Top categorías en menciones negativas
        if menciones_negativas > 0 and 'Categoría' in df_negativas.columns:
            pdf.set_xy(15, 75)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, "Top 5 Tipos de Reclamos (menciones negativas):", ln=True)
            
            cat_counts = df_negativas['Categoría'].value_counts().head(5)
            y_pos = 85
            
            color_map = {
                'Frustración Crítica': (255, 82, 82),
                'Seguridad': (255, 138, 128),
                'Cobros y Tarifas': (255, 179, 71),
                'Calidad de Servicio': (255, 213, 79),
                'Disponibilidad / App': (129, 212, 250),
                'Otros / Neutro': (180, 180, 180)
            }
            
            for cat, vol in cat_counts.items():
                color = color_map.get(cat, (150, 150, 150))
                pdf.set_fill_color(*color)
                bar_width = min(vol / cat_counts.max() * 150, 150)
                pdf.rect(15, y_pos, bar_width, 10, 'F')
                
                pdf.set_xy(bar_width + 20, y_pos + 2)
                pdf.set_font("Arial", '', 10)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 6, f"{clean_txt(cat)}: {vol}")
                y_pos += 14

    return pdf.output(dest='S').encode('latin-1')


# ============================================
# === INTERFAZ PRINCIPAL ===
# ============================================
st.markdown(f"<h1 style='color: {CABIFY_PURPLE};'>🚕 C_OPS Support Dashboard</h1>", unsafe_allow_html=True)
st.write("Dashboard de indicadores de Support. Configura los filtros y sube tu archivo CSV.")

# === CONFIGURACIÓN INICIAL EN SIDEBAR ===
st.sidebar.markdown(f"<h3 style='color: {CABIFY_PURPLE};'>⚙️ Configuración</h3>", unsafe_allow_html=True)

# Filtro 1: Tipo de período
period_type = st.sidebar.radio(
    "📅 Tipo de Análisis",
    options=['monthly', 'weekly'],
    format_func=lambda x: 'Mensual' if x == 'monthly' else 'Semanal',
    index=0
)

# Filtro 2: Incluir Abi/Bot
include_abibot = st.sidebar.checkbox(
    "🤖 Incluir tickets Abi/Bot (Automatizados)",
    value=False,
    help="Si está desactivado, solo se incluyen tickets con Automated='Agent' (coincide con Tableau)"
)

st.sidebar.divider()

file_main = st.file_uploader("📁 Archivo Maestro (CSV)", type=['csv'], help="CSV principal con todas las audiencias")
file_whatsapp = st.file_uploader("📁 Aeropuerto WhatsApp (CSV) - Opcional", type=['csv'], help="CSV específico de [CL] Aeropuerto | WhatsApp")
file_brandwatch = st.file_uploader("🔍 Brandwatch (CSV) - Opcional", type=['csv', 'xlsx'], help="Archivo de menciones para Riesgo Reputacional")

if file_main is not None:
    with st.spinner('Aplicando reglas de negocio, limpiando y analizando datos...'):
        df_raw = load_main_data(file_main, include_abibot=include_abibot)
        
        # Si hay archivo de WhatsApp, cargarlo y combinar
        if file_whatsapp is not None:
            df_whatsapp = load_whatsapp_data(file_whatsapp)
            # Asegurar que ambos DataFrames tengan las mismas columnas básicas
            common_cols = ['Date_Time', 'Audience', 'Contact Type', 'NPS_Score', 'CSAT_Pct', 
                          'FRT_Hours', 'FuRT_Hours', 'Reopen_Count', 'Tag_1', 'Tag_2', 
                          'Tag_3', 'Ticket_Number']
            
            for col in common_cols:
                if col not in df_raw.columns:
                    df_raw[col] = np.nan
                if col not in df_whatsapp.columns:
                    df_whatsapp[col] = np.nan
            
            df_raw = pd.concat([df_raw[common_cols], df_whatsapp[common_cols]], ignore_index=True)
            st.sidebar.success("✅ Archivo WhatsApp cargado")
        
        # Si hay archivo de Brandwatch, cargarlo
        df_brandwatch = None
        if file_brandwatch is not None:
            df_brandwatch = load_brandwatch_data(file_brandwatch)
            if df_brandwatch is not None:
                st.sidebar.success("✅ Archivo Brandwatch cargado")
        
        df_metrics = aggregate_data(df_raw, period_type=period_type)
        
        # Agregar columnas de período a df_raw
        if period_type == 'weekly':
            df_raw['Week'] = df_raw['Date_Time'].dt.isocalendar().week
            df_raw['Year'] = df_raw['Date_Time'].dt.isocalendar().year
        else:
            df_raw['YearMonth'] = df_raw['Date_Time'].dt.to_period('M')
    
    st.sidebar.markdown(f"<h3 style='color: {CABIFY_PURPLE};'>📊 Filtros de Visualización</h3>", unsafe_allow_html=True)
    all_audiences = ['Rider', 'Driver', 'B2B', 'Emergencias', 'Aeropuerto', 'Aeropuerto WhatsApp']
    audiences = [a for a in all_audiences if a in df_metrics['Audience'].unique()]
    
    selected_audience = st.sidebar.selectbox("Selecciona la Audiencia", audiences)
    
    df_filtered = df_metrics[df_metrics['Audience'] == selected_audience].sort_values('Period')
    available_periods = sorted(df_filtered['Period'].dropna().unique(), reverse=True)
    
    # Para no mezclar períodos incompletos, excluir el actual si es mensual
    if period_type == 'monthly':
        today = datetime.now()
        current_period = pd.Period(f"{today.year}-{today.month:02d}", freq='M')
        available_periods = [p for p in available_periods if p < current_period]
    
    if len(available_periods) == 0:
        st.warning("No hay períodos completos disponibles para analizar.")
        st.stop()
    
    # Formatear períodos para display
    if period_type == 'weekly':
        period_options = {str(p): f"Semana {p}" for p in available_periods}
    else:
        period_options = {str(p): f"{calendar.month_name[p.month]} {p.year}" for p in available_periods}
    
    selected_period_str = st.sidebar.selectbox(
        f"Selecciona {'Semana' if period_type == 'weekly' else 'Mes'} a visualizar", 
        options=list(period_options.keys()),
        format_func=lambda x: period_options[x],
        index=0
    )
    
    if period_type == 'weekly':
        selected_period = int(selected_period_str)
    else:
        selected_period = pd.Period(selected_period_str)
    
    period_display = period_options[selected_period_str]
    
    # REPORTE VERTICAL
    st.sidebar.divider()
    st.sidebar.subheader("📄 Reportes Ejecutivos (PDF)")
    st.sidebar.caption("Formato documento clásico.")
    _pdf_vertical = generar_pdf_resumen(df_metrics, df_raw.copy(), selected_period, period_type, df_brandwatch)
    _ = st.sidebar.download_button(
        label=f"📄 Descargar Informe (Vertical)",
        data=_pdf_vertical,
        file_name=f"COPS_Informe_{selected_period_str}.pdf",
        mime="application/pdf"
    )

    # PRESENTACIÓN HORIZONTAL
    st.sidebar.caption("Formato diapositivas visuales.")
    _pdf_horizontal = generar_pdf_presentacion(df_metrics, df_raw.copy(), selected_period, period_type, df_brandwatch)
    _ = st.sidebar.download_button(
        label=f"📊 Descargar Presentacion (Horizontal)",
        data=_pdf_horizontal,
        file_name=f"COPS_Presentacion_{selected_period_str}.pdf",
        mime="application/pdf"
    )

    # EXCEL DATOS CRUDOS
    st.sidebar.divider()
    st.sidebar.subheader("📥 Exportar Datos Crudos")
    
    # Filtrar datos del período seleccionado
    if period_type == 'weekly':
        df_period_raw = df_raw[df_raw['Week'] == selected_period]
    else:
        df_period_raw = df_raw[df_raw['YearMonth'] == selected_period]
    
    if not df_period_raw.empty:
        # Usar función de Excel con formato Cabify
        excel_data = generar_excel_cabify(df_raw.copy(), audiences, period_type, selected_period, period_display)
        _ = st.sidebar.download_button(
            label=f"📥 Descargar Datos {period_display} (.xlsx)",
            data=excel_data,
            file_name=f"Datos_Crudos_Cabify_{selected_period_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.sidebar.warning(f"No hay datos para exportar en {period_display}.")
    
    # === TABS PRINCIPALES ===
    tab_names = [f"📈 KPIs {'Semanales' if period_type == 'weekly' else 'Mensuales'}", "📊 Evolución Histórica", "🔍 Deep Dive (Análisis de Detractores)"]
    if df_brandwatch is not None:
        tab_names.append("⚠️ Riesgo Reputacional")
    
    tabs = st.tabs(tab_names)
    
    with tabs[0]:
        st.markdown("### 💬 Copiar Resumen para Slack")
        st.info("Pasa el mouse sobre la caja de abajo y haz clic en el ícono de copiar para pegarlo directamente en Slack.")
        slack_msg = generar_texto_slack(df_metrics, df_raw, selected_period, period_type, df_brandwatch)
        st.code(slack_msg, language="markdown")
        st.divider()

        current_data = df_filtered[df_filtered['Period'] == selected_period]
        prev_period = selected_period - 1
        prev_data = df_filtered[df_filtered['Period'] == prev_period]
        
        if not current_data.empty:
            curr = current_data.iloc[0]
            prev = prev_data.iloc[0] if not prev_data.empty else current_data.iloc[0] * 0 
            
            def calc_delta_pct(current, previous):
                if previous == 0 or pd.isna(previous): return "0.0%"
                return f"{((current - previous) / previous) * 100:+.2f}%"
                
            def calc_delta_abs(current, previous):
                if pd.isna(previous): return "+0.0"
                return f"{current - previous:+.2f}"
                
            st.markdown(f"### Resumen **{period_display}** - {selected_audience}")
            
            st.markdown("#### I. Performance General de Gestión")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Contactos Recibidos", f"{curr['Contactos Recibidos']:,.0f}", calc_delta_pct(curr['Contactos Recibidos'], prev['Contactos Recibidos']), delta_color="inverse")
            with c2: st.metric("Contactos Ticket", f"{curr['Contactos Ticket']:,.0f}", calc_delta_pct(curr['Contactos Ticket'], prev['Contactos Ticket']), delta_color="inverse")
            with c3: st.metric("Contactos Chat", f"{curr['Contactos Chat']:,.0f}", calc_delta_pct(curr['Contactos Chat'], prev['Contactos Chat']), delta_color="inverse")
            with c4: st.metric("Contactos Call", f"{curr['Contactos Call']:,.0f}", calc_delta_pct(curr['Contactos Call'], prev['Contactos Call']), delta_color="inverse")
            
            c5, c6, c7 = st.columns(3)
            period_label = "este período" if period_type == 'weekly' else "este mes"
            with c5: st.metric("NPS Score", f"{curr['NPS']:.2f}", calc_delta_abs(curr['NPS'], prev['NPS']), delta_color="normal", help=f"Basado en {int(curr.get('NPS_Count', 0))} encuestas {period_label}")
            with c6: st.metric("CSAT", f"{curr['CSAT (%)']:.1f}%", calc_delta_abs(curr['CSAT (%)'], prev['CSAT (%)']) + "%", delta_color="normal")
            
            st.divider()
            
            st.markdown("#### II. Calidad Gestión de Tickets")
            
            # Obtener SLAs de la audiencia seleccionada
            sla_config = SLA_CONFIG.get(selected_audience, {'firt': 24, 'furt': 36})
            firt_label = f"FiRT <{sla_config['firt']}h" if sla_config['firt'] >= 1 else f"FiRT <{int(sla_config['firt']*60)}min"
            furt_label = f"FuRT <{sla_config['furt']}h"
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("TMO Promedio (Hrs)", f"{curr['TMO (Hrs)']:.2f}", calc_delta_abs(curr['TMO (Hrs)'], prev['TMO (Hrs)']), delta_color="inverse")
            with c2: st.metric(f"SLA 1ra Rsp ({firt_label})", f"{curr['FiRT SLA (%)']:.2f}%", calc_delta_abs(curr['FiRT SLA (%)'], prev['FiRT SLA (%)']) + "%", delta_color="normal")
            with c3: st.metric(f"SLA Resolución ({furt_label})", f"{curr['FuRT SLA (%)']:.2f}%", calc_delta_abs(curr['FuRT SLA (%)'], prev['FuRT SLA (%)']) + "%", delta_color="normal")
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

    # === TAB 2: EVOLUCIÓN HISTÓRICA ===
    with tabs[1]:
        st.markdown(f"### 📊 Evolución de Indicadores - {selected_audience}")
        st.info(f"Visualiza la tendencia histórica de todos los indicadores clave. Solo se muestran períodos completos hasta {period_display}.")
        
        # Filtrar solo hasta el período seleccionado (no incluir períodos futuros o incompletos)
        df_trend = df_filtered[df_filtered['Period'] <= selected_period].sort_values('Period').copy()
        
        if len(df_trend) > 0:
            # Gráfico 1: Volumen
            st.markdown("#### 📦 Volumen de Contactos")
            fig_vol = px.bar(df_trend, x='PeriodLabel', y='Contactos Recibidos', 
                           color_discrete_sequence=[CABIFY_PURPLE],
                           text='Contactos Recibidos')
            fig_vol.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_vol.update_layout(plot_bgcolor="white", xaxis_title="Período", yaxis_title="Contactos")
            fig_vol.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_vol, use_container_width=True)
            
            st.divider()
            
            # Gráfico 2: NPS y CSAT
            st.markdown("#### ⭐ Experiencia del Cliente (NPS y CSAT)")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_nps = px.line(df_trend, x='PeriodLabel', y='NPS', markers=True,
                                color_discrete_sequence=[CABIFY_PURPLE])
                fig_nps.update_traces(line=dict(width=3), marker=dict(size=10))
                fig_nps.update_layout(plot_bgcolor="white", title="NPS Score", yaxis_range=[-100, 100])
                fig_nps.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig_nps, use_container_width=True)
            
            with col2:
                fig_csat = px.line(df_trend, x='PeriodLabel', y='CSAT (%)', markers=True,
                                 color_discrete_sequence=[CABIFY_SECONDARY])
                fig_csat.update_traces(line=dict(width=3), marker=dict(size=10))
                fig_csat.update_layout(plot_bgcolor="white", title="CSAT (%)", yaxis_range=[0, 100])
                st.plotly_chart(fig_csat, use_container_width=True)
            
            st.divider()
            
            # Gráfico 3: SLA FiRT
            st.markdown("#### ⏱️ Cumplimiento de SLAs")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_firt = px.area(df_trend, x='PeriodLabel', y='FiRT SLA (%)', 
                                  color_discrete_sequence=[CABIFY_PURPLE])
                fig_firt.update_layout(plot_bgcolor="white", title="SLA 1ra Respuesta (<24h)", yaxis_range=[0, 100])
                fig_firt.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7, 
                                  annotation_text="Meta 80%", annotation_position="top right")
                st.plotly_chart(fig_firt, use_container_width=True)
            
            with col2:
                fig_furt = px.area(df_trend, x='PeriodLabel', y='FuRT SLA (%)',
                                  color_discrete_sequence=[CABIFY_SECONDARY])
                fig_furt.update_layout(plot_bgcolor="white", title="SLA Resolución (<36h)", yaxis_range=[0, 100])
                fig_furt.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7,
                                  annotation_text="Meta 80%", annotation_position="top right")
                st.plotly_chart(fig_furt, use_container_width=True)
            
            st.divider()
            
            # Tabla resumen
            st.markdown("#### 📋 Tabla Resumen de Indicadores")
            
            df_table = df_trend[['PeriodLabel', 'Contactos Recibidos', 'NPS', 'CSAT (%)', 
                                'FiRT SLA (%)', 'FuRT SLA (%)', 'TMO (Hrs)', 
                                'Ratio Reopen/Tickets (%)']].copy()
            df_table.columns = ['Período', 'Volumen', 'NPS', 'CSAT (%)', 'FiRT SLA (%)', 'FuRT SLA (%)', 'TMO (Hrs)', 'Reopen (%)']
            df_table = df_table.round(2)
            
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            
            # Exportar evolución a Excel
            output_evol = io.BytesIO()
            with pd.ExcelWriter(output_evol, engine='xlsxwriter') as writer:
                df_table.to_excel(writer, sheet_name='Evolución', index=False)
            st.download_button(
                label="📊 Descargar Evolución (.xlsx)",
                data=output_evol.getvalue(),
                file_name=f"Evolucion_{selected_audience}_{selected_period_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No hay suficientes datos para mostrar la evolución histórica.")

    # === TAB 3: DEEP DIVE ===
    with tabs[2]:
        st.markdown(f"### 🔍 Deep Dive: Análisis de Motivos ({selected_audience} - {period_display})")
        
        # Filtrar datos del período
        if period_type == 'weekly':
            df_raw_filtered = df_raw[(df_raw['Audience'] == selected_audience) & (df_raw['Week'] == selected_period)]
        else:
            df_raw_filtered = df_raw[(df_raw['Audience'] == selected_audience) & (df_raw['YearMonth'] == selected_period)]
        
        # Para Aeropuerto WhatsApp: no hay encuestas, solo motivos generales
        if selected_audience == 'Aeropuerto WhatsApp':
            st.info("ℹ️ Aeropuerto WhatsApp no tiene encuestas NPS. Mostramos los motivos de contacto más frecuentes.")
            
            if not df_raw_filtered.empty and 'Tag_3' in df_raw_filtered.columns:
                st.markdown("#### 📊 Top 5 Motivos de Contacto (Tag Nivel 3)")
                top_tags = df_raw_filtered['Tag_3'].dropna().value_counts().reset_index()
                top_tags.columns = ['Motivo', 'Volumen']
                top_tags = top_tags.head(5)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig_tags = px.bar(top_tags, x='Volumen', y='Motivo', orientation='h',
                                      color_discrete_sequence=[CABIFY_PURPLE], text='Volumen')
                    fig_tags.update_traces(textposition='outside')
                    fig_tags.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white", height=300)
                    st.plotly_chart(fig_tags, use_container_width=True)
                with col2:
                    st.dataframe(top_tags, use_container_width=True, hide_index=True)
        else:
            # Para otras audiencias: análisis de detractores NPS -100
            resumen_app = analizar_detractores(df_raw, selected_audience, selected_period, period_type)
            if "Excelente" in resumen_app:
                st.success("✅ " + resumen_app)
            else:
                st.error("🚨 " + resumen_app)
            
            st.divider()
            
            # Top 5 motivos en detractores NPS -100
            st.markdown("#### 📊 Top 5 Motivos de Detractores NPS -100 (Tag Nivel 3)")
            
            df_detractores = df_raw_filtered[df_raw_filtered['NPS_Score'] == -100]
            
            if not df_detractores.empty and 'Tag_3' in df_detractores.columns:
                top_tags_nps = df_detractores['Tag_3'].dropna().value_counts().reset_index()
                top_tags_nps.columns = ['Motivo', 'Volumen']
                top_tags_nps = top_tags_nps.head(5)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig_tags_nps = px.bar(top_tags_nps, x='Volumen', y='Motivo', orientation='h',
                                          color_discrete_sequence=['#FF5252'], text='Volumen')
                    fig_tags_nps.update_traces(textposition='outside')
                    fig_tags_nps.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white", height=300)
                    st.plotly_chart(fig_tags_nps, use_container_width=True)
                with col2:
                    st.dataframe(top_tags_nps, use_container_width=True, hide_index=True)
            else:
                st.info("No hay detractores NPS -100 en este período.")
            
            st.divider()
            
            # También mostrar distribución general
            st.markdown("#### 📋 Distribución General de Motivos (Tag Nivel 3)")
            if not df_raw_filtered.empty and 'Tag_3' in df_raw_filtered.columns:
                top_tags_all = df_raw_filtered['Tag_3'].dropna().value_counts().reset_index()
                top_tags_all.columns = ['Motivo', 'Volumen']
                top_tags_all = top_tags_all.head(10)
                
                fig_tags = px.bar(top_tags_all, x='Volumen', y='Motivo', orientation='h',
                                  color_discrete_sequence=[CABIFY_SECONDARY])
                fig_tags.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white")
                st.plotly_chart(fig_tags, use_container_width=True)

    # === TAB 4: RIESGO REPUTACIONAL (Solo si hay Brandwatch) ===
    if df_brandwatch is not None and len(tabs) > 3:
        with tabs[3]:
            st.markdown("### ⚠️ Riesgo Reputacional (Brandwatch)")
            st.info("📊 Análisis de menciones negativas clasificadas por tipo de reclamo.")
            
            total_menciones = len(df_brandwatch)
            
            # Filtrar por Sentiment = 'negative'
            if 'Sentimiento' in df_brandwatch.columns:
                df_negativas = df_brandwatch[df_brandwatch['Sentimiento'].astype(str).str.lower() == 'negative'].copy()
            elif 'Sentiment' in df_brandwatch.columns:
                df_negativas = df_brandwatch[df_brandwatch['Sentiment'].astype(str).str.lower() == 'negative'].copy()
            else:
                df_negativas = pd.DataFrame()
            
            menciones_negativas = len(df_negativas)
            pct_negativas = (menciones_negativas / total_menciones * 100) if total_menciones > 0 else 0
            
            # Métricas principales
            st.markdown("#### 📊 Resumen de Menciones")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Menciones", f"{total_menciones:,}")
            with col2:
                st.metric("Menciones Negativas", f"{menciones_negativas:,}", delta=f"{pct_negativas:.1f}%", delta_color="inverse")
            with col3:
                if 'Categoría' in df_negativas.columns:
                    top_cat = df_negativas['Categoría'].value_counts().index[0] if len(df_negativas) > 0 else "N/A"
                    st.metric("Principal Reclamo", top_cat)
            
            st.divider()
            
            # Top 5 tipos de reclamos en menciones negativas
            if menciones_negativas > 0 and 'Categoría' in df_negativas.columns:
                st.markdown("#### 📈 Top 5 Tipos de Reclamos (menciones negativas)")
                cat_counts = df_negativas['Categoría'].value_counts().reset_index()
                cat_counts.columns = ['Categoría', 'Menciones']
                cat_counts = cat_counts.head(5)
                
                color_map = {
                    'Frustración Crítica': '#FF5252',
                    'Seguridad': '#FF8A80',
                    'Cobros y Tarifas': '#FFB347',
                    'Calidad de Servicio': '#FFD54F',
                    'Disponibilidad / App': '#81D4FA',
                    'Otros / Neutro': '#BDBDBD'
                }
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig_cat = px.bar(cat_counts, x='Menciones', y='Categoría', orientation='h',
                                    color='Categoría', color_discrete_map=color_map, text='Menciones')
                    fig_cat.update_traces(textposition='outside')
                    fig_cat.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white", showlegend=False, height=300)
                    st.plotly_chart(fig_cat, use_container_width=True)
                with col2:
                    st.dataframe(cat_counts, use_container_width=True, hide_index=True)
            
            # Distribución por Sentimiento
            st.divider()
            st.markdown("#### 😊 Distribución por Sentimiento")
            
            if 'Sentimiento' in df_brandwatch.columns:
                sent_col = 'Sentimiento'
            elif 'Sentiment' in df_brandwatch.columns:
                sent_col = 'Sentiment'
            else:
                sent_col = None
            
            if sent_col:
                col1, col2 = st.columns(2)
                with col1:
                    sent_counts = df_brandwatch[sent_col].value_counts().reset_index()
                    sent_counts.columns = ['Sentimiento', 'Volumen']
                    fig_sent = px.pie(sent_counts, values='Volumen', names='Sentimiento',
                                     color='Sentimiento',
                                     color_discrete_map={'negative': '#FF5252', 'neutral': '#9E9E9E', 'positive': '#00D1A3'})
                    fig_sent.update_layout(showlegend=True)
                    st.plotly_chart(fig_sent, use_container_width=True)
                
                with col2:
                    st.dataframe(sent_counts, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Exportar
            st.markdown("#### 📥 Exportar Datos")
            excel_bw = generar_excel_brandwatch(df_negativas)
            st.download_button(
                label="📊 Excel Menciones Negativas",
                data=excel_bw,
                file_name="Riesgo_Reputacional_Support.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # === INFORMACIÓN DE FILTROS APLICADOS ===
    st.sidebar.divider()
    st.sidebar.subheader("ℹ️ Estado")
    abibot_status = "✅ Incluidos" if include_abibot else "❌ Excluidos"
    period_type_label = "Semanal" if period_type == 'weekly' else "Mensual"
    st.sidebar.caption(f"""
    **Configuración activa:**
    - Análisis: {period_type_label}
    - Tickets Abi/Bot: {abibot_status}
    
    **Registros procesados:** {len(df_raw):,}
    """)
