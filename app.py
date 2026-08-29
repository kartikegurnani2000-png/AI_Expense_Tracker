# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import datetime
import json
import pdfplumber
from google import genai
from google.genai import types
import time

# --- DEFAULT API KEY ---
DEFAULT_GEMINI_KEY = "YOUR_API_KEY_HERE" # Put your key here!

# --- 🎨 MODERN SAAS UI/UX CONFIGURATION ---
st.set_page_config(page_title="Clera Finance UI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# SAFE CSS INJECTION
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Safely apply fonts ONLY to text, avoiding Streamlit internal icons */
    h1, h2, h3, h4, h5, h6, p, .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Protect internal icons */
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
    }
    
    .stApp { background-color: #F8FAFC; }

    .gradient-text {
        background: linear-gradient(90deg, #4F46E5 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    .hero-banner {
        width: 100%;
        height: 140px;
        border-radius: 20px;
        background-image: url('https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=2564&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        margin-bottom: -40px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(180deg, rgba(255,255,255,0) 0%, #F8FAFC 100%);
        border-radius: 20px;
    }

    /* Custom SaaS KPI Cards */
    .saas-card {
        background-color: #FFFFFF;
        border: 1px solid #F1F5F9;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .saas-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.1);
        border-color: #E2E8F0;
    }
    .card-icon {
        position: absolute;
        top: 20px;
        right: 20px;
        width: 45px;
        height: 45px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    .icon-green { background: #DCFCE7; color: #16A34A; }
    .icon-red { background: #FEE2E2; color: #DC2626; }
    .icon-blue { background: #DBEAFE; color: #2563EB; }

    .metric-label {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 8px;
    }
    .trend-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .trend-up { background: #DCFCE7; color: #15803D; }
    .trend-down { background: #FEE2E2; color: #B91C1C; }

    /* Smart AI Insight Panel */
    .ai-panel {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid #E2E8F0;
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.05);
        display: flex;
        gap: 20px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .ai-avatar {
        width: 80px;
        height: 80px;
        border-radius: 20px;
        background-image: url('https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=200&auto=format&fit=crop');
        background-size: cover;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        flex-shrink: 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- USER DATABASE MANAGEMENT ---
USERS_FILE = 'users.csv'
DELETION_LOG = 'deletion_logs.csv'

def load_users():
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame([['admin', 'admin321']], columns=['username', 'password'])
        df.to_csv(USERS_FILE, index=False)
        return df
    return pd.read_csv(USERS_FILE)

def delete_user_account(username):
    users_df = pd.read_csv(USERS_FILE)
    users_df = users_df[users_df['username'] != username]
    users_df.to_csv(USERS_FILE, index=False)
    for file in [f"expenses_{username}.csv", f"upload_history_{username}.csv", f"income_{username}.csv"]:
        if os.path.exists(file): os.remove(file)

users_df = load_users()

# --- AUTHENTICATION ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = ''

if not st.session_state['logged_in']:
    st.markdown("<br><br><h1 style='text-align: center;' class='gradient-text'>Clera Finance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B; margin-bottom: 3rem;'>Intelligent wealth management, visualized.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        tab1, tab2 = st.tabs(["🔐 Sign In", "✨ Create Account"])
        with tab1:
            with st.form("login_form"):
                login_user = st.text_input("Workspace ID").strip().lower()
                login_pass = st.text_input("Password", type="password")
                if st.form_submit_button("Access Dashboard"):
                    if not users_df[(users_df['username'] == login_user) & (users_df['password'] == login_pass)].empty:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = login_user
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("Choose Workspace ID").strip().lower()
                new_pass = st.text_input("Secure Password", type="password")
                if st.form_submit_button("Deploy Account"):
                    if new_user in users_df['username'].values:
                        st.error("ID already exists.")
                    elif len(new_user) < 3:
                        st.warning("Minimum 3 characters required.")
                    else:
                        pd.DataFrame([[new_user, new_pass]], columns=['username', 'password']).to_csv(USERS_FILE, mode='a', header=False, index=False)
                        st.success("Deployed successfully. Please sign in.")
    st.stop()

# --- SIDEBAR ---
active_user = st.session_state['username']
with st.sidebar:
    st.markdown("<h2 class='gradient-text' style='margin-bottom: 0;'>⚡ Clera.</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #64748B; font-size: 0.875rem; margin-bottom: 2rem;'>{active_user}.workspace</p>", unsafe_allow_html=True)
    
    menu = st.radio("MENU", ["📊 Overview", "📥 Data Ingestion", "💵 Income Hub", "📝 Expense Ledger", "📈 Portfolio", "⚙️ Settings"], label_visibility="collapsed")
    st.write("---")
    if st.button("Log out"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
        st.rerun()

# --- DATA FUNCTIONS ---
USER_CSV_FILE = f"expenses_{active_user}.csv"
UPLOAD_HISTORY_FILE = f"upload_history_{active_user}.csv"
INCOME_CSV_FILE = f"income_{active_user}.csv"

@st.cache_data 
def load_data(filename):
    if not os.path.exists(filename):
        pd.DataFrame(columns=['date', 'description', 'amount', 'category', 'source_file']).to_csv(filename, index=False)
        return pd.DataFrame(columns=['date', 'description', 'amount', 'category', 'source_file'])
    df = pd.read_csv(filename)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df = df.dropna(subset=['date']) 
    return df

@st.cache_data 
def load_income(filename):
    if not os.path.exists(filename):
        pd.DataFrame(columns=['date', 'source', 'amount', 'notes', 'source_file']).to_csv(filename, index=False)
        return pd.DataFrame(columns=['date', 'source', 'amount', 'notes', 'source_file'])
    df = pd.read_csv(filename)
    if 'amount' not in df.columns or 'source_file' not in df.columns:
        pd.DataFrame(columns=['date', 'source', 'amount', 'notes', 'source_file']).to_csv(filename, index=False)
        return pd.DataFrame(columns=['date', 'source', 'amount', 'notes', 'source_file'])
    if not df.empty and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df = df.dropna(subset=['date'])
    return df

df = load_data(USER_CSV_FILE)
income_df = load_income(INCOME_CSV_FILE)

@st.cache_resource
def train_prediction_model(data):
    if len(data) < 5: return None, 0.0 
    monthly_spending = data.groupby(data['date'].dt.to_period("M"))['amount'].sum().reset_index()
    monthly_spending['month_index'] = np.arange(len(monthly_spending)) 
    model = LinearRegression()
    model.fit(monthly_spending[['month_index']], monthly_spending['amount'])
    prediction = model.predict(np.array([[len(monthly_spending)]]))
    return monthly_spending, prediction[0]

monthly_data, predicted_spend = train_prediction_model(df)

# --- HELPER: RICH KPI CARD ---
def rich_kpi_card(title, value, icon, color_theme, trend_text, trend_type):
    return f"""
    <div class="saas-card">
        <div class="card-icon {color_theme}">{icon}</div>
        <div class="metric-label">{title}</div>
        <div class="metric-value">₹{value:,.2f}</div>
        <div class="trend-pill {trend_type}">{trend_text}</div>
    </div>
    """

def parse_pdf_with_gemini(uploaded_file, api_key: str):
    raw_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text: raw_text += text + "\n"
    if not raw_text.strip(): raise ValueError("Could not extract text. Might be an image scan.")

    client = genai.Client(api_key=api_key)
    prompt = f"Extract EXPENSES and INCOME from this bank statement. Normalize dates to YYYY-MM-DD. Amount positive float. Category: Groceries, Transport, Dining, Shopping, Wellness, Other.\n\n{raw_text[:14000]}"
    response = client.models.generate_content(
        model='gemini-3.6-flash', contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={"type": "OBJECT", "properties": {"expenses": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"date": {"type": "STRING"}, "description": {"type": "STRING"}, "amount": {"type": "NUMBER"}, "category": {"type": "STRING"}}, "required": ["date", "description", "amount", "category"]}}, "income": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"date": {"type": "STRING"}, "source": {"type": "STRING"}, "amount": {"type": "NUMBER"}, "notes": {"type": "STRING"}}, "required": ["date", "source", "amount"]}}}, "required": ["expenses", "income"]}
        )
    )
    res = response.text.strip()
    if res.startswith("```json"): res = res[7:-3].strip()
    elif res.startswith("```"): res = res[3:-3].strip()
    parsed_json = json.loads(res)
    return pd.DataFrame(parsed_json.get("expenses", [])), pd.DataFrame(parsed_json.get("income", []))

# --- MAIN PAGES ---
if menu == "📊 Overview":
    st.markdown("<div class='hero-banner'></div>", unsafe_allow_html=True)
    
    st.markdown(f"<div style='display: flex; align-items: center; gap: 15px; margin-bottom: 24px;'><img src='https://api.dicebear.com/7.x/notionists/svg?seed={active_user}&backgroundColor=e2e8f0' style='width: 60px; height: 60px; border-radius: 50%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h1 style='margin: 0; font-weight: 800;'>Welcome back, {active_user.title()}</h1></div>", unsafe_allow_html=True)

    if df.empty:
        st.info("Your workspace is empty. Head to Data Ingestion to connect your data.")
    else:
        total_spent = float(df['amount'].sum())
        total_income = float(income_df['amount'].sum()) if not income_df.empty else 0.0
        net_balance = total_income - total_spent
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(rich_kpi_card("Total Inflow", total_income, "📈", "icon-green", "Active Credits", "trend-up"), unsafe_allow_html=True)
        with col2:
            st.markdown(rich_kpi_card("Total Outflow", total_spent, "📉", "icon-red", "Tracked Debits", "trend-down"), unsafe_allow_html=True)
        with col3:
            bal_icon = "💰" if net_balance >= 0 else "⚠️"
            bal_theme = "icon-blue" if net_balance >= 0 else "icon-red"
            st.markdown(rich_kpi_card("Net Position", net_balance, bal_icon, bal_theme, "Current Balance", "trend-up" if net_balance >=0 else "trend-down"), unsafe_allow_html=True)
            
        st.write("<br>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='font-weight: 700; margin-bottom: 1rem;'>Cash Flow Trajectory</h3>", unsafe_allow_html=True)
        chart_data = df.groupby(df['date'].dt.to_period("M"))['amount'].sum().reset_index()
        chart_data['date'] = chart_data['date'].dt.strftime('%b %Y') 
        
        fig = px.area(chart_data, x="date", y="amount", markers=True)
        fig.update_traces(line_color='#4F46E5', fillcolor='rgba(79, 70, 229, 0.15)', line_width=4, marker=dict(size=10, color='#EC4899', line=dict(width=3, color='white')))
        fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, title=""), yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=""), margin=dict(t=10, b=20, l=0, r=0))
        # CRITICAL FIX: disabled modebar to fix the overlapping glitch
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}) 

        st.markdown("""
        <div class="ai-panel">
            <div class="ai-avatar"></div>
            <div style="flex-grow: 1;">
                <h3 class="gradient-text" style="margin: 0 0 10px 0;">Clera AI Synthesis</h3>
                <p style="color: #475569; font-size: 1.05rem; line-height: 1.6; margin-bottom: 15px;">
                    I have analyzed your recent transaction history. Based on your spending velocity, your predicted outflow for the next cycle is projected at <strong>₹{:,.2f}</strong>. Keep an eye on your discretionary spending categories this week.
                </p>
            </div>
        </div>
        """.format(predicted_spend), unsafe_allow_html=True)
        
        st.write("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight: 700; margin-bottom: 1rem;'>Resource Allocation</h3>", unsafe_allow_html=True)
        fig_pie = px.pie(df, names='category', values='amount', hole=0.6, color_discrete_sequence=['#4F46E5', '#EC4899', '#38BDF8', '#F59E0B', '#10B981'])
        fig_pie.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

elif menu == "📥 Data Ingestion":
    st.markdown("<h2 style='margin-bottom: 24px; font-weight: 700; letter-spacing: -1px;'>Document Pipeline</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type=["csv", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Process Pipeline"):
            if DEFAULT_GEMINI_KEY == "YOUR_API_KEY_HERE":
                st.error("Hold up! You need to put your actual Gemini API key into the app.py code at line 18.")
                st.stop()
            success_count = 0
            for file in uploaded_files:
                try:
                    if file.name.lower().endswith(".pdf"):
                        with st.spinner(f"🤖 AI is reading {file.name}..."):
                            new_data, new_income = parse_pdf_with_gemini(file, DEFAULT_GEMINI_KEY)
                    else:
                        with st.spinner(f"Processing CSV: {file.name}..."):
                            new_data = pd.read_csv(file)
                            new_income = pd.DataFrame(columns=['date', 'source', 'amount', 'notes'])
                    if not new_data.empty:
                        new_data['source_file'] = file.name
                        if not os.path.exists(USER_CSV_FILE): new_data.to_csv(USER_CSV_FILE, index=False)
                        else: new_data.to_csv(USER_CSV_FILE, mode='a', header=False, index=False)
                    if not new_income.empty:
                        new_income['source_file'] = file.name
                        if not os.path.exists(INCOME_CSV_FILE): new_income.to_csv(INCOME_CSV_FILE, index=False)
                        else: new_income.to_csv(INCOME_CSV_FILE, mode='a', header=False, index=False)
                    pd.DataFrame([[file.name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(new_data) + len(new_income)]], columns=["File Name", "Upload Date", "Rows Added"]).to_csv(UPLOAD_HISTORY_FILE, mode='a', header=not os.path.exists(UPLOAD_HISTORY_FILE), index=False)
                    st.success(f"✅ Processed {file.name} successfully!")
                    success_count += 1
                except Exception as e: st.error(f"Error processing {file.name}: {e}")
            if success_count > 0:
                time.sleep(2) 
                load_data.clear(); load_income.clear()
                st.rerun()
    st.write("---")
    if os.path.exists(UPLOAD_HISTORY_FILE):
        hist_df = pd.read_csv(UPLOAD_HISTORY_FILE)
        st.dataframe(hist_df.iloc[::-1], use_container_width=True, hide_index=True)
        files_uploaded = hist_df['File Name'].unique().tolist()
        if files_uploaded:
            file_to_delete = st.selectbox("Undo an extraction:", ["-- Select File --"] + files_uploaded)
            if file_to_delete != "-- Select File --" and st.button(f"Undo {file_to_delete}"):
                if not df.empty: df[df['source_file'] != file_to_delete].to_csv(USER_CSV_FILE, index=False)
                if not income_df.empty and 'source_file' in income_df.columns: income_df[income_df['source_file'] != file_to_delete].to_csv(INCOME_CSV_FILE, index=False)
                hist_df[hist_df['File Name'] != file_to_delete].to_csv(UPLOAD_HISTORY_FILE, index=False)
                st.success("✅ Undone!")
                load_data.clear(); load_income.clear()
                st.rerun()

elif menu == "💵 Income Hub":
    st.markdown("<h2 style='margin-bottom: 24px; font-weight: 700; letter-spacing: -1px;'>Revenue Streams</h2>", unsafe_allow_html=True)
    
    # CRITICAL FIX: Ensure income sum is displayed correctly
    total_earned = float(income_df['amount'].sum()) if not income_df.empty else 0.0
    st.markdown(rich_kpi_card("Total Verified Income", total_earned, "💵", "icon-green", "Active", "trend-up"), unsafe_allow_html=True)
        
    st.write("<br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("### Log Manual Income")
        with st.form("income_form"):
            col1, col2 = st.columns(2)
            with col1:
                inc_date = st.date_input("Date")
                inc_source = st.selectbox("Source", ["Freelance Post Designing", "Tutoring Students", "Client Credit", "Other"])
            with col2:
                inc_amount = st.number_input("Amount (₹)", min_value=1.00, format="%.2f")
                inc_notes = st.text_input("Notes")
            if st.form_submit_button("Record Transaction"):
                pd.DataFrame([[inc_date, inc_source, inc_amount, inc_notes, 'Manual Entry']], columns=['date', 'source', 'amount', 'notes', 'source_file']).to_csv(INCOME_CSV_FILE, mode='a', header=not os.path.exists(INCOME_CSV_FILE), index=False)
                st.success("✅ Income added!")
                load_income.clear(); st.rerun()
    st.write("<br>", unsafe_allow_html=True)
    if not income_df.empty:
        edited_income = st.data_editor(income_df, num_rows="dynamic", use_container_width=True, hide_index=True)
        if st.button("Save Changes"):
            edited_income.to_csv(INCOME_CSV_FILE, index=False)
            st.success("✅ Changes saved!"); load_income.clear(); st.rerun()

elif menu == "📝 Expense Ledger":
    st.markdown("<h2 style='margin-bottom: 24px; font-weight: 700; letter-spacing: -1px;'>Master Ledger</h2>", unsafe_allow_html=True)
    with st.expander("➕ Log Manual Expense"):
        with st.form("expense_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_date = st.date_input("Date")
                new_desc = st.text_input("Description", placeholder="E.g., Canteen lunch")
            with col2:
                new_amount = st.number_input("Amount (₹)", min_value=1.00, format="%.2f")
                new_cat = st.selectbox("Category", ['Groceries', 'Transport', 'Dining', 'Shopping', 'Wellness', 'Other'])
            if st.form_submit_button("Log Expense"):
                pd.DataFrame([[new_date, new_desc, new_amount, new_cat, 'Manual Entry']], columns=['date', 'description', 'amount', 'category', 'source_file']).to_csv(USER_CSV_FILE, mode='a', header=not os.path.exists(USER_CSV_FILE), index=False)
                st.success("Expense logged!"); load_data.clear(); st.rerun()
    st.write("---")
    if not df.empty:
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True)
        if st.button("Save Ledger Changes"):
            edited_df.to_csv(USER_CSV_FILE, index=False)
            st.success("✅ Changes saved!"); load_data.clear(); st.rerun()

elif menu == "📈 Portfolio":
    st.markdown("<h2 style='margin-bottom: 24px; font-weight: 700; letter-spacing: -1px;'>Wealth & Portfolio</h2>", unsafe_allow_html=True)
    invest_data = pd.DataFrame({'Asset Class': ['Index Funds (Nifty 50)', 'Crypto (BTC/ETH)', 'Bonds', 'High-Yield Savings'], 'Amount (₹)': [250000, 45000, 100000, 150000]})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Asset Allocation")
        with st.container(border=True):
            fig_pie = px.pie(invest_data, values='Amount (₹)', names='Asset Class', hole=0.5, color_discrete_sequence=['#4F46E5', '#38BDF8', '#F59E0B', '#EC4899'])
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
    with col2:
        st.markdown("### Portfolio Holdings")
        with st.container(border=True):
            st.dataframe(invest_data, use_container_width=True, hide_index=True)
            st.info("💡 **AI Tip:** Your portfolio is heavily weighted in Index Funds, which provides excellent long-term stability.")

elif menu == "⚙️ Settings":
    st.markdown("<h2 style='margin-bottom: 24px; font-weight: 700; letter-spacing: -1px;'>Preferences</h2>", unsafe_allow_html=True)
    current_users_df = pd.read_csv(USERS_FILE)
    if st.session_state['username'] == 'admin':
        st.dataframe(current_users_df, use_container_width=True)
    else:
        with st.expander("💾 Export Data Archive"):
            if not df.empty: st.download_button("Download CSV Archive", data=df.to_csv(index=False).encode('utf-8'), file_name=f"archive_{active_user}.csv", mime="text/csv")
        with st.expander("🔑 Access Control"):
            old_pass = st.text_input("Current Password", type="password")
            new_pass = st.text_input("New Password", type="password")
            if st.button("Update Security Credentials"):
                user_real_pass = current_users_df.loc[current_users_df['username'] == st.session_state['username'], 'password'].values[0]
                if old_pass == user_real_pass and len(new_pass) >= 3:
                    current_users_df.loc[current_users_df['username'] == st.session_state['username'], 'password'] = new_pass
                    current_users_df.to_csv(USERS_FILE, index=False); st.success("✅ Password updated!")
        with st.expander("🗑️ Destructive Actions"):
            st.warning("This action cannot be undone.")
            if st.button("Purge Workspace") and st.text_input("Confirm Password", type="password") == current_users_df.loc[current_users_df['username'] == st.session_state['username'], 'password'].values[0]:
                delete_user_account(st.session_state['username'])
                st.session_state['logged_in'] = False; st.session_state['username'] = ''; st.rerun()