"""
Volleyball Email to WhatsApp - Streamlit Dashboard
Converted from your existing Python code

Setup:
pip install streamlit

Run:
streamlit run app.py

Deploy to Streamlit Cloud for free hosting!
"""

import streamlit as st
import imaplib
import email
import re
from datetime import datetime
import urllib.parse

# ==================== CONFIGURATION ====================
# Credentials are read from .streamlit/secrets.toml
EMAIL = st.secrets["EMAIL_USER"]
PASSWORD = st.secrets["EMAIL_PASSWORD"]
IMAP_SERVER = "imap.gmail.com"
SENDER_EMAIL = "contact@gomammoth.co.uk"
WHATSAPP_GROUP_URL = "https://chat.whatsapp.com/FMDm5wA8PIo5MJtH8hQ6dC"

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🏐 Volleyball Email Checker",
    page_icon="🏐",
    layout="wide"
)

# ==================== STYLING ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #2E7D32 0%, #4CAF50 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #2E7D32 0%, #4CAF50 100%);
        color: white;
        font-size: 1.2rem;
        padding: 0.8rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
    }
    .match-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================
def parse_match_date(date_string: str):
    """Parse dates like 'on Tuesday 9 December' into a structured dict."""
    clean = re.sub(r"^on\s+", "", date_string.strip(), flags=re.IGNORECASE)
    m = re.search(r"(\w+day)\s+(\d{1,2})\s+(\w+)", clean, flags=re.IGNORECASE)
    if not m:
        return date_string

    day_name = m.group(1)
    day_num = int(m.group(2))
    month_name = m.group(3).lower()

    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    month = month_map.get(month_name)
    if not month:
        return date_string

    year = datetime.now().year
    try:
        dt = datetime(year, month, day_num)
    except ValueError:
        return date_string

    formatted = dt.strftime("%A, %d %B %Y")
    return {
        "original": date_string,
        "formatted": formatted,
        "day": day_num,
        "month": month_name,
        "year": year,
        "iso": dt.strftime("%Y-%m-%d"),
        "datetime": dt,
    }

def build_match_message(match_details: dict) -> str:
    """Build the WhatsApp message."""
    date_val = match_details.get("date", "Date not parsed")
    if isinstance(date_val, dict):
        date_display = date_val.get("formatted") or date_val.get("original") or str(date_val)
    else:
        date_display = date_val

    venue = match_details.get("venue") or "Check email for venue"

    return f"""🏐 VOLLEYBALL MATCH GAME!

{match_details.get('team1', '')} vs {match_details.get('team2', '')}
📅 {date_display}
⏰ {match_details.get('time', '')}
📍 {venue}

React to confirm your availability:
✅ = I'm coming
❌ = Can't make it
🤔 = Maybe

See you on the court! 🏐"""

def parse_volleyball_email(email_body):
    """Extract match details from GO Mammoth email."""
    patterns = [
        re.compile(
            r"Your next fixture is\s+(.+?)\s+vs\s+(.+?),\s+at\s+(\d{1,2}:\d{2})\s+on\s+([A-Za-z]+\s+\d{1,2}\s+[A-Za-z]+)\s+at\s+(.+?)\.",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"Your next fixture is\s+(.+?)\s+vs\s+(.+?)\s+at\s+(\d{1,2}:\d{2})\s+on\s+([A-Za-z]+\s+\d{1,2}\s+[A-Za-z]+)\s+at\s+(.+?)\.",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"fixture is\s+(.+?)\s+vs\s+(.+?)[,\s]+at\s+(\d{1,2}:\d{2})[,\s]+on\s+([A-Za-z]+\s+\d{1,2}\s+[A-Za-z]+)\s+at\s+(.+?)[\.\s]",
            re.IGNORECASE | re.DOTALL,
        ),
    ]

    for i, pat in enumerate(patterns):
        m = pat.search(email_body)
        if m:
            details = {
                "team1": m.group(1).strip(),
                "team2": m.group(2).strip(),
                "time": m.group(3).strip(),
                "date": parse_match_date(m.group(4).strip()),
                "venue": m.group(5).strip(),
            }

            if details["team1"] and details["team2"] and details["time"] and details["venue"]:
                return details

    # Fallback extraction
    time_match = re.search(r"(\d{1,2}:\d{2})", email_body)
    vs_match = re.search(r"([A-Za-z\s]+)\s+vs\s+([A-Za-z\s]+)", email_body, flags=re.IGNORECASE)
    date_match = re.search(r"on\s+([A-Za-z]+\s+\d{1,2}\s+[A-Za-z]+)", email_body, flags=re.IGNORECASE)
    venue_match = re.search(r"at ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", email_body)

    if time_match and vs_match:
        details = {
            "team1": vs_match.group(1).strip(),
            "team2": vs_match.group(2).strip(),
            "time": time_match.group(1),
            "date": parse_match_date(date_match.group(1)) if date_match else "Check email for date",
            "venue": venue_match.group(1).strip() if venue_match else "Check email for venue",
        }
        return details

    return None

def connect_to_email():
    """Connect to Gmail via IMAP"""
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        return mail
    except Exception as e:
        st.error(f"Failed to connect to email: {e}")
        return None

def get_latest_volleyball_email(mail):
    """Fetch the latest email from GO Mammoth with subject starting 'Coffee Stains' (including read)"""
    try:
        mail.select("inbox")
        # Include seen emails and filter by subject prefix
        status, messages = mail.search(None, f'(FROM "{SENDER_EMAIL}" SUBJECT "Coffee Stains")')
        
        if status != "OK" or not messages[0]:
            return None
        
        email_ids = messages[0].split()
        if not email_ids:
            return None
            
        latest_email_id = email_ids[-1]
        status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
        
        if status != "OK":
            return None
        
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode()
                            break
                else:
                    body = msg.get_payload(decode=True).decode()
                
                return body
        
        return None
        
    except Exception as e:
        st.error(f"Error fetching email: {e}")
        return None

# ==================== MAIN UI ====================
st.markdown('<div class="main-header">🏐 Volleyball Email Checker</div>', unsafe_allow_html=True)
st.markdown("### Check for new volleyball matches and send to WhatsApp")

# Sidebar
with st.sidebar:
    st.header('⚙️ Settings')
    st.info(f'📧 **Email:** {EMAIL}')
    st.info(f'🔍 **Monitoring:** {SENDER_EMAIL}')
    st.markdown('---')
    st.markdown('### 📱 WhatsApp Group')
    st.code(WHATSAPP_GROUP_URL, language=None)
    st.markdown('---')
    st.markdown('### ℹ️ How to Use')
    st.markdown("""
    1. Click **Check Email** 🔍
    2. See match details 📋
    3. Click **Copy Message** 📋
    4. Click **Open WhatsApp** 💬
    5. Paste and send! ✅
    """)
    st.markdown('---')
    st.caption('Made with ❤️ for Volleyball Team')

# Initialize session state
if 'match_details' not in st.session_state:
    st.session_state.match_details = None
if 'message' not in st.session_state:
    st.session_state.message = None
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Main button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button('🔍 Check Email', type='primary', use_container_width=True):
        st.session_state.logs = []
        st.session_state.logs.append(f"{datetime.now().strftime('%H:%M:%S')} - 🔍 Checking email...")
        
        with st.spinner('Connecting to Gmail...'):
            mail = connect_to_email()
            
            if mail:
                st.session_state.logs.append(f"{datetime.now().strftime('%H:%M:%S')} - ✅ Connected to Gmail")
                
                email_body = get_latest_volleyball_email(mail)
                
                if email_body:
                    st.session_state.logs.append(f"{datetime.now().strftime('%H:%M:%S')} - 📧 Found email from GO Mammoth")
                    
                    match_details = parse_volleyball_email(email_body)
                    
                    if match_details:
                        st.session_state.match_details = match_details
                        st.session_state.message = build_match_message(match_details)
                        st.session_state.logs.append(f"{datetime.now().strftime('%H:%M:%S')} - ✅ Match details extracted!")
                        st.success('✅ Match details found!')
                        st.balloons()
                    else:
                        st.warning('⚠️ Could not parse match details from email')
                        st.session_state.logs.append(f"{datetime.now().strftime('%H:%M:%S')} - ⚠️ Parsing failed")
                else:
                    st.info('📭 No new volleyball emails found')
                    st.session_state.logs.append(f"{datetime.now().strftime('%H:%M:%S')} - 📭 No unread emails")
                
                mail.close()
                mail.logout()

# Display match details
if st.session_state.match_details:
    st.markdown('---')
    
    # Match details card
    st.markdown('## ✅ Match Found!')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('### 👥 Teams')
        st.markdown(f"**{st.session_state.match_details['team1']}**  \nvs  \n**{st.session_state.match_details['team2']}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('### 📅 Date')
        date_val = st.session_state.match_details['date']
        if isinstance(date_val, dict):
            st.markdown(f"**{date_val['formatted']}**")
        else:
            st.markdown(f"**{date_val}**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('### ⏰ Time')
        st.markdown(f"**{st.session_state.match_details['time']}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('### 📍 Venue')
        st.markdown(f"**{st.session_state.match_details['venue']}**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('---')
    
    # WhatsApp message
    st.markdown('## 💬 WhatsApp Message')
    
    st.text_area(
        'Message ready to send:',
        value=st.session_state.message,
        height=300,
        disabled=True,
        key='message_display'
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('📋 Copy to Clipboard', use_container_width=True, type='secondary'):
            # Create a copy button using JavaScript
            st.components.v1.html(
                f"""
                <script>
                navigator.clipboard.writeText(`{st.session_state.message}`);
                </script>
                <p style="color: green;">✅ Message copied!</p>
                """,
                height=50
            )
            st.success('✅ Message copied to clipboard!')
    
    with col2:
        st.link_button(
            '💬 Open WhatsApp Group',
            WHATSAPP_GROUP_URL,
            use_container_width=True,
            type='primary'
        )

# Activity log
if st.session_state.logs:
    with st.expander('📋 Activity Log', expanded=False):
        for log in st.session_state.logs:
            st.text(log)

# Instructions at bottom
if not st.session_state.match_details:
    st.markdown('---')
    st.info("""
    👆 **Click the "Check Email" button above to get started!**
    
    This will:
    1. Connect to your Gmail account
    2. Check for new volleyball match emails from GO Mammoth
    3. Extract match details automatically
    4. Format a WhatsApp message for you to send
    """)