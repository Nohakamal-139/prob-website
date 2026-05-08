import filecmp

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
STOPWORDS = set([
    'the', 'to', 'a', 'i', 'and', 'is', 'in', 'it', 'you', 'of', 
    'for', 'on', 'my', 'me', 'with', 'at', 'have', 'your', 'be', 
    'this', 'that', 'so', 'are', 'it', 'its', 'from', 'but', 'ur','2','4', 'u', 'im', 
    'do', 'not', 'we', 'he', 'she', 'they','or','1','3','5','6','7','8','9','0'
])
st.set_page_config(page_title="Spam Detection System",page_icon="✉️" ,layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-DataSet-/master/spam.csv"
    df = pd.read_csv(url, encoding='latin-1')
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    df = df[['label', 'message']]
    return df
df = load_data()
def train (df):
    total_msgs = len(df)
    spam_msgs = len(df[df['label'] == 'spam'])
    ham_msgs = len(df[df['label'] == 'ham'])
    spam_prob = spam_msgs / total_msgs
    ham_prob = ham_msgs / total_msgs
    def get_word_counts(emails):
      all_words = []
      for msg in emails:
        words = re.findall(r'\b\w+\b', msg.lower())
        filtered_words = [w for w in words if w not in STOPWORDS]
        all_words.extend(filtered_words)
      return Counter(all_words),len(all_words)
    spam_counts , total_spam_words = get_word_counts(df[df['label'] == 'spam']['message'])
    ham_counts , total_ham_words = get_word_counts(df[df['label'] == 'ham']['message'])
    vocab_size = len(set(spam_counts.keys()).union(set(ham_counts.keys())))
    return spam_prob, ham_prob, spam_counts, ham_counts, total_spam_words, total_ham_words, vocab_size
spam_prob, ham_prob, spam_counts, ham_counts, total_spam_words, total_ham_words, vocab_size = train(df)
with st.sidebar:
    st.image("https://img.icons8.com/ios/100/ffffff/mail.png", width=60)
    st.subheader("Project Dashboard")
    st.markdown("---")
    st.info(f"📁 Dataset: {len(df)} records")
    st.success(f"✅ Normal (Ham): {len(df[df['label']=='ham'])}")
    st.error(f"🚨 Spam: {len(df[df['label']=='spam'])}")
    st.markdown("---")
    st.caption("Developed by us to save your inbox! ❤️")


st.subheader("🛡️ Smart AI Spam Detection Shield")
st.markdown("##### Probabilistic text classification using Naive Bayes Algorithm")
st.write("---")


st.write("---")
col1,col2 = st.columns([1.5,1])
with col1:
    st.subheader("Test Your Message")
    user_input = st.text_area("Enter a message to analyze:", height=150, placeholder="e.g., Congratulations! You've won a $1000 Walmart gift card. Click here to claim.")
    if st.button("Analyze"):
        if user_input:
            words = [w for w in re.findall(r'\w+', user_input.lower()) if w not in STOPWORDS]
            alpha =1
            spam_log_prob =spam_prob
            ham_log_prob =ham_prob
            for word in words:
                spam_log_prob *= (spam_counts.get(word, 0) + alpha) / (total_spam_words +  vocab_size)
                ham_log_prob *= (ham_counts.get(word, 0) + alpha) / (total_ham_words +  vocab_size)
            prob_sum = spam_log_prob + ham_log_prob
            final_spam = (spam_log_prob / prob_sum)*100
            final_ham = (ham_log_prob / prob_sum)*100
            if final_spam > final_ham:
                st.markdown(f'<div class="result-card" style="background-color: rgba(255, 75, 75, 0.2); border: 2px solid #ff4b4b;">'
                   f'<h2 style="color: #ff4b4b; margin:0;">🚨 SPAM DETECTED</h2>'
                   f'<p style="margin:5px;">This message shows patterns of malicious content.</p>'
                   f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-card" style="background-color: rgba(46, 204, 113, 0.2); border: 2px solid #2ecc71;">'
                   f'<h2 style="color: #2ecc71; margin:0;">✅ SAFE MESSAGE</h2>'
                   f'<p style="margin:5px;">This looks like a normal conversational message.</p>'
                   f'</div>', unsafe_allow_html=True)
            m1,m2 = st.columns(2)
            m1.metric("Spam Probability", f"{final_spam:.2f}%")
            m2.metric("Ham Probability", f"{final_ham:.2f}%")
            fig_res = px.bar(
                x=[final_spam, final_ham], 
                y=["Spam", "Ham"], 
                orientation='h',
                color=["Spam", "Ham"],
                color_discrete_map={"Spam": "#ff4b4b", "Ham": "#2ecc71"},
                labels={'x': 'Probability %', 'y': 'Type'}
            )
            fig_res.update_layout(showlegend=False, height=250, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.warning("Please enter a message to analyze.")
    if st.checkbox("Show Raw Dataset (Samples)"):
       st.dataframe(df.sample(10)) 
with col2:
    st.subheader("📈 Model Statistics")
    fig_pie = px.pie(
        df, names='label', 
        hole=0.4, 
        color='label',
        color_discrete_map={'ham': '#2ecc71', 'spam': '#ff4b4b'}
    )
    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption("Distribution of training data (Spam vs Ham)")
    st.subheader("🔠 Common Spam Words")
    top_words = pd.DataFrame(spam_counts.most_common(10), columns=['Word', 'Count'])
    fig_words = px.bar(top_words, x='Word', y='Count', color_discrete_sequence=['#ff4b4b'])
    fig_words.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_words, use_container_width=True)
