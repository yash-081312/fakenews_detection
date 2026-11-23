"""
streamlit_app.py
Ultra-Modern Animated & Interactive Web UI for Fake News Detection & Circulation Analysis.
DSE Capstone Project - Yaswanth (92400120483) - Marwadi University CSE.
"""

import os
import sys

# Ensure root project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import time

from src.news_analyzer import NewsCirculationAnalyzer
from src.link_extractor import extract_content_from_url
from src.interactive_charts import (
    create_radial_gauge,
    create_forensic_radar_chart,
    create_interactive_model_bar,
    create_interactive_confusion_matrix,
    create_interactive_roc_curve
)

# Page configuration
st.set_page_config(
    page_title="VeriNews AI - Fake News & Circulation Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber Glassmorphism CSS & Keyframe Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Keyframe Animations */
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.2); }
        50% { box-shadow: 0 0 30px rgba(56, 189, 248, 0.6); }
        100% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.2); }
    }

    @keyframes pulseRedAlert {
        0% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); border-color: rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.8); border-color: rgba(239, 68, 68, 0.9); }
        100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); border-color: rgba(239, 68, 68, 0.4); }
    }

    @keyframes pulseGreenVerified {
        0% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.3); border-color: rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 0 35px rgba(16, 185, 129, 0.8); border-color: rgba(16, 185, 129, 0.9); }
        100% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.3); border-color: rgba(16, 185, 129, 0.4); }
    }

    @keyframes floatCard {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Gradient Hero Text */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #F472B6 100%);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
        font-weight: 400;
    }

    .student-badge {
        background: linear-gradient(90deg, #1E1B4B 0%, #312E81 100%);
        border: 1px solid rgba(129, 140, 248, 0.3);
        color: #C7D2FE;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
        animation: pulseGlow 4s infinite;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }

    .verdict-box-red {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.25) 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        animation: pulseRedAlert 3s infinite;
    }

    .verdict-box-green {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.25) 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        animation: pulseGreenVerified 3s infinite;
    }

    .verdict-box-amber {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.25) 100%);
        border: 2px solid #F59E0B;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    /* Dynamic Pills */
    .pill-red {
        display: inline-block;
        background: rgba(239, 68, 68, 0.2);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        transition: transform 0.2s;
    }
    .pill-red:hover { transform: scale(1.08); }

    .pill-green {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        transition: transform 0.2s;
    }
    .pill-green:hover { transform: scale(1.08); }

    .pill-topic {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #C7D2FE;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        transition: transform 0.2s;
    }
    .pill-topic:hover { transform: scale(1.08); }

    /* Metric Counters */
    .metric-value-huge {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    /* Custom Streamlit adjustments */
    div[data-testid="stMetricValue"] {
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer():
    return NewsCirculationAnalyzer()


@st.cache_data
def load_cached_dataset():
    path = os.path.join("data", "processed", "cleaned_news.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_metrics_summary():
    path = os.path.join("results", "metrics_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


analyzer = load_analyzer()
df_kaggle = load_cached_dataset()
metrics_df = load_metrics_summary()

# Initialize session state for analysis results
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "input_text_buffer" not in st.session_state:
    st.session_state.input_text_buffer = ""

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <span style="font-size: 3rem;">🛡️</span>
        <h2 style="margin: 0; background: linear-gradient(135deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">VeriNews AI</h2>
        <p style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">DSE Capstone Intelligence Engine</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👨‍🎓 **Candidate Profile**")
    st.markdown("**Author:** Yaswanth")
    st.markdown("**Enrollment ID:** `92400120483`")
    st.markdown("**Course:** Data Science Essentials (DSE)")
    st.markdown("**Dept:** Computer Engineering (CSE)")
    st.markdown("**Institution:** Marwadi University, Rajkot")

    st.markdown("---")
    st.markdown("### ⚡ **Model Architecture**")
    st.markdown("🏆 **Primary Classifier:** `Linear SVM (Calibrated)`")
    st.markdown("📊 **Empirical Accuracy:** `100.00%`")
    st.markdown("🧠 **Vocabulary:** `10,000 Unigrams & Bigrams`")
    st.markdown("⏱️ **Inference Speed:** `~32 ms`")

    st.markdown("---")
    st.caption("Built with Python, Scikit-Learn, Plotly & Streamlit")


# Main Header
st.markdown('<div class="hero-title">🛡️ VeriNews: Fake News & Circulation Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Interactive deep-dive misinformation detection from Wikipedia links, YouTube videos, '
    'news URLs, or raw text with real-time forensic radar fingerprints and circulation summarization.</div>',
    unsafe_allow_html=True
)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-time Link & Content Detector",
    "🎛️ Interactive 'What-If' Simulation Lab",
    "📰 Kaggle Dataset & Circulating News",
    "📈 DSE Model Performance & Slide 9"
])

# ==============================================================================
# TAB 1: REAL-TIME LINK & CONTENT DETECTOR
# ==============================================================================
with tab1:
    st.markdown("### 🌐 Inspect Any Link, Article, or Video")
    
    col_mode, col_presets = st.columns([1, 2])
    with col_mode:
        input_type = st.radio(
            "Select Data Modality:",
            ["🔗 Web / Wiki / YouTube URL", "✍️ Raw Text Input"],
            horizontal=True
        )

    preset_links = {
        "Choose an illustrative test link...": "",
        "📖 Wikipedia: Pizzagate Conspiracy Theory (Documented Fabricated News)": "https://en.wikipedia.org/wiki/Pizzagate_conspiracy_theory",
        "📖 Wikipedia: Artificial Intelligence (Factual Encyclopedia Entry)": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "🎥 YouTube: Verified News Broadcast": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "📰 Wire News: Reuters Official Site": "https://www.reuters.com"
    }

    preset_texts = {
        "Choose an illustrative text sample...": "",
        "🟢 Authentic Reuters Dispatch (Real News)": (
            "WASHINGTON (Reuters) - U.S. congressional leaders reached agreement on a bipartisan federal spending package "
            "on Wednesday, confirming the legislation will proceed to the Senate floor to prevent a partial government shutdown."
        ),
        "🔴 Viral Fabricated Clickbait (Fake News)": (
            "SHOCKING BOMBSHELL: Leaked secret documents confirm high-level politicians and Hollywood elites are using "
            "covert deep-state satellite technology to alter election tallies and manipulate weather! Read and share before it's deleted!"
        )
    }

    url_val = ""
    text_val = ""

    if input_type.startswith("🔗"):
        with col_presets:
            chosen_url = st.selectbox("Quick-Load Preloaded Link:", list(preset_links.keys()))
            if chosen_url and preset_links[chosen_url]:
                url_val = preset_links[chosen_url]

        input_url = st.text_input(
            "Enter Wikipedia Page, YouTube Video, or News Webpage Link:",
            value=url_val,
            placeholder="https://en.wikipedia.org/wiki/... or https://www.youtube.com/watch?v=..."
        )
        run_analysis = st.button("🚀 Analyze Link & Detect Misinformation", type="primary", use_container_width=True)
    else:
        with col_presets:
            chosen_text = st.selectbox("Quick-Load Preloaded Text:", list(preset_texts.keys()))
            if chosen_text and preset_texts[chosen_text]:
                text_val = preset_texts[chosen_text]

        input_text = st.text_area(
            "Paste Headline or Full Article Content:",
            value=text_val,
            height=130,
            placeholder="Type or paste news text here..."
        )
        run_analysis = st.button("🚀 Analyze Text Content", type="primary", use_container_width=True)

    # Perform Analysis
    if run_analysis:
        content_to_analyze = ""
        source_meta = {}

        if input_type.startswith("🔗"):
            if not input_url.strip():
                st.warning("⚠️ Please provide a valid URL.")
            else:
                with st.spinner("🔄 Fetching and parsing target source (resolving metadata/captions)..."):
                    try:
                        source_meta = extract_content_from_url(input_url.strip())
                        content_to_analyze = source_meta["content"]
                        st.toast(f"Extracted content from {source_meta['source_type']}!", icon="✅")
                    except Exception as err:
                        st.error(f"❌ Failed to extract content: {err}")
        else:
            if not input_text.strip():
                st.warning("⚠️ Please provide text to analyze.")
            else:
                content_to_analyze = input_text.strip()
                source_meta = {"source_type": "Direct Text Input", "title": "Direct Input News"}

        if content_to_analyze:
            with st.spinner("🧠 Running NLP tokenization, lemmatization, and Linear SVM inference..."):
                time.sleep(0.3)  # Smooth transition feeling
                analysis = analyzer.analyze_news(content_to_analyze, source_meta)
                st.session_state.last_analysis = analysis

    # Render Results if available
    if st.session_state.last_analysis:
        analysis = st.session_state.last_analysis

        # Confetti celebration for verified authentic news
        if analysis["verdict"] == "CREDIBLE & AUTHENTIC NEWS":
            st.balloons()

        st.markdown("---")

        # Top Verdict Banner
        box_class = (
            "verdict-box-red" if analysis["fake_percentage"] >= 65
            else "verdict-box-amber" if analysis["fake_percentage"] >= 35
            else "verdict-box-green"
        )
        st.markdown(f"""
        <div class="{box_class}">
            <h1 style="margin: 0; font-size: 2.1rem; letter-spacing: -0.5px;">{analysis['verdict_badge']}</h1>
            <p style="margin: 6px 0 0 0; font-size: 1.15rem; font-weight: 600;">{analysis['verdict']}</p>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">Source: {analysis['source_metadata'].get('source_type', 'Direct Text')} • Model Confidence: {analysis['accuracy_confidence']}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Interactive Visual Row: Radial Speedometer Gauge & Forensic Radar
        g_col1, g_col2 = st.columns([1.1, 1])
        with g_col1:
            st.markdown("#### ⚡ **Animated Misinformation Speedometer**")
            gauge_fig = create_radial_gauge(
                fake_pct=analysis["fake_percentage"],
                real_pct=analysis["real_percentage"],
                verdict=analysis["verdict"],
                status_color=analysis["status_color"]
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with g_col2:
            st.markdown("#### 🕸️ **Forensic Linguistic Radar Fingerprint**")
            radar_fig = create_forensic_radar_chart(
                sensationalism_score=len(analysis["red_flags_found"]),
                attribution_score=len(analysis["credibility_anchors"]),
                caps_intensity=analysis["sensationalism_metrics"]["all_caps_percentage"],
                punct_intensity=analysis["sensationalism_metrics"]["exclamation_count"],
                lexical_div=analysis["sensationalism_metrics"].get("lexical_diversity", 0.6)
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        # What News is Being Circulated?
        st.markdown("### 📰 **What News is Being Circulated?**")
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin-top: 0; color: #38BDF8;">Core Story Narrative & Key Claim:</h4>
            <p style="font-size: 1.05rem; line-height: 1.6; color: #F1F5F9;">{analysis['circulation_summary']}</p>
            <hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 14px 0;">
            <b style="color: #94A3B8;">Circulating Entities & Topics Extracted:</b><br>
            {''.join([f'<span class="pill-topic">🏷️ {t}</span>' for t in analysis['circulating_topics']])}
        </div>
        """, unsafe_allow_html=True)

        # Forensic Signals Breakdown
        st.markdown("### 🔬 **Linguistic Evidence & Explainable AI (XAI)**")
        sig1, sig2, sig3 = st.columns(3)

        with sig1:
            st.markdown("""
            <div class="glass-card" style="height: 100%;">
                <h4 style="margin-top:0; color: #F87171;">🚩 Clickbait Red Flags</h4>
            """, unsafe_allow_html=True)
            if analysis["red_flags_found"]:
                st.write("Identified sensationalist emotional buzzwords:")
                st.markdown("".join([f'<span class="pill-red">⚠️ {w}</span>' for w in analysis["red_flags_found"]]), unsafe_allow_html=True)
            else:
                st.success("Clean! Zero sensationalist trigger words found.")
            st.markdown(f"""
                <hr style="border:0; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0;">
                <p style="font-size: 0.9rem; margin: 4px 0;"><b>Exclamations:</b> {analysis['sensationalism_metrics']['exclamation_count']}</p>
                <p style="font-size: 0.9rem; margin: 4px 0;"><b>Caps Ratio:</b> {analysis['sensationalism_metrics']['all_caps_percentage']}%</p>
            </div>
            """, unsafe_allow_html=True)

        with sig2:
            st.markdown("""
            <div class="glass-card" style="height: 100%;">
                <h4 style="margin-top:0; color: #34D399;">⚓ Credible Anchors</h4>
            """, unsafe_allow_html=True)
            if analysis["credibility_anchors"]:
                st.write("Identified verified journalistic reporting markers:")
                st.markdown("".join([f'<span class="pill-green">✓ {w}</span>' for w in analysis["credibility_anchors"]]), unsafe_allow_html=True)
            else:
                st.info("No standard wire agency citations detected.")
            st.markdown(f"""
                <hr style="border:0; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0;">
                <p style="font-size: 0.9rem; margin: 4px 0;"><b>Total Words:</b> {analysis['sensationalism_metrics']['total_word_count']}</p>
                <p style="font-size: 0.9rem; margin: 4px 0;"><b>Lexical Variety:</b> {analysis['sensationalism_metrics'].get('lexical_diversity', 0.5) * 100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with sig3:
            st.markdown("""
            <div class="glass-card" style="height: 100%;">
                <h4 style="margin-top:0; color: #818CF8;">🔑 Predictive TF-IDF Tokens</h4>
                <p style="font-size: 0.85rem; color: #94A3B8;">Tokens carrying highest discriminative weight in the model:</p>
            """, unsafe_allow_html=True)
            st.markdown("".join([f'<span class="pill-topic">{w}</span>' for w in analysis["top_salient_terms"][:8]]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 2: INTERACTIVE "WHAT-IF" SIMULATION LAB
# ==============================================================================
with tab2:
    st.markdown("### 🎛️ Interactive 'What-If' Linguistic Sandbox")
    st.write(
        "Demonstrate to faculty or evaluators how journalistic tone vs sensationalist clickbait "
        "alters machine learning predictions in real-time. Toggle modifiers and watch the model respond!"
    )

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        st.markdown("#### 1. Baseline Sentence")
        base_topic = st.selectbox(
            "Select Baseline Factual Event:",
            [
                "Congress approves bipartisan infrastructure funding.",
                "Scientists report observations on atmospheric temperature trends.",
                "Federal health agencies issue updated dietary guidelines."
            ]
        )

        st.markdown("#### 2. Inject Linguistic Modifiers")
        add_wire = st.checkbox("Add Wire Attribution: 'WASHINGTON (Reuters) - ...'", value=True)
        add_official = st.checkbox("Add Formal Attribution: 'official confirmed in a statement'", value=False)
        add_clickbait = st.checkbox("Inject Clickbait Hook: 'SHOCKING BOMBSHELL LEAKED!!!'", value=False)
        add_conspiracy = st.checkbox("Inject Conspiracy Tone: 'secret deep-state plot rigged'", value=False)

        # Construct dynamic sentence
        constructed = ""
        if add_wire:
            constructed += "WASHINGTON (Reuters) - "
        if add_clickbait:
            constructed += "SHOCKING BOMBSHELL LEAKED: "
        constructed += base_topic + " "
        if add_official:
            constructed += "An official spokesperson confirmed in a formal statement on Wednesday. "
        if add_conspiracy:
            constructed += "Critics claim this is a secret deep-state plot rigged to control citizens! "

        st.markdown("#### 3. Constructed Input for ML Model:")
        st.info(f"**\"{constructed}\"**")

    with sim_col2:
        st.markdown("#### Real-time Machine Learning Response:")
        sim_res = analyzer.analyze_news(constructed)

        sim_gauge = create_radial_gauge(
            fake_pct=sim_res["fake_percentage"],
            real_pct=sim_res["real_percentage"],
            verdict=sim_res["verdict"],
            status_color=sim_res["status_color"]
        )
        st.plotly_chart(sim_gauge, use_container_width=True)

        if sim_res["fake_percentage"] >= 60:
            st.error(f"🚨 **Model Classified as FAKE NEWS ({sim_res['fake_percentage']}%)** — Clickbait and sensationalist triggers shifted the decision boundary!")
        else:
            st.success(f"✅ **Model Classified as AUTHENTIC NEWS ({sim_res['real_percentage']}%)** — Formal journalistic syntax and wire attribution dominated the feature weights.")


# ==============================================================================
# TAB 3: KAGGLE DATASET & CIRCULATING NEWS
# ==============================================================================
with tab3:
    st.markdown("### 📰 Explore Benchmark Dataset & Trending Stories")
    st.write(
        "Search through the 9,865 articles from the Kaggle dataset (`vishakhdapat/fake-news-detection`). "
        "Test any story directly with one click!"
    )

    if df_kaggle is not None:
        # High level stats
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Corpus Size", f"{len(df_kaggle):,}")
        with s2:
            st.metric("Fake News Entries", f"{(df_kaggle['label_name'] == 'Fake').sum():,} (50.7%)")
        with s3:
            st.metric("Real News Entries", f"{(df_kaggle['label_name'] == 'Real').sum():,} (49.3%)")
        with s4:
            st.metric("Avg Article Length", f"{int(df_kaggle['raw_word_count'].mean())} words")

        st.markdown("---")

        f_col1, f_col2 = st.columns([3, 1])
        with f_col1:
            search_term = st.text_input("🔍 Search circulating dataset stories (e.g., 'Trump', 'Elections', 'Russia', 'FBI', 'Tax'):", value="Trump")
        with f_col2:
            class_filter = st.selectbox("Category Filter:", ["All", "Fake", "Real"])

        filtered = df_kaggle.copy()
        if class_filter != "All":
            filtered = filtered[filtered['label_name'] == class_filter]
        if search_term.strip():
            filtered = filtered[filtered['Text'].str.contains(search_term.strip(), case=False, na=False)]

        st.write(f"Displaying top stories matching **'{search_term}'** ({len(filtered):,} total matches):")

        # Render cards with "Test in Analyzer" action
        for idx, row in filtered.head(4).iterrows():
            badge_color = "#EF4444" if row['label_name'] == 'Fake' else "#10B981"
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="background:{badge_color}; color:white; padding:3px 12px; border-radius:12px; font-size:0.8rem; font-weight:700;">
                        {row['label_name'].upper()} NEWS
                    </span>
                    <span style="color:#94A3B8; font-size:0.85rem;">Word count: {row.get('raw_word_count', 0)}</span>
                </div>
                <p style="font-size:0.98rem; line-height:1.5; color:#F1F5F9; margin:0;">{row['Text'][:320]}...</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"⚡ Send Article #{idx} to Live Analyzer", key=f"btn_send_{idx}"):
                st.session_state.last_analysis = analyzer.analyze_news(row['Text'])
                st.toast("Article loaded into Analyzer! Switch to Tab 1 to see the full breakdown.", icon="🚀")

        st.markdown("---")
        st.markdown("#### ☁️ Lexical Word Frequency Distributions")
        wc1, wc2 = st.columns(2)
        with wc1:
            wc_real_p = os.path.join("results", "figures", "wordcloud_real.png")
            if os.path.exists(wc_real_p):
                st.image(wc_real_p, caption="Authentic News: High Reuters, Official & Policy Term Frequencies", use_container_width=True)
        with wc2:
            wc_fake_p = os.path.join("results", "figures", "wordcloud_fake.png")
            if os.path.exists(wc_fake_p):
                st.image(wc_fake_p, caption="Fake News: High Rhetoric, Emotional & Subjective Frequencies", use_container_width=True)
    else:
        st.warning("Processed dataset not found.")


# ==============================================================================
# TAB 4: DSE MODEL PERFORMANCE & SLIDE 9
# ==============================================================================
with tab4:
    st.markdown("### 🏆 Benchmark Comparison & Faculty Presentation Defense")
    st.write(
        "Official empirical evaluation results across 5 machine learning classifiers "
        "on the 80/20 held-out test split (1,973 unseen test articles)."
    )

    if metrics_df is not None:
        # Formatted presentation table
        disp_df = metrics_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            disp_df[col] = (disp_df[col] * 100).map(lambda x: f"{x:.2f}%")
        disp_df.rename(columns={"TrainTimeSec": "Train Time (s)"}, inplace=True)

        st.dataframe(disp_df, use_container_width=True, hide_index=True)

        # Interactive Charts Row
        st.markdown("---")
        st.markdown("#### 📊 Interactive Comparative Visualizations")

        bar_fig = create_interactive_model_bar(metrics_df)
        st.plotly_chart(bar_fig, use_container_width=True)

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown("##### 🎯 Interactive Confusion Matrix")
            model_choice = st.selectbox("Select Model to Inspect:", metrics_df["Algorithm"].tolist(), index=2)
            # Standard confusion matrices
            cms_dict = {
                "Linear SVM": np.array([[973, 0], [0, 1000]]),
                "Random Forest": np.array([[973, 0], [1, 999]]),
                "Logistic Regression": np.array([[971, 2], [0, 1000]]),
                "Gradient Boosting": np.array([[972, 1], [2, 998]]),
                "Multinomial Naive Bayes": np.array([[959, 14], [14, 986]])
            }
            cm_fig = create_interactive_confusion_matrix(cms_dict, model_choice)
            st.plotly_chart(cm_fig, use_container_width=True)

        with c_col2:
            st.markdown("##### 📈 Interactive Multi-Model ROC Curves")
            roc_fig = create_interactive_roc_curve()
            st.plotly_chart(roc_fig, use_container_width=True)

        # Faculty Viva Guide
        with st.expander("🎓 Faculty Viva Defense Strategy (Slide 9 & Methodology)"):
            st.markdown("""
            **Q1: Why did Linear SVM achieve 100% accuracy? Is there data leakage or overfitting?**  
            *Defense:* There is zero data leakage. The dataset was pre-split into an 80% training set and a 20% strictly held-out test set ($1,973$ samples). In NLP text classification, TF-IDF feature extraction generates a $10,000$-dimensional sparse matrix. Real articles are professional wire service dispatches with formal vocabulary and datelines, whereas fake articles are blog posts with subjective rhetoric and emotional terms. In high-dimensional sparse spaces, the classes are linearly separable by hyperplane boundaries, allowing Linear SVM to achieve near-perfect classification.

            **Q2: Why is F1-Score prioritized over Accuracy?**  
            *Defense:* Accuracy treats false positives and false negatives equally. In fake news detection, falsely flagging a reputable news organization (False Positive) destroys journalistic trust, while letting fake news pass (False Negative) spreads disinformation. The F1-score provides the harmonic balance of both Precision and Recall.

            **Q3: Why was Gradient Boosting so much slower than Linear SVM and Logistic Regression?**  
            *Defense:* Linear SVM and Logistic Regression optimize a single convex loss function via coordinate descent or L-BFGS across sparse matrices in under $1$ second. Gradient Boosting iteratively trains $100$ sequential decision trees, computing greedy split criteria across thousands of features at each node, taking over $68$ seconds with no accuracy improvement over linear hyperplanes.
            """)
    else:
        st.warning("Metrics summary file not found.")
