# ============================================================
# NLP TEXT ANALYZER - PROFESSIONAL STREAMLIT WEB APPLICATION
# ============================================================

import streamlit as st
import nltk
import spacy
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NLP Text Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main Background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f5f7ff 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    /* Main Content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Main Title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #64748b;
        margin-bottom: 30px;
    }

    /* Student Card */
    .student-card {
        background: linear-gradient(
            135deg,
            #1e3a8a,
            #4f46e5
        );
        padding: 22px;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0px 8px 25px rgba(30, 58, 138, 0.25);
        margin-bottom: 25px;
    }

    .student-name {
        font-size: 24px;
        font-weight: 700;
    }

    .student-info {
        font-size: 16px;
        margin-top: 5px;
        opacity: 0.9;
    }

    /* Info Cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.07);
        border: 1px solid #e2e8f0;
    }

    .info-number {
        font-size: 28px;
        font-weight: 800;
        color: #4f46e5;
    }

    .info-label {
        font-size: 14px;
        color: #64748b;
    }

    /* Section Headers */
    .section-title {
        color: #1e3a8a;
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Pipeline */
    .pipeline {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        margin: 25px 0;
    }

    .pipeline-item {
        background: white;
        padding: 12px 18px;
        border-radius: 25px;
        color: #3730a3;
        font-weight: 600;
        border: 2px solid #c7d2fe;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
    }

    .arrow {
        color: #6366f1;
        font-size: 22px;
        font-weight: bold;
    }

    /* Result Box */
    .result-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4f46e5;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 25px;
        margin-top: 40px;
        border-top: 1px solid #e2e8f0;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
        font-size: 16px;
        font-weight: 700;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# DOWNLOAD NLTK RESOURCES
# ============================================================

@st.cache_resource
def download_nltk_resources():

    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng"
    ]

    for resource in resources:

        nltk.download(
            resource,
            quiet=True
        )


download_nltk_resources()


# ============================================================
# LOAD SPACY MODEL
# ============================================================

@st.cache_resource
def load_spacy_model():

    return spacy.load(
        "en_core_web_sm"
    )


nlp = load_spacy_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 NLP Analyzer"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### 📚 NLP Operations"
    )

    st.write(
        "✅ Sentence Segmentation"
    )

    st.write(
        "✅ Word Tokenization"
    )

    st.write(
        "✅ Stop Word Removal"
    )

    st.write(
        "✅ Stemming"
    )

    st.write(
        "✅ Lemmatization"
    )

    st.write(
        "✅ POS Tagging"
    )

    st.write(
        "✅ Named Entity Recognition"
    )

    st.write(
        "✅ Dependency Parsing"
    )

    st.markdown(
        "---"
    )

    st.info(
        "Enter any English paragraph "
        "and click Analyze Text to "
        "perform NLP processing."
    )


# ============================================================
# STUDENT INFORMATION
# ============================================================

st.markdown("""
<div class="student-card">

<div class="student-name">
👩‍💻 SEJAL KASHYAP
</div>

<div class="student-info">
Roll No: 44 &nbsp; | &nbsp;
MCA (AI & ML) &nbsp; | &nbsp;
NLP Practical
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🧠 NLP Text Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore Natural Language Processing through interactive text analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# NLP PIPELINE
# ============================================================

st.markdown("""
<div class="pipeline">

<div class="pipeline-item">📝 Input</div>

<div class="arrow">→</div>

<div class="pipeline-item">🔤 Tokenization</div>

<div class="arrow">→</div>

<div class="pipeline-item">🛑 Stop Words</div>

<div class="arrow">→</div>

<div class="pipeline-item">🌱 Stemming</div>

<div class="arrow">→</div>

<div class="pipeline-item">📖 Lemmatization</div>

<div class="arrow">→</div>

<div class="pipeline-item">🏷️ POS</div>

<div class="arrow">→</div>

<div class="pipeline-item">🔎 NER</div>

<div class="arrow">→</div>

<div class="pipeline-item">🔗 Dependency</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# TEXT INPUT
# ============================================================

st.markdown(
    '<div class="section-title">📝 Enter Your Text</div>',
    unsafe_allow_html=True
)

text = st.text_area(

    "Enter paragraph",

    height=180,

    placeholder=(
        "Example: Barack Obama was born in Hawaii. "
        "He served as the 44th President of the United States. "
        "He visited India in 2015."
    ),

    label_visibility="collapsed"
)


# ============================================================
# BUTTONS
# ============================================================

col1, col2, col3 = st.columns(
    [1, 1, 2]
)

with col1:

    analyze_button = st.button(
        "🔍 Analyze Text",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


# ============================================================
# CLEAR BUTTON
# ============================================================

if clear_button:

    st.rerun()


# ============================================================
# ANALYZE BUTTON
# ============================================================

if analyze_button:

    if text.strip() == "":

        st.warning(
            "⚠️ Please enter some text before analyzing."
        )

    else:

        # ====================================================
        # PROCESS TEXT
        # ====================================================

        doc = nlp(
            text
        )

        sentences = sent_tokenize(
            text
        )

        words = word_tokenize(
            text
        )

        stop_words = set(
            stopwords.words(
                "english"
            )
        )

        filtered_words = [

            word

            for word in words

            if word.lower()
            not in stop_words

        ]


        # ====================================================
        # STEMMING
        # ====================================================

        stemmer = PorterStemmer()

        stemmed_words = [

            stemmer.stem(
                word
            )

            for word in filtered_words

        ]


        # ====================================================
        # LEMMATIZATION
        # ====================================================

        lemmatizer = WordNetLemmatizer()

        lemmatized_words = [

            lemmatizer.lemmatize(
                word
            )

            for word in filtered_words

        ]


        # ====================================================
        # POS TAGGING
        # ====================================================

        pos_tags = nltk.pos_tag(
            words
        )


        # ====================================================
        # DASHBOARD METRICS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Text Statistics'
            '</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3, m4, m5 = st.columns(
            5
        )

        with m1:

            st.metric(
                "Characters",
                len(text)
            )

        with m2:

            st.metric(
                "Sentences",
                len(sentences)
            )

        with m3:

            st.metric(
                "Total Tokens",
                len(words)
            )

        with m4:

            st.metric(
                "After Stop Words",
                len(filtered_words)
            )

        with m5:

            st.metric(
                "Named Entities",
                len(doc.ents)
            )


        # ====================================================
        # RESULTS TABS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🔬 NLP Analysis Results'
            '</div>',
            unsafe_allow_html=True
        )


        tab1, tab2, tab3, tab4 = st.tabs(

            [
                "📑 Basic Processing",
                "🌱 Stemming & Lemmatization",
                "🏷️ POS Tagging",
                "🔎 NER & Dependency"
            ]

        )


        # ====================================================
        # TAB 1
        # ====================================================

        with tab1:

            st.subheader(
                "1️⃣ Sentence Segmentation"
            )

            for i, sentence in enumerate(
                sentences,
                1
            ):

                st.info(
                    f"Sentence {i}: {sentence}"
                )


            st.subheader(
                "2️⃣ Word Tokenization"
            )

            st.write(
                words
            )


            st.subheader(
                "3️⃣ Stop Word Removal"
            )

            st.write(
                filtered_words
            )


        # ====================================================
        # TAB 2
        # ====================================================

        with tab2:

            col1, col2 = st.columns(
                2
            )


            with col1:

                st.subheader(
                    "🌱 Stemming"
                )

                stemming_df = pd.DataFrame(

                    {

                        "Original Word":
                        filtered_words,

                        "Stemmed Word":
                        stemmed_words

                    }

                )

                st.dataframe(

                    stemming_df,

                    use_container_width=True,

                    hide_index=True

                )


            with col2:

                st.subheader(
                    "📖 Lemmatization"
                )

                lemma_df = pd.DataFrame(

                    {

                        "Original Word":
                        filtered_words,

                        "Lemma":
                        lemmatized_words

                    }

                )

                st.dataframe(

                    lemma_df,

                    use_container_width=True,

                    hide_index=True

                )


        # ====================================================
        # TAB 3
        # ====================================================

        with tab3:

            st.subheader(
                "🏷️ Part-of-Speech Tagging"
            )

            pos_df = pd.DataFrame(

                pos_tags,

                columns=[
                    "Word",
                    "POS Tag"
                ]

            )

            st.dataframe(

                pos_df,

                use_container_width=True,

                hide_index=True

            )


        # ====================================================
        # TAB 4
        # ====================================================

        with tab4:

            # ----------------------------------------------
            # NER
            # ----------------------------------------------

            st.subheader(
                "🔎 Named Entity Recognition"
            )

            if len(doc.ents) == 0:

                st.info(
                    "No Named Entities Found."
                )

            else:

                ner_data = [

                    {

                        "Entity":
                        entity.text,

                        "Label":
                        entity.label_,

                        "Description":
                        spacy.explain(
                            entity.label_
                        )

                    }

                    for entity in doc.ents

                ]

                ner_df = pd.DataFrame(
                    ner_data
                )

                st.dataframe(

                    ner_df,

                    use_container_width=True,

                    hide_index=True

                )


            # ----------------------------------------------
            # DEPENDENCY PARSING
            # ----------------------------------------------

            st.subheader(
                "🔗 Dependency Parsing"
            )

            dependency_data = [

                {

                    "Word":
                    token.text,

                    "Dependency":
                    token.dep_,

                    "Head":
                    token.head.text,

                    "POS":
                    token.pos_

                }

                for token in doc

            ]

            dependency_df = pd.DataFrame(

                dependency_data

            )

            st.dataframe(

                dependency_df,

                use_container_width=True,

                hide_index=True

            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🧠 <b>NLP Text Analyzer</b><br>

Built using Python • NLTK • spaCy • Streamlit<br>

MCA (AI & ML) | NLP Practical

</div>
""", unsafe_allow_html=True)