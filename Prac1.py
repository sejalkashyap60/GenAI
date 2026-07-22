# ============================================================
# NLP TEXT ANALYZER - STREAMLIT WEB APPLICATION
# ============================================================

import streamlit as st
import nltk
import spacy

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NLP Text Analyzer",
    page_icon="📝",
    layout="wide"
)


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
# TITLE
# ============================================================
st.title("Developed by  SEJAL KASHYAP (44)")
st.title("📝 NLP Text Analyzer")

st.write(
    "Enter a paragraph below and perform "
    "different Natural Language Processing operations."
)


# ============================================================
# TEXT INPUT
# ============================================================

text = st.text_area(
    "Enter your paragraph here:",
    height=200,
    placeholder=(
        "Example: Barack Obama was born in Hawaii. "
        "He served as the 44th President of the United States."
    )
)


# ============================================================
# BUTTONS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    analyze_button = st.button(
        "🔍 Analyze Text",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "🗑 Clear",
        use_container_width=True
    )


# ============================================================
# CLEAR BUTTON
# ============================================================

if clear_button:

    st.rerun()


# ============================================================
# ANALYZE TEXT
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # Check Empty Text
    # --------------------------------------------------------

    if text.strip() == "":

        st.warning(
            "⚠️ Please enter some text before analyzing."
        )

    else:

        # ====================================================
        # PROCESS TEXT WITH SPACY
        # ====================================================

        doc = nlp(text)


        # ====================================================
        # 1. SENTENCE SEGMENTATION
        # ====================================================

        st.header(
            "1️⃣ Sentence Segmentation"
        )

        sentences = sent_tokenize(
            text
        )

        for i, sentence in enumerate(
            sentences,
            1
        ):

            st.write(
                f"**{i}.** {sentence}"
            )


        # ====================================================
        # 2. WORD TOKENIZATION
        # ====================================================

        st.header(
            "2️⃣ Word Tokenization"
        )

        words = word_tokenize(
            text
        )

        st.write(
            words
        )

        st.write(
            f"**Total Tokens:** {len(words)}"
        )


        # ====================================================
        # 3. STOP WORD REMOVAL
        # ====================================================

        st.header(
            "3️⃣ Stop Word Removal"
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

        st.write(
            filtered_words
        )


        # ====================================================
        # 4. STEMMING
        # ====================================================

        st.header(
            "4️⃣ Stemming"
        )

        stemmer = PorterStemmer()

        stemmed_words = [

            stemmer.stem(
                word
            )

            for word in filtered_words

        ]

        stemming_results = []

        for original, stemmed in zip(
            filtered_words,
            stemmed_words
        ):

            stemming_results.append(
                f"{original} → {stemmed}"
            )

        st.write(
            stemming_results
        )


        # ====================================================
        # 5. LEMMATIZATION
        # ====================================================

        st.header(
            "5️⃣ Lemmatization"
        )

        lemmatizer = WordNetLemmatizer()

        lemmatized_words = [

            lemmatizer.lemmatize(
                word
            )

            for word in filtered_words

        ]

        lemmatization_results = []

        for original, lemma in zip(
            filtered_words,
            lemmatized_words
        ):

            lemmatization_results.append(
                f"{original} → {lemma}"
            )

        st.write(
            lemmatization_results
        )


        # ====================================================
        # 6. POS TAGGING
        # ====================================================

        st.header(
            "6️⃣ POS Tagging"
        )

        pos_tags = nltk.pos_tag(
            words
        )

        for word, tag in pos_tags:

            st.write(
                f"**{word}** → `{tag}`"
            )


        # ====================================================
        # 7. NAMED ENTITY RECOGNITION
        # ====================================================

        st.header(
            "7️⃣ Named Entity Recognition"
        )

        if len(doc.ents) == 0:

            st.info(
                "No Named Entities Found."
            )

        else:

            for entity in doc.ents:

                st.write(
                    f"**{entity.text}** → "
                    f"`{entity.label_}`"
                )


        # ====================================================
        # 8. DEPENDENCY PARSING
        # ====================================================

        st.header(
            "8️⃣ Dependency Parsing"
        )

        st.write(
            "Word → Dependency → Head"
        )

        for token in doc:

            st.write(
                f"**{token.text}** → "
                f"`{token.dep_}` → "
                f"**{token.head.text}**"
            )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            "✅ NLP Text Analysis Completed Successfully!"
        )