import streamlit as st
from datetime import datetime
from pipelines import PipelineSentiment, PipelineNMT, PipelineLangDetect


@st.cache(allow_output_mutation=True)
def initialize():
    return {
        "lang": PipelineLangDetect(),
        "translate": PipelineNMT(),
        "sentiment": PipelineSentiment(),
    }


def timestamp(time):
    ret = 0
    ret += time.seconds * 1000
    ret += time.microseconds // 1000
    return ret


def SentimentDemo(pipeline, runtime):
    desc = "Sentiment Analyzer focus on predicting text polarity (Negative, Neutral, Positive). Focusing on English Text"
    st.title("Sentiment Analyzer")
    st.write(desc)

    text = st.text_input("Input Text")
    if st.button("Analyze"):
        start = datetime.now()
        sentiment, confidence = pipeline(text)
        stop = datetime.now()
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence}")
        if runtime:
            st.write(f"Time Taken: {timestamp(stop - start)} ms")


def LanguageDemo(pipeline, runtime):
    desc = "Language Detection, for now, it can detect 10 languages"
    st.title("Language Detection")
    st.write(desc)

    text = st.text_input("Input Text")
    if st.button("Analyze"):
        start = datetime.now()
        language = pipeline(text)
        stop = datetime.now()
        st.write(f"Language: {language}")
        if runtime:
            st.write(f"Time Taken: {timestamp(stop - start - start)} ms")


def MachineTranslationDemo(pipeline, runtime):
    desc = "Machine Translation, for now, it can translate Indonesian to English"
    st.title("Machine Translation")
    st.write(desc)

    text = st.text_input("Input Text")
    if st.button("Translate"):
        start = datetime.now()
        translated = pipeline(text)
        stop = datetime.now()
        st.write(f"Translated: {translated}")
        if runtime:
            st.write(f"Time Taken: {timestamp(stop - start)} ms")


def MainDemo(pipelines, runtime):
    desc = "Sentiment Analyzer with language detection and machine translation"
    st.title("Interactive Sentiment Analyzer")
    st.write(desc)

    text = st.text_input("Input Text")
    if st.button("Analyze"):
        start = datetime.now()
        lang, cont = pipelines["lang"](text)
        stop = datetime.now()
        st.write(f"Input Language: {lang}")
        if runtime:
            st.write(f"Time Taken: {timestamp(stop - start)} ms")

        if cont:
            if lang == "ind":
                start = datetime.now()
                text = pipelines["translate"](text)
                stop = datetime.now()
                st.write(f"Translated: {text}")
                if runtime:
                    st.write(f"Time Taken: {timestamp(stop - start)} ms")

            start = datetime.now()
            sentiment, confidence = pipelines["sentiment"](text)
            stop = datetime.now()
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Confidence: {confidence}")
            if runtime:
                st.write(f"Time Taken: {timestamp(stop - start)} ms")


if __name__ == "__main__":
    st.set_page_config(
        page_title="IU Korea",
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    pipelines = initialize()

    st.sidebar.header("IU Korea")
    option = st.sidebar.radio(
        "Select Pipelines",
        ("All", "Language Detection", "Machine Translation", "Sentiment Analysis"),
    )
    runtime = st.sidebar.radio("See Runtime?", ("Yes", "No"))

    runtime_map = {
        "Yes": True,
        "No": False,
    }
    val_runtime = runtime_map[runtime]
    if option == "All":
        MainDemo(pipelines, runtime_map)
    elif option == "Language Detection":
        LanguageDemo(pipelines["lang"], runtime_map)
    elif option == "Machine Translation":
        MachineTranslationDemo(pipelines["translate"], runtime_map)
    elif option == "Sentiment Analysis":
        SentimentDemo(pipelines["sentiment"], runtime_map)
