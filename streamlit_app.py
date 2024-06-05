import streamlit as st
from transformers import pipeline

# Cargar modelos de NLP
sentiment_analysis = pipeline('sentiment-analysis')
summarization = pipeline('summarization')
text_classification = pipeline('zero-shot-classification')
named_entity_recognition = pipeline('ner', grouped_entities=True)

st.title("Aplicación Avanzada de NLP con Streamlit")

# Sección de Análisis de Sentimientos
st.header("Análisis de Sentimientos")
user_input_sentiment = st.text_area("Ingrese un texto para analizar sentimientos:")
if st.button("Analizar Sentimientos"):
    if user_input_sentiment:
        sentiment_result = sentiment_analysis(user_input_sentiment)
        sentiment = sentiment_result[0]['label']
        score = sentiment_result[0]['score']
        st.write(f"Sentimiento: {sentiment}")
        st.write(f"Confianza: {score:.2f}")
    else:
        st.write("Por favor, ingrese un texto.")

# Sección de Resumen de Textos
st.header("Resumen de Textos")
user_input_summary = st.text_area("Ingrese un texto para resumir:")
if st.button("Resumir Texto"):
    if user_input_summary:
        summary_result = summarization(user_input_summary, max_length=50, min_length=25, do_sample=False)
        summary = summary_result[0]['summary_text']
        st.write("Resumen:")
        st.write(summary)
    else:
        st.write("Por favor, ingrese un texto.")

# Sección de Clasificación de Textos
st.header("Clasificación de Textos")
user_input_classification = st.text_area("Ingrese un texto para clasificar:")
labels = st.text_input("Ingrese categorías separadas por comas:", "deportes, política, tecnología")
if st.button("Clasificar Texto"):
    if user_input_classification:
        labels_list = [label.strip() for label in labels.split(',')]
        classification_result = text_classification(user_input_classification, candidate_labels=labels_list)
        st.write("Clasificación:")
        for i in range(len(classification_result['labels'])):
            st.write(f"Categoría: {classification_result['labels'][i]}, Confianza: {classification_result['scores'][i]:.2f}")
    else:
        st.write("Por favor, ingrese un texto.")

# Sección de Extracción de Entidades Nombradas
st.header("Extracción de Entidades Nombradas")
user_input_ner = st.text_area("Ingrese un texto para extraer entidades nombradas:")
if st.button("Extraer Entidades"):
    if user_input_ner:
        ner_results = named_entity_recognition(user_input_ner)
        st.write("Entidades Nombradas:")
        for entity in ner_results:
            st.write(f"Entidad: {entity['word']}, Tipo: {entity['entity_group']}, Confianza: {entity['score']:.2f}")
    else:
        st.write("Por favor, ingrese un texto.")

# Ejecución de la Aplicación
if __name__ == '__main__':
    st.run()
