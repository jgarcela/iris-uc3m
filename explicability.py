import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer
from collections import Counter, defaultdict
from wordcloud import WordCloud
from gensim.models import Word2Vec
import pipeline
from transformers import PreTrainedTokenizerBase


def create_explainer(class_names):
    return LimeTextExplainer(class_names=class_names)


def predict_proba_lime(texts, vectorizer, model):
    # Verificar si el vectorizer es Word2Vec
    if isinstance(vectorizer, Word2Vec):
        text_vectorized = np.array([pipeline.vectorize_text_word2vec(text, vectorizer) for text in texts])  # Vectorizar manualmente
    elif isinstance(vectorizer, PreTrainedTokenizerBase):
        text_vectorized = np.array([pipeline.vectorize_text_salamandra(text, vectorizer) for text in texts])  # Vectorizar manualmente
    else:
        text_vectorized = vectorizer.transform(texts)  # Usar el método transform en otros casos


    # text_vectorized = vectorizer.transform(texts) # Convertimos el texto a la representación vectorizada
    return model.predict_proba(text_vectorized) # Devolvemos la probabilidad de cada clase


def explain_instance_predicted_class(text, vectorizer, model, explainer, id2label, num_words=10):
    # Verificar si el vectorizer es Word2Vec
    if isinstance(vectorizer, Word2Vec):
        text_vectorized = np.array([pipeline.vectorize_text_word2vec(text, vectorizer)])  # Vectorizar manualmente
    elif isinstance(vectorizer, PreTrainedTokenizerBase):
        text_vectorized = np.array([pipeline.vectorize_text_salamandra(text, vectorizer)])  # Vectorizar manualmente
    else:
        text_vectorized = vectorizer.transform([text])  # Usar el método transform en otros casos
    
    # Predecir la clase para el texto
    predicted_class = model.predict(text_vectorized)[0] # Obtenemos las probabilidades
    print(f"Clase predicha: {id2label[predicted_class]}")

    # Explicar la instancia
    exp = explainer.explain_instance(
        text, 
        lambda x: predict_proba_lime(x, vectorizer, model),
        num_features=num_words,
        labels=[predicted_class]
    )
    
    return exp.show_in_notebook()


def explain_instance_all_classes_heatmap(text, vectorizer, model, explainer, id2label, num_words=5):
    # Verificar si el vectorizer es Word2Vec
    if isinstance(vectorizer, Word2Vec):
        text_vectorized = np.array([pipeline.vectorize_text_word2vec(text, vectorizer)])  # Vectorizar manualmente
    elif isinstance(vectorizer, PreTrainedTokenizerBase):
        text_vectorized = np.array([pipeline.vectorize_text_salamandra(text, vectorizer)])  # Vectorizar manualmente
    else:
        text_vectorized = vectorizer.transform([text])  # Usar el método transform en otros casos
    
    proba = model.predict_proba(text_vectorized)[0] # Obtenemos las probabilidades
    predicted_class = np.argmax(proba) # Clase predicha

    # Generar la explicación con LIME
    exp = explainer.explain_instance(
        text, 
        lambda x: predict_proba_lime(x, vectorizer, model),
        num_features=num_words,
        labels=list(range(len(id2label)))
    )
    
    print(f"Clase predicha: {id2label[predicted_class]}")
    print("Probabilidades de cada clase:")
    for i, prob in enumerate(proba):
        print(f"  {id2label[i]}: {prob:.2%}")

    # Crear una matriz donde cada fila es una palabra y cada columna una clase
    word_importance = {}
    for label in range(len(id2label)):
        exp_list = exp.as_list(label=label) # Explicación de la clase actual
        for word, weight in exp_list:
            if word not in word_importance:
                word_importance[word] = [0] * len(id2label) # Inicializar todas las clases en 0
            word_importance[word][label] = weight # Asignar peso

    # Convertir a matriz para seaborn
    words = list(word_importance.keys())
    importance_matrix = np.array(list(word_importance.values()))

    # Crear el heatmap
    plt.figure(figsize=(12, len(words) * 0.5))
    sns.heatmap(
        importance_matrix, 
        xticklabels=[id2label[i] for i in range(len(id2label))], 
        yticklabels=words, 
        cmap="coolwarm", 
        annot=True, 
        fmt=".2f"
    )
    plt.xlabel("Clases")
    plt.ylabel("Palabras")
    plt.title("Importancia de palabras en todas las clases - Texto seleccionado")
    plt.show()


def explain_top_confidence_texts(X, vectorizer, model, explainer, id2label, top_n=5):
    """
    Explica los textos con mayor confianza de predicción.
    
    Args:
        X (pd.Series): Datos de prueba en formato de texto.
        vectorizer (object): Vectorizador para transformar el texto.
        model (object): Modelo entrenado con predict_proba.
        top_n (int): Número de textos con mayor confianza a explicar.
    """
    # Obtener las predicciones de todo X (vectorizado)
    # Verificar si el vectorizer es Word2Vec
    if isinstance(vectorizer, Word2Vec):
        X_vec = np.array([pipeline.vectorize_text_word2vec(text, vectorizer) for text in X])  # Vectorizar manualmente
    elif isinstance(vectorizer, PreTrainedTokenizerBase):
        X_vec = np.array([pipeline.vectorize_text_salamandra(text, vectorizer) for text in X])  # Vectorizar manualmente
    else:
        X_vec = vectorizer.transform(X)  # Usar el método transform en otros casos

    # X_vec = vectorizer.transform(X)  # Convertimos el texto en su representación numérica
    probs = model.predict_proba(X_vec)  # Obtenemos las probabilidades de cada clase

    # Obtener la confianza más alta por cada texto
    confidence_scores = np.max(probs, axis=1)

    # Ordenar los textos por confianza descendente
    top_indices = np.argsort(confidence_scores)[::-1][:top_n]  # Top textos mejor clasificados

    # Explicar los mejores clasificados
    for i in top_indices:
        print(f"\nExplicando texto {i} con confianza {confidence_scores[i]:.2f}")
        explain_instance_predicted_class(X.iloc[i], vectorizer, model, explainer, id2label)



def explain_classes_global(X, y, vectorizer, model, explainer, id2label, num_samples=10, num_words=10):
    class_word_importance = defaultdict(lambda: defaultdict(float)) # {clase: {palabra: importancia}}
    # Verificar si el vectorizer es Word2Vec
    if isinstance(vectorizer, Word2Vec):
        X_vec = np.array([pipeline.vectorize_text_word2vec(text, vectorizer) for text in X])  # Vectorizar manualmente
    elif isinstance(vectorizer, PreTrainedTokenizerBase):
        X_vec = np.array([pipeline.vectorize_text_salamandra(text, vectorizer) for text in X])  # Vectorizar manualmente
    else:
        X_vec = vectorizer.transform(X)  # Usar el método transform en otros casos
    y_pred = model.predict(X_vec) # Predicciones del modelo

    # Seleccionar muestras representativas para cada clase
    for label in set(y):
        indices = np.where(y_pred == label)[0] # Índices de textos predichos como esta clase
        selected_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False) # Seleccionamos ejemplos
        
        for i in selected_indices:
            exp = explainer.explain_instance(
                X.iloc[i],
                lambda x: predict_proba_lime(x, vectorizer, model),
                num_features=num_words,
                labels=[label]
            )

            # Recopilar importancia de palabras para la clase
            for word, weight in exp.as_list(label=label):
                class_word_importance[label][word] += weight

    # Convertir a matriz para el heatmap
    all_words = list(set(word for words in class_word_importance.values() for word in words))
    importance_matrix = np.zeros((len(all_words), len(id2label)))
    
    for label, word_weights in class_word_importance.items():
        for word, weight in word_weights.items():
            word_idx = all_words.index(word)
            importance_matrix[word_idx, label] = weight

    # Crear el heatmap
    plt.figure(figsize=(12, len(all_words) * 0.5))
    sns.heatmap(
        importance_matrix, 
        xticklabels=[id2label[i] for i in range(len(id2label))], 
        yticklabels=all_words, 
        cmap="coolwarm", 
        annot=True, 
        fmt=".2f"
    )
    plt.xlabel("Clases")
    plt.ylabel("Palabras")
    plt.title("Palabras más relevantes por clase - Explicación Global")
    plt.show()


def explain_classes_summary(X, y, vectorizer, model, explainer, id2label, num_samples=10, num_words=10):
    class_word_importance = {label: Counter() for label in id2label.keys()} # Diccionario para almacenar palabras clave por clase
    # Verificar si el vectorizer es Word2Vec
    if isinstance(vectorizer, Word2Vec):
        X_vec = np.array([pipeline.vectorize_text_word2vec(text, vectorizer) for text in X])  # Vectorizar manualmente
    elif isinstance(vectorizer, PreTrainedTokenizerBase):
        X_vec = np.array([pipeline.vectorize_text_salamandra(text, vectorizer) for text in X])  # Vectorizar manualmente
    else:
        X_vec = vectorizer.transform(X)  # Usar el método transform en otros casos
    y_pred = model.predict(X_vec) # Predicciones del modelo en los datos de entrenamiento

    # Recorrer cada clase y encontrar sus palabras clave
    for label in set(y):
        indices = np.where(y_pred == label)[0] # Índices de textos predichos como esta clase
        selected_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False) # Seleccionar muestras
        
        for i in selected_indices:
            exp = explainer.explain_instance(
                X.iloc[i],
                lambda x: predict_proba_lime(x, vectorizer, model),
                num_features=num_words,
                labels=[label]
            )

            # Contar la frecuencia de las palabras más importantes en cada clase
            for word, weight in exp.as_list(label=label):
                class_word_importance[label][word] += abs(weight) # Sumar el peso absoluto

    # Mostrar un resumen de cada clase con sus palabras más influyentes
    for label, words in class_word_importance.items():
        most_common_words = words.most_common(num_words)
        print(f"\n🔹 Clase: {id2label[label]}")
        print(f"  Palabras clave más influyentes: {', '.join([f'{word} ({weight:.2f})' for word, weight in most_common_words])}")


def generate_wordclouds(X, y, vectorizer, model, explainer, id2label, num_samples=10, num_words=20):
    class_word_importance = {label: Counter() for label in id2label.keys()}
    # Verificar si el vectorizer es Word2Vec
    if isinstance(vectorizer, Word2Vec):
        X_vec = np.array([pipeline.vectorize_text_word2vec(text, vectorizer) for text in X])  # Vectorizar manualmente
    elif isinstance(vectorizer, PreTrainedTokenizerBase):
        X_vec = np.array([pipeline.vectorize_text_salamandra(text, vectorizer) for text in X])  # Vectorizar manualmente
    else:
        X_vec = vectorizer.transform(X)  # Usar el método transform en otros casos
    y_pred = model.predict(X_vec) # Predicciones del modelo en los datos de entrenamiento

    # Recorrer cada clase y encontrar sus palabras clave
    for label in set(y):
        indices = np.where(y_pred == label)[0]
        selected_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
        
        for i in selected_indices:
            exp = explainer.explain_instance(
                X.iloc[i],
                lambda x: predict_proba_lime(x, vectorizer, model),
                num_features=num_words,
                labels=[label]
            )

            # Sumar importancia de palabras
            for word, weight in exp.as_list(label=label):
                class_word_importance[label][word] += abs(weight)

    # Generar nubes de palabras
    for label, words in class_word_importance.items():
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Nube de palabras - {id2label[label]}")
        plt.show()
