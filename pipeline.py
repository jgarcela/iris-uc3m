import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def preparar_datos_train(data, test_size=0.2, random_state=42):
    """Divide los datos en entrenamiento y prueba."""
    X_train, X_test, y_train, y_test = train_test_split(
        data['contenido_clean'], data['Tema'], test_size=test_size, random_state=random_state, stratify=data['Tema']
    )
    return X_train, X_test, y_train, y_test

def balancear_datos_train(X_train, y_train, strategy="none"):
    """Aplica balanceo de datos si es necesario."""
    if strategy == "smote":
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)  # No es necesario convertir a denso
    elif strategy == "undersampling":
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    return X_train, y_train


def train_and_evaluate(X_train, X_test, y_train, y_test, id2label, label2id,
                        vectorizer_name="tfidf", classifier_name="logistic", balance="none"):
    """Entrena y evalúa un modelo de clasificación de texto con opción de balanceo."""
    
    # Seleccionar vectorizador
    vectorizers = {
        "tfidf": TfidfVectorizer(max_features=5000),
        "count": CountVectorizer(max_features=5000)
    }
    vectorizer = vectorizers.get(vectorizer_name, TfidfVectorizer(max_features=5000))
    
    # Vectorizar datos
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Aplicar balanceo si es necesario
    X_train_vec, y_train = balancear_datos_train(X_train_vec, y_train, balance)

    # Seleccionar modelo
    classifiers = {
        "logistic": LogisticRegression(max_iter=1000),
        "svm": SVC(),
        "random_forest": RandomForestClassifier()
    }
    model = classifiers.get(classifier_name, "logistic") # logistic es el default

    # Entrenar modelo
    model.fit(X_train_vec, y_train)

    # Evaluar modelo
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, target_names=id2label.values()))

    return model, vectorizer

