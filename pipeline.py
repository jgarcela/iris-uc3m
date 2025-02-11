import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import uuid
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import Counter


def preparar_datos_train(data, target_column_name, contenido_column_name, test_size=0.2, random_state=42):
    """Divide los datos en entrenamiento y prueba."""
    X_train, X_test, y_train, y_test = train_test_split(
        data[contenido_column_name], data[target_column_name], test_size=test_size, random_state=random_state, stratify=data[target_column_name]
    )
    return X_train, X_test, y_train, y_test


def balancear_datos_train(X_train, y_train, strategy="none"):
    """Aplica balanceo de datos si es necesario e imprime los conteos antes y después."""
    
    print("Distribución antes del balanceo:", Counter(y_train))
    
    if strategy == "smote":
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    elif strategy == "downsampling":
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    
    print("Distribución después del balanceo:", Counter(y_train))
    
    return X_train, y_train


def save_results_to_excel(X_test, y_test, y_pred, id2label, vectorizer_name, classifier_name, balance, target_column_name):
    """
    Guarda los resultados de cada ejecución en un archivo Excel sin sobrescribir los datos anteriores.
    """
    # Crear filename
    filename = f"results/results_{target_column_name}.xlsx"

    # Generar un ID único para la ejecución
    execution_id = str(uuid.uuid4())

    # Convertir etiquetas de números a nombres
    y_test_labels = [id2label[y] for y in y_test]
    y_pred_labels = [id2label[y] for y in y_pred]

    # Crear DataFrame con los datos de test y predicciones
    results_df = pd.DataFrame({
        "Execution ID": execution_id,
        "Texto": X_test,
        "Etiqueta Real": y_test_labels,
        "Etiqueta Predicha": y_pred_labels
    })

    # Calcular métricas de clasificación y añadir nombre de la clase
    metrics_dict = classification_report(y_test, y_pred, target_names=id2label.values(), output_dict=True)
    metrics_df = pd.DataFrame(metrics_dict).transpose()  # Convertir dict a DataFrame
    metrics_df.insert(0, "Execution ID", execution_id)  # Agregar Execution ID
    metrics_df.insert(1, "Clase", list(id2label.values()) + ["accuracy", "macro avg", "weighted avg"])  # Agregar nombres de clases

    # Crear DataFrame con la configuración del modelo e ID de ejecución
    config_df = pd.DataFrame({
        "Execution ID": [execution_id],
        "Vectorizer": [vectorizer_name],
        "Classifier": [classifier_name],
        "Balance": [balance]
    })

    # Crear Matriz de Confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=id2label.values(), columns=id2label.values())
    conf_matrix_df.insert(0, "Execution ID", execution_id)  # Agregar Execution ID

    # Guardar en Excel (añadiendo en lugar de sobrescribir)
    if os.path.exists(filename):
        with pd.ExcelWriter(filename, mode='a', if_sheet_exists='overlay') as writer:
            results_df.to_excel(writer, sheet_name="Predicciones", index=False, startrow=writer.sheets["Predicciones"].max_row)  
            metrics_df.to_excel(writer, sheet_name="Métricas", index=False, startrow=writer.sheets["Métricas"].max_row)
            config_df.to_excel(writer, sheet_name="Configuración", index=False, startrow=writer.sheets["Configuración"].max_row)
            conf_matrix_df.to_excel(writer, sheet_name="Matriz de Confusión", index=False, startrow=writer.sheets["Matriz de Confusión"].max_row)
    else:
        with pd.ExcelWriter(filename) as writer:
            results_df.to_excel(writer, sheet_name="Predicciones", index=False)
            metrics_df.to_excel(writer, sheet_name="Métricas", index=False)
            config_df.to_excel(writer, sheet_name="Configuración", index=False)
            conf_matrix_df.to_excel(writer, sheet_name="Matriz de Confusión", index=False)

    print(f"Resultados guardados en {filename} con Execution ID: {execution_id}")

    return execution_id


def save_model_and_vectorizer(model, vectorizer, target_column_name, execution_id):
    """Guarda el modelo y el vectorizador en sus respectivas carpetas."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("vectorizers", exist_ok=True)

    model_path = f"models/{target_column_name}/model_{execution_id}.joblib"
    vectorizer_path = f"vectorizers/{target_column_name}/vectorizer_{execution_id}.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Modelo guardado en {model_path}")
    print(f"Vectorizador guardado en {vectorizer_path}")



# Integrar la función en el flujo de entrenamiento y evaluación
def train_and_evaluate(target_column_name, X_train, X_test, y_train, y_test, id2label, label2id,
                        vectorizer_name="tfidf", classifier_name="logistic", balance="none"):
    """Entrena y evalúa un modelo de clasificación de texto con opción de balanceo y guarda resultados en Excel."""
    
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
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "svm": SVC(class_weight="balanced"),
        "random_forest": RandomForestClassifier(class_weight="balanced")
    }
    model = classifiers.get(classifier_name, LogisticRegression(max_iter=1000))  # Logistic por defecto

    # Entrenar modelo
    model.fit(X_train_vec, y_train)

    # Evaluar modelo
    y_pred = model.predict(X_test_vec)

    # Guardar resultados en Excel sin sobrescribir los datos previos
    execution_id = save_results_to_excel(X_test, y_test, y_pred, id2label, vectorizer_name, classifier_name, balance, target_column_name)

    # Guardar el modelo y el vectorizador
    save_model_and_vectorizer(model, vectorizer, target_column_name, execution_id)

    return model, vectorizer

