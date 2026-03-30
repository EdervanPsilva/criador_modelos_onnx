import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. Ler dataset
df = pd.read_csv("data/dataset.csv")

# Separar features e target
X = df.drop("target", axis=1)
y = df["target"]

# 2. Converter colunas de texto em números automaticamente
X = pd.get_dummies(X)

# 3. Converter target em números se for texto
if y.dtype == "object":
    y = pd.factorize(y)[0]  # transforma categorias em 0,1,2...

# 4. Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Definir modelos
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),  # Bom para dados lineares, rápido e interpretável; útil como baseline.
    "RandomForest": RandomForestClassifier(n_estimators=100),  # Robusto, lida bem com dados tabulares e desbalanceados; ótimo para importância de features.
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),  # Simples, interpretável; bom para entender regras, mas tende a overfitting se não regularizado.
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42),  # Captura relações complexas; exige mais dados e tempo de treino.
    "KNN": KNeighborsClassifier(n_neighbors=5),  # Funciona bem em datasets pequenos e com fronteiras claras; não escala bem para grandes volumes.
    "SVM": SVC(kernel="rbf", probability=True),  # Excelente para margens bem definidas; pode ser pesado em datasets grandes.
    "GradientBoosting": GradientBoostingClassifier(),  # Forte em dados tabulares; bom para problemas desbalanceados; costuma ter alta acurácia.
    "AdaBoost": AdaBoostClassifier(),  # Útil para melhorar modelos fracos; funciona bem em dados limpos e sem muito ruído.
    "ExtraTrees": ExtraTreesClassifier(),  # Similar ao RandomForest, mas mais aleatório; rápido e bom para generalização.
    "NaiveBayes": GaussianNB()  # Muito rápido; ótimo para dados textuais ou probabilísticos; pode ser fraco em dados complexos.
}


results = {}

# 6. Treinar e avaliar cada modelo
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    results[name] = (acc, model)

# 7. Mostrar melhor modelo
best_model_name = max(results, key=lambda k: results[k][0])
print(f"\n🏆 Melhor modelo: {best_model_name} (Accuracy: {results[best_model_name][0]:.4f})")

# 8. Solicitar escolha do usuário
print("\nDigite o nome do modelo que deseja salvar:")
choice = input().strip()

if choice not in results:
    print("❌ Escolha inválida. Nenhum modelo será salvo.")
else:
    chosen_model = results[choice][1]
    initial_type = [('input', FloatTensorType([None, X.shape[1]]))]

    try:
        onnx_model = convert_sklearn(chosen_model, initial_types=initial_type)

        # Criar pasta se não existir
        os.makedirs("models", exist_ok=True)

        with open(f"models/{choice}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"✅ Modelo {choice} salvo em models/{choice}.onnx")
    except Exception as e:
        print(f"⚠️ Não foi possível exportar {choice} para ONNX. Erro: {e}")
