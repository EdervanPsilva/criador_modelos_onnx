import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
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
print("\nDigite o nome do modelo que deseja salvar (LogisticRegression, RandomForest, DecisionTree, NeuralNetwork):")
choice = input().strip()

if choice not in results:
    print("❌ Escolha inválida. Nenhum modelo será salvo.")
else:
    chosen_model = results[choice][1]
    initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(chosen_model, initial_types=initial_type)

    with open(f"models/{choice}.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ Modelo {choice} salvo em models/{choice}.onnx")
