import onnxruntime as ort
import numpy as np
import pandas as pd

# 1. Carregar o modelo ONNX
modelo_path = "models/DecisionTree.onnx"  # altere para LogisticRegression.onnx ou DecisionTree.onnx se quiser
session = ort.InferenceSession(modelo_path)

# 2. Preparar os dados de entrada (mesmo pré-processamento usado no treino)
df = pd.read_csv("data/dataset.csv")
X = df.drop("target", axis=1)
X = pd.get_dummies(X)

# 3. Selecionar uma linha de teste
sample = X.iloc[0].values.astype(np.float32).reshape(1, -1)

# 4. Rodar a inferência
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

pred = session.run([output_name], {input_name: sample})[0]

print("🔮 Classe prevista:", pred)

# 5. Se o modelo exportou também probabilidades
if len(session.get_outputs()) > 1:
    output_proba = session.get_outputs()[1].name
    proba = session.run([output_proba], {input_name: sample})[0]
    # print("📊 Probabilidades:", proba)

# 6. Opcional: mapear de volta para os rótulos originais
labels = pd.factorize(df["target"])[1]  # pega os nomes originais
print("📝 Classe original correspondente:", labels[pred[0]])
