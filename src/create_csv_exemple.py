import pandas as pd
import random

data = []
for i in range(1, 501):
    idade = random.randint(20, 65)
    renda = random.randint(1500, 15000)
    divida = random.randint(0, 10000)
    score = random.randint(400, 800)
    target = "bom" if score > 650 and divida < renda*0.6 else "ruim"
    data.append([f"Cliente{i}", idade, renda, divida, score, target])

df = pd.DataFrame(data, columns=["cliente","idade","renda","divida","score_credito","target"])
df.to_csv("data/dataset.csv", index=False)
print("✅ Dataset gerado com 500 linhas em data/dataset.csv")
