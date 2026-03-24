# Conversor de Modelos para ONNX 🚀

Este projeto é um pipeline completo de ciência de dados para treinamento de modelos de classificação e exportação para o formato **ONNX (Open Neural Network Exchange)**. Ele permite gerar dados sintéticos, treinar múltiplos algoritmos de Machine Learning e salvar o melhor modelo para uso em produção.

## 📋 Estrutura do Projeto

```text
C:\Users\c1010568\Documents\conversor_modelos\
├── requirements.txt         # Dependências do projeto
├── data/                    # Pasta para armazenamento dos datasets
│   └── dataset.csv          # Dataset gerado (após execução do script)
├── models/                  # Pasta para armazenamento dos modelos exportados
│   └── DecisionTree.onnx    # Exemplo de modelo exportado
└── src/                     # Código fonte
    ├── create_csv_exemple.py # Gerador de dados sintéticos
    └── train.py             # Script principal de treino e conversão
```

---

## 🛠️ Pré-requisitos

Certifique-se de ter o Python instalado (recomendado 3.8+). Instale as dependências necessárias executando:

```bash
pip install -r requirements.txt
```

As principais bibliotecas utilizadas são:
- **Pandas/Numpy**: Manipulação de dados.
- **Scikit-learn**: Algoritmos de Machine Learning.
- **skl2onnx**: Conversão de modelos Scikit-learn para ONNX.
- **ONNX Runtime**: Execução eficiente de modelos ONNX.

---

## 🚀 Como Utilizar (Passo a Passo)

### Passo 1: Gerar o Dataset
Antes de treinar, você precisa de dados. O script `create_csv_exemple.py` cria um cenário fictício de análise de crédito.

```bash
python src/create_csv_exemple.py
```
*O que ele faz:* Gera 500 linhas com informações de idade, renda, dívida e score de crédito, salvando em `data/dataset.csv`.

### Passo 2: Treinar e Avaliar Modelos
O script `train.py` é o coração do projeto. Ele realiza o pré-processamento automático e testa 4 algoritmos diferentes:

```bash
python src/train.py
```

**O pipeline de execução inclui:**
1. **Leitura e Limpeza:** Carrega o CSV e converte colunas de texto (como nomes de clientes) em formatos numéricos via One-Hot Encoding.
2. **Divisão de Dados:** Separa 80% para treino e 20% para teste.
3. **Competição de Modelos:** Treina simultaneamente:
   - Regressão Logística
   - Random Forest
   - Decision Tree
   - Neural Network (MLP)
4. **Relatório de Performance:** Exibe a acurácia e o relatório de classificação (Precision, Recall, F1-Score) de cada um.

### Passo 3: Escolher e Exportar para ONNX
Ao final do treinamento, o script indicará qual foi o **vencedor (Melhor Modelo)** baseado na acurácia.
- O sistema solicitará que você digite o nome do modelo que deseja salvar.
- Após a escolha, o modelo é convertido para o formato `.onnx` e salvo na pasta `models/`.

---

## 🧠 Por que usar ONNX?

O formato **ONNX** permite que você treine seu modelo em Python e o execute em quase qualquer linguagem ou plataforma (C#, Java, C++, Mobile, Web) com alta performance, sem depender do ambiente original de treinamento.

## 📝 Notas de Implementação
- **Pré-processamento:** O script utiliza `pd.get_dummies` para lidar com variáveis categóricas.
- **Flexibilidade:** Você pode adicionar novos modelos no dicionário `models` dentro de `src/train.py` para expandir os testes.
