# Regras para construção do dataset.csv

Este documento descreve as regras e boas práticas para montar o arquivo `dataset.csv` usado no treinamento dos modelos.

---

## 1. Estrutura básica

- O arquivo deve estar em formato **CSV** (valores separados por vírgula).
- Deve conter uma coluna chamada **`target`**, que representa a variável de saída (classe).
- As demais colunas são **features** (variáveis de entrada).
- O cabeçalho (primeira linha) deve conter os nomes das colunas.

---

## 2. Tipos de dados permitidos

- **Numéricos**: inteiros ou decimais (ex.: idade, renda, avaliação).
- **Categóricos**: texto curto (ex.: região, produto, status).  
  - Esses serão automaticamente convertidos em variáveis numéricas via `pd.get_dummies`.
- **Target**:
  - Pode ser texto (`bom/mau`, `alta/baixa`, `positivo/negativo`) ou número (`0/1`).
  - Se for texto, será convertido automaticamente em números (`factorize`).

---

## 3. Boas práticas

- Evite colunas irrelevantes (ex.: nomes de clientes) como feature.  
- Mantenha o dataset **balanceado** (quantidade semelhante de exemplos para cada classe).  
- Use pelo menos **50 linhas** para testes básicos.  
- Para redes neurais, recomenda-se **100+ linhas** para evitar overfitting.  

---

## 4. Exemplos

### Exemplo A: Risco de Crédito
```csv
cliente,idade,renda,divida,score_credito,target
João,25,2000,500,650,bom
Maria,40,5000,2000,720,bom
Carlos,35,3000,1500,580,mau
Ana,50,7000,5000,600,mau


### Exemplo B: Performance de Funcionários
```csv
funcionario,tempo_empresa,treinamentos,avaliacao,target
João,2,3,8,alta
Maria,5,10,9,alta
Carlos,1,1,5,baixa
Ana,3,2,6,baixa


### Exemplo C: Fluxo de Caixa
```csv
mes,receita,despesa,investimentos,target
Jan,100000,80000,5000,positivo
Fev,95000,100000,7000,negativo
Mar,120000,90000,10000,positivo
Abr,85000,95000,3000,negativo


## 5. Onde salvar

data/dataset.csv
