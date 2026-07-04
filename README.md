# Quora Question Pairs: Siamese BiLSTM e Features Manuais

Projeto de Processamento de Linguagem Natural para identificar pares de perguntas duplicadas do Quora. A solução principal está no notebook [`quora_question_pairs_siamese_bilstm.ipynb`](quora_question_pairs_siamese_bilstm.ipynb), que compara dois experimentos:

- **Modelo A:** Siamese BiLSTM com pesos compartilhados.
- **Modelo B:** Siamese BiLSTM combinada com features manuais de similaridade textual e TF-IDF.

A métrica oficial do problema no Kaggle é **Log Loss**, mas o notebook também calcula Accuracy, Precision, Recall, F1 e ROC AUC para uma análise mais completa.

## Objetivo

Dado um par de perguntas, prever se elas têm o mesmo significado semântico. O problema é tratado como classificação binária:

- `1` indica perguntas duplicadas.
- `0` indica perguntas não duplicadas.

## Estrutura do projeto

- [`quora_question_pairs_siamese_bilstm.ipynb`](quora_question_pairs_siamese_bilstm.ipynb): notebook principal com o pipeline completo.
- [`datasets/train.csv`](datasets/train.csv): dados rotulados de treino.
- [`datasets/test.csv`](datasets/test.csv): dados de teste sem rótulo.
- [`datasets/sample_submission.csv`](datasets/sample_submission.csv): exemplo do formato de submissão.
- [`requirements.txt`](requirements.txt): dependências do ambiente.
- [`submission.csv`](submission.csv): arquivo de saída gerado pelo notebook.

## Metodologia

O notebook segue as etapas abaixo:

1. leitura dos dados;
2. análise exploratória;
3. limpeza textual;
4. tokenização e padding;
5. divisão treino/validação;
6. extração de features manuais;
7. treinamento do Modelo A;
8. treinamento do Modelo B;
9. avaliação das métricas;
10. geração da submissão final.

### Features manuais usadas no Modelo B

- diferença no número de palavras;
- Jaccard Similarity;
- Word Overlap;
- mesma primeira palavra;
- mesma última palavra;
- quantidade de números compartilhados;
- similaridade cosseno via TF-IDF.

## Requisitos

O ambiente esperado é Python com os pacotes listados em [`requirements.txt`](requirements.txt):

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- ipykernel
- jinja2

## Como executar

1. Crie e ative um ambiente virtual Python.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Abra o notebook [`quora_question_pairs_siamese_bilstm.ipynb`](quora_question_pairs_siamese_bilstm.ipynb).
4. Execute as células na ordem apresentada.
5. Ao final, os arquivos [`submission_A.csv`](submission_A.csv) e [`submission_B.csv`](submission_B.csv) serão gerados na raiz do projeto.

## Configuração do notebook

O notebook centraliza os principais hiperparâmetros em um dicionário `CONFIG`, incluindo:

- tamanho da amostra de treino;
- tamanho da validação;
- tamanho máximo do vocabulário;
- comprimento máximo das sequências;
- dimensão dos embeddings;
- número de unidades da LSTM;
- batch size;
- número de épocas;
- taxa de aprendizado.

Para acelerar experimentos locais, a amostra de treino pode ser reduzida por meio de `TRAIN_SAMPLE_SIZE`. Para usar todo o conjunto de treino, basta definir esse valor como `None`.

## Saída esperada

Ao final da execução, o notebook salva uma submissão com as colunas:

- `test_id`
- `is_duplicate`

Essa é a estrutura esperada pelo Kaggle para a competição Quora Question Pairs.

## Observações

- O notebook foi organizado para comparação direta entre o baseline e a versão com features manuais.
- O Modelo B tende a ser mais informativo quando sinais lexicais simples ajudam a distinguir duplicidade.
- Melhorias futuras possíveis incluem embeddings pré-treinados, ajuste de limiar, validação cruzada, Transformers como BERT e calibração probabilística.
