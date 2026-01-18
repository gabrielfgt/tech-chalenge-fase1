# Tech Challenge Fase 1 - IA em Saúde

Solução com foco em Inteligência Artificial para processamento de exames médicos e documentos clínicos, utilizando Machine Learning e Visão Computacional para apoiar a tomada de decisão médica.

## Sobre o Projeto

Este projeto é parte do Tech Challenge da FIAP e apresenta duas soluções principais de IA aplicadas à área da saúde:

1. **Triagem Preventiva de AVC (Acidente Vascular Cerebral)** - Utiliza Machine Learning para identificar padrões de risco em pacientes que podem desenvolver AVC, criando uma base para um sistema de apoio à decisão médica (CDSS - Clinical Decision Support System).

2. **Classificação de Pneumonia em Raio-X** - Utiliza Deep Learning com Redes Neurais Convolucionais (CNN) para classificar automaticamente imagens de raio-X de tórax em categorias NORMAL ou PNEUMONIA.

## Objetivos

- Desenvolver modelos de Machine Learning para predição de risco de AVC
- Criar uma CNN para classificação de imagens médicas (raio-X)
- Avaliar a performance dos modelos usando métricas apropriadas para problemas médicos
- Criar soluções que possam servir como base para sistemas de apoio à decisão médica

## Estrutura do Projeto

```
tech-chalenge-fase1/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências do projeto
├── main.ipynb                         # Notebook: Triagem Preventiva de AVC
├── cv.ipynb                           # Notebook: Classificação de Pneumonia
└── data/
    └── healthcare-dataset-stroke-data.csv  # Dataset de casos clínicos
```

## Tecnologias Utilizadas

- **Python 3.11+**
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Deep Learning**: TensorFlow, Keras
- **Visão Computacional**: OpenCV, Pillow
- **Análise de Dados**: pandas, numpy, matplotlib, seaborn
- **Explicabilidade**: SHAP
- **Dataset**: Kaggle Hub

## Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)
- Jupyter Notebook ou JupyterLab (recomendado)

## Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositório>
cd tech-chalenge-fase1
```

### 2. Crie um ambiente virtual (recomendado)

```bash
# No macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# No Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Instale o Jupyter (se ainda não tiver)

```bash
pip install jupyter jupyterlab
```

## Como Executar

### Executando os Notebooks

1. **Inicie o Jupyter Notebook ou JupyterLab:**

```bash
jupyter notebook
# ou
jupyter lab
```

2. **Abra o notebook desejado:**
   - `main.ipynb` - Para análise de triagem preventiva de AVC
   - `cv.ipynb` - Para classificação de pneumonia em raio-X

3. **Execute as células sequencialmente:**
   - Os notebooks estão organizados em seções lógicas
   - Execute as células na ordem apresentada
   - Algumas células podem demorar para executar (especialmente o treinamento de modelos)

### Notebook: Triagem Preventiva de AVC (`main.ipynb`)

Este notebook contém:
- Análise exploratória de dados (EDA)
- Pré-processamento e limpeza de dados
- Treinamento de múltiplos modelos de classificação:
  - Random Forest
  - Logistic Regression
  - SVM
  - Decision Tree
  - K-Nearest Neighbors
- Avaliação de performance dos modelos
- Análise de importância de features

**Dataset:** O dataset `healthcare-dataset-stroke-data.csv` contém 5.110 casos clínicos com informações sobre pacientes e histórico de AVC.

### Notebook: Classificação de Pneumonia (`cv.ipynb`)

Este notebook contém:
- Download automático do dataset de raio-X do Kaggle
- Pré-processamento de imagens
- Construção e treinamento de CNN
- Uso de transfer learning (VGG16, ResNet50, MobileNetV2)
- Avaliação com métricas médicas apropriadas
- Análise de resultados e visualizações

**Dataset:** O dataset de raio-X é baixado automaticamente do Kaggle durante a execução do notebook.

## Datasets

### Dataset de AVC
- **Arquivo:** `data/healthcare-dataset-stroke-data.csv`
- **Descrição:** Contém 5.110 casos clínicos com informações demográficas, histórico médico e diagnóstico de AVC
- **Variáveis:** idade, gênero, hipertensão, doença cardíaca, estado civil, tipo de trabalho, tipo de residência, nível de glicose, IMC, status de fumante, e diagnóstico de AVC

### Dataset de Pneumonia
- **Fonte:** Kaggle (baixado automaticamente)
- **Descrição:** Imagens de raio-X de tórax classificadas como NORMAL ou PNEUMONIA
- **Download:** O dataset é baixado automaticamente ao executar o notebook `cv.ipynb`

## Observações Importantes

- **Tempo de execução:** O treinamento de modelos, especialmente as CNNs, pode levar bastante tempo dependendo do hardware disponível
- **GPU:** O projeto funciona em CPU, mas ter uma GPU acelera significativamente o treinamento das CNNs
- **Memória:** Certifique-se de ter memória RAM suficiente (recomendado: 8GB+)
- **Kaggle API:** Para o notebook `cv.ipynb`, pode ser necessário configurar as credenciais do Kaggle (o kagglehub tenta fazer isso automaticamente)

## Notas

- Este projeto é educacional e não deve ser usado para diagnóstico médico real
- Os modelos apresentados são protótipos e requerem validação clínica adequada antes de uso em produção
- Sempre consulte profissionais médicos para diagnósticos reais

## Autor
Gabriel Francico Gonsalves Teixeira - RM368786

Projeto desenvolvido para o Tech Challenge FIAP - Fase 1

## Licença

Este projeto é um desafio acadêmico, para a especialização em Inteligência Artificial - FIAP
