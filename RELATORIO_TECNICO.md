# Relatório Técnico - Tech Challenge Fase 1
## Inteligência Artificial em Saúde

---

## 1. Estratégias de Pré-processamento

### 1.1. Dataset de Triagem Preventiva de AVC

#### 1.1.1. Limpeza e Tratamento de Dados Faltantes

O dataset inicial continha **5.110 casos clínicos** com informações sobre pacientes e histórico de AVC. Durante a análise exploratória, foi identificado que a variável **IMC (BMI)** apresentava valores faltantes.

**Estratégia de Imputação:**
- Utilização da **média** do IMC para preencher os valores ausentes
- Justificativa: A média preserva a distribuição estatística da variável e evita introduzir viés significativo no dataset

#### 1.1.2. Codificação de Variáveis Categóricas

**Features Categóricas Identificadas:**
- `gender` (gênero)
- `ever_married` (estado civil)
- `work_type` (tipo de trabalho)
- `Residence_type` (tipo de residência)
- `smoking_status` (status de fumante)

**Técnica Aplicada:**
- **One-Hot Encoding** para transformar variáveis categóricas em variáveis binárias
- Resultado: Expansão de 5 features categóricas para 11 features binárias, mantendo a informação sem criar hierarquias artificiais

#### 1.1.3. Normalização de Variáveis Numéricas

**Features Numéricas:**
- `age` (idade)
- `hypertension` (hipertensão)
- `heart_disease` (doença cardíaca)
- `avg_glucose_level` (nível médio de glicose)
- `bmi` (índice de massa corporal)

**Técnica Aplicada:**
- **StandardScaler** (normalização Z-score): Transformação para média 0 e desvio padrão 1
- Fórmula: `z = (x - μ) / σ`
- Justificativa: Garante que todas as features numéricas tenham a mesma escala, essencial para algoritmos sensíveis à magnitude (SVM, KNN, Regressão Logística)

#### 1.1.4. Tratamento de Desbalanceamento de Classes

**Problema Identificado:**
- Distribuição altamente desbalanceada: aproximadamente **4.9%** de casos positivos (AVC) vs **95.1%** de casos negativos
- Risco: Modelos tendem a classificar tudo como classe majoritária, ignorando casos de AVC

**Estratégia Implementada:**
- **SMOTE (Synthetic Minority Oversampling Technique)**
  - Geração sintética de exemplos da classe minoritária
  - Antes: 3.403 casos negativos vs 175 casos positivos
  - Depois: 3.403 casos negativos vs 3.403 casos positivos (balanceamento perfeito)
  - Resultado: Dataset de treino expandido de 3.578 para 6.806 amostras

**Justificativa:**
- SMOTE cria exemplos sintéticos baseados em interpolação entre exemplos reais da classe minoritária
- Preserva a distribuição original dos dados enquanto aumenta a representatividade da classe rara
- Mais eficaz que simples oversampling (duplicação) ou undersampling (perda de informação)

#### 1.1.5. Separação dos Dados

**Estratégia de Divisão:**
- **Train (70%)**: 3.578 amostras → 3.806 após SMOTE
- **Validation (15%)**: 765 amostras
- **Test (15%)**: 767 amostras

**Técnica:**
- **Stratified Split**: Mantém a proporção de classes em cada conjunto
- Garante que cada partição tenha aproximadamente 4.9% de casos positivos
- Evita vieses na avaliação do modelo

### 1.2. Dataset de Classificação de Pneumonia em Raio-X

#### 1.2.1. Pré-processamento de Imagens

**Transformações Aplicadas:**
- **Redimensionamento**: Todas as imagens redimensionadas para **224x224 pixels**
  - Padrão para modelos de transfer learning (VGG16, ResNet50, MobileNetV2)
  - Reduz complexidade computacional mantendo informações relevantes

- **Normalização de Pixels**: Valores normalizados para o intervalo **[0, 1]**
  - Divisão por 255 (valores originais em escala 0-255)
  - Facilita convergência durante o treinamento de redes neurais

- **Conversão para Escala de Cinza**: Modelo CNN customizado utiliza imagens monocromáticas
  - Reduz dimensionalidade (1 canal vs 3 canais RGB)
  - Mantém informações essenciais para diagnóstico de pneumonia

#### 1.2.2. Data Augmentation

**Técnica Aplicada:**
Data augmentation aplicada **apenas no conjunto de treino** para aumentar a variabilidade dos dados e prevenir overfitting.

**Transformações Implementadas:**
1. **Rotação**: ±20 graus aleatórios
2. **Deslocamento Horizontal/Vertical**: 10% da largura/altura
3. **Cisalhamento (Shear)**: 20% de transformação
4. **Zoom**: 20% de ampliação/redução aleatória
5. **Flip Horizontal**: Espelhamento horizontal
6. **Variação de Brilho**: Intervalo [0.8, 1.2]

**Justificativa:**
- Simula variações naturais em imagens médicas (posicionamento do paciente, condições de captura)
- Aumenta efetivamente o tamanho do dataset sem necessidade de coletar mais dados
- Reduz overfitting ao expor o modelo a mais variações durante o treinamento
- **Não aplicado em validação/teste**: Preserva dados originais para avaliação realista

#### 1.2.3. Estrutura do Dataset

**Distribuição:**
- **Treino**: 5.216 imagens (2 classes)
- **Validação**: 16 imagens (2 classes)
- **Teste**: 624 imagens (2 classes)

**Classes:**
- `NORMAL`: 0
- `PNEUMONIA`: 1

---

## 2. Modelos Usados e Por Quê

### 2.1. Modelos de Machine Learning para Predição de AVC

Foram testados **5 algoritmos diferentes** com otimização de hiperparâmetros via **GridSearchCV**:

#### 2.1.1. Logistic Regression (Regressão Logística)

**Por que foi escolhido:**
- **Interpretabilidade**: Coeficientes fornecem insights sobre importância de cada feature
- **Eficiência computacional**: Treinamento rápido mesmo com grandes volumes de dados
- **Boa performance em problemas binários**: Adequado para classificação de risco de AVC
- **Probabilidades calibradas**: Outputs probabilísticos úteis para sistemas de apoio à decisão médica

**Hiperparâmetros Otimizados:**
- `C`: [0.1, 1, 10, 100] - Regularização L2
- `class_weight`: [None, 'balanced'] - Balanceamento de classes

**Melhor Configuração:**
- `C = 0.1`, `class_weight = None`

#### 2.1.2. Random Forest

**Por que foi escolhido:**
- **Robustez**: Menos sensível a overfitting que árvores individuais
- **Importância de Features**: Identifica variáveis mais relevantes para predição
- **Não-linearidade**: Captura relações complexas entre features
- **Performance**: Geralmente apresenta boa acurácia em problemas médicos

**Hiperparâmetros Otimizados:**
- `n_estimators`: [50, 100, 200] - Número de árvores
- `max_depth`: [5, 10, 15, None] - Profundidade máxima
- `min_samples_split`: [2, 5, 10] - Mínimo de amostras para divisão
- `min_samples_leaf`: [1, 2, 4] - Mínimo de amostras por folha
- `class_weight`: [None, 'balanced']

**Melhor Configuração:**
- `n_estimators = 100`, `max_depth = None`, `min_samples_split = 2`, `min_samples_leaf = 1`, `class_weight = 'balanced'`

#### 2.1.3. Support Vector Machine (SVM)

**Por que foi escolhido:**
- **Margem de separação**: Maximiza a distância entre classes
- **Eficácia em espaços de alta dimensionalidade**: Dataset possui 16 features após encoding
- **Kernels não-lineares**: Captura padrões complexos (kernel RBF)

**Hiperparâmetros Otimizados:**
- `C`: [0.1, 1, 10] - Penalidade de erro
- `kernel`: ['linear', 'rbf'] - Tipo de kernel
- `gamma`: ['scale', 'auto'] - Coeficiente do kernel RBF
- `class_weight`: [None, 'balanced']

**Melhor Configuração:**
- `C = 10`, `kernel = 'rbf'`, `gamma = 'scale'`, `class_weight = None`

#### 2.1.4. Decision Tree

**Por que foi escolhido:**
- **Interpretabilidade visual**: Árvores de decisão são facilmente compreensíveis
- **Baseline**: Serve como referência para modelos mais complexos
- **Não requer normalização**: Menos sensível a escala de features

**Hiperparâmetros Otimizados:**
- `max_depth`: [3, 5, 7, 10, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `class_weight`: [None, 'balanced']

**Melhor Configuração:**
- `max_depth = None`, `min_samples_split = 2`, `min_samples_leaf = 1`, `class_weight = 'balanced'`

#### 2.1.5. K-Nearest Neighbors (KNN)

**Por que foi escolhido:**
- **Simplicidade**: Algoritmo baseado em instâncias, sem treinamento explícito
- **Não-paramétrico**: Não assume distribuição específica dos dados
- **Eficaz para problemas locais**: Classifica baseado em vizinhança

**Hiperparâmetros Otimizados:**
- `n_neighbors`: [3, 5, 7, 9, 11] - Número de vizinhos
- `weights`: ['uniform', 'distance'] - Peso dos vizinhos
- `metric`: ['euclidean', 'manhattan'] - Distância utilizada

**Melhor Configuração:**
- `n_neighbors = 3`, `weights = 'distance'`, `metric = 'manhattan'`

#### 2.1.6. Estratégia de Otimização

**GridSearchCV com Validação Cruzada:**
- **5-fold Stratified Cross-Validation**: Mantém proporção de classes em cada fold
- **Métrica Principal**: F1-Score (balanceia precision e recall)
- **Métricas Secundárias**: Recall, Precision, ROC-AUC
- **Justificativa**: Em medicina, **recall é crítico** - não podemos perder casos reais de AVC

### 2.2. Modelos de Deep Learning para Classificação de Pneumonia

#### 2.2.1. CNN Customizada (Do Zero)

**Arquitetura Implementada:**
```
Input (224x224x1) → Conv2D → MaxPooling → Conv2D → MaxPooling → 
Conv2D → MaxPooling → Flatten → Dense → Dropout → Dense (sigmoid)
```

**Por que foi escolhida:**
- **Otimizada para imagens em escala de cinza**: Reduz parâmetros e complexidade
- **Arquitetura simples**: Treinamento mais rápido que modelos pré-treinados
- **Adequada para dataset médio**: 5.216 imagens de treino suficientes para treinar do zero
- **Controle total**: Permite ajuste fino da arquitetura para o problema específico

**Características:**
- **Camadas Convolucionais**: Extração de features hierárquicas
- **MaxPooling**: Redução de dimensionalidade e invariância a translações
- **Dropout**: Regularização para prevenir overfitting
- **Ativação Sigmoid**: Output probabilístico para classificação binária

#### 2.2.2. Transfer Learning (Preparado, não utilizado no modelo final)

**Modelos Pré-treinados Considerados:**
- **VGG16**: Arquitetura profunda com 16 camadas
- **ResNet50**: Arquitetura residual com 50 camadas
- **MobileNetV2**: Arquitetura leve e eficiente

**Por que Transfer Learning seria útil:**
- **Features pré-aprendidas**: Modelos treinados no ImageNet já aprenderam padrões visuais gerais
- **Eficiência**: Requer menos dados e tempo de treinamento
- **Performance**: Geralmente supera modelos treinados do zero com datasets pequenos

**Decisão:**
- CNN customizada foi escolhida por ser otimizada para imagens monocromáticas
- Modelos pré-treinados esperam imagens RGB (3 canais)

#### 2.2.3. Configuração de Treinamento

**Otimizador:**
- **Adam**: Adaptativo, eficiente para CNNs

**Função de Perda:**
- **Binary Crossentropy**: Adequada para classificação binária

**Métricas Monitoradas:**
- Accuracy, Precision, Recall, AUC-ROC

**Callbacks:**
- **Early Stopping**: Interrompe treinamento se não houver melhoria
- **Learning Rate Reduction**: Reduz taxa de aprendizado em platôs

---

## 3. Resultados e Interpretação dos Dados

### 3.1. Resultados - Predição de AVC

#### 3.1.1. Performance no Conjunto de Validação

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.7582 | 0.1022 | **0.5135** | **0.1704** | 0.7421 |
| KNN | 0.8314 | 0.0818 | 0.2432 | 0.1224 | 0.5601 |
| SVM | 0.8222 | 0.0696 | 0.2162 | 0.1053 | 0.7020 |
| Random Forest | 0.8863 | 0.0690 | 0.1081 | 0.0842 | 0.7175 |
| Decision Tree | 0.8523 | 0.0581 | 0.1351 | 0.0813 | 0.5119 |

#### 3.1.2. Performance no Conjunto de Teste (Dados Nunca Vistos)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.7953 | 0.1591 | **0.7568** | **0.2629** | **0.8136** |
| Random Forest | 0.8905 | 0.1594 | 0.2973 | 0.2075 | 0.7502 |
| Decision Tree | 0.8696 | 0.1461 | 0.3514 | 0.2063 | 0.6236 |
| SVM | 0.8279 | 0.1008 | 0.3243 | 0.1538 | 0.7363 |
| KNN | 0.8357 | 0.0680 | 0.1892 | 0.1000 | 0.6245 |

#### 3.1.3. Modelo Selecionado: Logistic Regression

**Justificativa da Seleção:**
1. **Melhor Recall (75.68%)**: Crítico em medicina - detecta 3 de cada 4 casos reais de AVC
2. **Melhor F1-Score (0.2629)**: Melhor equilíbrio entre precision e recall
3. **Melhor ROC-AUC (0.8136)**: Boa capacidade discriminativa geral
4. **Interpretabilidade**: Coeficientes permitem explicar decisões (importante para CDSS)

**Métricas Finais no Teste:**
- **Accuracy**: 79.53%
- **Precision**: 15.91% (baixa, mas esperada em classes desbalanceadas)
- **Recall**: 75.68% (prioridade: não perder casos reais)
- **F1-Score**: 26.29%
- **ROC-AUC**: 81.36%

#### 3.1.4. Interpretação dos Resultados

**Desafios Identificados:**
1. **Desbalanceamento Severo**: Mesmo com SMOTE, o modelo enfrenta dificuldades
   - Apenas ~5% de casos positivos no dataset original
   - Precision baixa (15.91%) indica muitos falsos positivos
   - Trade-off necessário: Priorizar recall sobre precision

2. **Performance em Contexto Médico:**
   - **Recall alto (75.68%)**: O modelo identifica corretamente a maioria dos casos de AVC
   - **Precision baixa (15.91%)**: Muitos pacientes serão classificados como risco, mas não terão AVC
   - **Interpretação**: Em triagem preventiva, é preferível "sobre-alertar" do que perder casos reais

3. **ROC-AUC de 0.8136**: 
   - Indica boa capacidade discriminativa
   - Modelo consegue distinguir bem entre pacientes de risco e não-risco
   - Acima de 0.80 é considerado bom desempenho

4. **Aplicação Prática:**
   - Modelo adequado para **triagem inicial** e **priorização de pacientes**
   - Não deve ser usado como diagnóstico definitivo
   - Pacientes com score alto devem ser encaminhados para avaliação médica detalhada

### 3.2. Resultados - Classificação de Pneumonia

#### 3.2.1. Performance no Conjunto de Teste

**Métricas Obtidas:**
- **Accuracy**: 43.59%
- **Precision**: 91.89%
- **Recall**: 9.09%
- **F1-Score**: 16.55%
- **AUC-ROC**: ~0.60

#### 3.2.2. Interpretação dos Resultados

**Análise das Métricas:**

1. **Precision Alta (91.89%)**:
   - Quando o modelo classifica como PNEUMONIA, está correto em 91.89% dos casos
   - Baixa taxa de falsos positivos
   - **Interpretação**: Modelo é conservador - só classifica como pneumonia quando muito confiante

2. **Recall Baixo (9.09%)**:
   - Modelo detecta apenas 9.09% dos casos reais de pneumonia
   - **Problema crítico**: 90.91% dos casos de pneumonia não são detectados
   - **Risco médico**: Muitos casos reais de pneumonia seriam perdidos

3. **Accuracy Baixa (43.59%)**:
   - Performance geral abaixo do acaso (50%)
   - Indica que o modelo não está generalizando bem

4. **F1-Score Baixo (16.55%)**:
   - Reflete o desequilíbrio entre precision e recall
   - Modelo não está balanceado adequadamente

#### 3.2.3. Problemas Identificados

1. **Overfitting ou Underfitting**:
   - Modelo pode estar muito simples ou muito complexo
   - Necessário ajuste de arquitetura ou hiperparâmetros

2. **Dataset de Validação Pequeno**:
   - Apenas 16 imagens de validação
   - Insuficiente para monitoramento adequado durante treinamento
   - Pode levar a ajustes incorretos de hiperparâmetros

3. **Desequilíbrio de Classes**:
   - Possível desbalanceamento entre NORMAL e PNEUMONIA
   - Necessário verificar distribuição e aplicar técnicas de balanceamento

4. **Necessidade de Melhorias**:
   - Aumentar dataset de validação
   - Ajustar threshold de classificação (priorizar recall)
   - Considerar transfer learning com modelos pré-treinados
   - Implementar class weights para balancear classes

#### 3.2.4. Recomendações para Melhoria

1. **Ajuste de Threshold**:
   - Reduzir threshold de classificação para aumentar recall
   - Aceitar trade-off: menor precision, maior recall

2. **Transfer Learning**:
   - Utilizar VGG16, ResNet50 ou MobileNetV2 pré-treinados
   - Converter imagens para RGB ou usar modelos adaptados para escala de cinza

3. **Balanceamento de Classes**:
   - Aplicar class weights no modelo
   - Usar técnicas de data augmentation específicas para classe minoritária

4. **Aumento do Dataset**:
   - Coletar mais imagens de validação
   - Usar técnicas de data augmentation mais agressivas

### 3.3. Comparação Geral dos Projetos

| Aspecto | Predição de AVC | Classificação de Pneumonia |
|---------|----------------|---------------------------|
| **Tipo de Dados** | Tabelar (features clínicas) | Imagens (raio-X) |
| **Modelo Final** | Logistic Regression | CNN Customizada |
| **Melhor Métrica** | Recall (75.68%) | Precision (91.89%) |
| **Principal Desafio** | Desbalanceamento de classes | Baixo recall |
| **Aplicabilidade** | Alta (triagem preventiva) | Baixa (necessita melhorias) |
| **Status** | ✅ Pronto para uso em triagem | ⚠️ Requer refinamento |

### 3.4. Conclusões e Próximos Passos

#### 3.4.1. Predição de AVC
- **Status**: Modelo funcional para triagem preventiva
- **Próximos Passos**:
  - Coletar mais dados de casos positivos
  - Implementar sistema de explicação (SHAP) para transparência
  - Validar com médicos especialistas
  - Integrar em sistema CDSS

#### 3.4.2. Classificação de Pneumonia
- **Status**: Modelo requer melhorias significativas
- **Próximos Passos**:
  - Implementar transfer learning
  - Ajustar threshold para priorizar recall
  - Aumentar dataset de validação
  - Reavaliar arquitetura da CNN

#### 3.4.3. Considerações Finais
- Ambos os projetos demonstram o potencial da IA em saúde
- É fundamental priorizar métricas apropriadas para contexto médico (recall em triagem)
- Modelos devem ser interpretáveis e transparentes para ganhar confiança médica
- Validação clínica é essencial antes de uso em produção

---

**Projeto**: Tech Challenge FIAP - Fase 1  
**Autor**: Gabriel Francisco Gonsalves Teixeira
