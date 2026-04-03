#  Model Summary

Clasificar automáticamente requisitos de software en español:
- **Funcional (F):** Describe QUÉ debe hacer el sistema
- **No-Funcional (NF):** Describe CÓMO debe comportarse (rendimiento, seguridad, etc.)

## Architecture
* Scikit Learn Pipeline
* NLTK Snowball stemmer

## Inputs and Outputs
* Input: Document text (requisito en español)
* Output: Class (FUNCTIONAL: 0, NON FUNCTIONAL: 1)

## Terms and Links
* [NLTK Snowball stemmer](https://www.nltk.org/api/nltk.stem.SnowballStemmer.html)

## Data
HuggingFace Dataset [FR_NFR_Spanish_requirements_classification](https://huggingface.co/datasets/MariaIsabel/FR_NFR_Spanish_requirements_classification)

# Model Usage and Limitations

## Uso previsto
El modelo está diseñado para clasificar requisitos de software escritos en español en dos categorías: funcionales (F) y no funcionales (NF). Su principal caso de uso es automatizar el enrutamiento de tickets de requisitos a los equipos de desarrollo apropiados.

## Limitaciones
- **Idioma:** Solo funciona con texto en español. Texto en otros idiomas produce resultados incorrectos.
- **Vocabulario limitado:** Con 128 features máximos, términos muy especializados pueden no ser capturados.
- **Independencia de palabras:** Naive Bayes asume independencia entre palabras, lo cual no es cierto en lenguaje natural.
- **Contexto:** No captura negaciones ni contexto complejo (ej. "NO debe tener latencia").
- **Distribución de datos:** Si los datos nuevos tienen distribución muy diferente al entrenamiento, el rendimiento se degrada.

## Sesgos potenciales
- Sesgo lingüístico: entrenado con español de cierta región/variante.
- Sesgo de selección: solo incluye requisitos etiquetados manualmente.

# Implementation

## Pipeline
```
Texto → Tokenización (NLTK) → Stemming (SnowballStemmer) → CountVectorizer (binario, 128 features) → BernoulliNB → Predicción
```

## Componentes
1. **Tokenizador:** NLTK word_tokenize + eliminación de puntuación + eliminación de stopwords españolas
2. **Stemmer:** SnowballStemmer para español (reduce palabras a su raíz)
3. **Vectorizador:** CountVectorizer con binary=True, min_df=1, max_df=0.5, max_features=128
4. **Clasificador:** BernoulliNB (Naive Bayes para features binarias)

## Hiperparámetros seleccionados
- min_df: 1 (incluir términos que aparecen en al menos 1 documento)
- max_df: 0.5 (excluir términos en más del 50% de documentos)
- max_features: 128 (vocabulario máximo)

## Reproducibilidad
- random_state=42 para splits y validación cruzada
- Estratificación en train/test split

# Evaluation

## Métricas (3-Fold Cross-Validation)
| Métrica | Valor |
|---------|-------|
| F1-Score | 0.82 |
| Precisión | 0.81 |
| Recall | 0.83 |
| Accuracy | 0.82 |

## Métricas en Test (63 muestras)
| Métrica | Valor |
|---------|-------|
| Accuracy | 0.889 |
| F1-Score | 0.759 |
| Precisión | 0.733 |
| Recall | 0.786 |

### Matriz de Confusión
|  | Pred F | Pred NF |
|---|---|---|
| **Real F** | 45 | 4 |
| **Real NF** | 3 | 11 |

## Experimentos realizados
1. **Naive Bayes (GANADOR):** CountVectorizer + BernoulliNB → F1=0.82
2. **Gradient Boosting:** TfidfVectorizer + GradientBoostingClassifier → F1=0.79

## Decisión del modelo campeón
Se selecciona Naive Bayes porque:
- Mayor F1-Score (0.82 vs 0.79)
- 20x más rápido en entrenamiento
- 100x más ligero en tamaño de modelo
- Mejor relación simplicidad/rendimiento

## Features más importantes
Los términos más discriminativos incluyen: sistema, datos, proceso, usuario, rendimiento, seguridad, disponibilidad, respuesta, carga, interfaz.

## Robustez
- Validación cruzada 3-fold reduce overfitting
- Estratificación mantiene balance de clases
- Tests unitarios validan tokenización y pipeline (16/18 tests pasados)

## Recomendaciones
- Monitoreo continuo de predicciones en producción
- Reentrenamiento periódico con nuevos datos
- Validación humana en decisiones críticas
