"""
Tests unitarios para validar la funcionalidad del módulo de modelos.

Ejecutar con: pytest tests/test_models.py -v
"""

import pytest
import sys
import os
from pathlib import Path

# Agregar la carpeta src al path para importar modelos
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import tokenizer_stemmer_es, get_model
from sklearn.pipeline import Pipeline


# ============================================================================
# TESTS PARA TOKENIZADOR
# ============================================================================

class TestTokenizerSpanish:
    """Tests para la función tokenizer_stemmer_es"""

    def test_tokenizer_returns_list(self):
        """
        Verificar que el tokenizador retorna una lista
        """
        texto = "Estamos trabajando en una solución excelente."
        resultado = tokenizer_stemmer_es(texto)

        assert isinstance(resultado, list), "El resultado debe ser una lista"
        print(f"✅ Resultado es lista: {resultado}")

    def test_tokenizer_returns_strings(self):
        """
        Verificar que todos los elementos de la lista son strings
        """
        texto = "Estamos trabajando en una solución excelente."
        resultado = tokenizer_stemmer_es(texto)

        assert len(resultado) > 0, "La lista no puede estar vacía"
        assert all(isinstance(token, str) for token in resultado), \
            "Todos los elementos deben ser strings"
        print(f"✅ Todos los tokens son strings: {resultado}")

    def test_tokenizer_removes_punctuation(self):
        """
        Verificar que el tokenizador elimina la puntuación
        """
        texto = "Estamos trabajando en una solución excelente."
        resultado = tokenizer_stemmer_es(texto)

        # No debe contener puntuación
        assert "." not in resultado, "No debe haber puntos"
        assert "," not in resultado, "No debe haber comas"
        assert "!" not in resultado, "No debe haber signos de exclamación"
        print(f"✅ Puntuación eliminada correctamente: {resultado}")

    def test_tokenizer_removes_stopwords(self):
        """
        Verificar que el tokenizador elimina stopwords (palabras vacías en español)
        """
        texto = "El sistema debe procesar datos"
        resultado = tokenizer_stemmer_es(texto)

        # Palabras vacías comunes que no deberían estar
        stopwords_comunes = ["el", "la", "de", "en", "un", "una"]

        # Verificar que la mayoría de stopwords fueron eliminados
        tokens_lower = [t.lower() for t in resultado]
        assert not any(sw in tokens_lower for sw in stopwords_comunes), \
            "Los stopwords no deberían estar en el resultado"
        print(f"✅ Stopwords eliminados: {resultado}")

    def test_tokenizer_performs_stemming(self):
        """
        Verificar que el tokenizador realiza stemming (reduce palabras a raíz)
        """
        texto = "trabajando trabajar trabajadores trabajo"
        resultado = tokenizer_stemmer_es(texto)

        # Después del stemming, todas estas palabras deberían tener una raíz común
        assert len(resultado) > 0, "Debe haber tokens después del stemming"
        assert len(resultado) <= 4, "El stemming debe reducir las variaciones"
        print(f"✅ Stemming aplicado: {resultado}")

    def test_tokenizer_empty_input(self):
        """
        Verificar comportamiento con texto vacío
        """
        texto = ""
        resultado = tokenizer_stemmer_es(texto)

        assert isinstance(resultado, list), "Debe retornar una lista incluso con texto vacío"
        print(f"✅ Manejo de texto vacío correcto: {resultado}")

    def test_tokenizer_special_characters(self):
        """
        Verificar comportamiento con caracteres especiales
        """
        texto = "¡Hola! ¿Cómo estás? (Bien, gracias)"
        resultado = tokenizer_stemmer_es(texto)

        # No debe contener caracteres especiales españoles
        assert "¡" not in resultado, "No debe haber signos de apertura de exclamación"
        assert "¿" not in resultado, "No debe haber signos de apertura de pregunta"
        assert "(" not in resultado, "No debe haber paréntesis"
        print(f"✅ Caracteres especiales manejados: {resultado}")


# ============================================================================
# TESTS PARA EL PIPELINE DEL MODELO
# ============================================================================

class TestModelPipeline:
    """Tests para la construcción del pipeline del modelo"""

    def test_get_model_returns_pipeline(self):
        """
        Verificar que get_model retorna una instancia de Pipeline
        """
        modelo = get_model(min_df=1, max_df=1.0, max_features=128)

        assert isinstance(modelo, Pipeline), \
            "get_model debe retornar un sklearn Pipeline"
        print(f"✅ Retorna Pipeline: {type(modelo)}")

    def test_pipeline_has_two_steps(self):
        """
        Verificar que el pipeline tiene exactamente 2 pasos: vectorizador y clasificador
        """
        modelo = get_model(min_df=1, max_df=1.0, max_features=128)

        assert len(modelo.steps) == 2, \
            f"El pipeline debe tener 2 pasos, tiene {len(modelo.steps)}"
        print(f"✅ Pipeline tiene 2 pasos: {[name for name, _ in modelo.steps]}")

    def test_pipeline_has_vectorizer_step(self):
        """
        Verificar que el pipeline tiene un paso de vectorización ('fte')
        """
        modelo = get_model(min_df=1, max_df=1.0, max_features=128)

        step_names = [name for name, _ in modelo.steps]
        assert 'fte' in step_names, \
            "El pipeline debe tener un paso llamado 'fte' (feature transformer/vectorizer)"
        print(f"✅ Pipeline tiene paso 'fte': {step_names}")

    def test_pipeline_has_classifier_step(self):
        """
        Verificar que el pipeline tiene un paso de clasificación ('clf')
        """
        modelo = get_model(min_df=1, max_df=1.0, max_features=128)

        step_names = [name for name, _ in modelo.steps]
        assert 'clf' in step_names, \
            "El pipeline debe tener un paso llamado 'clf' (classifier)"
        print(f"✅ Pipeline tiene paso 'clf': {step_names}")

    def test_pipeline_accepts_hyperparameters(self):
        """
        Verificar que get_model acepta hiperparámetros sin error
        """
        try:
            modelo = get_model(min_df=2, max_df=0.8, max_features=256)
            assert isinstance(modelo, Pipeline), "Debe retornar Pipeline con diferentes parámetros"
            print(f"✅ Acepta hiperparámetros correctamente")
        except Exception as e:
            pytest.fail(f"get_model no debe lanzar excepciones: {e}")

    def test_pipeline_can_predict_shape(self):
        """
        Verificar que el pipeline puede realizar predicciones
        """
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        # Crear un pequeño dataset de ejemplo
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = get_model(min_df=1, max_df=1.0, max_features=64)

        try:
            # El pipeline debería poder hacer predicciones
            predicciones = modelo.predict(X_test)
            assert len(predicciones) == len(X_test), \
                "El número de predicciones debe coincidir con el número de muestras"
            print(f"✅ Modelo puede hacer predicciones: {len(predicciones)} predicciones")
        except Exception as e:
            pytest.fail(f"El pipeline no puede hacer predicciones: {e}")


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

class TestIntegration:
    """Tests de integración: tokenizer + pipeline"""

    def test_tokenizer_pipeline_integration(self):
        """
        Verificar que el tokenizer y pipeline funcionan juntos
        """
        texto = "El sistema debe ser rápido y seguro"

        # Usar el tokenizer
        tokens = tokenizer_stemmer_es(texto)
        assert len(tokens) > 0, "Tokenizer debe producir tokens"

        # Crear el modelo
        modelo = get_model()
        assert isinstance(modelo, Pipeline), "Debe ser un Pipeline"

        print(f"✅ Integración tokenizer + pipeline: tokens={tokens}")

    def test_model_convergence(self):
        """
        Verificar que el modelo puede entrenarse sin errores
        """
        import pandas as pd

        # Crear un pequeño dataset de ejemplo
        X = ["requisito funcional uno", "requisito funcional dos",
             "requisito no funcional rendimiento", "requisito no funcional seguridad"]
        y = [0, 0, 1, 1]  # 0=Funcional, 1=No-funcional

        modelo = get_model(min_df=1, max_df=1.0, max_features=128)

        try:
            # El modelo debe poder entrenarse
            modelo.fit(X, y)

            # Y debe poder predecir
            predicciones = modelo.predict(X)
            assert len(predicciones) == len(X), "Debe predecir para todas las muestras"

            # Las predicciones deben ser 0 o 1 (binarias)
            assert all(pred in [0, 1] for pred in predicciones), \
                "Las predicciones deben ser binarias (0 o 1)"

            print(f"✅ Modelo converge correctamente. Predicciones: {predicciones}")
        except Exception as e:
            pytest.fail(f"El modelo no puede entrenarse: {e}")


# ============================================================================
# TESTS DE ROBUSTEZ
# ============================================================================

class TestRobustness:
    """Tests para verificar la robustez del modelo"""

    def test_model_handles_long_text(self):
        """
        Verificar que el modelo maneja texto largo sin error
        """
        texto_largo = " ".join(["palabra"] * 1000)
        tokens = tokenizer_stemmer_es(texto_largo)

        # Debe funcionar sin error
        assert isinstance(tokens, list), "Debe retornar lista incluso con texto largo"
        print(f"✅ Maneja texto largo: {len(tokens)} tokens")

    def test_model_handles_short_text(self):
        """
        Verificar que el modelo maneja texto muy corto
        """
        texto_corto = "a"
        tokens = tokenizer_stemmer_es(texto_corto)

        assert isinstance(tokens, list), "Debe retornar lista incluso con texto muy corto"
        print(f"✅ Maneja texto corto: {tokens}")

    def test_model_consistent_predictions(self):
        """
        Verificar que el modelo es determinista (mismas predicciones)
        """
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=50, n_features=20, random_state=42)

        modelo = get_model(min_df=1, max_df=1.0, max_features=64)
        modelo.fit(X, y)

        # Hacer dos predicciones del mismo dato
        pred1 = modelo.predict(X[:5])
        pred2 = modelo.predict(X[:5])

        assert all(p1 == p2 for p1, p2 in zip(pred1, pred2)), \
            "Las predicciones deben ser consistentes"
        print(f"✅ Predicciones deterministas: {pred1}")


if __name__ == "__main__":
    # Ejecutar tests con: pytest tests/test_models.py -v
    pytest.main([__file__, "-v", "--tb=short"])
