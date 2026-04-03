"""
Script para ejecutar predicciones batch usando el modelo de producción.

Lee un archivo CSV con una columna de texto, genera predicciones
y guarda los resultados en data/processed/.

Argumentos:
    - path_data: Ruta al archivo CSV de entrada.
    - output_file: Nombre del archivo de salida con predicciones.

El modelo se carga desde: models/prod/model.pkl
Las predicciones se guardan en: data/processed/
"""

import argparse
import os
import pandas as pd
import pickle
import logging

from src.models import TARGET_MAP_REV

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def main(path_data: str, output_file: str) -> None:
    """
    Función principal: carga modelo de producción y genera predicciones.

    Args:
        path_data (str): Ruta al CSV de entrada.
        output_file (str): Nombre del archivo de salida.
    """

    path_processed = os.path.join("data", "processed")
    path_model_prod = os.path.join("models", "prod")

    # Crear carpeta si no existe
    os.makedirs(path_processed, exist_ok=True)

    # Verificar que el archivo de entrada existe
    if not os.path.isfile(path_data):
        logging.error(f"Archivo de entrada no encontrado: {path_data}")
        return

    # Verificar que el modelo existe
    model_path = os.path.join(path_model_prod, "model.pkl")
    if not os.path.isfile(model_path):
        logging.error(f"Modelo de producción no encontrado en: {model_path}")
        return

    # ============================
    # 1. Cargar datos
    # ============================
    logging.info(f"Leyendo datos desde: {path_data}")
    df_score = pd.read_csv(path_data)
    logging.info(f"Forma de los datos: {df_score.shape}")
    logging.info(f"Columnas: {df_score.columns.tolist()}")

    # ============================
    # 2. Preparar datos
    # ============================
    df_prep = df_score.copy()

    # Detectar columna de texto
    if 'x_text' in df_prep.columns:
        text_col = 'x_text'
    elif 'REQUIREMENT' in df_prep.columns:
        text_col = 'REQUIREMENT'
        df_prep['x_text'] = df_prep['REQUIREMENT']
    elif 'requirement' in df_prep.columns:
        text_col = 'requirement'
        df_prep['x_text'] = df_prep['requirement']
    elif 'text' in df_prep.columns:
        text_col = 'text'
        df_prep['x_text'] = df_prep['text']
    else:
        # Usar la primera columna de tipo texto
        text_cols = [col for col in df_prep.columns if df_prep[col].dtype == 'object']
        if text_cols:
            df_prep['x_text'] = df_prep[text_cols[0]]
            text_col = text_cols[0]
            logging.info(f"Usando columna '{text_col}' como texto de entrada")
        else:
            logging.error("No se encontró columna de texto en el archivo de entrada")
            return

    logging.info(f"Datos de entrada: {df_prep.shape}")

    # ============================
    # 3. Cargar modelo
    # ============================
    logging.info(f"Cargando modelo desde: {model_path}")
    with open(model_path, "rb") as file:
        skl_pl = pickle.load(file)

    # ============================
    # 4. Generar predicciones
    # ============================
    logging.info("Generando predicciones...")
    X_score = df_prep['x_text']
    df_prep['y_pred'] = skl_pl.predict(X_score)
    df_prep['y_pred_label'] = df_prep['y_pred'].map(TARGET_MAP_REV)

    # Probabilidades
    try:
        probas = skl_pl.predict_proba(X_score)
        df_prep['prob_F'] = probas[:, 0]
        df_prep['prob_NF'] = probas[:, 1]
    except Exception:
        logging.warning("No se pudieron calcular probabilidades")

    # ============================
    # 5. Guardar resultados
    # ============================
    output_path = os.path.join(path_processed, output_file)
    df_prep.to_csv(output_path, index=False)

    logging.info(f"Predicciones guardadas en: {output_path}")
    logging.info(f"Total predicciones: {len(df_prep)}")
    logging.info(f"Distribución:")
    logging.info(f"\n{df_prep['y_pred_label'].value_counts()}")
    logging.info("=" * 60)
    logging.info("SCORING COMPLETADO EXITOSAMENTE")
    logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cargar modelo de producción y predecir en datos nuevos."
    )

    parser.add_argument("path_data", type=str,
        help="Ruta al archivo de entrada, ej. ./data/raw/mi_archivo.csv")
    parser.add_argument("output_file", type=str,
        help="Nombre del archivo de salida, ej. scoring_202604.csv")

    args = parser.parse_args()
    main(args.path_data, args.output_file)
