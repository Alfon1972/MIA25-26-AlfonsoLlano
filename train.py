"""
Script para entrenar el modelo campeón y registrarlo en el model registry.

Toma como entrada un archivo CSV con datos etiquetados y genera
una nueva instancia del modelo en la carpeta models/prod/ (archivando
la versión anterior en models/archive/).

Argumentos obligatorios:
    - path_data: Ruta al archivo CSV con datos de entrenamiento.
    - model_version_id: Identificador de versión del modelo (ej. YYYYMM).

Argumentos opcionales:
    - min_df: Frecuencia mínima de documento para vectorización (default: 1).
    - max_df: Frecuencia máxima de documento (default: 0.5).
    - max_features: Número máximo de features para vectorización (default: 128).
"""

import argparse
import os
import pandas as pd
import pickle
import shutil
from datetime import datetime
import logging
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Importar módulos del proyecto
from src import models

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def register_model(model, metadata, version_id, path_model_prod, path_model_arch):
    """
    Registra el modelo en producción y archiva la versión anterior.
    """
    # Archivar modelo anterior si existe
    model_file = os.path.join(path_model_prod, "model.pkl")
    meta_file = os.path.join(path_model_prod, "metadata.pkl")

    if os.path.exists(model_file):
        archive_folder = os.path.join(path_model_arch, version_id)
        os.makedirs(archive_folder, exist_ok=True)
        shutil.copy2(model_file, os.path.join(archive_folder, "model.pkl"))
        if os.path.exists(meta_file):
            shutil.copy2(meta_file, os.path.join(archive_folder, "metadata.pkl"))
        logging.info(f"Modelo anterior archivado en: {archive_folder}")

    # Guardar nuevo modelo en producción
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)

    logging.info(f"Nuevo modelo guardado en: {path_model_prod}")


def main(
    path_data: str,
    model_version_id: str,
    min_df: int,
    max_df: float,
    max_features: int
) -> None:
    """
    Pipeline principal de entrenamiento.
    """

    # Rutas
    path_interim = os.path.join("data", "interim")
    path_model_prod = os.path.join("models", "prod")
    path_model_arch = os.path.join("models", "archive")

    # Crear carpetas si no existen
    os.makedirs(path_interim, exist_ok=True)
    os.makedirs(path_model_prod, exist_ok=True)
    os.makedirs(path_model_arch, exist_ok=True)

    # ============================
    # 1. Cargar datos
    # ============================
    logging.info(f"Leyendo datos desde: {path_data}")
    df_raw = pd.read_csv(path_data)
    logging.info(f"Forma del dataset: {df_raw.shape}")
    logging.info(f"Columnas: {df_raw.columns.tolist()}")

    # ============================
    # 2. Preparar datos
    # ============================
    df_prep = df_raw.copy()

    # Detectar formato de datos (raw vs preprocesado)
    if 'x_text' in df_prep.columns and 'y_is_nf' in df_prep.columns:
        # Datos ya preprocesados (vienen de notebook 01)
        logging.info("Datos ya preprocesados (x_text, y_is_nf)")
    elif 'FINAL_LABEL' in df_prep.columns and 'REQUIREMENT' in df_prep.columns:
        # Datos raw originales
        logging.info("Datos raw detectados (FINAL_LABEL, REQUIREMENT)")
        df_prep['y_is_nf'] = df_prep['FINAL_LABEL'].map(models.TARGET_MAP)
        df_prep['x_text'] = df_prep['REQUIREMENT']
    else:
        logging.error(f"Formato de datos no reconocido. Columnas: {df_prep.columns.tolist()}")
        return

    # Mantener solo columnas necesarias
    df_prep = df_prep[["x_text", "y_is_nf"]].copy()

    # Eliminar filas con valores nulos
    nulos_antes = len(df_prep)
    df_prep = df_prep.dropna()
    nulos_eliminados = nulos_antes - len(df_prep)
    if nulos_eliminados > 0:
        logging.info(f"Eliminadas {nulos_eliminados} filas con valores nulos")

    # Asegurar que y_is_nf es entero
    df_prep['y_is_nf'] = df_prep['y_is_nf'].astype(int)

    logging.info(f"Distribución de clases:\n{df_prep['y_is_nf'].value_counts(normalize=True)}")

    # ============================
    # 3. Dividir datos
    # ============================
    df_train, df_test = train_test_split(
        df_prep,
        test_size=0.2,
        random_state=42,
        stratify=df_prep['y_is_nf']
    )

    logging.info(f"Datos de entrenamiento: {df_train.shape}")
    logging.info(f"Datos de test: {df_test.shape}")

    # Guardar splits
    df_train.to_csv(os.path.join(path_interim, "train.csv"), index=False)
    df_test.to_csv(os.path.join(path_interim, "test.csv"), index=False)
    logging.info(f"Datos guardados en {path_interim}")

    # ============================
    # 4. Entrenar modelo
    # ============================
    skl_pl = models.get_model(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )

    X_train, y_train = df_train['x_text'], df_train['y_is_nf']
    X_test, y_test = df_test['x_text'], df_test['y_is_nf']

    logging.info("Entrenando modelo...")
    skl_pl.fit(X_train, y_train)

    # ============================
    # 5. Evaluar modelo
    # ============================
    y_pred_train = skl_pl.predict(X_train)
    y_pred_test = skl_pl.predict(X_test)

    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    logging.info(f"F1 Score (Train): {f1_train:.4f}")
    logging.info(f"F1 Score (Test): {f1_test:.4f}")

    # ============================
    # 6. Registrar modelo
    # ============================
    metadata = {
        "score": "f1",
        "f1_train": f1_train,
        "f1_test": f1_test,
        "version_id": model_version_id,
        "exe_dt": datetime.now().strftime("%Y%m%d"),
        "sklearn": sklearn.__version__,
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
        "train_shape": df_train.shape,
        "test_shape": df_test.shape
    }

    register_model(
        skl_pl,
        metadata,
        model_version_id,
        path_model_prod=path_model_prod,
        path_model_arch=path_model_arch
    )

    logging.info(f"Modelo registrado con versión: {model_version_id}")
    logging.info("=" * 60)
    logging.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenar y evaluar el modelo de clasificación de texto."
    )

    parser.add_argument("path_data", type=str,
        help="Ruta al archivo CSV con datos etiquetados.")
    parser.add_argument("model_version_id", type=str,
        help="ID de versión del modelo (ej. 202604).")
    parser.add_argument("--min_df", type=int, default=1,
        help="Frecuencia mínima de documento (default: 1).")
    parser.add_argument("--max_df", type=float, default=0.5,
        help="Frecuencia máxima de documento (default: 0.5).")
    parser.add_argument("--max_features", type=int, default=128,
        help="Número máximo de features (default: 128).")

    args = parser.parse_args()
    main(
        args.path_data,
        args.model_version_id,
        args.min_df,
        args.max_df,
        args.max_features
    )
