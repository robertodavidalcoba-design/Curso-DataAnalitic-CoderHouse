# Predicción de Rendimiento Académico (Trabajo Final — Data Science II)

Pipeline de Machine Learning para predecir la **nota final** (`nota_final`, escala 0–20) de estudiantes de dos escuelas secundarias de Portugal (Gabriel Pereira y Mousinho da Silveira), a partir de variables socioeconómicas, hábitos y entorno familiar — **sin usar las notas parciales previas** (G1/G2) como predictoras, para evitar fuga de información y evaluar el valor predictivo real de los factores de contexto.

**Autor:** Roberto David Alcoba

---

## 1. Problema y objetivo

- **Tarea:** Regresión.
- **Variable objetivo:** `nota_final` (G3 original), rango 0–20.
- **Métricas de éxito:** R², R² ajustado, RMSE y MAE sobre un conjunto *holdout* (20%) no visto durante el entrenamiento.
- **Relevancia de negocio:** el modelo busca servir como señal complementaria de alerta temprana para priorizar intervención pedagógica (tutorías, apoyo socioeconómico), no como reemplazo del criterio docente. Por eso se excluyen deliberadamente `nota_periodo1` y `nota_periodo2` (G1/G2): incluirlas resuelve el problema de forma casi trivial (están altamente correlacionadas con G3) pero no aporta valor de negocio, ya que para cuando existen esas notas parciales gran parte del período académico ya transcurrió.

## 2. Dataset

- **Fuente:** [Student Performance Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/320/student+performance), recolectado en escuelas secundarias de Portugal.
- **Archivo esperado:** `data/student_data.csv`
- **Filas / columnas originales:** 395 estudiantes, 33 variables (renombradas al español en el pipeline).
- **Escuelas:** `GP` = Gabriel Pereira (Évora), `MS` = Mousinho da Silveira (Portalegre).

> El dataset no se distribuye en este repositorio por tamaño/licencia. Descargarlo del enlace de UCI y colocarlo en `data/student_data.csv` antes de ejecutar el pipeline (ver sección 5).

## 3. Estructura del repositorio
```text
project-root/
│
├── README.md           # Documentación del proyecto
├── requirements.txt    # Librerías necesarias
├── outputs/          # Notebooks ordenados por etapa
├── scripts/            # Módulos Python reutilizables (utils.py)
├── data/               # Datasets de entrada
└── doc/                # reportes Informe Final, Resultados del Pipeline
├── data/               # Datasets de entrada
└── img/                # Gráficos y reportes generados

## 4. Requisitos

- Python 3.10+
- Dependencias principales:

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
xgboost
lightgbm
shap
joblib
```

Instalación:

```bash
pip install -r requirements.txt
```

> `xgboost` y `lightgbm` son opcionales: si no están instalados, el pipeline continúa y salta esos modelos con un aviso por consola.

## 5. Cómo ejecutar el proyecto

### Opción A — Notebook (Google Colab)

El notebook fue desarrollado originalmente en Google Colab y monta Google Drive para localizar automáticamente la raíz del proyecto (carpetas `data/`, `doc/`, `img/`, `outputs/`, `scripts/`) a partir del propio nombre del `.ipynb`.

1. Subir la carpeta completa del proyecto (con `data/student_data.csv` incluido) a Google Drive.
2. Abrir `Trabajo Final DataII_RobertoDavidAlcoba.ipynb` en Colab.
3. Ejecutar todas las celdas (`Entorno de ejecución → Ejecutar todas`). La primera celda solicitará autorización para montar Drive.

### Opción B — Local / fuera de Colab

El script depende de `google.colab.drive`, por lo que para correrlo localmente hay que reemplazar el bloque de montaje de Drive por una ruta local fija. Pasos sugeridos:

1. Clonar el repositorio y colocar el dataset en `data/student_data.csv`.
2. En `scripts/pipeline_trabajo_final.py`, sustituir:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   por la definición directa del diccionario `paths` apuntando a las carpetas locales del repo (`data/`, `doc/`, `img/`, `outputs/`, `scripts/`).
3. Ejecutar:
   ```bash
   python scripts/pipeline_trabajo_final.py
   ```

Al finalizar, el pipeline genera en `outputs/`:
- CSV con el ranking de variables (`f_regression` vs `mutual_info_regression`).
- Modelo entrenado serializado: `modelo_lineal_nota_final.pkl` (pipeline completo: preprocesamiento + `LinearRegression`).
- Gráficos de diagnóstico y SHAP en `img/` e `img/shap/`.

## 6. Resumen del pipeline

1. **Carga y estandarización:** lectura de `student_data.csv`, renombre de 33 columnas al español.
2. **EDA y detección de atípicos:** análisis univariado del target, IQR y puntaje Z, revisión de consistencia de categorías.
3. **Codificación categórica:** binarias/ordinales por mapeo numérico, nominales por One-Hot Encoding (`drop_first=True`).
4. **Escalado híbrido:** `RobustScaler` para variables con `|skew| > 1.0` (ej. ausencias), `MinMaxScaler` para el resto.
5. **Feature engineering:** creación de 4 índices sintéticos (`indice_riesgo_academico`, `nivel_educativo_parental`, `indice_riesgo_social`, `indice_compromiso_academico`).
6. **Análisis bivariado y VIF:** verificación de ausencia de multicolinealidad severa (VIF < 2.0).
7. **Selección de variables:** dos *feature sets* — Set A (10 variables, `f_regression`, familia lineal) y Set B (13 variables, `f_regression` + `mutual_info_regression`, familia de ensambles).
8. **Entrenamiento y optimización:**
   - Familia lineal: `LinearRegression`, `RidgeCV`, `LassoCV` sobre Set A.
   - Familia de ensambles: `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor` sobre Set B, tuneados con `RandomizedSearchCV` (`n_iter=25`) — se prefirió sobre `GridSearchCV` por el tamaño combinado del espacio de hiperparámetros de los tres modelos, con costo computacional acotado.
   - Validación: K-Fold (`cv=5`) + *holdout* 80/20.
9. **Diagnóstico de residuos:** pruebas de Shapiro-Wilk (normalidad), Breusch-Pagan (homocedasticidad) y Durbin-Watson (independencia) sobre el modelo lineal ganador.
10. **Explicabilidad (SHAP):** `shap.LinearExplainer` y `shap.TreeExplainer`, *summary plots* guardados por modelo.
11. **Exportación:** pipeline completo (preprocesador + modelo) serializado con `joblib`.

## 7. Resultados (holdout 20%)

| Familia | Feature Set | Modelo | R² | R² ajustado | RMSE | MAE |
|---|---|---|---|---|---|---|
| Lineal | A | **Linear Regression** | **0.112** | -0.018 | 0.213 | 0.167 |
| Lineal | A | Lasso (LassoCV) | 0.106 | -0.026 | 0.214 | 0.168 |
| Ensamble | B | Random Forest | 0.098 | -0.083 | 0.215 | 0.172 |
| Lineal | A | Ridge (RidgeCV) | 0.089 | -0.045 | 0.216 | 0.170 |
| Ensamble | B | XGBoost | 0.068 | -0.119 | 0.219 | 0.176 |
| Ensamble | B | LightGBM | -0.081 | -0.298 | 0.235 | 0.198 |

*RMSE y MAE están en la escala procesada/escalada.*

**Modelo elegido:** `LinearRegression` sobre Feature Set A. Nota importante: al excluir G1/G2 del conjunto de predictores, el poder explicativo es intencionalmente bajo (R² ajustado negativo en todos los modelos); el valor del proyecto está en la interpretabilidad de los coeficientes/SHAP más que en la precisión predictiva pura. Se recomienda usarlo como señal complementaria de priorización, no como decisión automática.

## 8. Explicabilidad — hallazgos principales (SHAP)

- `indice_riesgo_academico` (materias reprobadas, ausencias severas, tiempo de estudio) es el factor con mayor impacto **negativo** sobre la nota final.
- `nivel_educativo_parental` y `clases_particulares_pagadas` son los factores protectores con mayor impacto **positivo**.
- `indice_riesgo_social` (alcohol, salidas, relación amorosa) tiene impacto negativo moderado pero constante.

Ver gráficos en `img/shap/`.

## 9. Licencia y créditos

Dataset: P. Cortez y A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, UCI Machine Learning Repository.
Uso académico — Trabajo Final, materia Data Science II.


