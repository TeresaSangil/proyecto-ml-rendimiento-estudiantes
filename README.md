# Proyecto Final – Aprendizaje Automático

Comillas ICAI · Curso 2025/2026

---

## Descripción

Este proyecto analiza el abandono, permanencia y éxito académico de estudiantes de educación superior. Se desarrollan tres tareas complementarias de Machine Learning:

1. **Clasificación** — predecir la situación final del estudiante: `Abandono`, `Matriculado` o `Graduado`.
2. **Regresión** — predecir la nota media del segundo semestre (`nota_media_2sem`) sin usar datos del propio segundo semestre.
3. **Aprendizaje no supervisado** — descubrir perfiles latentes de estudiantes mediante clustering y PCA.

---

## Estructura del repositorio

```
.
├── data/
│   └── rendimiento_estudiantes.csv   # Dataset original (separador: ;)
├── notebooks/
│   ├── 1_EDA.ipynb                   # Exploración y análisis descriptivo
│   ├── 2_preprocesado.ipynb          # Codificación, split y escalado
│   ├── 3_clasificacion.ipynb         # Modelos de clasificación supervisada
│   ├── 4_regresion.ipynb             # Modelos de regresión supervisada
│   └── 5_no_supervisado.ipynb        # PCA, K-Means y clustering jerárquico
├── datos_procesados/                 # Objetos y features ya preprocesados
│   ├── features_clf.csv / features_reg.csv
│   ├── scaler_clf.pkl / scaler_reg.pkl / label_encoder_obj.pkl
│   └── (los arrays .npy no se versionan, ver nota más abajo)
├── plots/                            # Figuras generadas por los notebooks
├── docs/
│   ├── 2026 ML proyecto final.pdf    # Enunciado del proyecto
│   └── Informe.docx                  # Informe final
└── README.md
```

> **Nota sobre `datos_procesados/`:** los arrays `.npy` (`X_train_*`, `X_test_*`, `y_train_*`, `y_test_*`) son artefactos derivados y pesados (~28 MB en total). No se versionan en el repositorio (ver `.gitignore`) — se regeneran automáticamente al ejecutar `notebooks/2_preprocesado.ipynb`. Sí se incluyen los objetos livianos (`scaler_*.pkl`, `label_encoder_obj.pkl`, `features_*.csv`) para facilitar la reproducibilidad sin tener que rehacer todo el preprocesado.

---

## Requisitos

### Python
Versión recomendada: **Python 3.9 – 3.11**

### Dependencias

Instalar con pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

O con conda:

```bash
conda install numpy pandas matplotlib seaborn scikit-learn scipy
```

No se requieren librerías externas adicionales.

---

## Instrucciones de ejecución

Los notebooks deben ejecutarse **en orden**, ya que cada uno depende de los artefactos generados por el anterior.

### Paso 1 — EDA

```bash
jupyter notebook notebooks/1_EDA.ipynb
```

- Lee `data/rendimiento_estudiantes.csv`.
- Genera figuras en `plots/`.
- No genera ningún artefacto persistente; es solo exploratorio.

### Paso 2 — Preprocesado *(ejecutar obligatoriamente antes del 3, 4 y 5)*

```bash
jupyter notebook notebooks/2_preprocesado.ipynb
```

- Codifica variables categóricas (LabelEncoder para binarias, OneHotEncoder para nominales).
- Elimina columnas con varianza cero tras OHE.
- Crea dos conjuntos de datos: uno para clasificación (todas las variables) y otro para regresión (excluye variables del 2º semestre para evitar data leakage).
- Guarda en `datos_procesados/` los arrays `.npy`, los scalers/encoder (`.pkl`) y los `features_*.csv`.

### Paso 3 — Clasificación

```bash
jupyter notebook notebooks/3_clasificacion.ipynb
```

- Carga los arrays de `datos_procesados/`.
- Entrena y compara: Regresión Logística, Árbol de Decisión (con GridSearchCV), Random Forest (con GridSearchCV) y Gradient Boosting (con GridSearchCV).
- Métrica principal: **F1-macro**.
- Genera matrices de confusión e importancia de variables del mejor modelo.
- Realiza validación cruzada sobre el conjunto de entrenamiento.

### Paso 4 — Regresión

```bash
jupyter notebook notebooks/4_regresion.ipynb
```

- Predice `nota_media_2sem` usando únicamente características del estudiante al ingreso, variables del primer semestre y variables macroeconómicas.
- Variables excluidas por leakage: todas las del segundo semestre (`asignaturas_2sem_*`, `nota_media_2sem`) y `objetivo_enc`.
- Solo se usan estudiantes con `nota_media_2sem > 0` (excluye abandonos sin nota registrada).
- Modelos: Regresión Lineal, Ridge, Lasso (con GridSearchCV) y Random Forest / Gradient Boosting Regressor (con GridSearchCV).
- Métricas: **RMSE** y **R²**.
- Incluye diagnóstico de residuos (Q-Q plot, residuos vs ajustados, error por rango de nota).

### Paso 5 — Aprendizaje no supervisado

```bash
jupyter notebook notebooks/5_no_supervisado.ipynb
```

- Combina train + test para usar todos los datos disponibles (apropiado en contexto no supervisado).
- Aplica PCA completo para estudiar varianza explicada.
- Reduce dimensionalidad hasta el número de componentes que explican el 90% de la varianza antes del clustering.
- Selecciona k óptimo mediante el método del codo y el coeficiente de silueta.
- Compara K-Means con clustering jerárquico (Ward).
- Interpreta los clusters usando el CSV original (variables sin OHE) y la variable `objetivo` solo como referencia.

---

## Decisiones metodológicas relevantes

| Decisión | Justificación |
|---|---|
| Split estratificado (80/20) con `random_state=42` | Reproducibilidad y preservación de proporciones de clase |
| `class_weight='balanced'` en clasificación | Compensa el desbalance (~49% Graduados, ~18% Matriculados) |
| F1-macro como métrica de clasificación | Penaliza igual errores en todas las clases, incluyendo la minoritaria |
| Exclusión de variables 2º semestre en regresión | Evita data leakage respecto al target `nota_media_2sem` |
| Filtro `nota_media_2sem > 0` en regresión | Elimina filas sin nota real (abandonos tempranos sin dato de 2º semestre) |
| PCA antes de K-Means | Reduce ruido del espacio OHE de alta dimensionalidad y mejora la métrica de silueta |

---

## Reproducibilidad

Todos los modelos con aleatoriedad usan `random_state=42`. Los GridSearchCV usan `n_jobs=-1` para paralelizar; los tiempos de ejecución dependen del hardware, pero los resultados son deterministas.

Tiempos orientativos en un portátil estándar (8 núcleos):

| Notebook | Tiempo aproximado |
|---|---|
| 1 – EDA | < 1 min |
| 2 – Preprocesado | < 1 min |
| 3 – Clasificación | 10–20 min (GridSearch RF y GB) |
| 4 – Regresión | 5–15 min (GridSearch RF y GB) |
| 5 – No supervisado | 2–5 min |

## Documentación adicional

- [`docs/2026 ML proyecto final.pdf`](docs/2026%20ML%20proyecto%20final.pdf) — enunciado original del proyecto.
- [`docs/Informe.docx`](docs/Informe.docx) — informe final con resultados y conclusiones.
