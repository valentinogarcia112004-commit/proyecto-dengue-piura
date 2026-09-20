# Guía de organización — Notebooks del proyecto (Dengue Piura)

Esta guía reorganiza el código que ya construimos (`eda_dengue_piura.py` y
`modelado_dengue.py`) en **5 notebooks**, siguiendo la estructura de
repositorio exigida en la sección 29.1 de tu proyecto. Cada notebook
corresponde a una fase de CRISP-DM y a secciones específicas del informe,
así sabes exactamente qué capturas/evidencia pegar en cada parte del paper.

```
PROYECTO-IA/
├── data/
│   ├── raw/                     -> dataset_dengue_actualizado.csv
│   └── processed/                -> dengue_piura_procesado.csv, resultados_modelos.csv, etc.
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
```

Regla general que pide tu profesor: **cada notebook alterna código y celdas
Markdown de interpretación.** No basta con ejecutar y mostrar un gráfico o
una tabla — cada bloque de código debe ir seguido de una celda de texto que
responda "¿qué evidencia aporta esto?". Abajo te indico exactamente qué
interpretar en cada caso.

---

## 01_data_understanding.ipynb
**Corresponde a:** Fase 2 — Comprensión de los datos (sección 15 del proyecto)
**Objetivo:** Describir el dataset crudo, sin modificarlo todavía.

### Código a incluir
- Carga del CSV original (`dataset_dengue_actualizado.csv`)
- `df.shape`, `df.dtypes`
- Conteo de registros por `ANO`
- Conteo de nulos por columna
- Cardinalidad de las categóricas: `DEPARTAMENTO`, `PROVINCIA`, `DISTRITO`, `SEXO`, `TIPO_EDAD`

```python
# Bloque: Ficha técnica del dataset
import pandas as pd
df = pd.read_csv("../data/raw/dataset_dengue_actualizado.csv")
print("Registros:", df.shape[0], "| Variables:", df.shape[1])
print(df.dtypes)
```

```python
# Bloque: Cobertura temporal y calidad general
print(df['ANO'].value_counts().sort_index())   # ¿hay años faltantes?
print(df.isnull().sum())                       # nulos por columna
for c in ['DEPARTAMENTO','PROVINCIA','DISTRITO','SEXO','TIPO_EDAD']:
    print(c, "->", df[c].nunique(), "categorías únicas")
```

### Markdown de interpretación a agregar
- **Ficha del dataset** (tabla): Fuente = Gobierno Regional de Piura; Periodo = 2023, 2025-2026 (falta 2024); Registros = 47,082; Variable objetivo = no viene predefinida, se construye; Formato = CSV; Procedencia = datos abiertos regionales.
- Explicar el hueco de 2024: ¿es un vacío real de reporte o un recorte del dataset? (limitación a declarar en sección 23).
- Explicar por qué `DISTRITO` (155 categorías) no es viable como feature directa, pero `PROVINCIA` (8) sí.

**Va en el informe:** sección 15 (Fase 2) y parte de la Data Card (16.1).

---

## 02_eda.ipynb
**Corresponde a:** Fase 3.3 — EDA (sección 16.3)
**Objetivo:** Explorar visualmente el dataset ya con limpieza ligera, para *descubrir* el problema y la variable objetivo — todavía no es la preparación final para modelar.

### Código a incluir (del script `eda_dengue_piura.py`, bloques 2 y 3-5)
1. Limpieza ligera: normalizar texto, filtrar `DEPARTAMENTO == "PIURA"`, convertir fechas
2. Cálculo exploratorio de `DELAY_DIAS`
3. Histograma de `DELAY_DIAS`
4. Casos por año (gráfico de barras)
5. Casos por mes de inicio de síntomas (estacionalidad)
6. Distribución de edad
7. Retraso promedio por provincia (tabla/gráfico)

```python
# Bloque: Limpieza ligera para exploración (NO es la preparación final de datos)
for col in ["DEPARTAMENTO","PROVINCIA","DISTRITO","SEXO","TIPO_EDAD"]:
    df[col] = df[col].astype(str).str.strip().str.upper()
df = df[df["DEPARTAMENTO"] == "PIURA"].copy()

for col in ["FECHA_INICIO_SINTOMAS","FECHA_NOTIFICACION"]:
    df[col+"_DT"] = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="coerce")

df["DELAY_DIAS"] = (df["FECHA_NOTIFICACION_DT"] - df["FECHA_INICIO_SINTOMAS_DT"]).dt.days
```

```python
# Bloque: Distribución del retraso, estacionalidad, demografía, geografía
# (pega aquí los bloques 5.1 a 5.7 de eda_dengue_piura.py)
```

### Markdown de interpretación a agregar (una celda después de CADA gráfico)
- Histograma de retraso → "la mayoría de casos se notifica en pocos días (mediana 2), pero hay una cola larga de retrasos extremos."
- Casos por mes → "el pico marzo-mayo coincide con la temporada de lluvias/calor, ciclo reproductivo del vector *Aedes aegypti*."
- Retraso por provincia → "**Huancabamba (19.8 días) es un outlier claro** frente a Piura (4.8) y Sullana (2.6) — evidencia cuantitativa de brecha de acceso geográfico a salud."
- Esta última tabla es tu evidencia más fuerte: cítala explícitamente en la sección 5 (Situación problemática) de tu informe.

**Va en el informe:** sección 16.3 (EDA) y aporta evidencia directa a la sección 5 (Situación problemática) y sección 9 (Justificación).

---

## 03_preprocessing.ipynb
**Corresponde a:** Fase 3.4 — Preparación e ingeniería de datos + Fase 3.5 — Prevención de data leakage (secciones 16.4 y 16.5)
**Objetivo:** Dejar el dataset listo para modelar, con cada transformación justificada.

### Código a incluir (bloques 3.1, 3.2, 4 y parte de 6 de `eda_dengue_piura.py`; bloque 3 de `modelado_dengue.py`)
1. Eliminar retrasos negativos (errores de registro)
2. Eliminar outliers extremos (> 90 días)
3. Probar umbrales (3, 5, 7, 10, 14 días) y tabla de balance de clases
4. Justificar y fijar el umbral definitivo = 5 días → construir `RETRASO_NOTIFICACION`
5. Selección final de features: `EDAD`, `SEXO`, `PROVINCIA`, `MES_SINTOMAS`
6. One-hot encoding
7. Guardar `dengue_piura_procesado.csv`

```python
# Bloque: Limpieza de inconsistencias (justificar cada filtro)
n_negativos = (df["DELAY_DIAS"] < 0).sum()   # errores de registro
df = df[df["DELAY_DIAS"] >= 0].copy()
df = df[df["DELAY_DIAS"] <= 90].copy()        # outliers extremos
```

```python
# Bloque: Selección del umbral para la variable objetivo (probar y justificar)
for umbral in [3, 5, 7, 10, 14]:
    print(umbral, (df["DELAY_DIAS"] > umbral).mean())

UMBRAL_DIAS = 5   # justificación: mediana=2 días, 5 días = más del doble de lo típico
df["RETRASO_NOTIFICACION"] = (df["DELAY_DIAS"] > UMBRAL_DIAS).astype(int)
```

```python
# Bloque: Codificación de variables y verificación de prevención de data leakage
features_cat = ["SEXO","PROVINCIA","MES_SINTOMAS"]
df_model = pd.get_dummies(df[["EDAD"]+features_cat+["RETRASO_NOTIFICACION"]], columns=features_cat, drop_first=True)
```

### Markdown de interpretación a agregar
- **Tabla de balance de clases por umbral** con la justificación explícita del umbral 5 (ya la escribimos juntos: "mediana=2 días, por lo tanto 5 días representa más del doble del tiempo típico").
- **Prevención de data leakage (obligatorio, sección 16.5):** escribe explícitamente que todas las variables usadas (`EDAD`, `SEXO`, `PROVINCIA`, `MES_SINTOMAS`) están disponibles **en el momento en que el paciente llega al centro de salud** — ninguna depende de información posterior al evento de notificación, por lo que no hay fuga temporal.
- Justificar por qué se excluye `DISTRITO` (alta cardinalidad) y `TIPO_EDAD` (98.9% un solo valor, sin varianza informativa).

**Va en el informe:** secciones 16.4, 16.5 completas.

---

## 04_modeling.ipynb
**Corresponde a:** Diseño de la solución + Desarrollo e implementación (secciones 17 y 18)
**Objetivo:** Entrenar el baseline y los dos modelos, justificando cada elección técnica.

### Código a incluir (bloques 4-8 de `modelado_dengue.py`)
1. Cargar `dengue_piura_procesado.csv`
2. Train/test split estratificado
3. Escalado (solo para logística)
4. Baseline (`DummyClassifier`)
5. Modelo 1: Regresión Logística
6. Modelo 2: Random Forest

```python
# Bloque: Split estratificado (mantiene proporción de clases en train y test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

```python
# Bloque: Baseline obligatorio — referencia mínima que todo modelo debe superar
baseline = DummyClassifier(strategy="most_frequent", random_state=42)
```

```python
# Bloque: Modelo 1 - Regresión Logística (interpretable, rápida, bajo riesgo de overfitting)
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
```

```python
# Bloque: Modelo 2 - Random Forest (captura no linealidades, feature importance nativa)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
```

### Markdown de interpretación a agregar
- **Sección 18.1 (obligatoria):** justificar el algoritmo — no basta con "porque funciona bien". Aquí ya tienes el argumento: logística por interpretabilidad directa (coeficientes = odds ratio), Random Forest por capacidad de capturar interacciones no lineales (ej. edad × provincia) y por dar `feature_importance` sin preprocesamiento adicional.
- **Sección 18.2 (Baseline):** explicar por qué el Dummy es necesario aunque tenga "buen accuracy" — para exponer la paradoja del accuracy en clases desbalanceadas.
- Explicar por qué `class_weight="balanced"` sustituye a técnicas más complejas de balanceo (SMOTE, undersampling) sin añadir complejidad innecesaria.

**Va en el informe:** secciones 17, 18.1, 18.2 (Diseño de la solución, Diseño del modelo, Baseline).

---

## 05_evaluation.ipynb
**Corresponde a:** Evaluación (sección 20 completa)
**Objetivo:** Comparar modelos, interpretar resultados y justificar decisiones — es el notebook con más peso de interpretación.

### Código a incluir (bloques 9-13 de `modelado_dengue.py`, más el ajuste de umbral)
1. Tabla comparativa de métricas (baseline, logística, RF)
2. Matrices de confusión (3 paneles)
3. Curvas ROC comparadas
4. Coeficientes de la logística (top 10)
5. Feature importance de Random Forest (top 10)
6. Ajuste de umbral de decisión (tabla + curva Precision-Recall)

```python
# Bloque: Tabla comparativa (nunca reportar solo accuracy)
resultados = pd.DataFrame([...])  # Accuracy, Precision, Recall, F1, ROC-AUC de los 3 modelos
```

```python
# Bloque: Interpretabilidad - comparar variables top de ambos modelos
coef_df = pd.DataFrame({"Variable": X.columns, "Coeficiente": logreg.coef_[0]})
importancia_df = pd.DataFrame({"Variable": X.columns, "Importancia": rf.feature_importances_})
```

```python
# Bloque: Ajuste del umbral de decisión
for umbral in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_umbral = (y_proba_rf >= umbral).astype(int)
    # calcular precision, recall, f1
```

### Markdown de interpretación a agregar (esto es lo que más valora tu profesor)
- **Paradoja del accuracy:** explicar por qué el baseline "gana" en accuracy (83.3%) pero tiene Precision/Recall = 0, y por qué eso lo descalifica como modelo útil.
- **Logística vs Random Forest — empate técnico:** F1 casi idéntico (0.345 vs 0.344), AUC ~0.65 en ambos → argumento de parsimonia: "el modelo más simple (logística) es preferible porque logra rendimiento equivalente con mayor interpretabilidad."
- **Coincidencia de variables importantes entre ambos modelos:** `PROVINCIA_HUANCABAMBA`, `MES_SINTOMAS_5` y `EDAD` aparecen como top en ambos → esta convergencia entre dos algoritmos distintos es evidencia de que la señal es real, no ruido del modelo. Conecta esto explícitamente con el hallazgo del EDA (notebook 02).
- **Justificación del umbral final (0.4):** no es el matemáticamente óptimo en F1 (eso sería 0.5), pero se elige por el costo asimétrico de los errores en un sistema de alerta epidemiológica (un falso negativo — no detectar un caso tardío — es más costoso que un falso positivo).
- **Veredicto de viabilidad:** ROC-AUC ~0.65 es un rendimiento modesto pero significativamente mejor que el azar; limitación honesta por ausencia de variables clínicas/socioeconómicas (va también en sección 23, Amenazas a la validez).

**Va en el informe:** sección 20 completa (20.1 a 20.3), y alimenta directamente las secciones 21 (Resultados), 22 (Discusión) y 23 (Limitaciones).

---

## Resumen rápido: qué notebook → qué sección del informe

| Notebook | Secciones del informe que alimenta |
|---|---|
| 01_data_understanding | 15 (Fase 2), parte de 16.1 (Data Card) |
| 02_eda | 16.3 (EDA), evidencia para 5 (Situación problemática) y 9 (Justificación) |
| 03_preprocessing | 16.4, 16.5 (Preparación y prevención de data leakage) |
| 04_modeling | 17, 18.1, 18.2 (Diseño de la solución, modelo, baseline) |
| 05_evaluation | 20 completa, alimenta 21 (Resultados), 22 (Discusión), 23 (Limitaciones) |

Cuando retomes el formato completo del paper, cada sección ya tiene su
evidencia lista: solo hay que trasladar las tablas, gráficos e
interpretaciones de estos notebooks al documento.
