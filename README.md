# Predicción de Retraso en la Notificación de Dengue — Región Piura

Proyecto de investigación en Inteligencia Artificial: clasificación del riesgo de
retraso en la notificación de casos de dengue en la región Piura (Perú), mediante
Regresión Logística y Random Forest, usando datos reales del Gobierno Regional de
Piura.

## 1. ¿Cuál es el problema?

Piura concentra la mayor cantidad de casos y muertes por dengue a nivel nacional
en el Perú. La notificación de un caso sospechoso no es un registro administrativo
pasivo: es el disparador operativo del control vectorial y la asignación de
recursos de salud. Cuando esa notificación se retrasa, se pierde la ventana de
oportunidad para contener la propagación local. El análisis exploratorio de este
proyecto encontró que provincias como Huancabamba presentan un retraso promedio
(~19.8 días) casi 4 veces mayor que Piura capital (~4.8 días) — una brecha que hoy
no se identifica de forma anticipada.

## 2. ¿Cuál es el objetivo?

Desarrollar un modelo de aprendizaje automático de clasificación para predecir el
riesgo de retraso en la notificación de casos de dengue en la región Piura,
utilizando datos históricos reales de vigilancia epidemiológica, con la finalidad
de apoyar la focalización oportuna de los recursos de vigilancia y respuesta
epidemiológica.

## 3. ¿Qué dataset se utilizó?

`dataset_dengue_actualizado.csv` — línea de casos de dengue notificados en la
región Piura, publicado como dato abierto por el Gobierno Regional de Piura (GRP).
47,082 registros crudos (46,757 tras filtrar solo Piura), con variables
geográficas, temporales y demográficas. Ver ficha técnica completa en
`docs/guia_notebooks_proyecto.md`.

## 4. ¿Cómo obtener los datos?

El archivo crudo se encuentra en `data/raw/dataset_dengue_actualizado.csv`. Si no
está incluido en este repositorio (por tamaño), puede descargarse desde el portal
de datos abiertos del Gobierno Regional de Piura. El dataset ya procesado
(limpio, con la variable objetivo construida) se guarda automáticamente en
`data/processed/dengue_piura_procesado.csv` al ejecutar el notebook.

## 5. ¿Cómo instalar el proyecto?

```bash
git clone https://github.com/<tu-usuario>/proyecto-dengue-piura.git
cd proyecto-dengue-piura
pip install -r requirements.txt
```

También puede ejecutarse directamente en Google Colab, sin instalación local:
solo se debe subir el archivo `dataset_dengue_actualizado.csv` al entorno de
Colab antes de correr el notebook.

## 6. ¿Cómo ejecutar el código?

Abrir `notebooks/Proyecto_Dengue_Piura.ipynb` en Jupyter o Google Colab y
ejecutar todas las celdas en orden (Entorno de ejecución → Ejecutar todas). El
notebook está organizado en 8 secciones secuenciales: Configuración,
Comprensión de datos, EDA, Preparación de datos, Modelado, Evaluación, Ajuste de
umbral y Conclusión.

## 7. ¿Cómo entrenar el modelo?

El entrenamiento ocurre automáticamente al ejecutar la Sección 5 del notebook
("Diseño y entrenamiento de modelos"), que entrena tres modelos en secuencia:
`DummyClassifier` (baseline), `LogisticRegression` y `RandomForestClassifier`,
ambos con `class_weight="balanced"` y `random_state=42` para reproducibilidad.

## 8. ¿Cómo reproducir los experimentos?

Todos los experimentos (comparación de algoritmos y ajuste de umbral de
decisión) están documentados con código y celdas de interpretación en el mismo
notebook (Secciones 6 y 7). Al usar semillas aleatorias fijas, los resultados
numéricos deben ser idénticos en cada ejecución.

## 9. ¿Cómo ejecutar el prototipo?

**Pendiente.** El proyecto, en su estado actual, no cuenta con un prototipo
desplegado (interfaz web/API) — el modelo se ejecuta y consulta directamente
desde el notebook. Está planificado como trabajo futuro (ver sección 27.2 del
informe).

## 10. ¿Cuáles fueron los resultados?

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (Dummy) | 0.833 | 0.000 | 0.000 | 0.000 | — |
| Regresión Logística | 0.589 | 0.235 | 0.648 | 0.345 | 0.645 |
| Random Forest | 0.585 | 0.233 | 0.652 | 0.344 | 0.655 |

Ambos modelos superan ampliamente al baseline en Recall y F1. Las variables más
influyentes en ambos modelos fueron `PROVINCIA_HUANCABAMBA`, `MES_SINTOMAS_5`
(mayo) y `EDAD`. Umbral de decisión final seleccionado: 0.4 (prioriza Recall por
el costo asimétrico de los errores en salud pública). Detalle completo de la
interpretación en el informe (secciones 20-22).

## Estructura del repositorio

```
proyecto-dengue-piura/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            -> dataset original
│   └── processed/      -> dataset limpio, generado por el notebook
├── notebooks/
│   └── Proyecto_Dengue_Piura.ipynb
└── docs/
    └── guia_notebooks_proyecto.md
```

## Autoría y fuente de datos

Código de autoría propia. Dataset: Gobierno Regional de Piura (dato abierto).
