# Predicción de Fuga de Clientes (Churn) en Telecomunicaciones

Proyecto de Machine Learning aplicado al problema de **Customer Churn** en el sector telecomunicaciones, utilizando el dataset *Telco Customer Churn* de IBM.

## Objetivo

Predecir qué clientes tienen mayor probabilidad de cancelar su servicio para que el negocio pueda priorizar campañas de retención y reducir la pérdida de ingresos.

## Estructura del repositorio

```
.
├── Segunda_Tarea_Acumulativa_EA2.ipynb   # Notebook principal con todo el análisis
├── Telco-Customer-Churn.csv              # Dataset (7,043 clientes, 21 variables)
├── index.html                            # Dashboard autocontenido (~62 KB) listo para GitHub Pages
├── .gitignore
└── README.md
```

El archivo `index.html` es **autocontenido**: los 4 gráficos Plotly están embebidos como `<div>` directamente en el HTML, y `plotly.js` se carga desde un CDN. No hay archivos satélite. Se regenera al ejecutar el notebook.

## Pipeline de análisis (en el notebook)

1. **Entendimiento del negocio** — costo de retener ($10) vs. perder un cliente ($100).
2. **Carga e ingesta** del dataset Telco Customer Churn.
3. **Limpieza y EDA** — manejo de `TotalCharges` con espacios en blanco, detección de outliers con IQR.
4. **Correlación** — variables numéricas vs. `Churn` (heatmap).
5. **Transformación** — One-Hot Encoding + `StandardScaler`.
6. **Modelado** — Regresión Logística, SVM (kernel lineal) y Árbol de Decisión.
7. **Evaluación** — `accuracy`, `F1-score` por clase, matriz de confusión.
8. **Desbalance de clases** — aplicación de **SMOTE** y reentrenamiento.
9. **Importancia de variables** — coeficientes y `feature_importances_`.
10. **Análisis costo-beneficio** — ahorro económico de cada modelo vs. modelo ingenuo.

## Resultados clave

| Métrica                       | Valor               |
| ----------------------------- | ------------------- |
| Tasa de churn del dataset     | **26.5%**           |
| Mejor modelo (por ahorro)     | **Decision Tree (SMOTE)** |
| Ahorro vs. modelo ingenuo     | **$25,950**         |
| Clientes en riesgo detectados | **~3,140**          |

**Factores de mayor influencia en el churn**: contratos mes-a-mes, fibra óptica, ausencia de `OnlineSecurity`, método de pago `Electronic Check` y baja antigüedad (`tenure < 12`).

## Requisitos

- Python 3.9+
- Paquetes: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `imbalanced-learn`, `colorama`, `nbconvert`, `jupyter`.

Instalación rápida:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly imbalanced-learn colorama nbconvert jupyter
```

## Cómo regenerar el dashboard

Ejecuta el notebook completo: la última celda construye `index.html` automáticamente con los KPIs y gráficos actualizados.

```bash
python -m nbconvert --to notebook --execute "Segunda_Tarea_Acumulativa_EA2.ipynb" \
       --output "Segunda_Tarea_Acumulativa_EA2.ipynb" \
       --ExecutePreprocessor.timeout=600
```

## Cómo ver el dashboard

Como el HTML es autocontenido y solo consume Plotly desde un CDN, puedes:

**Opción A — Abrirlo localmente con doble clic.** Funciona en cualquier navegador moderno con conexión a internet (para cargar Plotly desde el CDN).

**Opción B — Servidor local** (recomendado para desarrollo):

```bash
python -m http.server 8000
```

Abre [http://localhost:8000/index.html](http://localhost:8000/index.html).

**Opción C — GitHub Pages.** En el repo: *Settings → Pages → Branch: `main` / `(root)` → Save*. En unos segundos estará público en `https://andrenany.github.io/Predicci-n-de-Churn/`.

## Dataset

[Telco Customer Churn — IBM Sample Data Set](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 filas × 21 columnas).
