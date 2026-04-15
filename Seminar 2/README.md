# Seminar 2 — Support Vector Machine (SVM)
**ML4Net · Universitat Pompeu Fabra**

Clasificación de poses humanas usando señales Wi-Fi (CSI) y un clasificador SVM.

---

## ¿De qué va el trabajo?

Tenemos mediciones reales de Wi-Fi capturadas mientras personas hacían poses (wave, push, crouch, sitdown, bend). El objetivo es entrenar un modelo SVM que, dado una medición Wi-Fi, prediga qué pose estaba haciendo la persona — **sin cámara, solo con la señal Wi-Fi**.

---

## Archivos del proyecto

| Archivo | Contenido |
|---|---|
| `Train_features.csv` | 1000 mediciones Wi-Fi (CSI) para entrenar |
| `Train_labels.csv` | 1000 etiquetas de pose (1–5) para entrenar |
| `Train_skelletonpoints.csv` | 1000 esqueletos capturados con cámara (solo visualización) |
| `Test_features.csv` | 200 mediciones Wi-Fi para evaluar |
| `Test_labels.csv` | 200 etiquetas reales para comparar con las predicciones |
| `Test_skelletonpoints.csv` | 200 esqueletos de test |
| `Seminar_2.ipynb` | Notebook con todo el código |

---

## Estructura de los datos

### Features (X) — `Train_features.csv`
Cada fila es una medición Wi-Fi. Tiene **270 columnas** que representan una matriz CSI aplanada de forma `30 × 3 × 3`:
- **30** = número de subcarriers (frecuencias Wi-Fi)
- **3 × 3** = 3 antenas transmisoras × 3 antenas receptoras

```
Una fila → [val1, val2, ..., val270]
              ↑
    H(subcarrier, antena_tx, antena_rx) = amplitud de la señal
```

### Labels (y) — `Train_labels.csv`
Cada valor es un entero del 1 al 5:

| Número | Pose |
|---|---|
| 1 | wave (saludar) |
| 2 | push (empujar) |
| 3 | crouch (agacharse) |
| 4 | sitdown (sentarse) |
| 5 | bend (inclinarse) |

### Skeleton points — `Train_skelletonpoints.csv`
Solo para visualización. Cada fila tiene **54 columnas**:
- Columnas 0–17 → coordenada X de 18 puntos del cuerpo
- Columnas 18–35 → coordenada Y de 18 puntos del cuerpo
- Columnas 36–53 → confianza de cada punto

---

## Paso 1 — Carga de datos

Lo primero es cargar todos los CSV en variables numpy.

```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Features: 1000 muestras x 270 columnas
X_train = pd.read_csv('Train_features.csv', header=None).values
X_test  = pd.read_csv('Test_features.csv',  header=None).values

# Labels: enteros del 1 al 5
y_train = pd.read_csv('Train_labels.csv', header=None).values.flatten()
y_test  = pd.read_csv('Test_labels.csv',  header=None).values.flatten()

# Skeleton points
skel_train = pd.read_csv('Train_skelletonpoints.csv', header=None).values
skel_test  = pd.read_csv('Test_skelletonpoints.csv',  header=None).values

POSE_NAMES = {1: 'wave', 2: 'push', 3: 'crouch', 4: 'sitdown', 5: 'bend'}
```

**Por qué `header=None`**: los CSV no tienen fila de cabecera, son datos directamente.  
**Por qué `.values`**: convierte el DataFrame de pandas a array numpy (más rápido para cálculos).  
**Por qué `.flatten()`**: los labels están en una sola fila, `flatten()` los convierte a un array 1D.

---

## Paso 2 — Visualización del esqueleto

Los 18 puntos del cuerpo siguen el estándar **MPII**. Los conectamos con líneas para dibujar el esqueleto.

```python
CONNECTIONS = [
    (0, 1),   # Nose - Neck
    (1, 2),   # Neck - Right Shoulder
    (2, 3),   # Right Shoulder - Right Elbow
    (3, 4),   # Right Elbow - Right Wrist
    (1, 5),   # Neck - Left Shoulder
    (5, 6),   # Left Shoulder - Left Elbow
    (6, 7),   # Left Elbow - Left Wrist
    (1, 8),   # Neck - Right Hip
    (8, 9),   # Right Hip - Right Knee
    (9, 10),  # Right Knee - Right Ankle
    (1, 11),  # Neck - Left Hip
    (11, 12), # Left Hip - Left Knee
    (12, 13), # Left Knee - Left Ankle
    (0, 14),  # Nose - Right Eye
    (0, 15),  # Nose - Left Eye
    (14, 16), # Right Eye - Right Ear
    (15, 17), # Left Eye - Left Ear
    (2, 5),   # Right Shoulder - Left Shoulder
    (8, 11),  # Right Hip - Left Hip
]

def plot_skeleton(skeleton_row, label, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))

    x_coords = skeleton_row[:18]   # columnas 0-17
    y_coords = skeleton_row[18:36] # columnas 18-35

    # Dibujar líneas entre keypoints conectados
    for (i, j) in CONNECTIONS:
        ax.plot([y_coords[i], y_coords[j]],
                [x_coords[i], x_coords[j]], 'b-', linewidth=1.5)

    # Dibujar puntos encima
    ax.scatter(y_coords, x_coords, c='red', s=30, zorder=3)
    ax.set_title(f'Pose = {POSE_NAMES[label]}', fontsize=13)
    ax.invert_yaxis()  # invertir eje Y para que la cabeza quede arriba
```

**Por qué `invert_yaxis()`**: las coordenadas Y en imagen crecen hacia abajo, pero en matplotlib crecen hacia arriba. Hay que invertirlo para que el esqueleto salga en la posición correcta.

```python
# Mostrar un ejemplo de cada pose
fig, axes = plt.subplots(1, 5, figsize=(18, 6))
for pose_id, ax in zip(range(1, 6), axes):
    idx = np.where(y_train == pose_id)[0][0]  # primer ejemplo de esa clase
    plot_skeleton(skel_train[idx], y_train[idx], ax=ax)
plt.tight_layout()
plt.show()
```

---

## Paso 3 — Amplitud CSI media por subcarrier

Para entender los datos Wi-Fi, visualizamos la amplitud media de la señal en cada subcarrier. Esto muestra cómo varía el canal en distintas frecuencias.

```python
def plot_csi_amplitude(features_row, label, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Paso clave: reshape de (270,) a (30, 3, 3)
    csi_matrix = features_row.reshape(30, 3, 3)

    # Media sobre las 9 combinaciones de antenas → shape (30,)
    csi_mean = csi_matrix.mean(axis=(1, 2))

    ax.plot(range(1, 31), csi_mean, marker='o', markersize=4)
    ax.set_title(f'Amplitud CSI media — Pose: {POSE_NAMES[label]}')
    ax.set_xlabel('Subcarrier')
    ax.set_ylabel('Amplitud media')
    ax.grid(True, alpha=0.3)
```

**Por qué `reshape(30, 3, 3)`**: los 270 valores son en realidad una matriz tridimensional. Al hacer reshape podemos operar sobre cada dimensión por separado.  
**Por qué `mean(axis=(1, 2))`**: promediamos sobre las dimensiones de antenas (TX y RX) y nos quedamos solo con la variación por subcarrier.

---

## Paso 4 — Distribución de amplitud CSI por clase

Visualizamos si las diferentes poses producen patrones CSI distintos. Si las distribuciones son separables, el SVM podrá clasificar bien.

```python
# Boxplot: un box por subcarrier para cada clase
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

for pose_id, ax, color in zip(range(1, 6), axes, colors):
    mask = y_train == pose_id          # seleccionar solo las muestras de esa clase
    samples = X_train[mask]            # (n_samples, 270)

    # Reshape y media sobre antenas → (n_samples, 30)
    csi_reshaped = samples.reshape(len(samples), 30, 9).mean(axis=2)

    ax.boxplot(csi_reshaped, positions=range(1, 31), patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.6))
    ax.set_title(f'{POSE_NAMES[pose_id]} (n={mask.sum()})')
    ax.set_xlabel('Subcarrier')

plt.tight_layout()
plt.show()
```

**Por qué boxplot**: muestra la mediana, percentiles y outliers de cada subcarrier. Si los boxplots de distintas clases tienen medianas diferentes en algún subcarrier, esa frecuencia es informativa para el clasificador.

```python
# Vista superpuesta: media ± desviación estándar de cada clase
for pose_id, color in zip(range(1, 6), colors):
    mask = y_train == pose_id
    samples = X_train[mask].reshape(mask.sum(), 30, 9).mean(axis=2)
    mean_per_sub = samples.mean(axis=0)  # media entre muestras de la misma clase
    std_per_sub  = samples.std(axis=0)   # variabilidad

    ax.plot(range(1, 31), mean_per_sub, color=color, label=POSE_NAMES[pose_id])
    ax.fill_between(range(1, 31),
                    mean_per_sub - std_per_sub,
                    mean_per_sub + std_per_sub,
                    color=color, alpha=0.15)  # banda de incertidumbre
```

**Por qué `fill_between`**: la banda sombreada representa la variabilidad de cada clase. Si las bandas de distintas clases se solapan mucho, el SVM lo tendrá difícil.

---

## Paso 5 — Entrenar el SVM

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Normalizar los datos (importante para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Crear y entrenar el modelo
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train_scaled, y_train)
```

**Por qué `StandardScaler`**: el SVM es sensible a la escala de los datos. Si una columna tiene valores entre 0–1 y otra entre 0–1000, la segunda dominará. El scaler centra cada columna en 0 con desviación estándar 1.  
**Por qué `fit_transform` en train y solo `transform` en test**: el scaler aprende la media y std del train. Aplicamos esos mismos valores al test para no "contaminar" la evaluación con información del test.  
**Por qué `kernel='rbf'`**: el kernel RBF (Radial Basis Function) transforma los datos a un espacio de mayor dimensión donde pueden ser separables linealmente.

---

## Paso 6 — Predecir y evaluar

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Predicciones
y_pred = model.predict(X_test_scaled)

# Accuracy global
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2%}')

# Reporte por clase
print(classification_report(y_test, y_pred, target_names=list(POSE_NAMES.values())))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=POSE_NAMES.values(),
            yticklabels=POSE_NAMES.values())
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de confusión')
plt.show()
```

**Qué mide cada métrica**:
- **Accuracy**: % de predicciones correctas sobre el total
- **Precision**: de las veces que predije clase X, ¿cuántas eran realmente X?
- **Recall**: de todas las muestras reales de clase X, ¿cuántas detecté?
- **F1-score**: media armónica de precision y recall (útil si las clases están desbalanceadas)
- **Matriz de confusión**: muestra dónde se confunde el modelo (qué clases mezcla)

---

## Resumen del flujo completo

```
Train_features.csv ──┐
Train_labels.csv   ──┤──► Entrenar SVM ──► modelo entrenado
                      │                          │
Test_features.csv  ──┼──────────────────► Predecir poses
Test_labels.csv    ──┘                          │
                                                 ▼
                                          Evaluar (accuracy,
                                          F1, matriz confusión)
```

---

## Requisitos

```bash
pip install pandas numpy matplotlib scikit-learn seaborn
```
