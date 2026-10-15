# Detección, clasificación y conteo de granos de maíz con YOLOv11

Trabajo de titulación: *Evaluación de YOLOv11n y YOLOv11s para la detección, clasificación y conteo de granos de maíz sanos y con defectos*.

Los modelos detectan cada grano y lo clasifican en dos clases:
- **`Healthy`:** granos sanos (`Healthy` e `Immature`);
- **`Defect`:** granos con defectos (`Broken`, `Damage`, `Fungus`, `Shriveled` y `Weeveled`).

El conteo se obtiene de las detecciones.

## Estructura

```
maiz-yolo11/
├── oe1/                 Objetivo específico 1: ingeniería de datos
│   ├── 1.x_*.ipynb      notebooks, en orden de ejecución
│   ├── comun.py         rutas del repositorio, lectura de datasets, partición, experimentos
│   ├── docs/            informe de cada actividad
│   ├── resultados/      partición por lote, registro de corridas, decisión
│   └── salida/          figuras de cada notebook
├── oe2/                 Objetivo específico 2: entrenamiento y validación de los modelos
│   ├── 2.x_*.ipynb      notebooks, en orden de ejecución
│   ├── comun_oe2.py     entrenamiento, registro de corridas, diagnóstico
│   ├── docs/            plan del OE2 e informe de cada actividad
│   ├── resultados/      experimentos/<fase>.csv, configuraciones adoptadas, entornos
│   └── salida/          figuras de cada notebook
├── antecedentes/        PIIC y figuras de la asignatura Proyectos Tecnológicos (julio de 2026)
├── fotos/               fotos propias (sin metadatos)
│   ├── hojas_vacias/    fondos sin granos, por sesión de captura
│   └── propias/         fotos de prueba con smartphone y su referencia
├── datos/               (local) datasets originales y derivados
├── runs/                (local) corridas de Ultralytics: oe1/, oe2/
├── modelos/             (local) pesos preentrenados en COCO (yolo11n.pt, yolo11s.pt)
├── nube/                (local) paquetes para entrenar en Google Colab
└── zenodo/              (local) zips publicados en Zenodo: datasets derivados y pesos
```

Git solo versiona el código, los notebooks, los informes, los resultados, las figuras y las fotos propias. Las carpetas marcadas como *(local)* no se suben.

## Entorno

- **Instalación:** Python 3.14 y [uv](https://docs.astral.sh/uv/). `uv sync` crea `.venv/` con las versiones exactas de `uv.lock`: PyTorch 2.14.1 con CUDA 13.2, Ultralytics 8.4.174 y Albumentations 2.0.8.
- **Local:** preparación de datos, análisis, figuras e inferencia, en una RTX 3060 Laptop de 6 GB.
- **Google Colab:** el entrenamiento a 1280 px, en una A100 de 40 GB.

## Datos

El dataset de partida es *Maize with prev dataset* (Roboflow Universe, licencia CC BY 4.0), re-anotado en varias versiones. Sus exportaciones se colocan en `datos/originales/`:

| Carpeta | Versión |
| :--- | :--- |
| `datos/originales/Data_original` | v0: exportación pública original |
| `datos/originales/Data` | v1: primera re-anotación |
| `datos/originales/Data_corr` | v2: curaduría de `Weeveled` |

Los notebooks generan el resto de `datos/`:
- `Data_corr_v6`: etiquetas corregidas;
- `Data_2c_*`: variantes del diseño factorial;
- `Data_final`: dataset final, 1280 px, partición por lote;
- `fondos`: fondos negativos.

`Data_corr_v6`, `Data_final`, `fondos` y los pesos de los modelos finales se publican en Zenodo, porque no caben en git: [10.5281/zenodo.23400166](https://doi.org/10.5281/zenodo.23400166). Los zips se generan en `zenodo/`.

## Orden de ejecución

| Notebook | Qué hace | Informe |
| :--- | :--- | :--- |
| `oe1/1.1_eda.ipynb` | Análisis exploratorio del dataset original | `oe1/docs/1.1_eda.md` |
| `oe1/1.2_limpieza_etiquetas.ipynb` | Corrección de etiquetas (v0 → v3) | `oe1/docs/1.2_limpieza_etiquetas.md` |
| `oe1/1.2_ingenieria_datos.ipynb` | Consolidación a 2 clases, partición por lote, variantes del diseño factorial | `oe1/docs/1.2_ingenieria_datos.md` |
| `oe1/1.2_experimentos.ipynb` | Diseño factorial (resolución × fotométrico × balanceo) en Colab y su análisis | `oe1/docs/1.2_ingenieria_datos.md` |
| `oe1/1.3_estructuracion_dataset.ipynb` | Construcción de `Data_final` | `oe1/docs/1.3_estructuracion_dataset.md` |
| `oe2/2.1_entorno_arquitecturas.ipynb` | Entorno, arquitecturas, transferencia y memoria | `oe2/docs/2.1_configuracion_entorno.md` |
| `oe2/2.1_hiperparametros.ipynb` | Análisis de los hiperparámetros con evidencias | `oe2/docs/2.1_configuracion_entorno.md` |
| `oe2/2.1_aumento_datos.ipynb` | Transformaciones de Albumentations y configuraciones de aumento | `oe2/docs/2.1_configuracion_entorno.md` |
| `oe2/2.2_entrenamiento.ipynb` | Cribado, confirmación y entrenamiento final en Colab, y su análisis | `oe2/docs/2.2_entrenamiento.md` |
| `oe2/2.2b_diagnostico_test.ipynb` | Diagnóstico de los modelos finales en el test y con impurezas | `oe2/docs/2.2b_diagnostico_test.md` |
| `oe2/2.2c_fondos_dataset.ipynb` | Dataset de fondos negativos | `oe2/docs/2.2c_fondos_negativos.md` |
| `oe2/2.2c_fondos_entrenamiento.ipynb` | Reentrenamiento con fondos en Colab y su análisis | `oe2/docs/2.2c_fondos_negativos.md` |
| `oe2/2.2d_fotos_propias.ipynb` | Comportamiento de los modelos con fotos propias | `oe2/docs/2.2d_fotos_propias.md` |

`oe2/docs/plan_oe2.md` resume el diseño del OE2 y las decisiones de cada etapa.

**Modelos vigentes:** `oe2/resultados/configuracion_fondos.json` contiene YOLOv11n y YOLOv11s entrenados con fondos negativos, 3 semillas cada uno. Incluye la configuración, el umbral de conteo y las rutas de sus `best.pt`.

## Entrenamiento en Google Colab

Los notebooks que entrenan (`1.2_experimentos`, `2.2_entrenamiento` y `2.2c_fondos_entrenamiento`) se usan así:
1. **En local**, su última sección genera un paquete en `nube/` con el código, los datos y los registros previos.
2. **El paquete se sube a Google Drive**, que solo se lee.
3. **En Colab** se ejecuta el mismo notebook con una A100. Las corridas ya registradas se omiten.
4. **Al terminar**, cada sesión descarga un zip con sus corridas y su registro. En el OE2 se incorpora en local con:

   ```python
   import comun_oe2 as E
   E.integrar_descarga('oe2_<fase>.zip')   # corridas a runs/oe2/, filas a oe2/resultados/experimentos/<fase>.csv
   ```
