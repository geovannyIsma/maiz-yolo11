"""Funciones del OE2: entorno, figuras de evidencia, entrenamiento y registro de corridas, y diagnóstico de los modelos."""
import json
import os
import platform
from pathlib import Path


def plataforma():
    if 'COLAB_RELEASE_TAG' in os.environ:   # Colab también tiene una carpeta /kaggle
        return 'colab'
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    return 'local'


def entorno():
    """Hardware y versiones del equipo actual."""
    import cv2
    import numpy as np
    import psutil
    import torch
    import ultralytics
    import albumentations
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor())
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {'plataforma': plataforma(), 'CPU': cpu, 'núcleos': os.cpu_count(),
            'RAM (GB)': round(psutil.virtual_memory().total / 1e9, 1),
            'GPU': gpu.name if gpu else 'sin GPU', 'VRAM (GB)': round(gpu.total_memory / 1e9, 1) if gpu else 0.0,
            'Python': platform.python_version(), 'PyTorch': torch.__version__, 'CUDA': torch.version.cuda,
            'Ultralytics': ultralytics.__version__, 'Albumentations': albumentations.__version__,
            'OpenCV': cv2.__version__, 'NumPy': np.__version__}


def registrar_entorno(carpeta):
    """Guarda el entorno actual en carpeta/entorno_<plataforma>.json y lo devuelve."""
    info = entorno()
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / f"entorno_{info['plataforma']}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    return info


# =============================================================================
# Figuras de evidencia (Actividad 2.1)
# =============================================================================
import sys as _sys

RAIZ = Path(__file__).resolve().parent        # oe2/
_OE1 = RAIZ.parent / 'oe1'
_sys.path.insert(0, str(_OE1))
import comun as C   # noqa: E402  (módulo del OE1: rutas del repositorio, lectura de etiquetas y estilo de gráficas)

REPO = C.REPO
RUNS = REPO / 'runs' / 'oe2'
TRABAJO = REPO / 'trabajo' / 'oe2'
FOTOS = REPO / 'fotos'
REGISTRO = RAIZ / 'resultados' / 'experimentos'   # un CSV por fase: cribado, confirmacion, final, fondos
FINAL = C.DATOS / 'Data_final'
NOMBRE = {0: 'Broken', 1: 'Damage', 2: 'Fungus', 3: 'Healthy', 4: 'Immature', 8: 'Shriveled', 9: 'Weeveled'}
LOTE_CAPTURA = {s: v['lote'] for s, v in json.loads((_OE1 / 'resultados' / 'particion.json').read_text())['imagenes'].items()}
stems = sorted(p.stem for p in (FINAL / 'train' / 'images').glob('*.jpg'))
SUBTIPOS = {s: C.leer_yolo(FINAL / 'train' / 'labels_subtipo' / f'{s}.txt')[1] for s in stems}

# Borde verde: se mantiene o se prueba; rojo: se descarta
OK, MAL, NEUTRO, GRIS = '#1E8449', '#C0392B', '#2C3E50', '#6B7280'
CAJA_OK, CAJA_MAL, CAJA_B = (46, 204, 113), (231, 76, 60), (52, 152, 219)


def mostrar(ax, img, titulo, nota=None, estado=None):
    """Panel de imagen con borde de color según la decisión ('ok', 'mal' o None)."""
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    color = {'ok': OK, 'mal': MAL}.get(estado, NEUTRO)
    for borde in ax.spines.values():
        borde.set_visible(True); borde.set_edgecolor(color); borde.set_linewidth(4 if estado else 1)
    ax.set_title(titulo, fontsize=11.5, color=color, fontweight='bold', pad=6)
    if nota:
        ax.set_xlabel(nota, fontsize=9.5, color=GRIS, labelpad=5)


def cargar(stem, split='train'):
    """Imagen RGB, cajas en píxeles (x1, y1, x2, y2), clases binarias y subtipos."""
    import cv2
    import numpy as np
    img = cv2.cvtColor(cv2.imread(str(FINAL / split / 'images' / f'{stem}.jpg')), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    c, cl = C.leer_yolo(FINAL / split / 'labels' / f'{stem}.txt')
    _, sub = C.leer_yolo(FINAL / split / 'labels_subtipo' / f'{stem}.txt')
    c = np.array(c, dtype=float).reshape(-1, 4)
    xyxy = np.stack([(c[:, 0] - c[:, 2] / 2) * W, (c[:, 1] - c[:, 3] / 2) * H,
                     (c[:, 0] + c[:, 2] / 2) * W, (c[:, 1] + c[:, 3] / 2) * H], 1)
    return img, xyxy, np.array(cl), np.array(sub)


def dibujar(img, cajas, color, grosor=2):
    import cv2
    import numpy as np
    img = img.copy()
    for x1, y1, x2, y2 in np.round(cajas).astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)
    return img


def recorte(img, centro, lado, salida=None):
    """Recorte cuadrado de `lado` px centrado en `centro` (redimensionado a `salida` px) y su esquina superior izquierda."""
    import cv2
    import numpy as np
    x, y = (int(v) for v in centro); h = lado // 2
    x = int(np.clip(x, h, img.shape[1] - h)); y = int(np.clip(y, h, img.shape[0] - h))
    r = img[y - h:y + h, x - h:x + h]
    return (cv2.resize(r, (salida, salida), interpolation=cv2.INTER_CUBIC) if salida else r), (x - h, y - h)


def grano(subtipo, k=0, lado=224, margen=1.5):
    """Recorte cuadrado de un grano grande del subtipo pedido (k elige una imagen de otro lote) y su caja en el recorte."""
    import numpy as np
    candidatas = sorted((s for s in stems if subtipo in SUBTIPOS[s]), key=lambda s: -list(SUBTIPOS[s]).count(subtipo))
    vistos, elegidas = set(), []
    for s in candidatas:
        if LOTE_CAPTURA[s] not in vistos:
            vistos.add(LOTE_CAPTURA[s]); elegidas.append(s)
    s = elegidas[k % len(elegidas)]
    img, xyxy, _, sub = cargar(s)
    b = xyxy[sub == subtipo]; b = b[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))]
    L = int(max(b[2] - b[0], b[3] - b[1]) * margen)
    r, (ox, oy) = recorte(img, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), L, lado)
    return r, s, (b - [ox, oy, ox, oy]) * lado / L


def hsv(img, h=1.0, s=1.0, v=1.0):
    """Misma operación que RandomHSV de Ultralytics, con factores fijos (extremos de 1 ± ganancia)."""
    import cv2
    import numpy as np
    x = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    x[..., 0] = (x[..., 0] * h) % 180; x[..., 1] = np.clip(x[..., 1] * s, 0, 255); x[..., 2] = np.clip(x[..., 2] * v, 0, 255)
    return cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_HSV2RGB)


def ejemplos_ultralytics(hyp, indices=(2, 9, 16, 23), semilla=1):
    """Imágenes de entrenamiento generadas por el flujo de aumento de Ultralytics, con las cajas dibujadas
    (verde: Healthy; rojo: Defect). Usa una copia temporal de una parte de train para no escribir cachés en Data_final."""
    import random
    import shutil
    import tempfile
    import numpy as np
    from ultralytics.data.dataset import YOLODataset
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); (tmp / 'images').mkdir(); (tmp / 'labels').mkdir()
        for s in stems[::12]:
            shutil.copy(FINAL / 'train' / 'images' / f'{s}.jpg', tmp / 'images')
            shutil.copy(FINAL / 'train' / 'labels' / f'{s}.txt', tmp / 'labels')
        ds = YOLODataset(img_path=str(tmp / 'images'), imgsz=1280, augment=True, hyp=hyp, batch_size=4, stride=32,
                         data={'names': {0: 'Healthy', 1: 'Defect'}, 'nc': 2, 'channels': 3})
        np.random.seed(semilla); random.seed(semilla)
        for i in range(len(ds)):   # se recorren todas las imágenes antes de tomar los ejemplos
            ds[i]
        salida = []
        for i in indices:
            e = ds[i]
            img = np.ascontiguousarray(e['img'].permute(1, 2, 0).numpy())
            H, W = img.shape[:2]
            bx = e['bboxes'].numpy() * [W, H, W, H]
            xyxy = np.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2, bx[:, 0] + bx[:, 2] / 2, bx[:, 1] + bx[:, 3] / 2], 1)
            cl = e['cls'].numpy().ravel()
            salida.append((dibujar(dibujar(img, xyxy[cl == 0], CAJA_OK, 4), xyxy[cl == 1], CAJA_MAL, 4), len(cl)))
    return salida


# =============================================================================
# Entrenamiento y ajuste (Actividad 2.2)
# =============================================================================
MODELOS = ('yolo11n', 'yolo11s')
IMGSZ = 1280
OPTIMIZADORES = {'SGD': 0.01, 'MuSGD': 0.01, 'Adam': 0.001, 'AdamW': 0.001}   # tasa estándar de cada optimizador
REFERENCIA = {'optimizer': 'SGD', 'lr0': 0.01, 'batch': 8, 'aumento': 'A'}       # protocolo del OE1 (el cribado usa lote 8)
LOTE = 16                                                                       # lote por defecto de Ultralytics: confirmación y final
GANANCIA_MIN = 0.005                                                            # mejora mínima del mAP50-95 medio
ALFA = 0.05
EPOCAS_AJUSTE, EPOCAS_FINAL, PACIENCIA_FINAL = 100, 200, 50
SEMILLAS_CRIBADO = (13, 29, 51)   # cada configuración del cribado se entrena con 3 semillas y se compara por su media
SEMILLAS_CONFIRMACION = (13, 29)  # la confirmación se amplió de una semilla (13) a dos: 4 pliegues × 2 semillas = 8 comparaciones
SEMILLAS_FINAL = (42, 7, 28)
CONF_APLICACION = 0.25        # umbral para las métricas de aplicación durante el ajuste (el final usa el de máximo F1)
WORKERS = 4
_PESOS_LOCALES = C.MODELOS


def aumento_albumentations():
    """Conjunto propio (2.1_aumento_datos.ipynb): una transformación por familia, con probabilidad baja."""
    import albumentations as A
    return [A.OneOf([A.RandomGamma(gamma_limit=(75, 125)), A.RandomToneCurve(scale=0.25)], p=0.35),
            A.OneOf([A.Sharpen(alpha=(0.25, 0.40), lightness=(0.95, 1.05)), A.Defocus(radius=(2, 4))], p=0.40),
            A.GaussNoise(std_range=(0.06, 0.12), p=0.20),
            A.ImageCompression(quality_range=(80, 95), p=0.15)]


def aumento_geometrico():
    import albumentations as A
    return [A.RandomRotate90(p=0.5)]


def args_aumento(configuracion):
    """Argumentos de Ultralytics de cada configuración de aumento: A (por defecto), B (+ Albumentations), C (B + geometría exacta)."""
    if configuracion == 'A':
        return {}
    if configuracion == 'B':
        return {'augmentations': aumento_albumentations()}
    if configuracion == 'C':
        return {'augmentations': aumento_albumentations() + aumento_geometrico(), 'flipud': 0.5}
    raise ValueError(configuracion)


def id_corrida(modelo, cfg, semilla, particion):
    return f"{modelo}_{cfg['optimizer']}_lr{cfg['lr0']:g}_b{cfg['batch']}_aug{cfg['aumento']}_s{semilla}_{particion}"


def _imagenes(split):
    return sorted((FINAL / split / 'images').glob('*.jpg'))


def datos_particion(particion, trabajo):
    """data.yaml e imágenes de validación de una partición:
    'valid' y 'final' usan Data_final (train / valid = pliegue 2); 'fondos', Data_final más los fondos negativos;
    'p0' a 'p3' son pliegues de la validación cruzada del OE1."""
    if particion in ('valid', 'final'):
        return FINAL / 'data.yaml', _imagenes('valid')
    if particion == 'fondos':
        return FONDOS / 'data.yaml', _imagenes('valid')
    k = int(particion[1:])
    pliegue = {s: v['pliegue'] for s, v in json.loads((_OE1 / 'resultados' / 'particion.json').read_text())['imagenes'].items()
               if v['conjunto'] == 'dev'}
    dev = _imagenes('train') + _imagenes('valid')
    train = [p for p in dev if pliegue[p.stem] != k]
    val = [p for p in dev if pliegue[p.stem] == k]
    carpeta = Path(trabajo) / particion
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / 'train.txt').write_text('\n'.join(map(str, train)) + '\n')
    (carpeta / 'val.txt').write_text('\n'.join(map(str, val)) + '\n')
    (carpeta / 'data.yaml').write_text(f'path: {carpeta}\ntrain: train.txt\nval: val.txt\nnc: 2\nnames: ["Healthy", "Defect"]\n')
    return carpeta / 'data.yaml', val


def leer_registro(registro=REGISTRO):
    """Registro de corridas: combina los CSV de todas las fases de la carpeta `registro`, sin duplicar corridas."""
    import pandas as pd
    archivos = sorted(Path(registro).glob('*.csv'))
    if not archivos:
        return pd.DataFrame(columns=['id'])
    return pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True).drop_duplicates('id', keep='last').reset_index(drop=True)


def anotar_registro(registro, fase, filas):
    """Añade filas al CSV de su fase (`registro/<fase>.csv`), sustituyendo las corridas con el mismo id."""
    import pandas as pd
    archivo = Path(registro) / f'{fase}.csv'
    archivo.parent.mkdir(parents=True, exist_ok=True)
    previo = pd.read_csv(archivo) if archivo.exists() else pd.DataFrame(columns=['id'])
    nuevo = pd.DataFrame(filas)
    pd.concat([previo[~previo['id'].isin(nuevo['id'])], nuevo], ignore_index=True).to_csv(archivo, index=False)


def integrar_descarga(zip_descarga, registro=REGISTRO):
    """Incorpora una descarga de Colab (`oe2_<fase>*.zip`): copia sus corridas a runs/oe2/, añade sus filas al CSV
    de su fase sin duplicar y copia el registro del entorno. Devuelve los id incorporados."""
    import shutil
    import tempfile
    import zipfile
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zip_descarga).extractall(tmp)
        tmp = Path(tmp)
        for carpeta in sorted((tmp / 'runs' / 'oe2').glob('*')):
            destino = RUNS / carpeta.name
            shutil.rmtree(destino, ignore_errors=True)
            shutil.copytree(carpeta, destino)
        for f in (tmp / 'oe2' / 'resultados').glob('entorno_*.json'):   # hardware y versiones de la nube
            shutil.copy(f, RAIZ / 'resultados' / f.name)
        ids = []
        for csv in sorted((tmp / 'oe2' / 'resultados' / 'experimentos').glob('*.csv')):
            filas = pd.read_csv(csv)
            anotar_registro(registro, csv.stem, filas.to_dict('records'))
            ids += list(filas['id'])
    return ids


def entrenar(modelo, cfg, semilla, particion, fase, runs, registro, trabajo):
    """Entrena una corrida, la evalúa en su validación y añade una fila al registro. Conserva solo best.pt."""
    import shutil
    import time
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import torch
    from ultralytics import YOLO

    rid = id_corrida(modelo, cfg, semilla, particion)
    data_yaml, val = datos_particion(particion, trabajo)
    final = particion in ('final', 'fondos')
    epocas, paciencia = (EPOCAS_FINAL, PACIENCIA_FINAL) if final else (EPOCAS_AJUSTE, EPOCAS_AJUSTE)
    pesos = _PESOS_LOCALES / f'{modelo}.pt'
    kwargs = dict(data=str(data_yaml), imgsz=IMGSZ, epochs=epocas, patience=paciencia, batch=cfg['batch'],
                  optimizer=cfg['optimizer'], lr0=cfg['lr0'], seed=semilla, deterministic=True, workers=WORKERS,
                  project=str(runs), name=rid, exist_ok=True, plots=True, verbose=False, **args_aumento(cfg['aumento']))
    if cfg['optimizer'] in ('Adam', 'AdamW'):
        kwargs['warmup_bias_lr'] = 0.0   # como hace Ultralytics en modo 'auto' con AdamW: una tasa de sesgo de 0.1 es alta para Adam
    carpeta = Path(runs) / rid
    shutil.rmtree(carpeta, ignore_errors=True)   # restos de una corrida interrumpida: Ultralytics añadiría filas a su results.csv
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m = YOLO(str(pesos) if pesos.exists() else f'{modelo}.pt')
    m.train(**kwargs)
    minutos = (time.time() - t0) / 60
    curvas = pd.read_csv(carpeta / 'results.csv'); curvas.columns = [c.strip() for c in curvas.columns]
    mejor = YOLO(str(carpeta / 'weights' / 'best.pt'))
    data_val = FINAL / 'data.yaml' if particion == 'fondos' else data_yaml   # métricas sobre los granos de valid, comparables con la 2.2
    v = mejor.val(data=str(data_val), split='val', imgsz=IMGSZ, batch=cfg['batch'], plots=False, verbose=False,
                  project=str(runs), name=f'{rid}_val', exist_ok=True)
    from ultralytics.utils.metrics import smooth
    f1 = smooth(np.asarray(v.box.f1_curve).mean(0), 0.1)   # mismo suavizado que Ultralytics para el máximo F1
    conf_f1 = float(np.asarray(v.box.px)[int(np.argmax(f1))])
    fila = {'id': rid, 'fase': fase, 'modelo': modelo, **cfg, 'semilla': semilla, 'particion': particion,
            'epocas': epocas, 'paciencia': paciencia, 'epocas_ejecutadas': len(curvas),
            'mejor_epoca': int(curvas['metrics/mAP50-95(B)'].idxmax()) + 1,
            'mAP50_95': float(v.box.map), 'mAP50': float(v.box.map50), 'mAP75': float(v.box.map75),
            'precision': float(v.box.mp), 'recall': float(v.box.mr),
            'AP50_95_Healthy': float(v.box.ap[0]), 'AP50_95_Defect': float(v.box.ap[1]),
            'conf_f1': conf_f1, 'f1_max': float(f1.max()),
            **C.metricas_aplicacion(mejor, val, IMGSZ, conf_f1 if final else CONF_APLICACION),
            'minutos': round(minutos, 1),
            'vram_gb': round(torch.cuda.max_memory_reserved() / 1e9, 2) if torch.cuda.is_available() else 0.0,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'plataforma': plataforma(), 'fecha': datetime.now().isoformat(timespec='seconds')}
    shutil.rmtree(Path(runs) / f'{rid}_val', ignore_errors=True)
    (carpeta / 'weights' / 'last.pt').unlink(missing_ok=True)
    anotar_registro(registro, fase, [fila])
    C.liberar_gpu(m, mejor)
    del m, mejor
    C.liberar_gpu()
    return fila


def obtener_map(registro, modelo, cfg, semilla, particion, entrenar_si_falta=None):
    """mAP50-95 de una corrida del registro; si falta y se pasa `entrenar_si_falta`, la entrena primero."""
    reg = leer_registro(registro)
    rid = id_corrida(modelo, cfg, semilla, particion)
    fila = reg[reg['id'] == rid]
    if len(fila):
        return float(fila['mAP50_95'].iloc[0])
    if entrenar_si_falta is None:
        return None
    return entrenar_si_falta(modelo, cfg, semilla, particion)['mAP50_95']


def cribado(modelo, registro, entrenar_si_falta=None):
    """Cribado secuencial en train / valid: optimizador -> tasa de aprendizaje -> aumento. Cada configuración se
    entrena con las SEMILLAS_CRIBADO y se compara por su mAP50-95 medio. En cada paso se adopta la mejor alternativa
    si supera a la base en GANANCIA_MIN; si no, se mantiene la base.
    Devuelve (tabla por configuración, configuración ganadora o None si falta alguna corrida)."""
    import numpy as np
    import pandas as pd
    def medir(cfg):
        valores = [obtener_map(registro, modelo, cfg, s, 'valid', entrenar_si_falta) for s in SEMILLAS_CRIBADO]
        completo = all(v is not None for v in valores)
        return (float(np.mean(valores)) if completo else None), valores
    def fila(paso, cfg, media, valores, elegida):
        return {'paso': paso, **cfg, 'mAP50_95': media, 'sd': float(np.std([v for v in valores if v is not None], ddof=1)) if all(v is not None for v in valores) else None,
                **{f'semilla {s}': v for s, v in zip(SEMILLAS_CRIBADO, valores)}, 'elegida': elegida}
    base = dict(REFERENCIA)
    m_base, val_base = medir(base)
    filas = [fila('Referencia', base, m_base, val_base, True)]
    pasos = [('Optimizador', lambda b: [dict(b, optimizer=o, lr0=lr) for o, lr in OPTIMIZADORES.items() if o != b['optimizer']]),
             ('Tasa de aprendizaje', lambda b: [dict(b, lr0=round(OPTIMIZADORES[b['optimizer']] * f, 6)) for f in (0.5, 2)]),
             ('Aumento', lambda b: [dict(b, aumento=a) for a in ('B', 'C')])]
    for nombre, alternativas in pasos:
        res = [(c, *medir(c)) for c in alternativas(base)]
        completo = m_base is not None and all(m is not None for _, m, _ in res)
        nuevo, m_nuevo = base, m_base
        if completo:
            c, m, _ = max(res, key=lambda t: t[1])
            if m - m_base >= GANANCIA_MIN:
                nuevo, m_nuevo = c, m
        filas += [fila(nombre, c, m, v, completo and c == nuevo) for c, m, v in res]
        if not completo:
            return pd.DataFrame(filas), None
        base, m_base = nuevo, m_nuevo
    return pd.DataFrame(filas), base


def confirmacion(modelo, candidata, registro, entrenar_si_falta=None):
    """Validación cruzada de 4 pliegues con 2 semillas y el lote por defecto (LOTE): candidata frente a la referencia,
    emparejadas por pliegue y semilla (8 comparaciones), con corridas propias. Se adopta la candidata si mejora al menos
    GANANCIA_MIN de media con p < ALFA (t pareada). Devuelve (tabla de comparaciones, configuración adoptada con LOTE
    o None si falta alguna corrida)."""
    import pandas as pd
    from scipy import stats
    ref, cand = dict(REFERENCIA, batch=LOTE), dict(candidata, batch=LOTE)
    if cand == ref:
        return pd.DataFrame(), ref
    filas = []
    for semilla in SEMILLAS_CONFIRMACION:
        for p in ('p0', 'p1', 'p2', 'p3'):
            r = obtener_map(registro, modelo, ref, semilla, p, entrenar_si_falta)
            c = obtener_map(registro, modelo, cand, semilla, p, entrenar_si_falta)
            filas.append({'pliegue': int(p[1:]), 'semilla': semilla, 'referencia': r, 'candidata': c})
    t = pd.DataFrame(filas).set_index(['pliegue', 'semilla']).sort_index()
    if t.isna().any().any():
        return t, None
    d = t['candidata'] - t['referencia']
    p = stats.ttest_rel(t['candidata'], t['referencia']).pvalue
    t.attrs.update({'diferencia': float(d.mean()), 'valor_p': float(p), 'a_favor': int((d > 0).sum())})
    return t, (cand if d.mean() >= GANANCIA_MIN and p < ALFA else ref)


# =============================================================================
# Diagnóstico de los modelos finales (test e impurezas)
# =============================================================================
import numpy as np   # noqa: E402

IOU_ACIERTO = 0.5   # un grano está detectado si una predicción lo cubre con IoU >= 0.5 (criterio de mAP50)
IOU_FONDO = 0.1     # una predicción sobrante con IoU < 0.1 con todo grano está sobre el fondo o sobre un objeto no anotado
IMPUREZAS = {5: 'Inorganic', 6: 'Organic'}   # 'Other Edible Seeds' (7) queda fuera: se descartó en el OE1 por salir del objetivo (maíz)
_DATA_CORR = C.DATOS / 'Data_corr_v6'
FONDOS = C.DATOS / 'fondos'


def letterbox(img):
    """Mismo letterbox que Data_final: lado mayor a 1280 px (INTER_AREA) y relleno gris hasta un múltiplo de 32.
    Devuelve la imagen, la escala y el desplazamiento (x, y) del relleno."""
    import cv2
    redim, _ = C._letterbox(IMGSZ)
    red = redim(image=img)['image']
    H, W = red.shape[:2]
    ph, pw = -H % C.PAD_DIVISOR, -W % C.PAD_DIVISOR
    out = cv2.copyMakeBorder(red, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_CONSTANT, value=C.PAD_COLOR)
    return out, W / img.shape[1], (pw // 2, ph // 2)


def configuracion(nombre='final'):
    """Configuración adoptada (`resultados/configuracion_<nombre>.json`: 'final' de la 2.2 o 'fondos'): por modelo,
    hiperparámetros, umbral de conteo y rutas absolutas de sus best.pt."""
    cfg = json.loads((RAIZ / 'resultados' / f'configuracion_{nombre}.json').read_text())
    for m in cfg.values():
        m['pesos'] = [REPO / p for p in m['pesos']]
    return cfg


def config_final():
    return configuracion('final')


def predecir(pesos, imagenes, conf, lote=8):
    """Predicciones con NMS agnóstico a la clase (un grano, una caja): stem -> (cajas xyxy, clases, confianzas)."""
    from ultralytics import YOLO
    modelo, salida = YOLO(str(pesos)), {}
    for i in range(0, len(imagenes), lote):
        for r in modelo.predict([str(p) for p in imagenes[i:i + lote]], imgsz=IMGSZ, conf=conf,
                                agnostic_nms=True, verbose=False):
            b = r.boxes
            salida[Path(r.path).stem] = (b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy())
    C.liberar_gpu(modelo)
    return salida


def emparejar(gt, pred):
    """Emparejamiento uno a uno por IoU descendente (IoU >= IOU_ACIERTO): {índice gt: índice pred} y la matriz de IoU."""
    m = C.iou(gt, pred)
    pares, usadas = {}, set()
    for g, q in sorted(zip(*np.where(m >= IOU_ACIERTO)), key=lambda t: -m[t]):
        if g not in pares and q not in usadas:
            pares[g] = q; usadas.add(q)
    return pares, m


def diagnosticar(predicciones, split='test'):
    """Desglose por grano y por predicción sobrante de un split de Data_final.
    granos: estado 'acierto', 'clase cambiada' o 'no detectado'.
    sobrantes: predicciones sin grano emparejado; 'sobre un grano' (duplicado o caja mal ajustada) o 'fondo'."""
    import pandas as pd
    granos, sobrantes = [], []
    for stem, (pb, pc, pf) in predicciones.items():
        _, gt, cl, sub = cargar(stem, split)
        pares, m = emparejar(gt, pb)
        for g in range(len(gt)):
            q = pares.get(g)
            estado = 'no detectado' if q is None else 'acierto' if pc[q] == cl[g] else 'clase cambiada'
            granos.append({'stem': stem, 'lote': LOTE_CAPTURA[stem], 'grano': g, 'subtipo': NOMBRE[sub[g]], 'clase': int(cl[g]),
                           'estado': estado, 'clase_pred': None if q is None else int(pc[q]),
                           'conf': None if q is None else float(pf[q]), 'iou': None if q is None else float(m[g, q]),
                           'caja': gt[g]})
        for q in sorted(set(range(len(pb))) - set(pares.values())):
            iou_max = float(m[:, q].max()) if len(gt) else 0.0
            sobrantes.append({'stem': stem, 'lote': LOTE_CAPTURA[stem], 'clase_pred': int(pc[q]), 'conf': float(pf[q]),
                              'iou_max': iou_max, 'tipo': 'fondo' if iou_max < IOU_FONDO else 'sobre un grano', 'caja': pb[q]})
    return pd.DataFrame(granos), pd.DataFrame(sobrantes, columns=['stem', 'lote', 'clase_pred', 'conf', 'iou_max', 'tipo', 'caja'])


def imagenes_impurezas(destino):
    """Imágenes de Data_corr_v6 que solo contienen impurezas inorgánicas u orgánicas, retiradas en el OE1 y nunca vistas
    por los modelos (las de otras semillas no se usan). Se llevan a 1280 px con el mismo letterbox de Data_final. Devuelve las rutas y
    stem -> (cajas xyxy, clases originales)."""
    import cv2
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    rutas, anotaciones = [], {}
    for r in C.leer_dataset(_DATA_CORR):
        if not r['clases'] or not set(r['clases']) <= set(IMPUREZAS):
            continue
        out, esc, (dx, dy) = letterbox(cv2.imread(str(r['img'])))
        ruta = destino / f"{r['stem']}.jpg"
        if not ruta.exists():
            cv2.imwrite(str(ruta), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        anotaciones[r['stem']] = (r['xyxy'] * esc + [dx, dy, dx, dy], np.array(r['clases']))
        rutas.append(ruta)
    return rutas, anotaciones
