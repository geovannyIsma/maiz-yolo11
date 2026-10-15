"""Funciones comunes del OE1 (EDA, limpieza, ingeniería de datos y experimentos).

Secciones:
    1. Clases y estilo de las gráficas
    2. Lectura de datasets, sesiones y lotes de captura
    3. Calidad del etiquetado (detección y comparación de versiones)
    4. Partición por lote y pliegues de validación cruzada
    5. Preprocesamiento (letterbox y normalización fotométrica)
    6. Balanceo con transformaciones que no repiten el aumento online
    7. Experimentos (preparación de pliegues, entrenamiento, métricas)
    8. Análisis del diseño factorial 2^3
"""
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent        # oe1/
REPO = RAIZ.parent
DATOS = REPO / 'datos'                          # datasets originales y generados (solo en local)
ORIGINALES = DATOS / 'originales'               # exportaciones de Roboflow: Data_original, Data, Data_corr
RUNS = REPO / 'runs' / 'oe1'
TRABAJO = REPO / 'trabajo' / 'oe1'
NUBE = REPO / 'nube'
MODELOS = REPO / 'modelos'                      # pesos preentrenados en COCO

# =============================================================================
# 1. CLASES Y ESTILO
# =============================================================================
CLASES = ['Broken', 'Damage', 'Fungus', 'Healthy', 'Immature', 'Inorganic foreign material',
          'Organic foreign material', 'Other Edible Seeds', 'Shriveled', 'Weeveled']
CLASES_MAIZ = {0, 1, 2, 3, 4, 8, 9}
MAPEO_BINARIO = {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 8: 1, 9: 1}          # 0 = Healthy, 1 = Defect
CLASES_BINARIAS = ['Healthy', 'Defect']
SUBTIPOS = [0, 1, 2, 3, 4, 8, 9]
SPLITS = ('train', 'valid', 'test')

COLOR_SPLIT = {'train': '#4A90D9', 'valid': '#50C878', 'test': '#FF6B6B', 'dev': '#4A90D9'}
COLOR_CATEGORIA = {'Defecto': '#FF6B6B', 'Sano': '#50C878', 'Contaminante': '#FFD93D', 'Otro': '#6C5CE7'}
COLOR_BINARIO = {'Healthy': '#50C878', 'Defect': '#FF6B6B'}
CATEGORIA = {'Broken': 'Defecto', 'Damage': 'Defecto', 'Fungus': 'Defecto', 'Shriveled': 'Defecto',
             'Weeveled': 'Defecto', 'Healthy': 'Sano', 'Immature': 'Sano', 'Inorganic foreign material': 'Contaminante',
             'Organic foreign material': 'Contaminante', 'Other Edible Seeds': 'Otro'}


def estilo():
    """Estilo único de las gráficas (el mismo del EDA original)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
    plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 130, 'font.family': 'sans-serif',
                         'axes.titlesize': 14, 'axes.titleweight': 'bold'})


def paleta_clases():
    import seaborn as sns
    return dict(zip(CLASES, sns.color_palette('husl', len(CLASES))))


def guardar(carpeta, nombre):
    import matplotlib.pyplot as plt
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(carpeta / nombre, bbox_inches='tight')
    plt.show()


# =============================================================================
# 2. LECTURA, SESIONES Y LOTES
# =============================================================================
GAP_SESION = (60, 2)      # (segundos, pasos de contador): regla de sesión del proyecto
GAP_LOTE = (300, 5)       # agrupación conservadora: probable mismo lote de granos


def nombre_base(stem):
    """Nombre sin el sufijo '.rf.<hash>' que Roboflow cambia en cada exportación."""
    return re.sub(r'\.rf\.[0-9a-f]+$', '', stem)


def nombre_roboflow(stem):
    return re.sub(r'_JPG$', '', re.sub(r'(_jpg)?\.rf\.[0-9a-f]+$', '', stem))


def clave_captura(stem):
    base = re.sub(r'(_jpg)?\.rf\.[0-9a-f]+$', '', stem)
    m = re.match(r'^(?:IMG_)?(\d{8})_(\d{6})$', base)
    if m:
        return 'tiempo', datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').timestamp()
    m = re.match(r'^(IMG|DSC)_(\d+)_JPG$', base)
    if m:
        return f'contador_{m.group(1)}', float(m.group(2))
    return f'suelto_{base}', 0.0


def agrupar(stems, gap=GAP_SESION):
    """Identificador de grupo (sesión o lote) por stem, según los saltos de tiempo o de contador."""
    espacios = defaultdict(list)
    for s in stems:
        ns, v = clave_captura(s)
        espacios[ns].append((v, s))
    grupo = {}
    for ns in sorted(espacios):
        umbral = gap[0] if ns == 'tiempo' else gap[1]
        k, previo = 0, None
        for v, s in sorted(espacios[ns]):
            if previo is not None and v - previo > umbral:
                k += 1
            grupo[s] = f'{ns}_{k:03d}'
            previo = v
    return grupo


def leer_dataset(raiz, solo=None, splits=SPLITS):
    """Inventario de un dataset YOLO. solo: nombres base a conservar."""
    raiz = Path(raiz)
    inv = []
    for split in splits:
        for img in sorted((raiz / split / 'images').glob('*')):
            base = nombre_base(img.stem)
            if solo is not None and base not in solo:
                continue
            with Image.open(img) as im:
                W, H = im.size
            lbl = raiz / split / 'labels' / f'{img.stem}.txt'
            filas = [l.split() for l in lbl.read_text().splitlines() if len(l.split()) == 5] if lbl.exists() else []
            c = np.array([[float(x) for x in f[1:]] for f in filas]).reshape(-1, 4)
            xyxy = np.stack([(c[:, 0] - c[:, 2] / 2) * W, (c[:, 1] - c[:, 3] / 2) * H,
                             (c[:, 0] + c[:, 2] / 2) * W, (c[:, 1] + c[:, 3] / 2) * H], 1)
            inv.append({'stem': img.stem, 'base': base, 'split': split, 'img': img, 'lbl': lbl, 'W': W, 'H': H,
                        'xyxy': xyxy, 'yolo': c, 'clases': [int(float(f[0])) for f in filas]})
    sesion = agrupar([r['stem'] for r in inv], GAP_SESION)
    for r in inv:
        r['sesion'] = sesion[r['stem']]
    return inv


def tabla_cajas(inv):
    filas = []
    for r in inv:
        for (x1, y1, x2, y2), c in zip(r['xyxy'], r['clases']):
            filas.append({'base': r['base'], 'split': r['split'], 'clase': CLASES[c], 'id_clase': c,
                          'cx': (x1 + x2) / 2 / r['W'], 'cy': (y1 + y2) / 2 / r['H'],
                          'w': (x2 - x1) / r['W'], 'h': (y2 - y1) / r['H'],
                          'lado_menor_960': min(x2 - x1, y2 - y1) * 960 / max(r['W'], r['H'])})
    return pd.DataFrame(filas)


# =============================================================================
# 3. CALIDAD DEL ETIQUETADO
# =============================================================================
UMBRAL_DEGENERADA_PX = 3
UMBRAL_DUPLICADA_IOU = 0.85
UMBRAL_TAMANO_Z = 3.5


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def detectar(inv):
    """Un problema por fila; cada caja se informa una sola vez."""
    problemas, degeneradas = [], set()

    def anotar(tipo, r, caja, detalle, pareja=None):
        problemas.append({'tipo': tipo, 'stem': r['stem'], 'base': r['base'], 'split': r['split'],
                          'caja': caja, 'pareja': pareja, 'detalle': detalle})

    for r in inv:
        escala = 960 / max(r['W'], r['H'])
        for i, ((x1, y1, x2, y2), c) in enumerate(zip(r['xyxy'], r['clases'])):
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) * escala < UMBRAL_DEGENERADA_PX:
                anotar('Caja degenerada', r, i, f'lado menor {min(bw, bh) * escala:.1f} px a 960')
                degeneradas.add((r['stem'], i))
            elif c in CLASES_MAIZ and (bw / bh > 4 or bw / bh < 0.25):
                anotar('Relación de aspecto atípica', r, i, f'ancho/alto = {bw / bh:.2f}')
            if x1 < -1 or y1 < -1 or x2 > r['W'] + 1 or y2 > r['H'] + 1:
                anotar('Caja fuera de la imagen', r, i, '')
        m = iou(r['xyxy'], r['xyxy'])
        for a, b in zip(*np.where(np.triu(m, 1) > UMBRAL_DUPLICADA_IOU)):
            anotar('Cajas duplicadas', r, int(b), f"IoU {m[a, b]:.2f} con otra caja ({CLASES[r['clases'][a]]})", int(a))
    areas = defaultdict(list)
    for r in inv:
        for i, ((x1, y1, x2, y2), c) in enumerate(zip(r['xyxy'], r['clases'])):
            if (r['stem'], i) not in degeneradas and (x2 - x1) > 0 and (y2 - y1) > 0:
                areas[c].append((np.log((x2 - x1) * (y2 - y1) / (r['W'] * r['H'])), r, i))
    for c, lst in areas.items():
        v = np.array([a for a, _, _ in lst]); med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
        for a, r, i in lst:
            z = (a - med) / max(mad, 1e-9)
            if abs(z) > UMBRAL_TAMANO_Z:
                anotar('Tamaño atípico', r, i, f"{'muy pequeña' if z < 0 else 'muy grande'} para {CLASES[c]} (z = {z:+.1f})")
    por_sesion = defaultdict(list)
    for r in inv:
        por_sesion[r['sesion']].append(r)
    for rs in por_sesion.values():
        if len(rs) < 3:
            continue
        n = np.array([len(r['clases']) for r in rs], float); med = np.median(n)
        mad = max(1.4826 * np.median(np.abs(n - med)), 1.0)
        for r, ni in zip(rs, n):
            if (ni - med) / mad < -2 and ni < 0.6 * med:
                anotar('Posible etiquetado incompleto', r, -1, f'{int(ni)} cajas; mediana de su sesión {med:.0f}')
        comp = Counter(c for r in rs for c in r['clases'] if c in CLASES_MAIZ)
        total = sum(comp.values())
        if total:
            dom, ndom = comp.most_common(1)[0]
            if ndom / total >= 0.85:
                for r in rs:
                    for i, c in enumerate(r['clases']):
                        if c in CLASES_MAIZ and c != dom and comp[c] / total < 0.03:
                            anotar('Clase rara en sesión casi monoclase', r, i,
                                   f'{CLASES[c]} ({comp[c]} de {total} cajas) en una sesión de {CLASES[dom]}')
    return pd.DataFrame(problemas, columns=['tipo', 'stem', 'base', 'split', 'caja', 'pareja', 'detalle'])


def _dibujar(vista, r, escala, ox=0, oy=0, caja=None, pareja=None):
    for i, b in enumerate(r['xyxy']):
        color, grosor = ((0, 0, 255), 3) if i == caja else ((255, 120, 0), 3) if i == pareja else ((0, 170, 0), 1)
        cv2.rectangle(vista, (int((b[0] - ox) * escala), int((b[1] - oy) * escala)),
                      (int((b[2] - ox) * escala), int((b[3] - oy) * escala)), color, grosor)


def imagen_evidencia(r, caja, pareja, titulo, alto=600):
    """Izquierda: imagen completa con un círculo en el problema. Derecha: recorte ampliado."""
    img = cv2.imread(str(r['img']))
    e = alto / r['H']
    completa = cv2.resize(img, None, fx=e, fy=e, interpolation=cv2.INTER_AREA)
    _dibujar(completa, r, e, caja=caja, pareja=pareja)
    paneles = [completa]
    if caja is not None and caja >= 0:
        x1, y1, x2, y2 = r['xyxy'][caja]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cv2.circle(completa, (int(cx * e), int(cy * e)), 35, (0, 0, 255), 3)
        lado = max(4 * max(x2 - x1, y2 - y1), 0.25 * max(r['W'], r['H']))
        ox, oy = int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2))
        recorte = img[oy:int(min(r['H'], cy + lado / 2)), ox:int(min(r['W'], cx + lado / 2))]
        er = alto / recorte.shape[0]
        zoom = cv2.resize(recorte, None, fx=er, fy=er, interpolation=cv2.INTER_AREA)
        _dibujar(zoom, r, er, ox, oy, caja, pareja)
        if max(x2 - x1, y2 - y1) * er < 25:
            cv2.circle(zoom, (int((cx - ox) * er), int((cy - oy) * er)), 30, (0, 0, 255), 3)
        paneles += [np.full((alto, 12, 3), 255, np.uint8), zoom]
    lienzo = np.hstack(paneles)
    barra = np.full((40, lienzo.shape[1], 3), 255, np.uint8)
    cv2.putText(barra, titulo, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return np.vstack([barra, lienzo])


def comparar(inv_a, inv_b, umbral=0.5):
    """Cambios caja a caja entre dos versiones (emparejadas por nombre base e IoU)."""
    B = {r['base']: r for r in inv_b}
    cambios = []
    for ra in inv_a:
        rb = B.get(ra['base'])
        if rb is None:
            continue
        m = iou(ra['xyxy'], rb['xyxy'])
        usados_a, usados_b = set(), set()
        for i, j in sorted(zip(*np.where(m >= umbral)), key=lambda t: -m[t]):
            if i in usados_a or j in usados_b:
                continue
            usados_a.add(i); usados_b.add(j)
            if ra['clases'][i] != rb['clases'][j]:
                cambios.append({'base': ra['base'], 'tipo': 'reclasificada', 'caja_a': i, 'caja_b': j,
                                'clase_antes': CLASES[ra['clases'][i]], 'clase_despues': CLASES[rb['clases'][j]]})
            elif m[i, j] < 0.95:
                cambios.append({'base': ra['base'], 'tipo': 'reajustada', 'caja_a': i, 'caja_b': j,
                                'clase_antes': CLASES[ra['clases'][i]], 'clase_despues': CLASES[rb['clases'][j]]})
        for i in set(range(len(ra['clases']))) - usados_a:
            cambios.append({'base': ra['base'], 'tipo': 'eliminada', 'caja_a': i, 'caja_b': None,
                            'clase_antes': CLASES[ra['clases'][i]], 'clase_despues': None})
        for j in set(range(len(rb['clases']))) - usados_b:
            cambios.append({'base': ra['base'], 'tipo': 'añadida', 'caja_a': None, 'caja_b': j,
                            'clase_antes': None, 'clase_despues': CLASES[rb['clases'][j]]})
    return pd.DataFrame(cambios, columns=['base', 'tipo', 'caja_a', 'caja_b', 'clase_antes', 'clase_despues'])


# =============================================================================
# 4. PARTICIÓN POR LOTE Y PLIEGUES
# =============================================================================
def _composicion(stems, sub_de):
    c = Counter()
    for s in stems:
        c.update(sub_de[s])
    return c


def _peor_representacion(grupos_stems, sub_de, global_c):
    total = sum(global_c.values())
    peor = 1e9
    for stems in grupos_stems:
        c = _composicion(stems, sub_de); t = sum(c.values()) or 1
        for k, n in global_c.items():
            peor = min(peor, (c.get(k, 0) / t) / (n / total))
    return peor


def particionar(inv_maiz, fraccion_test=0.15, k=4, semillas=300):
    """Partición por lote: test (lotes completos, ~fraccion_test) + k pliegues sobre el desarrollo.

    Se usa StratifiedGroupKFold (grupo = lote, estrato = subtipo dominante del lote) y se elige la
    semilla que mejor representa al subtipo peor cubierto. Solo usa etiquetas."""
    import warnings
    from sklearn.model_selection import StratifiedGroupKFold

    stems = [r['stem'] for r in inv_maiz]
    lote = agrupar(stems, GAP_LOTE)
    sub_de = {r['stem']: Counter(CLASES[c] for c in r['clases'] if c in CLASES_MAIZ) for r in inv_maiz}
    comp_lote = defaultdict(Counter)
    for s in stems:
        comp_lote[lote[s]].update(sub_de[s])
    dominante = {g: c.most_common(1)[0][0] for g, c in comp_lote.items()}

    def dividir(lista, n, objetivo=None):
        y = [dominante[lote[s]] for s in lista]; g = [lote[s] for s in lista]
        glob = _composicion(lista, sub_de)
        mejor = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for semilla in range(semillas):
                partes = [[lista[i] for i in v] for _, v in
                          StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=semilla).split(np.zeros(len(lista)), y, g)]
                if objetivo is None:
                    tam = np.array([len(p) for p in partes])
                    puntaje = _peor_representacion(partes, sub_de, glob) - 0.5 * tam.std() / tam.mean()
                    if mejor is None or puntaje > mejor[0]:
                        mejor = (puntaje, semilla, partes)
                else:
                    for p in partes:
                        frac = len(p) / len(lista)
                        puntaje = _peor_representacion([p], sub_de, glob) - 4 * abs(frac - objetivo)
                        if mejor is None or puntaje > mejor[0]:
                            mejor = (puntaje, semilla, p)
        return mejor

    _, semilla_test, test = dividir(stems, int(round(1 / fraccion_test)), objetivo=fraccion_test)
    dev = sorted(set(stems) - set(test))
    _, semilla_pl, pliegues = dividir(dev, k)
    pliegue_de = {s: i for i, p in enumerate(pliegues) for s in p}
    lotes_test = {lote[s] for s in test}
    assert not lotes_test & {lote[s] for s in dev}, 'Un lote quedó en test y en desarrollo'
    lp = {}
    for s, i in pliegue_de.items():
        assert lp.setdefault(lote[s], i) == i, 'Un lote quedó en dos pliegues'
    return {'gap_lote_s': GAP_LOTE[0], 'gap_lote_contador': GAP_LOTE[1], 'fraccion_test': fraccion_test, 'k': k,
            'semilla_test': semilla_test, 'semilla_pliegues': semilla_pl,
            'imagenes': {s: {'conjunto': 'test' if s in set(test) else 'dev', 'pliegue': pliegue_de.get(s),
                             'lote': lote[s], 'subtipos': dict(sub_de[s])} for s in stems}}


def resumen_particion(part):
    filas = []
    imgs = part['imagenes']
    grupos = [('test', [s for s, v in imgs.items() if v['conjunto'] == 'test'])]
    grupos += [(f"pliegue {k}", [s for s, v in imgs.items() if v['pliegue'] == k]) for k in range(part['k'])]
    glob = Counter()
    for v in imgs.values():
        glob.update(v['subtipos'])
    for nombre, stems in grupos:
        c = Counter()
        for s in stems:
            c.update(imgs[s]['subtipos'])
        t = sum(c.values())
        bin_ = Counter()
        for k, n in c.items():
            bin_[CLASES_BINARIAS[MAPEO_BINARIO[CLASES.index(k)]]] += n
        filas.append({'conjunto': nombre, 'imágenes': len(stems), 'lotes': len({imgs[s]['lote'] for s in stems}),
                      'cajas': t, '% Healthy': round(100 * bin_['Healthy'] / t, 1),
                      **{CLASES[i]: round(100 * c.get(CLASES[i], 0) / t, 1) for i in SUBTIPOS}})
    t = sum(glob.values())
    filas.append({'conjunto': 'global', 'imágenes': len(imgs), 'lotes': len({v['lote'] for v in imgs.values()}),
                  'cajas': t, '% Healthy': round(100 * (glob['Healthy'] + glob['Immature']) / t, 1),
                  **{CLASES[i]: round(100 * glob.get(CLASES[i], 0) / t, 1) for i in SUBTIPOS}})
    return pd.DataFrame(filas)


# =============================================================================
# 5. PREPROCESAMIENTO
# =============================================================================
PAD_DIVISOR = 32
PAD_COLOR = (114, 114, 114)
NORM = {'brillo_objetivo': 175, 'factor_contraste': 1.1, 'offset_brillo': 5, 'factor_saturacion': 1.15}


def normalizar_imagen(img):
    """Gray World + brillo + contraste + saturación (variante 'norm' de la iteración anterior)."""
    f = img.astype(np.float32); media = f.mean(axis=(0, 1))
    img = np.clip(f * (media.mean() / (media + 1e-6))[None, None, :], 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    if hsv[:, :, 2].mean() > 0:
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.clip(NORM['brillo_objetivo'] / hsv[:, :, 2].mean(), 0.7, 1.4), 0, 255)
    img = cv2.convertScaleAbs(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR),
                              alpha=NORM['factor_contraste'], beta=NORM['offset_brillo'])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * NORM['factor_saturacion'], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _letterbox(resolucion):
    import albumentations as A
    redim = A.LongestMaxSize(max_size=resolucion, interpolation=cv2.INTER_AREA)
    relleno = A.Compose([A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=PAD_DIVISOR,
                                       pad_width_divisor=PAD_DIVISOR, border_mode=cv2.BORDER_CONSTANT,
                                       fill=PAD_COLOR, position='center')],
                        bbox_params=A.BboxParams(format='yolo', label_fields=['clases', 'subtipos'],
                                                 min_visibility=0.0, clip=True))
    return redim, relleno


def guardar_yolo(ruta, cajas, clases):
    with open(ruta, 'w') as f:
        for c, (cx, cy, w, h) in zip(clases, cajas):
            f.write(f'{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')


def leer_yolo(ruta):
    filas = [l.split() for l in Path(ruta).read_text().splitlines() if len(l.split()) == 5] if Path(ruta).exists() else []
    return [[float(x) for x in f[1:]] for f in filas], [int(float(f[0])) for f in filas]


def generar_datasets(inv_maiz, part, destino, resoluciones=(960, 1280), variantes=('base', 'norm'), hilos=8):
    """datos/Data_2c_<res>_<variante>/{dev,test}/{images,labels,labels_subtipo}: un solo pase por imagen.
    labels: clase binaria; labels_subtipo: subtipo original en la misma línea."""
    from concurrent.futures import ThreadPoolExecutor
    destino = Path(destino)
    for r_ in resoluciones:
        for v in variantes:
            for conj in ('dev', 'test'):
                for sub in ('images', 'labels', 'labels_subtipo'):
                    (destino / f'Data_2c_{r_}_{v}' / conj / sub).mkdir(parents=True, exist_ok=True)

    def procesar(r):
        conj = part['imagenes'][r['stem']]['conjunto']
        img = cv2.imread(str(r['img']))
        keep = [i for i, c in enumerate(r['clases']) if c in CLASES_MAIZ]
        cajas = [list(r['yolo'][i]) for i in keep]
        subt = [r['clases'][i] for i in keep]
        binc = [MAPEO_BINARIO[c] for c in subt]
        for res in resoluciones:
            redim, relleno = _letterbox(res)
            red = redim(image=img)['image']
            for v in variantes:
                x = normalizar_imagen(red) if v == 'norm' else red
                out = relleno(image=cv2.cvtColor(x, cv2.COLOR_BGR2RGB), bboxes=cajas, clases=binc, subtipos=subt)
                assert len(out['bboxes']) == len(cajas), r['stem']
                d = destino / f'Data_2c_{res}_{v}' / conj
                cv2.imwrite(str(d / 'images' / f"{r['stem']}.jpg"), cv2.cvtColor(out['image'], cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                bb = [list(map(float, b)) for b in out['bboxes']]
                guardar_yolo(d / 'labels' / f"{r['stem']}.txt", bb, [int(c) for c in out['clases']])
                guardar_yolo(d / 'labels_subtipo' / f"{r['stem']}.txt", bb, [int(c) for c in out['subtipos']])
        return len(cajas)

    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=hilos) as ex:
        n = sum(ex.map(procesar, inv_maiz))
    return n


# =============================================================================
# 6. BALANCEO
# =============================================================================
# Las copias de balanceo usan solo transformaciones que Ultralytics NO aplica por defecto en línea
# (rotación de 90/180/270 grados, volteo vertical, ruido, compresión JPEG, gamma, nitidez, curva tonal),
# para no repetir el aumento online y que no sean duplicados exactos.
def transformacion_unica():
    import albumentations as A
    return A.Compose(
        [A.RandomRotate90(p=1.0), A.VerticalFlip(p=0.5),
         A.OneOf([A.GaussNoise(std_range=(0.03, 0.08)), A.ImageCompression(quality_range=(60, 85)),
                  A.RandomGamma(gamma_limit=(80, 120)), A.Sharpen(alpha=(0.15, 0.3), lightness=(0.95, 1.05)),
                  A.RandomToneCurve(scale=0.15)], p=1.0)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['clases'], min_visibility=0.0, clip=True))


def _copia(transf, img_path, destino, nombre):
    lbl = Path(str(img_path.parent).replace('/images', '/labels')) / f'{img_path.stem}.txt'
    cajas, clases = leer_yolo(lbl)
    img = cv2.imread(str(img_path))
    out = transf(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bboxes=cajas, clases=clases)
    ruta = destino / 'images' / f'{nombre}.jpg'
    cv2.imwrite(str(ruta), cv2.cvtColor(out['image'], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    guardar_yolo(destino / 'labels' / f'{nombre}.txt', [list(map(float, b)) for b in out['bboxes']],
                 [int(c) for c in out['clases']])
    return ruta, Counter(int(c) for c in out['clases'])


def balancear(imagenes, destino, semilla=42, tolerancia=0.02):
    """Copias por déficit (priorizando imágenes de la clase deficitaria) hasta igualar las clases."""
    destino = Path(destino)
    (destino / 'images').mkdir(parents=True, exist_ok=True); (destino / 'labels').mkdir(parents=True, exist_ok=True)
    random.seed(semilla); np.random.seed(semilla)
    transf = transformacion_unica(); transf.set_random_seed(semilla)
    inv = []
    for p in sorted(imagenes, key=lambda p: p.stem):
        _, cl = leer_yolo(Path(str(p.parent).replace('/images', '/labels')) / f'{p.stem}.txt')
        inv.append((p, Counter(cl)))
    conteo = Counter()
    for _, c in inv:
        conteo.update(c)
    objetivo = max(conteo.values())
    fuentes = {k: sorted([x for x in inv if x[1][k]], key=lambda x: -x[1][k] / max(sum(x[1].values()), 1))
               for k in (0, 1)}
    nuevas, ronda = [], 0
    while True:
        deficit = {k: (objetivo - conteo[k]) / objetivo for k in (0, 1)}
        clase = max(deficit, key=deficit.get)
        if deficit[clase] <= tolerancia or not fuentes[clase] or ronda > 2000:
            break
        p, _ = fuentes[clase][ronda % len(fuentes[clase])]
        ronda += 1
        ruta, c = _copia(transf, p, destino, f'{p.stem}_bal{ronda}')
        conteo.update(c); nuevas.append(ruta)
    return nuevas


# =============================================================================
# 7. EXPERIMENTOS
# =============================================================================
_PESOS_PROYECTO = MODELOS / 'yolo11s.pt'
PROTOCOLO = {'modelo': str(_PESOS_PROYECTO) if _PESOS_PROYECTO.exists() else 'yolo11s.pt', 'epocas_base': 50, 'batch': 8, 'optimizer': 'SGD', 'lr0': 0.01,
             'cos_lr': False, 'seed': 42, 'deterministic': True, 'workers': 4, 'conf': 0.25}

# Diseño factorial completo 3 x 2 x 2: resolución x fotométrico x balanceo (niveles de menor a mayor)
FACTORES = {'resolucion': (640, 960, 1280), 'fotometrico': ('base', 'norm'), 'balanceo': ('sinbal', 'bal')}


# Experimentos complementarios sobre la configuración elegida: cada uno quita una técnica aplicada
COMPLEMENTARIOS = {'etiquetas_originales': 'Limpieza de etiquetas (corregidas − originales)',
                   'sin_aumento': 'Aumento en línea (con − sin)'}
# Desactiva todo el aumento en línea de Ultralytics, incluidas sus transformaciones de Albumentations (p = 0.01)
SIN_AUMENTO = {'mosaic': 0.0, 'fliplr': 0.0, 'flipud': 0.0, 'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0, 'degrees': 0.0,
               'translate': 0.0, 'scale': 0.0, 'shear': 0.0, 'perspective': 0.0, 'mixup': 0.0, 'cutmix': 0.0,
               'copy_paste': 0.0, 'augmentations': []}


def experimentos_complementarios(part, decision):
    """Una corrida por pliegue y por técnica quitada, con la configuración elegida; se comparan con las corridas
    del diseño factorial de esa misma configuración."""
    cfg = {f: decision[f]['elegido'] for f in FACTORES}
    return [{**cfg, 'complementario': c, 'pliegue': k,
             'id': f"{cfg['resolucion']}_{cfg['fotometrico']}_{cfg['balanceo']}_{c}_p{k}"}
            for c in COMPLEMENTARIOS for k in range(part['k'])]


def con_etiquetas_originales(maiz, original):
    """Las mismas imágenes de maíz de la versión corregida, con las etiquetas de la versión original (v0)."""
    v0 = {r['base']: r for r in leer_dataset(original)}
    assert all(r['base'] in v0 and (v0[r['base']]['W'], v0[r['base']]['H']) == (r['W'], r['H']) for r in maiz)
    return [{**r, 'yolo': v0[r['base']]['yolo'], 'clases': v0[r['base']]['clases']} for r in maiz]


def configuraciones():
    return [{'resolucion': r, 'fotometrico': f, 'balanceo': b}
            for r in FACTORES['resolucion'] for f in FACTORES['fotometrico'] for b in FACTORES['balanceo']]


def matriz_experimentos(part, resoluciones=None):
    exps = []
    for cfg in configuraciones():
        if resoluciones and cfg['resolucion'] not in resoluciones:
            continue
        for k in range(part['k']):
            exps.append({**cfg, 'pliegue': k,
                         'id': f"{cfg['resolucion']}_{cfg['fotometrico']}_{cfg['balanceo']}_p{k}"})
    return exps


def preparar_pliegue(exp, part, datos, trabajo):
    """Escribe train.txt, val.txt y data.yaml del pliegue y, si hay balanceo, genera (con caché) las
    copias a partir de las imágenes de entrenamiento del pliegue. Devuelve (data.yaml, val, n_base, n_train)."""
    nombre = f"Data_2c_{exp['resolucion']}_{exp['fotometrico']}"
    raiz = Path(datos) / nombre / 'dev' / 'images'
    # Con etiquetas originales, solo el entrenamiento las usa; la validación siempre usa las corregidas
    raiz_t = Path(datos) / 'etiquetas_v0' / nombre / 'dev' / 'images' if exp.get('complementario') == 'etiquetas_originales' else raiz
    imgs = part['imagenes']
    train = [raiz_t / f'{s}.jpg' for s, v in sorted(imgs.items()) if v['conjunto'] == 'dev' and v['pliegue'] != exp['pliegue']]
    val = [raiz / f'{s}.jpg' for s, v in sorted(imgs.items()) if v['pliegue'] == exp['pliegue']]
    base_t = Path(trabajo) / f"{exp['resolucion']}_{exp['fotometrico']}_p{exp['pliegue']}"
    extra = []
    if exp['balanceo'] == 'bal':
        carpeta = base_t / 'balanceo'
        if (carpeta / 'hecho.json').exists():
            extra = sorted((carpeta / 'images').glob('*.jpg'))
        else:
            shutil.rmtree(carpeta, ignore_errors=True)
            extra = balancear(train, carpeta)
            (carpeta / 'hecho.json').write_text(json.dumps({'copias': len(extra)}))
    carpeta = base_t / (exp.get('complementario') or exp['balanceo'])
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / 'train.txt').write_text('\n'.join(map(str, train + extra)) + '\n')
    (carpeta / 'val.txt').write_text('\n'.join(map(str, val)) + '\n')
    (carpeta / 'data.yaml').write_text(f"path: {carpeta}\ntrain: train.txt\nval: val.txt\nnc: 2\nnames: {json.dumps(CLASES_BINARIAS)}\n")
    return carpeta / 'data.yaml', val, len(train), len(train) + len(extra)


def metricas_aplicacion(modelo, imagenes, imgsz, conf=0.25):
    """Acierto por subtipo (clase binaria correcta, IoU >= 0.5) y MAE de conteo con NMS agnóstico."""
    total, acierto, err = Counter(), Counter(), []
    for i in range(0, len(imagenes), 8):
        for res in modelo.predict([str(p) for p in imagenes[i:i + 8]], imgsz=imgsz, conf=conf,
                                  agnostic_nms=True, verbose=False):
            p = Path(res.path)
            H, W = res.orig_shape
            cajas, clases = leer_yolo(p.parent.parent / 'labels' / f'{p.stem}.txt')
            _, subt = leer_yolo(p.parent.parent / 'labels_subtipo' / f'{p.stem}.txt')
            c = np.array(cajas).reshape(-1, 4)
            gt = np.stack([(c[:, 0] - c[:, 2] / 2) * W, (c[:, 1] - c[:, 3] / 2) * H,
                           (c[:, 0] + c[:, 2] / 2) * W, (c[:, 1] + c[:, 3] / 2) * H], 1)
            pb, pc = res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)
            err.append(abs(len(pb) - len(gt)))
            m = iou(gt, pb); usados, pares = set(), {}
            for g, q in sorted(zip(*np.where(m >= 0.5)), key=lambda t: -m[t]):
                if g not in pares and q not in usados:
                    pares[g] = q; usados.add(q)
            for g, (cl, st) in enumerate(zip(clases, subt)):
                total[st] += 1
                if g in pares and pc[pares[g]] == cl:
                    acierto[st] += 1
    fila = {'mae_conteo': float(np.mean(err))}
    for s in SUBTIPOS:
        fila[f'acierto_{CLASES[s]}'] = acierto[s] / total[s] if total[s] else np.nan
    return fila


def plataforma():
    import os
    if 'COLAB_RELEASE_TAG' in os.environ:   # Colab también tiene una carpeta /kaggle
        return 'colab'
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    return 'local'


def correr(exp, part, datos, trabajo, runs, registro, guardar_pesos=True, epocas=None):
    """Entrena un experimento con el mismo presupuesto de iteraciones que la configuración sin
    copias, evalúa en su pliegue de validación y añade una fila al registro."""
    import time
    import torch
    from ultralytics import YOLO

    data_yaml, val, n_base, n_train = preparar_pliegue(exp, part, datos, trabajo)
    epocas = epocas or max(10, round(PROTOCOLO['epocas_base'] * n_base / n_train))
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    modelo = YOLO(PROTOCOLO['modelo'])
    modelo.train(data=str(data_yaml), imgsz=exp['resolucion'], epochs=epocas, patience=epocas,
                 batch=PROTOCOLO['batch'], optimizer=PROTOCOLO['optimizer'], lr0=PROTOCOLO['lr0'],
                 cos_lr=PROTOCOLO['cos_lr'], seed=PROTOCOLO['seed'], deterministic=PROTOCOLO['deterministic'],
                 workers=PROTOCOLO['workers'], project=str(runs), name=exp['id'], exist_ok=True,
                 plots=True, verbose=False, **(SIN_AUMENTO if exp.get('complementario') == 'sin_aumento' else {}))
    minutos = (time.time() - t0) / 60
    carpeta = Path(runs) / exp['id']
    mejor = YOLO(str(carpeta / 'weights' / 'best.pt'))
    v = mejor.val(data=str(data_yaml), split='val', imgsz=exp['resolucion'], batch=PROTOCOLO['batch'],
                  plots=False, verbose=False, project=str(runs), name=f"{exp['id']}_val", exist_ok=True)
    fila = {**exp, 'n_train_base': n_base, 'n_train': n_train, 'epocas': epocas,
            'mAP50_95': float(v.box.map), 'mAP50': float(v.box.map50), 'mAP75': float(v.box.map75),
            'precision': float(v.box.mp), 'recall': float(v.box.mr),
            'AP50_95_Healthy': float(v.box.ap[0]), 'AP50_95_Defect': float(v.box.ap[1]),
            **metricas_aplicacion(mejor, val, exp['resolucion'], PROTOCOLO['conf']),
            'minutos': round(minutos, 1),
            'vram_gb': round(torch.cuda.max_memory_reserved() / 1e9, 2) if torch.cuda.is_available() else 0.0,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'plataforma': plataforma(), 'fecha': datetime.now().isoformat(timespec='seconds')}
    shutil.rmtree(Path(runs) / f"{exp['id']}_val", ignore_errors=True)
    (carpeta / 'weights' / 'last.pt').unlink(missing_ok=True)   # solo se conserva best.pt
    if not guardar_pesos:
        shutil.rmtree(carpeta / 'weights', ignore_errors=True)
    registro = Path(registro)
    df = pd.read_csv(registro) if registro.exists() else pd.DataFrame()
    pd.concat([df, pd.DataFrame([fila])], ignore_index=True).to_csv(registro, index=False)
    liberar_gpu(modelo, mejor)
    del modelo, mejor
    liberar_gpu()
    return fila


def liberar_gpu(*modelos):
    """Suelta las referencias del entrenador (modelo, optimizador, dataloaders) y vacía la caché de CUDA,
    para que la corrida siguiente empiece con la VRAM libre."""
    import gc
    import torch
    for m in modelos:
        for attr in ('trainer', 'predictor', 'validator'):
            if getattr(m, attr, None) is not None:
                setattr(m, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =============================================================================
# 8. ANÁLISIS DEL DISEÑO FACTORIAL
# =============================================================================
def _modelo(d):
    """Matrices del modelo lineal: bloque (intercepto + pliegue) y, por término, sus columnas con
    codificación por efectos (suma cero), de modo que cada término se contrasta con el modelo completo."""
    import itertools
    cod = {f: np.column_stack([(d[f] == n).astype(float).values - (d[f] == niv[0]).astype(float).values for n in niv[1:]])
           for f, niv in _niveles(d).items()}
    terminos = {}
    for n in (1, 2, 3):
        for combo in itertools.combinations(FACTORES, n):
            terminos[' x '.join(combo)] = np.column_stack([np.prod(cols, axis=0) for cols in itertools.product(*[cod[f].T for f in combo])])
    pl = sorted(d.pliegue.unique())
    bloque = np.column_stack([np.ones(len(d))] + [(d.pliegue == q).astype(float).values for q in pl[1:]])
    return bloque, terminos


def _rss(M, y):
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    return ((y - M @ beta) ** 2).sum(), np.linalg.matrix_rank(M)


def _niveles(d):
    """Niveles de cada factor presentes en los datos (el análisis funciona aunque falte alguno)."""
    return {f: [n for n in niv if (d[f] == n).any()] for f, niv in FACTORES.items()}


def _diseno(df, respuesta):
    d = df.dropna(subset=[respuesta])
    if 'complementario' in d:
        d = d[d['complementario'].isna()]
    return d[np.logical_and.reduce([d[f].isin(niv) for f, niv in FACTORES.items()])].copy()


def anova(df, respuesta='mAP50_95', alfa=0.05):
    """Indica si cada factor o interacción influye en la respuesta (prueba F con el pliegue como bloque)."""
    from scipy import stats
    d = _diseno(df, respuesta); y = d[respuesta].values
    bloque, terminos = _modelo(d)
    rss, rango = _rss(np.column_stack([bloque, *terminos.values()]), y)
    gl_res = len(y) - rango
    filas = []
    for n, X in terminos.items():
        rss_sin, _ = _rss(np.column_stack([bloque, *[v for k, v in terminos.items() if k != n]]), y)
        F = (rss_sin - rss) / X.shape[1] / (rss / gl_res)
        pv = stats.f.sf(F, X.shape[1], gl_res)
        filas.append({'factor': n, 'valor p': pv, 'influye': 'Sí' if pv < alfa else 'No'})
    return pd.DataFrame(filas)


def analizar(df, respuesta='mAP50_95', alfa=0.05):
    """Efectos principales como diferencias entre niveles consecutivos de cada factor (promediando los demás),
    con IC y p a partir de la varianza residual del modelo completo con el pliegue como bloque."""
    from scipy import stats
    d = _diseno(df, respuesta); y = d[respuesta].values
    bloque, terminos = _modelo(d)
    rss, rango = _rss(np.column_stack([bloque, *terminos.values()]), y)
    gl = len(y) - rango
    s2 = rss / gl
    tcrit = stats.t.ppf(1 - alfa / 2, gl)
    filas = []
    for f, niv in _niveles(d).items():
        m, n = d.groupby(f)[respuesta].mean(), d.groupby(f)[respuesta].size()
        for bajo, alto in zip(niv, niv[1:]):
            ef, ee = m[alto] - m[bajo], np.sqrt(s2 * (1 / n[alto] + 1 / n[bajo]))
            pv = 2 * stats.t.sf(abs(ef / ee), gl)
            filas.append({'comparación': f'{f}: {alto} - {bajo}', 'diferencia': ef, 'mínimo probable': ef - tcrit * ee,
                          'máximo probable': ef + tcrit * ee, 'valor p': pv, 'significativo': 'Sí' if pv < alfa else 'No'})
    return pd.DataFrame(filas)


def decidir(tabla, alfa=0.05, ganancia_min_resolucion=0.010):
    """Resolución: se sube al nivel siguiente mientras mejore al menos `ganancia_min_resolucion` con p < alfa.
    Fotométrico y balanceo: se adopta el nivel alto solo si su efecto es positivo con p < alfa."""
    t = tabla.set_index('comparación')
    dec = {}
    for f, todos in FACTORES.items():
        niv = [n for n in todos if any(str(n) in x.split(': ')[1].split(' - ') for x in t.index if x.startswith(f'{f}:'))]   # presentes
        umbral = ganancia_min_resolucion if f == 'resolucion' else 0.0
        elegido, pasos = niv[0], {}
        for bajo, alto in zip(niv, niv[1:]):
            ef, p = t.loc[f'{f}: {alto} - {bajo}', ['diferencia', 'valor p']]
            pasos[f'{alto} - {bajo}'] = {'diferencia': round(float(ef), 4), 'valor_p': round(float(p), 4)}
            if elegido == bajo and ef >= umbral and ef > 0 and p < alfa:
                elegido = alto
        dec[f] = {'elegido': elegido, **pasos}
    return dec


# =============================================================================
# 8. DATASET FINAL (Actividad 1.3)
# =============================================================================
SPLITS_FINALES = ('train', 'valid', 'test')


def elegir_pliegue_validacion(part, tamano=(0.15, 0.25)):
    """Pliegue de validación: entre los que tienen entre el 15 y el 25 % de las imágenes, el de composición por
    subtipo más parecida a la del conjunto de desarrollo (menor diferencia máxima en puntos porcentuales).
    Solo usa etiquetas. Devuelve (pliegue, tabla con el criterio)."""
    imgs = part['imagenes']
    dev = Counter()
    for v in imgs.values():
        if v['conjunto'] == 'dev':
            dev.update(v['subtipos'])
    T = sum(dev.values())
    filas = []
    for k in range(part['k']):
        c = Counter()
        stems = [s for s, v in imgs.items() if v['pliegue'] == k]
        for s in stems:
            c.update(imgs[s]['subtipos'])
        t = sum(c.values())
        filas.append({'pliegue': k, 'imágenes': len(stems), '% del total': round(100 * len(stems) / len(imgs), 1),
                      'diferencia máxima por subtipo (pp)': round(max(abs(100 * c[CLASES[i]] / t - 100 * dev[CLASES[i]] / T)
                                                                    for i in SUBTIPOS), 2)})
    tabla = pd.DataFrame(filas)
    aptos = tabla[tabla['% del total'].between(100 * tamano[0], 100 * tamano[1])]
    return int(aptos.loc[aptos['diferencia máxima por subtipo (pp)'].idxmin(), 'pliegue']), tabla


def particion_final(part, pliegue_val):
    """{stem: 'train' | 'valid' | 'test'}: el test fijo, el pliegue elegido como validación y el resto como entrenamiento."""
    return {s: 'test' if v['conjunto'] == 'test' else 'valid' if v['pliegue'] == pliegue_val else 'train'
            for s, v in part['imagenes'].items()}


def estructurar_dataset(origen, splits, destino):
    """Copia con os/shutil las imágenes y etiquetas ya preprocesadas de `origen` ({dev,test}/...) a la jerarquía de YOLO
    destino/{train,valid,test}/{images,labels,labels_subtipo} y escribe data.yaml (rutas relativas al propio yaml)."""
    origen, destino = Path(origen), Path(destino)
    shutil.rmtree(destino, ignore_errors=True)
    for sp in SPLITS_FINALES:
        for sub in ('images', 'labels', 'labels_subtipo'):
            os.makedirs(destino / sp / sub, exist_ok=True)
    for stem, sp in sorted(splits.items()):
        conj = 'test' if sp == 'test' else 'dev'
        for sub, ext in (('images', '.jpg'), ('labels', '.txt'), ('labels_subtipo', '.txt')):
            shutil.copy2(origen / conj / sub / f'{stem}{ext}', destino / sp / sub / f'{stem}{ext}')
    (destino / 'data.yaml').write_text('# Dataset final: maíz sano / defectuoso (Actividad 1.3)\n'
                                       'train: train/images\nval: valid/images\ntest: test/images\n'
                                       f'nc: {len(CLASES_BINARIAS)}\nnames: {json.dumps(CLASES_BINARIAS)}\n')
    return destino / 'data.yaml'


def pipeline_dataset_final(limpio, part, decision, datos, destino, pliegue_val):
    """Pipeline completo, desde el dataset con etiquetas corregidas hasta el dataset final:
    consolidación a 2 clases -> letterbox a la resolución elegida -> variante fotométrica elegida ->
    partición train/valid/test por lote -> jerarquía de YOLO -> balanceo del entrenamiento si se eligió."""
    res, foto, bal = (decision[f]['elegido'] for f in ('resolucion', 'fotometrico', 'balanceo'))
    inv = leer_dataset(limpio)
    maiz = [r for r in inv if any(c in CLASES_MAIZ for c in r['clases'])]
    generar_datasets(maiz, part, datos, resoluciones=(res,), variantes=(foto,))
    yaml = estructurar_dataset(Path(datos) / f'Data_2c_{res}_{foto}', particion_final(part, pliegue_val), destino)
    if bal == 'bal':
        balancear(sorted((Path(destino) / 'train' / 'images').glob('*.jpg')), Path(destino) / 'train')
    return yaml
