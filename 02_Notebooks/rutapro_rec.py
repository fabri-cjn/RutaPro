"""
rutapro_rec.py — Funciones compartidas del proyecto RutaPro.

Contiene la carga de datos, la construcción de los conjuntos de entrenamiento y
evaluación, las métricas Top-N y los cuatro recomendadores:

    Popularidad      (baseline)
    ContenidoTFIDF   (filtrado basado en contenido)
    KNNColaborativo  (filtrado colaborativo por vecinos, variante usuario o ítem)
    LightFMHibrido   (factorización híbrida con atributos; algoritmo de innovación)

Todos los modelos comparten la misma interfaz:
    modelo.fit(datos_entrenamiento)
    modelo.puntajes(user_id) -> np.array con un puntaje por curso (orden de datos.cursos)
y la función `recomendar` aplica la regla común: no recomendar cursos que el
estudiante ya tuvo como positivos en entrenamiento.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------

def ruta_base() -> Path:
    cwd = Path.cwd()
    return cwd.parent if cwd.name in ("02_Notebooks", "03_App") else cwd


def cargar_datos(base: Path | None = None):
    base = base or ruta_base()
    d = base / "01_Datos"
    cursos = pd.read_csv(d / "cursos.csv")
    usuarios = pd.read_csv(d / "usuarios.csv")
    pares = pd.read_csv(d / "pares_usuario_curso.csv", parse_dates=["fecha_evento"])
    return cursos, usuarios, pares


@dataclass
class Particion:
    """Conjunto de entrenamiento + conjunto a evaluar, con índices consistentes."""
    cursos: pd.DataFrame
    usuarios: pd.DataFrame
    train: pd.DataFrame            # pares usados para entrenar (con columnas positivo y peso)
    evaluar: pd.DataFrame          # pares positivos a predecir
    uidx: dict = field(init=False)
    iidx: dict = field(init=False)

    def __post_init__(self):
        self.uidx = {u: k for k, u in enumerate(self.usuarios["user_id"])}
        self.iidx = {c: k for k, c in enumerate(self.cursos["course_id"])}
        self.n_u, self.n_i = len(self.uidx), len(self.iidx)

    def matriz(self, solo_positivos=True, usar_peso=True) -> sp.csr_matrix:
        """Matriz usuario x curso con las interacciones de entrenamiento."""
        t = self.train[self.train.positivo == 1] if solo_positivos else self.train
        vals = t["peso"].astype(float).values if usar_peso else np.ones(len(t))
        return sp.csr_matrix((vals, (t.user_id.map(self.uidx), t.course_id.map(self.iidx))),
                             shape=(self.n_u, self.n_i))

    def vistos(self, user_id) -> np.ndarray:
        """Cursos positivos del estudiante en entrenamiento (no se vuelven a recomendar)."""
        t = self.train
        return t.loc[(t.user_id == user_id) & (t.positivo == 1), "course_id"].map(self.iidx).values

    def relevantes(self) -> dict:
        return self.evaluar.groupby("user_id")["course_id"].apply(lambda s: set(s.map(self.iidx))).to_dict()

    def usuarios_elegibles(self) -> list:
        """Usuarios con positivos a evaluar y al menos un positivo en entrenamiento."""
        con_hist = set(self.train.loc[self.train.positivo == 1, "user_id"])
        return sorted(set(self.evaluar.user_id) & con_hist)


def construir_particion(cursos, usuarios, pares, etapa="validacion", esquema="usuario") -> Particion:
    """
    etapa='validacion': entrena con train y evalúa en valid (ajuste de hiperparámetros).
    etapa='prueba'    : entrena con train + valid y evalúa en test (comparación final).
    esquema='usuario' : partición temporal por usuario (adoptada).
    esquema='global'  : partición temporal con corte global (prueba de robustez).
    """
    col = "particion" if esquema == "usuario" else "particion_global"
    if etapa == "validacion":
        tr, ev = pares[pares[col] == "train"], pares[(pares[col] == "valid") & (pares.positivo == 1)]
    else:
        tr, ev = pares[pares[col].isin(["train", "valid"])], pares[(pares[col] == "test") & (pares.positivo == 1)]
    return Particion(cursos, usuarios, tr.copy(), ev.copy())


# ---------------------------------------------------------------------------
# 2. Métricas Top-N
# ---------------------------------------------------------------------------

def metricas_usuario(ranking, relevantes, k):
    top = list(ranking[:k])
    hits = [1 if i in relevantes else 0 for i in top]
    n_hit = sum(hits)
    dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(relevantes), k)))
    return {"precision": n_hit / k, "recall": n_hit / len(relevantes),
            "hit_rate": float(n_hit > 0), "ndcg": dcg / idcg if idcg > 0 else 0.0}


def recomendar(modelo, datos: Particion, user_id, n=10, excluir_vistos=True):
    s = np.asarray(modelo.puntajes(user_id), dtype=float).copy()
    if excluir_vistos:
        s[datos.vistos(user_id)] = -np.inf
    # desempate determinista por índice del curso
    return np.lexsort((np.arange(len(s)), -s))[:n]


def evaluar(modelo, datos: Particion, ks=(5, 10), usuarios=None, devolver_detalle=False):
    rel = datos.relevantes()
    usuarios = usuarios if usuarios is not None else datos.usuarios_elegibles()
    filas, recs_all = [], {}
    kmax = max(ks)
    for u in usuarios:
        r = recomendar(modelo, datos, u, n=kmax)
        recs_all[u] = r
        for k in ks:
            filas.append({"user_id": u, "K": k, **metricas_usuario(r, rel[u], k)})
    det = pd.DataFrame(filas)
    res = det.groupby("K")[["precision", "recall", "hit_rate", "ndcg"]].mean()
    # cobertura de catálogo: % de cursos distintos que aparecen en algún Top-K
    res["cobertura"] = [len({i for r in recs_all.values() for i in r[:k]}) / datos.n_i for k in res.index]
    res["usuarios"] = len(usuarios)
    return (res, det, recs_all) if devolver_detalle else res


def tabla_metricas(resultados: dict, k=10):
    """resultados: {nombre: DataFrame de evaluar()} -> tabla comparativa para un K."""
    return pd.DataFrame({n: r.loc[k] for n, r in resultados.items()}).T


# ---------------------------------------------------------------------------
# 3. Modelos
# ---------------------------------------------------------------------------

class Popularidad:
    """Baseline: ordena los cursos por número de inscripciones en entrenamiento."""
    nombre = "Popularidad"

    def fit(self, datos: Particion):
        self.datos = datos
        m = datos.matriz(solo_positivos=True, usar_peso=False)
        conteo = np.asarray(m.sum(axis=0)).ravel()
        # desempate: popularidad relativa en la plataforma de origen (percentil de reseñas)
        pct = datos.cursos.groupby("plataforma")["n_resenas"].rank(pct=True).values
        self._s = conteo + 0.01 * pct
        return self

    def puntajes(self, user_id):
        return self._s


# --- Contenido ---------------------------------------------------------------

def _sin_tildes(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


# Términos bilingües por categoría: unen los textos de Coursera (inglés) con los de Platzi/DataCamp/edX (español)
TERMINOS_CATEGORIA = {
    "Python": "python programacion programming pandas",
    "SQL y Bases de Datos": "sql bases_de_datos databases consultas queries",
    "Excel": "excel hojas_de_calculo spreadsheets tablas_dinamicas",
    "Power BI": "power_bi dashboards dax bi",
    "Tableau y Visualización": "tableau visualizacion visualization dashboards storytelling",
    "Estadística": "estadistica statistics probabilidad probability inferencia",
    "R": "lenguaje_r r_programming tidyverse estadistica",
    "Machine Learning": "machine_learning aprendizaje_automatico modelos scikit_learn",
    "Análisis de Datos": "analisis_de_datos data_analysis negocio business",
    "Gestión de Proyectos": "gestion_de_proyectos project_management agile scrum",
    "Habilidades Blandas": "habilidades_blandas soft_skills comunicacion communication liderazgo",
}

# Perfil textual de cada objetivo laboral (se usa para el perfil declarado y el cold start)
TERMINOS_OBJETIVO = {
    "Data Analyst": "analisis_de_datos sql excel python power_bi estadistica",
    "BI Analyst": "power_bi tableau sql excel dashboards bi",
    "Data Scientist": "python machine_learning estadistica lenguaje_r modelos",
    "Analista Financiero": "excel power_bi finanzas financial analisis_de_datos",
    "Marketing Analyst": "analisis_de_datos marketing tableau excel estadistica",
    "Project Analyst": "gestion_de_proyectos project_management habilidades_blandas comunicacion excel",
}


def texto_curso(c: pd.Series) -> str:
    cat = TERMINOS_CATEGORIA.get(c["categoria"], "")
    txt = f"{c['titulo']} {c['habilidades']} {cat} {cat} nivel_{c['dificultad']} idioma_{c['idioma']}"
    return _sin_tildes(txt.lower())


def texto_perfil(u: pd.Series) -> str:
    intereses = " ".join(TERMINOS_CATEGORIA.get(x.strip(), "") for x in str(u["intereses"]).split(";"))
    return _sin_tildes(f"{TERMINOS_OBJETIVO.get(u['objetivo_laboral'], '')} {intereses} "
                       f"nivel_{u['nivel']} idioma_es".lower())


class ContenidoTFIDF:
    """
    Filtrado basado en contenido.
    - Cada curso se representa con TF-IDF de su título, habilidades, términos de categoría, nivel e idioma.
    - El perfil del estudiante es el promedio ponderado (por peso implícito) de los cursos de su historial,
      mezclado con el texto de su perfil declarado (objetivo + intereses) con peso `alpha_perfil`.
    - Puntaje = similitud del coseno entre perfil y curso.
    """
    nombre = "Contenido (TF-IDF)"

    def __init__(self, ngram=(1, 2), min_df=1, usar_vistas=False, alpha_perfil=0.0, sublinear=True):
        self.ngram, self.min_df, self.usar_vistas = ngram, min_df, usar_vistas
        self.alpha_perfil, self.sublinear = alpha_perfil, sublinear

    def fit(self, datos: Particion):
        self.datos = datos
        self.vec = TfidfVectorizer(ngram_range=self.ngram, min_df=self.min_df, sublinear_tf=self.sublinear,
                                   token_pattern=r"(?u)\b\w+\b")
        self.X = normalize(self.vec.fit_transform(datos.cursos.apply(texto_curso, axis=1)))
        R = datos.matriz(solo_positivos=not self.usar_vistas, usar_peso=True)
        self.P_hist = normalize(R @ self.X)                       # perfil por historial
        perfiles = datos.usuarios.apply(texto_perfil, axis=1)
        self.P_decl = normalize(self.vec.transform(perfiles))     # perfil declarado
        P = (1 - self.alpha_perfil) * self.P_hist + self.alpha_perfil * self.P_decl
        self.S = (normalize(P) @ self.X.T).toarray()
        return self

    def puntajes(self, user_id):
        return self.S[self.datos.uidx[user_id]]

    def puntajes_perfil(self, perfil: dict):
        """Cold start: puntajes a partir de un perfil declarado (sin historial)."""
        q = normalize(self.vec.transform([texto_perfil(pd.Series(perfil))]))
        return (q @ self.X.T).toarray().ravel()


# --- KNN colaborativo --------------------------------------------------------

def _top_k_por_fila(S: np.ndarray, k: int) -> np.ndarray:
    S = S.copy()
    np.fill_diagonal(S, 0)
    if k < S.shape[1]:
        idx = np.argpartition(-S, k, axis=1)[:, k:]
        np.put_along_axis(S, idx, 0, axis=1)
    return S


class KNNColaborativo:
    """
    Filtrado colaborativo por vecinos más cercanos (similitud del coseno).
    - variante='item'   : puntaje(u,i) = sum_{j en historial de u} sim_k(i,j) * r_uj
    - variante='usuario': puntaje(u,i) = sum_{v en N_k(u)} sim(u,v) * r_vi
    `k` es el número de vecinos que se conservan; `usar_vistas` incluye vistas/favoritos con su peso.
    """
    def __init__(self, k=20, variante="item", usar_vistas=False, usar_peso=True):
        self.k, self.variante, self.usar_vistas, self.usar_peso = k, variante, usar_vistas, usar_peso
        self.nombre = f"KNN {'ítem' if variante == 'item' else 'usuario'}"

    def fit(self, datos: Particion):
        self.datos = datos
        R = datos.matriz(solo_positivos=not self.usar_vistas, usar_peso=self.usar_peso)
        Rn = normalize(R, axis=0) if self.variante == "item" else normalize(R, axis=1)
        if self.variante == "item":
            S = (Rn.T @ Rn).toarray()
            self.S = _top_k_por_fila(S, self.k)              # vecinos de cada curso
            self.scores = (R @ self.S.T)
        else:
            S = (Rn @ Rn.T).toarray()
            self.S = _top_k_por_fila(S, self.k)              # vecinos de cada usuario
            self.scores = self.S @ R.toarray()
        self.scores = np.asarray(self.scores)
        return self

    def puntajes(self, user_id):
        return self.scores[self.datos.uidx[user_id]]


# --- LightFM híbrido ---------------------------------------------------------

def atributos_usuario(u: pd.Series) -> list:
    f = [f"carrera:{u['carrera']}", f"nivel:{u['nivel']}", f"idioma:{u['idioma_preferido']}",
         f"objetivo:{u['objetivo_laboral']}", f"ciclo:{'5-6' if u['ciclo'] <= 6 else ('7-8' if u['ciclo'] <= 8 else '9-10')}"]
    f += [f"interes:{x.strip()}" for x in str(u["intereses"]).split(";") if x.strip()]
    f += [f"domina:{x.strip()}" for x in str(u["habilidades_actuales"]).split(";") if x.strip()]
    return f


def atributos_curso(c: pd.Series) -> list:
    return [f"categoria:{c['categoria']}", f"dificultad:{c['dificultad']}", f"idioma:{c['idioma']}",
            f"plataforma:{c['plataforma']}", f"tipo:{c['tipo']}",
            f"duracion:{'corta' if c['duracion_horas'] <= 5 else ('media' if c['duracion_horas'] <= 40 else 'larga')}"]


class LightFMHibrido:
    """
    LightFM (Kula, 2015): factorización matricial híbrida. Cada usuario y cada curso se representan como
    la suma de los embeddings de sus atributos (más un atributo identidad propio). El puntaje es
    el producto interno de ambas representaciones más los sesgos. Se entrena con pérdida de ranking
    (WARP o BPR) usando solo interacciones implícitas.
    """
    nombre = "LightFM híbrido"

    def __init__(self, no_components=32, loss="warp", epochs=30, learning_rate=0.05,
                 item_alpha=1e-6, user_alpha=1e-6, usar_atributos=True, usar_peso=True, seed=42):
        self.p = dict(no_components=no_components, loss=loss, learning_rate=learning_rate,
                      item_alpha=item_alpha, user_alpha=user_alpha, random_state=seed)
        self.epochs, self.usar_atributos, self.usar_peso = epochs, usar_atributos, usar_peso

    def fit(self, datos: Particion):
        from lightfm import LightFM
        from lightfm.data import Dataset
        self.datos = datos
        ds = Dataset()
        uf = datos.usuarios.apply(atributos_usuario, axis=1)
        itf = datos.cursos.apply(atributos_curso, axis=1)
        ds.fit(users=datos.usuarios.user_id, items=datos.cursos.course_id,
               user_features=sorted({x for l in uf for x in l}) if self.usar_atributos else None,
               item_features=sorted({x for l in itf for x in l}) if self.usar_atributos else None)
        pos = datos.train[datos.train.positivo == 1]
        w = pos["peso"].astype(float).values if self.usar_peso else np.ones(len(pos))
        inter, pesos = ds.build_interactions(zip(pos.user_id, pos.course_id, w))
        self.ds = ds
        if self.usar_atributos:
            self.UF = ds.build_user_features(zip(datos.usuarios.user_id, uf))
            self.IF = ds.build_item_features(zip(datos.cursos.course_id, itf))
        else:
            self.UF = self.IF = None
        self.model = LightFM(**self.p)
        self.model.fit(inter, sample_weight=pesos,
                       user_features=self.UF, item_features=self.IF, epochs=self.epochs, num_threads=1)
        # mapeo interno de LightFM (coincide con el orden de datos, pero se verifica)
        umap, _, imap, _ = ds.mapping()
        self._u = np.array([umap[u] for u in datos.usuarios.user_id])
        self._i = np.array([imap[c] for c in datos.cursos.course_id])
        self._cache = {}
        return self

    def puntajes(self, user_id):
        if user_id not in self._cache:
            uid = int(self._u[self.datos.uidx[user_id]])
            self._cache[user_id] = self.model.predict(uid, self._i, user_features=self.UF,
                                                      item_features=self.IF, num_threads=1)
        return self._cache[user_id]

    # ---- Cold start y exportación para la app ----
    def vector_usuario_nuevo(self, perfil: dict):
        """Representación de un usuario nuevo solo con atributos (sin columna identidad)."""
        _, ufmap, _, _ = self.ds.mapping()
        feats = [f for f in atributos_usuario(pd.Series(perfil)) if f in ufmap]
        x = np.zeros(len(ufmap))
        for f in feats:
            x[ufmap[f]] = 1.0 / len(feats)          # misma normalización que Dataset.build_user_features
        return x

    def puntajes_perfil(self, perfil: dict):
        ib, ie = self.model.get_item_representations(self.IF)
        x = self.vector_usuario_nuevo(perfil)
        ue = x @ self.model.user_embeddings
        ub = x @ self.model.user_biases
        return (ie @ ue + ib + ub)[self._i]

    def exportar(self):
        """Arrays para calcular puntajes en la app con numpy (sin instalar LightFM)."""
        ub, ue = self.model.get_user_representations(self.UF)
        ib, ie = self.model.get_item_representations(self.IF)
        _, ufmap, _, _ = self.ds.mapping()
        return {"user_emb": ue[self._u], "user_bias": ub[self._u], "item_emb": ie[self._i], "item_bias": ib[self._i],
                "feat_emb": self.model.user_embeddings, "feat_bias": self.model.user_biases,
                "user_feature_names": np.array(sorted(ufmap, key=ufmap.get))}


# ---------------------------------------------------------------------------
# 4. Cold start por reglas (respaldo)
# ---------------------------------------------------------------------------

def mascara_perfil(cursos, perfil: dict, margen_nivel=1) -> np.ndarray:
    """Reglas de negocio: solo cursos en español si el estudiante no lee inglés y
    sin cursos más de `margen_nivel` niveles por encima de su nivel declarado."""
    niv = {"Básico": 0, "Intermedio": 1, "Avanzado": 2}
    ok = cursos["dificultad"].map(niv).values <= niv[perfil["nivel"]] + margen_nivel
    if perfil.get("idioma_preferido") == "es":
        ok &= cursos["idioma"].eq("es").values
    return ok


def aplicar_reglas(puntajes, cursos, perfil: dict, margen_nivel=1) -> np.ndarray:
    """Devuelve los puntajes con -inf en los cursos que no cumplen las reglas del perfil."""
    return np.where(mascara_perfil(cursos, perfil, margen_nivel), puntajes, -np.inf)


def popularidad_filtrada(pop_scores, cursos, perfil: dict, n=10, margen_nivel=1):
    """Respaldo para usuarios nuevos: cursos populares que cumplen las reglas del perfil."""
    s = aplicar_reglas(pop_scores, cursos, perfil, margen_nivel)
    return np.lexsort((np.arange(len(s)), -s))[:n]
