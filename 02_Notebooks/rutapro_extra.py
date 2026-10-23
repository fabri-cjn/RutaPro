"""
rutapro_extra.py: extensiones de rigor para RutaPro (no modifica rutapro_rec.py).

    construir_pares        reproduce la logica del EDA (positivo, pesos, particiones) para CUALQUIER dataset simulado
    Aleatorio              baseline piso
    PopularidadSegmento    baseline "personalizado simple": popularidad dentro del segmento del estudiante
    ConReglas              aplica las reglas de idioma y nivel sobre cualquier modelo
    Oraculo                techo de referencia que conoce las reglas del simulador (no el azar del muestreo)
    ic_bootstrap, tabla_con_ic, comparar_pareado, holm      estadistica de la comparacion
    evaluar_semillas       promedia por estudiante varias semillas de un modelo estocastico
    particion_cold_start   deja estudiantes fuera de TODO el entrenamiento (cold start sin trampa)
    exposicion, perdida_por_reglas, ndcg_por_grupo          metricas de sesgo y equidad
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import rutapro_rec as rr


# ---------------------------------------------------------------------------
# 1. Pares usuario-curso y particiones (misma logica que 01_EDA.ipynb)
# ---------------------------------------------------------------------------
def construir_pares(inter: pd.DataFrame, min_pos: int = 3, q_global=(0.70, 0.85)) -> pd.DataFrame:
    """Una fila por par usuario-curso con positivo (=inscripcion), peso implicito y dos particiones:
    `particion` (corte temporal por usuario, esquema B) y `particion_global` (corte unico, esquema A)."""
    inter = inter.copy()
    inter["timestamp"] = pd.to_datetime(inter["timestamp"])
    par = (inter.sort_values("timestamp").groupby(["user_id", "course_id"])
           .agg(primer_evento=("timestamp", "min"), peso=("peso_evento", "max"),
                calificacion=("calificacion", "max")).reset_index())
    ts_ins = (inter[inter.evento == "inscripcion"].groupby(["user_id", "course_id"])["timestamp"]
              .min().rename("fecha_positivo"))
    par = par.merge(ts_ins, on=["user_id", "course_id"], how="left")
    par["positivo"] = par["fecha_positivo"].notna().astype(int)
    par["fecha_evento"] = par["fecha_positivo"].fillna(par["primer_evento"])

    t1, t2 = par.loc[par.positivo == 1, "fecha_evento"].quantile(list(q_global)).dt.floor("D")
    par["particion_global"] = np.select([par.fecha_evento < t1, par.fecha_evento < t2], ["train", "valid"], "test")

    par = par.sort_values(["user_id", "fecha_evento"]).reset_index(drop=True)
    par["particion"] = "train"
    cortes = {}
    for uid, g in par[par.positivo == 1].groupby("user_id"):
        n = len(g)
        if n < min_pos:
            continue
        n_test = max(1, int(round(0.2 * n)))
        cortes[uid] = (g["fecha_evento"].iloc[n - n_test - 1], g["fecha_evento"].iloc[n - n_test])
    for uid, (fv, ft) in cortes.items():
        m = par.user_id == uid
        par.loc[m & (par.fecha_evento >= fv), "particion"] = "valid"
        par.loc[m & (par.fecha_evento >= ft), "particion"] = "test"
    return par


# ---------------------------------------------------------------------------
# 2. Baselines nuevos y reglas
# ---------------------------------------------------------------------------
class Aleatorio:
    """Piso: puntajes aleatorios reproducibles por (semilla, estudiante)."""
    nombre = "Aleatorio"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, datos):
        self.datos = datos
        return self

    def puntajes(self, user_id):
        return np.random.default_rng([self.seed, self.datos.uidx[user_id]]).random(self.datos.n_i)


class PopularidadSegmento:
    """
    Baseline personalizado simple: cursos mas inscritos DENTRO del segmento del estudiante
    (por ejemplo su objetivo laboral), suavizado con la popularidad global:
        puntaje_i = inscripciones_segmento_i + m * proporcion_global_i
    `m` (pseudo-conteo) se ajusta en validacion como cualquier otro hiperparametro.
    """
    def __init__(self, cols=("objetivo_laboral",), m=10.0):
        self.cols, self.m = tuple(cols), m
        self.nombre = "Popularidad por " + " + ".join(self.cols)

    def _llave(self, fila):
        return tuple(fila[c] for c in self.cols)

    def fit(self, datos):
        self.datos = datos
        t = datos.train[datos.train.positivo == 1].merge(datos.usuarios[["user_id", *self.cols]], on="user_id")
        n_i = datos.n_i
        glob = np.bincount(t.course_id.map(datos.iidx), minlength=n_i).astype(float)
        self.glob = glob / max(glob.sum(), 1)
        self.seg = {}
        for llave, g in t.groupby(list(self.cols)):
            llave = llave if isinstance(llave, tuple) else (llave,)
            self.seg[llave] = np.bincount(g.course_id.map(datos.iidx), minlength=n_i).astype(float)
        self.u_llave = datos.usuarios.set_index("user_id")[list(self.cols)].apply(tuple, axis=1).to_dict()
        if {"plataforma", "n_resenas"} <= set(datos.cursos.columns):       # RutaPro: desempate por resenas dentro de la plataforma
            pct = datos.cursos.groupby("plataforma")["n_resenas"].rank(pct=True).values
        else:                                                              # otros datasets: sin desempate externo
            pct = np.zeros(n_i)
        self._desempate = 0.001 * pct
        return self

    def _s(self, llave):
        return self.seg.get(llave, np.zeros_like(self.glob)) + self.m * self.glob + self._desempate

    def puntajes(self, user_id):
        return self._s(self.u_llave[user_id])

    def puntajes_perfil(self, perfil: dict):
        return self._s(self._llave(perfil))


class ConReglas:
    """Aplica las reglas de idioma y nivel (rr.aplicar_reglas) sobre los puntajes de otro modelo."""
    def __init__(self, base, cursos, perfil_de, nombre=None):
        self.base, self.cursos, self.perfil_de = base, cursos, perfil_de
        self.nombre = nombre or getattr(base, "nombre", "modelo") + " + reglas"

    def puntajes(self, u):
        return rr.aplicar_reglas(self.base.puntajes(u), self.cursos, self.perfil_de[u])


class SoloPerfil:
    """Ignora el historial y puntua solo con el perfil declarado (estudiante nuevo)."""
    def __init__(self, fn, perfil_de, nombre=""):
        self.fn, self.perfil_de, self.nombre = fn, perfil_de, nombre

    def puntajes(self, u):
        return self.fn(self.perfil_de[u])


class Oraculo:
    """
    Techo de referencia. Para cada inscripcion de prueba usa la probabilidad REAL del simulador de que ESE paso
    terminara en inscripcion en cada curso (P(elegir c) x P(inscribirse | c), dado el nivel efectivo y el estado
    del estudiante en ese momento). Condicionado a que el paso termino en inscripcion, la distribucion del curso es
    proporcional a ese vector, asi que es el ranking optimo por paso. Si un estudiante tiene varias inscripciones de
    prueba se promedian sus vectores. Conoce las reglas y el momento exacto, pero NO el sorteo aleatorio: ningun
    modelo realista puede igualarlo. Requiere rutapro_sim.generar_dataset(con_traza=True).
    """
    nombre = "Oráculo del simulador"

    def __init__(self, traza):
        self.traza = traza

    def fit(self, datos):
        self.datos = datos
        rel = datos.relevantes()
        self.S = {}
        for u, items in rel.items():
            vs = [v for c, ins, v in self.traza.get(u, []) if ins and c in items]
            if vs:
                self.S[u] = np.mean(vs, axis=0)
        return self

    def puntajes(self, u):
        return self.S.get(u, np.zeros(self.datos.n_i))


class MezclaRRF:
    """
    Modelo hibrido por FUSION DE RANKINGS (Reciprocal Rank Fusion) de dos recomendadores `a` y `b`:

        puntaje(u, i) = w / (k + rango_a(u, i)) + (1 - w) / (k + rango_b(u, i))

    Se fusionan RANGOS y no puntajes porque los puntajes de LightFM (sin escala) y de KNN (similitudes, con muchos ceros)
    no son comparables. Los cursos ya tomados por el estudiante se mandan al final de ambos rankings antes de fusionar,
    y los empates (los ceros de KNN) reciben el rango promedio del bloque, asi no inyectan cursos arbitrarios arriba.
    `w` pesa a `a` (w = 1 reproduce a `a`; w = 0 reproduce a `b`). `k` suaviza el peso de las primeras posiciones.
    Para estudiantes SIN historial (`puntajes_perfil`) usa solo `a`: KNN no puede atender el cold start.
    `preajustados=True` reutiliza modelos ya entrenados (por ejemplo para ajustar `w` en validacion sin reentrenar).
    """
    def __init__(self, a, b, w=0.5, k=20, preajustados=False, nombre=None):
        self.a, self.b, self.w, self.k, self.preajustados = a, b, float(w), int(k), preajustados
        self.nombre = nombre or f"{getattr(a, 'nombre', 'A')} + {getattr(b, 'nombre', 'B')}"

    def fit(self, datos):
        self.datos = datos
        if not self.preajustados:
            self.a.fit(datos)
            self.b.fit(datos)
        return self

    def _rangos(self, s, vistos):
        from scipy.stats import rankdata
        s = np.asarray(s, dtype=float).copy()
        if len(vistos):
            s[np.asarray(vistos, dtype=int)] = -np.inf            # lo ya tomado, al final
        return rankdata(-s, method="average")                      # 1 = mejor; empates con rango promedio

    def puntajes(self, user_id):
        vistos = self.datos.vistos(user_id)
        ra = self._rangos(self.a.puntajes(user_id), vistos)
        rb = self._rangos(self.b.puntajes(user_id), vistos)
        return self.w / (self.k + ra) + (1 - self.w) / (self.k + rb)

    def puntajes_perfil(self, perfil):
        return self.a.puntajes_perfil(perfil)


# ---------------------------------------------------------------------------
# 3. Estadistica de la comparacion
# ---------------------------------------------------------------------------
def ic_bootstrap(x, n_boot=5000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return x.mean(), np.percentile(m, 100 * alpha / 2), np.percentile(m, 100 * (1 - alpha / 2))


def tabla_con_ic(dets: dict, K=10, metricas=("precision", "recall", "hit_rate", "ndcg"), n_boot=5000):
    """Media e IC95 bootstrap (remuestreo de estudiantes) de cada metrica y modelo."""
    filas = []
    for nombre, d in dets.items():
        x = d[d.K == K].set_index("user_id")
        fila = {"modelo": nombre, "estudiantes": len(x)}
        for m in metricas:
            mu, lo, hi = ic_bootstrap(x[m].values, n_boot=n_boot)
            fila[m] = mu
            fila[m + " IC95"] = f"[{lo:.3f}, {hi:.3f}]"
        filas.append(fila)
    return pd.DataFrame(filas).set_index("modelo")


def comparar_pareado(det_a, det_b, K=10, metrica="ndcg", n_boot=5000, seed=0):
    """Diferencia pareada (a - b) por estudiante: media, IC95 bootstrap, Wilcoxon y conteo de mejoras/empeoras."""
    a = det_a[det_a.K == K].set_index("user_id")[metrica]
    b = det_b[det_b.K == K].set_index("user_id")[metrica].reindex(a.index)
    d = (a - b).dropna()
    mu, lo, hi = ic_bootstrap(d.values, n_boot=n_boot, seed=seed)
    nz = d[d != 0]
    p = stats.wilcoxon(nz).pvalue if len(nz) > 0 else 1.0
    return {"dif. media": mu, "IC95 inf": lo, "IC95 sup": hi, "mejora": int((d > 0).sum()),
            "empeora": int((d < 0).sum()), "p (Wilcoxon)": p}


def holm(pvals: pd.Series) -> pd.Series:
    """Correccion de Holm-Bonferroni (p ajustados, monotonos)."""
    o = pvals.sort_values()
    m = len(o)
    aj = pd.Series([min(1.0, p * (m - i)) for i, p in enumerate(o.values)], index=o.index).cummax()
    return aj.reindex(pvals.index)


def promediar_det(dets: list) -> pd.DataFrame:
    """Promedia por estudiante y K el detalle de varias semillas (misma poblacion de estudiantes)."""
    return pd.concat(dets).groupby(["user_id", "K"], as_index=False).mean(numeric_only=True)


def evaluar_semillas(fabrica, datos, semillas=(42, 43, 44, 45, 46), ks=(5, 10), envolver=None):
    """
    Entrena `fabrica(seed)` con cada semilla y evalua. Devuelve:
      det_prom  : detalle por estudiante PROMEDIADO entre semillas (para IC y pruebas pareadas)
      res_sem   : metricas globales por semilla (para media y desviacion entre semillas)
    `envolver(modelo)` permite aplicar reglas u otra capa antes de evaluar.
    """
    dets, resumenes = [], []
    for s in semillas:
        m = fabrica(s).fit(datos)
        if envolver:
            m = envolver(m)
        res, det, _ = rr.evaluar(m, datos, ks=ks, devolver_detalle=True)
        dets.append(det)
        resumenes.append(res.assign(semilla=s).reset_index())
    return promediar_det(dets), pd.concat(resumenes, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Cold start sin trampa
# ---------------------------------------------------------------------------
def particion_cold_start(cursos, usuarios, pares, fuera: set, esquema="usuario", objetivo="test"):
    """
    Particion donde los estudiantes de `fuera` NO aportan NINGUN evento al entrenamiento (ni embeddings de curso,
    ni popularidad, ni vecinos). Se entrena con train+valid de los demas. `objetivo='test'` evalua las mismas
    inscripciones de prueba que el protocolo con historial (comparacion directa con/sin historial);
    `objetivo='todo'` evalua todas las inscripciones del estudiante retenido.
    """
    col = "particion" if esquema == "usuario" else "particion_global"
    tr = pares[pares[col].isin(["train", "valid"]) & ~pares.user_id.isin(fuera)]
    ev = pares[(pares.positivo == 1) & pares.user_id.isin(fuera)]
    if objetivo == "test":
        ev = ev[ev[col] == "test"]
    return rr.Particion(cursos, usuarios, tr.copy(), ev.copy())


def pliegues_usuarios(user_ids, n_pliegues=5, seed=0):
    ids = np.array(sorted(user_ids))
    np.random.default_rng(seed).shuffle(ids)
    return [set(ids[k::n_pliegues]) for k in range(n_pliegues)]


# ---------------------------------------------------------------------------
# 5. Sesgos y equidad
# ---------------------------------------------------------------------------
def exposicion(recs: dict, datos, k=10) -> dict:
    """Concentracion de la exposicion: cobertura, Gini de exposicion y % de recomendaciones en el 20 % mas popular."""
    pop = np.asarray(datos.matriz(solo_positivos=True, usar_peso=False).sum(axis=0)).ravel()
    top20 = set(np.argsort(-pop)[: int(np.ceil(0.2 * datos.n_i))])
    cont = np.zeros(datos.n_i)
    en_top20 = []
    for r in recs.values():
        for i in r[:k]:
            cont[i] += 1
            en_top20.append(i in top20)
    x = np.sort(cont)
    n = len(x)
    gini = (2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum()) - (n + 1) / n) if x.sum() else np.nan
    return {"cobertura": float((cont > 0).mean()), "Gini de exposición": float(gini),
            "% recs en el 20 % más popular": 100 * float(np.mean(en_top20))}


def perdida_por_reglas(datos, perfil_de, cursos, usuarios=None, margen_nivel=1) -> pd.DataFrame:
    """
    Costo de las reglas de negocio: de las inscripciones REALES de prueba, cuantas quedan vetadas por las reglas
    (curso en ingles para quien solo estudia en espanol, o mas de `margen_nivel` niveles por encima).
    Es el recall maximo que se pierde por diseno.
    """
    rel = datos.relevantes()
    usuarios = usuarios if usuarios is not None else datos.usuarios_elegibles()
    filas = []
    for u in usuarios:
        ok = rr.mascara_perfil(cursos, perfil_de[u], margen_nivel)
        items = list(rel[u])
        filas.append({"user_id": u, "idioma_preferido": perfil_de[u]["idioma_preferido"],
                      "inscripciones_prueba": len(items), "vetadas": int((~ok[items]).sum())})
    d = pd.DataFrame(filas)
    out = d.groupby("idioma_preferido")[["inscripciones_prueba", "vetadas"]].sum()
    out.loc["Todos"] = d[["inscripciones_prueba", "vetadas"]].sum()
    out["% vetado"] = 100 * out["vetadas"] / out["inscripciones_prueba"]
    return out


def ndcg_por_grupo(det: pd.DataFrame, usuarios: pd.DataFrame, columna: str, K=10, metrica="ndcg") -> pd.DataFrame:
    d = det[det.K == K].merge(usuarios[["user_id", columna]], on="user_id")
    g = d.groupby(columna)[metrica].agg(["mean", "count"])
    return g.rename(columns={"mean": metrica, "count": "estudiantes"})


def puntajes_cold_filtrado(lfm, perfil: dict, prefijos=("carrera", "nivel", "idioma", "objetivo", "ciclo", "interes", "domina")):
    """Cold start de LightFM usando solo algunos campos del formulario (prefijos de atributo). Con todos los
    prefijos coincide con LightFMHibrido.puntajes_perfil. Sirve para medir cuantas preguntas necesita el formulario."""
    _, ufmap, _, _ = lfm.ds.mapping()
    feats = [f for f in rr.atributos_usuario(pd.Series(perfil)) if f in ufmap and f.split(":")[0] in prefijos]
    x = np.zeros(len(ufmap))
    for f in feats:
        x[ufmap[f]] = 1.0 / len(feats)
    ib, ie = lfm.model.get_item_representations(lfm.IF)
    return (ie @ (x @ lfm.model.user_embeddings) + ib + x @ lfm.model.user_biases)[lfm._i]


# ---------------------------------------------------------------------------
# 6. Experimentos sobre datasets simulados (replicas y escenarios)
# ---------------------------------------------------------------------------
def _elegir(filas, orden, tol=0.01, tol_rel=None):
    """Regla de parsimonia del notebook 02: dentro de la tolerancia del mejor NDCG@10, la configuracion mas simple.
    `tol` es ABSOLUTA (0.01 va bien cuando el NDCG ronda 0.2). `tol_rel` es RELATIVA al mejor (0.05 = 5 %) y es la
    correcta cuando el NDCG es pequeno: con NDCG ~ 0.05 una tolerancia absoluta de 0.01 equivale a 20 % y deja pasar
    cualquier configuracion, eligiendo siempre la mas simple aunque sea claramente peor."""
    g = pd.DataFrame(filas)
    umbral = g.ndcg.max() * (1 - tol_rel) if tol_rel is not None else g.ndcg.max() - tol
    return g[g.ndcg >= umbral].sort_values(orden).iloc[0]


def ajustar_hiperparametros(cursos, usuarios, pares, esquema="global", seed=42, tol=0.01,
                            seg_cols=("objetivo_laboral",), alphas_cont=(0.0, 0.3, 0.6, 0.9), ngram_cont=(1, 1),
                            comps=(32, 64), epocas=(50, 100, 200), tol_rel=None, con_blend=False,
                            pesos_blend=(0.25, 0.5, 0.75)) -> dict:
    """Rejilla pequena, SOLO en validacion, con la regla de parsimonia del notebook 02 (tolerancia absoluta `tol` o
    relativa `tol_rel`). `con_blend=True` ajusta ademas el hibrido LightFM + KNN (variante de KNN y peso w)."""
    val = rr.construir_particion(cursos, usuarios, pares, etapa="validacion", esquema=esquema)
    nd = lambda m: rr.evaluar(m.fit(val), val).loc[10, "ndcg"]
    el = lambda filas, orden: _elegir(filas, orden, tol, tol_rel)
    seg = el([{"m": m, "ndcg": nd(PopularidadSegmento(tuple(seg_cols), m))} for m in (0, 20)], ["m"])
    cont = el([{"alpha": a_, "ndcg": nd(rr.ContenidoTFIDF(ngram=ngram_cont, alpha_perfil=a_))} for a_ in alphas_cont], ["alpha"])
    ku = el([{"k": k, "ndcg": nd(rr.KNNColaborativo(k=k, variante="usuario"))} for k in (10, 20, 50)], ["k"])
    ki = el([{"k": k, "ndcg": nd(rr.KNNColaborativo(k=k, variante="item"))} for k in (5, 20, 50)], ["k"])
    lf = el([{"no_components": c, "epochs": e_, "ndcg": nd(rr.LightFMHibrido(no_components=c, loss="warp", epochs=e_, seed=seed))}
             for c in comps for e_ in epocas], ["no_components", "epochs"])
    hp = {"seg_m": float(seg.m), "cont_alpha": float(cont.alpha), "knn_u_k": int(ku.k), "knn_i_k": int(ki.k),
          "lfm": {"no_components": int(lf.no_components), "loss": "warp", "epochs": int(lf.epochs)}}
    if con_blend:
        lfm_v = rr.LightFMHibrido(**hp["lfm"], seed=seed).fit(val)
        knn_v = {"usuario": rr.KNNColaborativo(k=hp["knn_u_k"], variante="usuario").fit(val),
                 "item": rr.KNNColaborativo(k=hp["knn_i_k"], variante="item").fit(val)}
        filas = []
        for var, knn in knn_v.items():
            for w in pesos_blend:
                filas.append({"variante": var, "w": w, "dist": abs(w - 0.5),
                              "ndcg": rr.evaluar(MezclaRRF(lfm_v, knn, w=w, preajustados=True).fit(val), val).loc[10, "ndcg"]})
        bl = el(filas, ["dist", "variante"])             # parsimonia: el peso mas cercano a 0.5 dentro de la tolerancia
        hp["blend"] = {"variante": str(bl.variante), "w": float(bl.w), "k_rrf": 20}
        hp["blend_rejilla"] = [{k_: (float(v) if k_ in ("w", "ndcg") else v) for k_, v in f.items() if k_ != "dist"} for f in filas]
    return hp


def evaluar_dataset(cursos, usuarios, inter, traza, esquema, hp, seed=42, ks=(5, 10),
                    seg_cols=("objetivo_laboral",), con_reglas=True, ngram_cont=(1, 1)):
    """Entrena con train+valid y evalua una vez en prueba. Devuelve (dets por modelo, resumen K=10, n evaluables).
    traza=None omite el oraculo (datos reales); con_reglas=False omite las reglas de idioma y nivel."""
    pares = construir_pares(inter)
    prueba = rr.construir_particion(cursos, usuarios, pares, etapa="prueba", esquema=esquema)
    perfil_de = usuarios.set_index("user_id").to_dict("index")
    lfm = rr.LightFMHibrido(**hp["lfm"], seed=seed).fit(prueba)
    seg = PopularidadSegmento(tuple(seg_cols), hp["seg_m"]).fit(prueba)
    modelos = {"Popularidad": rr.Popularidad().fit(prueba)}
    modelos["Pop. por segmento + reglas" if con_reglas else "Pop. por segmento"] = \
        ConReglas(seg, cursos, perfil_de) if con_reglas else seg
    modelos["Contenido (TF-IDF)"] = rr.ContenidoTFIDF(ngram=ngram_cont, alpha_perfil=hp["cont_alpha"]).fit(prueba)
    modelos["KNN usuario"] = rr.KNNColaborativo(k=hp["knn_u_k"], variante="usuario").fit(prueba)
    modelos["KNN ítem"] = rr.KNNColaborativo(k=hp["knn_i_k"], variante="item").fit(prueba)
    modelos["LightFM híbrido"] = lfm
    if hp.get("blend"):                       # hibrido LightFM + KNN (reutiliza los modelos ya entrenados)
        knn_b = modelos["KNN usuario"] if hp["blend"]["variante"] == "usuario" else modelos["KNN ítem"]
        modelos["LightFM + KNN"] = MezclaRRF(lfm, knn_b, w=hp["blend"]["w"], k=hp["blend"].get("k_rrf", 20), preajustados=True).fit(prueba)
    if con_reglas:
        modelos["LightFM + reglas"] = ConReglas(lfm, cursos, perfil_de)
    if traza is not None:
        modelos["Oráculo"] = Oraculo(traza).fit(prueba)
    dets, filas = {}, []
    for n, m in modelos.items():
        res, det, _ = rr.evaluar(m, prueba, ks=ks, devolver_detalle=True)
        dets[n] = det
        filas.append({"modelo": n, **res.loc[10, ["precision", "recall", "hit_rate", "ndcg", "cobertura"]].to_dict()})
    return dets, pd.DataFrame(filas).set_index("modelo"), len(prueba.usuarios_elegibles())


def descriptivos_dataset(usuarios, inter, cursos) -> dict:
    """Indicadores de plausibilidad del dataset simulado (embudo, concentracion, idioma)."""
    pares_ev = inter.groupby(["user_id", "course_id"])["evento"].agg(set)
    n_ins = pares_ev.map(lambda s: "inscripcion" in s)
    n_fin = pares_ev.map(lambda s: "finalizacion" in s)
    ins = inter[inter.evento == "inscripcion"].merge(usuarios[["user_id", "idioma_preferido"]], on="user_id") \
        .merge(cursos[["course_id", "idioma"]], on="course_id")
    cnt = ins.groupby("course_id").size().reindex(cursos.course_id, fill_value=0).sort_values().values.astype(float)
    n = len(cnt)
    gini = 2 * np.sum(np.arange(1, n + 1) * cnt) / (n * cnt.sum()) - (n + 1) / n
    solo_es = ins[ins.idioma_preferido == "es"]
    return {"estudiantes": len(usuarios), "eventos": len(inter), "inscripciones": int(n_ins.sum()),
            "inscripciones por estudiante": n_ins.sum() / len(usuarios),
            "% vista → inscripción": 100 * n_ins.mean(), "% inscripción → finalización": 100 * n_fin.sum() / max(n_ins.sum(), 1),
            "Gini de inscripciones por curso": gini,
            "% inscripciones en inglés de 'solo español'": 100 * (solo_es.idioma == "en").mean()}


def correr_escenario(cursos, nombre, P0, R=10, cache=None, verbose=True):
    """Ajusta hiperparametros (dataset aparte, solo validacion) y corre R replicas independientes del escenario con corte
    global y por usuario. Si `cache` (ruta .pkl) existe se reutiliza; si no, se calcula y se guarda."""
    import pickle, time
    from pathlib import Path
    from scipy import stats as _st
    import rutapro_sim as _sim
    if cache and Path(cache).exists():
        return pickle.load(open(cache, "rb"))
    t0 = time.time()
    u_aj, i_aj, _ = _sim.generar_dataset(cursos, P0.con(seed=1000))
    hp = ajustar_hiperparametros(cursos, u_aj, construir_pares(i_aj), esquema="global")
    filas, dif = [], []
    for r in range(R):
        P = P0.con(seed=2000 + 10 * r)
        usu, inter, traza = _sim.generar_dataset(cursos, P, con_traza=True)
        for esq in ("global", "usuario"):
            dets, res, n_eval = evaluar_dataset(cursos, usu, inter, traza, esq, hp, seed=42)
            for m, fila in res.iterrows():
                filas.append({"escenario": nombre, "esquema": esq, "replica": r, "modelo": m, "evaluables": n_eval, **fila.to_dict()})
            for a, b in [("LightFM + reglas", "Pop. por segmento + reglas"), ("LightFM + reglas", "Popularidad"),
                         ("Pop. por segmento + reglas", "Popularidad")]:
                x = dets[a].query("K == 10").set_index("user_id")["ndcg"]
                y = dets[b].query("K == 10").set_index("user_id")["ndcg"]
                d = (x - y.reindex(x.index)).dropna(); nz = d[d != 0]
                dif.append({"escenario": nombre, "esquema": esq, "replica": r, "comparacion": f"{a} − {b}", "dif": d.mean(),
                            "p": _st.wilcoxon(nz).pvalue if len(nz) else 1.0})
    out = (pd.DataFrame(filas), pd.DataFrame(dif), hp)
    if cache:
        Path(cache).parent.mkdir(exist_ok=True)
        pickle.dump(out, open(cache, "wb"))
    if verbose:
        print(f"{nombre}: {time.time() - t0:.0f} s | hiperparametros: {hp}", flush=True)
    return out
