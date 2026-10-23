"""
rutapro_sim.py: simulador parametrizable de estudiantes e interacciones de RutaPro.

Con los parametros por defecto (ParamsSim()) y seed=42 reproduce EXACTAMENTE los archivos
usuarios.csv e interacciones.csv del dataset v1 (notebook 00, secciones 4 y 5). Eso se verifica en
04_Robustez_Baselines.ipynb. Los demas parametros permiten escenarios de sensibilidad:

    w_afinidad, w_interes, w_pop    pesos de la utilidad (perfil vs popularidad)
    w_secuencia                     bonus por "ruta de aprendizaje tipica" (curso siguiente segun el anterior)
    w_latente                       gustos individuales NO observables en los atributos del estudiante
    p_explora, temperatura          ruido de la eleccion
    p_ins_base, p_ins_pend, p_fin_base   embudo vista -> inscripcion -> finalizacion
    n_usuarios, ventana_registro_dias, ritmo_dias   tamano y calendario

Ademas puede devolver una traza con la probabilidad real de inscripcion de cada curso en cada paso,
que se usa para construir el "oraculo" (techo de referencia que conoce las reglas del simulador).

Todo supuesto es del equipo y esta declarado: los datos siguen siendo SIMULADOS.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Supuestos del simulador (identicos al notebook 00)
# ---------------------------------------------------------------------------
CATS = ["Python", "SQL y Bases de Datos", "Excel", "Power BI", "Tableau y Visualización", "Estadística", "R",
        "Machine Learning", "Análisis de Datos", "Gestión de Proyectos", "Habilidades Blandas"]

AFINIDAD = pd.DataFrame({
    #                      Py   SQL  Exc  PBI  Tab  Est  R    ML   AD   GP   HB
    "Data Analyst":       [.9, 1.0, .9,  .8,  .7,  .8,  .5,  .3,  1.0, .2,  .5],
    "BI Analyst":         [.4, 1.0, .9,  1.0, 1.0, .5,  .2,  .2,  .8,  .3,  .5],
    "Data Scientist":     [1.0, .7, .3,  .3,  .3,  1.0, .7,  1.0, .7,  .1,  .3],
    "Analista Financiero": [.5, .6,  1.0, .8,  .5,  .6,  .3,  .3,  .7,  .4,  .5],
    "Marketing Analyst":  [.5, .6,  .8,  .7,  .8,  .7,  .4,  .4,  .9,  .3,  .7],
    "Project Analyst":    [.2, .4,  .8,  .6,  .4,  .3,  .1,  .1,  .5,  1.0, 1.0],
}, index=CATS).T

CARRERAS = {"Estadística e Informática": .25, "Economía": .20, "Ingeniería Industrial": .20,
            "Ingeniería de Sistemas": .15, "Administración": .20}
OBJETIVOS = list(AFINIDAD.index)
P_OBJETIVO = pd.DataFrame({
    #                            DA   BI   DS   AF   MK   PA
    "Estadística e Informática": [.35, .15, .35, .05, .05, .05],
    "Economía":                  [.25, .15, .10, .30, .15, .05],
    "Ingeniería Industrial":     [.25, .25, .05, .10, .05, .30],
    "Ingeniería de Sistemas":    [.25, .25, .35, .00, .00, .15],
    "Administración":            [.15, .15, .00, .25, .25, .20],
}, index=OBJETIVOS).T
BASE_CARRERA = {
    "Estadística e Informática": ["Estadística", "R", "Python", "Excel"],
    "Economía": ["Excel", "Estadística"],
    "Ingeniería Industrial": ["Excel", "Gestión de Proyectos", "Estadística"],
    "Ingeniería de Sistemas": ["Python", "SQL y Bases de Datos"],
    "Administración": ["Excel", "Habilidades Blandas"],
}
NIVELES = ["Básico", "Intermedio", "Avanzado"]
NIV_NUM = {"Básico": 0, "Intermedio": 1, "Avanzado": 2}
P_NIVEL = {5: [.70, .25, .05], 6: [.60, .32, .08], 7: [.40, .45, .15],
           8: [.30, .50, .20], 9: [.20, .50, .30], 10: [.15, .45, .40]}
PESO = {"vista": 1, "favorito": 2, "inscripcion": 3, "finalizacion": 4}
FECHA_CORTE = pd.Timestamp("2026-10-15")
AJUSTE_NIVEL = np.array([-0.6, -0.2, 0.4, -0.5, -1.5])        # brecha -2, -1, 0, +1, +2

# Ruta de aprendizaje tipica (SOLO para el escenario secuencial): categoria previa -> categorias que suelen seguir.
SIGUE = {
    "Excel": ["SQL y Bases de Datos", "Power BI", "Análisis de Datos"],
    "SQL y Bases de Datos": ["Power BI", "Python", "Análisis de Datos", "Tableau y Visualización"],
    "Python": ["Machine Learning", "Estadística", "Análisis de Datos"],
    "Estadística": ["R", "Machine Learning", "Python"],
    "R": ["Machine Learning", "Estadística"],
    "Power BI": ["Tableau y Visualización", "Análisis de Datos"],
    "Tableau y Visualización": ["Power BI", "Análisis de Datos"],
    "Análisis de Datos": ["Machine Learning", "Power BI", "Python"],
    "Machine Learning": ["Python", "Estadística"],
    "Gestión de Proyectos": ["Habilidades Blandas", "Excel"],
    "Habilidades Blandas": ["Gestión de Proyectos"],
}


@dataclass
class ParamsSim:
    nombre: str = "Base (v1)"
    seed: int = 42
    n_usuarios: int = 150
    ventana_registro_dias: int = 243      # registros entre el 01/01/2026 y +N dias
    ritmo_dias: float = 6.0               # media (exponencial) de dias entre un curso y el siguiente
    n_cursos_media: float = 13.0          # media de cursos revisados por estudiante (binomial negativa)
    w_afinidad: float = 2.0
    w_interes: float = 0.6
    w_pop: float = 1.5
    w_secuencia: float = 0.0              # 0 = sin ruta de aprendizaje (v1)
    w_latente: float = 0.0                # 0 = sin gustos latentes (v1)
    p_explora: float = 0.15
    temperatura: float = 0.6
    p_favorito: float = 0.25
    p_ins_base: float = 0.30
    p_ins_pend: float = 0.35
    p_fin_base: float = 0.45

    def con(self, **kw) -> "ParamsSim":
        return replace(self, **kw)


def escenarios(n_usuarios=400, ventana=60, ritmo=20.0) -> dict:
    """Escenarios de sensibilidad. Todos usan el mismo tamano y calendario; solo cambia el mecanismo."""
    b = ParamsSim(nombre="Base", n_usuarios=n_usuarios, ventana_registro_dias=ventana, ritmo_dias=ritmo)
    return {
        "E0 Base (reglas v1)": b.con(nombre="E0 Base (reglas v1)"),
        "E1 Perfil débil": b.con(nombre="E1 Perfil débil", w_afinidad=1.0, w_interes=0.3, p_explora=0.30),
        "E2 Popularidad dominante": b.con(nombre="E2 Popularidad dominante", w_afinidad=1.0, w_interes=0.3, w_pop=3.0),
        "E3 Ruta secuencial": b.con(nombre="E3 Ruta secuencial", w_afinidad=1.0, w_interes=0.3, w_secuencia=1.5),
        "E4 Gustos latentes": b.con(nombre="E4 Gustos latentes", w_afinidad=1.0, w_interes=0.3, w_latente=1.5),
        "E5 Embudo MOOC bajo": b.con(nombre="E5 Embudo MOOC bajo", p_ins_base=0.10, p_ins_pend=0.20,
                                      p_fin_base=0.15, n_cursos_media=18.0),
    }


# ---------------------------------------------------------------------------
# Generacion
# ---------------------------------------------------------------------------
def _simular_usuario(i, P: ParamsSim, rng, AF):
    carrera = rng.choice(list(CARRERAS), p=list(CARRERAS.values()))
    ciclo = int(rng.integers(5, 11))
    objetivo = rng.choice(OBJETIVOS, p=P_OBJETIVO.loc[carrera].values)
    nivel = rng.choice(NIVELES, p=P_NIVEL[ciclo])
    idioma = rng.choice(["es", "es+en"], p=[.4, .6])
    w = AF.loc[objetivo].values ** 2
    w = w / w.sum()
    intereses = list(rng.choice(CATS, size=int(rng.integers(2, 4)), replace=False, p=w))
    base = BASE_CARRERA[carrera]
    habilidades = list(rng.choice(base, size=min(len(base), int(rng.integers(1, 4))), replace=False))
    registro = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, P.ventana_registro_dias)))
    return dict(user_id=f"U{i:03d}", carrera=carrera, ciclo=ciclo, nivel=nivel, idioma_preferido=idioma,
                objetivo_laboral=objetivo, intereses="; ".join(intereses),
                habilidades_actuales="; ".join(habilidades), fecha_registro=registro.date())


def generar_dataset(cursos: pd.DataFrame, P: ParamsSim = ParamsSim(), con_traza: bool = False, afinidad: pd.DataFrame = None):
    """Devuelve (usuarios, interacciones, traza). traza = {user_id: [(idx_curso, inscrito, vector_prob_inscripcion), ...]}
    `afinidad` (opcional) reemplaza la matriz objetivo x categoria supuesta por el equipo (por ejemplo, una derivada de avisos reales)."""
    AF = AFINIDAD if afinidad is None else afinidad.loc[OBJETIVOS, CATS]
    rng = np.random.default_rng(P.seed)
    rng_lat = np.random.default_rng(P.seed + 10_000)          # flujo aparte: no altera el v1 cuando w_latente = 0

    usuarios = pd.DataFrame([_simular_usuario(i, P, rng, AF) for i in range(1, P.n_usuarios + 1)])

    cur = cursos.reset_index(drop=True)
    n_c = len(cur)
    cat_idx = cur["categoria"].map({c: k for k, c in enumerate(CATS)}).values
    dif = cur["dificultad"].map(NIV_NUM).values
    es_en = (cur["idioma"] == "en").values
    es_basico = dif == 0
    dur = cur["duracion_horas"].values
    rating = cur["rating"].values
    pop = cur.groupby("plataforma")["n_resenas"].rank(pct=True).values
    ids = cur["course_id"].values
    afin_por_obj = {o: np.array([AF.loc[o, c] for c in cur["categoria"]]) for o in OBJETIVOS}
    sigue_mask = {c: np.isin(cur["categoria"].values, SIGUE[c]) for c in CATS}

    filas, traza = [], {}
    for u in usuarios.itertuples(index=False):
        ints = u.intereses.split("; ")
        dominadas = u.habilidades_actuales.split("; ")
        m_int = np.isin(cur["categoria"].values, ints).astype(float)
        m_dom_basico = np.isin(cur["categoria"].values, dominadas) & es_basico
        solo_es = u.idioma_preferido == "es"
        afin = afin_por_obj[u.objetivo_laboral]
        gusto = rng_lat.normal(0, 1, len(CATS)) if P.w_latente else None

        def utilidad_vec(nivel_ef, cat_prev):
            s = P.w_afinidad * afin
            s = s + P.w_interes * m_int
            s = s + P.w_pop * pop
            s = s + AJUSTE_NIVEL[(dif - nivel_ef) + 2]
            if solo_es:
                s = np.where(es_en, s - 1.5, s)
            s = np.where(m_dom_basico, s - 0.5, s)
            if P.w_secuencia and cat_prev is not None:
                s = s + P.w_secuencia * sigue_mask[cat_prev]
            if P.w_latente:
                s = s + P.w_latente * gusto[cat_idx]
            return s

        n_cursos = max(3, int(rng.negative_binomial(4, 4 / (4 + P.n_cursos_media))))
        nivel_ef = NIV_NUM[u.nivel]
        tomados = np.zeros(n_c, dtype=bool)
        cats_tomadas = np.zeros(len(CATS), dtype=bool)
        cat_prev, completados = None, 0
        t = pd.Timestamp(u.fecha_registro) + pd.Timedelta(days=int(rng.integers(0, 21)), hours=int(rng.integers(8, 23)))
        tr = []
        for _ in range(n_cursos):
            disp = np.flatnonzero(~tomados)
            s_all = utilidad_vec(nivel_ef, cat_prev)
            util_n_all = 1 / (1 + np.exp(-(s_all - 1.5)))
            s = s_all[disp]
            prereq = np.where((dif[disp] == 2) & (~cats_tomadas[cat_idx[disp]]), 0.3, 1.0)
            w = np.exp((s - s.max()) / P.temperatura) * prereq
            p = w / w.sum()
            if rng.random() < P.p_explora:
                c_i = int(rng.choice(disp))
            else:
                c_i = int(rng.choice(disp, p=p))
            if con_traza:
                prob = np.zeros(n_c)
                prob[disp] = P.p_explora / len(disp) + (1 - P.p_explora) * p
                vec_ins = prob * (P.p_ins_base + P.p_ins_pend * util_n_all)
            tomados[c_i] = True
            cats_tomadas[cat_idx[c_i]] = True
            cat_prev = CATS[cat_idx[c_i]]
            util_n = util_n_all[c_i]
            encaja = dif[c_i] == nivel_ef

            filas.append((u.user_id, ids[c_i], t, "vista"))
            if rng.random() < P.p_favorito:
                filas.append((u.user_id, ids[c_i], t + pd.Timedelta(hours=int(rng.integers(1, 48))), "favorito"))
            inscrito = False
            if rng.random() < P.p_ins_base + P.p_ins_pend * util_n:
                inscrito = True
                t_ins = t + pd.Timedelta(hours=int(rng.integers(2, 96)))
                filas.append((u.user_id, ids[c_i], t_ins, "inscripcion"))
                p_fin = P.p_fin_base + 0.25 * encaja - 0.15 * (dur[c_i] > 100)
                if rng.random() < p_fin:
                    semanas = dur[c_i] / 6 * rng.uniform(0.7, 1.4)
                    t_fin = t_ins + pd.Timedelta(days=max(1, int(semanas * 7)))
                    nota = rating[c_i] + rng.normal(0, 0.6) - (0 if encaja else 1.0)
                    filas.append((u.user_id, ids[c_i], t_fin, "finalizacion", int(np.clip(round(nota), 1, 5))))
                    completados += 1
                    if completados % 2 == 0:
                        nivel_ef = min(2, nivel_ef + 1)
            if con_traza:
                tr.append((c_i, inscrito, vec_ins))
            t = t + pd.Timedelta(days=max(1, int(rng.exponential(P.ritmo_dias))), hours=int(rng.integers(0, 24)))
        if con_traza:
            traza[u.user_id] = tr

    inter = pd.DataFrame([f if len(f) == 5 else (*f, np.nan) for f in filas],
                         columns=["user_id", "course_id", "timestamp", "evento", "calificacion"])
    inter = inter[inter["timestamp"] <= FECHA_CORTE].sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    inter["peso_evento"] = inter["evento"].map(PESO)
    inter.insert(0, "interaction_id", [f"I{i:05d}" for i in range(1, len(inter) + 1)])
    return usuarios, inter, traza


def validar(usuarios: pd.DataFrame, inter: pd.DataFrame, cursos: pd.DataFrame) -> pd.Series:
    """Las mismas 9 validaciones de integridad del notebook 00."""
    reg = inter.merge(usuarios[["user_id", "fecha_registro"]], on="user_id")
    primero = inter.sort_values("timestamp").groupby(["user_id", "course_id"])["evento"].first()
    return pd.Series({
        "IDs de usuario únicos": usuarios["user_id"].is_unique,
        "IDs de interacción únicos": inter["interaction_id"].is_unique,
        "Toda interacción apunta a un usuario existente": inter["user_id"].isin(usuarios["user_id"]).all(),
        "Toda interacción apunta a un curso existente": inter["course_id"].isin(cursos["course_id"]).all(),
        "Calificación solo en finalizaciones": inter.loc[inter["evento"] != "finalizacion", "calificacion"].isna().all(),
        "Calificación entre 1 y 5": inter["calificacion"].dropna().between(1, 5).all(),
        "Sin eventos antes del registro": bool((reg["timestamp"] >= pd.to_datetime(reg["fecha_registro"])).all()),
        "Inscripción no ocurre sin vista previa": bool((primero == "vista").all()),
    })
