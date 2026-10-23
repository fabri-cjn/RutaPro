"""
RutaPro · Recomendador de cursos profesionales (prototipo)

Ejecutar desde la carpeta 03_App:
    pip install -r requirements.txt
    streamlit run app.py

Modelo: LightFM híbrido (WARP, 64 factores) + reglas de idioma y nivel.
Los puntajes se calculan con numpy a partir de los embeddings exportados en
02_Notebooks/03_Evaluacion.ipynb, por lo que la app no necesita instalar LightFM.
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DIR = Path(__file__).parent / "modelo"
NIV = {"Básico": 0, "Intermedio": 1, "Avanzado": 2}
CATEGORIAS = ["Python", "SQL y Bases de Datos", "Excel", "Power BI", "Tableau y Visualización", "Estadística", "R",
              "Machine Learning", "Análisis de Datos", "Gestión de Proyectos", "Habilidades Blandas"]
CARRERAS = ["Estadística e Informática", "Economía", "Ingeniería Industrial", "Ingeniería de Sistemas", "Administración"]
OBJETIVOS = ["Data Analyst", "BI Analyst", "Data Scientist", "Analista Financiero", "Marketing Analyst", "Project Analyst"]
# Categorías clave de cada puesto (afinidad ≥ 0.8 en la matriz del proyecto); se usan solo para explicar
CLAVE_OBJETIVO = {
    "Data Analyst": ["SQL y Bases de Datos", "Análisis de Datos", "Python", "Excel", "Power BI", "Estadística"],
    "BI Analyst": ["Power BI", "Tableau y Visualización", "SQL y Bases de Datos", "Excel", "Análisis de Datos"],
    "Data Scientist": ["Python", "Machine Learning", "Estadística"],
    "Analista Financiero": ["Excel", "Power BI"],
    "Marketing Analyst": ["Análisis de Datos", "Excel", "Tableau y Visualización"],
    "Project Analyst": ["Gestión de Proyectos", "Habilidades Blandas", "Excel"],
}

st.set_page_config(page_title="RutaPro · Recomendador de cursos", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.rp-meta {color: #52514e; font-size: 0.86rem; margin: 0.1rem 0 0.4rem 0;}
.rp-chip {display: inline-block; padding: 1px 8px; margin: 0 4px 4px 0; border-radius: 10px;
          font-size: 0.78rem; background: #eef3fb; color: #1d4f8f; border: 1px solid #d6e2f3;}
.rp-chip.es {background: #fdf0ea; color: #8a3a17; border-color: #f4d6c8;}
.rp-why {font-size: 0.84rem; color: #2f2e2b;}
.rp-rank {font-size: 1.6rem; font-weight: 700; color: #2a78d6; line-height: 1;}
</style>""", unsafe_allow_html=True)

# Fondo de la app: dibujo minimalista en la esquina inferior derecha
def poner_fondo(nombre):
    ruta = Path(__file__).parent / nombre
    if not ruta.exists():          # si falta la imagen, la app sigue funcionando sin fondo
        return
    img = base64.b64encode(ruta.read_bytes()).decode()
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/jpeg;base64,{img}") right bottom / cover no-repeat fixed;
    }}
    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
    </style>
    """, unsafe_allow_html=True)

poner_fondo("fondo.jpg")

# ---------------------------------------------------------------------------
# Carga de datos y modelo
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar():
    cursos = pd.read_csv(DIR / "cursos.csv")
    usuarios = pd.read_csv(DIR / "usuarios.csv")
    inscritos = json.loads((DIR / "inscritos.json").read_text(encoding="utf-8"))
    cfg = json.loads((DIR / "config.json").read_text(encoding="utf-8"))
    try:
        m = dict(np.load(DIR / "lightfm_final.npz", allow_pickle=True))
        assert list(m["course_ids"]) == list(cursos.course_id), "El orden de cursos no coincide con el modelo"
    except Exception as e:                       # incidencia: sin modelo → respaldo por popularidad
        st.warning(f"No se pudo cargar el modelo LightFM ({e}). Se usará la popularidad por objetivo laboral (con reglas) como respaldo.")
        m = None
    return cursos, usuarios, inscritos, cfg, m


cursos, usuarios, inscritos, cfg, M = cargar()
POP = M["popularidad"] if M is not None else np.zeros(len(cursos))
_CID = {c: i for i, c in enumerate(cursos.course_id)}
_OBJ = usuarios.set_index("user_id")["objetivo_laboral"].to_dict()
SEG = {}                                   # popularidad por objetivo laboral (respaldo mas fuerte que la global)
for _u, _cs in inscritos.items():
    v = SEG.setdefault(_OBJ[_u], np.zeros(len(cursos)))
    for _c in _cs:
        v[_CID[_c]] += 1

FEAT = {f: i for i, f in enumerate(M["user_feature_names"])} if M is not None else {}
UIDX = {u: i for i, u in enumerate(M["user_ids"])} if M is not None else {}


def atributos_usuario(p: dict) -> list:
    """Misma definición de atributos que en el entrenamiento (rutapro_rec.atributos_usuario)."""
    ciclo = int(p["ciclo"])
    f = [f"carrera:{p['carrera']}", f"nivel:{p['nivel']}", f"idioma:{p['idioma_preferido']}",
         f"objetivo:{p['objetivo_laboral']}", f"ciclo:{'5-6' if ciclo <= 6 else ('7-8' if ciclo <= 8 else '9-10')}"]
    f += [f"interes:{x.strip()}" for x in str(p["intereses"]).split(";") if x.strip()]
    f += [f"domina:{x.strip()}" for x in str(p["habilidades_actuales"]).split(";") if x.strip()]
    return f


def puntajes(perfil: dict, user_id=None) -> np.ndarray:
    if M is None:                                          # respaldo: popularidad por objetivo laboral (+ reglas en top_n)
        return SEG.get(perfil["objetivo_laboral"], POP).astype(float) + 0.001 * POP
    if user_id is not None and user_id in UIDX:            # estudiante registrado: embedding aprendido
        k = UIDX[user_id]
        return M["user_emb"][k] @ M["item_emb"].T + M["item_bias"] + M["user_bias"][k]
    feats = [f for f in atributos_usuario(perfil) if f in FEAT]   # estudiante nuevo: solo atributos
    x = np.zeros(len(FEAT))
    for f in feats:
        x[FEAT[f]] = 1.0 / len(feats)
    return (x @ M["feat_emb"]) @ M["item_emb"].T + M["item_bias"] + x @ M["feat_bias"]


def mascara_reglas(perfil: dict, margen=1) -> np.ndarray:
    ok = cursos["dificultad"].map(NIV).values <= NIV[perfil["nivel"]] + margen
    if perfil["idioma_preferido"] == "es":
        ok &= cursos["idioma"].eq("es").values
    return ok


def top_n(s, perfil, excluir, n, reglas):
    s = np.asarray(s, dtype=float).copy()
    s[cursos.course_id.isin(excluir).values] = -np.inf
    if reglas:
        s[~mascara_reglas(perfil)] = -np.inf
    orden = np.lexsort((np.arange(len(s)), -s))
    return [i for i in orden if np.isfinite(s[i])][:n]


def explicar(c: pd.Series, perfil: dict) -> list:
    razones = []
    intereses = [x.strip() for x in str(perfil["intereses"]).split(";")]
    if c.categoria in intereses:
        razones.append(f"Coincide con tu interés en <b>{c.categoria}</b>")
    if c.categoria in CLAVE_OBJETIVO.get(perfil["objetivo_laboral"], []):
        razones.append(f"Es una habilidad clave para <b>{perfil['objetivo_laboral']}</b>")
    gap = NIV[c.dificultad] - NIV[perfil["nivel"]]
    razones.append({0: f"Está en tu nivel ({c.dificultad})", 1: f"Te ayuda a subir de nivel ({c.dificultad})",
                    -1: f"Refuerza bases ({c.dificultad})", -2: f"Repaso ({c.dificultad})"}.get(gap, c.dificultad))
    if c.idioma == "es":
        razones.append("En español")
    return razones


def tarjeta(rank: int, c: pd.Series, perfil: dict):
    with st.container(border=True):
        a, b = st.columns([0.07, 0.93])
        a.markdown(f"<div class='rp-rank'>{rank}</div>", unsafe_allow_html=True)
        dur = f"{c.duracion_horas:.0f} h" if c.duracion_horas >= 1 else f"{c.duracion_horas * 60:.0f} min"
        rating = "" if c.rating_imputado else f" · ★ {c.rating:.1f}"
        origen = c.institucion if str(c.institucion).startswith(c.plataforma) else f"{c.plataforma} · {c.institucion}"
        idioma = "Español" if c.idioma == "es" else "Inglés"
        b.markdown(
            f"<a href='{c.url}' target='_blank'><b>{c.titulo}</b></a>"
            f"<div class='rp-meta'>{origen} · {c.tipo} · {dur}{rating}</div>"
            f"<span class='rp-chip'>{c.categoria}</span><span class='rp-chip'>{c.dificultad}</span>"
            f"<span class='rp-chip {'es' if c.idioma == 'es' else ''}'>{idioma}</span>"
            f"<div class='rp-why'>¿Por qué? " + " · ".join(explicar(c, perfil)) + "</div>",
            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.title("🎓 RutaPro")
st.caption("Recomendador de cursos profesionales para estudiantes universitarios · prototipo académico "
           "(organización ficticia; perfiles e historiales simulados, catálogo real de Coursera, edX, Platzi y DataCamp)")

with st.sidebar:
    st.header("Configuración")
    modo = st.radio("Tipo de estudiante", ["Estudiante registrado", "Estudiante nuevo (sin historial)"])
    n = st.slider("Número de recomendaciones (Top-N)", 5, 15, int(cfg.get("K_mostrado", 10)))
    reglas = st.toggle("Aplicar reglas de idioma y nivel", value=True,
                       help="No mostrar cursos en inglés a quien solo estudia en español ni cursos más de un nivel por encima.")
    comparar = st.toggle("Comparar con el baseline de popularidad", value=False)
    st.divider()
    met = cfg.get("metricas_prueba_K10", {})
    st.markdown("**Modelo en uso**")
    st.markdown(f"{cfg.get('modelo', 'LightFM')}  \n"
                f"NDCG@10 en prueba: **{met.get('ndcg', 0):.3f}** · Hit Rate@10: **{met.get('hit_rate', 0):.0%}**  \n"
                f"Popularidad global: NDCG@10 0.134 · Hit Rate@10 35 %  \n"
                f"Popularidad por objetivo + reglas: NDCG@10 0.183 · Hit Rate@10 44 %  \n"
                f"Techo del simulador (oráculo): NDCG@10 0.293")

# ---- Perfil
if modo == "Estudiante registrado":
    etiquetas = {r.user_id: f"{r.user_id} · {r.objetivo_laboral} · {r.carrera} · nivel {r.nivel}" for r in usuarios.itertuples()}
    uid = st.selectbox("Estudiante", list(etiquetas), format_func=etiquetas.get)
    perfil = usuarios.set_index("user_id").loc[uid].to_dict()
    hist = inscritos.get(uid, [])
else:
    uid, hist = None, []
    with st.form("registro"):
        st.subheader("Formulario de registro")
        c1, c2, c3 = st.columns(3)
        carrera = c1.selectbox("Carrera", CARRERAS)
        ciclo = c2.slider("Ciclo", 5, 10, 7)
        nivel = c3.selectbox("Nivel técnico actual", list(NIV))
        c4, c5 = st.columns(2)
        objetivo = c4.selectbox("Objetivo laboral", OBJETIVOS)
        idioma = c5.radio("¿En qué idioma puedes estudiar?", ["Solo español", "Español e inglés"], horizontal=True)
        intereses = st.multiselect("Intereses (elige 1 a 3)", CATEGORIAS, default=["SQL y Bases de Datos", "Power BI"],
                                   max_selections=3, placeholder="Elige hasta 3 temas")
        domina = st.multiselect("Temas que ya dominas", CATEGORIAS, default=[], placeholder="Elige uno o más temas")
        enviado = st.form_submit_button("Recomendarme cursos", type="primary")
    perfil = {"carrera": carrera, "ciclo": ciclo, "nivel": nivel, "objetivo_laboral": objetivo,
              "idioma_preferido": "es" if idioma == "Solo español" else "es+en",
              "intereses": "; ".join(intereses), "habilidades_actuales": "; ".join(domina)}
    if not intereses:
        st.info("Elige al menos un interés para personalizar mejor tus recomendaciones. "
                "Mientras tanto se usan tu carrera, nivel y objetivo laboral.")

col_p, col_h = st.columns([0.55, 0.45])
with col_p:
    st.markdown(f"**Objetivo:** {perfil['objetivo_laboral']} · **Carrera:** {perfil['carrera']} · "
                f"**Ciclo:** {perfil['ciclo']} · **Nivel:** {perfil['nivel']} · "
                f"**Idioma:** {'solo español' if perfil['idioma_preferido'] == 'es' else 'español e inglés'}")
    st.markdown(f"**Intereses:** {perfil['intereses'] or '—'}")
with col_h:
    if modo == "Estudiante registrado":
        with st.expander(f"Historial: {len(hist)} cursos inscritos (no se vuelven a recomendar)"):
            for cid in hist:
                c = cursos.set_index("course_id").loc[cid]
                st.markdown(f"- {c.titulo} · *{c.categoria}, {c.dificultad}*")
    else:
        st.markdown("**Sin historial:** se recomienda a partir de los datos del formulario (*cold start*).")

# ---- Recomendaciones
top = top_n(puntajes(perfil, uid), perfil, hist, n, reglas)
if len(top) < n:
    st.warning(f"Solo {len(top)} cursos cumplen las reglas de idioma y nivel para este perfil. "
               "Desactiva las reglas en la barra lateral para ver más opciones.")

if comparar:
    izq, der = st.columns(2)
    with izq:
        st.subheader("Recomendación personalizada (LightFM)")
        for r, i in enumerate(top, 1):
            tarjeta(r, cursos.iloc[i], perfil)
    with der:
        st.subheader("Baseline: más populares")
        for r, i in enumerate(top_n(POP, perfil, hist, n, False), 1):
            tarjeta(r, cursos.iloc[i], perfil)
else:
    st.subheader(f"Tus {len(top)} cursos recomendados")
    resumen = cursos.iloc[top]
    m1, m2, m3 = st.columns(3)
    m1.metric("Categorías distintas", resumen.categoria.nunique())
    m2.metric("Cursos en español", f"{(resumen.idioma == 'es').mean():.0%}")
    m3.metric("Horas totales aprox.", f"{resumen.duracion_horas.sum():.0f}")
    for r, i in enumerate(top, 1):
        tarjeta(r, cursos.iloc[i], perfil)

st.divider()
st.caption("Proyecto académico de Sistemas de Recomendación. Las recomendaciones provienen de un modelo entrenado con "
           "interacciones simuladas; los enlaces llevan a las fichas públicas de cada curso.")
