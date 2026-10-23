"""
rutapro_mooc.py: lleva el protocolo de RutaPro a MOOCCube/MOOCCubeX (inscripciones REALES en cursos de XuetangX).

Sirve para las DOS versiones publicas (el modulo detecta cual es):

  A) MOOCCube v1 -- RECOMENDADA: un solo zip de 1.01 GB, con respaldo en Google Drive y licencia CC0.
     Pagina: http://moocdata.cn/data/MOOCCube | repositorio: https://github.com/thukg/MOOCCube
     Usa: entities/course.json (name, about, video_order) y entities/user.json (course_order, enroll_time).

  B) MOOCCubeX -- version ampliada: https://github.com/THU-KEG/MOOCCubeX
     Los archivos NO estan en GitHub: el repositorio solo trae el script que los baja de lfs.aminer.cn.
     Usa: entities/course.json, entities/user.json (770 MB) y, si estan, relations/course-field.json
     y relations/course-school.txt. NO hace falta relations/user-problem.json (21 GB).

En ambas, la interaccion positiva es la inscripcion (`course_order`) con su fecha (`enroll_time`).

Interaccion positiva = inscripcion (curso en `course_order`), con su fecha en `enroll_time`: es la misma definicion
que en RutaPro. Se reutilizan sin cambios las particiones, las metricas, los modelos y las pruebas estadisticas.

Lo que NO existe en estos datos (y por eso no se evalua): idioma, nivel de dificultad, objetivo laboral, carrera.
Sin esos campos no hay reglas de negocio de idioma y nivel, y no hay oraculo (no hay simulador).

CITAR: Yu et al. (2020) MOOCCube, ACL. Yu et al. (2021) MOOCCubeX, CIKM. Revisar los terminos de uso del repositorio.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import rutapro_rec as rr

_TOKEN = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+")


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------
def leer_json_flexible(ruta, max_lineas=None, fraccion=1.0, seed=0):
    """Itera objetos de un .json que puede ser JSON Lines (un objeto por linea) o un arreglo JSON completo.
    `fraccion` < 1 toma cada linea con esa probabilidad ANTES de parsearla (muestra aleatoria y rapida de archivos
    enormes). Leer solo las primeras lineas NO es una muestra aleatoria: en MOOCCubeX las primeras filas son del personal de Tsinghua."""
    ruta = Path(ruta)
    rng = np.random.default_rng(seed) if fraccion < 1.0 else None
    with open(ruta, encoding="utf-8") as f:
        primera = f.readline().lstrip()
        f.seek(0)
        if primera.startswith("["):                       # arreglo completo: se carga entero (solo archivos pequenos)
            for k, obj in enumerate(json.load(f)):
                if max_lineas and k >= max_lineas:
                    return
                yield obj
            return
        for k, linea in enumerate(f):
            if max_lineas and k >= max_lineas:
                return
            if rng is not None and rng.random() > fraccion:
                continue
            linea = linea.strip().rstrip(",")
            if not linea:
                continue
            try:
                yield json.loads(linea)
            except json.JSONDecodeError:
                continue


def _primer_campo(x):
    if isinstance(x, list):
        return str(x[0]) if x else "sin_campo"
    return str(x) if x not in (None, "") else "sin_campo"


def _ruta(d: Path, *candidatas):
    """Devuelve la primera ruta que exista (busca en la carpeta y en una subcarpeta, por si el zip trae un nivel extra)."""
    for c in candidatas:
        for base in [d, *[p for p in d.iterdir() if p.is_dir()]] if d.exists() else []:
            p = base / c
            if p.exists():
                return p
    return None


def _n_recursos(c) -> int:
    for k in ("resource", "video_order", "video_name"):
        v = c.get(k)
        if isinstance(v, list):
            return len(v)
    return 0


def cargar_cursos(dir_datos, max_cursos=None) -> pd.DataFrame:
    """Catalogo de cursos. Toma lo que exista en esta version: disciplina, escuela, tamano (n de videos) y texto."""
    d = Path(dir_datos)
    f_cursos = _ruta(d, "entities/course.json", "course.json")
    if f_cursos is None:
        raise FileNotFoundError(f"No encuentro entities/course.json dentro de {d}")
    filas = []
    for c in leer_json_flexible(f_cursos, max_cursos):
        filas.append({"course_id": str(c.get("id")), "titulo": str(c.get("name", "")), "descripcion": str(c.get("about", ""))[:400],
                      "campos": "; ".join(c["field"]) if isinstance(c.get("field"), list) else str(c.get("field", "")),
                      "n_recursos": _n_recursos(c)})
    cur = pd.DataFrame(filas).dropna(subset=["course_id"]).drop_duplicates("course_id")
    f_field = _ruta(d, "relations/course-field.json")
    if f_field is not None:
        cf = {str(o.get("course_id")): o.get("field") for o in leer_json_flexible(f_field)}
        mejor = cur.course_id.map(lambda k: "; ".join(cf[k]) if isinstance(cf.get(k), list) and cf.get(k) else None)
        cur["campos"] = mejor.fillna(cur["campos"])
    cur["campos"] = cur["campos"].replace("", "sin_campo").fillna("sin_campo")
    cur["escuela_curso"] = "sin_dato"
    f_sch = _ruta(d, "relations/course-school.txt", "relations/school-course.txt")
    if f_sch is not None:                       # el archivo puede venir como curso->escuela o escuela->curso
        m = {}
        for linea in open(f_sch, encoding="utf-8"):
            p = linea.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            a_, b_ = p[0], p[1]
            if a_.startswith("S_"):
                a_, b_ = b_, a_
            m[a_] = b_
        cur["escuela_curso"] = cur.course_id.map(m).fillna("sin_dato")
        f_nom = _ruta(d, "entities/school.json")
        if f_nom is not None:                   # S_123 -> nombre de la escuela (solo para que sea legible)
            nombres = {str(o.get("id")): str(o.get("name") or o.get("name_en") or o.get("id")) for o in leer_json_flexible(f_nom)}
            cur["escuela_curso"] = cur["escuela_curso"].map(lambda x: nombres.get(x, x))
    cur["plataforma"], cur["n_resenas"] = "XuetangX", 0          # constantes: solo compatibilidad con rutapro_rec (no influyen en ningun ranking)
    q = cur["n_recursos"].rank(pct=True)
    cur["tamano"] = np.where(q <= 1 / 3, "pequeno", np.where(q <= 2 / 3, "mediano", "grande"))
    return cur.reset_index(drop=True)


def _grupo_edad(v):
    """Cohorte de nacimiento. El campo viene sucio (por ejemplo 6558, o 2015 para un adulto): solo se aceptan anos 1940-2008."""
    m = re.match(r"\s*(\d{4})", str(v))
    if not m:
        return "sin_dato"
    y = int(m.group(1))
    if not 1940 <= y <= 2008:
        return "sin_dato"
    return "<1980" if y < 1980 else ("1980-1989" if y < 1990 else ("1990-1999" if y < 2000 else "2000+"))


def _genero(v) -> str:
    """El genero viene como entero (0, 1, 2) o nulo. Se conserva el codigo SIN interpretarlo: g0, g1, g2."""
    if v is None:
        return "sin_dato"
    try:
        f = float(v)
    except (TypeError, ValueError):
        t = str(v).strip().lower()
        return t if t and t not in ("nan", "none", "null") else "sin_dato"
    return "sin_dato" if f != f else f"g{int(f)}"


def _escuela(v) -> str:
    t = "" if v is None else str(v).strip()
    return "sin_dato" if t.lower() in ("", "nan", "none", "null") else t


def _variantes_id(x) -> list:
    """Un mismo curso puede venir como 'C_123', '123' o 123 segun la version y el archivo."""
    t = str(x).strip()
    v = [t]
    if t.startswith("C_"):
        v.append(t[2:])
    else:
        v.append("C_" + t)
    return v


def cargar_inscripciones(dir_datos, cursos_ids: set, max_lineas=None, verbose=True, fraccion=1.0, seed=0):
    """Lee entities/user.json en streaming. Devuelve (usuarios, inscripciones) SIN filtrar por actividad.
    Resuelve los ids de curso probando variantes ('C_123' vs '123'), y avisa si casi nada coincide."""
    d = Path(dir_datos)
    f_users = _ruta(d, "entities/user.json", "user.json")
    if f_users is None:
        raise FileNotFoundError(f"No encuentro entities/user.json dentro de {d}")
    idx = {}
    for cid in cursos_ids:
        for v in _variantes_id(cid):
            idx.setdefault(v, cid)
    usu, filas, n_ref, n_ok_tot = [], [], 0, 0
    for u in leer_json_flexible(f_users, max_lineas, fraccion=fraccion, seed=seed):
        uid = u.get("id")
        cursos, tiempos = u.get("course_order") or [], u.get("enroll_time") or []
        if not uid or not cursos:
            continue
        n_ok = 0
        for k, c in enumerate(cursos):
            n_ref += 1
            cid = idx.get(str(c).strip())
            if cid is not None and k < len(tiempos):
                filas.append((str(uid), cid, tiempos[k]))
                n_ok += 1
        n_ok_tot += n_ok
        if n_ok:
            usu.append({"user_id": str(uid), "genero": _genero(u.get("gender")), "escuela": _escuela(u.get("school")),
                        "edad_grupo": _grupo_edad(u.get("year_of_birth"))})
    pct = 100 * n_ok_tot / max(n_ref, 1)
    if verbose:
        print(f"Referencias a cursos en user.json: {n_ref:,} | resueltas contra el catalogo: {n_ok_tot:,} ({pct:.1f} %)")
    if pct < 5:
        raise ValueError("Casi ningun curso de user.json coincide con course.json: los identificadores no son los mismos. "
                         "Ejecuta diagnosticar(DIR) para ver ejemplos de ambos archivos antes de continuar.")
    ins = pd.DataFrame(filas, columns=["user_id", "course_id", "timestamp"])
    ins["timestamp"] = pd.to_datetime(ins["timestamp"], errors="coerce")
    ins = ins.dropna(subset=["timestamp"]).drop_duplicates(["user_id", "course_id"])
    return pd.DataFrame(usu).drop_duplicates("user_id"), ins


def raiz_datos(dir_datos) -> Path:
    """Carpeta que contiene entities/ (aunque el zip o la descarga hayan dejado un nivel extra)."""
    f = _ruta(Path(dir_datos), "entities/user.json", "user.json")
    if f is None:
        return Path(dir_datos)
    return f.parent.parent if f.parent.name == "entities" else f.parent


def tiene_atributos(dir_datos, n=50) -> bool:
    """True si el user.json trae gender / school / year_of_birth (MOOCCubeX). False con MOOCCube v1 (solo id, name, course_order, enroll_time)."""
    f = _ruta(Path(dir_datos), "entities/user.json", "user.json")
    if f is None:
        return False
    claves = set()
    for o in leer_json_flexible(f, n):
        claves |= set(o.keys())
    return bool(claves & {"gender", "school", "year_of_birth"})


def resumen_atributos(usu: pd.DataFrame) -> pd.Series:
    """Cuanto del perfil esta realmente informado (despues de limpiar valores imposibles)."""
    return pd.Series({
        "estudiantes": len(usu),
        "% con escuela informada": 100 * (usu.escuela != "sin_dato").mean(),
        "% con año de nacimiento plausible (1940-2008)": 100 * (usu.edad_grupo != "sin_dato").mean(),
        "% con género distinto de 'sin_dato'": 100 * (usu.genero != "sin_dato").mean(),
        "% con escuela o cohorte informada": 100 * ((usu.escuela != "sin_dato") | (usu.edad_grupo != "sin_dato")).mean(),
    })


# ---------------------------------------------------------------------------
# Descarga automatica de MOOCCubeX (idempotente: no vuelve a bajar lo que ya esta completo)
# ---------------------------------------------------------------------------
URL_MOOCCUBEX = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2"
# (ruta relativa, tamano conocido en bytes, opcional)
ARCHIVOS_MOOCCUBEX = [
    ("entities/school.json", 627_336, False),
    ("relations/course-field.json", None, True),
    ("relations/course-school.txt", None, True),
    ("entities/course.json", 44_660_123, False),
    ("entities/user.json", 806_781_726, False),
]


def descargar(url, destino, verbose=True) -> Path:
    """Descarga `url` en `destino`. Usa wget -c (reanuda si quedo a medias) y, si no hay wget, urllib."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        if subprocess.run(["wget", "-c", "-q", "-O", str(destino), url]).returncode != 0:
            if destino.exists():
                destino.unlink()                  # algunos servidores no permiten reanudar: reintento limpio
            if subprocess.run(["wget", "-q", "-O", str(destino), url]).returncode != 0:
                if destino.exists() and destino.stat().st_size == 0:
                    destino.unlink()
                raise RuntimeError(f"No se pudo descargar {url}. Revisa tu conexion o descarga el archivo a mano en {destino}.")
        return destino
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = destino.with_name(destino.name + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        total, leidos = int(resp.headers.get("Content-Length") or 0), 0
        while True:
            bloque = resp.read(1 << 20)
            if not bloque:
                break
            out.write(bloque)
            antes, leidos = leidos, leidos + len(bloque)
            if verbose and total and antes // (100 << 20) != leidos // (100 << 20):
                print(f"   {leidos / 1e6:,.0f} MB de {total / 1e6:,.0f} MB")
    tmp.replace(destino)
    return destino


def descargar_mooccubex(dir_destino, base_url=URL_MOOCCUBEX, verbose=True) -> Path:
    """Baja a `dir_destino` los archivos necesarios de MOOCCubeX (~820 MB, entre 4 y 8 minutos). Los opcionales pueden fallar sin problema."""
    d = Path(dir_destino)
    for rel, tam, opcional in ARCHIVOS_MOOCCUBEX:
        destino = d / rel
        existente = _ruta(d, rel, Path(rel).name)          # tambien vale si ya esta en otra subcarpeta o suelto
        if existente is not None and existente.stat().st_size > 0 and (tam is None or existente.stat().st_size == tam):
            if verbose:
                print(f"ya existe: {rel}")
            continue
        if verbose:
            print(f"descargando {rel}" + ("  (769 MB: unos 5 minutos, no interrumpas)" if rel.endswith("user.json") else "") + " ...", flush=True)
        try:
            descargar(f"{base_url}/{rel}", destino, verbose)
        except Exception as e:
            if opcional:
                print(f"   aviso: no se pudo bajar {rel} (es opcional): {e}")
                continue
            raise
        if verbose and tam and destino.stat().st_size != tam:
            print(f"   aviso: {rel} pesa {destino.stat().st_size:,} bytes y se esperaban {tam:,} (puede que el dataset se haya actualizado)")
    return d


def diagnosticar(dir_datos, n=2):
    """Imprime que archivos hay y un par de registros de cada uno. Usar cuando algo no cuadre."""
    d = Path(dir_datos)
    print("Carpeta:", d.resolve())
    if not d.exists():
        print("  (no existe)")
        return
    for p in sorted(d.rglob("*.json"))[:25] + sorted(d.rglob("*.txt"))[:15]:
        print(f"  {p.relative_to(d)}  ({p.stat().st_size / 1e6:.1f} MB)")
    for nombre, cands in [("course.json", ("entities/course.json", "course.json")), ("user.json", ("entities/user.json", "user.json"))]:
        f = _ruta(d, *cands)
        print(f"\n--- {nombre}: {f if f else 'NO ENCONTRADO'}")
        if f:
            for k, o in enumerate(leer_json_flexible(f, n)):
                campos = {kk: (str(vv)[:60] + "..." if isinstance(vv, (list, dict, str)) and len(str(vv)) > 60 else vv)
                          for kk, vv in o.items()}
                print(f"  registro {k}: {campos}")


# ---------------------------------------------------------------------------
# Preparacion documentada (filtros + muestreo)
# ---------------------------------------------------------------------------
def preparar(usuarios, ins, cursos, n_usuarios=5000, seed=0, min_pop_curso=20, min_inscripciones=3, min_usuarios_escuela=30,
             solo_con_perfil=False):
    """
    1. Muestra aleatoria de `n_usuarios` estudiantes con >= min_inscripciones inscripciones validas.
    2. k-core iterativo: se eliminan cursos con < min_pop_curso inscripciones en la muestra y luego estudiantes con
       < min_inscripciones, hasta que no cambie nada (practica estandar; se reporta cuanto se pierde).
    3. Escuelas con < min_usuarios_escuela estudiantes se agrupan en 'otra' (atributo y segmento).
    4. Con solo_con_perfil=True se parte solo de estudiantes con escuela o cohorte de nacimiento informadas.
    Devuelve (usuarios, cursos, inter, resumen) con `inter` en el formato de RutaPro (evento = inscripcion).
    """
    rng = np.random.default_rng(seed)
    if solo_con_perfil:        # solo estudiantes con escuela o cohorte informadas (el genero no se usa: no se sabe que significa el 0)
        ok = usuarios[(usuarios.escuela != "sin_dato") | (usuarios.edad_grupo != "sin_dato")].user_id
        ins = ins[ins.user_id.isin(set(ok))]
    n0_u, n0_i = ins.user_id.nunique(), len(ins)
    act = ins.groupby("user_id").size()
    elegibles = act[act >= min_inscripciones].index.to_numpy()
    if len(elegibles) > n_usuarios:
        elegibles = rng.choice(elegibles, n_usuarios, replace=False)
    d = ins[ins.user_id.isin(set(elegibles))]
    while True:
        pop = d.groupby("course_id").size()
        d2 = d[d.course_id.isin(pop[pop >= min_pop_curso].index)]
        act = d2.groupby("user_id").size()
        d2 = d2[d2.user_id.isin(act[act >= min_inscripciones].index)]
        if len(d2) == len(d):
            break
        d = d2
    usu = usuarios[usuarios.user_id.isin(set(d.user_id))].copy()
    cnt = usu.escuela.value_counts()
    usu["escuela"] = np.where(usu.escuela.map(cnt) >= min_usuarios_escuela, usu.escuela, "otra")
    cur = cursos[cursos.course_id.isin(set(d.course_id))].reset_index(drop=True)
    usu = usu.sort_values("user_id").reset_index(drop=True)
    inter = d.rename(columns={"timestamp": "timestamp"}).assign(evento="inscripcion", peso_evento=3, calificacion=np.nan)
    inter = inter.sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    inter.insert(0, "interaction_id", [f"I{i:07d}" for i in range(1, len(inter) + 1)])
    resumen = {"usuarios con >= min inscripciones (antes de muestrear)": int((ins.groupby('user_id').size() >= min_inscripciones).sum()),
               "estudiantes en la muestra final": len(usu), "cursos finales": len(cur), "inscripciones finales": len(inter),
               "solo con perfil": bool(solo_con_perfil),
               "% de inscripciones conservadas (de la lectura)": 100 * len(inter) / max(n0_i, 1),
               "dispersion de la matriz (%)": 100 * (1 - len(inter) / (len(usu) * len(cur)))}
    return usu, cur, inter, resumen


# ---------------------------------------------------------------------------
# Adaptadores: reemplazan temporalmente las funciones de atributos y texto de rutapro_rec
# ---------------------------------------------------------------------------
def _atributos_usuario(u):
    return [f"genero:{u['genero']}", f"escuela:{u['escuela']}", f"edad:{u['edad_grupo']}"]


def _atributos_curso(c):
    f = [f"campo:{x.strip()}" for x in str(c["campos"]).split(";") if x.strip()]
    return f + [f"escuela_curso:{c['escuela_curso']}", f"tamano:{c['tamano']}"]


def _tokens(texto: str) -> str:
    """Tokeniza caracteres CJK uno a uno y palabras latinas completas (para que TF-IDF con n-gramas funcione en chino)."""
    return " ".join(_TOKEN.findall(texto.lower()))


def _texto_curso(c):
    return _tokens(f"{c['titulo']} {c['campos']} {c['campos']} {c['descripcion']}")


def _texto_perfil(u):
    return ""        # los estudiantes no tienen texto: el contenido usa solo el historial (alpha = 0)


_vistos_original = rr.Particion.vistos


def _vistos_rapido(self, user_id):
    """Equivalente a Particion.vistos pero con cache por usuario: con miles de estudiantes el filtro original es muy lento."""
    cache = self.__dict__.get("_vistos_cache")
    if cache is None:
        t = self.train[self.train.positivo == 1]
        cache = {u: g.map(self.iidx).values for u, g in t.groupby("user_id")["course_id"]}
        self._vistos_cache = cache
    return cache.get(user_id, np.array([], dtype=int))


@contextmanager
def adaptadores_mooc():
    """Dentro del bloque, rutapro_rec usa los atributos y el texto de MOOCCube. Al salir se restauran los originales."""
    orig = (rr.atributos_usuario, rr.atributos_curso, rr.texto_curso, rr.texto_perfil)
    rr.atributos_usuario, rr.atributos_curso, rr.texto_curso, rr.texto_perfil = (
        _atributos_usuario, _atributos_curso, _texto_curso, _texto_perfil)
    rr.Particion.vistos = _vistos_rapido
    try:
        yield
    finally:
        rr.atributos_usuario, rr.atributos_curso, rr.texto_curso, rr.texto_perfil = orig
        rr.Particion.vistos = _vistos_original


def instalar_adaptadores():
    """Version sin bloque `with`: deja activos los atributos de MOOCCube hasta llamar a restaurar_adaptadores()."""
    global _ORIG
    _ORIG = (rr.atributos_usuario, rr.atributos_curso, rr.texto_curso, rr.texto_perfil)
    rr.atributos_usuario, rr.atributos_curso, rr.texto_curso, rr.texto_perfil = (
        _atributos_usuario, _atributos_curso, _texto_curso, _texto_perfil)
    rr.Particion.vistos = _vistos_rapido


def restaurar_adaptadores():
    rr.atributos_usuario, rr.atributos_curso, rr.texto_curso, rr.texto_perfil = _ORIG
    rr.Particion.vistos = _vistos_original


# ---------------------------------------------------------------------------
# Datos SINTETICOS en el formato de MOOCCubeX: SOLO para comprobar que el codigo corre
# ---------------------------------------------------------------------------
def generar_sintetico(dir_salida, n_usuarios=4000, n_cursos=300, seed=0, esquema="x"):
    """
    Escribe archivos con el mismo formato que MOOCCubeX (esquema="x") o MOOCCube v1 (esquema="v1"). Sus resultados NO significan
    nada: es un simulacro para verificar la instalacion y el flujo. NO usar para sacar conclusiones.
    El esquema "x" reproduce los defectos reales vistos en user.json: ids de curso C_<numero> con course_order numerico,
    genero entero o nulo, escuela vacia en la mayoria, anos de nacimiento imposibles (6558, 2015) o nulos.
    """
    rng = np.random.default_rng(seed)
    d = Path(dir_salida)
    (d / "entities").mkdir(parents=True, exist_ok=True)
    (d / "relations").mkdir(parents=True, exist_ok=True)
    campos = ["计算机科学", "数学", "心理学", "经济学", "物理学", "管理学", "外语", "艺术"]
    escuelas = [f"S_{k}" for k in range(8)]
    nombres_esc = ["清华大学", "北京大学", "复旦大学", "浙江大学", "南京大学", "武汉大学", "中山大学", "四川大学"]
    cur_campo = rng.integers(0, len(campos), n_cursos)
    cur_esc = rng.integers(0, len(escuelas), n_cursos)
    cids = [f"C_{100000 + k}" if esquema == "x" else f"C_{k:05d}" for k in range(n_cursos)]
    with open(d / "entities" / "course.json", "w", encoding="utf-8") as f, \
            open(d / "relations" / "course-field.json", "w", encoding="utf-8") as g, \
            open(d / "relations" / "course-school.txt", "w", encoding="utf-8") as h:
        for k in range(n_cursos):
            f.write(json.dumps({"id": cids[k], "name": f"{campos[cur_campo[k]]}课程{k}", "about": f"介绍{k} {campos[cur_campo[k]]}",
                                "field": [campos[cur_campo[k]]], "resource": [{}] * int(rng.integers(3, 60))}, ensure_ascii=False) + "\n")
            g.write(json.dumps({"course_id": cids[k], "course_name": f"课程{k}", "field": [campos[cur_campo[k]]]}, ensure_ascii=False) + "\n")
            h.write(f"{cids[k]}\t{escuelas[cur_esc[k]]}\n")
    with open(d / "entities" / "school.json", "w", encoding="utf-8") as f:
        for k, e in enumerate(escuelas):
            f.write(json.dumps({"id": e, "name": nombres_esc[k], "name_en": f"School {k}"}, ensure_ascii=False) + "\n")
    pop = rng.pareto(1.2, n_cursos) + 1
    with open(d / "entities" / "user.json", "w", encoding="utf-8") as f:
        for u in range(n_usuarios):
            pref = int(rng.integers(0, len(campos)))
            w = pop * np.where(cur_campo == pref, 4.0, 1.0)
            n = int(min(40, 3 + rng.negative_binomial(2, 0.25)))
            elegidos = rng.choice(n_cursos, size=min(n, n_cursos), replace=False, p=w / w.sum())
            t0 = pd.Timestamp("2017-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
            tiempos = [str(t0 + pd.Timedelta(days=int(k * rng.integers(3, 30)))) for k in range(len(elegidos))]
            if esquema == "x":
                informada = rng.random() < 0.3
                anio = [None, 6558, 2015, int(rng.integers(1970, 2003)), int(rng.integers(1970, 2003))][int(rng.integers(0, 5))]
                reg = {"id": f"U_{u}", "name": "匿名", "gender": [0, 1, 2, None][int(rng.integers(0, 4))],
                       "school": nombres_esc[pref % len(escuelas)] if informada else "", "year_of_birth": anio,
                       "course_order": [100000 + int(k) for k in elegidos], "enroll_time": tiempos}
            else:
                reg = {"id": f"U_{u:06d}", "name": "x", "gender": int(rng.integers(0, 3)), "school": escuelas[pref % len(escuelas)],
                       "year_of_birth": f"{int(rng.integers(1980, 2004))}-05",
                       "course_order": [cids[int(k)] for k in elegidos], "enroll_time": tiempos}
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
    return d
