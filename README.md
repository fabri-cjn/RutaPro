# RutaPro: sistema de recomendación de cursos profesionales

Sistema de recomendación Top-N que sugiere a estudiantes universitarios peruanos los 10 cursos en línea más convenientes según su carrera, nivel, idioma, objetivo laboral e historial. Proyecto de medio curso de Ciencia de Datos I (UNALM, 2026).

- **Catálogo real:** 107 cursos de Coursera, edX, Platzi y DataCamp (55 en español y 52 en inglés), obtenidos por web scraping el 19/10/2026.
- **Usuarios e interacciones simulados:** 150 estudiantes y 3 680 eventos, generados con reglas explícitas y semilla fija (`SEED = 42`).
- **Modelos:** popularidad (baseline), contenido TF-IDF, KNN usuario, KNN ítem y LightFM híbrido (innovación).
- **Modelo seleccionado:** LightFM + reglas de idioma y nivel, con K = 10 (NDCG@10 = 0.195; Hit Rate@10 = 43.8 %).
- **Aplicación:** prototipo en Streamlit con recomendaciones para estudiantes registrados y nuevos.

> RutaPro es una organización **ficticia**. Las interacciones son simuladas, así que las métricas miden qué tan bien cada modelo recupera los patrones del simulador, no el comportamiento de estudiantes reales.

---

## Contenido

1. [Estructura del proyecto](#1-estructura-del-proyecto)
2. [¿Qué quieres hacer?](#2-qué-quieres-hacer)
3. [Opción A: solo la aplicación (la más rápida)](#3-opción-a-solo-la-aplicación)
4. [Opción B: notebooks en Google Colab (recomendada)](#4-opción-b-notebooks-en-google-colab-recomendada)
5. [Opción C: notebooks en tu computadora (VS Code o Jupyter)](#5-opción-c-notebooks-en-tu-computadora)
6. [Orden de ejecución y tiempos](#6-orden-de-ejecución-y-tiempos)
7. [Publicar la aplicación en Streamlit Community Cloud](#7-publicar-la-aplicación-en-streamlit-community-cloud)
8. [Qué subir a GitHub y qué no](#8-qué-subir-a-github-y-qué-no)
9. [Problemas frecuentes](#9-problemas-frecuentes)
10. [Créditos y referencias de datos](#10-créditos-y-referencias-de-datos)

---

## 1. Estructura del proyecto

```
RutaPro/
├── README.md                     ← este archivo
├── celda_inicial_colab.txt       ← celdas de arranque para Colab (por si un notebook no las tiene)
├── CAMBIOS_DOCUMENTACION.md      ← correcciones de la fase de robustez
├── README_robustez.md            ← detalle de los notebooks 04 a 08
├── lightfm_src/                  ← LightFM compilado (lo crea Colab; NO subir a GitHub)
│
├── 01_Datos/
│   ├── raw/cursos_scraping_raw.txt      captura del scraping (171 registros, 19/10/2026)
│   ├── cursos.csv                       catálogo final: 107 cursos
│   ├── usuarios.csv                     150 estudiantes simulados
│   ├── interacciones.csv                3 680 eventos simulados
│   ├── pares_usuario_curso.csv          pares estudiante-curso con particiones (lo crea el notebook 01)
│   ├── diccionario_datos_v2.md          diccionario de datos
│   ├── evidencia_avisos_plantilla.csv   plantilla para el notebook 07
│   └── mooccubex_x/                     datos reales de MOOCCubeX (los descarga el notebook 08; ≈ 820 MB; NO subir)
│
├── 02_Notebooks/
│   ├── 00_Construccion_Dataset.ipynb    scraping, curación y simulación
│   ├── 01_EDA.ipynb                     calidad, exploración y partición temporal
│   ├── 02_Modelos.ipynb                 ajuste de hiperparámetros en validación
│   ├── 03_Evaluacion.ipynb              evaluación en prueba y exportación del modelo a la app
│   ├── 04_Robustez_Baselines.ipynb      baselines justos, techo, IC95, pruebas pareadas, sesgos
│   ├── 05_ColdStart_Honesto.ipynb       cold start con estudiantes fuera del entrenamiento
│   ├── 06_Replicas_Sensibilidad.ipynb   6 escenarios × 10 réplicas del simulador
│   ├── 07_Evidencia_Afinidad.ipynb      validación de la matriz de afinidad (requiere avisos tabulados)
│   ├── 08_MOOCCube_Protocolo.ipynb      mismo protocolo con inscripciones reales (MOOCCubeX)
│   ├── rutapro_rec.py                   modelos, métricas y reglas (lo usan notebooks y app)
│   ├── rutapro_sim.py                   simulador reproducible y escenarios
│   ├── rutapro_extra.py                 baselines, oráculo, pruebas estadísticas, fusión de rankings
│   ├── rutapro_mooc.py                  descarga y preparación de MOOCCubeX
│   ├── resultados/                      hiperparámetros y métricas (notebooks 02 y 03)
│   ├── resultados_robustez/             resultados de los notebooks 04 a 06
│   └── resultados_mooc/                 resultados del notebook 08
│
├── 03_App/
│   ├── app.py                           aplicación Streamlit (app_v2.py es una copia idéntica)
│   ├── requirements.txt                 streamlit, numpy, pandas
│   ├── modelo/                          modelo exportado por el notebook 03
│   └── capturas/                        capturas de pantalla de la app
│
├── 04_Informe/
│   ├── Informe_RutaPro.docx / .pdf      informe técnico
│   └── figuras/                         figuras generadas por los notebooks
│
└── 05_Presentacion/                     diapositivas
```

**Importante:** la carpeta debe llamarse exactamente `RutaPro` y conservar esta estructura. Los notebooks calculan las rutas a partir de `02_Notebooks/`, así que no muevas archivos entre carpetas.

---

## 2. ¿Qué quieres hacer?

| Objetivo | Opción | Necesita LightFM | Tiempo |
|---|---|---|---|
| Ver y probar la aplicación | **A** | No | 2 minutos |
| Ver los notebooks con sus resultados | Ninguna: ábrelos en GitHub, VS Code o Colab sin ejecutarlos | No | — |
| Volver a ejecutar los notebooks | **B** (Colab) o **C** (local) | Sí, en 02 a 06 y 08 | Ver sección 6 |
| Tener un enlace público de la app | Sección 7 | No | 5 minutos |

Todos los resultados ya están calculados y guardados. **No hace falta volver a ejecutar nada** para revisar el proyecto ni para usar la aplicación.

---

## 3. Opción A: solo la aplicación

La aplicación no necesita LightFM: usa los vectores del modelo exportados por el notebook 03 y calcula los puntajes con numpy.

Requisitos: Python 3.9 o superior.

```bash
cd RutaPro/03_App
pip install -r requirements.txt
streamlit run app.py
```

Se abre en el navegador en `http://localhost:8501`. Si no se abre solo, copia esa dirección.

**Qué probar en la app:**
1. **Estudiante registrado:** elige `U001` (Project Analyst, solo español) y revisa su historial y sus 10 recomendaciones con su explicación.
2. **Comparación:** activa la comparación con los cursos más populares.
3. **Reglas:** desactiva y activa las reglas de idioma y nivel con un estudiante que solo estudia en español.
4. **Estudiante nuevo:** cambia a "Estudiante nuevo (sin historial)", llena el formulario y pulsa **Recomendarme cursos**.

Para detenerla, presiona `Ctrl + C` en la terminal.

---

## 4. Opción B: notebooks en Google Colab (recomendada)

Colab es la forma más sencilla de volver a ejecutar los notebooks, porque cada uno ya trae una primera celda que conecta tu Google Drive, entra a la carpeta del proyecto y compila LightFM si hace falta.

### 4.1 Subir el proyecto a Google Drive

1. Abre [drive.google.com](https://drive.google.com) con tu cuenta.
2. Arrastra la carpeta **`RutaPro`** completa a **Mi unidad** (la raíz, no dentro de otra carpeta).
3. Comprueba que la ruta quede así: `Mi unidad/RutaPro/02_Notebooks/00_Construccion_Dataset.ipynb`.

> Si subiste un `.zip`, descomprímelo antes en tu computadora y sube la carpeta. Drive no descomprime archivos.

### 4.2 Abrir un notebook

1. En Drive, entra a `RutaPro/02_Notebooks/`.
2. Clic derecho sobre el notebook → **Abrir con** → **Google Colaboratory**.
   - Si no aparece Colab: **Abrir con → Conectar más aplicaciones**, busca *Colaboratory* e instálalo.
   - Alternativa: entra a [colab.research.google.com](https://colab.research.google.com) → **Archivo → Abrir cuaderno → Google Drive** y busca el notebook.

### 4.3 Ejecutar

1. Ejecuta la **primera celda de código**. Hace tres cosas:
   - Pide permiso para conectar tu Drive (acepta con tu cuenta).
   - Entra a `/content/drive/MyDrive/RutaPro/02_Notebooks` y muestra si falta algún archivo.
   - Si el notebook necesita LightFM (`NECESITA_LIGHTFM = True`), lo compila la primera vez (≈ 2 minutos) y guarda una copia en `RutaPro/lightfm_src/`. Las siguientes veces la reutiliza.
2. Luego ejecuta el resto: **Entorno de ejecución → Ejecutar todas**, o celda por celda con `Shift + Enter`.

**Cada vez que abras un notebook** (o se reinicie la sesión de Colab) debes ejecutar de nuevo la primera celda. Si algún notebook no la tuviera, cópiala desde `celda_inicial_colab.txt`.

**Si tu carpeta no está en la raíz de Mi unidad**, cambia esta línea de la primera celda:

```python
PROY = "/content/drive/MyDrive/RutaPro"   # ← pon aquí la ruta real de tu carpeta
```

---

## 5. Opción C: notebooks en tu computadora

### 5.1 Crear el entorno

**Con conda (recomendado, porque instala LightFM sin compilar):**

```bash
conda create -n rutapro python=3.11 -y
conda activate rutapro
conda install -c conda-forge lightfm numpy pandas scipy scikit-learn matplotlib requests beautifulsoup4 jupyter ipykernel -y
pip install streamlit
```

**Con pip y Python 3.10 u 3.11:**

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib requests beautifulsoup4 jupyter ipykernel streamlit
pip install lightfm
```

> `pip install lightfm` **falla en Python 3.12 o superior**. Usa conda, o Python 3.10/3.11, o compílalo como se explica en la sección 9.

### 5.2 Quitar la celda de Colab

La primera celda de cada notebook usa `google.colab`, que solo existe en Colab. En tu computadora:

- **No ejecutes esa celda** (o bórrala), y
- abre el notebook desde la carpeta `02_Notebooks/`, para que las rutas se calculen bien.

### 5.3 Ejecutar en VS Code

1. Instala las extensiones **Python** y **Jupyter**.
2. **Archivo → Abrir carpeta** → elige `RutaPro`.
3. Abre un notebook de `02_Notebooks/`.
4. Arriba a la derecha, **Seleccionar kernel** → elige el entorno `rutapro` (o `.venv`).
5. Ejecuta desde la segunda celda en adelante.

### 5.4 Ejecutar en Jupyter

```bash
cd RutaPro/02_Notebooks
jupyter notebook
```

O sin abrir el navegador, ejecutando todo el notebook de una vez (primero borra la celda de Colab):

```bash
cd RutaPro/02_Notebooks
jupyter nbconvert --to notebook --execute --inplace 01_EDA.ipynb
```

---

## 6. Orden de ejecución y tiempos

```
00 → 01 → 02 → 03 → 04 → 05
                  └→ 06, 07 y 08 (independientes entre sí)
```

| Notebook | Lee | Genera | LightFM | Tiempo aprox. |
|---|---|---|---|---|
| `00_Construccion_Dataset` | `raw/cursos_scraping_raw.txt` | `cursos.csv`, `usuarios.csv`, `interacciones.csv` | No | < 1 min |
| `01_EDA` | CSV de `01_Datos/` | `pares_usuario_curso.csv`, figuras 01 a 11 | No | < 1 min |
| `02_Modelos` | `pares_usuario_curso.csv` | `resultados/hiperparametros.json`, figuras 12 a 15 | Sí | 5 a 10 min |
| `03_Evaluacion` | Hiperparámetros | Métricas en prueba, figuras 16 a 18 y **`03_App/modelo/`** | Sí | 2 a 4 min |
| `04_Robustez_Baselines` | Lo anterior | `resultados_robustez/`, figuras 19 a 21 | Sí | ≈ 1 min |
| `05_ColdStart_Honesto` | Lo anterior + 04 | Cold start sin fuga, figura 22 | Sí | ≈ 1 min |
| `06_Replicas_Sensibilidad` | `rutapro_sim.py` | 6 escenarios × 10 réplicas, figuras 23 y 24 | Sí | **≈ 12 min la primera vez** (luego usa `_cache/`) |
| `07_Evidencia_Afinidad` | `01_Datos/evidencia_avisos.csv` | Contraste de la matriz de afinidad | No | Segundos |
| `08_MOOCCube_Protocolo` | Descarga MOOCCubeX (≈ 820 MB) | `resultados_mooc/`, figuras 26 a 28 | Sí | **20 a 40 min** |

**Notas:**
- Todo usa semilla fija, así que volver a ejecutar produce los mismos archivos. LightFM puede variar en el tercer decimal según el sistema operativo.
- **Notebook 00:** el scraping viene desactivado (`EJECUTAR_SCRAPING = False`) y trabaja sobre la captura guardada. Si lo activas, el resultado puede diferir porque las páginas cambian con el tiempo.
- **Notebook 03:** si lo vuelves a ejecutar, **sobrescribe el modelo de la app** en `03_App/modelo/`.
- **Notebook 07:** está pendiente. Necesita que el equipo tabule al menos 20 avisos de prácticas por puesto en `01_Datos/evidencia_avisos.csv`, siguiendo `evidencia_avisos_plantilla.csv`. Sin ese archivo, solo muestra el protocolo.
- **Notebook 08:** descarga los datos solo (desde `lfs.aminer.cn`) y no repite lo que ya está completo. En Colab conviene usar **Entorno de ejecución → Cambiar tipo de entorno** con más RAM si está disponible.

---

## 7. Publicar la aplicación en Streamlit Community Cloud

1. Sube el proyecto a un repositorio de GitHub (ver sección 8). Puede ser público o privado.
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Pulsa **Create app** (o **New app**) → **Deploy a public app from GitHub**.
4. Completa:
   - **Repository:** `tu-usuario/RutaPro`
   - **Branch:** `main`
   - **Main file path:** `03_App/app.py`
5. Pulsa **Deploy**. Streamlit instala `03_App/requirements.txt` y en 2 a 3 minutos entrega un enlace del tipo `https://<nombre>.streamlit.app`.

Cada vez que hagas *push* a GitHub, la aplicación se actualiza sola. Si la app pasa varios días sin visitas, Streamlit la suspende; basta con abrir el enlace y pulsar **Wake up** (conviene abrirla unos minutos antes de la sustentación).

**Enlace de la aplicación:** [insertar enlace de la aplicación]

---

## 8. Qué subir a GitHub y qué no

**Subir:** `01_Datos/` (sin `mooccubex_x/`), `02_Notebooks/`, `03_App/`, `04_Informe/`, `05_Presentacion/` y los archivos `.md` y `.txt` de la raíz.

**No subir:**

| Carpeta | Motivo |
|---|---|
| `01_Datos/mooccubex_x/` | Pesa ≈ 820 MB (GitHub rechaza archivos de más de 100 MB) y `user.json` contiene nombres de estudiantes reales |
| `lightfm_src/` | Es una compilación para Colab; no sirve en otro sistema |
| `02_Notebooks/resultados_robustez/_cache/` | Caché temporal del notebook 06 |
| `__pycache__/`, `.ipynb_checkpoints/` | Archivos temporales de Python y Jupyter |

Para que Git los ignore, crea en la raíz un archivo llamado `.gitignore` con este contenido:

```
01_Datos/mooccubex_x/
lightfm_src/
02_Notebooks/resultados_robustez/_cache/
__pycache__/
.ipynb_checkpoints/
.venv/
*.pyc
```

Si subes desde la web de GitHub (**Add file → Upload files**), simplemente no arrastres esas carpetas.

---

## 9. Problemas frecuentes

**`ModuleNotFoundError: No module named 'google.colab'`**
Estás en tu computadora. No ejecutes la primera celda del notebook (sección 5.2).

**`FileNotFoundError` con `cursos.csv`, `pares_usuario_curso.csv` o `rutapro_rec.py`**
El notebook no se está ejecutando desde `02_Notebooks/`. En Colab, revisa que la primera celda imprima `Carpeta actual: .../RutaPro/02_Notebooks` y que la lista de archivos faltantes diga "ninguno". En tu computadora, abre el notebook desde esa carpeta. Si falta `pares_usuario_curso.csv`, ejecuta antes el notebook 01; si falta `hiperparametros.json`, el 02.

**`pip install lightfm` falla con errores de compilación**
Pasa en Python 3.12 o superior. Opciones, de la más simple a la más laboriosa:
1. `conda install -c conda-forge lightfm`
2. Un entorno con Python 3.10 u 3.11 y `pip install lightfm`
3. Compilarlo con Cython 3 (es lo que hace la primera celda en Colab):
   ```bash
   pip install cython
   git clone --depth 1 https://github.com/lyst/lightfm.git
   cd lightfm
   # En setup.py:
   #   compiler_directives={'language_level': "3", 'legacy_implicit_noexcept': True}
   #   y reemplazar "-march=native" por "-O2"
   python setup.py cythonize
   python setup.py build_ext --inplace
   ```
   Luego agrega esa carpeta al inicio del notebook: `import sys; sys.path.insert(0, "ruta/a/lightfm")`.

   En Windows este último camino requiere *Microsoft C++ Build Tools*; es más fácil usar conda o Colab.

**LightFM dejó de importarse en Colab**
Colab cambió de versión de Python. Borra la carpeta `RutaPro/lightfm_src` de tu Drive y vuelve a ejecutar la primera celda para recompilar.

**La app avisa "No se pudo cargar el modelo LightFM... Se usará la popularidad por objetivo laboral como respaldo"**
No encuentra o no puede leer `03_App/modelo/lightfm_final.npz`. Verifica que la carpeta `modelo/` esté completa o vuelve a ejecutar el notebook 03.

**La app dice "Solo N cursos cumplen las reglas de idioma y nivel"**
Las reglas de idioma y nivel dejaron pocos candidatos para ese perfil. Desactiva las reglas o cambia el nivel o el idioma del estudiante.

**`streamlit: command not found`**
Usa `python -m streamlit run app.py`, o activa el entorno donde instalaste Streamlit.

**El notebook 08 se queda sin memoria en Colab**
Reduce `N_USUARIOS` (5 000 por defecto) en la celda de configuración o cambia a un entorno con más RAM.

**El scraping del notebook 00 devuelve 0 cursos o error 429**
Las plataformas cambian su HTML y limitan las peticiones. Vuelve a `EJECUTAR_SCRAPING = False` para usar la captura guardada, o aumenta `PAUSA_SEG`.

---

## 10. Créditos y referencias de datos

- **Autor:** Fabricio Joaquín Chávez Ccorahua, Facultad de Economía y Planificación, Universidad Nacional Agraria La Molina, 2026.
- **Catálogo:** fichas públicas de [Coursera](https://www.coursera.org/courses), [edX](https://www.edx.org/es/learn), [Platzi](https://platzi.com/escuela/datos/) y [DataCamp](https://www.datacamp.com/es/courses-all), consultadas el 19/10/2026 respetando sus `robots.txt`.
- **Datos reales del notebook 08:** Yu, J., et al. (2021). MOOCCubeX: A large knowledge-centered repository for adaptive learning in MOOCs. *CIKM '21*, 4643–4652.
- **LightFM:** Kula, M. (2015). Metadata embeddings for user and item cold-start recommendations. *CBRecSys 2015*, CEUR 1448, 14–21.

El informe técnico completo, con metodología, resultados y limitaciones, está en `04_Informe/Informe_RutaPro.pdf`.
