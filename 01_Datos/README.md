# Datos del proyecto RutaPro

Sistema de recomendación de cursos profesionales para estudiantes universitarios. RutaPro es una organización **ficticia** creada para el trabajo grupal del curso de Sistemas de Recomendación.

## Contenido
| Archivo | Descripción | Origen |
|---|---|---|
| `raw/cursos_scraping_raw.txt` | 171 registros extraídos de Coursera, edX, Platzi y DataCamp el 19/10/2026 | Real |
| `cursos.csv` | Catálogo final curado: 107 cursos (52 en inglés y 55 en español) | Real, con imputaciones marcadas |
| `usuarios.csv` | 150 estudiantes | **Simulado** |
| `interacciones.csv` | 3 680 eventos (vista, favorito, inscripción, finalización) | **Simulado** |
| `raw/regla_afinidad_objetivo_categoria.csv` | Matriz de afinidad usada en la simulación | Supuesto del equipo |
| `diccionario_datos.md` | Descripción de cada variable | |
| `pares_usuario_curso.csv` | Una fila por par usuario-curso: positivo, peso implícito y particiones (generado en `01_EDA.ipynb`) | Derivado |

## Procedencia y construcción
1. **Catálogo (real).** Como RutaPro se dirige a estudiantes peruanos, el catálogo se equilibró en mitad inglés y mitad español.
   - *Inglés:* páginas públicas de búsqueda de Coursera (10 consultas temáticas).
   - *Español:* fichas de 35 cursos de **Platzi** (obtenidas desde `platzi.com/escuela/datos` y las rutas de gestión de proyectos y comunicación), 9 cursos en español de **DataCamp** (`datacamp.com/es/courses/...`), temas de edX en español (Python y Excel) y 3 fichas de Coursera en español.
   - Se respetó `robots.txt`: no se usó la API de Coursera ni Udemy, porque ambos excluyen la extracción automática. Crehana se descartó porque su listado se carga dinámicamente. Tres listados de edX en español no se pudieron leer por límite de peticiones del servidor (HTTP 429).
2. **Curación.** Eliminación de duplicados por URL, exclusión de cursos fuera del público objetivo, asignación de categoría por reglas sobre el título, conversión de "23K" a números y de la duración a horas, e imputación marcada de los valores no publicados: nivel, duración y reseñas de los 8 cursos de edX, y rating de esos 8 cursos más los 9 de DataCamp (DataCamp no publica un promedio numérico). Se seleccionan hasta 5 cursos en inglés por categoría, priorizando un curso por nivel de dificultad, y todos los cursos en español.
3. **Usuarios e interacciones (simulados).** Se generan con reglas explícitas (afinidad objetivo-categoría, intereses, popularidad real, ajuste de nivel con progresión, prerrequisitos, idioma, exploración aleatoria del 15 % y embudo vista → inscripción → finalización). El detalle completo está en la sección 5 del notebook.

## Cómo reproducir
```bash
pip install pandas numpy requests beautifulsoup4 jupyter
jupyter nbconvert --to notebook --execute 02_Notebooks/00_Construccion_Dataset.ipynb
```
Semilla fija (`SEED = 42`), así que se obtienen exactamente los mismos archivos. Para volver a extraer el catálogo desde la web, poner `EJECUTAR_SCRAPING = True` en la sección 2. La web cambia con el tiempo, así que los resultados pueden diferir de la captura del 19/10/2026.

## Limitaciones
- Las interacciones simuladas reflejan las reglas definidas por el equipo, no el comportamiento real de estudiantes. Las métricas de los modelos deben leerse con esa advertencia.
- Los estudiantes "solo español" toman algunos cursos en inglés (13 % de sus inscripciones). Se asume que usan los subtítulos en español que ofrece Coursera.
- Los cursos de Platzi y DataCamp son cortos (1 a 5 horas) frente a los programas de Coursera (hasta ~110 horas). Esto influye en la probabilidad simulada de finalizarlos.
- El número de reseñas no es del todo comparable entre plataformas (Coursera cuenta reseñas, Platzi cuenta opiniones y DataCamp cuenta reseñas sobre millones de estudiantes).
- Ratings y reseñas son una fotografía del 19/10/2026.

## Fuentes
- Coursera, páginas de búsqueda: https://www.coursera.org/courses
- edX, páginas de temas: https://www.edx.org/es/learn
- Platzi, escuela de datos y rutas: https://platzi.com/escuela/datos/ · https://platzi.com/ruta/gestion-de-proyectos-y-metodologias-agiles/ · https://platzi.com/ruta/comunicacion/
- DataCamp en español: https://www.datacamp.com/es/courses-all
