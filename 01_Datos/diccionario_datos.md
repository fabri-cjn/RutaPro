# Diccionario de datos: RutaPro

Llaves: `course_id` (cursos), `user_id` (usuarios), `interaction_id` (interacciones). La tabla `interacciones` se relaciona con `usuarios` por `user_id` y con `cursos` por `course_id` (relación muchos a uno en ambos casos).

## cursos.csv (107 filas, una por curso). Origen: web scraping (real)
| Variable | Tipo | Descripción | Valores / ejemplo |
|---|---|---|---|
| course_id | texto (PK) | Identificador asignado por el equipo | C001 … C107 |
| titulo | texto | Nombre del curso en la plataforma | "SQL for Data Science" |
| plataforma | categórica | Plataforma de origen | Coursera, edX, Platzi, DataCamp |
| institucion | texto | Universidad, empresa o plataforma (con docente) que dicta el curso | IBM, UNAM, Platzi (Freddy Vega) |
| idioma | categórica | Idioma del curso | en, es |
| categoria | categórica | Categoría asignada por reglas de palabras clave del título | 11 categorías |
| tipo | categórica | Formato del programa | Curso, Especialización, Certificado profesional, Proyecto guiado |
| dificultad | ordinal | Nivel publicado por la plataforma (o inferido del título en edX) | Básico, Intermedio, Avanzado |
| duracion_horas | numérico | Horas aproximadas (en Coursera se asume ~10 h por semana) | 0.75 – 110 |
| rating | numérico | Calificación promedio publicada | 1 – 5 |
| n_resenas | numérico | Número de reseñas u opiniones publicadas | 14 – 281 000 |
| habilidades | texto | Habilidades (*skills*) o temas del temario, separados por "; " (en inglés para Coursera, en español para el resto) | "SQL; Data Analysis" |
| url | texto | Enlace a la ficha del curso | |
| fecha_extraccion | fecha | Fecha del scraping | 2026-10-19 |
| dificultad_imputada, duracion_imputada, rating_imputado, n_resenas_imputado | booleano | `True` si el valor fue imputado | |
| habilidades_fuente | categórica | Origen de las habilidades | skills publicadas por la plataforma / temario publicado por la plataforma / inferidas del título |

## usuarios.csv (150 filas, una por estudiante). Origen: SIMULADO
| Variable | Tipo | Descripción | Valores |
|---|---|---|---|
| user_id | texto (PK) | Identificador | U001 … U150 |
| carrera | categórica | Carrera universitaria | 5 carreras |
| ciclo | entero | Ciclo académico | 5 – 10 |
| nivel | ordinal | Nivel técnico autodeclarado | Básico, Intermedio, Avanzado |
| idioma_preferido | categórica | Idiomas en los que estudia | es (solo español), es+en |
| objetivo_laboral | categórica | Puesto al que apunta | Data Analyst, BI Analyst, Data Scientist, Analista Financiero, Marketing Analyst, Project Analyst |
| intereses | texto | 2–3 categorías de interés, separadas por "; " | |
| habilidades_actuales | texto | Categorías que ya domina, separadas por "; " | |
| fecha_registro | fecha | Fecha de alta en RutaPro | 2026-01-01 – 2026-08-31 |

## interacciones.csv (3 680 filas, una por evento). Origen: SIMULADO
| Variable | Tipo | Descripción | Valores |
|---|---|---|---|
| interaction_id | texto (PK) | Identificador del evento | I00001 … |
| user_id | texto (FK) | Estudiante | |
| course_id | texto (FK) | Curso | |
| timestamp | fecha-hora | Momento del evento | 2026-01 – 2026-10-15 |
| evento | categórica | Tipo de interacción | vista, favorito, inscripcion, finalizacion |
| calificacion | entero | Nota del estudiante al curso (solo en `finalizacion`) | 1 – 5 |
| peso_evento | entero | Intensidad de la señal implícita | vista 1, favorito 2, inscripcion 3, finalizacion 4 |

**Interacción positiva** (para evaluar): par usuario-curso con `inscripcion` o `finalizacion`, o con calificación de 4 o más.
