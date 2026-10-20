# RutaPro: Sistema de recomendación de cursos profesionales

## 1. Definición del caso (Fase 1)

**Organización (ficticia).** RutaPro es una plataforma de empleabilidad universitaria que orienta a estudiantes sobre qué cursos en línea tomar para fortalecer su perfil profesional. No dicta cursos propios: reúne en un solo lugar cursos de Coursera, edX, Platzi y DataCamp y los recomienda.

**Problema.** Coursera y edX ofrecen miles de cursos sobre los mismos temas. Un estudiante de Estadística que quiere ser Data Analyst ve la misma lista de "más populares" que un estudiante de Administración que apunta a gestión de proyectos. El resultado es pérdida de tiempo, cursos abandonados por estar por encima (o por debajo) del nivel del estudiante y rutas de aprendizaje sin orden.

**Usuarios.** Estudiantes universitarios de 5.º a 10.º ciclo de carreras cuantitativas y de negocios (Estadística e Informática, Economía, Ingeniería Industrial, Ingeniería de Sistemas, Administración).

**Catálogo (ítems).** 107 cursos, especializaciones y certificados de Coursera, edX, Platzi y DataCamp (mitad en español y mitad en inglés), sobre Python, SQL, Excel, Power BI, Tableau y visualización, Estadística, R, Machine Learning, Gestión de proyectos, Habilidades blandas y Análisis de datos.

**Decisión que apoya el sistema.** ¿Qué 5 o 10 cursos debería mirar primero este estudiante, dado su perfil y lo que ya ha hecho?

**Escenarios de uso.**
- A. Estudiante con historial: entra a RutaPro y recibe su Top-N personalizado.
- B. Estudiante nuevo (cold start): llena un formulario corto (carrera, ciclo, nivel, idioma, intereses, objetivo laboral) y recibe recomendaciones sin tener historial.

**Interacción positiva.** Inscripción o finalización de un curso, o una calificación de 4 o más. Las vistas y los favoritos se registran como señales débiles.

**Restricciones.** No existen interacciones reales entre estudiantes y cursos, por lo que se simulan con reglas explícitas y se declaran como simuladas. El catálogo sí es real (obtenido por web scraping). Los cursos de edX no publican rating ni duración en su listado y DataCamp no publica un rating promedio, así que esos valores se imputan y se marcan.

## 2. Objetivos del proyecto

### Objetivo general
Diseñar, construir, evaluar e implementar un sistema de recomendación Top-N de cursos profesionales para RutaPro, basado en un dataset propio (catálogo real obtenido por web scraping de Coursera, edX, Platzi y DataCamp, junto con perfiles e interacciones simuladas con reglas documentadas), que ayude a estudiantes universitarios a elegir cursos acordes con su nivel, sus intereses y su objetivo laboral.

### Objetivos específicos
1. Definir el problema de negocio de RutaPro, precisando el usuario objetivo, el catálogo de cursos y la decisión que apoyará el sistema.
2. Construir mediante web scraping un catálogo de al menos 100 cursos, con aproximadamente la mitad en español (Coursera, edX, Platzi y DataCamp), y complementarlo con 150 perfiles de estudiantes y unas 3 700 interacciones simuladas con reglas explícitas y reproducibles.
3. Caracterizar mediante análisis exploratorio la actividad de usuarios y cursos, la popularidad (cola larga) y la dispersión de la matriz usuario-curso.
4. Implementar un modelo de referencia por popularidad y tres algoritmos de familias distintas: filtrado basado en contenido (TF-IDF y similitud del coseno), filtrado colaborativo (KNN) y un modelo híbrido de factorización (LightFM), este último como aporte de innovación.
5. Comparar los modelos con un protocolo reproducible de partición temporal, usando Precision@K, Recall@K, Hit Rate@K y NDCG@K para K = 5 y K = 10.
6. Seleccionar el modelo más conveniente para RutaPro considerando desempeño, costo computacional y facilidad de explicación al estudiante.
7. Desplegar un prototipo en Streamlit que genere recomendaciones Top-N para un estudiante existente y para un estudiante nuevo sin historial.

### Criterios de éxito
- **Técnico:** el modelo seleccionado supera al baseline de popularidad en NDCG@10 y Hit Rate@10 sobre el conjunto de prueba.
- **Cobertura:** el modelo recomienda una parte amplia del catálogo (no solo los 10 cursos más populares), medida como cobertura de catálogo.
- **Cold start:** un estudiante nuevo recibe 10 recomendaciones coherentes con su objetivo laboral y su nivel declarado.
- **Negocio (después del despliegue):** porcentaje de estudiantes que se inscriben en al menos un curso recomendado y tasa de finalización de esos cursos.
