# RutaPro: prototipo en Streamlit

Aplicación que genera recomendaciones Top-N de cursos con el modelo seleccionado en `02_Notebooks/03_Evaluacion.ipynb`: **LightFM híbrido (WARP, 64 factores, 200 épocas) + reglas de idioma y nivel**.

## Ejecutar localmente
```bash
cd 03_App
pip install -r requirements.txt     # streamlit, numpy, pandas (no requiere LightFM)
streamlit run app.py
```
La app se abre en `http://localhost:8501`.

## Publicarla con un enlace (Streamlit Community Cloud, gratis)
1. Subir la carpeta del proyecto a un repositorio de GitHub.
2. Entrar a https://share.streamlit.io, elegir **New app**, seleccionar el repositorio e indicar `03_App/app.py` como archivo principal.
3. Streamlit instala `03_App/requirements.txt` y entrega un enlace público para la sustentación.

## Qué hace
| Función | Cómo |
|---|---|
| **Estudiante registrado** | Se elige un estudiante, se muestran su perfil y su historial, y se recomiendan cursos con su embedding aprendido. No se repiten cursos en los que ya se inscribió. |
| **Estudiante nuevo (*cold start*)** | Formulario de registro (carrera, ciclo, nivel, idioma, objetivo laboral, intereses, temas que domina). La representación del estudiante se construye sumando los embeddings de esos atributos. |
| **Reglas de negocio** | Activadas por defecto: solo cursos en español para quien no estudia en inglés y nunca más de un nivel por encima del suyo. Se pueden desactivar para la demostración. |
| **Explicación** | Cada curso muestra por qué se recomienda: interés coincidente, habilidad clave para el objetivo laboral, ajuste de nivel e idioma. |
| **Comparación** | Muestra la recomendación personalizada al lado del baseline de popularidad. |
| **Incidencias** | Si falta el archivo del modelo, se usa la popularidad como respaldo. Si las reglas dejan menos de N cursos, se avisa y se sugiere desactivarlas. Si el formulario no tiene intereses, se recomienda con el resto del perfil. |

## Archivos
| Archivo | Contenido |
|---|---|
| `app.py` | Interfaz y cálculo de puntajes (numpy) |
| `modelo/lightfm_final.npz` | Embeddings y sesgos de LightFM, popularidad y orden de IDs (exportados por el notebook 03) |
| `modelo/config.json` | Modelo, hiperparámetros y métricas en prueba |
| `modelo/inscritos.json` | Cursos en los que ya se inscribió cada estudiante |
| `modelo/cursos.csv`, `modelo/usuarios.csv` | Copias del catálogo y de los perfiles |
| `capturas/` | Capturas de pantalla como evidencia del funcionamiento |

## Guion sugerido para la demostración (4 minutos)
1. **Estudiante registrado (U001, Project Analyst, solo español):** mostrar el historial y las recomendaciones, y leer la explicación del curso n.º 1.
2. **Comparación con popularidad:** activar el interruptor y señalar que la popularidad recomienda Excel avanzado y BI, mientras que el modelo prioriza gestión de proyectos y habilidades blandas.
3. **Reglas:** elegir un estudiante bilingüe y luego uno de solo español, para mostrar que con las reglas activas no aparecen cursos en inglés.
4. **Estudiante nuevo:** llenar el formulario en vivo con un perfil que proponga el jurado y pulsar **Recomendarme cursos**.
