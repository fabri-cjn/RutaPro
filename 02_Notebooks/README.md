# Notebooks de RutaPro

Ejecutar en orden. Cada notebook lee lo que genera el anterior.

| Notebook | Fase | Genera |
|---|---|---|
| `00_Construccion_Dataset.ipynb` | 2. Dataset | `01_Datos/cursos.csv`, `usuarios.csv`, `interacciones.csv` |
| `01_EDA.ipynb` | 3. Preparación y EDA | `01_Datos/pares_usuario_curso.csv`, figuras 01-11 |
| `02_Modelos.ipynb` | 4. Modelamiento | `resultados/hiperparametros.json`, grillas de ajuste, figuras 12-15 |
| `03_Evaluacion.ipynb` | 5. Evaluación y selección | `resultados/metricas_prueba.csv`, `metricas_mas_alla.csv`, figuras 16-18 y el modelo final en `03_App/modelo/` |
| `rutapro_rec.py` | — | Módulo con métricas y modelos, compartido por los notebooks y la app |

## Entorno
Probado con Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, scipy 1.18, matplotlib 3.11 y LightFM 1.17.

```bash
pip install numpy pandas scipy scikit-learn matplotlib requests beautifulsoup4 jupyter
```

### Instalar LightFM
LightFM 1.17 no compila con `pip install lightfm` en Python 3.12 o superior. Hay tres opciones:

1. **Recomendada:** usar conda: `conda install -c conda-forge lightfm`.
2. Crear un entorno con Python 3.10 o 3.11 y luego `pip install lightfm`.
3. Compilar desde el código fuente con Cython 3. Es lo que se hizo para este proyecto:
   ```bash
   git clone --depth 1 https://github.com/lyst/lightfm.git && cd lightfm
   # en setup.py, dentro de cythonize(...):
   #   compiler_directives={'language_level': "3", 'legacy_implicit_noexcept': True}
   # y reemplazar "-march=native" por "-O2"
   pip install cython && python setup.py cythonize && python setup.py build_ext --inplace
   # copiar la carpeta lightfm/ al site-packages del entorno
   ```

La aplicación no necesita LightFM: usa los embeddings exportados y calcula los puntajes con numpy.
