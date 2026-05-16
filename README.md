# PERK-ID

![vaites, ocurreu un erro ca imaxe](https://github.com/MarceloFerreiro/PERK-ID/blob/main/data/icono.jpeg?raw=true)

<details> 
<summary> TODO </summary>

- transformaciones mas chulas
- hacer un script que sea postprocess index
- [x] query con nn de scikit learn
- [x] nn 
</details>

<details>
<summary> Enunciado de la práctica </summary>

El objetivo del examen final es realizar una presentación oral del proyecto, en la que todos los miembros del grupo (máximo tres estudiantes) participen de forma activa. La evaluación se basará en criterios específicos que permitan demostrar una comprensión completa del trabajo realizado.

Fecha del examen final: 19 maio 2026 (15:30-18:30).

Calculo de la nota final = Presentacion final (50-70%) + Prácticas - Ej. 1 & 2 (30-50%).

El proyecto consistirá en una aplicación web o de escritorio (con modelos ML/DL/LLMs) desarrollada con Vibe Coding, utilizando cualquier agente que consideres oportuno (Antigravity, AI Studio, etc.). Si necesitas realizar llamadas a APIs, utiliza versiones gratuitas (por ejemplo, Nvidia, Gemini, OpenRouter, GitHub, Qwen, entre otros).

El uso de Vibe Coding no exime de comprender el código. Mantén el proyecto lo más sencillo posible. En el caso de las aplicaciones web, procura crear apps simples, preferiblemente estáticas y sin servidor, empleando únicamente HTML, CSS y JavaScript, de modo que puedan desplegarse en GitHub Pages, Vercel u otro servicio gratuito.

Forma tu equipo y selecciona el proyecto en el siguiente enlace (tienes acceso completo al archivo):

https://udcgal-my.sharepoint.com/:x:/g/personal/c_munteanu_udc_es/IQAqyRn8ONduQKOTDQfj7r4FAXh3QeMVZIopBd5JBNdDs5E?e=VOitnI

La presentación debe reflejar un entendimiento profundo del proyecto, utilizando un lenguaje claro y conciso. Las respuestas a las preguntas deberán evidenciar un dominio sólido de todo el proceso llevado a cabo.

Recuerda: debes comprender cada decisión que tomes en tu proyecto. Pregúntate siempre por qué. Este trabajo busca demostrar que realmente entiendes lo que haces, no solo que lo implementas.

Cada grupo dispondrá de 15 minutos para la presentación (aproximadamente 5 minutos por persona), tras los cuales se plantearán 1 o 2 preguntas por estudiante. El tiempo total de evaluación por grupo será de 30 minutos. Dado el límite de tiempo, se recomienda utilizar alrededor de 15 diapositivas.

Finalmente, envía en el Campus Virtual el enlace al repositorio de GitHub de tu proyecto, donde todos los miembros del grupo aparezcan como colaboradores. El repositorio debe incluir:

    Todo el código fuente.
    Una página README que indique que se trata de un proyecto de la asignatura “Bioinformática y Medicina” del Grado en Inteligencia Artificial de la Universidade da Coruña.
    El enlace a la aplicación web (si aplica).
    El enlace al DOI del reléase GitHub en el repositorio europeo Zenodo (https://zenodo.org).
    El enlace a la presentación para el examen.

</details>

> [!WARNING]
> Ojo si cambiais `config.json`, eso afecta a todo. Es decir, el `.npz` de características que generais con una configuracion (con `src/images2index.py`) solo se puede evaluar (con `src/eval.py`) con la misma configuracion. Hay que tenerlo presente y bautizar bien a esos archivos `.npz`. 

Entorno
    
    git clone https://github.com/MarceloFerreiro/PERK-ID
    cd PERK-ID
    ./scripts/entorno.sh

**Descargar dataset**, necesita el `.env`

    python -m src.scraper 


**Construir índice**


    (venv) » python -m src.images2index -h
    usage: python -m src.images2index [-h] [--dir DIR] [--workers WORKERS] [--out OUT] [--save-every SAVE_EVERY] [-v]
    
    options:
      -h, --help            show this help message and exit
      --dir DIR
      --workers WORKERS     Número de hilos a usar
      --out OUT             Ruta al índice .npz
      --save-every SAVE_EVERY
                            Cada cuantas imágenes se actualiza el indice en disco

    
**Evaluar el indice**. Importante para ver como de buenas y/o eficientes son las características. Antes de evaluar nos tenemos que asegurar que el aumento de datos es bueno, o si no vamos a tener métricas cojonudas aunque las características sean malas.

    
    (venv) » python -m src.features.transform --h 5 --w 8 
    open tmp/transform_grid.png

![erro ca imaxe](https://github.com/MarceloFerreiro/PERK-ID/blob/main/data/docs/transform_grid.png?raw=true)

Tenemos que ser conscientes que la calidad de la evaluación depende de lo bueno que sea el aumento de datos.


    
    (venv) » python -m src.eval -h
    usage: python -m src.eval [-h] [--features FEATURES] [--images-dir IMAGES_DIR] [--eval-size EVAL_SIZE] [--topk TOPK] [--seed SEED]
    
    Evalua Top-1/Top-K sobre un indice de features
    
    options:
      -h, --help            show this help message and exit
      --features FEATURES   Ruta al índice .npz
      --images-dir IMAGES_DIR
                            Directorio con imagenes
      --eval-size EVAL_SIZE
                            Numero de imagenes para evaluar (muestreo aleatorio)
      --topk TOPK           K para Top-K accuracy
      --seed SEED

Verás algo como:

    Evaluadas: 50  Omitidas: 0
    Top-1 Accuracy: 0.6400
    Top-10 Accuracy: 0.9400
    Distancia media a la pastilla más cercana: 0.2383
    Duración media inferencia: 0.0871s
    Ranking medio: 1.0200
    La pastilla quedo fuera del top-10 en: 3/50 casos
                           P(Ranking <= X)
    ┌────────────────────────────────────────────────────────────┐
    │                                                       ▗▄▄▞▀│
    │                                      ▗▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▘    │
    │                                    ▗▞▘                     │ 0.9
    │                                  ▗▞▘                       │
    │                               ▄▄▞▘                         │
    │                          ▄▄▀▀▀                             │
    │                       ▄▞▀                                  │
    │                    ▄▞▀                                     │ 0.8
    │              ▄▄▄▀▀▀                                        │
    │            ▄▀                                              │
    │          ▗▞                                                │
    │         ▄▘                                                 │
    │       ▗▞                                                   │
    │      ▟▘                                                    │ 0.7
    │    ▄▀                                                      │
    │  ▄▀                                                        │
    │▄▀                                                          │
    └────────────────────────────────────────────────────────────┘
           2             4            6            8           10


![erro ca imaxe](https://github.com/MarceloFerreiro/PERK-ID/blob/main/data/docs/eval.png?raw=true)

Tanto para la construcción del índice como para la evaluación, se hará uso interno de `config.toml`

Generar dataset de bouding box [el json se genera en este enlace](https://2d-on-2d.annotate.photo/):

    (venv) » python -m src.json2csv 2D-on-2D_labeling_save\(1\).json tmp.csv
    Done — 36 row(s) written to: tmp.csv

**Entrenamiento de CNN para Bounding Box regression**. Naturalmente para este paso es fundamental tener un `.csv` con anotaciones de bb, ver paso de arriba.

    (venv)  » python -m src.models.train -h
    usage: python -m src.models.train [-h] [--images-dir IMAGES_DIR] [--bbox-csv BBOX_CSV] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--lambda-bbox LAMBDA_BBOX] [--output-dir OUTPUT_DIR]
                                      [--checkpoint-interval CHECKPOINT_INTERVAL] [--latent-dim LATENT_DIM]
    
    Train masked autoencoder on pill images
    
    options:
      -h, --help            show this help message and exit
      --images-dir IMAGES_DIR
                            Directory containing pill images
      --bbox-csv BBOX_CSV   Path to CSV with bounding box annotations
      --batch-size BATCH_SIZE
                            Batch size for training
      --epochs EPOCHS       Number of epochs to train
      --lr LR               Learning rate
      --lambda-bbox LAMBDA_BBOX
                            Weight for bounding box loss
      --output-dir OUTPUT_DIR
                            Directory to save model checkpoints
      --checkpoint-interval CHECKPOINT_INTERVAL
                            Save checkpoint every N epochs
      --latent-dim LATENT_DIM
                            Dimensionality of latent space


La ejecución de ese script produce pesos (`.pt`) y logs (`.csv`). Una vez entrenado podemos evaluarlo de la siguiente forma:

    (venv) » python -m src.models.net --params data/params/model_epoch_005.pt --latent-dim 16 --output tmp/prueba_reconstruccion_modelo.png
     open tmp/prueba_reconstruccion_modelo.png

La calidad de la reconstrucción es una proxy de la calidad de las características.


![erro ca imaxe](https://github.com/MarceloFerreiro/PERK-ID/blob/main/data/docs/reconstruccion.png?raw=true)

---

## App móvil / web (FastAPI + Flutter)

El prototipo CLI se migró a un servidor FastAPI y una app Flutter multiplataforma (Android, web). La app abre la cámara directamente, hace la foto y muestra los resultados más similares del índice.

### Requisitos previos

- Python 3.10+, virtualenv (`venvPerk`)
- Flutter SDK (`~/flutter/bin` en el PATH)
- Android SDK con platform-tools (`~/Android/Sdk`)
- El índice ya construido: `data/features.npz` + `data/bow.pkl`

### 1. Levantar el servidor

```bash
source venvPerk/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

El servidor carga el índice en memoria al arrancar y expone:

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servidor y tamaño del índice |
| `POST /query?topk=10` | Sube una imagen, devuelve los top-K más similares en JSON |
| `GET /images/<filename>` | Sirve las imágenes del dataset |
| `GET /` | App web (si está compilada, ver más abajo) |

La IP de la máquina en la red local se ve con `hostname -I`. Todos los dispositivos en la misma WiFi pueden acceder al servidor.

### 2. App Android

**Primera vez** (compila en ~20-40 min, luego es rápido):

```bash
cd pillsearch
flutter run
```

Conectar el móvil por **depuración inalámbrica** (Android 11+):
1. Activar en Ajustes → Opciones de desarrollador → Depuración inalámbrica
2. Parear el dispositivo:
    ```bash
    ~/Android/Sdk/platform-tools/adb pair <IP>:<puerto-pareo>
    ```
3. Conectar:
    ```bash
    ~/Android/Sdk/platform-tools/adb connect <IP>:<puerto>
    ```
4. `flutter run` detecta el dispositivo automáticamente

En la app, pulsar el icono de ajustes (⚙) e introducir la URL del servidor: `http://<IP-maquina>:8000`

### 3. App web

**Modo desarrollo** (abre Chrome directamente):
```bash
cd pillsearch
flutter run -d chrome
```

**Build estático** (queda servido por el propio servidor FastAPI):
```bash
cd pillsearch
flutter build web
# Reiniciar el servidor — la app web estará en http://<IP>:8000
```

Para instalarla como PWA: Chrome muestra un botón de instalación en la barra de direcciones. En móvil, Safari → Compartir → "Añadir a pantalla de inicio".

> **Nota sobre HTTP y cámara en web:** los navegadores solo permiten acceso a la cámara en `localhost` o con HTTPS. Para usar la cámara desde otro dispositivo en la red local, añadir la IP como origen seguro en Chrome:
> `chrome://flags/#unsafely-treat-insecure-origin-as-secure`

### Workflow de desarrollo

```
# Terminal 1 — servidor siempre corriendo
source venvPerk/bin/activate && uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — app (hot reload activo con 'r', reinicio con 'R')
cd pillsearch && flutter run
```

Cambios en `api/main.py` → reiniciar el servidor (Ctrl+C y volver a lanzar).  
Cambios en `pillsearch/lib/` → Flutter recarga automáticamente con hot reload.

---

**Interfaz con el Kotlin.** Dicen por [aqui](https://discuss.kotlinlang.org/t/integrating-a-python-code-with-kotlin/24639) que se puede meter python en kotlín, en concreto usando una cosa de java, estilo asi ([lee esto de este enlace bien de todas formas](https://www.baeldung.com/java-lang-processbuilder-api)):

    Process process = new ProcessBuilder("python", "-m", "src.Ranker", "data/imagenes_alt/012c6e037f099712479ede765f82e3f3.jpeg").start();


Desconozco el tema la verdad, en cualquier caso la _API_ es esta:

    (venv) python -m src.Ranker -h
    usage: python -m src.Ranker [-h] [--features FEATURES] [--output_dir OUTPUT_DIR] [--images-dir IMAGES_DIR] [--topk TOPK] image_path
    
    Ranking
    
    positional arguments:
      image_path
    
    options:
      -h, --help            show this help message and exit
      --features FEATURES   Ruta al índice .npz
      --output_dir OUTPUT_DIR
                            directorio donde se guardan los resultados.
      --images-dir IMAGES_DIR
                            Directorio con imagenes
      --topk TOPK           K para Top-K accuracy

El k es cuantas imagenes te devuelve así yo diria de devolver flow 100 o así y que se pueda deslizar o que haya lo típico de _ver 10 pastillas más_.

En cualquier caso ese programa escribre un csv con:

    cat results/012c6e037f099712479ede765f82e3f3.csv

    Path,Distance
    data/imagenes_alt/012c6e037f099712479ede765f82e3f3.jpeg,0.0
    data/imagenes_alt/802c610734641e0a2722164eb0047124.jpeg,51.75687026977539
    data/imagenes_alt/48b982d8e992e0ba05e3828f0a24a430.jpeg,52.08281707763672
    data/imagenes_alt/dc693a6ae970bd0dcd68be0d68430a9e.jpeg,52.757781982421875
    data/imagenes_alt/9f406249a791c8a666b74ebd0617c391.jpeg,53.97950744628906
    data/imagenes_alt/54e5a307cb22dc4b426240a897e6d584.jpeg,54.46134948730469
    data/imagenes_alt/9db772ce1ce36de5e860738865233a22.jpeg,56.36448287963867
    data/imagenes_alt/7ba008c62b07e9853c85b229fa8a7a36.jpeg,57.79072189331055
    data/imagenes_alt/38940f4c9c13479dac706d89a17708bd.jpeg,58.11359786987305
    data/imagenes_alt/f88a7a7987f79968cb84c99e6e5e9971.jpeg,58.468685150146484

