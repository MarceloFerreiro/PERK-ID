# PERK-ID

<details> 
<summary> TODO </summary>

- transformaciones mas chulas
- hacer un script que sea postprocess index
- query con nn de scikit learn
- nn 
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


<details>
<summary> Como usar </summary>

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





</details>
