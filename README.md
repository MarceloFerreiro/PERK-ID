# PERK-ID

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

link a las imágenes: https://drive.google.com/file/d/1c4pNwsGYY6k2csR25O8CRM-CscworUFK/view?usp=sharing

El .csv que nos interesa es el ``data/dataset_con_imágenes.csv``. 

El `src/query.py` te permite pasarle una imagen por argumento y te genera un html con los resultados. Hay una pastilla `data/test/test.jpg` para probar.

Las caroetas pruebas y preprocesado ambas tienen un `src/preprocesado/extract_features.py`, ambos son iguales.

La app tocha está en la carpeta PillIdentifier.

</details>
