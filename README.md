# MultispectralEarthSimulator

En este repositorio esta el código y datos necesarios para replicar mi trabajo final de grado, en el que
hemos diseñado un modelo de Deep Learning capaz de generar una imagen de alta calidad de 25 cm/píxel de resolución a partir de una imagen de satélite de 8 m/píxel de resolución, en concreto el
de Sentinel-2. Para más detalles sobre el proceso de realización y los detalles, podéis consultar el informe final en la carpeta de documentos.

La primera arquitectura utilizada es la pix2pix para la fase de recreación.

[PAPER](https://arxiv.org/pdf/1611.07004v3.pdf) [GitHub](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

El primer paso a realizar es descargar la implementación en pytorch de la pix2pix, para ello podéis usar el siguiente comando:

git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git

Para poder Generar los distintos datasets debeis seguir los siguientes pasos:

* Descargar el TCI y SCL de Sentinel-2 'clipSentinel.py': proporcionamos archivo .shp en la carpeta de datos perteneciente a la zona de la provincia de Barcelona. Este código recibe como entrada el path al archivo .shp, es necesario un usuario y contraseña de Copernicus access Hub para descargar datos. Si no dispone de cuenta en la carpeta de datos se encuentra la imagen TCI y SCL de la zona utilizada.
* Generar recortes: Es necesario pasar el archivo TCI y SCL a una resolución de 8 m/píxel (los datos que proporcionamos ya la tienen) y más adelante generar los distintos recortes de 32x32 del SCL y el TCI, estos recortes siguen estando georeferenciados. El archivo 'generarRasterTiles.py' recibe el path de la imagen y genera los recortes.
* Seleccionar dataset: mediante la función 'selectForest' del archivo 'utils.py' comparamos los recortes de TCI con SCL y descartamos las imágenes que el clasificador (SCL) no considere bosque, con este dataset nos quedamos con casi 3000 imágenes de bosque.
* Descargar imágenes alta calidad: Una vez obtenemos las imágenes que nos interesan, procedemos a hacer peticiones al ICGC para descargar cada una de estas imágenes en alta calidad (25 cm/píxel), para realizarlo se utiliza archivo 'servicioWMS_ICGC.py'.
* Generar datasets: En el archivo 'utils.py' se encuentran las distintas funciones para generar los distintos datasets que deseemos, se aplica validación cruzada, dividimos los datos entre train, validation, test. Se le pasa al archivo 'utils.py' el tipo de dataset entre '25cmDecimado', '50cmDecimado', '1mDecimado', '25cmDistorsionado' o '25cmSentinel', es necesario también el path a las imágenes de alta calidad y en el tipo '25cmSentinel' el path a las imágenes de satélite.
* Combinar datasets: Con el archivo 'combine_A_and_B.py' le damos el path a la carpeta con las imágenes de entrada y groundtruth de la red, este las combina para que la red pueda trabajar con ellas. A continuación tenéis el ejemplo de uso:

python codigo/combine_A_and_B.py --fold_A /path/images/entrada --fold_B /path/images/groundtrhuth --fold_AB /path/dataset/generado

A continuación os mostramos ejemplos de como hemos realizado el entrenamiento y posteriormente el testeo mediante la pix2pix:

Train:

python pytorch-CycleGAN-and-pix2pix/train.py --dataroot path/a/dataset/ --model pix2pix --display_id -1 --serial_batches --load_size TAMAÑO_IMAGEN --crop_size TAMAÑO_PATCHES --name NOMBRE_MODELO --niter 80 --niter_decay 80

Test:

python pytorch-CycleGAN-and-pix2pix/test.py --dataroot path/a/dataset/ --load_size TAMAÑO_IMAGEN --name NOMBRE_MODELO --model pix2pix --results_dir ./resultados/

Para la segunda fase, la de super Resolución se utiliza la arquitectura CARN:

[PAPER](https://arxiv.org/pdf/1803.08664v5.pdf) [GitHub](https://github.com/nmhkahn/CARN-pytorch)

En el archivo 'CARN/checkpoint/' se encuentran los pesos de los modelos entrenados, uno aplica una aumento de x4 de 1m/pixel a 25 cm/pixel, '1m-25cm.pth' y otro aumenta un x2 de 50 cm/pixel a 25 cm/pixel. Para que se nos generen muestras de ejemplo se debe ejecutar lo siguiente:

python CARN/sample.py --model carn --ckpt_path PATH/A/CHECKPOINT/ --test_data_dir PATH/DATASET/TEST/ --scale ESCALA(2 o 4)
