# AviDetect
Este repositorio contiene la documentación, dataset y modelo para el proyecto del módulo de inteligencia artificial TC3002B.


# Descripción del Proyecto
En este proyecto se creará un modelo de clasificación de imagenes con la capacidad de distinguir entre drones y aves.


# Contexto
En el módulo se revisan redes neurolanes de capas densas y convolutivas cuyas funciones y parámetros le permitirán al modelo a partir de las imagenes.


# Descripción del dataset
El dataset utilizado es [Drone vs Bird:Aerial Object Classification Dataset](https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird) disponible públicamente [Kaggle](https://www.kaggle.com) bajo la licencia **Apache 2.0**.

"Kaggle es una plataforma web que reúne la comunidad Data Science más grande del mundo, con más de 536 mil miembros activos en 194 países, recibe más de 150 mil publicaciones por mes, que brindan todas las herramientas y recursos más importantes para progresar al máximo en data science. Kaggle, al igual que Liora, tiene una interfaz Jupyter Notebooks personalizable y sin configuración." (Mary, 2026) 

<div align="center">
  <img src="./resources/kaggleDataset.png" alt="Figura 1: Dataset en Kaggle">
  <em>Figura 1: Dataset en Kaggle</em>
</div>

<div align="center">
  <img src="./resources/kaggleDatasetImages.png" alt="Figura 2: Imágenes en el dataset">
  <em>Figura 2: Imágenes en el dataset</em>
</div>

Existen otros datasets que contienen imágenes de drones y aves con el mismo objetivo. Este dataset se eligió puesto que algunas imágenes están en alta resolución y requiere de un espacio en disco de 1.72 GB.

Un dataset alternativo es [Drone-vs-Bird](https://www.kaggle.com/datasets/romsham/dronevsbird-foryolo) igualmente disponible en **Kaggle**, que contiene más imágenes y en alta resolución, sin embargo, requiere de 7.1 GB de espacio en disco.

# Metodología

## Separación

El dataset contiene dos carpetas con un total de 4106 imágenes: 
* bird: 1607 imágenes de aves en diversos escenarios
* drone: 2499 imágenes de drones en diversos escenarios

Es notorio que el dataset está desbalanceado, drones contiene 892 imágenes más que bird, por lo que la proporción es de aproximadamente 60/40 a favor de drones. 

Es necesario hacer la separación del dataset en train, validation y test. Típicamente esta división es 70/15/15 u 80/10/10. Para datasets pequeños como el caso de *Drone vs Bird*, preservar entre 10 y 30% para test suele estar bien. Se utilizó la librería split-folders para hacer la división, finalmente se obtuvo:

* 70% para train
* 15% para validation
* 15% para test

## Preprocesamiento

Dado que el dataset está desbalanceado, es esperado que el modelo sea mejor para identificar la clase mayoritaria, es decir, drones. Para mitigar el desbalanceo de clases, se consideró una técnica de oversampling sobre la clase minoritaria, seguido de data augmentation a todas las clases, siguiendo enfoques similares a los propuestos por Guerrero et al. (2024) en su paper [A Data Augmentation Methodology to Reduce the Class Imbalance in Histopathology Images](https://link.springer.com/article/10.1007/s10278-024-01018-9), donde la aumentación dirigida permitió reducir el sesgo hacia las clases mayoritarias en tareas de clasificación de imágenes.
Es importante destacar que estas alteraciones en el dataset se hacen únicamente en el set de entrenamiento y que previamente se realizó una normalización de color. Los set de validation y test se dejan como están.


# Referencias
Escobar Díaz Guerrero, R., Carvalho, L., Bocklitz, T. et al. A Data Augmentation Methodology to Reduce the Class Imbalance in Histopathology Images. J Digit Imaging. Inform. med. 37, 1767–1782 (2024). https://doi.org/10.1007/s10278-024-01018-9
Galdran, A., Carneiro, G., González Ballester, M.A. (2021). Balanced-MixUp for Highly Imbalanced Medical Image Classification. In: de Bruijne, M., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science(), vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_31
Mary. (2026, February 25). Kaggle: todo lo que hay que saber sobre esta plataforma. Liora. https://liora.io/es/kaggle-todo-lo-que-hay-que-saber-sobre-esta-plataforma