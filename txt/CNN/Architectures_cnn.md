# intro

Поговорим про историю моделей связанных с CV так как это
* поможет узнать о компонентах из которых построенны современные модели. Как и зачем они возникли.
* поможет лучше запомнить (nrrative)

# ImageNet

# AlexNet

* from University (Йорк Англия)
* разр - меньше, каналов больше
* 2 видеокарты
* Нововведения: ReLU, Dropout,  SGD + momentum + эвристика,  ансамбль

# ZFNet

* Йоркский университет
* сильно упала ошибка
* никаких архитектурных новшеств - только подбор гипер-параметров

# VGG

* from University (Оксфорд Англия)
* 2-е место на ImageNet
* basic block из чего он
* свертки 3x3 без сжатия
* Глубина и размеры (сжати)

# GoogleNet

* количество параметров
* Inception
* 22 честных слоя
* StemNetwork
* 1 FC слой
* GAP
* Auxilarry classifier

# ResNet
* Bottleneck vs VGG param number
* Resudial connection
* Common modules: Stem, VGG block, AvgPooling, 1xFC

# Grouped Convolution
Resnext
MobileNet 
Depthwise separable convolution
Inverted residual block