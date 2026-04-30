# Сегментация

## Темы

- [Сегментация](Semantic_segmentation.ipynb)
- [Датасет COCO](COCO_dataset.ipynb)
- [Albumentation](Albumentations.ipynb)
- [Segmentation models PyTorch (SMP)](SMP.ipynb)
- [Panoptic segmentation](Panoptic_segmentation.ipynb)

## Сегментация

В отличии от классификации, в задаче сегментации пространственные размерности(H,W) выхода модели (CxHxW где С- количество предсказываемых классов) должен совпадать с размерностью входа (3xHxW где 3-количество каналов для RGB изображения).

В качестве backbone можно использовать архитектуру аналогичную той что использовалась в сетях для классификации, но при этом придется сохранать признаки полученные на промежуточных слоях, так как информация о местоположении объектов важна:

Так же для сегментации популярны Unet подобные архитектуры состоящии из энкодера(encoder) и декодера(decoder) со skip-connection

В качестве метрики используется IoU/Dice

В качестве loss может использоваться попиксельный BCE, либо дифференцируемые варианты IoU/Dice и их комбинации с лосс функциями других видов:

Материалы: [Semantic_segmentation.ipynb](Semantic_segmentation.ipynb)

## Датасет COCO

Для задачи сегментации каждый пиксель должен иметь метку соответствующую классу объекта к которому он относиться*.

Классическим датасетом с данными в таком формате является [COCO](https://cocodataset.org) (Common Objects in Context)

*В задачах Instance и Panoptic segmentation к меткам классов добавляются еще и id объектов.

Материалы: [COCO_dataset.ipynb](COCO_dataset.ipynb), [colab.research.google.com](https://colab.research.google.com/drive/1YsQsXifAgixBW6jALbvBSio4coGPHUXm#scrollTo=JvGV8hTB9N01), [pycocotools](https://pypi.org/project/pycocotools/), [cocodataset.org](https://cocodataset.org/#format-data)

## Albumentation

При работе с масками или BoundingBox аугментации(трансформации) которые применялись к изображению должны применяться и к таргету (маске или BoundingBox) аналогичным образом.

Обычные transform из [torchvision](https://pytorch.org/vision/0.9/transforms.html) не поддерживают такой функционал. Его поддержвают трансформации из пакета [torchvision.v2](https://pytorch.org/vision/stable/transforms.html) а так же сторонние библиотеки, например [Albumentaton](https://albumentations.ai/docs/)

Материалы: [Albumentations.ipynb](Albumentations.ipynb), [mask_augmentation](https://albumentations.ai/docs/getting_started/mask_augmentation/), [plot_transforms_getting_started](https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html#detection-segmentation-videos), [additional-targets](https://albumentations.ai/docs/4-advanced-guides/additional-targets/#core-mechanism-additional-targets-in-acompose)

## Segmentation models PyTorch (SMP)

Большинство моделей для сегментации содержит encoder и decoder блоки.

Библиотека [SMP](https://github.com/qubvel-org/segmentation_models.pytorch#architectures-) позволяет удобным образом комбинировать их между собой. И поддерживает работу с [TIMM](https://github.com/huggingface/pytorch-image-models).

Материалы: [SMP.ipynb](SMP.ipynb), [quickstart](https://smp.readthedocs.io/en/latest/quickstart.html)

## Panoptic segmentation

Panoptic segmentation бъединяет задачи Instance segmentation и Semantic segmentation.

Для каждого пикселя на изображении задаются два значения — номер класса(как в instance segmentation) и id объекта если объект счетный(thing: человек, машина)

Метрика: Panoptic Quality

[Блог 1](https://iq.opengenus.org/pq-sq-rq/)

[Блог 2](https://segments.ai/blog/panoptic-segmentation-datasets/)

Материалы: [Panoptic_segmentation.ipynb](Panoptic_segmentation.ipynb)


## Современные модели для сегментации

После
появления моделей на основе трансформерных архитектур их  стали применять для решения многих задач компьютерного
зрения. Классический ViT плохо подходит для выделения точных границ объектов.
Так как изображение превращается в последовательность путём нарезки на
фрагменты большого размера (16x16).

Что бы сохранить информацию о пиксельных
границах используют иерархические
архитектуры с меньшим размером patch и
локальным self-attention.

Примером такой архитектуры является[SWIN](https://arxiv.org/abs/2103.14030).

На базе иерархической трансформерной архитектуры построена модель SegFormer которая решает задачу попиксельной сегментации так же как и классические модели.

Затем появилась [MaskFormer](https://arxiv.org/abs/2107.06278) архитектуры. Принципиальным их отличием является то что предсказывается не класс отдельного пикселя а сразу маска класса.

В модели есть блок который предсказывает
вектор признаков соответствующий классу
(размерность С), а пиксельный декодер
возвращает  пространственную карту
признаков (CxHxW). Их перемножение позволяет
получить маску класса. Такая архитектура
позволяет решать задачи Instance и Panoptic
Segmentation. Дальнейшим развитием идеи стала
модель [OneFormer](https://arxiv.org/abs/2211.06220)
способная решать любую из этих задач.

Тип задачи задаётся текстовым промптом.

Модели семейства Segment Anything ([SAM](https://arxiv.org/abs/2304.02643)/[SAM2](https://arxiv.org/abs/2408.00714)) используют подобную архитектуру и позволяют использовать разные типы промпта.

Например координаты точки на изображении, текст или bounding box.

[Презентация](https://docs.google.com/presentation/d/19tKhAVJubedsiTjoJkLjjsT0YnzyOnq_QRJGz64w4Rw/edit#slide=id.g3355bc1050f_0_175)

Материалы: [SAM.ipynb](SAM.ipynb), [presentation](https://docs.google.com/presentation/d/1Z6K-GqKVCSfhOVYasbRySO-2QPcQkrvpqsrNtq01yeY/edit?slide=id.g3355bc1050f_0_175#slide=id.g3355bc1050f_0_175)