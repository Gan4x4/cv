# Программа курса

## Classic_ML

### Dimensionality_reduction.ipynb

- **Задача понижения размерности**
  - Manifold assumption
  - PCA (Метод главных компонент)
  - Kernel PCA (нелинейный) метод главных компонент
  - t-SNE (t-distributed Stochastic Neighbor Embedding)
  - UMAP

### KNN.ipynb

- **Описание модели k-NN**
  - Близость данных согласно метрике
  - Задача классификации
  - Задача регрессии

### SVM.ipynb

- **Метод опорных векторов (SVM)**
  - Многомерная классификация
- **Обобщенные линейные модели**
  - Полиномиальная модель
  - Kernel SVM

### classification_metrics.ipynb

- **Метрики классификации**
  - Accuracy
  - Confusion matrix
  - Balanced accuracy
  - Precision, Recall
  - F-мера
  - AUC-ROC
  - PR-кривая
  - Multiclass accuracy
  - Multilabel

### clustering.ipynb

- **Кластеризация**
  - Алгоритм K-Means
  - Алгоритм DBSCAN

### gradient_descent.ipynb

- **Метод градиентного спуска**
  - Градиент
  - Идея градиентного спуска
  - Выбор скорости обучения
  - Единый подход к учету смещения
  - Необходимость нормализации
  - Cтохастический градиентный спуск

### hyperparameters_optimization.ipynb

- **GridSearch**
- **RandomizedSearch**
- **Optuna**

### linear_classifier.ipynb

- **Линейная классификация**
  - Постановка задачи
  - Переход к вероятностям
  - Многоклассовая классификация
  - Cross-Entropy loss

### linear_regression.ipynb

- **Линейная регрессия**
  - Модель и ее параметры
  - Функция потерь
  - Поиск локального минимума
  - Метод наименьших квадратов
  - Метрики регрессии
  - Модель линейной регрессии из библиотеки scikit-learn

### regression_metrics.ipynb

- **Метрики регрессии**
  - MAE (mean absolute error)
  - MSE (mean squared error)
  - RMSE (root mean squared error)
  - R²
  - MSLE (mean squared logarithmic error)

### regularization.ipynb

- **Проблема корреляции признаков**
  - Регуляризация

### train_test_split_kfold.ipynb

- **Параметры и гиперпараметры модели**
- **Разделение train-validation-test**
  - Стратификация
- **Кросс-валидация**

## Decision_trees

### Bias, Variance, Irreducible error.ipynb

- **Bias, Variance, Irreducible error**
  - Bias
  - Variance
  - Irreducible error
  - Bias vs variance

### Blending_stacking.ipynb

- **Блендинг и Стэкинг**
  - Blending (Блендинг)
  - Стэкинг

### Bootstrap.ipynb

- **Бутстрэп**
  - Построение доверительного интервала для качества метрики

### Decision_tree.ipynb

- **Деревья решений**
  - Принцип работы дерева решений
  - Классификация
  - Регрессия
  - Свойства деревьев решений

### Ensembles.ipynb

- **Ансамбли**
  - Bagging = **B**ootstrap **agg**regat**ing**
  - Метод случайных подпространств (RSM, random subspace method)
  - Комбинация RSM и Bagging

### Imbalanced_dataset.ipynb

- **Проблемы при работе с реальными данными**
  - Дисбаланс классов
- **Обнаружение аномалий**
  - Оценка качества в задаче обнаружения аномалий

### Random_forest.ipynb

- **Случайный лес**
  - Зависимость качества случайного леса от числа деревьев
  - Зависимость качества случайного леса от глубины дерева
  - Минимальное число объектов в листе
  - Про другие реализации случайного леса
  - Переобучение случайного леса
  - Валидация случайного леса на Out-Of-Bag (OOB) объектах

### gradient_boosting.ipynb

- **Boosting**
  - Gradient boosting (градиентный бустинг)
  - Модификации градиентного бустинга

## Neural_network

### Backpropagation.ipynb

- **Обучение нейронной сети**
  - Прямое и обратное распространение
  - Метод обратного распространения ошибки

### Batchnorm.ipynb

- **Трудности при обучении глубоких нейронных сетей**
- **Затухание градиента**
- **Нормализация входов и выходов**
  - Нормализация входных данных
  - Нормализация целевых значений в задаче регрессии
- **Инициализация весов**
  - Инициализация Ксавье (Xavier Glorot)
  - Инициализация Каймин Хе (Kaiming He)
  - Важность инициализации весов
  - Инициализация весов в PyTorch
- **Слои нормализации**
  - Internal covariate shift
  - Batch Normalization
- **Pseudocode for training a model in pure pytorch**

### Dropout.ipynb

- **Вспомогательный код**
- **Регуляризация**
  - L1, L2 регуляризации
  - Dropout
  - DropConnect
  - DropBlock
  - Советы по использованию Dropout

### Lightning.ipynb

- **Lightning**
  - Мотивация множить сущности
  - Pipeline обучения в чистом PyTorch
  - Основы работы с Lightning
- **define optimizer -> configure_optimizers**
- **Loss function -> __init__**
- **number epoch -> L.trainer(max_epochs=...)**

### Multilayer_perceptron.ipynb

- **Ограничения линейных моделей**
  - Проблемы классификации более сложных объектов
- **Многослойные нейронные сети**
  - Веса и смещения
  - Нейронная сеть как универсальный аппроксиматор

### Optimizers.ipynb

- **Вспомогательный код**
- **Оптимизация параметров нейросетей**
  - Обзор популярных оптимизаторов
  - Использование оптимизаторов

### activation_functions.ipynb

- **Функции активации**
  - Требования к функциям активации
  - Различные функции активации
  - Логистическая функция
  - Гиперболический тангенс
  - ReLU
  - Leaky ReLU
  - GELU (Gaussian Error Linear Unit)
  - Визуализация функций активации

### learning_techniques.ipynb

- **Вспомогательный код**
- **Режимы обучения**
  - Ранняя остановка
  - Уменьшение скорости обучения на плато
  - Понижение скорости обучения на каждой эпохе
  - Neural Network WarmUp
  - Cyclical learning schedule
  - Ландшафт функции потерь
  - Model soup
  - Взаимодействие learning schedule и адаптивного изменения learning rate

### loss_function.ipynb

- **Функции потерь (loss functions)**
  - Mean Squared Error
  - Mean Absolute Error
  - Huber Loss
  - Cross-Entropy
  - Focal Loss
  - Negative Log Likelihood
  - Binary Cross-Entropy
  - Binary Cross-Entropy With Logits

### normalization.ipynb

- **Трудности при обучении глубоких нейронных сетей**
- **Затухание градиента**
- **Нормализация входов и выходов**
  - Нормализация входных данных
  - Нормализация целевых значений в задаче регрессии
- **Инициализация весов**
  - Инициализация Ксавье (Xavier Glorot)
  - Инициализация Каймин Хе (Kaiming He)
  - Важность инициализации весов
  - Инициализация весов в PyTorch

### pytorch_example.ipynb

- **Углубление в PyTorch. Пример нейронной сети на MNIST**
  - Dataset и DataLoader
  - Трансформации (Transforms)
  - Создание нейронной сети
  - Обучение нейронной сети

### tensorboard.ipynb

- **Знакомство с TensorBoard**
- **Логирование результатов и Summary**
- **Обучение полносвязной нейронной сети**
  - Подготовка данных
  - SummaryWriter
  - Запуск TensorBoard в Google Colab
  - Логирование изображений датасета
  - Логирование структуры модели
  - Проектор и отображение данных в пространстве меньшей размерности
  - Обучение модели и логирование loss и accuracy по эпохам
  - Визуализация результатов
  - PR-кривая
- **Анализ логов**

## Convolutional_neural_network

### Adversarial_attack.ipynb


### Architectures_cnn.ipynb

- **ImageNet**
  - Метрики ImageNet
- **Baseline (AlexNet 2012)**
  - Тюнинг гиперпараметров (ZFnet)
- **Базовый блок (VGGNet 2014)**
  - Вычислительные ресурсы
- **Inception module (GoogLeNet 2014)**
  - Stem network
  - Global Average Pooling
  - Затухание градиента
- **BatchNorm (революция глубины)**
- **Skip connection (ResNet 2015)**
  - Архитектура ResNet
  - BasicBlock в PyTorch
  - Bottleneck layer
  - Stage ratio
  - Обучение ResNet
- **Grouped Convolution**
  - Grouped Convolution in PyTorch
  - ResNeXt
- **Сравнение моделей**
  - Много skip connection (DenseNet 2016)
  - Ширина вместо глубины (WideResNet 2016)
- **Squeeze-and-Excitation (SENet 2017)**
- **Поиск хорошей архитектуры**
  - Обзор сети EfficientNet (2019 г.)
- **Трансформеры**
- **ConvNext (2022)**

### Augmentation.ipynb

- **Аугментация**
  - Random Rotation
  - Gaussian Blur
  - Random Erasing
  - ColorJitter
  - Совмещаем несколько аугментаций вместе
  - Совмещение нескольких аугментаций случайным образом
  - Пример создания собственной аугментации
  - Аугментация внутри `Dataset`
  - Аугментация в реальных задачах

### Convolution_1x1.ipynb

- **Свёртка фильтром $1\times1$**

### Convolution_layer.ipynb

- **Сверточный слой нейросети**
  - Обработка цветных/многоканальных изображений
  - Использование нескольких фильтров
  - Уменьшение размера карты признаков
  - Расширение (padding)
- **Batchnorm 2d**
  - Другие Normalization
- **Dropout 2d**

### Convolution_with_filter.ipynb

- **Скользящее окно (фильтр)**
- **Свертка с фильтром**
  - Дополнительная информация

### Covolution 1D, 3D.ipynb

- **Свертки 1D**
- **Свертки 3D**

### GradCam.ipynb

- **Load the data**
  - Apply Gradcam to Resnet18
  - Apply Gradcam to Densenet21
  - Densenet with modified classifier

### Receptive_field.ipynb

- **Полносвязная нейронная сеть**
  - Нарушение связей между соседними пикселями

### Transfer_learning.ipynb

- **Feature extractor**
- **Transfer learning**
  - Шаг 1. Получение предварительно обученной модели
  - Шаг 2. Заморозка предобученных слоев
  - Шаг 3. Добавление новых обучаемых слоев
  - Шаг 4. Обучение новых слоев
  - Шаг 5. Тонкая настройка модели (fine-tuning)

### Visualization_of_weights_and_feature_maps.ipynb

- **Визуализация весов**
  - Визуализация фильтров промежуточных слоев
- **Визуализация карт признаков**

### mlp_vs_conv.ipynb

- **Сравнение свёрточного и полносвязного слоев**
  - Сколько обучаемых праметров (весов) у свёрточного слоя?
  - Сколько обучаемых праметров у полносвязного слоя?
  - Сколько вычислительных ресурсов требуется полносвязному слою?
  - Сколько вычислительных ресурсов требуется свёрточному слою?

### stride_pooling.ipynb

- **Применение свёрточных слоёв**

### timm.ipynb

- **Torch Image Models (timm)**
  - Custom feature extractor

## Transformers

### DEIT.ipynb

  - Обучение ViT
  - DeiT: Data-efficient Image Transformers

### ViT.ipynb

- **Self Attention (ViT 2020)**
  - Сравнение со сверткой
- **Архитектура ViT**
  - Предсказание с помощью ViT

## Representation_learning

### Autoencoders.ipynb

- **Автоэнкодеры (AE)**
  - Архитектура автоэнкодера
  - Функции потерь в автоэнкодерах
  - Очищение изображения от шумов
  - Реализация автоэнкодера
  - Обнаружение аномалий
  - Предобучение на неразмеченных данных
  - Автоэнкодер как генератор и его ограничения
- **Вариационные автоэнкодеры (VAE)**
  - Семплирование в латентном пространстве
  - Регуляризация латентного пространства
  - Реализация VAE
  - Плавная интерполяция
  - Векторная арифметика
  - Ограничения VAE
- **Условные вариационные автоэнкодеры (CVAE)**
  - Реализация CVAE

### Metric_learning.ipynb

- **Metric learning**
  - Формирование векторов признаков
  - Сиамская сеть
  - Реализация сиамской сети

### Representation_learning.ipynb

- **Глубокие нейронные сети как модели обучения представлений**
  - Понижение размерности и гипотеза о многообразии

## Segmentation

### Albumentations.ipynb

- **Albumentations**
  - Особенности применения аугментаций при задаче сегментации
  - Пример использования Albumentations

### COCO_dataset.ipynb

  - Dataset COCO — Common Objects in Context

### Panoptic_segmentation.ipynb

- **Panoptic Segmentation**

### SMP.ipynb

- **Segmentation models PyTorch (SMP)**

### Semantic_segmentation.ipynb

- **Семантическая сегментация (Semantic segmentation)**
- **Семантическая сегментация (Semantic segmentation)**
  - Способы предсказания класса для каждого пикселя
  - Fully Convolutional Networks
  - Разжимающий слой
  - Пирамида признаков
  - Loss функции для сегментации
  - U-Net: Convolutional Networks for Biomedical Image Segmentation
  - Обзор DeepLabv3+ (2018)

## Detection

### OWL_ViT.ipynb


### RCNN.ipynb

- **Эвристика для поиска ROI**
  - Selective search
- **R-CNN (Region CNN)**
- **NMS**
- **Fast R-CNN**
- **ROI Pooling**
- **ROI Align**
- **Faster R-CNN**
  - Region proposal network (RPN)
  - Two stage detector
- **One Stage detector**
  - SSD: Single Shot MultiBox Detector

### RetinaNet.ipynb

- **Loss для детектора**
- **FocalLoss**
- **Нard Example Mining**
  - Online hard example mining
- **Feature pyramid network**

### SAM.ipynb


### Why_we_did't_predict_absolute_values_.ipynb


### YOLO.ipynb

- **YOLO**
  - YOLOv3
  - YOLOv4
  - YOLOv5
  - <font color="orange">YOLOX</font>
  - YOLOv6
  - YOLOv8

### mAP.ipynb

  - mAP — mean Average Precision

## Generative_models

### VAE.ipynb

- **Вариационные автоэнкодеры (VAE)**
  - Семплирование в латентном пространстве
  - Регуляризация латентного пространства
  - Реализация VAE
  - Плавная интерполяция
  - Векторная арифметика
  - Ограничения VAE
- **Условные вариационные автоэнкодеры (CVAE)**
  - Реализация CVAE

