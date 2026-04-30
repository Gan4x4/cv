# Обучение на реальных данных








## Темы

- [Multilabel классификация](extra/BCEWithLogitsLoss_invalid_targets.ipynb)
- [Schedulers и усреднение весов](Cyclic_learning_rate_schedulers.ipynb)
- [Анализ данных](WeightedRandomSampler.ipynb)


## Инструменты

### Lightning

Вот краткий список преимуществ, которые дает lightning:

улучшение читаемости кода: универсальные названия функций (вы всегда знаете, куда смотреть, чтобы найти нужную часть кода);

интеграция с TorchMetrics упрощает валидацию моделей: никаких вручную написанных расчетов accuracy и агрегации результатов — все происходит внутри готовых проверенных классов;

удобная система логирования: при выполнении вашей научной работы вы столкнетесь с необходимостью контроля воспроизводимости экспериментов, и Lightning предоставляет набор готовых решений для этого;

возможность восстановления: возможность легко продолжить обучение с сохраненной точки в случае отключения среды;

простота управления ресурсами: распараллеливание вычислений на несколько устройств или выбор устройства, на котором происходят вычисления, осуществляется одной строкой;

обратная совместимость с PyTorch: при помощи Lightning мы обучаем уже известные нам PyTorch модели, и при желании всегда сможем продолжить с ними работу на "чистом" PyTorch.

Материалы: [Lightning.ipynb](Lightning.ipynb), [stable](https://lightning.ai/docs/pytorch/stable/)


## Tensorboard

Для визуализации данных в коде можно использовать библиотеку matplotlib. Однако, если вы проводите реальные эксперименты, вам может понадобиться инструмент для сохранения и сравнения результатов без повторного обучения моделей. Один из способов — логирование результатов экспериментов. Для удобного отображения таких логов есть более мощные инструменты. В этом блоке вы узнаете об одном из них — TensorBoard.

Материалы: [tensorboard.ipynb](tensorboard.ipynb), [tensorboard](https://www.tensorflow.org/tensorboard?hl=ru), 

Альтернатива Tensorboard:
[WanDB](https://wandb.ai/site/)


## Multilabel классификация

Материалы: [BCEWithLogitsLoss_invalid_targets.ipynb](extra/BCEWithLogitsLoss_invalid_targets.ipynb), [towardsdatascience.com](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff), [Gjorgjioski_Multilabel](https://aile3.ijs.si/dunja/SiKDD2011/Papers/Gjorgjioski_Multilabel.pdf), [multilabel-classification-metrics-on-scikit](https://stats.stackexchange.com/questions/233275/multilabel-classification-metrics-on-scikit), [sklearn.metrics.hamming_loss](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.hamming_loss.html), [sklearn.preprocessing.MultiLabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html), [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn)

## Schedulers и усреднение весов

Например, для достаточно больших нейронных сетей практикуют следующую схему:

Поставить изначальный learning rate значительно ниже того, с которого мы обычно начинаем обучение.

За несколько эпох, например, 5, довести learning rate от этого значения до требуемого. Так мы не совершаем больших шагов, когда сеть еще ничего не знает о данных. За счет этого нейросеть лучше "адаптируется" к нашим данным.

Также такой learning schedule позволяет адаптивным оптимизаторам лучше оценить значения learning rate для разных параметров:

[Семинар(видео)](https://youtu.be/ErOVFZx5xX0)

Материалы: [Cyclic_learning_rate_schedulers.ipynb](Cyclic_learning_rate_schedulers.ipynb), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=BuNe1R-_SHUI), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=wrtSg-w-SHUM), [paper](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2203.05482.pdf)

## Анализ данных

**Проверка дисбаланса.**

Каждый класс должен быть представлен примерно равным числом изображений. Постройте гистограмму классов. Если дисбалланс значительный, то примените WeightedRandomSampler

**Проверка ошибок разметки**.

Все датасеты, даже такие популярные как ImageNet [содержат ошибки](https://labelerrors.com/) разметки. Новый датасет тем более будет их содержать. Перед началом обучения всегда нужно

Просмотреть на данные глазами, что бы оценить процент ошибок.

Получить эмбеддинги данных, кластеризовать их и найти выбросы, либо объекеты не попавшие в свой кластер. Для этого можно использовать [методы понижения размерности](https://www.kaggle.com/code/samuelcortinhas/intro-to-pca-t-sne-umap).

Если задача уже решалась, либо вы уже обучили свою модель. Можно найти ошибки разметки проанализировав изображения на которых модель(модели)

Можно искать ошибки в данных оценивая уверенность предсказания модели либо ансамбля моделей. Например включив слой Dropout на inference (но отключив BN)

**Утечка данных**

Через дубликаты:

Низкокачественная(грязная разметка) может привести к утечке данных из train в test.

Например данные могут содержать дубликаты: несколько изображений одного и того же объекта. Если не выявить их заранее то при разбиении датасета на train/val/test изображения одного и тогоже объекта могут окзаться и в тренировочной и проверочных подвыборках и оценка качества предсказания будет невалидной. Кроме того не зная количества дубликатов в данных вы не знаете реальный размер вашего датасета.

Для поиска дубликатов можно использовать техники кластеризации и понижения размерности, либо инструменты подобные [Ultralytics dataset explorer](https://docs.ultralytics.com/datasets/explorer/)

Материалы: [WeightedRandomSampler.ipynb](WeightedRandomSampler.ipynb), [Exploratory_Data_Analysis_image_dataset.ipynb](Exploratory_Data_Analysis_image_dataset.ipynb), [inter-annotator-agreement](https://www.innovatiana.com/en/post/inter-annotator-agreement), [Inter-rater_reliability](https://en.wikipedia.org/wiki/Inter-rater_reliability)
