# Классический ML

## Темы

- [Вводная лекция (ML + CV)](intro.ipynb)
- [Метод k ближайших соседей](KNN.ipynb)
- [Разделение на выборки и кросс-валидация](train_test_split_kfold.ipynb)
- [Оптимизация гиперпараметров](hyperparameters_optimization.ipynb)
- [Линейная регрессия](linear_regression.ipynb)
- [Функции потерь](Hinge_loss_vs_accuracy.ipynb)
- [Метрики регрессии](regression_metrics.ipynb)
- [Метод градиентного спуска](gradient_descent.ipynb)
- [Регуляризация](regularization.ipynb)
- [Линейный классификатор](linear_classifier.ipynb)
- [Метрики классификации](classification_metrics.ipynb)
- [Метод опорных векторов(SVM)](SVM.ipynb)
- [Кластеризация](clustering.ipynb)
- [Понижение размерности](Dimensionality_reduction.ipynb)
- [Генерация и отбор признаков](feature_engineering.ipynb)
- [Дисбаланс классов](feature_engineering.ipynb)
- [Анализ данных](clustering.ipynb)

## Вводная лекция (ML + CV)

Машинное обучение — это область искусственного интеллекта, которая занимается разработкой алгоритмов и моделей, позволяющих компьютерам принимать решения на основе данных.

Основная идея состоит в обучении систем выявлять закономерности в больших массивах данных и делать предсказания на основе выявленных закономерностей.

Материалы: [intro.ipynb](intro.ipynb), [presentation](https://docs.google.com/presentation/d/1OGcwPHsp_cKQsZfYx9uIyEJiJczsBAWNnC-d3HzBQbA/edit?slide=id.g37c856f7528_1_155#slide=id.g37c856f7528_1_155), [RGB](https://ru.wikipedia.org/wiki/RGB)

## Метод k ближайших соседей

Материалы: [KNN.ipynb](KNN.ipynb), [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [sklearn.neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html), [np-reshape-np-flatten-np-ravel](https://codingnomads.com/np-reshape-np-flatten-np-ravel#:~:text=Methods%20like%20np.-,flatten()%20%2C%20np.,copy%20for%20memory%20layout%20reasons.), [translations](https://tproger.ru/translations/3-basic-distances-in-data-science), [Chebyshev_distance](https://en.wikipedia.org/wiki/Chebyshev_distance), [neighbors](https://scikit-learn.org/stable/modules/neighbors.html), [faiss](https://github.com/facebookresearch/faiss), [annoy](https://github.com/spotify/annoy)

## Разделение на выборки и кросс-валидация

Кросс-валидация — это метод оценки модели, при котором данные делятся на несколько подмножеств, и модель обучается и тестируется на различных комбинациях этих подмножеств для более надежной оценки её качества

Материалы: [train_test_split_kfold.ipynb](train_test_split_kfold.ipynb), [cross_validation](https://scikit-learn.org/stable/modules/cross_validation.html), [plot_nested_cross_validation_iris](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [nested-cross-validation-for-machine-learning-with-python](https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/)

## Оптимизация гиперпараметров

GridSearch: Метод для поиска наилучших гиперпараметров модели, перебирая все возможные комбинации заданных параметров.

RandomizedSearch: Метод, который случайным образом выбирает комбинации гиперпараметров из заданного пространства, что позволяет быстрее находить оптимальные значения.

Optuna: Инструмент для автоматической оптимизации гиперпараметров с использованием методов байесовской оптимизации, который эффективно исследует пространство параметров, чтобы находить лучшие настройки.

Материалы: [hyperparameters_optimization.ipynb](hyperparameters_optimization.ipynb), [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [sklearn.model_selection.RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), [optuna.org](https://optuna.org/)

## Линейная регрессия

Материалы: [linear_regression.ipynb](linear_regression.ipynb), [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [linear-models](https://education.yandex.ru/handbook/ml/article/linear-models)

## Функции потерь

Функция потерь измеряет ошибку модели, оценивая расхождение между предсказанными результатами и истинными значениями. Она принимает два аргумента:

Вектор истинных значений.

Вектор предсказанных значений.

Для успешного обучения с использованием градиентного спуска функция потерь должна быть дифференцируемой и ограниченной снизу.

Материалы: [Hinge_loss_vs_accuracy.ipynb](Hinge_loss_vs_accuracy.ipynb), [Cross_entropy.ipynb](Cross_entropy.ipynb), [Loss_function](https://en.wikipedia.org/wiki/Loss_function), [@mlblogging](https://medium.com/@mlblogging.k/14-loss-functions-you-can-use-for-regression-b24db8dff987), [Mean_absolute_error](https://en.wikipedia.org/wiki/Mean_absolute_error), [Mean_squared_error](https://en.wikipedia.org/wiki/Mean_squared_error), [torch.nn.CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [Focal-loss-PyTorch](https://github.com/itakurah/Focal-loss-PyTorch), [torch.nn.BCELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

## Метрики регрессии

Материалы: [regression_metrics.ipynb](regression_metrics.ipynb), [metriki-klassifikacii-i-regressii](https://education.yandex.ru/handbook/ml/article/metriki-klassifikacii-i-regressii), [MSLE](https://permetrics.readthedocs.io/en/latest/pages/regression/MSLE.html)

## Метод градиентного спуска

[Семинар (видео)](https://youtu.be/vyipc0YCltA)

Материалы: [gradient_descent.ipynb](gradient_descent.ipynb), [presentation](https://docs.google.com/presentation/d/1GmzoAP1qvmPTULUoUc0M8off1T4vZK_CHoLYmgcu0X8/edit?slide=id.gedc961eaf4_0_879#slide=id.gedc961eaf4_0_879), [optimization-1](https://cs231n.github.io/optimization-1/#gradcompute), [Метод_наименьших_квадратов](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BD%D0%B0%D0%B8%D0%BC%D0%B5%D0%BD%D1%8C%D1%88%D0%B8%D1%85_%D0%BA%D0%B2%D0%B0%D0%B4%D1%80%D0%B0%D1%82%D0%BE%D0%B2), [Переобучение](https://ru.wikipedia.org/wiki/%D0%9F%D0%B5%D1%80%D0%B5%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5), [Стохастический_градиентный_спуск](https://ru.wikipedia.org/wiki/%D0%A1%D1%82%D0%BE%D1%85%D0%B0%D1%81%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B9_%D0%B3%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D0%B9_%D1%81%D0%BF%D1%83%D1%81%D0%BA)

## Регуляризация

L1-регуляризация (Lasso):

Добавляет штраф на сумму абсолютных значений весов

L2-регуляризация (Ridge):

Добавляет штраф на сумму квадратов весов

Практический смысл:

L1 способствует разреженности весов (многие веса становятся нулевыми), что может привести к выбору важнейших признаков.

L2 "наказывает" модель за слишком большие коэффициенты (веса), но не заставляет их стать нулевыми. Это приводит к тому, что модель учитывает все признаки, но с меньшими значениями весов, что помогает избежать переобучения и улучшает обобщающую способность модели.

Материалы: [regularization.ipynb](regularization.ipynb), [Regularization](https://deepmachinelearning.ru/docs/Machine-learning/Base-concepts/Regularization)

## Линейный классификатор

[Лекция\[видео\]](https://www.youtube.com/watch?v=vyipc0YCltA&t=1624s)

Материалы: [linear_classifier.ipynb](linear_classifier.ipynb), [presentation](https://docs.google.com/presentation/d/1GmzoAP1qvmPTULUoUc0M8off1T4vZK_CHoLYmgcu0X8/edit?slide=id.gedc961eaf4_0_1648#slide=id.gedc961eaf4_0_1648), [linear-classify](https://cs231n.github.io/linear-classify/#loss-function)

## Метрики классификации

Материалы: [classification_metrics.ipynb](classification_metrics.ipynb), [colab.research.google.com](https://colab.research.google.com/drive/1PZQZ-c6bLjFXnJES2i_AlnOWJLrR2Quy), [metriki-klassifikacii-i-regressii](https://education.yandex.ru/handbook/ml/article/metriki-klassifikacii-i-regressii), [colab.research.google.com](https://colab.research.google.com/drive/1AM6PIIvcI4m-D0NxIUwgnn55F7E8gkZH#scrollTo=oLz6PabZxfll), [colab.research.google.com](https://colab.research.google.com/drive/1AM6PIIvcI4m-D0NxIUwgnn55F7E8gkZH#scrollTo=FPM0I3d6xflm), [colab.research.google.com](https://colab.research.google.com/drive/1AM6PIIvcI4m-D0NxIUwgnn55F7E8gkZH#scrollTo=qS1Uzm2oxflm), [colab.research.google.com](https://colab.research.google.com/drive/1AM6PIIvcI4m-D0NxIUwgnn55F7E8gkZH#scrollTo=B4XKy_U4xfln), [colab.research.google.com](https://colab.research.google.com/drive/1AM6PIIvcI4m-D0NxIUwgnn55F7E8gkZH#scrollTo=PnIVfL0uxflo), [Harmonic_mean](https://en.wikipedia.org/wiki/Harmonic_mean), [10-klass](https://www.yaklass.ru/p/veroyatnost-i-statistika/10-klass/obobshchenie-i-sistematizatciia-znanii-7394653/opisatelnaia-statistika-7380283/re-7e9bab7e-2770-4e8e-a6b2-3f52710460f2), [colab.research.google.com](https://colab.research.google.com/drive/1AM6PIIvcI4m-D0NxIUwgnn55F7E8gkZH#scrollTo=w2WevAIxxflo), [colab.research.google.com](https://colab.research.google.com/drive/1AM6PIIvcI4m-D0NxIUwgnn55F7E8gkZH#scrollTo=YKmNXr15xflu), [micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin](https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin)

## Метод опорных векторов(SVM)

Материалы: [SVM.ipynb](SVM.ipynb), [SVM_extended.ipynb](SVM_extended.ipynb), [classification](https://cs231n.github.io/classification/#k---nearest-neighbor-classifier), [drive.google.com](https://drive.google.com/file/d/1Vs2B5M_lmhyOEPhROi-s5O9wmy7qUpCf/view?usp=sharing), [colab.research.google.com](https://colab.research.google.com/drive/1QZVjBaadqfyUWF5xpmU97qqI6N6DHGpc#scrollTo=gpPtNNoJR8Zs), [colab.research.google.com](https://colab.research.google.com/drive/1jDmQd35fyGauHbcO6jYyMH3-1pUAhSO_), [colab.research.google.com](https://colab.research.google.com/drive/1QZVjBaadqfyUWF5xpmU97qqI6N6DHGpc#scrollTo=Bv8pMk5DR8Zy)

## Кластеризация

![clustering_task.png](https://ml.gan4x4.ru/msu/dev-2.1/L04/out/clustering_task.png)

Материалы: [clustering.ipynb](clustering.ipynb), [klasterizaciya](https://education.yandex.ru/handbook/ml/article/klasterizaciya), [clustering](https://scikit-learn.org/stable/modules/clustering.html), [clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means), [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [clustering](https://scikit-learn.org/stable/modules/clustering.html#dbscan), [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [sklearn.cluster.AgglomerativeClustering](https://scikit-learn.org/dev/modules/generated/sklearn.cluster.AgglomerativeClustering.html), [python_ml_hierarchial_clustering](https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp), [articles](https://habr.com/ru/companies/yandex/articles/500742/), [rand-index-in-machine-learning](https://www.geeksforgeeks.org/rand-index-in-machine-learning/), [cluster-silhouette-index](https://wiki.loginom.ru/articles/cluster-silhouette-index.html), [colab.research.google.com](https://colab.research.google.com/drive/1wwRze_Qwi68jgg8RXTXKqSvVKwCDiwB6#scrollTo=0N9j3m3hSXrH)

## Понижение размерности

В машинном обучении часто используют предположение о многообразии (manifold assumption). Это предположение о том, что между признаками, описывающими реальные объекты, существуют некоторые нетривиальные связи. Вследствие этого данные заполняют не весь объем пространства признаков, а лежат на некоторой поверхности — на **многообразии** (manifold).

Если предположение верно, то каждый объект может быть достаточно точно описан новыми признаками в пространстве значительно меньшей размерности, чем исходное пространство признаков.

При этом мы будем терять часть информации об объектах. Но при выполнении предположения о многообразии (а оно почти всегда выполняется) и при правильных настройках алгоритма понижения размерности эти потери будут незначительны.

Интуиция: каждое изображение лица размером 300×300 содержит 90,000 пикселей, но на самом деле реальные лица занимают гораздо меньшее пространство. Это связано с тем, что пиксели на изображении связаны между собой и не могут принимать любые значения. Не все комбинации значений создадут узнаваемое лицо, поэтому мы можем использовать методы понижения размерности, чтобы выделить важные характеристики и структуру данных.

Материалы: [Dimensionality_reduction.ipynb](Dimensionality_reduction.ipynb), [intro-to-pca-t-sne-umap](https://www.kaggle.com/code/samuelcortinhas/intro-to-pca-t-sne-umap), [sklearn.manifold.TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [umap-learn](https://pypi.org/project/umap-learn/), [abs](https://arxiv.org/abs/2012.04456)

## Генерация и отбор признаков

![data_preparation.png](https://ml.gan4x4.ru/msu/dev-2.1/L04/out/data_preparation.png)

Материалы: [feature_engineering.ipynb](feature_engineering.ipynb), [feature_generation_rf_gb.ipynb](feature_generation_rf_gb.ipynb), [Polinomial_model.ipynb](Polinomial_model.ipynb), [colab.research.google.com](https://colab.research.google.com/drive/1vy0LIhb-0nLH4mErFmElm7Q615ylpMWB#scrollTo=j58cyuXkEyOQ), [colab.research.google.com](https://colab.research.google.com/drive/1vy0LIhb-0nLH4mErFmElm7Q615ylpMWB#scrollTo=4rtaDwHPEyOW), [colab.research.google.com](https://colab.research.google.com/drive/1vy0LIhb-0nLH4mErFmElm7Q615ylpMWB#scrollTo=ZyVIb43HEyOX), [colab.research.google.com](https://colab.research.google.com/drive/1vy0LIhb-0nLH4mErFmElm7Q615ylpMWB#scrollTo=SqemwDTxEyOY), [colab.research.google.com](https://colab.research.google.com/drive/1vy0LIhb-0nLH4mErFmElm7Q615ylpMWB#scrollTo=W9wkx47GEyOm), [colab.research.google.com](https://colab.research.google.com/drive/1vy0LIhb-0nLH4mErFmElm7Q615ylpMWB#scrollTo=UOLLQfK4EyOm), [permutation_importance](https://scikit-learn.org/stable/modules/permutation_importance.html), [colab.research.google.com](https://colab.research.google.com/drive/1vy0LIhb-0nLH4mErFmElm7Q615ylpMWB#scrollTo=XHbssPgEEyOx)

## Дисбаланс классов

Методы обработки дисбаланса:

Оценка моделей: самый главный момент работы с дисбалансом это выбор метрик, которые лучше подходят для оценки моделей на дисбалансированных данных, такие как F1-score, Balanced accuracy и другие.

Oversampling: Увеличение количества образцов менее представленного класса.

Undersampling: Уменьшение количества образцов более представленного класса.

Синтетические методы: Использование методов, таких как SMOTE, для создания новых образцов.

Использование специальных алгоритмов: Применение алгоритмов, которые учитывают дисбаланс.

Материалы: [feature_engineering.ipynb](feature_engineering.ipynb)

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

Материалы: [clustering.ipynb](clustering.ipynb), [inter-annotator-agreement](https://www.innovatiana.com/en/post/inter-annotator-agreement), [Inter-rater_reliability](https://en.wikipedia.org/wiki/Inter-rater_reliability)
