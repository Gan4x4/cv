# Нейронные сети



## Темы

- [Функции потерь](loss_function.ipynb)
- [Нормализация входных данных](normalization.ipynb)
- [Многослойные нейронные сети](Multilayer_perceptron.ipynb)
- [Функции активации](activation_functions.ipynb)
- [Введение в PyTorch.](pytorch_example.ipynb)
- [Backpropagation](Backpropagation.ipynb)
- [Оптимизаторы](Optimizers.ipynb)
- [Schedulers и усреднение весов](learning_techniques.ipynb)
- [Dropout](Dropout.ipynb)
- [Нормализация активаций](normalization.ipynb)
- [Lightning](Lightning.ipynb)
- [Tensorboard](tensorboard.ipynb)
- [Batch Normalization](Batchnorm.ipynb)


## Многослойные нейронные сети

Multi-layer perceptron(MLP) или Многослойная нейронная сеть

Многослойный персептрон это модель, где данные проходят через несколько линейных слоев с функцией активации. Входные данные обрабатываются последовательно, и каждый слой улучшает признаковое описание данных.


Мотивация:

    Линейные модели эффективны только при работе с линейно разделимыми данными.

    Для работы с произвольными данными требуется их предобработка и нелинейные модели(Feature engineering).

Объединив несколько линейных слоев при помощи нелинейной функции мы получаем возможность работать с любыми данными и избавляемся от необходимости ручной подготовки признаков.


![MLP](https://ml.gan4x4.ru/msu/dev-2.0/L05/out/nn_fully_connected.png)

Материалы: [Multilayer_perceptron.ipynb](Multilayer_perceptron.ipynb), 
[presentation](https://docs.google.com/presentation/d/1MCgXRalQYN4XMinNhaM49ZpEc39TnBxIwYJjiL2F5kU/edit?slide=id.p#slide=id.p)

Дополнительные материалы: 

[HTML c теорией](https://education.yandex.ru/handbook/ml/article/pervoe-znakomstvo-s-polnosvyaznymi-nejrosetyami)

Пример [реализации и обучения](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb) с использованиемфреймворка Pytorch и датасета MNIST

[Запись лекции](https://youtu.be/uJf-aWiXR-s)


### Функции активации

Функции активации вводят нелинейность в нейронные сети, что позволяет аппроксимировать сложные функции.

Требования к функциям активации:

**Нелинейность**: Функции активации добавляют нелинейность, необходимую для аппроксимации сложных функций, чего нельзя достичь простой линейной моделью. Без нелинейностей нейронные сети действуют как линейные модели.

**Дифференцируемость**: Функции активации должны быть дифференцируемыми, чтобы применять градиентные методы оптимизации.

![Basic activation functions](https://ml.gan4x4.ru/msu/dev-2.1/L05/popular_activation_functions.png)


Материалы: [activation_functions.ipynb](activation_functions.ipynb), [presentation](https://docs.google.com/presentation/d/1MCgXRalQYN4XMinNhaM49ZpEc39TnBxIwYJjiL2F5kU/edit?slide=id.g2860616d7ba_0_318#slide=id.g2860616d7ba_0_318)



## Обучение


### Функции потерь

Функция потерь измеряет ошибку модели, оценивая расхождение между предсказанными результатами и истинными значениями. Она принимает два аргумента:

Вектор истинных значений.

Вектор предсказанных значений.

Для успешного обучения с использованием градиентного спуска функция потерь должна быть дифференцируемой и ограниченной снизу.

Материалы: [loss_function.ipynb](loss_function.ipynb)

###  Backpropagation

Алгоритм обратного распространения ошибки(backpropagation) это обобщение метода градиентного спуска для обучения моделей с произвольным количеством слоев.

Так как градиент функции потерь зависит от всей модели. Ручной рассчет для больших моделей затруднителен. Алгоритм обратного распространения позволяет сделать это для модели любой сложности используя правило вычисления производной сложной функции([chain rule](https://en.wikipedia.org/wiki/Chain_rule)).

![Calculation graph](https://ml.gan4x4.ru/msu/dev-2.0/L05/out/graph_of_calculation_gradient.png)


Модель представляется в виде графа, каждый узел кторого это простая функция от которой несложно посчитать производную. Автоматический рассчет градиентов при помощи этого метода используется во фреймворках [Pytorch](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) и [Tensorflow](https://www.tensorflow.org/guide/autodiff).


Материалы: [Backpropagation.ipynb](Backpropagation.ipynb), 
[Презентация с описанием алгоритма обратного распространения](https://docs.google.com/presentation/d/1MCgXRalQYN4XMinNhaM49ZpEc39TnBxIwYJjiL2F5kU/edit?slide=id.gfa10e56b14_0_2#slide=id.gfa10e56b14_0_2)
[Функции потель](loss_function.ipynb)

Дополнительные материалы
[Блокнот с примерами использования на Pytorch](https://drive.google.com/file/d/1FIzS0gSlKag4u-lhRTG9F1DXgg7H68nx/view?usp=sharing)

[CS231 backprop explanation](https://cs231n.github.io/optimization-2/)

[Лекция (Michigan Justin Johnson)](https://youtu.be/YnQJTfbwBM8?si=U-0Ou-a_2G6U9PuC)

[Семинар backpropagation(видео)](https://youtu.be/HjhwvQ2t4xM)



### Оптимизаторы
Алгоритмы для обновления весов модели

Backprop позволяет получить градиент лосс для произвольной модели. Обучение по прежнему происходит при помощи [градиентного спуска](https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D0%B9_%D1%81%D0%BF%D1%83%D1%81%D0%BA).

Постепенно изменяем веса модели так что бы ошибка(функция потерь) уменьшалась:

w = w -lr*grad

Шаг обучения задается вручную.

![Optimization ](https://ml.gan4x4.ru/msu/dev-2.2/L07/out/stochastic_gradient_descent.gif)

Придуман ряд алгоритмов позволяющих сделать процесс обновления весов более эффективным:

Автоматически настраивать шаг обучения

Обновлять разные веса с разной скоростью

Не останавливаться в седловых точках(см. анимацию выше) и локальных минимумах

Они называются оптимизаторами(Optimizers).

![Optimizers](https://ml.gan4x4.ru/msu/dep-2.2/L07/methods_without_adaptive_learning_rate.gif)


Существует множество оптимизаторов, которые можно применять для поиска минимума функционала ошибки ([неполный список](https://paperswithcode.com/methods/category/stochastic-optimization)). Эти алгоритмы реализованы в модуле [torch.optim](https://pytorch.org/docs/stable/optim.html)


Материалы: [Optimizers.ipynb](Optimizers.ipynb), [Презентация "Оптимизаторы" (теория)](https://docs.google.com/presentation/d/1MCgXRalQYN4XMinNhaM49ZpEc39TnBxIwYJjiL2F5kU/edit?slide=id.g2860616d7ba_0_45#slide=id.g2860616d7ba_0_45),

[Семинар\[видео\]](https://youtu.be/ErOVFZx5xX0)

[Улучшение сходимости](learning_techniques.ipynb)

## Schedulers и усреднение весов

Адаптивные оптимизаторы такие как Adam хорошо рабатывают локально но не аптируются к глобальным стадиям обучения. Модель сначала должна попасть в облась глобального минимума лосс функции(exploration) а уже затем сойтись к одному из локальных(convergence).

![Exploration vs convergence](https://ml.gan4x4.ru/wb/crm/optimization/adam_weakness.png)

Поэтому используются планировщики(schedulers) и методы усреднения весов.

аптивные оптимизаторы такие как Adam хорошо рабатывают локально но не аптируются к глобальным стадиям обучения. Модель сначала должна попасть в облась глобального минимума лосс функции(exploration) а уже затем сойтись к одному из локальных(convergence).

![adam\_weakness.png](https://ml.gan4x4.ru/wb/crm/optimization/adam_weakness.png)

Поэтому используются планировщики(schedulers) и методы усреднения весов.

![schedulers.png](https://ml.gan4x4.ru/wb/crm/optimization/schedulers.png)

Проблемой планировщиков является необходимость подбора гиперпараметра - сколько шагов мы хотим сделать прежде чем изменить LR? Алгорити из статьи [Roadless scheduled](https://arxiv.org/abs/2405.15682) решает эту проблему за счет усреднения весов.


Материалы: 


[Алгоритмы оптимизации использующие усреднение весов](https://docs.google.com/presentation/d/1Gta36FIzbtgKYDxd0d6bG4bwdhbz9G0dOtBK4vC3ijw/edit?usp=sharing)

[learning_techniques.ipynb](learning_techniques.ipynb), 

## Регуляризация

### Штраф на веса

Регуляризация L1, L2

Cложная модель вместо закономерностей которые содержуться в данных может запомнить сами данные.

В результате модель будет отлично работеть на train выборке и очень плохо на val и test то есть переобучится.


![Overfitting](https://ml.gan4x4.ru/msu/dev-2.0/L02/out/l2_regularization.png)


Проше всего продемонстрировать переобучение на примере [полиномиальной модели](../Classic_ML/Polinomial_model.ipynb).

Техники для борьбы с переобучением называются [регуляризацией](../Classic_ML/regularization.ipynb). Одна из них это штраф на веса. Во большинстве оптимизаторов есть параметр который позволяет задать этот штраф.

### Dropout

Dropout — это слой в нейронной сети, используемый для предотвращения переобучения. Слой отключает  определенный процент нейронов (зануляет активации) на каждой итерации обучения. 

![Dropout](https://ml.gan4x4.ru/msu/dev-2.2/L07/out/dropout.png)

Материалы: [Dropout.ipynb](Dropout.ipynb)


## Нормализация

### Нормализация входных данных

Нормализация позволяет искать минимум целевой функции удобнее и быстрее:

Материалы: [normalization.ipynb](normalization.ipynb), [standartizaciya-dannyh](https://practicum.yandex.ru/blog/standartizaciya-dannyh/), [torch.nn.functional.normalize](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)

### Нормализация активаций


По [ряду причин](https://docs.google.com/presentation/d/1Gzjh0ywBrSGgBHF5D0VgYiI4RhuQCYUku4KmJR3oU9I/edit#slide=id.gcb79a262e3_0_36) (улучшение сходимости, стабильность и.т.д.) данные перед отправкой в модель нормализуют или [стандартизуют](https://wiki.loginom.ru/articles/data-standartization.html) (делают так что бы среднее было равно 0 а дисперсия 1).

Если модель состоит из нескольких линейных слоёв то нет гарантии что вход i-го слоя будет стандартизованным. Если сеть глубокая то к вышеописанным проблемам добавиться [затухание/взрыв градиента](https://docs.google.com/presentation/d/1Gzjh0ywBrSGgBHF5D0VgYiI4RhuQCYUku4KmJR3oU9I/edit#slide=id.g6afbbb2560_0_8)

![Vanishing gradient](https://ml.gan4x4.ru/wb/cv/images/grad_vanish.png)

Что бы не допустить затухание/взрыв градиента в сеть добавляют слои нормализации.

Так как на каждой итерации карты активации меняются, то слои нормализации выучивают параметры (среднее и стандартное отклонение) для нормализации активаций определенного слоя.


![Normalization types](https://ml.gan4x4.ru/wb/cv/images/normalization_types.png)

Они отличаются тем по какой части данных считаются статистики(mean, std).

Нормализация позволяет искать минимум целевой функции удобнее и быстрее:

Материалы: [normalization.ipynb](normalization.ipynb)

 [presentation](https://docs.google.com/presentation/d/1z9mFJ80hdQmRvhIZqQYKUKUt00bGKmrgocVNXe6C0tQ/edit?slide=id.ga128cc3c5d_0_114#slide=id.ga128cc3c5d_0_114)

#### Batch Normalization

В сверточных сетях наиболее эффективным показал себя слой BatchNorm:

Материалы: [Batchnorm.ipynb](Batchnorm.ipynb)

[Семинар(видео)](https://www.youtube.com/watch?v=FmQlk4NWSiY&t=1496s)


## Введение в PyTorch.

Набор инструментов для обработки и загрузки данных в модель с помощью фреймворка Pytorch.

![Torch training pipeline](https://ml.gan4x4.ru/msu/dev-2.2/L05/out/dataset_dataloader.png)


Материалы: [pytorch_example.ipynb](pytorch_example.ipynb), [data_tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)


