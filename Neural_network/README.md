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

## Функции потерь

Функция потерь измеряет ошибку модели, оценивая расхождение между предсказанными результатами и истинными значениями. Она принимает два аргумента:

Вектор истинных значений.

Вектор предсказанных значений.

Для успешного обучения с использованием градиентного спуска функция потерь должна быть дифференцируемой и ограниченной снизу.

Материалы: [loss_function.ipynb](loss_function.ipynb), [Loss_function](https://en.wikipedia.org/wiki/Loss_function), [@mlblogging](https://medium.com/@mlblogging.k/14-loss-functions-you-can-use-for-regression-b24db8dff987), [Mean_absolute_error](https://en.wikipedia.org/wiki/Mean_absolute_error), [Mean_squared_error](https://en.wikipedia.org/wiki/Mean_squared_error), [torch.nn.CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [Focal-loss-PyTorch](https://github.com/itakurah/Focal-loss-PyTorch), [torch.nn.BCELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

## Нормализация входных данных

Нормализация позволяет искать минимум целевой функции удобнее и быстрее:

Материалы: [normalization.ipynb](normalization.ipynb), [standartizaciya-dannyh](https://practicum.yandex.ru/blog/standartizaciya-dannyh/), [torch.nn.functional.normalize](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)

## Многослойные нейронные сети

Объединив несколько линейных слоев при помощи нелинейной функции мы получаем возможность [аппроксимировать любую функцию](https://ru.wikipedia.org/wiki/%D0%A2%D0%B5%D0%BE%D1%80%D0%B5%D0%BC%D0%B0_%D0%A6%D1%8B%D0%B1%D0%B5%D0%BD%D0%BA%D0%BE) и избавляемся от необходимости ручной подготовки признаков.

Материалы: [Multilayer_perceptron.ipynb](Multilayer_perceptron.ipynb), [presentation](https://docs.google.com/presentation/d/1MCgXRalQYN4XMinNhaM49ZpEc39TnBxIwYJjiL2F5kU/edit?slide=id.p#slide=id.p), [colab.research.google.com](https://colab.research.google.com/drive/1jP_wPi_zaijPvNQvZMq0Tt2YnHOtIF5-#scrollTo=pPP4g1gUUkk9), [colab.research.google.com](https://colab.research.google.com/drive/1jP_wPi_zaijPvNQvZMq0Tt2YnHOtIF5-#scrollTo=z4rKAfjWUklP)

## Функции активации

Требования к функциям активации:

**Нелинейность**: Функции активации добавляют нелинейность, необходимую для аппроксимации сложных функций, чего нельзя достичь простой линейной моделью. Без нелинейностей нейронные сети действуют как линейные модели.

**Дифференцируемость**: Функции активации должны быть дифференцируемыми, чтобы применять градиентные методы оптимизации.

Материалы: [activation_functions.ipynb](activation_functions.ipynb), [colab.research.google.com](https://colab.research.google.com/drive/1jP_wPi_zaijPvNQvZMq0Tt2YnHOtIF5-#scrollTo=6-0wqBsBUklq), [presentation](https://docs.google.com/presentation/d/1MCgXRalQYN4XMinNhaM49ZpEc39TnBxIwYJjiL2F5kU/edit?slide=id.g2860616d7ba_0_318#slide=id.g2860616d7ba_0_318)

## Введение в PyTorch.

Материалы: [pytorch_example.ipynb](pytorch_example.ipynb), [data_tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html), [colab.research.google.com](https://colab.research.google.com/drive/1jP_wPi_zaijPvNQvZMq0Tt2YnHOtIF5-#scrollTo=qynj642QUklv), [colab.research.google.com](https://colab.research.google.com/drive/1jP_wPi_zaijPvNQvZMq0Tt2YnHOtIF5-#scrollTo=d0zpXE7vUkly), [transforms_tutorial](https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html), [colab.research.google.com](https://colab.research.google.com/drive/1xbYegrbSVcOWiWpxMc1w4EZHAgVooFOG), [colab.research.google.com](https://colab.research.google.com/drive/1ZDTuPodj2NMB2RwLs89GW8pEkcMSgq6_#scrollTo=NGtAcQt3LGBC), [colab.research.google.com](https://colab.research.google.com/drive/1jP_wPi_zaijPvNQvZMq0Tt2YnHOtIF5-#scrollTo=xhLEmVoCUklo), [colab.research.google.com](https://colab.research.google.com/drive/1jP_wPi_zaijPvNQvZMq0Tt2YnHOtIF5-#scrollTo=NlRxeN36Ukll), [colab.research.google.com](https://colab.research.google.com/drive/1rM7zRySu8WulXbFiXzxBGVzILxvQ6K4A)

## Backpropagation

Так как градиент функции потерь зависит от всей модели. Его Аналитический подсчет для больших моделей затруднителен. Алгоритм обратного распространения позволяет сделать это для модели любой сложности используя правило вычисления производной сложной функции([chain rule](https://en.wikipedia.org/wiki/Chain_rule)).

Модель представляется в виде графа, каждый узел кторого это простая функция от которой несложно посчитать производную. Автоматический рассчет градиентов при помощи этого метода используется во фреймворках [Pytorch](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) и [Tensorflow](https://www.tensorflow.org/guide/autodiff).

[Презентация с описанием алгоритма обратного распространения](https://docs.google.com/presentation/d/1A2WY71ypO7QCRQVeK6x5IU5a69C4sFvHNQH-QtteQKw/edit#slide=id.gfa10e56b14_0_2)

[Блокнот с примерами использования на Pytorch](https://drive.google.com/file/d/1FIzS0gSlKag4u-lhRTG9F1DXgg7H68nx/view?usp=sharing)

[CS231 backprop explanation](https://cs231n.github.io/optimization-2/)

[Лекция (Michigan Justin Johnson)](https://youtu.be/YnQJTfbwBM8?si=U-0Ou-a_2G6U9PuC)

[Семинар backpropagation\[видео\]](https://youtu.be/HjhwvQ2t4xM)

Материалы: [Backpropagation.ipynb](Backpropagation.ipynb), [optimization-2](https://cs231n.github.io/optimization-2/), [WZDlNAPIFrL-bQ](https://disk.yandex.ru/d/WZDlNAPIFrL-bQ), [presentation](https://docs.google.com/presentation/d/1MCgXRalQYN4XMinNhaM49ZpEc39TnBxIwYJjiL2F5kU/edit?slide=id.gfa10e56b14_0_2#slide=id.gfa10e56b14_0_2)

## Оптимизаторы

Backprop позволяет получить градиент лосс для произвольной модели. Обучение по прежнему происходит при помощи [градиентного спуска](https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D0%B9_%D1%81%D0%BF%D1%83%D1%81%D0%BA).

Постепенно изменяем веса модели так что бы ошибка(функция потерь) уменьшалась:

w = w -lr*grad

Шаг обучения задается вручную.

Придуман ряд алгоритмов позволяющих сделать процесс обновления весов более эффективным:

Автоматически настраивать шаг обучения

Обновлять разные веса с разной скоростью

Не останавливаться в седловых точках(см. анимацию выше) и локальных минимумах

Они называются оптимизаторами(Optimizers).

Существует множество оптимизаторов, которые можно применять для поиска минимума функционала ошибки ([неполный список](https://paperswithcode.com/methods/category/stochastic-optimization)). Эти алгоритмы реализованы в модуле [torch.optim](https://pytorch.org/docs/stable/optim.html)

[Презентация "Оптимизаторы" (теория)](https://docs.google.com/presentation/d/1A2WY71ypO7QCRQVeK6x5IU5a69C4sFvHNQH-QtteQKw/edit#slide=id.g2860616d7ba_0_45)

[Алгоритмы оптимизации использующие усреднение весов](https://docs.google.com/presentation/d/1Gta36FIzbtgKYDxd0d6bG4bwdhbz9G0dOtBK4vC3ijw/edit?usp=sharing)

Материалы: [Optimizers.ipynb](Optimizers.ipynb), [presentation](https://docs.google.com/presentation/d/1MCgXRalQYN4XMinNhaM49ZpEc39TnBxIwYJjiL2F5kU/edit?slide=id.g2860616d7ba_0_45#slide=id.g2860616d7ba_0_45), [colab.research.google.com](https://colab.research.google.com/drive/1jP_wPi_zaijPvNQvZMq0Tt2YnHOtIF5-#scrollTo=07xMLT-NUkl5), [SaddlePoint](https://mathworld.wolfram.com/SaddlePoint.html), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=a49Fja0RSHUC), [optimization](https://www.deeplearningbook.org/contents/optimization.html), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=SSCrNkMGSHUD), [presentation](https://docs.google.com/presentation/d/1EYvgT3ZOUldAH1G4PBnPN0H5M9gaP9-8FyOLJs7Duhw/edit?slide=id.g39b5baca540_0_0#slide=id.g39b5baca540_0_0), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=f2QoyqmFSHUF), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=nOWaIkBaSHUF), [why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for), [presentation](https://docs.google.com/presentation/d/1A2WY71ypO7QCRQVeK6x5IU5a69C4sFvHNQH-QtteQKw/edit?slide=id.g35da52bd86f_0_0#slide=id.g35da52bd86f_0_0), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=gLOXOyPbSHUG), [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

## Schedulers и усреднение весов

Например, для достаточно больших нейронных сетей практикуют следующую схему:

Поставить изначальный learning rate значительно ниже того, с которого мы обычно начинаем обучение.

За несколько эпох, например, 5, довести learning rate от этого значения до требуемого. Так мы не совершаем больших шагов, когда сеть еще ничего не знает о данных. За счет этого нейросеть лучше "адаптируется" к нашим данным.

Также такой learning schedule позволяет адаптивным оптимизаторам лучше оценить значения learning rate для разных параметров:

[Семинар\[видео\]](https://youtu.be/ErOVFZx5xX0)

Материалы: [learning_techniques.ipynb](learning_techniques.ipynb), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=BuNe1R-_SHUI), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=wrtSg-w-SHUM), [paper](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2203.05482.pdf)

## Dropout

Материалы: [Dropout.ipynb](Dropout.ipynb), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=m3_mNXVDSHT5)

## Нормализация активаций

Нормализация позволяет искать минимум целевой функции удобнее и быстрее:

Материалы: [normalization.ipynb](normalization.ipynb), [Batchnorm.ipynb](Batchnorm.ipynb), [cs231n.github.io](https://cs231n.github.io/neural-networks-2/#batchnorm), [presentation](https://docs.google.com/presentation/d/1z9mFJ80hdQmRvhIZqQYKUKUt00bGKmrgocVNXe6C0tQ/edit?slide=id.ga128cc3c5d_0_114#slide=id.ga128cc3c5d_0_114), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=FumMqh2USHTq), [presentation](https://docs.google.com/presentation/d/1Gzjh0ywBrSGgBHF5D0VgYiI4RhuQCYUku4KmJR3oU9I/edit#slide=id.ga128cc3c5d_0_150), [colab.research.google.com](https://colab.research.google.com/drive/1wdPZuTSqfMjfkwAs7bnu2r4yIj0gU4ao), [colab.research.google.com](https://colab.research.google.com/drive/1otICabYgNg9FvUOXYAoOMr8vW9VX9vNC#scrollTo=EEGcisdRSHT2)

## Lightning

Вот краткий список преимуществ, которые дает lightning:

улучшение читаемости кода: универсальные названия функций (вы всегда знаете, куда смотреть, чтобы найти нужную часть кода);

интеграция с TorchMetrics упрощает валидацию моделей: никаких вручную написанных расчетов accuracy и агрегации результатов — все происходит внутри готовых проверенных классов;

удобная система логирования: при выполнении вашей научной работы вы столкнетесь с необходимостью контроля воспроизводимости экспериментов, и Lightning предоставляет набор готовых решений для этого;

возможность восстановления: возможность легко продолжить обучение с сохраненной точки в случае отключения среды;

простота управления ресурсами: распараллеливание вычислений на несколько устройств или выбор устройства, на котором происходят вычисления, осуществляется одной строкой;

обратная совместимость с PyTorch: при помощи Lightning мы обучаем уже известные нам PyTorch модели, и при желании всегда сможем продолжить с ними работу на "чистом" PyTorch.

Материалы: [Lightning.ipynb](Lightning.ipynb), [github.com](https://github.com/Gan4x4/cv/blob/6bb67b5cede8ea9202bcdedbe6c7696366dfe8e2/Neural_network/Lightning.ipynb), [stable](https://lightning.ai/docs/pytorch/stable/)

## Tensorboard

Для визуализации данных в коде можно использовать библиотеку matplotlib. Однако, если вы проводите реальные эксперименты, вам может понадобиться инструмент для сохранения и сравнения результатов без повторного обучения моделей. Один из способов — логирование результатов экспериментов. Для удобного отображения таких логов есть более мощные инструменты. В этом блоке вы узнаете об одном из них — TensorBoard.

Материалы: [tensorboard.ipynb](tensorboard.ipynb), [github.com](https://github.com/Gan4x4/cv/blob/6bb67b5cede8ea9202bcdedbe6c7696366dfe8e2/Neural_network/tensorboard.ipynb), [tensorboard](https://www.tensorflow.org/tensorboard?hl=ru), [site](https://wandb.ai/site/)

## Batch Normalization

[Семинар(видео)](https://www.youtube.com/watch?v=FmQlk4NWSiY&t=1496s)

Материалы: [Batchnorm.ipynb](Batchnorm.ipynb)
