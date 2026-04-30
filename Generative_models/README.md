# Генеративные модели

## Темы

- [Вариационные автоэнкодеры (VAE)](VAE.ipynb)
- [GAN](generative_models.ipynb)
- [Принципы работы диффузионных моделей](diffusers_models.ipynb)

## Постановка задачи генерации

В компьютере все объекты описываются числами. Допустим, мы генерируем одномерные объекты класса «дерево», которые описываются одним числом — высотой.
Если высота деревьев нормально распределена, мы можем семплировать числа из нормального распределения и считать каждое число новым деревом.

Когда параметров два, появляется зависимость: например, у дубов высота обычно больше, чем у берёз, поэтому высоту и породу нужно семплировать совместно (учитывая их взаимосвязь). Ещё сложнее дело обстоит с фотографиями: каждая из них — точка в многомерном пространстве (где измерения — это пиксели или признаки), а реальные снимки образуют «облако», описываемое сложным(мультимодальным) распределением которое нельзя описать аналитически и использовать генератор случайных чисел что бы семплировать из него.

![Задача генерации](https://ml.gan4x4.ru/wb/crm/generation/generation_task.png)

Поэтому семплируют точку z из простого распределения (например, нормального) , а затем обучают модель-генератор G, которая преобразует z в точку x=G(z) таким образом что каждому z соответствует свой x, а совокупность всех x должна приближать распределение реальных данных. Таким образом, задача сводится к сопоставлению распределений: «простое → данные».

[презентация](https://docs.google.com/presentation/d/1Hmd6dn_LmDEx4yRPZzJfnOI1G-3tESh87cn2Tib-wMM/edit#slide=id.p)

[видео](https://www.youtube.com/watch?v=PP5cI-sIF5o&t=1165s)


## Вариационные автоэнкодеры (VAE)

С помощью автоэнкодера (AE) можно было бы генерировать новые изображения, семплируя точки из латентного пространства и передавая их на вход декодеру. Однако на практике этот подход работает плохо, потому что в латентном пространстве могут возникать пустоты — из-за отсутствия ограничений на способ кодирования данных.

![vae.png](https://ml.gan4x4.ru/wb/crm/generation/vae.png)

Вариационные автоэнкодеры (VAE) решают эту проблему, добавляя в функцию потерь дополнительный компонент, который требует, чтобы распределение точек в латентном пространстве было близко к нормальному. Cэмплированные точки будут лежать в области, где декодер работает корректно.


Материалы: [VAE.ipynb](VAE.ipynb), [presentation](https://docs.google.com/presentation/d/1FdS5eB5QMXBaNR7cK_LhP947n-Xu1uVIGDliN2iqWec/edit?slide=id.g10c4be00ca3_0_106#slide=id.g10c4be00ca3_0_106), [variational-autoencoder-(vae)](https://education.yandex.ru/handbook/ml/article/variational-autoencoder-(vae)), [presentation](https://docs.google.com/presentation/d/1K24n4w18U9DqnzNw-Am41giPMNbFodZzXzyvb3Q037Y/edit?usp=sharing)

## GAN

Генеративно-состязательные сети (GAN), впервые предложенные Ианом Гудфеллоу в 2014 году в статье "Generative Adversarial Networks" (Goodfellow et al., 2014). Основная идея GAN заключается в использовании двух нейронных сетей: генератора и дискриминатора.

Генератор создает данные на основе случайного шума, стремясь обмануть дискриминатор, который, в свою очередь, обучается отличать истинные данные от сгенерированных. В процессе совместного обучения обе сети улучшаются, что позволяет генератору создавать всё более реалистичные данные.

![generative_adversarial_network_scheme.png](https://ml.gan4x4.ru/msu/dev-2.4/L13/out/generative_adversarial_network_scheme.png)

Материалы: [generative_models.ipynb](generative_models.ipynb)
 [presentation](https://docs.google.com/presentation/d/1FdS5eB5QMXBaNR7cK_LhP947n-Xu1uVIGDliN2iqWec/edit?slide=id.g10c4be00ca3_0_235#slide=id.g10c4be00ca3_0_235)
Дополнительные материалы: [problems](https://developers.google.com/machine-learning/gan/problems), [p](https://proglib.io/p/generativno-sostyazatelnaya-neyroset-vasha-pervaya-gan-model-na-pytorch-2020-08-11), [p](https://proglib.io/p/generativnaya-sostyazatelnaya-set-gan-dlya-chaynikov-poshagovoe-rukovodstvo-2021-07-16), [abs](https://arxiv.org/abs/1806.04304), [what-does-fid-do](https://sites.google.com/view/what-does-fid-do/), [frechet_inception_distance](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html), [kailashahirwar.medium.com](https://kailashahirwar.medium.com/a-very-short-introduction-to-inception-score-is-c9b03a7dd788), [inception_score](https://lightning.ai/docs/torchmetrics/stable/image/inception_score.html#inception-score), [html](https://arxiv.org/html/2408.15098v1), [clip_iqa](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html)

## Принципы работы диффузионных моделей

![distributions.png](https://ml.gan4x4.ru/wb/crm/generation/distributions.png)

Так же как и VAE модель учиться сопоставлять простое рапределение данных целевому. Но делается это за несколько шагов.

![diff_dataset.png](https://ml.gan4x4.ru/wb/crm/generation/diff_dataset.png)

Технически модель учиться на зашумленных изображениях. Количество шума который добавляется к изображению зависит от шага и конкретного алгоритма (DDPM, DDIM, FM, RF ...). Задача модели предсказать шум, лосс = MSE

![diff_base_scheme.png](https://ml.gan4x4.ru/wb/crm/generation/diff_base_scheme.png)

После того как модель обучилась из шума можно  генерировать изображения итеративно предсказывая шум для каждого шага и вычитая его.

[Презентация](https://docs.google.com/presentation/d/1tNHAvV2FLz2lI8xYqnPUK1zNOL3Zq6dZ6YUKBq_2zCU/edit?usp=sharing)

Материалы: [diffusers_models.ipynb](diffusers_models.ipynb), [handmade_diffusion.ipynb](handmade_diffusion.ipynb), [From_DDPM_to_ODE_and_Flow_Matching.ipynb](From_DDPM_to_ODE_and_Flow_Matching.ipynb)

### Условная генерация

Наиболее популярным условием для генерации изображений является текстовый prompt (подсказка).

1. Классический способ отправки промпта в модель это превращение текста в эмбеддинг и подача в cross-attention слой.

![text\_prompt.png](https://ml.gan4x4.ru/wb/crm/generation/text_prompt.png)

Важно что loss- функция при этом не изменяется: модель предсказывает только шум. Никакого компонента связанного с качеством следованию подсказке в лосс не добавляется.

![ada\_norm.png](https://ml.gan4x4.ru/wb/crm/generation/ada_norm.png)

2. Другой способ, который так же используется для подачи не только текста но в.т.ч. и таких условий как набросок, карта глубины или семантическая карта являются слои адаптивной нормализации.

3. [ControlNet](https://docs.google.com/presentation/d/1BqnRfLDRZQ-NmFLdJs7ZGN5zXjS9lZwBxSK9ip_TxB4/edit#slide=id.g30039e33537_0_118) и [T2I](https://docs.google.com/presentation/d/1BqnRfLDRZQ-NmFLdJs7ZGN5zXjS9lZwBxSK9ip_TxB4/edit#slide=id.g30039e33537_0_157) адаптеры используются для Layout2Image генерации. В качестве подсказки выступает семантическая карта, карта глубины, или canny edges. Так как никакой дополнительной информации о семантике (например название классов) модель не получает, то фактически все подсказки трактуются как информация о границах. Лосс без изменений - модель учиться предсказываеть только шум. Поэтому качественного следования сематической карте от этих моделей ожидать не следует.

4. Модели [FreeStyleNet](https://arxiv.org/abs/2303.14412) и [PLACE](https://arxiv.org/abs/2403.01852) преобразуют семантические классы в текстовые эмбеддинги и используют их как промпты. При этом они регулируют **cross-attention**, обнуляя веса для всех областей, кроме тех, где должен появиться объект нужного класса. Для работы таких методов требуется **дообучение** базовой модели на целевом датасете

### Редактирование изображений

Достаточно часто возникает задача изменить уже существующее изображение. Добавив/удалив на него новые объекты либо поменяв стиль. Можно делать это при помощи генеративных моделей, при этом возникает несколько проблем:

1. Как сохранить неизменным фон (часть)

2. Как подать на вход модели реальное изображение (см. Image Inversion)

3. Как управлять редактировнием

4. Как сохранить background при добавлении/изменении объекта

5. Как сгенерировать объект в подходящем месте (Affordance)

6. Как оценить качество генерации

Ответы на некоторые из этих вопросов в обзоре 2-х современных моделей: [StableFlow](https://arxiv.org/abs/2411.14430)(2025) и [MDE-Edit](https://arxiv.org/abs/2505.05101)(2025).

[Презентация Image editing](https://docs.google.com/presentation/d/1z7257tSYTU-a16zMSGO-3OyZgmSD3_pW0P6rlGJgI9k/edit?usp=sharing)

Разбор 5-ти статей посвященных методам редактирования изображений не требующих дообучения базовой модели (SEGA, CASO, FluxSpace, Concept Atention, AddIt).

Презентация [Image Editing & Inpainting](https://docs.google.com/presentation/d/1JeCfT1GQr-sLZRwe-f5gczEY6IEVBIlQVKDU5zrqiSQ/edit?slide=id.g3629ce7ecff_0_22#slide=id.g3629ce7ecff_0_22)

В задачах редактирования и layout2image бывает важно получить маску объекта еще на этапе генерации (по возможности на ранних шагах)

В презентации ниже сделан обзор техник позволяющих получать такие маски:

Презентация [Mask extraction](https://docs.google.com/presentation/d/186oqI9EQNti0qnEY-oemaaM5fe61v2EHNjEsy27vjEc/edit?slide=id.p#slide=id.p)

### Инверсия 

На практике часто возникает задача отредактировать уже существующее изображение. Что бы сделать это при помощи диффузионной модели нужно получить шум из соответствующий этому изображению. И уже затем изменив prompt или другий управляющий сигнал внести в нее изменения.

![inversion.png](https://ml.gan4x4.ru/wb/crm/generation/inversion.png)

Но получение такого шума не является тривиальной задачей. Ниже рассматриваются методы позволяющие более или менее успешно ее решат

[Презентация Inversion](https://docs.google.com/presentation/d/1ZFv50zFqWE6DEnIKjvthc4AJmeXYHdTzmeP0jwxFes8/edit?slide=id.g3847db70e73_0_192#slide=id.g3847db70e73_0_192)

### Генерация видео

[Презентация по Wan и VACE](https://docs.google.com/presentation/d/1ONqkptgesp3wXRlDVxuXw2YlbnG-s_yJsfSBAOzAwrk/edit?slide=id.p#slide=id.p)


### Unlearning

Современные диффузионные модели обучаются на огромных объемах неразмеченных данных из интернета. В процессе обучения модели усваивают различные паттерны, часть из которых может быть опасной: [NSFW](https://en.wikipedia.org/wiki/Not_safe_for_work)-контента, изображения защищенные авторским правом, и.т.п.

Для минимизации генерации нежелательного контента применяются специализированные техники для удаления информации о конкретных объектах или концепциях из памяти модели (unlearning), а так же постфильтрация вывода.

[Презентация](https://docs.google.com/presentation/d/1uup8ZzZq7MCjWqI_h5Jr5GIVXt0ERs2XcLsyY7IV9HU/edit?usp=sharing)


### Поиск аномалий в сгенерированных изображениях

В ходе генерации возникают ошибки и их нужно уметь находить. В частности это полезно для задачи обнаружения Deep Fake.

Ошибки можно разделить на две категории:

* Pixel-Level - Технические, такие как неестественные цвета, артефакты текстур и т.п.

* Semantic - level - Смысловые: включая несоразмерные масштабы, некорректные тени и нарушения законов физики.

Для их поиска существуют специальные методы. Некоторые из них описанны в презентации ниже:

[Презентация по поиску аномалий](https://docs.google.com/presentation/d/1zyT4xrNlX_fqyvHsuBCgsg2VxF65w32YWxB-O1RdY44/edit)
