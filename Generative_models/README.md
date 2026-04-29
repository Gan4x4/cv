# Генеративные модели

## Темы

- [Вариационные автоэнкодеры (VAE)](VAE.ipynb)
- [GAN](generative_models.ipynb)
- [Принципы работы диффузионных моделей](diffusers_models.ipynb)

## Вариационные автоэнкодеры (VAE)

Материалы: [VAE.ipynb](VAE.ipynb), [presentation](https://docs.google.com/presentation/d/1FdS5eB5QMXBaNR7cK_LhP947n-Xu1uVIGDliN2iqWec/edit?slide=id.g10c4be00ca3_0_106#slide=id.g10c4be00ca3_0_106), [variational-autoencoder-(vae)](https://education.yandex.ru/handbook/ml/article/variational-autoencoder-(vae)), [presentation](https://docs.google.com/presentation/d/1K24n4w18U9DqnzNw-Am41giPMNbFodZzXzyvb3Q037Y/edit?usp=sharing)

## GAN

Генератор создает данные на основе случайного шума, стремясь обмануть дискриминатор, который, в свою очередь, обучается отличать истинные данные от сгенерированных. В процессе совместного обучения обе сети улучшаются, что позволяет генератору создавать всё более реалистичные данные.

![generative_adversarial_network_scheme.png](https://ml.gan4x4.ru/msu/dev-2.4/L13/out/generative_adversarial_network_scheme.png)

Материалы: [generative_models.ipynb](generative_models.ipynb), [presentation](https://docs.google.com/presentation/d/1FdS5eB5QMXBaNR7cK_LhP947n-Xu1uVIGDliN2iqWec/edit?slide=id.g10c4be00ca3_0_235#slide=id.g10c4be00ca3_0_235), [problems](https://developers.google.com/machine-learning/gan/problems), [p](https://proglib.io/p/generativno-sostyazatelnaya-neyroset-vasha-pervaya-gan-model-na-pytorch-2020-08-11), [p](https://proglib.io/p/generativnaya-sostyazatelnaya-set-gan-dlya-chaynikov-poshagovoe-rukovodstvo-2021-07-16), [abs](https://arxiv.org/abs/1806.04304), [what-does-fid-do](https://sites.google.com/view/what-does-fid-do/), [frechet_inception_distance](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html), [kailashahirwar.medium.com](https://kailashahirwar.medium.com/a-very-short-introduction-to-inception-score-is-c9b03a7dd788), [inception_score](https://lightning.ai/docs/torchmetrics/stable/image/inception_score.html#inception-score), [html](https://arxiv.org/html/2408.15098v1), [clip_iqa](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html)

## Принципы работы диффузионных моделей

![distributions.png](https://ml.gan4x4.ru/wb/crm/generation/distributions.png)

Так же как и VAE модель учиться сопоставлять простое рапределение данных целевому. Но делается это за несколько шагов.

![diff_dataset.png](https://ml.gan4x4.ru/wb/crm/generation/diff_dataset.png)

Технически модель учиться на зашумленных изображениях. Количество шума который добавляется к изображению зависит от шага и конкретного алгоритма (DDPM, DDIM, FM, RF ...). Задача модели предсказать шум, лосс = MSE

![diff_base_scheme.png](https://ml.gan4x4.ru/wb/crm/generation/diff_base_scheme.png)

После того как модель обучилась из шума можно  генерировать изображения итеративно предсказывая шум для каждого шага и вычитая его.

[Презентация](https://docs.google.com/presentation/d/1tNHAvV2FLz2lI8xYqnPUK1zNOL3Zq6dZ6YUKBq_2zCU/edit?usp=sharing)

Материалы: [diffusers_models.ipynb](diffusers_models.ipynb), [handmade_diffusion.ipynb](handmade_diffusion.ipynb), [From_DDPM_to_ODE_and_Flow_Matching.ipynb](From_DDPM_to_ODE_and_Flow_Matching.ipynb)
