# Видео

## Темы

- [Видео](#видео)
  - [Темы](#темы)
  - [Кодирование видео](#кодирование-видео)
  - [Детектирование движения](#детектирование-движения)
  - [Трекинг](#трекинг)
  - [Распознавание действий](#распознавание-действий)
  - [Метрики для оценки качества видео](#метрики-для-оценки-качества-видео)

## Кодирование видео

Передача и хранение видеоданных невозможна без сильного сжатия. Современные кодеки используют ряд алгоритмов:

1. Цветовая субдискретизация

![BlockNote image](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdRxAzsEKO2DZcuSRqYCSHurP2xLhjQC307sRbzein0naI1zEfUAaNwZNMQKuMoIews1pQcKYJA7HZosp8NqPgomzwRrdwxtCi3H7scUbYAi9o-R13RH2-jhEoNxqntNA-7lnRU=s2048?key=DRkXMwQO6717RQgQpZiWj7U5)

Так как человеческий глаз воспринимает яркость лучше, чем цвет, то информацию о цветах можно хранить в меньшем количестве бит.

2. Компенсация движения

Можно сохранять информацию только о переместившихся в кадре объектах (P-frame). Часть кадров можно интерполировать по соседним (B-frame).

3. Всю информацию сжимают: опорные кадры и остатки с потерями (DCT), вектора движения без потерь (энтропийное кодирование).

Материалы: [Цветовая субдискретизация](https://docs.google.com/presentation/d/1I-1PYkLD6fDcLwuoCTXLsUWkxkIJM7ChfpFZHblFiVA/edit#slide=id.gc5e2ee76ed_0_36), [Компенсация движения](https://docs.google.com/presentation/d/1I-1PYkLD6fDcLwuoCTXLsUWkxkIJM7ChfpFZHblFiVA/edit#slide=id.gc5e2ee76ed_0_66), [Презентация](https://docs.google.com/presentation/d/1I-1PYkLD6fDcLwuoCTXLsUWkxkIJM7ChfpFZHblFiVA/edit#slide=id.g701582b916_0_74), [Видео](https://youtu.be/yct7wAgwyjk)

## Детектирование движения

Чтобы снизить нагрузку на анализатор, желательно пропускать кадры, в которых не было движения.

Простейший алгоритм — это вычитание одного кадра из другого:

![BlockNote image](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUeCor1Y1f9o4BC8hNAq0v3ip613Xwdb7GVGxCo9OzuRKC7E-Z7QIck4fv5OMOq4tS3S-Qr6RXqMCgXl39_jJRH_wE8KOOActkmPVxJn8btTRvGDO345JxZQFfba4mIttKdnodpAqA=s2048?key=DRkXMwQO6717RQgQpZiWj7U5)

Если разница больше порога — считаем, что движения нет.

В реальности фон может быть не статичным:

![BlockNote image](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUefB9VUsP_0YbLZTS5F7B-4pyH60Nbd67lcL7raGxysZPiUoDAWsPioVPUzRvMzRyVuA54MWUfOhXeyK4Q4WHZE2UiiYRT13xsOEB0ASb2MjWvkoJb1awfsBQsEJx-HQx9aZzRvwA=s2048?key=DRkXMwQO6717RQgQpZiWj7U5)

В таком случае используются алгоритмы для [моделирования фона](https://docs.google.com/presentation/d/1I-1PYkLD6fDcLwuoCTXLsUWkxkIJM7ChfpFZHblFiVA/edit?usp=sharing).

Материалы: [Презентация](https://docs.google.com/presentation/d/1I-1PYkLD6fDcLwuoCTXLsUWkxkIJM7ChfpFZHblFiVA/edit#slide=id.g701582b916_0_140)

## Трекинг

При анализе видеоконтента часто требуется отследить перемещение объекта в пространстве. Это задача трекинга.

Сначала детектор предсказывает bounding box или центр объекта, а tracker предсказывает положение bbox в следующем кадре.

Соответственно к ошибкам детектора добавляются ошибки трекера.

Классические алгоритмы трекинга используют фильтр Калмана для предсказания траектории и Венгерский алгоритм для соотнесения предсказаний с bbox. В качестве меры схожести используется IoU между предсказанным и детектированным bbox. Современные алгоритмы используют признаки, полученные из модели.

Материалы: [ByteTrack_example.ipynb](ByteTrack_example.ipynb), [Презентация](https://docs.google.com/presentation/d/1I-1PYkLD6fDcLwuoCTXLsUWkxkIJM7ChfpFZHblFiVA/edit#slide=id.g6d6b35b102_0_0)

## Распознавание действий

Для распознавания процесса (действия, action) недостаточно одного кадра. Хороший пример — спорт.

Модели, которые решают эту задачу, обрабатывают сразу несколько кадров.

![BlockNote image](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdQ3xfQjoEkKQY6GGai9MUyuagBgC59SC4Uqv_HEktpuPdShIwwtILLGA0nkgKXzA9tJQjfg8NnIn0PJ2hGoUXh2NiO42sriPZyPcYuqjQCjpFkve8qCQ3Xs6w80Bp-nc583b_ofg=s2048?key=DRkXMwQO6717RQgQpZiWj7U5)

Для этого используются 3-D, где 3-е измерение не пространственное, а временное. При этом решается задача классификации фрагмента.

![BlockNote image](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfFOlJGz7_pPzeTIkb3fDjPwvW48KG07MDIrivYyn2oD3GT19bNgFAU1ocU769WGGm2hTUA8DLpxSyAmajjuBFXtAItS44eA-l05BM_twy_wPWpdkuUOEVw0xuu77dz9npLoxbEOg=s2048?key=DRkXMwQO6717RQgQpZiWj7U5)

В более современных моделях (SlowFast) две ветви. Одна извлекает пространственные признаки, другая — временные.

Материалы: [Презентация](https://docs.google.com/presentation/d/1I-1PYkLD6fDcLwuoCTXLsUWkxkIJM7ChfpFZHblFiVA/edit#slide=id.g6190b0c50f_0_0)

## Метрики для оценки качества видео

[VMAF](https://en.wikipedia.org/wiki/Video_Multimethod_Assessment_Fusion)

[LPIPS](https://github.com/richzhang/PerceptualSimilarity)


