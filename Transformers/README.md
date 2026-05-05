# Transformers

## Темы

- [Visual Transformers (ViT)](ViT.ipynb)
- [DEIT.ipynb](DEIT.ipynb)
- [Efficient_Attention.ipynb](Efficient_Attention.ipynb)

## Visual Transformers (ViT)

Блок Attention можно использовать не только для работы с текстами но и с изображениями.

![ViT](https://ml.gan4x4.ru/msu/dev-2.2/L10/out/visual_transformer_architecture.png)

Для этого изображение разрезается на патчи, каждый из которых  преобразуется в embedding при помощи линейного слоя. Затем к эмбеддингам добавляются метки кодирующие позицию. Получившуюся последовательность можно обрабатывать при помщи трансформер - энкодера так же как и текстовую.


Материалы: [ViT.ipynb](ViT.ipynb), [Семинар(видео)](https://youtu.be/ssgmtr59K9w), [Презентация](https://docs.google.com/presentation/d/1B0-nmpAAhSy19mxW8hnUPnBqPwme6CPM6PTteUnDn9w/edit?slide=id.

Дополнительные материалы:
g39fb8dc98ac_0_114#slide=id.g39fb8dc98ac_0_114), [articles](https://habr.com/ru/companies/otus/articles/849756/)
