# Transformers

## Темы

- [Visual Transformers (ViT)](ViT.ipynb)

## Visual Transformers (ViT)

Блок Attention можно использовать не только для работы с текстами но и с изображениями.

![BlockNote image](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUf6NpnmBArXondTbopq7uKOVVAJo8vsmgPL4g1vEB9GZlEodr_4e4rK-h0y5sP_uQ4Iz4da91_uptxbRl7JtBlQmrXw6nZ04o0dMWxAZhyUANPWF6khuRp3FJEyQUjmwTISE7k_VFOyD-71SHF8LczPVbbh7QJA=s2048?key=RuvGefpjJ1y9lEFnhqk8Cw)

Для этого изображение разрезается на патчи, каждый из которых  преобразуется в embedding при помощи линейного слоя. Затем к эмбеддингам добавляются метки кодирующие позицию. Получившуюся последовательность можно обрабатывать при помщи трансформер - энкодера так же как и текстовую.

[Презентация](https://docs.google.com/presentation/d/1NarIAm-A7BiIiYgKB_pzPBglBp2zKfZ-dRqxtvLktRo/edit#slide=id.p)

[Семинар(видео)](https://youtu.be/ssgmtr59K9w)

Материалы: [ViT.ipynb](ViT.ipynb), [presentation](https://docs.google.com/presentation/d/1B0-nmpAAhSy19mxW8hnUPnBqPwme6CPM6PTteUnDn9w/edit?slide=id.g39fb8dc98ac_0_114#slide=id.g39fb8dc98ac_0_114), [articles](https://habr.com/ru/companies/otus/articles/849756/)
