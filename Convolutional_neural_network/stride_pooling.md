<a href="https://colab.research.google.com/github/Gan4x4/cv/blob/main/Convolutional_neural_network/stride_pooling.ipynb">
  ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
</a>

# Применение свёрточных слоёв

Поскольку операция свертки является линейной, то функция активации (например, ReLU) по-прежнему требуется.

>*Так как функция активации применяется к тензору поэлементно, не важно, какую именно форму имеет тензор, а значит и какой слой находился перед ней: полносвязный или сверточный.*

Простейшая модель для MNIST может выглядеть примерно так:

```python
import torch
from torch import nn

batch_size = 1
input = torch.randn((batch_size, 1, 28, 28))

model = torch.nn.Sequential(
    nn.Conv2d(
        in_channels=1, out_channels=3, kernel_size=5
    ),  # after conv shape: [batch_size,3,24,24]
    nn.ReLU(),  # Activation doesn't depend on input shape
    nn.Conv2d(
        in_channels=3, out_channels=6, kernel_size=3
    ),  # after conv shape: [batch_size,6,22,22]
    nn.ReLU(),
    nn.Flatten(),  # 6*22*22=2904
    nn.Linear(2904, 100),
    nn.ReLU(),  # Activation doesn't depend on input shape
    nn.Linear(100, 10),  # 10 classes, like a cifar10
)

out = model(input)
print(f"out shape: {out.shape}")
```

```stdout
out shape: torch.Size([1, 10])
```

Поскольку полносвязный слой принимает на вход набор векторов, а сверточный — возвращает набор трёхмерных тензоров, нам нужно превратить эти тензоры в вектора. Для этого используется объект класса `nn.Flatten` [🛠️[doc]](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html#torch.nn.Flatten).
Он преобразовывает данные на входе в вектор, сохраняя при этом первое (batch) измерение.

Ниже примеры других функций, которыми можно выполнить аналогичное преобразование:

```python
input = torch.randn((16, 3, 32, 32))

batch_size = input.shape[0]

print("class Flatten\t", nn.Flatten()(input).shape)
print(
    "view \t\t", input.view(batch_size, -1).shape
)  # data stay in same place in memory
print("reshape \t", input.reshape(batch_size, -1).shape)  # data may be moved
print("method flatten \t", input.flatten(1).shape)
```

```stdout
class Flatten	 torch.Size([16, 3072])
view 		 torch.Size([16, 3072])
reshape 	 torch.Size([16, 3072])
method flatten 	 torch.Size([16, 3072])
```

### Рецептивные поля нейронов

Нейросетевая модель из предыдущего примера позволяет в общем случае понять структуру свёрточных нейронных сетей: после некоторого количества свёрточных слоёв, извлекающих локальную пространственную информацию, идут полносвязные слои (как минимум в количестве одного), сопоставляющие извлечённую информацию.

Внутри свёрточных слоёв происходит следующий процесс: первые слои нейронных сетей имеют малые рецептивные поля, т. е. им соответствует малая площадь на исходном изображении. Такие нейроны могут активироваться лишь на некоторые простые шаблоны (по типу углов или освещённости).

Нейроны следующего слоя уже имеют большие рецептивные поля, в результате чего в картах признаков появляется информация о более сложных паттернах. С каждым слоем свёрточной нейронной сети рецептивное поле нейронов увеличивается. Увеличивается и сложность шаблонов, на которые может реагировать нейрон. В последних слоях рецептивное поле нейрона должно быть размером со всё исходное изображение. Пример можно увидеть на схеме ниже.

![image](https://ml.gan4x4.ru/msu/dev-2.2/L06/out/receptive_field_size.png)

Если на первом слое рецептивное поле имело размер $K \times K$, то после свёртки фильтром $K\times K$ оно стало иметь размер $(2K-1) \times (2K-1)$, то есть увеличилось на $K-1$ по каждому из направлений. Несложно самостоятельно убедиться, что данная закономерность сохранится при дальнейшем применении фильтров того или иного размера.

Однако при обработке больших изображений нам потребуется очень много слоев, чтобы нейрон "увидел" всю картинку.

К примеру, для изображения $1024\times1024$ понадобится сеть глубиной $\approx510$ сверточных слоев.

Такая модель потребует огромного количества памяти и вычислительных ресурсов.
Чтобы избежать этого, будем сами уменьшать размеры карт признаков, при этом рецептивные поля нейронов будут расти.

### Шаг свёртки (Stride)

Стандартная операция свертки двигает фильтр на один пиксель, то есть перемещает с шагом (stride) $= 1$.

Если двигать фильтр с большим шагом, то размер выходной карты признаков (feature map) будет уменьшаться кратно шагу, и рецептивные поля нейронов будут расти быстрее.

Для изменения шага свертки в конструкторе `nn.Conv2d` [🛠️[doc]](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) есть параметр `stride`.

```python
dummy_input = torch.randn(1, 1, 5, 5)
conv_s1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=(1, 1))
conv_s2 = nn.Conv2d(1, 3, 3, stride=2)  # bypass par. names, stride = (2, 2)

out_stride1 = conv_s1(dummy_input)
out_stride2 = conv_s2(dummy_input)

print("Out with stride 1", out_stride1.shape)
print("Out with stride 2", out_stride2.shape)
```

```stdout
Out with stride 1 torch.Size([1, 3, 3, 3])
Out with stride 2 torch.Size([1, 3, 2, 2])
```

<center>![image](https://ml.gan4x4.ru/msu/dev-2.2/L06/out/convolution_parameter_stride.gif)</center>
<center><em>Свёртка массива $5\times5$ фильтром размером $3\times3$ с шагом $2$ по вертикали и горизонтали.</em></center>

При этом важно заметить, что в некоторых случаях часть данных может не попасть в свёртку. К примеру, при $N = 7,\, K = 3,\, S = 3$. В данном случае: $$\large N' = 1 + \frac{7 - 3}{3} = 2\frac13.$$
 В подобных ситуациях часть изображения не захватывается, в чём мы можем убедиться на наглядном примере:

```python
# Create torch tensor 7x7
# fmt: off
input = torch.tensor([[[[1, 1, 1, 1, 1, 1, 99],
                        [1, 1, 1, 1, 1, 1, 99],
                        [1, 1, 1, 1, 1, 1, 99],
                        [1, 1, 1, 1, 1, 1, 99],
                        [1, 1, 1, 1, 1, 1, 99],
                        [1, 1, 1, 1, 1, 1, 99],
                        [1, 1, 1, 1, 1, 1, 99]]]], dtype=torch.float)
# fmt: on

print(f"input shape: {input.shape}")

conv = torch.nn.Conv2d(
    in_channels=1,  # Number of channels
    out_channels=1,  # Number of filters
    kernel_size=3,
    stride=3,
    bias=False,  # Don't use bias
)
conv.weight = torch.nn.Parameter(
    torch.ones((1, 1, 3, 3))
)  # Replace random weights to ones
out = conv(input)

print(f"out shape: {out.shape}")
print(f"out:\n{out}")
```

```stdout
input shape: torch.Size([1, 1, 7, 7])
out shape: torch.Size([1, 1, 2, 2])
out:
tensor([[[[9., 9.],
          [9., 9.]]]], grad_fn=<ConvolutionBackward0>)
```

Видно, что столбец с числами $99$ просто не попал в свертку.
Поэтому на практике подбирают padding таким образом, чтобы при `stride = 1`  размер карты признаков на выходе был равен входу, а затем делают свертку со `stride = 2`.

Казалось бы, с увеличением шага $S$ рецептивное поле не выросло — как увеличивалось с $1$ до $K$, так и увеличивается. Однако обратим внимание на другое: если раньше размерность $N$ становилась $N - F + 1$, то теперь она станет $\displaystyle 1 + \frac{N-F}{S}$.

В результате если раньше следующий фильтр с размером $K'$ имел рецептивное поле:$$\displaystyle N \cdot \frac{K'}{N'} = N \cdot \frac{K'}{N - F + 1},$$

то теперь: $$\displaystyle N \cdot \frac{K'}{N'} = N \cdot \frac{K'}{1 + \frac{N-F}{S}}.$$

Понятно, что $$\displaystyle \frac{K'}{N - F + 1} \leq \frac{K'}{1 + \frac{N-F}{S}},$$ поэтому рецептивное поле каждого нейрона увеличивается.

### Уплотнение (Субдискретизация, Pooling)

Другим вариантом стремительного увеличения размера рецептивного поля является использование дополнительных слоёв, требующих меньшего количества вычислительных ресурсов. Слои субдискретизации прекрасно выполняют эту функцию: подобно свёртке производится разбиение изображения на небольшие сегменты, внутри которых выполняются операции, не требующие использования обучаемых весов. Два популярных примера подобных операций: получение максимального значения (max pooling) и получение среднего значения (average pooling).

**Важно понимать**, что Pooling слои не являются сверточными слоями, так как в них нет фильтров с обучаемыми весами, т. е. они никак не настриваются в процессе обучения нейросети. Это просто эффективный способ уменьшить пространственные размеры карт признаков.


Аналогично разбиению на сегменты при свёртке, слои субдискретизации имеют два параметра: размер фильтра $K$ (то есть, каждого из сегментов) и шаг $S$ (stride). Аналогично свёрткам, при применении субдискретизации формула размера стороны:
$$N' = 1+ \frac{N-K}{S}.$$

Ниже приведён пример использования операций max pooling и average pooling при обработке массива.

![image](https://ml.gan4x4.ru/msu/dev-2.2/L06/out/subdiscretization_pooling.png)

Реализуем это в коде:

```python
# create tensor 4x4
# fmt: off
input = torch.tensor([[[[1, 1, 2, 4],
                        [5, 6, 7, 8],
                        [3, 2, 1, 0],
                        [1, 2, 3, 4]]]], dtype=torch.float)
# fmt: on

max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

print("Input:\n", input)
print("Max pooling:\n", max_pool(input))
print("Average pooling:\n", avg_pool(input))
```

```stdout
Input:
 tensor([[[[1., 1., 2., 4.],
          [5., 6., 7., 8.],
          [3., 2., 1., 0.],
          [1., 2., 3., 4.]]]])
Max pooling:
 tensor([[[[6., 8.],
          [3., 4.]]]])
Average pooling:
 tensor([[[[3.2500, 5.2500],
          [2.0000, 2.0000]]]])
```

**Важно отметить**, что субдискретизация выполняется по каждому из каналов отдельно, в результате чего количество каналов не меняется, в отличие от применения фильтра при свёртке. К примеру, ниже можно увидеть визуализацию применения max pooling к одному из каналов тензора, имеющего $64$ канала.

![image](https://ml.gan4x4.ru/msu/dev-2.2/L06/out/changing_size_of_image_after_pooling.png)
