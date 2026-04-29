# Production

## Темы

- [StreamLit](Streamlit_tutorial.ipynb)
- [Ускорение модели](#ускорение-модели)
- [Конвертация в формат для запуска](onnxruntime/README.md)

## StreamLit

Материалы: [Streamlit_tutorial.ipynb](Streamlit_tutorial.ipynb)

## Ускорение модели

При развертывании модели в production-среде критичным становится время инференса. При этом зачастую можно допустить небольшое понижение точности работы модели. Одним из способов уменьшить размер модели после обучения является прунинг.

![pruning.png](https://ml.gan4x4.ru/wb/crm/production/pruning.png)

Идея метода состоит в том, чтобы удалить из модели часть нейронов (фильтров/блоков), которые оказывают наименьшее влияние на результат. Обычно начинают с блоков/фильтров/голов внимания с наименьшей L1/L2 нормой, затем модель тестируется. Во многих моделях удаление до 50% весов незначительно ухудшает результат.

### Половинная точность

Другим способом сэкономить является конвертация весов модели в более компактный формат, например, во float16 или bfloat16. Для весов и активаций такая потеря точности не критична. При обучении используется гибридный подход, когда обновление весов происходит во float32.

### Квантизация

Для еще большей экономии при инференсе веса модели переводят в еще более компактный целочисленный формат int8 (иногда даже в int4).

![quant.png](https://ml.gan4x4.ru/wb/crm/production/quant.png)

Это дает многократный выигрыш в производительности, но требует масштабирования активаций. Для этого разработано несколько стратегий: считать коэффициенты масштабирования на калибровочных данных (static quantization), проводить вычисления во float (dynamic quantization) или использовать округление во время обучения (quantization aware training).

Материалы: [Прунинг](https://docs.google.com/presentation/d/1unSWOcr8VmPC6S-5yG5Qf6g-ngBoniXvLjJOklKWF2I/edit#slide=id.gc932c3f2bd_0_8), [Квантизация](https://docs.google.com/presentation/d/1unSWOcr8VmPC6S-5yG5Qf6g-ngBoniXvLjJOklKWF2I/edit?slide=id.g7f1c8b4f35_0_78#slide=id.g7f1c8b4f35_0_78)

## Конвертация в формат для запуска

Pytorch — это прежде всего фреймворк для обучения моделей. Когда модель запускается в production-окружении, инференс можно ускорить не только за счёт прунинга и квантизации, но и путём конвертации модели в более эффективный формат.

Torch script. Встроенный в PyTorch пакет позволяет экспортировать модель в формат, позволяющий запустить её в другом окружении, чаще всего на C++.

ONNX (Open Neural Network Exchange) — открытый формат для обмена моделями между фреймворками: PyTorch, TensorFlow и др. (хотя поддерживаются не все операторы). Позволяет запускать оптимизированный инференс без зависимостей от исходного фреймворка — например, через легковесный `onnxruntime` в Python или аналоги для C++/C#/Java.

TensorRT. Фреймворк, оптимизирующий модель для запуска на GPU Nvidia, поддерживает прунинг, квантизацию и другие оптимизации. Поддерживает конвертацию из PyTorch, ONNX и других форматов.

Материалы: [onnxruntime](onnxruntime/README.md), [1_onnxrt_colab.ipynb](onnxruntime/1_onnxrt_colab.ipynb), [2_onnxrt_paths.ipynb](onnxruntime/2_onnxrt_paths.ipynb), [3_onnxrt_cuda.ipynb](onnxruntime/3_onnxrt_cuda.ipynb), [4_onnxrt_bench.ipynb](onnxruntime/4_onnxrt_bench.ipynb), [Torch script](https://docs.google.com/presentation/d/1unSWOcr8VmPC6S-5yG5Qf6g-ngBoniXvLjJOklKWF2I/edit?slide=id.gc932c3f2bd_0_3#slide=id.gc932c3f2bd_0_3), [ONNX](https://docs.google.com/presentation/d/1unSWOcr8VmPC6S-5yG5Qf6g-ngBoniXvLjJOklKWF2I/edit?slide=id.g7de450b21e_0_0#slide=id.g7de450b21e_0_0), [TensorRT](https://docs.google.com/presentation/d/1unSWOcr8VmPC6S-5yG5Qf6g-ngBoniXvLjJOklKWF2I/edit?slide=id.g7de450b21e_0_0#slide=id.g7de450b21e_0_0)
