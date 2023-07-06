# Model conversion web service
Репозиторий содержит сервис преобразования нейронных моделей между форматами различных фреймворков машинного обучения

### Стандартный пайплайн сервиса
- Conversion: Torch -> Onnx -> Tensorflow -> Tensorflow Lite
- Verification: (Torch, Onnx), (Torch, Tensorflow Lite)

Не поддерживаются модели со слоями:
- **Adaptive avg pool** (не поддерживается в Onnx)
  - VGG 16
- Слой **resize** с параметром **pytorch_half_pixel** (не поддерживается в Tensorflow)
  - Faster RCNN
  - Mask RCNN


### Выполнение тестового запроса:
1. Скачать **.mar** архив с моделью из зоопарка Torch в папку **/tests/archives**
2. Файл **model.py** в архиве должен содержать класс **ImageClassifier**
2. Повторить 1-2 для всех нужных моделей
1. Выполнить `flask run` для запуска сервиса
2. Выполнить `python tests/sender.py` для запроса, отправляющего все подготовленные архивы из папки **/tests/archives**
3. Результаты конвертации сохранятся в **/tests/output**

### Формат логирования
Файл **.json**, содержащий список словарей

###### Формат логирования для конвертеров
Словарь, содержащий следующие поля: 
1. **process_type**
2. **model_name**
3. **input_format**
4. **output_format**
5. **model_size**
6. **process_time**

###### Формат логирования для верификаторов
Словарь, содержащий следующие поля: 
1. **process_type**
2. **model_name**
3. **input_format**
4. **output_format**
5. **model_size**
6. **process_time**
7. **threshold**
8. **max_difference**
9. **mismatched_percent**
