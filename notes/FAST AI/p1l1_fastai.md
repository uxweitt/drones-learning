DataBlock - блок данных
Deep learning - глубокое обучение
Машинный алгоритм, который генерирует изображения, используя только текст.
Языковая гугл модель PaLM, которая может ответить на вопрос и объяснить свой ответ.
Мы не даем NN функции, мы даем ей их изучить.
Многослойная структура.
PyTorch, TensorFlow, fast.ai.
Jupyter Notebook & kaggle
Просмотр данных на каждом этапе обучения модели.
train, valid - тренировочный и валидный набор данных, для проверки модели.
dataloader - загрузчик данных.
batch - порция.
*.pth - набор весов.
fine_tune - тонкая настройка.
predict - сделать прогноз.
Сегментация.
Табличный анализ.
Совместная фильтрайия в рек. системах.
Главное - экспериментировать.
___
**Resume**: суть урок ввести в курс fast.ai и рассказать, как будет проходить обучение и с чем.
___

**Problems**
- Необходим Numpy<2.0 для библиотек Torch
___

**Snippets**
1. **импорт** `from fastcore.all import *`
2. **скачивание фото:**
```
searches = 'drone','fly cat'
path = Path('cat_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    time.sleep(5)
    resize_images(path/o, max_size=400, dest=path/o)
```
*resize* - обычно vision моделям не нужно большой размер картинки, поэтому мы ограничиваем его для ускорения обучения.

3. **Проверка на ошибку скачивания**
```
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```
4. **Создание DataLoaders с использованием DataBlock**
```
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
```
Мы используем ***splitter***, чтобы разделить данные на тренировочные - на которых обучается модель, и валидные - на которых модель проверяет себя.

5. **Обучение модели**
```
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```
Библиотека fast.ai сама подбирает нам нужную модель для нашего случая (классификация картинки). resnet18 в свое время стало прорывом в vision models.

на вход ученику мы даем наш dataloader - загрузчик данных и название модели, которые мы можем посмотреть в библиотеке timm.

6. **Inferance модели**
```
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
```