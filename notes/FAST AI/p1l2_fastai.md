Data augmentation - аугментация данных
Epochs - эпохи обучения
Confusion matrix - матрица ошибок, имеет значение, только когда метки являются категориями.
loss - показатель того, насколько хороша наша модель, после каждого запуска на данных
Потери будут высоки, если мы неправильно предсказали и были в этом уверены.
confident - уверенность
ImageClassifierCleaner - очиститель классификатор изображений

*Интересный метод* - перед тем как очищать данные, обучите модель на своих данных, и посмотрите, с какими данными возникает ошибка, и найдите лучший способ избежать ее.

gradio, hugging face, wsl
zip, map
#| export 
GitHub pages

___

**Resume**:
 Данный видеоурок про просмотр и очищение данных во время обучения.

Также рассматривалось создание своего приложения с помощью gradio, и деплой через wsl ubuntu в hugging face / spaces. Импорт ячеек в один скрипт из jupyter notebook с помощью **#|export** и экспорт модели из colab / kaggle.
Создание бесплатного web приложения с помощью git pages и html кода.

___


*Saving a Cats v Dogs Model*
**Snippets**
1. Импорт инсрументов `from fastai.vision.all import *`
2. Скачивание и распаковка датасета `path = untar_data(URLs.PETS)/'images'`
3. Создание DataLoader 
```
    dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat,
    item_tfms=Resize(192))
```
4. Обучение классификатора 
```
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```
5. экспорт модели `learn.export('model.pkl')`


