# Блокноты с алгоритмами обучения

В данной директории расположены блокноты с обучением глубоких моделей.

Тренируемые алгоритмы:

SDG, KNN, DeepPavlov RuBert, LSTM и т.д.


В файлах `TNVED_2_digits_count.csv` и `TNVED_4_digits_count.csv` хранится подсчитанное количество объектов каждой группы и подгруппы ТН ВЭД в предоставленном наборе данных.


*Алгоритмы для предсказания первых 2 цифр:*

SDG:

```
              precision    recall  f1-score   support
...
    accuracy                           0.90    752022
   macro avg       0.70      0.91      0.76    752022
weighted avg       0.91      0.90      0.91    752022
```

DeepPavlov RuBERT:

```
              precision    recall  f1-score   support
...
    accuracy                           0.67     22465
   macro avg       0.46      0.55      0.47     22465
weighted avg       0.71      0.67      0.68     22465
```

LSTM:

```
              precision    recall  f1-score   support
...
    accuracy                           0.94    250672
   macro avg       0.75      0.82      0.78    250672
weighted avg       0.94      0.94      0.94    250672
```


*Алгоритмы для предсказания первых 4 цифр:*

SDG:

```
              precision    recall  f1-score   support
...
    accuracy                           0.84    752022
   macro avg       0.46      0.63      0.50    752022
weighted avg       0.88      0.84      0.85    752022
```

DeepPavlov RuBERT:

```
              precision    recall  f1-score   support
...
   micro avg       0.47      0.50      0.48       288
   macro avg       0.01      0.01      0.01       288
weighted avg       0.59      0.50      0.51       288
```

Датасет для обучения алгоритмов сохранен c нашей предобработкой в файле `dataset.csv` (~1 гигабайт). Скачать его можно по ссылке: [https://dropmefiles.com/GuV8W](https://dropmefiles.com/GuV8W)


Начать перетренировку можно с модели LSTM для предсказания 2 классов: [LSTM_Keras_2_digits.ipynb](LSTM_Keras_2_digits.ipynb)
Необходимо установить библиотеки tensorflow, keras, h5py

```
pip install tensoflow==2.8.0 keras==2.8.0
pip install h5py==2.10.0
``` 
