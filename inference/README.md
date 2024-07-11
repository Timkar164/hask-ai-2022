# Inference

В данной директории находятся блокноты, позволяющие проверить работоспособность алгоритмов без их тренеровки.


Лучше начать с файла [lstm/LSTM_inference.ipynb](lstm/LSTM_inference.ipynb), поскольку веса модели уже в репозитории. Для запуска понадобится установить зависимости
```
pip install tensoflow==2.8.0 keras==2.8.0
pip install h5py==2.10.0
``` 

Для запуста RuBERT требуется скачать с положить в папку к проекту файл `saved_weights_2_digits.pt` в папку к файлу `96_v1_transofrmers_inference.ipynb` и запустить его

Для запуска понадобится установить зависимости
```
pip install torch
pip install transformers[torch]
``` 

