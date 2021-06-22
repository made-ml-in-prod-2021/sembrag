# HW2 Online inference

Запуск скрипта ```predict_request.py```:

```python predict_request.py model_ip path/to/data```

Сборка и запуск модели в локальном Docker контейнере:
```
sudo docker build -t online_inference:v1 .
sudo docker run -p 8000:8000 online_inference:v1
```
Публикация в Docker hub:
```
sudo docker tag online_inference:v1 15042219/ml_in_prod_hw2
sudo docker push 15042219/ml_in_prod_hw2
```

Использование образа, опубликованного в хабе:
```
sudo docker pull 15042219/ml_in_prod_hw2:latest
sudo docker run -p 8000:8000 15042219/ml_in_prod_hw2
```

Самооценка:
----------
1) Оберните inference вашей модели в rest сервис (3 балла) - 3 балла

2) Напишите тест для /predict  (3 балла) - 2 балла

3) Скрипт, который будет делать запросы к вашему сервису  (2 балла) - 2 балла

4) Сделайте валидацию входных данных (3 доп балла) - 0 баллов

5) Напишите dockerfile, соберите на его основе образ и запустите локально, напишите в readme корректную команду сборки (4 балл) - 4 балла 

6) Оптимизируйте размер docker image (3 доп балла) - 0 баллов 

7) опубликуйте образ в https://hub.docker.com/ (2 балла) - 2 балла

8) напишите в readme корректные команды docker pull/run (1 балл) - 1 балл

5) проведите самооценку (1 доп балл) - 1 балл 
