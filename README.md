# Решение хакатона «Цифровой прорыв: сезон ИИ» (2023)

#### Команда flint3s

#### Кейс «Разработка модели по устранению и изменению некорректных адресов из городских баз данных»

## Демо

- Веб-демо: https://amm.flint3s.ru/
- API: https://amm.flint3s.ru/api

## Запуск и деплой

Самый простой вариант запустить модель - использовать подготовленные docker-образы и
файл `docker-compose.dev-images.yml`, для его запуска необходимо:

1. Скачать предобученную модель и
   вектора: https://drive.google.com/drive/folders/1BPg1wA6gAyfgf1rH8Bc15cFobZVoufbj?usp=sharing
2. Поместить файлы `512_new_fasttext.model` и `512_new_fasttext.model.wv.vectors_ngrams.npy` в папку `server/model`
3. Запустить docker-контейнеры: ` docker-compose -f docker-compose.dev-images.yml up -d`
4. Дождаться инициализации модели

В результате по адресу `http://localhost:8001/api` будет доступна серверная часть, а по адресу `http://localhost:3000` -
клиентское веб-приложение. Подробнее о настройках каждого из сервисов читайте ниже в соответствующих секциях

## ML Модель

Обучили на предоставленных данных FastText, построили эмбеддниги для всех адресов. Когда получаем новый адрес - ищем
ближайшие эмбеддинги, из них выбираем по совпадению фичей (номер дома, район, etc)

## Серверная часть

Серверная часть приложения выполнена с использованием Python и FastAPI для построения HTTP API.
Внутрь серверной части интегрирована работа с ML-моделью, поэтому для его работы также необходима установка зависимостей
для выполнения кода модели. Для уствноки всех зависимостей достаточно в папке `/server` выполнить команду:

```bash
pip install -r requirements.txt
```

Так же для запуска необходимо скачать модель (см. пункт "Запуск и деплой") и поместить её в папку `server/model`. После
этого можно запускать сервер в своем окружении: `python ./main.py`

После запуска серверная часть будет доступна по адресу `http://localhost:8000`, а также Swagger по
адресу `http://localhost:8000/api/docs`

## Веб-приложение

Для удобного использования модели был также разработан веб-интерфейс на TypeScript и Vue.JS. Веб-интерфейс представляет
доступ ко всем функциям сервиса:

- Единичное распознавание адреса, результатом которого является ответ с результатами, ранжированными по уверенности
  модели
- Пакетное распознавание, с помощью которого можно распознать сразу список из адресов - в нем, как и в остальных методах
  также поддерживается детекция неверной раскладки и транслитерации адреса
- Автодополнение, при вводе запроса система предлагает варианты, которые подходят под часть введенного адреса
- Распознавание адресов из файлов. На данный момент поддерживаются файлы txt и json, принцип работы схож с пакетной
  обработкой - все адреса из файла распознаются и пользователь получает JSON с корректными адресами

Веб-интерфейс имеет настройку API-пути - в правом верхнем углу доступна кнопка "Настройки", где можно вручную указать
URL серверной части

## CLI

Третий способ взаимодействия с системой - использование CLI. Он работает в двух режимах: API и локальный режим. В
локальном режиме используются ресурсы текущего компьютера и для него необходима сама модель. Плюс такого подхода в том,
что он может работать оффлайн. Примеры использования CLI:

```shell
python .\amm-cli.py --mode=local --file ./test.txt
python .\amm-cli.py --mode=local Санкт-Петербург
python .\amm-cli.py --mode=api --api-path=https://amm.flint3s.ru/api Санкт-Петербург
```