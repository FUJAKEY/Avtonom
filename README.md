# Avtonom

Этот проект демонстрирует использование Google Gemini API для создания
агента, который может читать и изменять файлы в каталоге `workspace`,
а также выполнять команды терминала. Скрипт `gemini_agent.py` запускает
чат с моделью `gemini-2.0-flash` и позволяет ей поэтапно строить
проект Discord‑бота.

Перед запуском убедитесь, что в файле `.env` указан рабочий API‑ключ.
Зависимости устанавливаются в папку `env` командой:

```bash
pip install --target ./env google-generativeai python-dotenv
```

Запуск агента:

```bash
python3 gemini_agent.py
```
