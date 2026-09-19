# schemas/ — производные JSON Schema контрактов фазы 2

Каталог содержит JSON Schema пяти контрактов второй фазы (`docs/tasklist-phase2.md`, ADR-007):

| Файл | Контракт | Модель |
|---|---|---|
| `raw_entry.json` | `raw_entry` | `diary.domain.contracts.raw_entry.RawEntry` |
| `stt_corrected.json` | `stt_corrected` | `diary.domain.contracts.stt_corrected.SttCorrection` |
| `daily_summary.json` | `daily_summary` | `diary.domain.contracts.daily_summary.DailySummary` |
| `weekly_summary.json` | `weekly_summary` | `diary.domain.contracts.weekly_summary.WeeklySummary` |
| `coach_note.json` | `coach_note` | `diary.domain.contracts.coach_note.CoachNote` |

## Почему файлы производные

Источник истины — Pydantic-модели, а не эти файлы. Схемы генерируются
(`model_json_schema()`) и потому:

- не могут разойтись с кодом, который валидирует вывод LLM (AGENTS.md §15);
- читаются человеком и внешними инструментами (промпты, harness, документация) без импорта пакета;
- версионируются в Git, так что изменение формы контракта видно в ревью.

Эти файлы **никогда не правятся руками**: правка — только в модели, дальше регенерация. Если файл отличается от регенерации, тест `tests/unit/test_contracts_schemas.py` падает (сравнение байт-в-байт).

## Регенерация

```bash
make contracts                                          # перезаписать schemas/*.json
.venv/bin/python scripts/generate_contract_schemas.py --check   # только проверить, ничего не писать
```

Формат детерминирован: `json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"`. Генератор зависит только от стандартной библиотеки и Pydantic.

## Версия контракта

В самом файле нет поля версии: `contract_name` и `schema_version` объявлены на модели как `ClassVar` (`diary.domain.contracts._base.ContractModel`) и намеренно не попадают ни в схему, ни в сериализуемые данные — это метаданные контракта, а не payload. Версия пары (вход, слой) описана в ADR-007 §2.

## Что считается ломающим изменением (veto)

Ломающее изменение — любое, из-за которого уже сохранённые артефакты фазы 2 или готовые промпты/интеграции перестают соответствовать контракту:

- удаление или переименование поля;
- смена типа поля, сужение допустимых значений (например, `energy` без `None`), ужесточение `minItems`/`maxItems`/`pattern`;
- переход `extra="forbid"` → `extra="allow"` (перестаёт ловить лишние поля);
- изменение смысла поля (например, отказ от «`None` = источник молчит»).

Порядок для ломающего изменения (ADR-007 §2, `docs/tasklist-phase2.md` §5):

1. новый ADR с обоснованием;
2. поднятая `schema_version` на модели;
3. обновление golden-набора (`eval/golden`) и промптов, которые уже используют контракт;
4. отдельная задача и явное подтверждение пользователя.

Совместимое добавление (новое необязательное поле, мягче ограничение) версию не поднимает, но всё равно проходит через регенерацию схем и тесты.

См. также: `../eval/golden/README.md`, `../docs/decisions/ADR-007-second-anchor-family-for-derivational-layers.md`, `../prompts/README.md`.
