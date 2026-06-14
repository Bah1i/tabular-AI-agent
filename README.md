# Tabular AI Agent

Система для автоматической трансформации табличных данных: пользователь загружает source и optional expected example, LLM генерирует `transform(df)`, код проходит static validation, выполняется в Docker sandbox и проверяется comparator-ом. Для FOOFAH используется отдельная prompt strategy и hidden-style проверка `TestingTable.csv -> TestAnswer.csv`.

## Запуск

```powershell
Copy-Item .env.example .env
docker build -t tabular-agent-sandbox:latest ./sandbox
docker compose up --build
```

UI:

```text
http://localhost:8000
```

## LLM

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

Старые переменные `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` сохранены для совместимости.

## Основные страницы

```text
/                     main UI
/ui/jobs/{job_id}     job details, previews, attempts, ala-lens
/ui/postgres          SQL read-only mode
/ui/benchmarks        FOOFAH runner
/ui/metrics           dashboard по всем job metrics
/metrics/summary      JSON metrics
```

## Ala-lens audit trail

Ala-lens хранится в таблице `ala_lens_events`. События:

- `GET`: синтез параметра `p`;
- `DELTA`: найдено несоответствие actual vs expected;
- `AMENDMENT`: repair-loop изменил параметр `p`;
- `STABILITY`: результат согласован с expected.

Для каждого события сохраняются:

- source model;
- view model;
- parameter before/after;
- delta;
- amendment;
- `prompt_strategy`;
- `code_hash`;
- `validation_status`;
- полный JSON.

На странице job сначала показывается человекочитаемая карточка, а полный JSON раскрывается отдельно.

## PostgreSQL read-only mode

Страница:

```text
http://localhost:8000/ui/postgres
```

Возможности:

- подключение к PostgreSQL по host/port/user/password;
- выбор database/schema/table или всей схемы;
- получение metadata: tables, columns, types, row count estimate, foreign keys;
- natural language query -> SQL SELECT;
- ручной SQL SELECT;
- optional expected CSV/XLSX для validation результата;
- preview результата;
- `EXPLAIN` без `ANALYZE`.

Безопасность:

- пароль не отправляется в LLM;
- LLM получает только schema metadata;
- разрешены только `SELECT` и `WITH ... SELECT`;
- запрещены `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, `COPY`, `CALL`, `DO`;
- backend добавляет `LIMIT 1000`, если LIMIT отсутствует;
- запрос выполняется внутри `BEGIN READ ONLY`;
- устанавливается `statement_timeout`.

Demo expected files:

```text
examples/sql_expected_job_metrics_summary.csv
examples/sql_expected_benchmark_dataset_summary.csv
```

Их можно использовать как expected-файлы в SQL mode, если в базе есть соответствующие данные и SQL возвращает такую же структуру.

## Metrics dashboard

`/ui/metrics` показывает глобальные метрики по `JobMetric`:

- success rate;
- first attempt success rate;
- repair success rate;
- average attempts;
- total tokens;
- cost per success;
- error type distribution;
- sandbox timeout rate;
- validation failure rate;
- cache hit rate;
- p95 latency.

В UI у метрик есть подсказки `?` с русским описанием и формулой.

## FOOFAH benchmark

FOOFAH tables читаются как 2D string matrix без доверенного header row. Для FOOFAH используется отдельная prompt strategy `foofah`; обычный business transform prompt не смешивается с FOOFAH prompt.

Case semantics:

```text
InputTable.csv   + OutputTable.csv  = example для synthesis
TestingTable.csv + TestAnswer.csv   = hidden-style generalization check
```

Benchmark success ближе всего к показателю perfect program из статьи Foofah: нужно пройти и example, и hidden/generalization.

Dashboard показывает:

- benchmark success;
- example success;
- generalization success/fail;
- cost per success;
- p95 latency;
- attempts distribution;
- prompt strategy;

## Exact task cache

Для повторных задач используется SHA-256 cache key:

- source file hash;
- expected file hash;
- instruction hash;
- mode;
- model name;
- prompt version.

Если успешная задача с таким ключом уже есть, новая задача копирует `result_path`, `generated_code`, `explanation` и не вызывает LLM.

## Keycloak

В dev mode:

```env
KEYCLOAK_ENABLED=false
```

Приложение работает без обязательной авторизации.

Docker Compose также поднимает dev Keycloak:

```text
http://localhost:8081
```

Автоматически импортируется realm `tabular-agent` из `infra/keycloak/realm-tabular-agent.json`.

Dev-пользователи:

```text
admin / admin  -> roles: admin, user
user  / user   -> roles: user
```

Это только локальные учетные записи для дипломной демонстрации. В реальном окружении пароли нужно заменить через Keycloak Admin Console или secrets. Keycloak хранит пароли как credentials в своей БД и хэширует их; приложение не хранит пароли пользователей.

В prod-like mode для приложения:

```env
KEYCLOAK_ENABLED=true
KEYCLOAK_ISSUER=http://localhost:8081/realms/tabular-agent
KEYCLOAK_BROWSER_ISSUER=http://localhost:8081/realms/tabular-agent
KEYCLOAK_BACKCHANNEL_ISSUER=http://keycloak:8080/realms/tabular-agent
KEYCLOAK_CLIENT_ID=tabular-ai-agent
KEYCLOAK_AUDIENCE=tabular-ai-agent
SESSION_SECRET_KEY=change-me-for-real-deployments
SESSION_COOKIE_SECURE=false
```

`KEYCLOAK_BROWSER_ISSUER` используется для редиректа пользователя в браузере, а `KEYCLOAK_BACKCHANNEL_ISSUER` — для обмена `code -> token` и JWKS из контейнера приложения. Cookie-сессия приложения хранит только подписанный минимальный user context, не access token.

Защищаются административные маршруты:

```text
/auth/login
/auth/callback
/auth/logout
/auth/me
/metrics/*
/postgres/*
/benchmarks/*
/jobs/*
```

Токены Keycloak используются только auth-слоем FastAPI. Они не передаются в prompt, не сохраняются в LLM-контекст и не отправляются в LLM-клиент.

Подробный сценарий тестирования:

```text
docs/keycloak_testing.md
```

