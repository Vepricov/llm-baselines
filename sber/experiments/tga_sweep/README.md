# tga_sweep

Свип работы token-grad-averaging: много строк, одна строка = один прогон на одном GPU,
8 строк на узел, Slurm array. Список строк: `sweep_<датасет>.txt` (формат описан в его шапке).

## Запуск (рабочая машина)

    # дым: все строки по 40 шагов, минуты
    curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/tga/sber/run.sh | SMOKE=1 bash -s -- tga_sweep
    # настоящий свип
    curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/tga/sber/run.sh | bash -s -- tga_sweep

`DATASET` выбирает файл свипа (по умолчанию shakespeare-char), `MAX_NODES` ограничивает
число одновременно занятых узлов. Команда ставит массив и сразу возвращается.

## Сводка

    curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/tga/sber/collect.sh | bash

Печатает summary.md и кладёт `~/gigalearn/runs/<свип>_summary.tar.gz`.

## Продолжить

    curl -sL .../run.sh | RESUME_RUN=<дата-время свипа> bash -s -- tga_sweep

Готовые строки (runs/<имя>/DONE) пропускаются, прерванные продолжают с exps/<имя>/ckpts/latest.
