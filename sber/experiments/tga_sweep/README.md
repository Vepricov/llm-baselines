# tga_sweep

Свип работы token-grad-averaging: много строк, одна строка = один прогон на одном GPU,
8 строк на узел, Slurm array. Список строк: `sweep_<датасет>.txt` (формат описан в его шапке).

## Запуск (рабочая машина)

Команды `giga` (sber/giga.sh) из папки проекта `/Users/23360187/gigalearn/tga`:

    giga smoke     дым: все строки по 40 шагов, минуты
    giga start     настоящий свип
    giga status    сводка
    giga resume    продолжить: готовые строки пропускаются, прерванные идут с exps/<имя>/ckpts/latest

`DATASET` выбирает файл свипа (по умолчанию shakespeare-char), `GPUS_PER_NODE` (1 по умолчанию,
8 = целые узлы), `PARTITION` (HGX), `TIME_LIMIT` (дым 30 минут), `MAX_NODES`. Пишутся перед командой:
`PARTITION=H200 giga start`.
