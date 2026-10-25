# mlmc_pretrain

Базовый претрейн llama на FineWeb через llm-baselines на GigaLearn (1 узел, 8 GPU).
Цель: воспроизводимая база, на которой дальше проверяются идеи прекондиционирования.

## Конфигурация по умолчанию

- dataset: fineweb (FineWeb sample-100BT, уже токенизированный, лежит в `~/gigalearn/base/data/fineweb-100BT`)
- model: llama, n_embd 768, n_head 12, n_layer 12 (около 124M параметров)
- optimizer: adamw, lr 3e-3, warmup 2000
- batch: 256 x 512, acc_steps 1
- iterations: 20000
- eval: каждые 500, лог: каждые 20
- partition: HGX, 1 узел, 8 GPU, лимит 24 часа

## Что можно менять без правки файлов

Переменными окружения при запуске:

- DATASET (fineweb, finewebedu, shakespeare-char)
- OPTIMIZER (adamw, soap, muon, dykaf, ...)
- LR, ITERATIONS, WARMUP, BATCH, SEQ, N_EMBD, N_HEAD, N_LAYER

Пример: `DATASET=shakespeare-char ITERATIONS=500 bash run.sh mlmc_pretrain`

## Результаты

Каждый запуск получает свою папку `~/gigalearn/runs/mlmc_pretrain_<timestamp>/results`
на кластере и копию в `~/gigalearn/runs/mlmc_pretrain_<timestamp>` на рабочей машине:
отчёт, metrics.csv, training_curve.svg (и png, если есть matplotlib), status.txt.

## Статус

Ожидает датасет FineWeb на кластере (9.9 ГБ, перенос через перекладчик).
