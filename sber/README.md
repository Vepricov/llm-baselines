# sber — запуск экспериментов llm-baselines на кластере GigaLearn

Эта папка превращает запуск обучения в одну команду с рабочей машины.
Всё остальное (код, данные, чекпоинты) живёт на кластере.

---

## 1. Кто где

```
 личный мак (правки кода)          рабочая машина (только sh)         кластер
 ────────────────────────          ──────────────────────────          ───────
 git push в ветку sber    ───────►  bash run.sh <эксперимент>  ───────►  sbatch
                                    (curl тянет архив, ssh ставит)      srun + контейнер
                                    ◄─────── результаты (scp) ◄───────  results/
```

Три машины, три роли:

- личный мак — там пишется код и делается `git push`;
- рабочая машина (сбер-мак) — только запускает `run.sh`, ничего не хранит и не правит;
- кластер — считает; всё, что нужно для повтора, лежит в каталоге прогона.

---

## 2. Разовая настройка

### 2.1. Ключ на кластер (один раз)

С рабочей машины:

```bash
ssh-keygen -t ed25519 -f $HOME/.ssh/id_ed25519 -N ""
ssh -o PreferredAuthentications=password aseveprikov@mkristo-dev01 \
  'mkdir -p $HOME/.ssh && chmod 700 $HOME/.ssh && cat >> $HOME/.ssh/authorized_keys && chmod 600 $HOME/.ssh/authorized_keys' \
  < $HOME/.ssh/id_ed25519.pub
ssh -o BatchMode=yes aseveprikov@mkristo-dev01 hostname
```

Последняя команда должна напечатать имя узла без запроса пароля.

### 2.2. Конфиг (один раз)

```bash
mkdir -p /Users/23360187/gigalearn
printf 'GIGALEARN_HOST=aseveprikov@mkristo-dev01\nGIGALEARN_IMAGE=/external/nfs/01-home/amaksimenko/docker_images/enroot_images/llm-foundry-train+2025_08_06_commit0fbe4534.sqsh\n' \
  > /Users/23360187/gigalearn/config.env
```

`GIGALEARN_IMAGE` — путь к контейнеру (sqsh) с torch. Если образ поменялся, правится только эта строка.

### 2.3. Постоянная база на кластере (один раз)

Тяжёлое (tiktoken, его словарь, данные) держим отдельно от прогонов:

```
$HOME/gigalearn/base/pylibs            tiktoken для python 3.12 (linux)
$HOME/gigalearn/base/tiktoken_cache    словарь gpt2
$HOME/gigalearn/base/data/fineweb-100BT/train.bin + val.bin
```

Один раз перенести из старой папки, если она есть:

```bash
ssh aseveprikov@mkristo-dev01 'mkdir -p $HOME/gigalearn/base && cp -r $HOME/llmb/pylibs $HOME/llmb/tiktoken_cache $HOME/llmb/data $HOME/gigalearn/base/ 2>/dev/null; du -sh $HOME/gigalearn/base/*'
```

---

## 3. Запуск: одна команда

```bash
curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/sber/sber/run.sh | bash -s -- mlmc_pretrain
```

Что произойдёт по шагам:

1. curl скачивает архив ветки `sber` с GitHub;
2. архив по ssh уезжает на кластер в `$HOME/gigalearn/repo`;
3. `cluster_run.sh` создаёт `$HOME/gigalearn/runs/mlmc_pretrain_<дата-время>/`,
   пишет туда README прогона, подставляет пути, ставит `sbatch` и ждёт;
4. готовые результаты копируются обратно в
   `/Users/23360187/gigalearn/runs/mlmc_pretrain_<дата-время>/`.

Формат имени: `ДД-ММ-ГГГГ_ЧЧ-ММ-СС`. Прогоны не перезатираются никогда.

### Примеры

```bash
# быстрый дым на shakespeare, 200 шагов
curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/sber/sber/run.sh \
  | DATASET=shakespeare-char ITERATIONS=200 bash -s -- mlmc_pretrain

# сравнение оптимизаторов на FineWeb
curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/sber/sber/run.sh \
  | OPTIMIZER=soap LR=1e-3 bash -s -- mlmc_pretrain

# маленькая модель, чтобы быстро перебрать идеи
curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/sber/sber/run.sh \
  | N_EMBD=384 N_HEAD=6 N_LAYER=6 ITERATIONS=5000 bash -s -- mlmc_pretrain
```

---

## 4. Параметры

Передаются переменными перед `bash` и попадают в README прогона.

| переменная | по умолчанию | смысл |
|---|---|---|
| DATASET | fineweb | fineweb, finewebedu, shakespeare-char |
| OPTIMIZER | adamw | adamw, soap, muon, dykaf и другие из src/optim |
| LR | 3e-3 | скорость обучения |
| ITERATIONS | 20000 | число шагов |
| WARMUP | 2000 | разогрев |
| BATCH | 256 | размер батча |
| SEQ | 512 | длина контекста |
| N_EMBD | 768 | ширина модели |
| N_HEAD | 12 | голов внимания |
| N_LAYER | 12 | слоёв |

Служебные: `GIGALEARN_HOST`, `GIGALEARN_IMAGE`, `GIGALEARN_REPO`, `GIGALEARN_BRANCH`.

---

## 5. Результаты

В каталоге прогона (и на кластере, и на рабочей машине):

```
runs/mlmc_pretrain_25-10-2026_20-14-03/
  README.md              что за идея, параметры, узел, sha кода
  idea.md                идея в чистом виде, как её написали
  results/
    status.txt           job id, состояние, код выхода, время
    report_<job>.txt     отчёт: sacct, хвост лога, хвост stderr
    metrics.csv          iter, train_loss, lr, val_loss, val_pp, val_acc
    training_curve.svg   график (открывается в браузере, зависимостей нет)
    training_curve.png   то же в png, если в контейнере есть matplotlib
```

Как читать:

- `metrics.csv` — таблица; по ней строятся любые свои графики;
- `training_curve.svg` — тот же график, открыть двойным кликом;
- `report` нужен только чтобы понять, почему задача упала.

Чекпоинты остаются на кластере в `exps/<имя эксперимента>/`.

---

## 6. Как добавить новый эксперимент

1. Создать папку `sber/experiments/<имя>/`.
2. Положить три файла:

`idea.md` — зачем эксперимент, что проверяем, что считаем успехом.
`train_slurm.sh` — директива sbatch и запуск контейнера, с плейсхолдерами
`@LOG_DIR@` и `@IMAGE@` (их подставит cluster_run.sh).
`slurm_startup.sh` — torchrun внутри контейнера; параметры берутся из переменных
окружения с дефолтами, как в `mlmc_pretrain`.

3. Скопировать пару `mlmc_pretrain` как образец:

```bash
cp -r sber/experiments/mlmc_pretrain sber/experiments/<имя>
# поправить --job-name, --run_prefix, idea.md, дефолты
git add sber/experiments/<имя>
git commit -m "feat(sber): эксперимент <имя>"
git push
```

4. Запустить:

```bash
curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/sber/sber/run.sh | bash -s -- <имя>
```

---

## 7. Воспроизводимость

В README прогона фиксируются:

- ветка и sha256 архива кода;
- все параметры запуска;
- дата и время;
- узел и job id.

Этого достаточно, чтобы повторить прогон: тот же sha, те же параметры, тот же датасет.
Чекпоинты и `exps/` остаются на кластере, поэтому повтор возможен и с середины.

---

## 8. Git и результаты

Результаты — обычные текстовые файлы, их можно коммитить:

```bash
cd /Users/23360187/gigalearn
git init                      # один раз
git add runs
git commit -m "run: mlmc_pretrain 25-10-2026_20-14-03"
```

Что стоит коммитить: `README.md`, `status.txt`, `metrics.csv`, `training_curve.svg`,
`report_*.txt`. Чего не стоит: `exps/` (чекпоинты живут на кластере), `data/`, `pylibs/`.

---

## 9. Диагностика

| симптом | причина | что делать |
|---|---|---|
| `error: задай GIGALEARN_HOST` | нет config.env | раздел 2.2 |
| `Permission denied (publickey)` | ключ не установлен | раздел 2.1 |
| `WARNING: нет датасета .../fineweb-100BT/train.bin` | датасет не занесён | раздел 10 |
| `No module named tiktoken` | нет pylibs в base | раздел 2.3 |
| `CUDA-capable device(s) is/are busy` | в srun не запрошены GPU | в train_slurm.sh должен быть `--gpus=8` |
| `plot_metrics skipped/failed` | не найден лог | проверить `@LOG_DIR@` и LOG_FILE |
| задача в `PENDING` долго | нет свободных узлов | `sinfo -s` на кластере |
| `report` пустой | задача ещё не закончилась | дождаться или `squeue -u aseveprikov` |

Полезное на кластере:

```bash
sinfo -s                      # свободные узлы
squeue -u aseveprikov         # свои задачи
scancel <job id>              # снять задачу
sacct -j <job id> --format=JobID,State,ExitCode,Elapsed
tail -50 $HOME/gigalearn/runs/<прогон>/logs/log_<job>.log
```

---

## 10. Датасеты

Ожидаемый формат — тот, что читает llm-baselines: `np.uint16` memmap.

```
$HOME/gigalearn/base/data/fineweb-100BT/train.bin
$HOME/gigalearn/base/data/fineweb-100BT/val.bin
```

Кластер интернета не имеет, поэтому датасет заносится заранее официальным каналом
(файловый перекладчик). После переноса достаточно положить файлы в этот каталог —
код ищет их по имени `fineweb-100BT` при `DATASET=fineweb`.

---

## 11. Правила для агентов

1. Ничего не писать в `$HOME/gigalearn/repo` на кластере руками: каталог перезаписывается
   при каждом запуске из GitHub.
2. Не менять `base/` без явной просьбы: там данные и зависимости, общие для всех прогонов.
3. Новый эксперимент — новая папка в `sber/experiments/`, старые не править.
4. Никогда не удалять каталоги `runs/`: это история.
5. Секреты (токены, пароли) в репозиторий не попадают; хост и образ живут в config.env
   на рабочей машине.
6. Перед запуском дорогого прогона сделать дешёвый дым: `DATASET=shakespeare-char ITERATIONS=50`.
7. Если задача упала, сначала читать `results/report_*.txt`, потом лог.

---

## 12. Ограничения

- Ветка `sber` публичная: внутренние имена хостов и пути кластера в неё не коммитим,
  они живут в `config.env`.
- Кластер без интернета: всё, что нужно, должно быть на NFS заранее.
- Один узел = 8 GPU. Многоузловые запуски потребуют правки `torchrun` в `slurm_startup.sh`.
