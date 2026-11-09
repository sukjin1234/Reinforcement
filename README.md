# 서버 상태 수집 및 강화학습 로드밸런서

각 서버의 수집기가 CPU, 메모리, NVIDIA GPU 사용량을 제공하고, 로드밸런서가 주기적으로 수집합니다. Ollama의 모델명과 digest를 확인해 다른 모델, 장애, 오래된 정보, 용량 초과 서버는 선택하지 않습니다. 사용자가 전달한 입력 토큰 수와 질문 유형으로 기존 출력 토큰 예측 모델을 실행한 뒤, 서버 상태와 예측값을 문맥으로 사용해 LinUCB 문맥형 밴딧이 서버를 선택합니다. 실제 응답 시간 및 실패 여부로 선택 정책을 온라인 갱신하고 SQLite에 저장합니다.

## 세 서버 구성

`ai`의 `~/round`에 로드밸런서를 설치했고 `127.0.0.1:8008`에서 실행합니다. `ai`, `ai1`, `ai2`의 수집기와 Ollama는 각 서버의 로컬 주소에 바인딩합니다. `ai`의 SSH 터널이 `ai1`, `ai2`의 수집기와 Ollama를 연결하며, 공개 네트워크에 서비스 포트를 열 필요가 없습니다. 모든 프로세스와 터널은 systemd가 관리합니다. 기존 예측 모델과 가상환경은 `~/qwen3_8b_100k_questions_bundle`을 사용합니다.

```bash
ssh ai
cd ~/round
curl -s http://127.0.0.1:8008/status
```

서비스 상태를 확인하거나 재시작하려면 다음 명령을 사용하세요.

```bash
systemctl status round-agent round-balancer round-tunnel-ai1 round-tunnel-ai2
sudo systemctl restart round-balancer
```

예측과 응답 요청:

```bash
curl -s http://127.0.0.1:8008/chat \
  -H 'Content-Type: application/json' \
  -d '{"input_tokens":38,"messages":[{"role":"user","content":"Reply with one word: hello."}]}'
```

위 38은 동작 확인을 위한 예시입니다. 실사용 시 **Qwen/Ollama의 채팅 템플릿을 포함한 실제 입력 토큰 수**를 호출자가 제공해야 합니다. 토큰 수는 자동 산출하지 않습니다. 질문 유형을 생략하면 마지막 사용자 메시지에서 자동 분류합니다. `category`를 직접 지정하면 해당 유형을 사용합니다. 응답에는 선택된 유형, 자동 분류 점수, 선택된 서버, 예측값, 학습 범위 밖 여부, Ollama 응답과 실제 토큰 수가 포함됩니다. 유형 목록은 기존 예측 모델의 README를 참고하세요.

분류 결과만 확인하려면 `POST /classify`에 `{"text":"요약해줘"}`를 전달합니다. 분류기는 첫 1,000자만 사용하며 여러 작업이 섞인 질문에 단일 유형을 출력합니다. 점수는 보정된 확률이나 정확도 보증이 아닙니다. 10만 개의 합성 템플릿 질문으로 학습했으므로 실제 사용자 질문에서 오류가 날 수 있습니다. 불확실한 결과는 `category`를 수동 지정하세요.

분류기 학습 및 검증:

```bash
cd ~/round
~/qwen3_8b_100k_questions_bundle/.token_predict_venv/bin/python train_category.py \
  --data ~/qwen3_8b_100k_questions_bundle/questions_100000.jsonl
```

문자 단위 TF-IDF와 선형 분류기를 사용합니다. 평가 데이터는 주제 영역을 분리하지만 템플릿 자체는 겹칠 수 있습니다. `category_report.json`에 평가를 저장한 후 전체 데이터로 분류기를 재학습합니다. `category_model.joblib`은 동일한 Python·scikit-learn 환경에서 사용하고 신뢰할 수 있는 파일만 불러오세요.

이번 주제 영역 분리 평가에서는 19,646개 합성 질문의 정확도가 99.995%였지만, 유사한 생성 템플릿이 학습·평가에 반복되어 일반화 성능을 나타내지 않습니다. 별도로 작성한 자연어 질문 8개에서는 기대한 유형 5개가 맞고 3개가 달랐습니다. 이 작은 예시는 정식 평가가 아니며, 실제 사용 전 별도로 라벨링된 실제 질문으로 검증해야 합니다.

실제 요청은 `/api/chat`으로 전달하며, 수집 당시와 같은 설정(`think=false`, 학습 모델의 `num_ctx`, `num_predict`, `temperature`, `top_p`, `seed`)을 사용합니다. `/chat`은 완료된 응답을 한 번에 반환하고, `/chat/stream`은 NDJSON 이벤트를 전송합니다. 프롬프트와 답변은 로드밸런서의 SQLite에 저장하지 않습니다. 성공한 응답의 실제 출력 토큰, 처리 시간, 서버, 예측값 및 실패 여부는 `round.sqlite3`에 남습니다.

```bash
curl -N -s http://127.0.0.1:8008/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"input_tokens":38,"messages":[{"role":"user","content":"Explain a network in one sentence."}]}'
```

스트리밍 이벤트는 `routed`, `chunk`, `complete` 또는 `error`입니다. `ttft_seconds`는 로드밸런서가 요청을 받은 뒤 처음으로 비어 있지 않은 답변 조각을 관측할 때까지의 시간으로 중앙 대기를 포함합니다. `mean_inter_chunk_seconds`는 답변 조각 사이의 평균 시간입니다. 조각은 한 토큰과 같다고 보장할 수 없으므로 토큰 간 시간(TBT)과 구분해야 합니다. 스트리밍 응답이 시작된 이후 오류가 나면 HTTP 상태 대신 NDJSON `error` 이벤트로 알려줍니다.

출력 구간 예측은 현재 응답의 `bucket_prediction`에 **실험용 참고값**으로 제공하며 서버 선택에는 쓰지 않습니다. `0-255`, `256-767`, `768-1535`, `1536+` 네 구간을 사용합니다. 독립 영어 질문 24건에서는 15건을 맞혀, 같은 평가 집합의 사후 최빈 구간(17건)보다 낮았습니다. 분류 점수는 보정된 확률이 아닙니다.

## 서버 추가

1. 각 서버에 이 폴더의 `agent.py` 및 의존성을 설치하고 `uvicorn agent:app --host <사설주소> --port 9109`로 실행합니다. NVIDIA 드라이버와 `nvidia-smi`가 필요합니다.
2. 로드밸런서에 `config.example.json`을 참고해 각 서버의 `id`, 수집기 URL, Ollama URL, `max_inflight`를 설정합니다. 모든 서버에 지정된 동일 모델과 digest가 있어야 합니다.
3. 설정을 바꾼 후 로드밸런서를 재시작하고 `/status`에서 각 서버를 확인합니다. 설정 파일, 학습 DB와 예측 모델을 읽을 수 있는 계정으로 실행해야 합니다.

수집기는 `METRICS_TOKEN` 환경변수가 설정되면 `Authorization: Bearer <토큰>`을 요구합니다. 로드밸런서에도 같은 환경변수를 지정하세요. 현재 서버 간 연결은 전용 키와 제한된 포트만 허용하는 SSH 터널을 사용합니다. 로드밸런서 API 자체에는 인증이 없으므로 외부에 직접 공개하지 마세요.

독립적인 수집 기록이 필요하면 다음 명령을 실행합니다. 기본적으로 계속 수집하며 `--once`를 추가하면 한 번만 수집합니다.

```bash
cd ~/round
~/qwen3_8b_100k_questions_bundle/.token_predict_venv/bin/python collect_metrics.py --config config.json --output metrics.jsonl
```

## 선택 및 평가의 한계

- 세 대 연결 후 충분한 실제 요청을 통해 라운드 로빈 등과 같은 조건에서 지연 시간과 실패율을 비교해야 합니다. 초기 정책은 학습 전 상태이며 최적 서버를 보장하지 않습니다.
- 보상은 성공 시 `-min(log(1 + 응답초/10), 5)`, 실패 시 `-5`입니다. 응답 시간이 짧을수록 높은 보상입니다. 각 서버별 선형 모델과 탐색 보너스를 사용합니다.
- 로드밸런서에는 최대 100건의 FIFO 대기열이 있으며 서버가 바쁘거나 사용 불가능할 때 최대 120초 기다립니다. 대기열이 가득 차거나 시간이 초과되면 503을 반환합니다. `/status`의 `queue.waiting`은 중앙 대기열 건수이며, 아직 서버가 배정되지 않아 서버별 대기 건수로 나눌 수 없습니다. `queue.rejected`, `queue.timed_out`은 프로세스 시작 이후의 횟수입니다. 재시작하면 대기 중인 요청과 이 카운터는 사라집니다.
- `max_inflight`와 전달된 요청 수는 이 로드밸런서 한 프로세스에서만 추적합니다. 다른 클라이언트의 Ollama 요청이나 Ollama 자체 내부 대기열은 집계하지 못합니다. 같은 SQLite로 여러 로드밸런서 프로세스를 띄우지 마세요.
- `/status`의 `recent_performance`는 서버별 최근 성공 20건을 기준으로 실제 입력·출력 토큰 수의 합을 각각의 Ollama 처리 시간 합으로 나눈 초당 토큰 수를 보여줍니다. 응답 지연은 중앙 대기 시간을 포함합니다. 완료 기록이 없거나 예전 기록에 처리 시간 정보가 없으면 해당 속도는 `null`입니다. 처리속도와 측정값 유무는 서버 선택의 상태에 포함하고 실제 완료 시간으로 밴딧 정책을 업데이트합니다.
- `input_tokens_per_second`는 기존 정책의 호환성을 위해 캐시 토큰을 포함한 유효 처리량으로 유지합니다. 새 `uncached_input_tokens_per_second`는 Ollama가 보고한 캐시 토큰을 빼고 계산합니다. 캐시 정보가 없는 이전 기록과 캐시되지 않은 입력이 32토큰 미만인 요청은 고정 오버헤드의 영향을 줄이기 위해 새 지표에서 제외합니다. 유효 표본 수는 `uncached_rate_samples`입니다. 기존 강화학습 정책은 여전히 기존 유효 처리량을 사용합니다.
- `/status`의 `estimated_work`는 로드밸런서가 전달한 진행 중 요청에 대해 토큰 예측과 캐시 제외 입력·출력 처리속도로 계산한 예상 처리시간 및 잔여 시간을 표시합니다. 속도가 없으면 `null`이고 예상 시간을 초과하면 `overdue=true`입니다. 시간 기반 근사치이며 모델 적재 시간·캐시 이득·Ollama 내부 대기열이나 다른 클라이언트의 작업은 포함하지 않습니다. 실제 미생성 토큰을 추적하지는 않습니다.
- 타임아웃 시 원격 Ollama 작업이 즉시 종료되는지 보장할 수 없습니다. 외부에서 직접 실행 중인 작업과 추론 대기열까지 반영하려면 별도 계측이 필요합니다.
- 출력 예측 모델의 테스트 MAE는 약 498토큰으로, 긴 출력을 과소예측할 수 있습니다. 모델 및 생성 설정이 달라지면 예측 모델을 다시 평가해야 합니다. 단일 랜덤 분할 평가는 유사 질문이 섞였을 가능성이 있습니다.

## 검증

```bash
python -m unittest -v test_round
```

## 강화학습 실험

`experiment.py`는 별도 로드밸런서 프로세스와 SQLite DB로 LinUCB를 학습한 후, 학습을 고정한 LinUCB와 라운드 로빈을 같은 평가 질문으로 비교합니다. 6건씩 동시 전송하고 완료될 때까지 기다린 뒤 다음 묶음을 보냅니다. 성공률, 평균·중앙값·95백분위 지연, 대기 시간, 처리량을 측정합니다. 서비스 중인 서버에 부하가 생기므로 다른 실험과 겹치지 않게 실행하세요.

```bash
cd ~/round
PY="$HOME/qwen3_8b_100k_questions_bundle/.token_predict_venv/bin/python"
"$PY" experiment.py \
  --questions ~/qwen3_8b_100k_questions_bundle/questions_100000.jsonl \
  --results ~/qwen3_8b_100k_questions_bundle/results_10000.jsonl \
  --config config.json --output experiments/new-run \
  --train-count 120 --test-count 36 --batch-size 6
```

평가에는 학습용과 다른 주제 영역을 사용하지만, 합성 데이터 생성 템플릿은 공유될 수 있습니다. 수집 결과의 실제 출력이 100~700토큰인 질문만 선택하므로 전체 운영 트래픽을 대표하지 않습니다. 출력 토큰 예측 모델도 같은 질문 자료로 학습했습니다. 평가 순서는 LinUCB 다음 라운드 로빈으로 고정돼 있어 시점 차이가 영향을 줄 수 있습니다. 실험 DB와 각 요청의 ID·지연 기록, 보고서는 지정한 폴더에 저장되며 질문·답변 원문은 저장하지 않습니다. 실험 정책을 운영 DB에 적용하려면 별도 적용 단계가 필요합니다.

현재 운영 정책은 `train_20261109_02`의 120건 학습 결과를 적용했습니다. 정책 적용 전 DB와 설정은 해당 실험 폴더에 백업돼 있습니다. 평가에는 `bandit_alpha=0`으로 학습을 고정한 정책을 사용했고, 운영 설정에서도 `bandit_alpha=0`을 사용하되 완료된 요청으로 온라인 학습을 계속합니다. 따라서 이후 운영 정책은 평가 당시와 달라집니다. 평가 결과는 [진행현황.md](진행현황.md)와 `reports/rl_20261109.json`에 기록했습니다.

독립 평가와 동시성 검토의 방법·한계·결과는 [개선_평가_20261109.md](개선_평가_20261109.md)에 기록했습니다. 운영 정책과 Ollama의 서버별 동시 처리 한도는 실험 전과 동일합니다.

독립 평가를 재현할 때는 기존 실행 결과를 덮어쓰지 않는 새 경로를 사용합니다. 자료는 [SQuAD v2 개발 세트](https://rajpurkar.github.io/SQuAD-explorer/)이며 CC BY-SA 4.0입니다. 실제 질문·응답 텍스트는 저장소에 커밋하지 않습니다.

```bash
cd ~/round
PY="$HOME/qwen3_8b_100k_questions_bundle/.token_predict_venv/bin/python"
"$PY" prepare_squad.py --data data/squad_dev_v2.json --output data/manifest_new.json --count 24 --seed 42
"$PY" calibrate_squad.py --manifest data/manifest_new.json --output data/calibration_new.jsonl --report data/calibration_new.json
"$PY" evaluate_squad.py --manifest data/manifest_new.json --calibration data/calibration_new.jsonl --output experiments/independent_new --repeats 2 --batch-size 6
```

평가 스크립트는 별도 포트와 SQLite DB를 사용하지만 실제 세 GPU에 부하를 발생시키므로, 운영 요청이 없는 시간에 실행해야 합니다. 평가 질문의 입력 토큰 수는 먼저 Qwen 응답으로 확인한 값이며 출력 길이로 질문을 거르지 않습니다. 평가 전에는 별도 준비 질문으로 속도를 측정합니다. 출력을 직접 생성하는 과정에서 같은 질문이 미리 처리되므로 캐시 효과가 남을 수 있습니다.
