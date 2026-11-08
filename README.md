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

실제 요청은 `/api/chat`으로 전달하며, 수집 당시와 같은 설정(`think=false`, `stream=false`, 학습 모델의 `num_ctx`, `num_predict`, `temperature`, `top_p`, `seed`)을 사용합니다. 스트리밍은 지원하지 않습니다. 프롬프트와 답변은 로드밸런서의 SQLite에 저장하지 않습니다. 성공한 응답의 실제 출력 토큰, 처리 시간, 서버, 예측값 및 실패 여부는 `round.sqlite3`에 남습니다.

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
- `/status`의 `recent_performance`는 서버별 최근 성공 20건을 기준으로 실제 입력·출력 토큰 수의 합을 각각의 Ollama 처리 시간 합으로 나눈 초당 토큰 수를 보여줍니다. 응답 지연은 중앙 대기 시간을 포함합니다. 완료 기록이 없거나 예전 기록에 처리 시간 정보가 없으면 해당 속도는 `null`입니다. 처리속도는 서버 선택의 독립 입력이 아니며, 실제 완료 시간으로 밴딧 정책을 업데이트합니다.
- 타임아웃 시 원격 Ollama 작업이 즉시 종료되는지 보장할 수 없습니다. 외부에서 직접 실행 중인 작업과 추론 대기열까지 반영하려면 별도 계측이 필요합니다.
- 출력 예측 모델의 테스트 MAE는 약 498토큰으로, 긴 출력을 과소예측할 수 있습니다. 모델 및 생성 설정이 달라지면 예측 모델을 다시 평가해야 합니다. 단일 랜덤 분할 평가는 유사 질문이 섞였을 가능성이 있습니다.

## 검증

```bash
python -m unittest -v test_round
```
