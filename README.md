# vLLM 기반 토큰 예측 로드밸런서

사용자 질문을 받으면 입력 토큰과 질문 유형을 자동으로 계산하고 예상 출력 토큰, 서버 부하와 처리 속도를 이용해 세 서버 중 하나를 선택합니다. 선택 정책은 LinUCB 문맥형 밴딧이며 실제 응답 시간과 실패 여부를 이용해 온라인으로 갱신할 수 있습니다. 논문의 Double DQN을 구현한 것은 아닙니다.

## 현재 구성

- `ai`, `ai1`, `ai2`: A10G GPU와 vLLM 0.33.0, `Qwen/Qwen3-8B-AWQ` 사용. 모델 리비전은 `4da05a8edb55c6046cce958586c33b61da07bb79`입니다.
- `ai`의 `127.0.0.1:8008`: 로드밸런서. 각 서버의 vLLM은 로컬 8100 포트, 머신 상태 수집기는 로컬 9109 포트를 사용합니다. `ai1`과 `ai2`는 제한된 SSH 터널로 접근합니다.
- 요청을 받은 뒤 vLLM `/tokenize`에서 채팅 템플릿을 적용해 입력 토큰을 계산합니다. 유형을 지정하지 않으면 마지막 사용자 질문을 분류합니다.
- vLLM `/metrics`의 실행·대기 건수, KV 캐시 사용량과 호스트 자원 상태, 입력·예상 출력 크기를 사용합니다. 서버가 비정상이거나 상태가 오래되면 선택하지 않습니다.
- 요청을 처리한 결과를 `vllm.sqlite3`에 기록합니다. 온라인 학습 여부, 서버당 동시 요청 한도와 중앙 대기열 용량은 설정 파일을 따릅니다.

기존 숫자 예측기는 Q4_K_M 형식의 기존 수집 데이터로 학습했으며, 현재 AWQ 모델에서의 정확도는 검증 중입니다. 서로 다른 양자화 방식의 결과가 같다고 가정하지 않습니다. 해당 데이터는 당시 Ollama에서 수집한 것이므로 출처를 vLLM으로 바꾸어 기록하지 않습니다.

## 실행 상태와 사용

```bash
ssh ai
systemctl is-active round-balancer round-vllm round-vllm-tunnel-ai1 round-vllm-tunnel-ai2
curl -s http://127.0.0.1:8008/status
curl -s http://127.0.0.1:8008/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"GPU 서버의 역할을 간단히 설명해줘."}]}'
```

`/chat` 응답의 `server_id`, `input_tokens`, `category`, `predicted_output_tokens`, `vllm_response.usage`를 확인합니다. `token_count_matches`는 사전 계산과 실제 사용량의 일치 여부입니다. `/chat/stream`은 `routed`, `chunk`, `complete`, `error` NDJSON 이벤트를 제공합니다. `/classify`는 `{"text":"질문"}`으로 유형만 분류합니다. 현재 스트리밍의 첫 답변 조각 시간과 조각 간 간격은 정확한 토큰 단위 시간과 다를 수 있습니다.

## 설정과 실행

```bash
python -m pip install -r requirements.txt
cp config.vllm.example.json config.vllm.json
# predictor_path와 category_model_path를 실제 파일 경로로 설정
ROUND_CONFIG=config.vllm.json python -m uvicorn balancer:app --host 127.0.0.1 --port 8008
```

세 서버 예시는 `config.three.example.json`, 단일 서버 예시는 `config.ai.example.json`입니다. `balancer:app`과 `vllm_balancer:app`은 같은 앱입니다. 실제 설치된 systemd 서비스, 터널, 고정 모델 설정은 `deploy/`에 있습니다. 서버 주소와 인증은 운영 환경에 맞게 설정하고 추론 API를 외부에 직접 공개하지 마세요. 여러 라우터 프로세스가 같은 정책 SQLite를 동시에 갱신하지 않도록 합니다.

`/status`에는 vLLM 내부 실행·대기 수, 캐시 사용량, 호스트 상태, 라우터 중앙 대기열, 자체 전달 요청의 예상 작업량 및 학습 관측 수가 표시됩니다. 완료 요청의 누적 히스토그램 기반 처리속도는 순간 속도가 아니며, 각 요청의 정확한 단계나 남은 출력 토큰을 나타내지는 않습니다. 속도를 계산할 근거가 없으면 추정 시간은 비어 있습니다.

## 검증과 실험 도구

```bash
python -m unittest test_round test_vllm -q
python collect_metrics.py --config config.vllm.json --once
python deploy/smoke_vllm.py http://127.0.0.1:8100
python deploy/smoke_router.py http://127.0.0.1:8008
python probe_concurrency.py --url http://127.0.0.1:8100 --output concurrency.json
```

동시성 시험은 실행 중인 인스턴스에 실제 요청을 보내므로 부하와 다른 작업의 영향을 고려하세요. 출력 파일은 기존 결과를 덮어쓰지 않습니다. `calibrate_squad.py`, `evaluate_squad.py`, `experiment.py`, `vllm_train_eval.py`, `vllm_compare.py`도 vLLM 환경을 사용합니다. 실험 정책 비교는 별도 SQLite를 사용하며, 훈련 결과를 운영 DB에 적용하는 `apply_policy.py`는 14차원 vLLM 정책만 검사합니다.

2026-11-09의 소규모 vLLM 시험에서 별도 영어 질문 24건의 출력 토큰 예측 MAE는 약 640토큰이었습니다. 추가 24건의 단일 비교에서는 평균 응답 시간이 라운드 로빈 1.901초, 예상 작업량 기반 1.877초, 고정 LinUCB 1.849초였습니다. 이 표본과 단일 반복으로 정책 우위를 결론낼 수 없습니다. 자세한 내용은 [진행현황.md](진행현황.md), [개선 평가](개선_평가_20261109.md), [논문 비교](논문_비교분석.md), [운영 문서](vLLM_운영.md)를 참고하세요.
