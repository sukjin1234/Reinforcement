# vLLM 전환과 운영

기존 Ollama 경로와 별도로 vLLM 0.33.0, `Qwen/Qwen3-8B-AWQ` 리비전 `4da05a8edb55c6046cce958586c33b61da07bb79`을 사용한다. 각 서버의 vLLM은 로컬 8100번 포트에서만 수신하며, ai에서 ai1과 ai2로 접근할 때 제한된 SSH 터널 18100/28100을 사용한다. GPU 메모리 경합을 막기 위해 전환 후 Ollama 서비스는 중지한다. 시스템 서비스 설정은 `deploy/`에 있다.

## 요청 처리

`POST http://127.0.0.1:8008/chat`에 `{"messages":[{"role":"user","content":"GPU란 무엇인가요?"}]}`를 보내면 된다. 입력 토큰을 직접 전달할 필요가 없다. `/tokenize`가 Qwen 채팅 템플릿과 `enable_thinking=false`를 적용하여 서버 선택 전에 토큰 수를 계산한다. 같은 설정을 실제 요청에도 적용하며, 응답의 `token_count_matches`로 사전 계산과 완료 후 사용량을 비교할 수 있다. `category`를 생략하면 자동 분류한다. `/chat/stream`은 NDJSON 이벤트를 전송한다.

`GET /status`에는 서버별 GPU/CPU/메모리, vLLM 내부의 실행·대기 요청 수와 KV 캐시 사용률, 라우터 대기 수, 라우터가 추정한 진행 중 작업량, 서버별 정책 학습 건수가 나타난다. 완료된 요청의 누적 히스토그램을 이용한 토큰 처리속도는 서버 재시작 이후의 평균이며 순간 속도가 아니다. 처리 속도를 충분히 얻지 못하면 예상 잔여 시간은 비어 있을 수 있다. 개별 요청의 정확한 내부 prefill/decode 단계나 남은 출력 토큰은 이 지표로 확인할 수 없다.

vLLM은 별도의 `vllm.sqlite3`에 14개 입력 특성의 LinUCB 정책과 관측치를 저장한다. 기존 Ollama 정책을 재사용하지 않는다. 제한된 병렬 요청과 지속 배칭 환경에서 논문의 요청 크기 및 서버 부하를 반영하되, 논문의 Double DQN 구현은 아니다. 기존 출력 토큰 예측기는 Ollama의 Q4_K_M 결과로 학습했으므로 AWQ/vLLM에서 검증 전이라는 표시를 응답에 포함한다. 새로운 실제 데이터로 정확도를 평가하고 충분한 데이터가 쌓이면 재학습해야 한다.

## 확인 및 복구

서버 접속 후 `systemctl status round-vllm`으로 각 엔진을, ai에서 `systemctl status round-balancer round-vllm-tunnel-ai1 round-vllm-tunnel-ai2`로 라우터와 연결을 확인한다. 테스트는 `python -m unittest test_vllm test_round -q`를 사용한다. `deploy/smoke_vllm.py`는 한국어·영어·다중 대화의 사전 토큰 수를 완료 후 실제 사용량과 비교한다.

Ollama로 복구하려면 트래픽을 비우고 각 서버의 `round-vllm`을 중지한 다음 Ollama를 다시 시작한다. ai의 `round-balancer.service.d/vllm.conf`를 제거하고 systemd 설정을 다시 읽은 뒤 `round-balancer`를 다시 시작하면 기존 설정과 기존 정책 데이터베이스를 사용한다. 두 엔진을 동일 GPU에서 동시에 모델 로드하지 않는다.
