# CacheBlend Fig.12 Reproduction Set (MuSiQue)

[CacheBlend (EuroSys '24)](https://arxiv.org/abs/2405.16444) Figure 12에서 보고한
실험을 재현하기 위해 만든 프롬프트 세트입니다. **MuSiQue** 멀티홉 QA 데이터에서
각 질문마다 후보 문단 20개 중 **L2 거리 기반 top-6**을 골라, **순서를 무작위로
섞어서** "system + 6 docs + question" 단일 프롬프트를 만듭니다.

## 1. 동기 — Figure 12가 무엇을 보는가

CacheBlend의 Fig.12는 RAG 형태의 멀티-document 입력에서 **KV 캐시 재사용** 효과를
검증합니다. 핵심은:

- 같은 문서들을 서로 다른 질문/순서로 재사용하는 시나리오
- 따라서 (a) 다수 문서를 컨텍스트에 모아 붙이는 형태, (b) 문서가 질문마다 부분적으로
  공유되는 분포, (c) 위치(prepix vs. middle)가 영향을 줄 수 있도록 **문서 순서가
  결정적이지 않음** — 이 세 조건을 만족하는 입력이 필요합니다.

이 디렉터리의 `prompts.jsonl`은 위 세 조건을 만족하도록 구성된 200개의 입력입니다.

## 2. 데이터 출처

- **MuSiQue v1.0** (StonyBrookNLP) — `musique_ans_v1.0_dev.jsonl`의 앞 200개.
- 라이선스: CC BY 4.0. 원본 https://github.com/StonyBrookNLP/musique
- 본 저장소에서는 같은 mydata 레포의 [`../musique/`](../musique) 디렉터리에 있는
  파일을 사용합니다.

각 MuSiQue 레코드는 한 multi-hop 질문에 대해 20개 paragraph 후보를 가집니다.
이 중 정답 추론에 필요한 paragraph는 `is_supporting=true`로 표시되어 있습니다
(보통 2개, MuSiQue-Ans dev의 경우 모두 2개).

## 3. 파이프라인

```
              MuSiQue dev (200 questions, 20 paragraphs each)
                              │
                              ▼
       SentenceTransformer all-mpnet-base-v2 (MPS/CPU)
        ├─ embed: question                    → q_emb (768-d)
        └─ embed: "{title}. {paragraph_text}" → p_emb (768-d) × 20
                              │
                              ▼
                ‖p_emb − q_emb‖₂   (L2 distance, ascending)
                              │
                              ▼
                  top-6 paragraphs by smallest L2
                              │
                              ▼
              random.Random(42).shuffle(...)   ← 결정적 셔플
                              │
                              ▼
   "{system}\n\nDocument 1:\n{d1}\n\n…\n\nDocument 6:\n{d6}\n\n
    Question: {q}\nAnswer:"
```

**구체적인 선택지:**

| 항목 | 값 |
|---|---|
| 임베딩 모델 | `sentence-transformers/all-mpnet-base-v2` (768-d) |
| 임베딩 정규화 | 끄고 raw 벡터로 L2 (CacheBlend 평가 코드 관행) |
| 거리 | L2 distance — `np.linalg.norm(p − q, axis=1)` |
| 후보 풀 | 한 질문의 20개 paragraph (MuSiQue 자체 후보) |
| Top-K | 6 |
| 셔플 | `random.Random(seed)` 단일 인스턴스, seed 기본값 42 |
| 디바이스 | Apple Silicon이면 MPS 자동 |
| 시스템 프롬프트 | `"You are a helpful assistant. Use the following documents to answer the question."` (CacheBlend 레포 톤) |
| Paragraph 직렬화 | `"{title}. {paragraph_text}"` — 임베딩과 프롬프트 모두 동일 |

## 4. 디렉터리 구조

```
cacheblend_fig12/
├── README.md          ← 본 문서
├── build_prompts.py   ← 빌드 스크립트
├── requirements.txt   ← Python 의존성
└── prompts.jsonl      ← 결과 (200줄, ~1.2 MB)
```

## 5. `prompts.jsonl` 레코드 스키마

한 줄 = 하나의 prompt record (JSON object).

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | string | MuSiQue 원본 id (예: `2hop__460946_294723`) |
| `question` | string | 멀티-hop 질문 |
| `answer` | string | 정답 (MuSiQue-Ans dev) |
| `answer_aliases` | list[string] | 정답 동의어 |
| `supporting_idxs` | list[int] | MuSiQue가 표시한 정답 근거 paragraph idx (참조용) |
| `selected_idxs_by_l2` | list[int] (len=6) | L2 거리 오름차순으로 뽑힌 paragraph idx |
| `selected_l2_distances` | list[float] (len=6) | 위 6개의 L2 거리 |
| `shuffled_idxs` | list[int] (len=6) | **프롬프트 안에서의 최종 등장 순서**의 idx |
| `supporting_recall_at_k` | float | `|supporting ∩ selected| / |supporting|` (sanity 지표) |
| `prompt_parts.system` | string | 시스템 프롬프트 |
| `prompt_parts.docs` | list[string] (len=6) | 셔플된 순서로 정렬된 문서 텍스트 |
| `prompt_parts.question` | string | `question`과 동일 |
| `prompt` | string | 모델에 그대로 넣을 수 있는 최종 단일 문자열 |

> `prompt_parts.docs[i]`의 idx ↔ `shuffled_idxs[i]`가 1:1 대응됩니다. 예를 들어
> KV 캐시 재사용 실험에서 문서 단위로 캐시 키를 만들고 싶다면 `selected_idxs_by_l2`로
> 식별하면 됩니다(셔플 전 안정적 ID).

### 한 레코드 예시

```json
{
  "id": "2hop__460946_294723",
  "question": "Who is the spouse of the Green performer?",
  "answer": "Miquette Giraudy",
  "answer_aliases": [],
  "supporting_idxs": [5, 10],
  "selected_idxs_by_l2": [19, 5, 12, 1, 18, 11],
  "selected_l2_distances": [1.062, 1.084, 1.132, 1.153, 1.203, 1.207],
  "shuffled_idxs": [1, 5, 12, 18, 19, 11],
  "supporting_recall_at_k": 0.5,
  "prompt_parts": { "system": "...", "docs": ["...", "..."], "question": "..." },
  "prompt": "You are a helpful assistant. Use the following documents to answer the question.\n\nDocument 1:\n...\n\n...\n\nQuestion: ...\nAnswer:"
}
```

## 6. 데이터셋 통계

200건 기준:

| 지표 | 값 |
|---|---:|
| 질문 수 | 200 |
| Top-K | 6 |
| supporting 평균 개수 | 2 (모두 2hop) |
| supporting recall@6 mean | **0.748** |
| recall@6 == 1.0 비율 | 51.0% |
| recall@6 ≥ 0.5 비율 | 98.5% |
| prompt 길이 (chars) mean / median / p95 / max | 3,005 / 2,792 / 5,011 / 6,527 |

> recall이 1.0이 아닌 경우 — 즉 retrieval이 supporting 문단을 놓친 경우 — 는 의도된
> 분포입니다. CacheBlend의 평가는 "현실적 retrieval 결과(noisy)" 위에서 이루어지므로
> 일부 supporting이 빠져 있어도 정상입니다. recall을 강제로 1.0으로 만들고 싶다면
> 별도 전처리(supporting을 강제 포함)가 필요한데, 본 스크립트의 기본 동작은 아닙니다.

## 7. 재현 방법

### 7.1 환경

```bash
# Python 3.11 권장 (PyTorch wheel 호환)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

또는:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

검증된 버전: `torch==2.11.0`, `sentence-transformers==5.4.1`, `numpy==2.4.4`,
디바이스 `mps` (Apple M2).

### 7.2 빌드

본 디렉터리에서:

```bash
.venv/bin/python build_prompts.py
```

기본값:

| 옵션 | 기본값 |
|---|---|
| `--input` | `../musique/musique_ans_v1.0_dev.jsonl` (mydata 레이아웃 기준) |
| `--output` | `./prompts.jsonl` |
| `--model` | `sentence-transformers/all-mpnet-base-v2` |
| `--n` | `200` |
| `--k` | `6` |
| `--seed` | `42` |

> `../musique/musique_*_train.jsonl` 은 mydata 레포에서 `.part-*` 로 분할되어 있습니다.
> train을 입력으로 쓰려면 [`../musique/README.md`](../musique/README.md)의 결합 명령으로 먼저 복원하세요.

### 7.3 결정성

같은 입력·옵션으로 재실행 시 `prompts.jsonl`이 비트 단위로 동일해야 합니다.
검증된 SHA-256:

```
791e1cf50d984f27b314c8abd49f25e3b27a0a1598a6cfcf53e28d13868a3e21  prompts.jsonl
```

검증:

```bash
shasum -a 256 prompts.jsonl   # macOS
sha256sum prompts.jsonl       # Linux
```

> 디바이스(MPS/CPU/CUDA)나 PyTorch 버전이 다르면 임베딩이 미세하게 달라져 top-K 경계
> 케이스가 바뀔 수 있고 그 결과 SHA가 다를 수 있습니다. **셔플 자체는 paragraph idx에
> 대해 결정적**이므로(셔플은 PyTorch가 아닌 stdlib `random` 사용), 같은 디바이스·버전
> 안에서는 항상 같은 결과를 얻습니다.

## 8. 활용 방법

### 8.1 모델에 넣기

`prompt` 필드를 그대로 쓰면 됩니다 (chat 템플릿 없는 raw text 컨텍스트).

```python
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

with open("prompts.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        ids = tok(rec["prompt"], return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=64, do_sample=False)
        pred = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
        gold = [rec["answer"], *rec["answer_aliases"]]
        print(rec["id"], "| pred:", pred, "| gold:", gold)
```

채팅 모델을 쓴다면 `prompt_parts.system`을 system role로, 나머지(`docs` + `question`)를
user role로 넣고 모델별 chat template을 적용하는 편이 정렬상 더 깔끔합니다.

### 8.2 평가

MuSiQue 표준 평가는 **answer EM/F1** + **answer alias** 매칭입니다.
간단 EM 예:

```python
def normalize(s):
    import re, string
    s = s.lower()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def em(pred, golds):
    p = normalize(pred)
    return any(p == normalize(g) for g in golds)
```

엄밀한 점수가 필요하면 MuSiQue 원본의 `metrics/` 평가 스크립트를 사용하세요.

### 8.3 KV 캐시 재사용 실험에 쓰기

이 데이터셋의 핵심 활용은 다음과 같은 비교입니다:

1. **Full prefill** (베이스라인): `prompt`를 그대로 한 번에 forward
2. **Cache reuse** (CacheBlend류): 같은 문서가 다른 prompt에 등장할 때 미리 계산해
   둔 KV를 재활용 — 이 때 **문서 단위 키**는 `selected_idxs_by_l2`(셔플 전 안정 ID)로
   잡으면 같은 문서 재출현을 정확히 식별할 수 있습니다.
3. 두 방식의 **출력 일치율**과 **TTFT/throughput**을 비교 → Fig.12 재현.

문서 셔플은 **위치 의존성**(어떤 문서가 prefix인지)이 결과에 미치는 영향을 평균화하기
위한 장치입니다. 이 때문에 같은 6개 문서 집합이라도 재사용 가능한 KV는 *위치별*로
제한적이라는 점이 CacheBlend가 강조하는 어려움이고, 본 데이터셋은 그 어려움을
보존합니다.

## 9. 파라미터를 바꾸고 싶다면

| 하고 싶은 것 | 명령 |
|---|---|
| 예제 수 늘리기 | `--n 1000` |
| Top-K 바꾸기 | `--k 8` |
| 다른 임베딩 모델 | `--model BAAI/bge-base-en-v1.5` |
| 다른 셔플 시드 | `--seed 0` |
| Train split 사용 | (먼저 musique 분할 복원 후) `--input ../musique/musique_ans_v1.0_train.jsonl` |
| Full split (unanswerable 포함) | `--input ../musique/musique_full_v1.0_dev.jsonl` |

## 10. 알려진 한계

- **Retrieval recall이 100%가 아님**: 위 §6의 통계 참조. 실험 목적에 따라 supporting을
  강제 포함하는 변형이 필요할 수 있습니다.
- **임베딩이 디바이스 의존적**: float32 연산이지만 MPS/CUDA/CPU 간 미세한 수치 차이로
  top-K 경계 케이스가 바뀔 수 있습니다. 비교를 엄밀하게 가져가려면 같은 디바이스에서
  항상 빌드하세요.
- **시스템 프롬프트는 raw text 형식**: chat-template 모델에 넣을 때는 §8.1의 권장
  변환을 참고하세요.

## 11. 라이선스 / 인용

- **MuSiQue**: CC BY 4.0. 사용 시 원 논문(Trivedi et al., TACL 2022) 인용.
- **CacheBlend**: 본 디렉터리는 EuroSys '24 논문의 Fig.12 실험 *조건*을 참고했습니다.
- 본 디렉터리의 스크립트/README는 mydata 저장소 라이선스를 따릅니다.
