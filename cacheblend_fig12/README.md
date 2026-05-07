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
├── README.md                  ← 본 문서
├── build_prompts.py           ← 프롬프트 빌드 스크립트 (MuSiQue → prompts.jsonl)
├── prompts.jsonl              ← 결과 (200줄, ~1.2 MB)
├── requirements.txt           ← Python 의존성 (build + harness)
└── harness/                   ← HF transformers 기반 평가 하네스 (§8.4 참조)
    ├── runner.py              ← CacheBlendRunner ABC + FullPrefillRunner (baseline)
    ├── metrics.py             ← F1, ROUGE-L (YaoJiayi/CacheBlend utils.py에서 포팅)
    ├── eval.py                ← argparse 기반 메인: 모델·러너·prompts.jsonl 로딩, 루프, 요약 출력
    └── example_runner.py      ← 사용자가 자신의 CacheBlend 구현을 끼우는 방법 예시
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

### 8.4 평가 하네스 (`harness/`) — HF transformers 기반 CacheBlend 측정용

[YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend)의 `example/blend_musique.py`
는 **vLLM fork**에 묶여 있어서 사용자가 직접 다른 백엔드(예: HuggingFace
transformers) 위에 CacheBlend를 재구현할 때는 그대로 가져다 쓸 수 없습니다.
`harness/`는 그 실험 루프 구조와 메트릭을 HF transformers 위로 포팅한
경량 벤치마크입니다. 사용자가 **자신의 CacheBlend 구현을 단일 클래스로 끼워
넣어** baseline(full-prefill)과 같은 데이터, 같은 모델, 같은 메트릭으로 비교할
수 있게 만들어졌습니다.

#### 8.4.1 구조

```
harness/
├── runner.py          # CacheBlendRunner ABC + FullPrefillRunner (baseline)
├── metrics.py         # F1, ROUGE-L (YaoJiayi/CacheBlend utils.py에서 포팅)
├── eval.py            # python -m harness.eval ... 의 메인
└── example_runner.py  # 사용자 구현용 stub (NotImplementedError로 시작)
```

#### 8.4.2 Runner 인터페이스

`harness/runner.py`:

```python
class CacheBlendRunner(ABC):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase): ...

    @abstractmethod
    def prepare(self, system: str, docs: list[str], question: str) -> None:
        """예제마다 한 번 호출. 청크별 prefill·KV 캐시 등 per-example state 세팅."""

    @abstractmethod
    def generate(self, max_new_tokens: int = 32) -> GenerationResult:
        """prepare()가 만든 state로 step-decoding. TTFT는 generate() 진입부터 첫 토큰까지."""
```

`GenerationResult = (text, ttft_seconds, total_seconds, n_generated_tokens)`.

**Runner는 예제 간 재사용됩니다.** 모델·토크나이저는 한 번 로드하고, 200개
예제에 대해 `prepare → generate` 사이클을 200번 돕니다. 그래서 `prepare()`
구현은 매번 이전 KV 등을 깨끗이 갈아끼워야 합니다.

#### 8.4.3 Baseline — `FullPrefillRunner`

`runner.py`에 함께 들어 있는 참조 구현. `system + docs + question`을 한
프롬프트로 합쳐 한 번에 prefill하고, `past_key_values`를 들고 step-decode
하면서 첫 토큰 시점까지의 시간을 TTFT로 잽니다.

이게 **정확히 사용자 CacheBlend가 이겨야 할 baseline TTFT**입니다 — 같은
입력으로 KV 재사용 없이 정직하게 forward한 비용.

#### 8.4.4 자기 구현 끼워 넣기

`harness/example_runner.py`를 복사해서 본인 패키지에 두고, 두 메서드를
채운 뒤 dotted spec으로 `--runner`에 넘겨줍니다.

```python
# my_pkg/my_cb.py
from harness.runner import CacheBlendRunner, GenerationResult

class MyCacheBlend(CacheBlendRunner):
    def prepare(self, system, docs, question):
        # 1) system, 각 doc, question을 따로 토크나이즈
        # 2) 청크별로 forward → 레이어별 KV 추출 → per-chunk cache 테이블에 저장
        # 3) 청크 KV를 stitch (position id 보정, 레이어별 blend, top-k refresh 등)
        # 4) self.{stitched_past, question_ids}에 보관
        ...

    def generate(self, max_new_tokens=32):
        # self.stitched_past를 past_key_values로 주입한 step-decode
        # TTFT는 이 함수 진입부터 첫 토큰 직후까지 측정
        ...
```

> 사용자 구현이 짊어지는 핵심 책임은 **(a)** 청크 KV 추출/저장, **(b)** 청크
> KV stitching (position·attention 정합), **(c)** generate() 시점의 TTFT가
> 정직하게 줄어들도록 (a)/(b)가 측정 외부에서 끝나 있을 것 — 입니다.

#### 8.4.5 실행

```bash
# 환경 (build_prompts.py와 venv 공유)
uv pip install --python .venv/bin/python -r requirements.txt

# baseline만 (default 러너)
.venv/bin/python -m harness.eval \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --n 200

# baseline vs 본인 구현 비교
.venv/bin/python -m harness.eval \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --runner harness.runner:FullPrefillRunner \
    --runner my_pkg.my_cb:MyCacheBlend \
    --n 200 \
    --report results.jsonl
```

기본값:

| 옵션 | 기본값 |
|---|---|
| `--prompts` | `./prompts.jsonl` (스크립트 기준 상대) |
| `--model` | `mistralai/Mistral-7B-Instruct-v0.2` |
| `--runner` | `harness.runner:FullPrefillRunner` (반복 가능) |
| `--n` | 전체 |
| `--max-new-tokens` | 32 |
| `--device` | `auto` (cuda → mps → cpu) |
| `--dtype` | `float16` (CPU에서는 자동 float32) |
| `--report` | 없음 (지정 시 per-example 결과를 JSONL로) |

요약 출력 예:

```
--- summary ---
runner                            TTFT(s)       F1  ROUGE-L   tok/ex
FullPrefillRunner                   1.823    0.412    0.451     14.2
MyCacheBlend                        0.214    0.398    0.439     14.0
```

`--report`로 받는 JSONL 한 줄에는
`{id, runner, ttft, total, n_tokens, pred, golds, f1, rouge_l}`이 들어갑니다 —
사후 분석/플로팅용.

#### 8.4.6 YaoJiayi 원본과의 매핑

| YaoJiayi/CacheBlend | 본 하네스 | 비고 |
|---|---|---|
| `example/blend_musique.py` 메인 루프 | `harness/eval.py` | argparse·모듈 임포트로 구조화 |
| `cache_fuse_metadata` dict 토글 | `CacheBlendRunner` 서브클래스 | "검사 vs 수집" 구분이 클래스 책임으로 들어감 |
| `model.layers[j].self_attn.hack_kv` | (사용자 구현 영역) | HF에서는 forward 후 `past_key_values`를 직접 다룸 |
| `model.old_kvs` 주입 | (사용자 구현 영역) | HF에서는 `past_key_values=...`로 주입 |
| `compute_f1` (utils.py) | `harness/metrics.py:compute_f1` | 토큰화 기반 F1, max-over-aliases 동일 |
| `prefix_prompt` / `query_prompt` (Mistral [INST]) | `_format_prompt` (raw concat) | §3에서 정한 우리 시스템 프롬프트 사용. 변경하려면 `runner.py`의 템플릿 수정 |
| 모델 하드코딩 | `--model` 인자 | 기본은 동일하게 Mistral-7B-Instruct-v0.2 |

> **중요**: YaoJiayi 원본은 Mistral `[INST]/[/INST]` 토큰을 raw로 박아 넣습니다.
> 본 하네스 baseline은 그렇게 하지 않고 `prompts.jsonl`이 가진 시스템 프롬프트를
> 그대로 텍스트로 합칩니다. 동일 모델에서 두 setup의 baseline F1이 다를 수 있는데,
> *상대 비교 (baseline 대비 자기 구현의 F1·TTFT)*는 같은 하네스 안에서만 유효합니다.

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
