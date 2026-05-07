# MuSiQue Dataset (mirror)

[MuSiQue](https://github.com/StonyBrookNLP/musique) (v1.0) 데이터셋의 개인 백업입니다.
원본 라이선스 CC BY 4.0 — 출처: StonyBrookNLP/musique.

## 파일 구성

| 파일 | 크기 | 설명 |
|---|---:|---|
| `dev_test_singlehop_questions_v1.0.json` | 894 KB | seed dataset 학습 시 누수 방지용 single-hop 질문 ID |
| `musique_ans_v1.0_dev.jsonl`  | 29 MB | MuSiQue-Ans dev |
| `musique_ans_v1.0_test.jsonl` | 27 MB | MuSiQue-Ans test |
| `musique_ans_v1.0_train.jsonl.part-{aa,ab,ac}`  | 총 241 MB | MuSiQue-Ans train (3분할) |
| `musique_full_v1.0_dev.jsonl`  | 57 MB | MuSiQue-Full dev |
| `musique_full_v1.0_test.jsonl` | 53 MB | MuSiQue-Full test |
| `musique_full_v1.0_train.jsonl.part-{aa..af}` | 총 476 MB | MuSiQue-Full train (6분할) |

## 분할 파일 복구 방법

GitHub의 100 MB 단일 파일 제한 때문에 두 개의 train 파일을 `split -b 90m`으로
바이트 단위 분할했습니다. `cat`으로 사전순(aa, ab, ac, …) 결합하면 원본과 동일한
바이트가 만들어집니다.

### 1. 결합 명령

리포지토리를 클론한 뒤 `musique/` 디렉터리에서 실행:

```bash
# MuSiQue-Ans train 복구
cat musique_ans_v1.0_train.jsonl.part-* > musique_ans_v1.0_train.jsonl

# MuSiQue-Full train 복구
cat musique_full_v1.0_train.jsonl.part-* > musique_full_v1.0_train.jsonl
```

> 셸의 글롭(`*`)은 사전순으로 확장되므로 part-aa → part-ab → … 순서가 보장됩니다.
> 명시적으로 순서를 지정하고 싶다면 `cat musique_ans_v1.0_train.jsonl.part-aa musique_ans_v1.0_train.jsonl.part-ab musique_ans_v1.0_train.jsonl.part-ac > musique_ans_v1.0_train.jsonl` 처럼 풀어 써도 됩니다.

### 2. 무결성 검증 (SHA-256)

복구한 파일이 원본과 동일한지 다음 해시값으로 확인:

```
83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a  musique_ans_v1.0_train.jsonl
b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034  musique_full_v1.0_train.jsonl
```

검증:

```bash
# macOS
shasum -a 256 musique_ans_v1.0_train.jsonl musique_full_v1.0_train.jsonl

# Linux
sha256sum musique_ans_v1.0_train.jsonl musique_full_v1.0_train.jsonl
```

출력된 해시가 위 값과 일치하면 정상입니다.

### 3. (선택) 분할 파일 정리

복구·검증이 끝났다면 분할 조각은 지워도 됩니다:

```bash
rm musique_ans_v1.0_train.jsonl.part-*
rm musique_full_v1.0_train.jsonl.part-*
```
