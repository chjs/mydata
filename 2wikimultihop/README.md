# 2WikiMultiHopQA Dataset (mirror)

[2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) (Ho et al., COLING 2020)
의 개인 백업입니다. 원본은
[Dropbox 링크](https://www.dropbox.com/s/npidmtadreo6df2/data.zip)로 배포됩니다.

## 파일 구성

| 파일 | 크기 | 설명 |
|---|---:|---|
| `dev.json`  | 53 MB | dev split |
| `test.json` | 51 MB | test split |
| `train.json.part-{aa..ah}` | 총 650 MB | train split (8분할) |

각 파일은 일반 JSON (한 줄 JSONL이 아님). Python으로 읽으려면
`json.load(open(path))`.

## 분할 파일 복구 방법

GitHub의 100 MB 단일 파일 제한 때문에 `train.json`을 `split -b 90m`으로
바이트 단위 분할했습니다. `cat`으로 사전순(aa, ab, … ah) 결합하면 원본과 동일한
바이트가 만들어집니다.

### 1. 결합 명령

리포지토리를 클론한 뒤 `2wikimultihop/` 디렉터리에서 실행:

```bash
cat train.json.part-* > train.json
```

> 셸의 글롭(`*`)은 사전순으로 확장되므로 part-aa → part-ab → … → part-ah 순서가
> 보장됩니다. 명시적으로 순서를 지정하려면 `cat train.json.part-aa
> train.json.part-ab ... train.json.part-ah > train.json` 처럼 풀어 쓰면 됩니다.

### 2. 무결성 검증 (SHA-256)

복구한 파일이 원본과 동일한지 다음 해시값으로 확인:

```
b3fddb4d5bb42cd797919cad67616545be51b24740e0a7dabdae7bf76b8f7bfa  train.json
48b9bdc69654dc580fda5f935a48b88cb89f11887587310af60d406c8d0111a6  dev.json
4f60ba6a108d3409ee039e0be9af42a6a139174fdc86c7451caca402e953fe7a  test.json
```

검증:

```bash
# macOS
shasum -a 256 train.json dev.json test.json

# Linux
sha256sum train.json dev.json test.json
```

출력된 해시가 위 값과 일치하면 정상입니다.

### 3. (선택) 분할 파일 정리

복구·검증이 끝났다면 분할 조각은 지워도 됩니다:

```bash
rm train.json.part-*
```

## 출처 및 인용

- 논문: Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, Akiko Aizawa,
  *"Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of
  Reasoning Steps"*, COLING 2020.
- 공식 레포: https://github.com/Alab-NII/2wikimultihop
- 데이터 라이선스는 원본 레포 기준을 따릅니다.
