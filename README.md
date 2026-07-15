# QC Agent Workflow

1. 입력: `QC workflow text` — 자유 텍스트로 omics 데이터를 설명
2. 분류: 데이터 종류를 파악하고 적합한 QC 에이전트를 선택
3. 툴 추천: 선택된 데이터 유형에 맞는 권장 QC 도구 목록 제공
4. 분석 목적 입력: 최종 분석 목적을 입력 (예: De novo assembly, 차등발현 분석 등)
5. 실험 데이터 입력: 사용자가 QC 결과값/측정값을 입력
6. 평가: QC 에이전트가 입력 데이터와 분석 목적을 기반으로 지표별 PASS/WARNING/FAIL 평가
7. 리포트: 최종 종합 리포트 생성 — 텍스트 리포트와 함께 `data/charts/` 폴더에 지표별 PNG 차트가 자동 저장됨

## 지원하는 데이터 카테고리

HiFi, ONT, Illumina, Hi-C, RNA-seq, Methylation, Single-cell, ATAC-seq, WGS

## 설치

```bash
pip install -r requirements.txt
```

`OPENAI_API_KEY` / `NCBI_API_KEY` 없이도 규칙 기반 fallback으로 전체 흐름이 동작하지만, 실제 LLM 평가와 문헌 참조가 필요하면 환경 변수 또는 `api_key.py`에 키를 설정하세요.

## 사용법

```bash
python bioQcAgent.py
```

## 입력 예시

딱딱한 형식을 맞출 필요 없이, 자연스러운 문장으로 입력해도 데이터 종류를 알아서 분류합니다.

- `하이파이 데이터야` → HiFi로 분류
- `RNA-seq 데이터 QC 확인하고 싶어` → RNA-seq로 분류
- `ONT 나노포어 시퀀싱 결과 확인` → ONT로 분류
- `메틸레이션 어레이 데이터 QC` → Methylation으로 분류

분류 및 권장 툴을 확인한 후, 분석 목적과 실제 QC 결과를 순서대로 입력합니다.
