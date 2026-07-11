# QC Agent Workflow

1. 입력: `QC workflow text`
2. 분류: 데이터 종류를 파악하고 적합한 QC 에이전트를 선택
3. 툴 추천: 선택된 데이터 유형에 맞는 권장 QC 도구 목록 제공
4. 실험 데이터 입력: 사용자가 QC 결과값/측정값을 입력
5. 평가: QC 에이전트가 입력 데이터를 기반으로 요약
6. 리포트: 최종 종합 리포트 생성

## 사용법

```bash
python bioQcAgent.py
```

## 입력 예시

- `RNA-seq fastq, 50M reads, mapping 90%, Q30 92%`

그리고 권장 툴을 확인한 후 실제 QC 결과를 입력합니다.
