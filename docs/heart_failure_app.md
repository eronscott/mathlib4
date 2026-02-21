# 심부전 측정 앱 (CLI)

`python3 scripts/heart_failure_app.py` 명령으로 동작하는 간이 위험도 계산기입니다.

## 사용 예시

```bash
python3 scripts/heart_failure_app.py \
  --age 72 \
  --systolic-bp 104 \
  --heart-rate 102 \
  --bnp 380 \
  --dyspnea
```

## 입력 항목

- `--age`: 나이
- `--systolic-bp`: 수축기 혈압(mmHg)
- `--heart-rate`: 심박수(bpm)
- `--bnp`: BNP/NT-proBNP 계열 수치
- `--edema`: 하지 부종 여부(옵션 플래그)
- `--dyspnea`: 호흡곤란 여부(옵션 플래그)

## 주의

이 도구는 교육/참고 목적의 간이 계산기이며 의학적 진단 도구가 아닙니다.
증상이 있으면 반드시 의료진과 상담하세요.
