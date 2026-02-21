#!/usr/bin/env python3
"""심부전 위험도 간이 측정 앱 (CLI).

주의: 본 도구는 의료 진단 도구가 아니며, 응급 증상 시 즉시 의료진에게 연락해야 합니다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Inputs:
    age: int
    systolic_bp: int
    heart_rate: int
    bnp: float
    edema: bool
    dyspnea: bool


def calculate_risk_score(data: Inputs) -> int:
    """단순 규칙 기반 점수 계산 (0~100 스케일 전 단계)."""
    score = 0

    if data.age >= 75:
        score += 12
    elif data.age >= 65:
        score += 8

    if data.systolic_bp < 100:
        score += 16
    elif data.systolic_bp < 110:
        score += 10

    if data.heart_rate >= 110:
        score += 12
    elif data.heart_rate >= 95:
        score += 6

    if data.bnp >= 400:
        score += 30
    elif data.bnp >= 100:
        score += 18

    if data.edema:
        score += 10
    if data.dyspnea:
        score += 10

    return min(score, 100)


def classify(score: int) -> str:
    if score >= 60:
        return "고위험"
    if score >= 35:
        return "중등도 위험"
    return "저위험"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="심부전 위험도 간이 측정 앱 (CLI)",
    )
    parser.add_argument("--age", type=int, required=True, help="나이")
    parser.add_argument("--systolic-bp", type=int, required=True, help="수축기 혈압 (mmHg)")
    parser.add_argument("--heart-rate", type=int, required=True, help="심박수 (bpm)")
    parser.add_argument("--bnp", type=float, required=True, help="BNP 또는 NT-proBNP 계열 수치")
    parser.add_argument("--edema", action="store_true", help="하지 부종 여부")
    parser.add_argument("--dyspnea", action="store_true", help="호흡곤란 여부")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    data = Inputs(
        age=args.age,
        systolic_bp=args.systolic_bp,
        heart_rate=args.heart_rate,
        bnp=args.bnp,
        edema=args.edema,
        dyspnea=args.dyspnea,
    )

    score = calculate_risk_score(data)
    risk = classify(score)

    print("=== 심부전 위험도 간이 측정 결과 ===")
    print(f"점수: {score}/100")
    print(f"분류: {risk}")
    print("\n※ 본 결과는 참고용이며 진단을 대체하지 않습니다.")

    if risk == "고위험":
        print("권고: 가능한 빠르게 심장내과 진료를 받으세요.")
    elif risk == "중등도 위험":
        print("권고: 조속히 외래 진료 예약 후 정밀 검사를 고려하세요.")
    else:
        print("권고: 생활습관 관리 및 정기 검진을 지속하세요.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
