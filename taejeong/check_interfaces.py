"""실행 전 인터페이스 사전 점검 (pre-flight).

세 경계를 검사한다:
  1. 클래스 어휘: 팀원 det_classes.py (DET_CLASSES) ↔ 우리 command_parser 어휘
     ↔ situation_resolver pose 클래스
  2. 가중치: best.pt 의 model.names ↔ DET_CLASSES (--weights 줬을 때만)
  3. 제출 타입: 우리 GroundingDecision.to_payload() ↔ marc_sdk GroundingResult

사용:
  <python> check_interfaces.py --det-classes ../marc-starter-kit/test/det_classes.py \
      [--weights weights/best.pt] [--sdk ../marc-starter-kit]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

from command_parser import LANDMARK_SYNONYMS, OBJECT_SYNONYMS
from grounding import CAR_CLASSES, GroundingDecision
from situation_resolver import POSE_CLASS


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-classes", required=True,
                    help="팀원 test/det_classes.py 경로")
    ap.add_argument("--weights", default=None, help="best.pt 경로 (선택)")
    ap.add_argument("--sdk", default=None,
                    help="marc-starter-kit 루트 (marc_sdk 임포트용, 선택)")
    args = ap.parse_args()
    issues, notes = [], []

    dc = load_module(args.det_classes, "det_classes")

    # ── 1. 클래스 어휘 대조 ──
    ours_objects = set(OBJECT_SYNONYMS)
    theirs_targets = set(dc.TARGET_CLASSES)
    for c in theirs_targets - ours_objects:
        issues.append(f"detector target '{c}' 를 파서 어휘가 모름 → 명령에서 못 뽑음")
    for c in ours_objects - theirs_targets:
        notes.append(f"파서는 아는데 탐지 불가: '{c}' (명령에 나오면 미탐지 → 폴백)")

    ours_landmarks = set(LANDMARK_SYNONYMS) | set(CAR_CLASSES)
    for c in set(dc.LANDMARK_CLASSES) - ours_landmarks:
        issues.append(f"detector landmark '{c}' 를 grounding._landmark_ok 가 거름")

    ours_poses = set(POSE_CLASS)
    theirs_poses = set(dc.POSE_CLASSES)
    if ours_poses != theirs_poses:
        issues.append(f"pose 클래스 불일치: 공식표에만 {ours_poses - theirs_poses}, "
                      f"탐지기에만 {theirs_poses - ours_poses}")

    print(f"[1] 클래스 어휘: targets {len(theirs_targets)}, "
          f"landmarks {len(dc.LANDMARK_CLASSES)}, poses {len(theirs_poses)}"
          f" — {'OK' if not issues else 'PROBLEMS'}")

    # ── 2. 가중치 클래스 ──
    if args.weights:
        if not os.path.exists(args.weights):
            issues.append(f"weights 파일 없음: {args.weights}")
        else:
            from ultralytics import YOLO
            from yolo_adapter import verify_model_names
            model = YOLO(args.weights)
            probs = verify_model_names(model, dc.DET_CLASSES)
            for p in probs:
                issues.append(f"weights: {p}")
            print(f"[2] weights model.names {len(model.names)}개 — "
                  f"{'OK' if not probs else 'PROBLEMS'}")
    else:
        notes.append("--weights 미지정: 가중치 클래스 검증 건너뜀")

    # ── 3. 제출 payload ↔ marc_sdk.GroundingResult ──
    if args.sdk:
        # marc_sdk 패키지 임포트는 __init__ 이 rclpy(ROS) 를 끌고 오므로
        # ROS 없는 머신에서도 돌도록 types.py 만 파일로 직접 로드한다.
        types_mod = load_module(
            os.path.join(args.sdk, "marc_sdk", "types.py"), "marc_sdk_types")
        GroundingResult = types_mod.GroundingResult
        sample = GroundingDecision(
            camera_id="rig_x", target_type="mug", landmark="bench",
            relation="on", situation=None,
            target_coord=[1.0, 2.0, 0.1], anchor_coord=[1.5, 2.0, 0.4],
        ).to_payload()
        gr = GroundingResult.from_payload(sample)
        back = gr.to_payload()
        for section in ("interpretation", "grounding"):
            for k, v in sample[section].items():
                if back.get(section, {}).get(k) != v and v is not None:
                    issues.append(f"payload 왕복 불일치: {section}.{k}: "
                                  f"{v!r} -> {back.get(section, {}).get(k)!r}")
        print(f"[3] GroundingResult.from_payload 왕복 — "
              f"{'OK' if not any('payload' in i for i in issues) else 'PROBLEMS'}")
    else:
        notes.append("--sdk 미지정: GroundingResult 왕복 검증 건너뜀")

    for n in notes:
        print("  note:", n)
    for i in issues:
        print("  ISSUE:", i)
    print("PASS" if not issues else f"FAIL ({len(issues)} issues)")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
