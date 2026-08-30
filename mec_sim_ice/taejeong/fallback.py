"""DINO/VLM 폴백 배선 -- 라이브 에이전트와 오프라인 채점기가 공유하는 순수 로직.

ROS 무의존(stdlib + numpy 만, dino_fallback/vlm_fallback 은 torch/transformers
필요) -- demo_agents/taejeong_agent_vla.py(라이브, marc_sdk 필요)와
taejeong/validation/offline_score.py(오프라인, records/ 재실행)가 똑같은
함수를 호출해야 두 경로의 동작이 갈라지지 않는다. 원래 taejeong_agent_vla.py
안에 메서드로 있던 걸 여기로 뽑아냈다 -- 오프라인 채점기가 마지막까지 이
로직을 안 거치고 있었기 때문(YOLO 결과로 바로 resolve() 호출).

호출 순서 (양쪽 다 동일):
  parsed = parse_command(command)
  target_phrases = vlm_assist(command, parsed, vlm)   # parsed 를 제자리 수정
  dets = infer_cameras(model, images, ...)
  dino_augment(parsed, images, dets, target_phrases, dino)  # dets 에 추가만
  decision = resolve(parsed, dets, cams)
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

try:                              # package import (e.g. `from taejeong.fallback import ...`)
    from .command_parser import LANDMARK_SYNONYMS, OBJECT_SYNONYMS, ParsedCommand
    from .grounding import Detection, _landmark_ok, _target_ok
except ImportError:               # flat import (offline_score.py 등, taejeong/ 자체가 sys.path)
    from command_parser import LANDMARK_SYNONYMS, OBJECT_SYNONYMS, ParsedCommand
    from grounding import Detection, _landmark_ok, _target_ok

log = logging.getLogger("participant_app")


def vlm_assist(command: str, parsed: ParsedCommand, vlm) -> Optional[list]:
    """VLM 보조 2건: behind 중의 중재 + OOV target 재해석 (parsed 제자리 수정).

    vlm 이 None 이면 아무것도 안 하고(비활성화), 아는 클래스면 동의어 사전을
    그대로 반환한다.

    반환: DINO 검색용 target phrase 목록. 아는 클래스면 동의어 사전, 진짜
    신규 물체면 VLM 이 만든 검색어 1개, 해석 불능/사람 문제면 None.
    """
    if (vlm is not None and not parsed.is_person
            and parsed.relation_hint == "behind" and not parsed.relation_confident):
        # 구동사 leave-behind vs 공간관계 -- command_parser 가 confident 를
        # 낮춘 중의 문장만 온다. phrasal 이면 물건은 그 자리 근처에 남은 것.
        spatial = vlm.behind_is_spatial(command)
        if spatial is False:
            parsed.relation_hint = "near"
            log.info("[fallback] VLM arbitration: 'behind' is phrasal -> near")
        elif spatial is True:
            log.info("[fallback] VLM arbitration: 'behind' is spatial -> keep hint")
    if parsed.is_person:
        return None
    if parsed.target_type is None and vlm is not None:
        matched, phrase = vlm.resolve(command, OBJECT_SYNONYMS.keys())
        if matched:
            parsed.target_type = matched     # 사전 미등재 표현 -> 아는 클래스로 복귀
            log.info("[fallback] VLM fallback resolved OOV wording -> %r", matched)
        elif phrase:
            # 진짜 신규 물체: phrase 를 클래스명처럼 쓴다. type 점수는 어차피
            # 닫힌 어휘 밖이라 못 얻지만, resolve() 의 _target_ok 가 label
            # 일치로 DINO 탐지를 통과시키려면 target_type 과 label 이 같아야
            # 하고, 그래야 좌표 점수는 살릴 수 있다.
            parsed.target_type = phrase
            log.info("[fallback] VLM fallback: unknown object, DINO phrase=%r", phrase)
            return [phrase]
    if parsed.target_type:
        return OBJECT_SYNONYMS.get(parsed.target_type,
                                   [parsed.target_type.replace("_", " ")])
    return None


def dino_augment(parsed: ParsedCommand, images: dict, detections: dict,
                 target_phrases: Optional[list], dino) -> None:
    """YOLO 가 어디서도 못 찾은 target/landmark 를 DINO 로 보충.

    detections dict 에 Detection 을 추가할 뿐 별도 제출 경로가 없다 --
    resolve() 의 카메라 게이트/기하 판정/relation 적합도가 폴백 탐지에도
    똑같이 적용된다. dino 가 None 이면 아무것도 안 함. 사람 문제는 pose
    분류가 필요해 대상 밖.
    """
    if dino is None or parsed.is_person:
        return
    if target_phrases and not any(_target_ok(d.label, parsed)
                                  for dets in detections.values() for d in dets):
        for cam_id, img in images.items():
            box = dino.best_box(img, target_phrases)
            if box is not None:
                detections.setdefault(cam_id, []).append(Detection(
                    label=parsed.target_type, bbox=tuple(box.xyxy[0]),
                    score=float(box.conf[0])))
                log.info("[fallback] DINO rescued target %r on %s",
                         parsed.target_type, cam_id)
    lm = parsed.landmark_type
    if lm and not any(_landmark_ok(d.label, lm)
                      for dets in detections.values() for d in dets):
        # 제너릭 "car" 힌트도 그대로 label 로 쓴다 -- grounding.CAR_CLASSES 에
        # "car" 가 포함돼 _landmark_ok 를 통과하고, situation 규칙의 support
        # 토큰 정규화도 "car" 를 안다.
        phrases = LANDMARK_SYNONYMS.get(lm, [lm.replace("_", " ")])
        for cam_id, img in images.items():
            box = dino.best_box(img, phrases)
            if box is not None:
                detections.setdefault(cam_id, []).append(Detection(
                    label=lm, bbox=tuple(box.xyxy[0]), score=float(box.conf[0])))
                log.info("[fallback] DINO rescued landmark %r on %s", lm, cam_id)


def pose_augment(parsed: ParsedCommand, images: dict, detections: dict, pose) -> None:
    """사람 탐지에 발목 가로좌표(``Detection.foot_u``)를 채운다 (제자리 수정).

    ``pose`` 가 None 이거나 사람 문항이 아니면 아무것도 하지 않는다. 사람
    문항에서만 도는 이유는 플랫폼 정답이 '사람의 발' 이라는 사실에 근거한
    보정이기 때문이다 -- 물체·랜드마크는 지금(bbox 가로 중앙)이 맞는 값이다
    (pose_keypoints.py 모듈 docstring 의 실측표 참조).

    dino_augment 와 마찬가지로 순수 추가 계층이다: 발목을 못 찾은 탐지는
    건드리지 않으므로 pose 모델이 없거나 실패해도 기존 동작 그대로다.
    """
    if pose is None or not parsed.is_person:
        return
    for cam_id, dets in detections.items():
        img = images.get(cam_id)
        if img is None:
            continue
        idx = [i for i, d in enumerate(dets)
               if d.label == "person" or d.label.startswith("person_")]
        if not idx:
            continue
        found = pose.foot_u(img, [dets[i].bbox for i in idx])
        for k, u in found.items():
            i = idx[k]
            dets[i] = replace(dets[i], foot_u=u)
        if found:
            log.info("[pose-kp] %s: %d/%d person detections got an ankle column",
                     cam_id, len(found), len(idx))
