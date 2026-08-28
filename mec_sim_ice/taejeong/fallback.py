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
from typing import Optional

try:                              # package import (e.g. `from taejeong.fallback import ...`)
    from .command_parser import LANDMARK_SYNONYMS, OBJECT_SYNONYMS, ParsedCommand
    from .grounding import Detection, _landmark_ok, _target_ok
except ImportError:               # flat import (offline_score.py 등, taejeong/ 자체가 sys.path)
    from command_parser import LANDMARK_SYNONYMS, OBJECT_SYNONYMS, ParsedCommand
    from grounding import Detection, _landmark_ok, _target_ok

log = logging.getLogger("participant_app")

# 8분할(4x2) 타일 재탐색 -- 실측(2026-08-28, hf02/pencilcase) 근거는
# _dino_best_box_tiled() 참고. 4/6/8/12분할을 비교했을 때 8분할이 정확도는
# 12분할과 동급이면서 카메라당 DINO 호출이 33% 적어 비용 대비 최선이었다.
_TILE_COLS, _TILE_ROWS, _TILE_OVERLAP = 4, 2, 0.2
# whole-frame confidence 가 이미 이 값 이상이면 타일링을 건너뛴다(비용 절감).
_TILE_UNCERTAIN_THR = 0.3


class _ShiftedBox:
    """타일 로컬 좌표 박스를 전역 좌표로 옮긴 duck-typed 박스 -- dino_fallback.
    _DinoBox 와 같은 최소 인터페이스(.xyxy, .conf)라 호출부를 안 고쳐도 된다."""

    def __init__(self, xyxy, conf: float):
        self.xyxy = [list(xyxy)]
        self.conf = [conf]


def _tile_grid(H: int, W: int, cols: int = _TILE_COLS, rows: int = _TILE_ROWS,
               overlap: float = _TILE_OVERLAP):
    """겹침 있는 (x1,y1,x2,y2) 타일 좌표 목록.

    겹침을 두는 이유: 실측에서 6분할(3x2)이 4/8분할보다 오히려 나빴다 --
    격자 경계에 물체가 걸쳐 잘리면 confidence 가 떨어진다. 20% 겹침으로
    경계 절단 위험을 줄인다.
    """
    tw, th = W / cols, H / rows
    ox, oy = tw * overlap, th * overlap
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, c * tw - ox)
            y1 = max(0, r * th - oy)
            x2 = min(W, (c + 1) * tw + ox)
            y2 = min(H, (r + 1) * th + oy)
            tiles.append((int(x1), int(y1), int(x2), int(y2)))
    return tiles


def _dino_best_box_tiled(dino, img, phrases, uncertain_thr: float = _TILE_UNCERTAIN_THR):
    """whole-frame 탐지 먼저, 그 confidence 가 낮을 때만(불확실) 8분할 재탐색.

    실측(2026-08-28, hf02/pencilcase, rig_3_b=정답 vs rig_1_a=오답 디코이):
    whole-frame 은 물체가 장면 전체 대비 작아 confidence 가 낮고 정답/오답이
    거의 구분이 안 됐다(정답 0.129 vs 오답 0.112). 같은 이미지를 8분할(4x2,
    20% 겹침)해서 타일별로 다시 태우면 정답 쪽만 두드러지게 오른다(0.129->
    0.428, 오답은 0.112->0.211) -- 물체가 로컬 프레임에서 상대적으로 커져
    DINO 의 attention 이 배경(사람/자전거/벤치 등)에 분산되지 않기 때문.
    DINO 전처리기 자체는 1280x720 을 ~1333x750 로 리샘플할 뿐이라(다운스케일
    없음), 이건 정보 손실 복구가 아니라 장면 희석 완화다.

    이전에 시도한 VLM crop 검증(verify_crop)은 30/30 케이스 전부 기각으로
    순손실만 나서 폐기했다(2026-08-28) -- 검증이 걸리는 시점 자체가 이미
    YOLO(전문 탐지기)도 놓친 최난도 잔여 케이스라, 그보다 도메인 적합도가
    낮은 제너럴리스트 VLM에게 재확인을 맡기는 게 구조적으로 불리했다.
    기하 정보(타일링)는 렌더 스타일에 영향받지 않아 이 문제가 없다.

    whole-frame conf 가 이미 uncertain_thr 이상이면 타일링 비용을 아낀다.
    타일 8장은 (dino 가 지원하면) best_box_multi_images() 로 한 번에 배치
    처리한다 -- 개별 순차 호출 대비 phrase 개수(보통 3~4개)만큼의 forward로
    끝나서 카메라당 호출 수가 최대 9배 줄어든다(실측/근거는 그 함수 참고).
    """
    whole = dino.best_box(img, phrases)
    whole_conf = float(whole.conf[0]) if whole is not None else 0.0
    if whole is not None and whole_conf >= uncertain_thr:
        return whole

    H, W = img.shape[:2]
    tiles = _tile_grid(H, W)
    tile_imgs = [img[y1:y2, x1:x2] for (x1, y1, x2, y2) in tiles]

    if hasattr(dino, "best_box_multi_images"):
        tile_boxes = dino.best_box_multi_images(tile_imgs, phrases)
    else:                          # 목/구버전 dino 폴백 (배치 메서드 없음)
        tile_boxes = [dino.best_box(t, phrases) for t in tile_imgs]

    best, best_conf = whole, whole_conf
    for (x1, y1, x2, y2), box in zip(tiles, tile_boxes):
        if box is None:
            continue
        conf = float(box.conf[0])
        if conf > best_conf:
            lx1, ly1, lx2, ly2 = box.xyxy[0]
            best = _ShiftedBox((x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2), conf)
            best_conf = conf
    return best


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
                 target_phrases: Optional[list], dino, vlm=None) -> None:
    """YOLO 가 어디서도 못 찾은 target/landmark 를 DINO 로 보충.

    detections dict 에 Detection 을 추가할 뿐 별도 제출 경로가 없다 --
    resolve() 의 카메라 게이트/기하 판정/relation 적합도가 폴백 탐지에도
    똑같이 적용된다. dino 가 None 이면 아무것도 안 함. 사람 문제는 pose
    분류가 필요해 대상 밖.

    target 탐지는 카메라별로 _dino_best_box_tiled() 를 거친다 -- whole-frame
    이 불확실하면(장면 대비 물체가 작아 confidence 가 낮으면) 8분할 재탐색으로
    보강한다. 근거/실측은 그 함수 docstring 참고.

    vlm 파라미터는 더 이상 안 쓴다(2026-08-28 폐기) -- crop 검증 방식을
    실측했더니, 검증이 걸리는 5개 케이스(30/30 개별 판정) 전부 기각으로
    순손실만 났다. 검증이 걸리는 시점 자체가 이미 YOLO 도 놓친 최난도
    잔여 케이스라, 그보다 도메인 적합도가 낮은 VLM에게 확인을 맡기는 게
    구조적으로 불리했다(sim 렌더 vs 실사 학습 데이터 갭). 호출부 시그니처
    호환을 위해 파라미터만 남겨둔다.

    landmark 를 먼저 찾고(2026-08-28), 그 결과로 target 타일 탐색 범위를
    좁힌다 -- resolve() 의 카메라 게이트(§1)가 어차피 landmark 가 검출된
    카메라로만 좁혀서 고르므로(gate 가 한 번 걸리면 안 풀림), 게이트 밖
    카메라에서 8분할까지 태워 target 을 찾아봐야 resolve() 가 버린다. 순서를
    바꿔 낭비를 없앤다 -- 정확도는 그대로, 비용만 준다(카메라 6대 중 보통
    1~2대만 그 landmark 를 보므로 target 쪽 DINO 호출이 크게 줄어든다).
    area_hint 게이트(랜드마크 미지목, 구역만 지목)는 YOLO 탐지 밀도로 정하는
    거라 이 축소 대상이 아니다 -- 지금은 손 안 댐(전체 카메라 스캔 유지).
    """
    if dino is None or parsed.is_person:
        return

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

    target_scope = images
    if lm:
        landmark_cams = {cid for cid, dets in detections.items()
                         if any(_landmark_ok(d.label, lm) for d in dets)}
        if landmark_cams:
            target_scope = {cid: img for cid, img in images.items()
                            if cid in landmark_cams}
            log.info("[fallback] target 탐색 범위를 landmark %r 검출된 %d/%d 카메라로 제한: %s",
                     lm, len(target_scope), len(images), sorted(target_scope))
        # landmark_cams 가 비면(끝내 못 찾음) 범위를 안 좁힌다 -- resolve() 도
        # 이 경우 게이트를 풀어(전체 카메라) 재탐색하므로 동작을 맞춘다.

    if target_phrases and not any(_target_ok(d.label, parsed)
                                  for dets in detections.values() for d in dets):
        for cam_id, img in target_scope.items():
            box = _dino_best_box_tiled(dino, img, target_phrases)
            if box is not None:
                detections.setdefault(cam_id, []).append(Detection(
                    label=parsed.target_type, bbox=tuple(box.xyxy[0]),
                    score=float(box.conf[0])))
                log.info("[fallback] DINO rescued target %r on %s (conf=%.2f)",
                         parsed.target_type, cam_id, float(box.conf[0]))
