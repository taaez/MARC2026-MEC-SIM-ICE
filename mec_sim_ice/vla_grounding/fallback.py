"""DINO/VLM 폴백 배선 -- 라이브 에이전트와 오프라인 채점기가 공유하는 순수 로직.

ROS 무의존(stdlib + numpy 만, dino_fallback/vlm_fallback 은 torch/transformers
필요) -- demo_agents/vla_grounding_agent.py(라이브, marc_sdk 필요)와
vla_grounding/validation/offline_score.py(오프라인, records/ 재실행)가 똑같은
함수를 호출해야 두 경로의 동작이 갈라지지 않는다. 원래 vla_grounding_agent.py
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

try:                              # package import (e.g. `from vla_grounding.fallback import ...`)
    from .command_parser import (LANDMARK_SYNONYMS, OBJECT_SYNONYMS, POSE_BUCKETS,
                                 ParsedCommand)
    from .grounding import (AREA_DOMINANCE_RATIO, AREA_INDICATOR_CLASSES,
                            Detection, _landmark_ok, _target_ok)
except ImportError:               # flat import (offline_score.py 등, vla_grounding/ 자체가 sys.path)
    from command_parser import (LANDMARK_SYNONYMS, OBJECT_SYNONYMS, POSE_BUCKETS,
                                ParsedCommand)
    from grounding import (AREA_DOMINANCE_RATIO, AREA_INDICATOR_CLASSES,
                           Detection, _landmark_ok, _target_ok)

_POSE_BUCKET_NAMES = [name for name, _ in POSE_BUCKETS]

log = logging.getLogger("participant_app")

# 8분할(4x2) 타일 재탐색 -- 실측(2026-08-28, hf02/pencilcase) 근거는
# _dino_best_box_tiled() 참고. 4/6/8/12분할을 비교했을 때 8분할이 정확도는
# 12분할과 동급이면서 카메라당 DINO 호출이 33% 적어 비용 대비 최선이었다.
_TILE_COLS, _TILE_ROWS, _TILE_OVERLAP = 4, 2, 0.2
# whole-frame confidence 가 이미 이 값 이상이면 타일링을 건너뛴다(비용 절감).
_TILE_UNCERTAIN_THR = 0.3

# landmark 구제 전용 최소 confidence -- DinoFallbackDetector 기본 box_thr(0.3)보다
# 높게 잡는다. 실측(2026-08-29, s1_007/s1_014, landmark="picnic_table"):
# 무관한 물체(taxi 0.479, vending machine 0.501/0.51, bench 0.592/0.60)가
# picnic_table 로 오탐됐는데, 진짜 피크닉 테이블은 rig_1_a(0.932)/rig_3_b
# (0.737)로 뚜렷하게 높았다. 오탐이 "table" 같은 범용 동의어 때문이라 보고
# strict("picnic table"/"outdoor table")/loose("table") 로 나눠 loose 만
# 더 높은 문턱을 매기는 방식을 시도했으나(같은 날), 재실측 결과 strict
# phrase 만으로도 같은 오탐(rig_2_b 0.59, rig_3_c 0.53, rig_7_a 0.48)이
# 그대로 났다 -- 원인이 특정 단어가 아니라 이 씬에서 Grounding DINO 가
# picnic_table 클래스 자체를 잘 구분 못 하는 것이었다. phrase 를 나눠도
# 문제가 안 사라지고 target_scope 만 도로 넓어져(2/6->6/6) 되돌렸다. 그래서
# phrase 구분 없이 최종 최고점 하나에만 문턱을 매기는 단순한 형태를 쓴다.
# target 재탐색(위 _TILE_UNCERTAIN_THR)은 반대로 낮은 확신(실측 0.11까지)도
# 진짜인 경우가 있어 그대로 두고, landmark 쪽에만 별도로 더 높은 문턱을 둔다
# -- 랜드마크는 설계상 "숨어있지 않은, 흔히 보이는 기준 물체"라 확신이 낮으면
# 진짜 없을 확률이 높다.
_LANDMARK_MIN_CONF = 0.65

# 같은 카메라 안에서 DINO 가 찾은 박스가 YOLO 가 이미 찾은 것과 "같은 물건"인지
# 판정하는 IoU 문턱. 이보다 겹치면 재탐지(새 정보 없음)로 보고 버리고, 이하면
# 카메라 안의 다른 인스턴스로 보고 추가한다(2026-08-29, sar01/sarH4 -- 아래
# dino_augment 랜드마크 루프 참고).
_SAME_INSTANCE_IOU = 0.3

# DINO 의 target 재탐지가 실제로는 YOLO 가 이미 다른 클래스로 확신하는 물건을
# 오분류한 것인지 거르는 문턱. 실측(2026-08-29): rig_3_b 의 콜라캔 하나가
# 질의 phrase 에 따라 "pencilcase"(conf 0.44, rtC2/lf07/hf02)로도,
# "mug"(conf 0.42, hf03)로도 잡혔다 -- 둘 다 실제로는 YOLO 가 0.95+ 로
# 확신하는 cola_can 과 픽셀 단위로 거의 같은 박스였다. DINO 는 열린 어휘라
# 이런 오분류를 스스로 못 거르고, confidence 도 진짜(약함)보다 가짜(강함)가
# 높을 수 있어(hf03: 진짜 mug 0.356 < 가짜 mug 0.419) confidence 비교만으로도
# 못 가린다. "YOLO 가 이미 높은 확신으로 다른 이름을 붙인 바로 그 박스"면
# 기각한다 -- _SAME_INSTANCE_IOU(0.3, "다른 인스턴스로 볼지" 판단)보다 높게
# 잡는다(0.5): 저건 "이 정도면 다른 물건일 수도 있다"는 관대한 기준이고,
# 여기는 "이건 확실히 같은 박스다"를 요구해야 진짜 탐지를 잘못 버리지 않는다.
#
# 클래스 이름으로 예외를 두는 방식은 두 번 틀렸다(2026-08-29):
#   1차: 제외 없음 -- hf04_park_sunblock 이 깨짐(DINO 가 sunblock 을 못
#        찾자 picnic_table 박스를 거의 그대로 반환, IoU=0.972 -- 랜드마크
#        위/옆에 있는 정상 관계라 오탐이 아닌데 걸러버림).
#   2차: "랜드마크 계열 클래스는 제외" -- hf03_table_mug 가 다시 깨짐.
#        진범이 picnic_table 이 아니라 "bicycle"(쓰러진 자전거, IoU=0.96)
#        이었는데, bicycle 도 LANDMARK_SYNONYMS 소속이라 똑같이 봐줘버렸다.
# 클래스가 랜드마크냐 아니냐가 진짜 구분선이 아니었다 -- **박스 크기**가
# 진짜 구분선이다. hf04 의 "테이블째 반환"은 프레임의 11.5%(가로세로 다
# 몇백 px)인데, hf03 의 자전거(3.7%)와 콜라캔(0.09%) 오인은 실제 소품
# 크기 그대로다. 즉 후보 박스가 충분히 크면 "특정 소품과 착각"한 게
# 아니라 "작은 물체를 못 찾아 그나마 유력한 넓은 지역째로 반환"한 것으로
# 보고 오탐 검사 자체를 건너뛴다 -- 이러면 랜드마크 여부와 무관하게 두
# 실패 모드가 정확히 갈린다.
_DECOY_OVERLAP_IOU = 0.5
_DECOY_MIN_CONF = 0.5
_DECOY_MAX_CANDIDATE_FRAC = 0.08   # 이 이상이면 "넓은 지역째 반환"으로 보고 예외


def _is_known_decoy(new_bbox: tuple, cam_dets: list, target_type: str,
                    frame_area: float) -> bool:
    """new_bbox 가 YOLO 가 이미 다른 클래스로 확신하는 박스와 사실상
    같은지. 후보 자체가 이미 프레임 대부분을 덮는 넓은 박스면(작은
    소품일 수 없는 크기) 검사하지 않는다."""
    x1, y1, x2, y2 = new_bbox
    cand_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if frame_area > 0 and cand_area / frame_area >= _DECOY_MAX_CANDIDATE_FRAC:
        return False
    return any(d.label != target_type and d.score >= _DECOY_MIN_CONF
              and _iou(new_bbox, d.bbox) >= _DECOY_OVERLAP_IOU
              for d in cam_dets)


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0

# whole-frame 이 불확실해 타일링으로 넘어갈 때, phrase 전부를 8타일에 태우지
# 않고 유력한 것만 추려서 태운다. 실측(2026-08-28, pencilcase): 카메라 1대당
# 8타일 x phrase(보통 3~4개)가 병목이었다 -- 여기서 phrase 를 topk 로 줄이면
# 타일 호출이 그 비율만큼 준다(4개->2개면 타일 호출 절반).
#
# 순위는 "정상 문턱을 넘긴 것" 이 아니라 낮은 probe 문턱(_PROBE_BOX_THR)으로
# 다시 재본 점수로 매긴다 -- 애초에 이 분기로 온 이유가 "물체가 작아 전부
# 0.3 문턱 미만"(whole_conf < uncertain_thr)이라, 정상 문턱으로는 phrase 간
# 우열을 가릴 신호 자체가 없다(전부 필터돼 안 보임). 점수 값 자체는 후처리
# 문턱과 무관하므로(문턱은 반환 여부만 결정), probe 로 다시 불러도 최종
# whole/uncertain 판정에는 영향이 없다 -- phrase 순위 매기기 전용이다.
_PROBE_BOX_THR = 0.05
_TILE_PHRASE_TOPK = 2

# 재귀 줌(2단계 세분 타일링) -- 8분할로도 물체가 전부 uncertain_thr(0.3) 미만이면
# (진짜 완전 미탐 후보, 예: hf02/hf03 류 target=0 케이스) 그중 가장 유력한 타일 1개를
# 한 번 더 2x2 로 쪼개 재검색한다. SAHI(Slicing Aided Hyper Inference)의 계층적
# 슬라이싱과 같은 원리 -- 8분할에서도 여전히 작으면 그 영역만 더 확대해서 로컬
# 프레임 대비 물체 비중을 키운다. 8분할 전부가 문턱 미만이면 정상 문턱으로는 어느
# 타일이 유력한지 신호가 아예 없으므로(전부 필터돼 안 보임), 낮은 probe 문턱으로
# 한 번 더 8타일을 조회해 후보를 고른다(최종 채택 문턱은 그대로 정상 문턱 -- probe
# 는 "어디를 더 볼지"만 정하지 결과를 낮은 신뢰로 그냥 통과시키지 않는다).
#
# 이미 uncertain_thr 를 넘긴 케이스(실측: pencilcase 0.32~0.48, mug 0.32~0.45)는
# 이 분기를 안 타므로 비용이 그대로다 -- 진짜 완전 미탐일 때만 추가 비용(약 4콜)이
# 붙는다.
_TILE2_COLS, _TILE2_ROWS, _TILE2_OVERLAP = 2, 2, 0.25


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

    타일링까지 왔다는 건 phrase 전부가 uncertain_thr 미만이란 뜻이라(작은
    물체), 8타일 x phrase 전부를 태우는 건 낭비다 -- 그중 유력한 phrase
    top-k(_TILE_PHRASE_TOPK)만 골라 타일링한다(phrase_detections() 지원 시).

    8분할로도 전부 uncertain_thr 미만이면(완전 미탐 후보) 가장 유력한 타일
    1개를 한 번 더 2x2 로 확대해서 재검색한다(SAHI 식 계층적 슬라이싱, 근거는
    _TILE2_* 상수 주석 참고) -- 이미 uncertain_thr 를 넘긴 경우는 이 분기를
    안 타 비용이 그대로다.
    """
    whole = dino.best_box(img, phrases)
    whole_conf = float(whole.conf[0]) if whole is not None else 0.0
    if whole is not None and whole_conf >= uncertain_thr:
        return whole

    tile_phrases = phrases
    if hasattr(dino, "phrase_detections") and len(phrases) > _TILE_PHRASE_TOPK:
        scored = dino.phrase_detections(img, phrases, box_thr=_PROBE_BOX_THR)
        ranked = sorted(phrases, key=lambda p: scored[p][1], reverse=True)
        tile_phrases = ranked[:_TILE_PHRASE_TOPK]
        log.info("[fallback] 타일링 phrase %d개->%d개로 축소: %s",
                 len(phrases), len(tile_phrases), tile_phrases)

    H, W = img.shape[:2]
    tiles = _tile_grid(H, W)
    tile_imgs = [img[y1:y2, x1:x2] for (x1, y1, x2, y2) in tiles]

    if hasattr(dino, "best_box_multi_images"):
        tile_boxes = dino.best_box_multi_images(tile_imgs, tile_phrases)
    else:                          # 목/구버전 dino 폴백 (배치 메서드 없음)
        tile_boxes = [dino.best_box(t, tile_phrases) for t in tile_imgs]

    best, best_conf = whole, whole_conf
    for (x1, y1, x2, y2), box in zip(tiles, tile_boxes):
        if box is None:
            continue
        conf = float(box.conf[0])
        if conf > best_conf:
            lx1, ly1, lx2, ly2 = box.xyxy[0]
            best = _ShiftedBox((x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2), conf)
            best_conf = conf

    # 재귀 줌: 8분할 전부가 uncertain_thr 미만(= 정상 문턱으로 아무 타일도 못
    # 건짐, best_conf 는 여기서 0.0)이면 어느 타일이 그나마 유력한지 낮은
    # probe 문턱으로 다시 조회해 후보를 고른다 -- probe 는 "어디를 더 볼지"만
    # 정하고, 최종 채택은 아래 2x2 재검색에서 정상 문턱으로 다시 판정하므로
    # 신뢰도가 낮은 결과를 그냥 통과시키는 게 아니다.
    if best_conf < uncertain_thr and hasattr(dino, "best_box_multi_images"):
        probe_boxes = dino.best_box_multi_images(tile_imgs, tile_phrases,
                                                  box_thr=_PROBE_BOX_THR)
        zoom_idx = max(
            (i for i, b in enumerate(probe_boxes) if b is not None),
            key=lambda i: float(probe_boxes[i].conf[0]), default=None)
        if zoom_idx is not None:
            zx1, zy1, zx2, zy2 = tiles[zoom_idx]
            zoom_img = img[zy1:zy2, zx1:zx2]
            zh, zw = zoom_img.shape[:2]
            sub_tiles = _tile_grid(zh, zw, cols=_TILE2_COLS, rows=_TILE2_ROWS,
                                   overlap=_TILE2_OVERLAP)
            sub_imgs = [zoom_img[sy1:sy2, sx1:sx2] for (sx1, sy1, sx2, sy2) in sub_tiles]
            sub_boxes = dino.best_box_multi_images(sub_imgs, tile_phrases)
            for (sx1, sy1, sx2, sy2), box in zip(sub_tiles, sub_boxes):
                if box is None:
                    continue
                conf = float(box.conf[0])
                if conf > best_conf:
                    lx1, ly1, lx2, ly2 = box.xyxy[0]
                    best = _ShiftedBox((zx1 + sx1 + lx1, zy1 + sy1 + ly1,
                                        zx1 + sx1 + lx2, zy1 + sy1 + ly2), conf)
                    best_conf = conf
                    log.info("[fallback] 재귀 줌으로 구제: 타일 %d 확대 재검색 "
                             "(conf=%.2f)", zoom_idx, conf)
    return best


def vlm_assist(command: str, parsed: ParsedCommand, vlm) -> Optional[list]:
    """VLM(텍스트 전용) 보조 5건 -- 전부 "이 문장이 뭘 말하는지" 순수 언어
    이해뿐이고, 이미지 판단은 절대 안 한다(parsed 제자리 수정):
      1. behind 중의 중재 (leave-behind 구동사 vs 공간관계)
      2. landmark OOV 재해석 (2026-08-29 추가)
      3. is_person 재분류 (2026-08-29 추가)
      4. pose_hint 재해석 (2026-08-29 추가)
      5. target OOV 재해석 (기존)

    vlm 이 None 이면 전부 스킵(비활성화). 2~4는 텍스트 규칙(command_parser)
    이 이미 확신하는 문장에는 안 건다 -- 규칙표가 이미 정답을 내는 39/39
    코퍼스에서는 전부 0회 호출이어야 정상이다(기존 target/behind 보조와
    같은 원칙). **TODO(테스트 필요)**: 2~4는 아직 실제 모델로 검증한 적이
    없다(vlm_fallback.py 모듈 docstring 참고) -- 검증 전에는 제출
    기본값(VLM 기본 OFF)을 바꾸지 말 것.

    반환: DINO 검색용 target phrase 목록. 아는 클래스면 동의어 사전, 진짜
    신규 물체면 VLM 이 만든 검색어 1개, 해석 불능/사람 문제면 None.
    """
    if vlm is None:
        return None

    if (not parsed.is_person
            and parsed.relation_hint == "behind" and not parsed.relation_confident):
        # 구동사 leave-behind vs 공간관계 -- command_parser 가 confident 를
        # 낮춘 중의 문장만 온다. phrasal 이면 물건은 그 자리 근처에 남은 것.
        spatial = vlm.behind_is_spatial(command)
        if spatial is False:
            parsed.relation_hint = "near"
            log.info("[fallback] VLM arbitration: 'behind' is phrasal -> near")
        elif spatial is True:
            log.info("[fallback] VLM arbitration: 'behind' is spatial -> keep hint")

    # landmark OOV -- 텍스트가 랜드마크도, 구역 힌트(area_hint)도, "빈 자리"
    # 표현(open_space)도 못 찾았을 때만 묻는다. 셋 다 "랜드마크 없음"을
    # 정상적으로 표현하는 경로라 여기서 막지 않으면 그런 문장에도 매번
    # VLM 이 랜드마크를 지어낼 위험이 있다. 반환값은 target 과 동일한
    # 패턴(아는 클래스 or DINO 검색용 phrase)이라 dino_augment() 의 기존
    # landmark 재탐지 경로(LANDMARK_SYNONYMS.get(lm, [lm 그대로]))를 그대로
    # 탄다 -- 새 배선이 필요 없다.
    if parsed.landmark_type is None and not parsed.area_hint and not parsed.open_space:
        matched_lm, lm_phrase = vlm.resolve_landmark(command, LANDMARK_SYNONYMS.keys())
        if matched_lm:
            parsed.landmark_type = matched_lm
            log.info("[fallback] VLM landmark resolved OOV wording -> %r", matched_lm)
        elif lm_phrase:
            parsed.landmark_type = lm_phrase
            log.info("[fallback] VLM landmark: unknown landmark, DINO phrase=%r", lm_phrase)

    # is_person 재분류 -- 텍스트가 물체도(target_type), 사람 단어도
    # (STRONG/WEAK_PERSON_WORDS) 못 찾았을 때만(진짜 중의). 둘 중 하나라도
    # 이미 확정됐으면 규칙표를 믿는다.
    if parsed.target_type is None and not parsed.is_person:
        is_person = vlm.classify_person(command)
        if is_person is True:
            parsed.is_person = True
            log.info("[fallback] VLM: 사람 문제로 재분류")

    if parsed.is_person:
        # pose_hint 재해석 -- POSE_BUCKETS 정규식 6종이 못 잡은 새 자세
        # 표현만 묻는다. situation 자체는 여전히 situation_resolver.py 의
        # 공식 규칙표가 결정한다 -- 여기서 나오는 건 여러 사람 후보 중
        # 고르는 데 쓰는 "코스 버킷" 힌트일 뿐이다(DESIGN.md 2.2 원칙 유지).
        if parsed.pose_hint is None:
            pose = vlm.classify_pose(command, _POSE_BUCKET_NAMES)
            if pose:
                parsed.pose_hint = pose
                log.info("[fallback] VLM pose_hint resolved -> %r", pose)
        return None

    if parsed.target_type is None:
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
    똑같이 적용된다. dino 가 None 이면 아무것도 안 함.

    사람 문제는 **target(사람) 재탐지만** 대상 밖이다 -- pose 분류가 필요해
    DINO 로는 못 한다. landmark 보강은 2026-08-29부터 사람 문제에도 켠다:
    pose 와 무관한 순수 물체 탐지라 못 할 이유가 없는데, 예전엔 이 함수
    맨 위에서 `parsed.is_person` 하나로 통째로 걸러 landmark 까지 같이
    막고 있었다. 실측(sar01/sarH4, 둘 다 "sitting on top of the trash
    can"): YOLO 가 그 카메라에서 trash_can 을 하나 찾긴 하는데, 그게
    사람이 앉은 진짜 trash_can 이 아니라 7.6m 떨어진 다른 인스턴스였다
    (같은 카메라에 같은 클래스 물체가 여러 개 있는 상황 -- RUNBOOK 의 rt01
    police_car/normal_car_1 사례와 뿌리가 같다). 그 결과 anchor 좌표가
    엉뚱한 물체를 가리키고
    (anchor=0), classify_relation() 이 "on" 을 못 내려 situation 도
    기본값으로 떨어진다(object_type=0) -- 사람 자체(target)는 GT 1.9m
    이내로 정확히 찾았는데도 두 필드가 통째로 0점이었다. landmark 보강을
    열어 두면 두 번째(더 가까운) 인스턴스를 DINO 가 찾을 기회가 생기고,
    resolve() 는 이미 카메라당 여러 랜드마크 인스턴스를 놓고 쌍 스코어링
    하도록 돼 있으니(§2) 그 중 실제로 맞는 짝을 고르는 건 기존 로직이
    그대로 한다 -- 새 선택 규칙이 필요 없다.

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

    area_hint(랜드마크 미지목, "parking space"/"park" 처럼 구역만 지목한 문제,
    예: s1_009 tumbler)도 2026-08-29부터 같은 방식으로 좁힌다 --
    grounding.AREA_INDICATOR_CLASSES/AREA_DOMINANCE_RATIO 를 그대로 재사용해
    resolve() 의 구역 게이트를 미리 앞당겨 적용한다. resolve() 가 최종적으로
    같은 판정을 내리므로 정확도 손실은 원리상 없다 -- 게이트 밖 카메라는
    어차피 버려지는데 실측(9번, landmark 없음)에서 이게 없어 6대 전부에
    8분할이 돌아 카메라당 최대 36 forward x 6 = 216 이 걸리고 있었다.
    """
    if dino is None:
        return

    lm = parsed.landmark_type
    if lm:
        # 제너릭 "car" 힌트도 그대로 label 로 쓴다 -- grounding.CAR_CLASSES 에
        # "car" 가 포함돼 _landmark_ok 를 통과하고, situation 규칙의 support
        # 토큰 정규화도 "car" 를 안다.
        # strict/loose 분리(구체적 동의어 vs "table" 등 범용 동의어)를 실제
        # 시도했으나(2026-08-29) 되돌렸다 -- 실측(s1_007, rig_2_b/rig_3_c/
        # rig_7_a): "table"을 뺀 strict 구성("picnic table", "outdoor table")
        # 만으로도 무관한 물체에서 0.48~0.59 로 걸렸다. 즉 오탐의 원인이
        # "table"이라는 특정 단어가 아니라 이 장면에서 Grounding DINO 가
        # picnic_table 클래스 자체를 잘 구분 못 하는 것이었다 -- phrase를
        # 나눠도 문제가 안 사라지고, 오히려 예전에 걸러지던 카메라들이 다시
        # target_scope 에 들어가(2/6 -> 6/6) 속도만 나빠졌다. 그래서 phrase
        # 구분 없이 최종 최고점 하나에만 문턱을 매기는 단순한 형태로 되돌린다.
        phrases = LANDMARK_SYNONYMS.get(lm, [lm.replace("_", " ")])
        # YOLO 가 이미 카메라를 1대로 좁혔으면 그 랜드마크는 변별력이 높다는 뜻이라
        # (grounding.py 의 landmark_discriminative 와 같은 판단) DINO 를 아예 안
        # 태운다. 아래 카메라별 가드는 "YOLO 가 못 본 카메라"를 열어 주는데, 개방
        # 어휘 DINO 는 클래스 변형을 구분하지 못해 거기서 오탐을 만든다.
        # 실측(2026-08-29, taxi 4문항 lf10/lf11/rt04/rt04b, GT 전부 rig_3_c):
        # YOLO 는 taxi 를 rig_3_c 한 대에서만 찾아 게이트가 정답 1대였는데, DINO
        # 가 rig_2_c(0.73)/rig_7_a(0.75) 의 평범한 승용차까지 taxi 로 건져 게이트가
        # 3대로 넓어졌고 그 안에서 오답 카메라가 이겼다(-136.8점). conf 문턱으로는
        # 못 거른다 -- 진짜 taxi 와 오탐이 똑같이 0.73~0.75 다.
        # trash_can 처럼 여러 카메라에 정상적으로 흩어진 랜드마크는 YOLO 게이트가
        # 애초에 2대 이상이라 이 가드에 안 걸리고 구제가 그대로 돈다(lf06/hf06/
        # lf07/rtC2 = +107.3점 유지). 즉 "나빠지는 경로가 없다"는 폴백 원칙 복구다.
        yolo_lm_cams = {cid for cid, dets in detections.items()
                        if any(_landmark_ok(d.label, lm) for d in dets)}

        def _try_rescue(cam_id, img, allow_second_instance):
            existing = [d for d in detections.get(cam_id, []) if _landmark_ok(d.label, lm)]
            if existing and not allow_second_instance:
                return          # 이 카메라는 YOLO 가 이미 찾음 -- 스킵 (원래 동작)
            box = dino.best_box(img, phrases)
            if box is not None and float(box.conf[0]) < _LANDMARK_MIN_CONF:
                log.info("[fallback] DINO landmark %r on %s 기각 (conf=%.2f < %.2f, "
                         "오탐 의심 -- picnic_table 등 이 클래스 자체를 이 장면에서 "
                         "잘 못 구분함)", lm, cam_id, float(box.conf[0]),
                         _LANDMARK_MIN_CONF)
                box = None
            if box is None:
                return
            new_bbox = tuple(float(v) for v in box.xyxy[0])
            if existing and any(_iou(new_bbox, d.bbox) > _SAME_INSTANCE_IOU
                                for d in existing):
                return          # YOLO 가 이미 찾은 것과 같은 물건 -- 새 정보 없음
            detections.setdefault(cam_id, []).append(Detection(
                label=lm, bbox=new_bbox, score=float(box.conf[0])))
            tag = "추가(다른 인스턴스)" if existing else "구제"
            log.info("[fallback] DINO %s landmark %r on %s (conf=%.2f)",
                     tag, lm, cam_id, float(box.conf[0]))

        # 카메라 범위(게이트 확장 방지, 물체/사람 문제 공통): 정확히 1대로
        # 확정됐을 때만 "변별력이 높다"는 판단이 성립해(위 taxi 주석) 그
        # 밖은 안 연다. 0대/2대+ 면 이 판단이 성립하지 않으니 원래(2026-08-28)
        # 동작대로 전체 카메라를 연다 -- 안 그러면 YOLO 가 하나도 못 본 진짜
        # 정답 카메라(hf06 의 rig_3_b 처럼)까지 막힌다.
        allowed_cams = ({next(iter(yolo_lm_cams))} if len(yolo_lm_cams) == 1
                        else set(images))

        # "이미 하나 찾았으면 그 카메라는 스킵" 은 물체 문제에만 적용한다
        # (2026-08-28 원래 동작, hf06/lf06/lf07/rtC2 검증 완료 -- 코퍼스
        # 전체에 걸어 두면 이미 성공한 카메라에도 매번 DINO 가 돌아 느려진다:
        # 실측으로 이 스킵을 빼자 라운드 평균 6.2s -> 10.0s, 30초 초과
        # 3건 -> 4건). 사람 문제는 스킵하지 않고 "다른 인스턴스"를 계속
        # 찾는다 -- 2026-08-29 실측(sar01/sarH4, "sitting on top of the
        # trash can"): YOLO 가 그 카메라에서 trash_can 을 찾긴 했는데(3대
        # 모두에서!) 사람이 앉은 진짜 물건이 아니라 7.6m 떨어진 다른
        # 인스턴스였다. person 문제는 14/39 로 적어 비용이 코퍼스 전체에
        # 번지지 않는다.
        for cam_id in allowed_cams:
            _try_rescue(cam_id, images[cam_id], allow_second_instance=parsed.is_person)

    if parsed.is_person:
        return   # target(사람) 재탐지는 여전히 대상 밖 -- pose 분류가 필요해
                 # DINO 로는 못 한다. landmark 보강은 위에서 이미 끝났다.

    target_scope = images
    scope_cams: Optional[set] = None

    if lm:
        landmark_cams = {cid for cid, dets in detections.items()
                         if any(_landmark_ok(d.label, lm) for d in dets)}
        if landmark_cams:
            scope_cams = landmark_cams
            log.info("[fallback] target 탐색 범위를 landmark %r 검출된 %d/%d 카메라로 제한: %s",
                     lm, len(landmark_cams), len(images), sorted(landmark_cams))
        # landmark_cams 가 비면(끝내 못 찾음) 범위를 안 좁힌다 -- resolve() 도
        # 이 경우 게이트를 풀어(전체 카메라) 재탐색하므로 동작을 맞춘다.

    # 구역 게이트(area_hint) -- grounding.resolve() 의 판정(§1)을 그대로
    # 재사용해서 앞당긴다. landmark 게이트와 독립이라 둘 다 있으면 교집합을
    # 쓴다(resolve() 와 동일 원칙) -- 교집합이 비면 landmark 게이트로 되돌아가
    # 나빠지는 경로가 없다.
    indicators = AREA_INDICATOR_CLASSES.get(parsed.area_hint or "")
    if indicators:
        counts = {cid: sum(1 for d in detections.get(cid, []) if d.label in indicators)
                 for cid in images}
        top = max(counts.values(), default=0)
        if top > 0:
            leaders = {cid for cid, n in counts.items() if n == top}
            runner_up = max((n for cid, n in counts.items() if cid not in leaders),
                           default=0)
            if top >= AREA_DOMINANCE_RATIO * runner_up:
                if scope_cams is not None:
                    both = scope_cams & leaders
                    scope_cams = both if both else scope_cams
                else:
                    scope_cams = leaders
                log.info("[fallback] target 탐색 범위를 area_hint %r 지표(%s) 밀집 "
                         "카메라로 제한: %s", parsed.area_hint, sorted(indicators),
                         sorted(scope_cams))

    if scope_cams is not None:
        target_scope = {cid: img for cid, img in images.items() if cid in scope_cams}

    if target_phrases:
        for cam_id, img in target_scope.items():
            # 카메라별로 판단한다(2026-08-28 수정, landmark 와 같은 이유) --
            # 예전엔 "어느 카메라에서든 YOLO가 하나라도 찾으면" 전체를
            # 건너뛰었는데, target 클래스도(umbrella 등 YOLO 가 잘 아는
            # 클래스라도) 카메라마다 있고 없고가 갈릴 수 있다. YOLO가 다른
            # 카메라에서 이미 찾았다고 해서 정답 카메라에도 있다는 보장이
            # 없다 -- 그 카메라만 DINO 로 못 보고 넘어가면 애초에 후보에도
            # 못 들어간다.
            if any(_target_ok(d.label, parsed) for d in detections.get(cam_id, [])):
                continue                  # 이 카메라는 YOLO 가 이미 찾음 -- 스킵
            box = _dino_best_box_tiled(dino, img, target_phrases)
            if box is None:
                continue
            new_bbox = tuple(float(v) for v in box.xyxy[0])
            frame_area = float(img.shape[0] * img.shape[1])
            if _is_known_decoy(new_bbox, detections.get(cam_id, []), parsed.target_type, frame_area):
                log.info("[fallback] DINO target %r on %s 기각 (YOLO가 이미 다른 "
                         "클래스로 확신하는 박스와 겹침, conf=%.2f)",
                         parsed.target_type, cam_id, float(box.conf[0]))
                continue
            detections.setdefault(cam_id, []).append(Detection(
                label=parsed.target_type, bbox=new_bbox,
                score=float(box.conf[0])))
            log.info("[fallback] DINO rescued target %r on %s (conf=%.2f)",
                     parsed.target_type, cam_id, float(box.conf[0]))


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
