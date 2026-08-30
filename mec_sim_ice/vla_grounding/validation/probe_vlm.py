"""진단: 소형 LM(Qwen) 폴백 5개 경로가 실제로 도는지 + 무엇을 답하는지.

## 왜 필요한가

vlm_fallback.py 는 기본 OFF 다 -- chungmu 39문항 전부가 command_parser 의 닫힌
동의어 사전 안에서 풀려서 5개 트리거 조건이 **한 번도 성립하지 않기 때문**이다
(이 파일의 --list 로 확인 가능). 그래서 지금까지 이 모듈은 "돌려본 적이 없는
코드"였고, vlm_fallback.py 모듈 docstring 의 TODO 세 줄이 그 상태를 적어 둔 것이다:

  (1) 소형 모델이 5개 프롬프트에 대해 JSON 을 제대로 뱉는가
  (2) 판정 품질 (특히 OOV 물체명 -> 클래스 복구)
  (3) 실제 VRAM 점유와 호출당 지연 (라이브 예산 30초)

본선에서 닫힌 어휘 밖 문항이 나오면 MARC_ENABLE_VLM=1 로 켜야 하는데, 그때
처음 돌려보는 건 도박이다. 이 스크립트는 **일부러 트리거를 성립시키는 명령
코퍼스**를 만들어서 세 가지를 GPU 데스크탑에서 미리 재 둔다.

파싱 실패 시 resolve()/classify_*() 가 조용히 None 을 돌려주므로(호출부는 기존
동작 유지), 모델이 형식을 못 지켜도 **아무 에러 없이 무력화된다** -- 그래서
"켰는데 아무 일도 안 일어남"을 로그로는 못 잡는다. 여기서 raw 출력을 직접
찍어 보는 이유다.

## 명령 코퍼스의 근거

전부 파서 규칙이 막다른 곳(None)에 닿도록 만든 문장이고, 씬을 참조하지 않는다
(본선에서 나올 법한 표현 일반화 -- 특정 레이아웃에 종속된 상수가 없다).
`expect` 는 "이 정도는 맞혀야 한다"는 기준이지 정답 강제가 아니다 -- None 은
"모델이 뭘 내든 형식만 맞으면 통과"를 뜻한다.

주의: "thermos"/"parasol" 같은 표현은 이미 OBJECT_SYNONYMS 에 들어 있어
파서가 혼자 푼다(각각 tumbler/umbrella). 즉 **사전에 없는 단어를 고르는 게
생각보다 어렵다** -- 그 자체가 좋은 소식이라 코퍼스에 그 사실을 남겨 둔다.

## 실행

    # GPU 없이: 어느 경로가 걸리는지만 (파서만 돌림, torch 불필요)
    python3 mec_sim_ice/vla_grounding/validation/probe_vlm.py --list

    # GPU 데스크탑: 실제로 모델을 올려 5경로 전부 호출 + 지연/VRAM 실측
    MARC_ENABLE_VLM=1 python3 mec_sim_ice/vla_grounding/validation/probe_vlm.py --run

    # 다른 모델과 비교 (예: 0.5B 로 낮춰도 되는지)
    MARC_ENABLE_VLM=1 MARC_VLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \\
        python3 mec_sim_ice/vla_grounding/validation/probe_vlm.py --run

    # chungmu 39문항이 정말 0회 호출인지 재확인 (회귀 감시)
    python3 mec_sim_ice/vla_grounding/validation/probe_vlm.py --corpus

라이브 예산이 문항당 30초이므로, 한 문항에서 이 경로가 최대 3번까지 겹칠 수
있다는 점(landmark + person/pose + target)을 --run 결과의 합계로 확인할 것.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_VLA_GROUNDING_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _VLA_GROUNDING_DIR)

from command_parser import (LANDMARK_SYNONYMS, OBJECT_SYNONYMS,  # noqa: E402
                           POSE_BUCKETS, parse_command)

_POSE_BUCKET_NAMES = [name for name, _ in POSE_BUCKETS]


def triggered_paths(p) -> list:
    """이 ParsedCommand 가 vlm_assist() 의 어느 경로를 깨우는가.

    조건은 fallback.py.vlm_assist() 와 1:1 로 맞춰 둔다 -- 그쪽이 바뀌면 여기도
    바뀌어야 하고, 안 맞으면 --corpus 가 거짓 안심을 준다.
    """
    out = []
    if not p.is_person and p.relation_hint == "behind" and not p.relation_confident:
        out.append("behind")
    if p.landmark_type is None and not p.area_hint and not p.open_space:
        out.append("landmark")
    if p.target_type is None and not p.is_person:
        out.append("is_person")
        out.append("target")       # 같은 조건 -- is_person 이 True 를 내면 target 은 건너뜀
    if p.is_person and p.pose_hint is None:
        out.append("pose")
    return out


# (명령, 노리는 경로, expect) -- expect 는 그 경로가 내야 할 값. None = 형식만 본다.
PROBES = [
    # --- behind 중의: "leave behind"(구동사) vs 공간관계 ---
    # 두 문장 모두 파서에는 똑같이 "left ... behind the X" 로 보여 confident=False
    # 로 넘어온다 -- 갈라주는 건 뜻뿐이라 정확히 이 모델이 할 일이다. 첫 문장은
    # 구동사(두고 왔다 -> near), 둘째는 공간관계(자판기 뒤쪽 -> behind 유지)여야 한다.
    ("I left my mug behind the bench near the entrance.", "behind", False),
    ("I left my umbrella behind the vending machine, right against its back panel.",
     "behind", True),

    # --- landmark OOV: 사전에 없는 기준물 ---
    # LANDMARK_SYNONYMS 11종 밖의 기준물. 모델은 클래스 복구를 못 하므로
    # search_phrase(1~4단어)를 내야 하고, 그게 DINO 검색어로 그대로 쓰인다.
    ("Find the tumbler on the water fountain.", "landmark", None),
    ("Find the mug next to the flower pot.", "landmark", None),
    ("Find the umbrella leaning against the lamp post.", "landmark", None),

    # --- is_person: 물체어도 사람어도 못 찾은 진짜 중의 ---
    # "someone" 은 WEAK_PERSON_WORD 라 pose 단서 없이는 is_person 이 안 선다 --
    # 즉 사람 문항인데 파서가 물체 문항처럼 넘기는 실제 구멍이다.
    ("Find the individual who appears to need assistance.", "is_person", True),
    ("Someone in a red jacket is waving for help.", "is_person", True),
    ("Find the guy slumped against the wall.", "is_person", True),
    # 반대 방향(물체인데 사람으로 오분류하면 안 됨) -- 오탐 감시용.
    ("Locate my spectacles on the bench.", "is_person", False),

    # --- target OOV: 사전에 없는 물체명 -> 클래스 복구 or 검색어 ---
    # 앞의 spectacles 는 우리 34클래스에 정말 없으므로 search_phrase 를 기대한다.
    ("Locate my spectacles on the bench.", "target", None),
    # 사전 미등재 표현이지만 실은 아는 클래스인 경우 -- 복구를 기대한다.
    # ("vacuum flask" 는 안 쓴다 -- "flask" 부분문자열이 이미 tumbler 에 걸려
    #  파서가 혼자 푼다. 사전에 정말 없는 단어를 고르는 게 생각보다 어렵다.)
    ("I misplaced my canteen by the picnic table.", "target", "tumbler"),
    ("Find the bumbershoot someone left at the bike rack.", "target", "umbrella"),

    # --- pose OOV: POSE_BUCKETS 정규식 6종이 못 잡은 자세 표현 ---
    ("Find the person sprawled out on the pavement.", "pose", "lying"),
    ("Find the person crouching by the trash can.", "pose", "bent"),
    ("Find the person curled up next to the bench.", "pose", "lying"),
]


def _fmt(v):
    return "-" if v is None else repr(v)


def cmd_list() -> int:
    """모델 없이: 각 프로브가 실제로 노린 경로를 깨우는지만 확인."""
    bad = 0
    print(f"{'노린 경로':10s} {'실제 깨운 경로':32s} 명령")
    for cmd, want, _ in PROBES:
        got = triggered_paths(parse_command(cmd))
        ok = want in got
        bad += not ok
        mark = " " if ok else "!"
        print(f"{mark}{want:9s} {','.join(got) or '(없음)':32s} {cmd}")
    if bad:
        print(f"\n{bad}건이 노린 경로를 못 깨웠다 -- 파서가 이미 그 문장을 "
              "풀어버린다는 뜻이다. 프로브 문장을 바꾸거나, 그 경로는 "
              "애초에 안 걸리는 것으로 기록할 것.")
    else:
        print("\n전부 노린 경로를 깨운다 -- --run 으로 실제 모델 판정을 볼 차례.")
    return 1 if bad else 0


def cmd_corpus() -> int:
    """chungmu 39문항이 정말 0회 호출인지 재확인 (회귀 감시).

    이게 깨지면(호출 > 0) 제출 기본값(VLM OFF)의 근거가 바뀐 것이다 --
    파서를 고쳤거나 시나리오가 바뀐 경우다. 어느 쪽이든 알아야 한다.
    """
    import yaml
    path = os.path.join(_VLA_GROUNDING_DIR, "content-image-2026.R03",
                        "marc2026_chungmu.yaml")
    if not os.path.exists(path):
        print(f"시나리오 파일 없음: {path}")
        return 2
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    cmds = []

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("voice_command"), str):
                cmds.append((o.get("id", "?"), o["voice_command"]))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(doc)
    hits = [(pid, c, triggered_paths(parse_command(c))) for pid, c in cmds]
    fired = [h for h in hits if h[2]]
    print(f"chungmu {len(cmds)}문항 중 소형 LM 이 걸리는 문항: {len(fired)}")
    for pid, c, paths in fired:
        print(f"  {pid:28s} {','.join(paths):24s} {c}")
    if not fired:
        print("  (0건 -- 제출 기본값 VLM OFF 의 근거가 그대로 유효하다)")
    return 0


def cmd_run() -> int:
    """GPU 데스크탑: 실제로 모델을 올려 5경로를 호출하고 지연/VRAM 을 잰다."""
    from vlm_fallback import VlmTargetResolver, is_disabled

    if is_disabled():
        print("소형 LM 이 비활성화 상태다. MARC_ENABLE_VLM=1 로 실행할 것 "
              "(MARC_DISABLE_VLM 이 켜져 있으면 그게 우선한다).")
        return 2

    try:
        import torch
    except ImportError:
        torch = None

    t0 = time.time()
    vlm = VlmTargetResolver()
    load_s = time.time() - t0
    model_name = vlm.backend.model_name
    device = str(getattr(vlm.backend.model, "device", "?"))
    print(f"모델: {model_name}  device={device}  로드 {load_s:.1f}s")
    if torch is not None and torch.cuda.is_available():
        print(f"로드 직후 VRAM: {torch.cuda.memory_allocated() / 2**30:.2f} GiB "
              f"할당 / {torch.cuda.memory_reserved() / 2**30:.2f} GiB 예약")
        torch.cuda.reset_peak_memory_stats()

    obj_classes = list(OBJECT_SYNONYMS.keys())
    lm_classes = list(LANDMARK_SYNONYMS.keys())

    rows, total_s, bad = [], 0.0, 0
    for cmd, want, expect in PROBES:
        parsed = parse_command(cmd)
        if want not in triggered_paths(parsed):
            rows.append((want, cmd, "SKIP", "파서가 이미 풀어버림", 0.0))
            continue

        t0 = time.time()
        if want == "behind":
            got = vlm.behind_is_spatial(cmd)
        elif want == "landmark":
            matched, phrase = vlm.resolve_landmark(cmd, lm_classes)
            got = matched or phrase
        elif want == "is_person":
            got = vlm.classify_person(cmd)
        elif want == "target":
            matched, phrase = vlm.resolve(cmd, obj_classes)
            got = matched or phrase
        elif want == "pose":
            got = vlm.classify_pose(cmd, _POSE_BUCKET_NAMES)
        else:
            continue
        dt = time.time() - t0
        total_s += dt

        # None = 파싱 실패(형식 이탈) -- 조용히 무력화되는 그 경우다.
        if got is None:
            verdict, note = "FAIL", "None (JSON 파싱 실패 또는 판정 불가)"
            bad += 1
        elif expect is None:
            verdict, note = "OK", f"-> {_fmt(got)}"
        elif got == expect:
            verdict, note = "OK", f"-> {_fmt(got)}"
        else:
            verdict, note = "DIFF", f"-> {_fmt(got)} (기대 {_fmt(expect)})"
        rows.append((want, cmd, verdict, note, dt))

    print()
    print(f"{'경로':10s} {'':5s} {'초':>6s}  결과")
    for want, cmd, verdict, note, dt in rows:
        print(f"{want:10s} {verdict:5s} {dt:6.2f}  {note}")
        print(f"{'':23s}{cmd}")
    called = [r for r in rows if r[2] != "SKIP"]
    if called:
        print(f"\n호출 {len(called)}건, 합계 {total_s:.1f}s, "
              f"호출당 평균 {total_s / len(called):.2f}s")
        print("라이브 한 문항에서 이 경로가 최대 3번 겹칠 수 있다 "
              "(landmark + is_person/pose + target) -- 평균 x3 이 "
              "30초 예산에서 DINO 몫을 얼마나 깎는지 볼 것.")
    if torch is not None and torch.cuda.is_available():
        print(f"추론 중 VRAM 피크: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if bad:
        print(f"\n{bad}건이 None 을 냈다 -- 이 경우 호출부는 **아무 에러 없이** "
              "기존 동작을 유지한다(조용히 무력화). 모델을 키우거나 프롬프트를 "
              "고치지 않으면, 켜 두는 것이 VRAM 만 쓰고 아무 일도 안 하는 상태다.")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true",
                   help="모델 없이: 각 프로브가 노린 경로를 깨우는지만 확인")
    g.add_argument("--corpus", action="store_true",
                   help="chungmu 39문항이 여전히 0회 호출인지 재확인")
    g.add_argument("--run", action="store_true",
                   help="GPU: 실제 모델 호출 + 지연/VRAM 실측")
    args = ap.parse_args()
    if args.list:
        return cmd_list()
    if args.corpus:
        return cmd_corpus()
    return cmd_run()


if __name__ == "__main__":
    raise SystemExit(main())
