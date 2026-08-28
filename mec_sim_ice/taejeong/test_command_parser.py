"""Tests for command_parser.

The parser itself is GENERAL (official vocabulary + linguistic rules, no
scenario references). This test uses two corpora:

  1. chungmu commands as a REALISTIC SAMPLE corpus (with reliable GT) —
     testing on them is not hardcoding; the parser never sees problem ids or
     coordinates, only sentences.
  2. NOVEL sentences that appear in no practice scenario, to verify the rules
     generalize beyond the sample (new objects, new landmarks, directional
     relations, paraphrases).

Invariants: target must match GT exactly; a confident relation must never
contradict GT ("no confident lie").

Runs with plain python3.
"""

from __future__ import annotations

from command_parser import parse_command

# (id, command, GT(target, landmark_name, relation),
#  expected parser (target, landmark, relation_hint, confident))
CASES = [
    ("lf01", "Find the mug on the picnic table.",
     ("mug", "picnic_table_01", "on"), ("mug", "picnic_table", "on", True)),
    ("lf02", "Find the tumbler on the bench.",
     ("tumbler", "bench_01", "on"), ("tumbler", "bench", "on", True)),
    ("lf03", "Find the tumbler near the fire hydrant in the parking lot.",
     ("tumbler", "hydrant_01", "near"), ("tumbler", "hydrant", "near", True)),
    ("lf04", "Find the sunblock beside the picnic table.",
     ("sunblock", "picnic_table_01", "beside"), ("sunblock", "picnic_table", "beside", True)),
    ("lf05", "Find the umbrella beside the vending machine.",
     ("umbrella", "vending_machine_01", "beside"), ("umbrella", "vending_machine", "beside", True)),
    ("lf06", "Find the umbrella next to the trash can.",
     ("umbrella", "trash_can_01", "near"), ("umbrella", "trash_can", "near", False)),
    ("lf07", "Find the pencil case near the picnic table.",
     ("pencilcase", "picnic_table_02", "near"), ("pencilcase", "picnic_table", "near", True)),
    ("lf08", "Find the umbrella dropped in the empty parking space.",
     ("umbrella", "police_car_01", "near"), ("umbrella", None, "near", False)),
    ("lf09", "Find the tumbler dropped in the empty parking space.",
     ("tumbler", "police_car_01", "near"), ("tumbler", None, "near", False)),
    ("lf10", "Find the tumbler on the road next to the taxi.",
     ("tumbler", "taxi_01", "near"), ("tumbler", "taxi", "near", False)),
    ("lf11", "Find the umbrella on the road next to the taxi.",
     ("umbrella", "taxi_01", "near"), ("umbrella", "taxi", "near", False)),
    ("hf01", "After working out I grabbed a drink from the machine and left my umbrella standing next to it.",
     ("umbrella", "vending_machine_01", "beside"), ("umbrella", "vending_machine", "near", False)),
    ("hf02", "I was taking notes at an outdoor table and left my pencil case behind.",
     ("pencilcase", "picnic_table_02", "near"), ("pencilcase", "picnic_table", "on", False)),
    ("hf03", "I was reading with a coffee at an outdoor table and left my mug there.",
     ("mug", "picnic_table_01", "on"), ("mug", "picnic_table", "on", False)),
    ("hf04", "I was resting in the sun at the park and left the sunblock I had been using.",
     ("sunblock", "picnic_table_01", "beside"), ("sunblock", None, None, False)),
    ("hf05", "After grabbing a drink from the vending machine I left my umbrella propped against its side.",
     ("umbrella", "vending_machine_01", "beside"), ("umbrella", "vending_machine", "beside", False)),
    ("hf06", "I stopped to throw something in the bin and left my umbrella leaning against the trash can.",
     ("umbrella", "trash_can_01", "near"), ("umbrella", "trash_can", "beside", False)),
    ("hf07", "I think I dropped my tumbler on the ground as I got out of my car in the parking lot.",
     ("tumbler", "police_car_01", "near"), ("tumbler", "car", "near", False)),
    ("hf08", "Getting out of my car I dropped my umbrella in the empty space next to it; find it in the parking lot.",
     ("umbrella", "police_car_01", "near"), ("umbrella", "car", "near", False)),
    ("rt01", "Pick up the tumbler dropped on the parking lot floor and bring it to the owner.",
     ("tumbler", "police_car_01", "near"), ("tumbler", None, "near", False)),
    ("rt02", "Pick up the umbrella in the empty parking space and bring it to the owner.",
     ("umbrella", "police_car_01", "near"), ("umbrella", None, None, False)),
    ("rt03", "Pick up the umbrella beside the vending machine and bring it to the owner.",
     ("umbrella", "vending_machine_01", "beside"), ("umbrella", "vending_machine", "beside", True)),
    ("rt04", "Pick up the tumbler on the road next to the taxi and bring it to the owner.",
     ("tumbler", "taxi_01", "near"), ("tumbler", "taxi", "near", False)),
    ("rt04b", "Pick up the umbrella on the road next to the taxi and bring it to the owner.",
     ("umbrella", "taxi_01", "near"), ("umbrella", "taxi", "near", False)),
    ("rtC2", "Pick up the pencil case near the picnic table and bring it to the owner.",
     ("pencilcase", "picnic_table_02", "near"), ("pencilcase", "picnic_table", "near", True)),
]

# SAR: (id, command, expected (landmark, pose_hint))
SAR_CASES = [
    ("sar01", "Find the person sitting on top of the trash can in the parking lot.",
     ("trash_can", "sitting")),
    ("sar02", "Find the person sitting on top of the vending machine.",
     ("vending_machine", "sitting")),
    ("sar03", "Find the person lying face-down on the bench near the taxi.",
     ("bench", "facedown")),
    ("sar04", "Find the person lying on top of the picnic table.",
     ("picnic_table", "lying")),
    ("sar06", "Find the person collapsed on the parking lot floor.",
     (None, "down")),
    ("sar07", "Find the person lying face-down and motionless in the parking lot.",
     (None, "facedown")),
    ("sar08", "Find the person fallen on the ground next to the toppled bicycle.",
     ("bicycle", "down")),
    ("sar09", "Find the person bent over and motionless near the trash can in the park rest area.",
     ("trash_can", "bent")),
    ("sarH1", "People are relaxing at the park and one is even lying on the grass; find the one near the trash can who is bent over and not moving and actually needs help.",
     ("trash_can", "bent")),
    ("sarH2", "In the parking lot a person is down next to a bicycle and looks like they had an accident; find them.",
     ("bicycle", "down")),
    ("sarH3", "Someone is up where people should not be; find the person who climbed on top of the drink machine.",
     ("vending_machine", "sitting")),
    ("sarH4", "In the parking lot, find the person who climbed up and is sitting on top of the trash can.",
     ("trash_can", "sitting")),
    ("sarH5", "Near the taxi, find the person lying down on top of the bench.",
     ("bench", "lying")),
    ("sarH6", "Several people are in the parking lot; one is completely face-down and immobile; find the most critical one.",
     (None, "facedown")),
]

# 일반화 검증: 어느 연습 시나리오에도 없는 신규 문장 (공식 로스터의 다른 클래스)
NOVEL_CASES = [
    ("Find the juice box on the bench.", ("juice", "bench", "on", True)),
    ("Find the tissue near the postbox.", ("tissue", "postbox", "near", True)),
    ("Pick up the paper cup next to the kick scooter and bring it to the owner.",
     ("disposable_cup", "kick_scooter", "near", False)),
    ("Find the cracker box behind the vending machine.",
     ("cracker_box", "vending_machine", "behind", True)),
    ("Find the cola can to the left of the mailbox.",
     ("cola_can", "postbox", "left", True)),
    ("I bought a coke and left the can beside the patrol car.",
     ("cola_can", "police_car", "beside", True)),
    ("Find the sunscreen in front of the bike.",
     ("sunblock", "bicycle", "front", True)),
    # -- "behind" 중의성: 구동사(leave ... behind) vs 공간관계 --
    # 공간 behind (leave-동사 없음) → confident 유지
    ("The tumbler fell behind the bench.",
     ("tumbler", "bench", "behind", True)),
    # 구동사+명사구 중의("left ... behind the X") → behind 힌트는 주되 confident 낮춤
    ("I left my pencil case behind the vending machine.",
     ("pencilcase", "vending_machine", "behind", False)),
    # 구동사 문말/비명사구("behind when ...") → behind 로 뽑지 않음
    ("I left my umbrella behind when I got up from the bench.",
     ("umbrella", "bench", None, False)),
]


def main() -> int:
    fails = 0
    CAR_TYPES = {"police_car", "taxi", "suv", "sports_car", "sports_car_2",
                 "normal_car_1", "normal_car_2"}

    def gt_cls(name):
        import re
        return re.sub(r"_\d+$", "", name)

    for cid, cmd, gt, exp in CASES:
        p = parse_command(cmd)
        got = (p.target_type, p.landmark_type, p.relation_hint, p.relation_confident)
        if got != exp:
            print(f"FAIL {cid}: got {got}, expected {exp}")
            fails += 1
        if p.target_type != gt[0]:
            print(f"FAIL {cid}: target {p.target_type} != GT {gt[0]}")
            fails += 1
        if p.relation_confident and p.relation_hint != gt[2]:
            print(f"FAIL {cid}: confident relation {p.relation_hint} != GT {gt[2]}")
            fails += 1
        if p.landmark_type is not None:
            ok = (p.landmark_type == gt_cls(gt[1]) or
                  (p.landmark_type == "car" and gt_cls(gt[1]) in CAR_TYPES))
            if not ok:
                print(f"FAIL {cid}: landmark {p.landmark_type} contradicts GT {gt[1]}")
                fails += 1
        if p.is_person or (cid.startswith("rt") != p.stage2):
            print(f"FAIL {cid}: person/stage2 flags wrong")
            fails += 1
    print(f"object corpus: {len(CASES)} checked")

    for cid, cmd, (exp_lm, exp_pose) in SAR_CASES:
        p = parse_command(cmd)
        if not p.is_person or p.target_type != "person" or p.stage2:
            print(f"FAIL {cid}: person/stage2 flags wrong")
            fails += 1
        if (p.landmark_type, p.pose_hint) != (exp_lm, exp_pose):
            print(f"FAIL {cid}: ({p.landmark_type}, {p.pose_hint}) != "
                  f"({exp_lm}, {exp_pose})")
            fails += 1
    print(f"SAR corpus: {len(SAR_CASES)} checked")

    for cmd, exp in NOVEL_CASES:
        p = parse_command(cmd)
        got = (p.target_type, p.landmark_type, p.relation_hint, p.relation_confident)
        if got != exp:
            print(f"FAIL novel: {cmd!r} -> {got}, expected {exp}")
            fails += 1
    print(f"novel sentences: {len(NOVEL_CASES)} checked")

    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
