"""Detection class taxonomy for the real VLA grounding model (YOLO11 DET_CLASSES).

Source: 정리폴더/MARC2026_객체_상태_분류.md (실측 dataset-gen 씬 1개 + mock_demo_data.yaml
교차검증), cross-checked against the dataset-gen "New Scene" object-selection panel
(Objects/Landmarks checklist screenshot) -- that panel lists 10 objects and 15 landmarks,
which is what TARGET_CLASSES/LANDMARK_CLASSES below now match exactly. POSE_CLASSES (9) also
matches the panel's stated pose count. Still worth re-confirming once more scenes are
generated (see that doc's warning) -- this file is the definitive list until proven otherwise.

Both `train_yolo11.py` (label conversion) and the eventual `record_agent_vla.py` (real-time
inference) must import from here so the two stay in lock-step.
"""

# Lost-item target objects (`target_type` in GroundingResult). 10 -- matches the dataset-gen
# Objects panel exactly (added `juice`, missing from the earlier single-scene sample).
TARGET_CLASSES = [
    "mug", "tumbler", "sunblock", "umbrella", "pencilcase", "cola_can",
    "cracker_box", "disposable_cup", "tissue", "juice",
]

# Reference landmarks (`landmark` in GroundingResult). 15 -- matches the dataset-gen
# Landmarks panel exactly. "parking"/"park"/"bikerack" are deliberately excluded: they are
# NOT selectable options in that panel either, confirming they're area/location references
# in voice_command text (e.g. "in the parking lot"), not standalone objects.
LANDMARK_CLASSES = [
    "picnic_table", "bench", "vending_machine", "trash_can",
    "police_car", "taxi", "bicycle", "hydrant",
    "normal_car_1", "normal_car_2", "sports_car", "sports_car_2", "suv",
    "kick_scooter", "postbox",
]

# Person pose subclasses -- the detector classifies these directly (no separate
# pose-estimation pass). `situation` (accident/emergency/abnormal/normal) is derived
# from the detected pose class *plus* voice_command context, not from pose alone --
# the same pose maps to different situations depending on the command's wording.
# 9 poses -- matches the dataset-gen panel's stated pose count. Character model (male/
# female student, professor, ...) does NOT create separate classes -- only pose does; the
# character identity lives in `instance_id` (e.g. "..._person_walking_male_student_01_0"),
# never in `class`. Confirmed from one character (male_student) in the sampled scene --
# worth re-checking once a scene with other characters is generated.
POSE_CLASSES = [
    "person_walking", "person_walking_away", "person_crouching",
    "person_bending", "person_reclining", "person_prone", "person_lying_down",
    "person_reaching", "person_sitting",
]

# One detector model covers all three groups. Order matters: it fixes the label id in
# the converted YOLO dataset (data.yaml `names:`). If you add/reorder classes AFTER
# running `train_yolo11.py convert`, the existing labels/*.txt files are invalidated --
# re-run convert (and retrain) from scratch instead of patching ids in place.
DET_CLASSES = TARGET_CLASSES + LANDMARK_CLASSES + POSE_CLASSES

# Command-parsing vocabulary (not detector classes -- these never get a bbox).
RELATION_WORDS = ["near", "beside", "behind", "front", "left", "right", "on", "above"]
SITUATION_WORDS = ["accident", "emergency", "abnormal", "normal"]
