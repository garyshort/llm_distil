#!/usr/bin/env python3
"""
Generate training_data.jsonl with prompt/completion pairs for insurance damage extraction.
Completions are correct by construction: we choose damage flags and severity first,
then build a narrative that describes exactly those conditions.
"""

import json
import random
from pathlib import Path

PROMPT_PREFIX = (
    "You are an insurance damage extraction system.\n\n"
    "Extract damage indicators from the narrative and return ONLY valid JSON matching the schema.\n\n"
    "NARRATIVE:\n"
)

# Building blocks for narratives. Each key matches a damage field.
# When the flag is True we pick from the "true" list; when False from the "false" list.
NARRATIVE_BLOCKS = {
    "floor_water_damage": {
        True: [
            "Water came into the basement and covered the floor.",
            "Water entered the basement and pooled on the floor.",
            "There was standing water across the basement floor.",
            "The basement flooded and water covered most of the floor.",
            "Water seeped in and left several inches on the floor.",
            "We had water across part of the basement floor.",
            "Water came in and covered most of the basement floor.",
            "The floor had water standing in the basement.",
            "Water spread across the basement floor.",
            "The basement had water on the floor after the storm.",
        ],
        False: [
            "The basement floor was dry.",
            "We did not get standing water on the floor.",
            "The floor stayed dry.",
            "No water reached the basement floor.",
            "The basement stayed mostly dry.",
            "We checked and the floor was dry.",
        ],
    },
    "carpet_damage": {
        True: [
            "The carpet is soaked and will likely need to be replaced.",
            "The carpet is completely soaked.",
            "The carpet is wet and smells musty.",
            "The carpet got wet and feels damp.",
            "The carpet is ruined and smells of mildew.",
            "The carpet is soaked through.",
            "Part of the carpet is wet and damp.",
            "The carpet near the wall is wet.",
            "The carpet is wet and giving off a mildew odor.",
            "The carpet feels damp and may be damaged.",
        ],
        False: [
            "The carpet is dry and does not appear damaged.",
            "The carpet is not soaked and looks okay.",
            "No carpet damage was found.",
            "The carpet appears dry.",
            "The carpet was not affected.",
            "We did not find any carpet damage.",
        ],
    },
    "broken_plaster": {
        True: [
            "The drywall is soft at the bottom and the plaster is damaged.",
            "The plaster is clearly damaged and the drywall is crumbling.",
            "The lower drywall feels soft and the plaster is breaking apart.",
            "The drywall at the bottom is falling apart and plaster is damaged.",
            "There is minor plaster damage and the drywall feels soft.",
            "The plaster is damaged in several areas.",
            "The drywall feels soft near the bottom and plaster is damaged.",
            "The lower wall plaster is damaged.",
            "The drywall is crumbling and the plaster is breaking apart.",
            "We are seeing plaster damage and soft drywall.",
        ],
        False: [
            "The drywall and plaster look intact.",
            "I did not see any plaster or drywall damage.",
            "The walls and plaster look normal.",
            "The plaster and drywall look solid.",
            "No soft drywall or plaster damage was found.",
            "The drywall and plaster appear unaffected.",
        ],
    },
    "mould": {
        True: [
            "We are starting to see mold forming along the wall.",
            "There are patches of mold visible.",
            "Mold is forming in several areas.",
            "We have noticed mold growing.",
            "There is visible mold on the walls.",
            "Small patches of mold have appeared.",
            "Mold has started to form.",
            "We see mold developing.",
            "There is mold along the wall.",
            "Mold is visible in the affected area.",
        ],
        False: [
            "I do not see any visible mold yet.",
            "No mold has been noticed.",
            "We have not seen any mold.",
            "There is no mold visible.",
            "No mold was found.",
            "We did not find any mold.",
        ],
    },
    "odor_present": {
        True: [
            "There is a strong musty smell.",
            "There is a noticeable musty odor.",
            "There is a slightly musty smell.",
            "The area smells musty.",
            "There is a faint damp smell.",
            "There is a mildew odor.",
            "We notice a musty odor.",
            "There is a strong mildew smell.",
            "There is a musty smell developing.",
            "A damp smell is present.",
        ],
        False: [
            "There are no unusual smells.",
            "No odors were noticed.",
            "We did not notice any musty smell.",
            "There is no unusual odor.",
            "No musty or damp smell.",
            "Odors are not present.",
        ],
    },
    "electrical_damage": {
        True: [
            "One electrical outlet near the floor sparked briefly.",
            "Several outlets near the floor stopped working.",
            "We had an outlet spark when we first noticed the water.",
            "Some electrical outlets are not working.",
            "There was a brief spark from an outlet.",
            "Electrical outlets near the floor have failed.",
            "We noticed an outlet sparking.",
            "One or two outlets are not functioning.",
        ],
        False: [
            "Electrical outlets seem to be working normally.",
            "Electrical outlets appear to still work.",
            "We have not had any electrical problems.",
            "The electrical outlets are functioning properly.",
            "No electrical issues were noted.",
            "Electrical systems appear unaffected.",
        ],
    },
    "ceiling_damage": {
        True: [
            "The ceiling has water stains and damage.",
            "The ceiling is stained and shows water damage.",
            "There is visible ceiling damage.",
            "The ceiling was affected by the water.",
            "We see ceiling stains and damage.",
        ],
        False: [
            "The ceiling looks fine with no stains.",
            "The ceiling upstairs appears unaffected.",
            "The ceiling looks okay.",
            "The ceiling is still fine.",
            "No ceiling damage was found.",
            "The ceiling appears normal.",
        ],
    },
    "cabinet_damage": {
        True: [
            "The storage cabinets have swollen at the base.",
            "The cabinets are warped at the bottom.",
            "Cabinets have swollen from the water.",
            "The cabinets are damp and warped.",
            "One or more cabinets are warped.",
            "The cabinets at the base are damaged.",
            "The wooden cabinets have swollen.",
            "Cabinets are warped from the water.",
        ],
        False: [
            "The cabinets got a bit damp but do not appear warped.",
            "Cabinets are slightly damp but not damaged.",
            "The storage cabinets appear unaffected.",
            "Cabinets and appliances were not impacted.",
            "Cabinets seem fine.",
            "No cabinet damage was found.",
        ],
    },
    "appliance_damage": {
        True: [
            "Our washer in the basement is showing an error and may have water damage.",
            "One appliance will not power on.",
            "The washer may have water damage.",
            "An appliance is not working properly.",
            "One of the basement appliances has failed.",
            "Our washer is showing an error.",
            "At least one appliance appears damaged.",
        ],
        False: [
            "Appliances in the basement seem to be running okay.",
            "Appliances appear fine.",
            "Appliances seem unaffected.",
            "No appliance damage was noted.",
            "The appliances are working normally.",
            "Cabinets and appliances were not affected.",
        ],
    },
    "structural_crack": {
        True: [
            "There is a new crack in the foundation wall.",
            "We noticed a structural crack that appeared after the water.",
            "A crack has developed in the wall.",
            "There is a crack that we believe is related to the water.",
        ],
        False: [
            "There is an old hairline crack near the window that existed before the storm.",
            "We did not see any new cracks.",
            "No structural cracks were found.",
            "No new cracks have appeared.",
            "Structural elements appear intact.",
        ],
    },
}

# Order to assemble narrative (so it reads naturally)
FIELD_ORDER = [
    "floor_water_damage",
    "carpet_damage",
    "broken_plaster",
    "mould",
    "odor_present",
    "electrical_damage",
    "ceiling_damage",
    "cabinet_damage",
    "appliance_damage",
    "structural_crack",
]


def make_damage_and_severity(rng: random.Random) -> tuple[dict, str]:
    """Choose damage flags with realistic correlations, then set severity."""
    # Base rates and correlations: floor water often leads to carpet, etc.
    has_floor = rng.random() < 0.75
    has_carpet = has_floor and rng.random() < 0.85
    has_plaster = has_floor and rng.random() < 0.6
    has_mould = has_floor and rng.random() < 0.5
    has_odor = has_floor and rng.random() < 0.8
    has_electrical = has_floor and rng.random() < 0.25
    has_ceiling = rng.random() < 0.2
    has_cabinet = has_floor and rng.random() < 0.4
    has_appliance = has_floor and rng.random() < 0.3
    has_crack = rng.random() < 0.15

    damage = {
        "broken_plaster": has_plaster,
        "mould": has_mould,
        "floor_water_damage": has_floor,
        "electrical_damage": has_electrical,
        "ceiling_damage": has_ceiling,
        "structural_crack": has_crack,
        "carpet_damage": has_carpet,
        "cabinet_damage": has_cabinet,
        "appliance_damage": has_appliance,
        "odor_present": has_odor,
    }
    true_count = sum(damage.values())
    if true_count <= 2:
        severity = "low"
    elif true_count <= 5:
        severity = "moderate"
    else:
        severity = "high"
    return damage, severity


def build_narrative(damage: dict, rng: random.Random) -> str:
    """Build a short narrative that matches the damage flags exactly."""
    parts = []
    for field in FIELD_ORDER:
        block = NARRATIVE_BLOCKS[field][damage[field]]
        parts.append(rng.choice(block))
    # Join with spaces; add a simple intro sometimes
    intro = rng.choice(
        [
            "Hi, I'm reporting damage from the recent flooding. ",
            "Calling about water damage in the basement. ",
            "We had water in the basement and I'm reporting the damage. ",
            "Hi, calling about some water that came into the basement. ",
            "I'm calling to report basement water damage. ",
            "",
        ]
    )
    return intro + " ".join(parts)


def main() -> None:
    out_path = Path(__file__).resolve().parent / "training_data.jsonl"
    rng = random.Random(42)
    count = 1000

    with open(out_path, "w") as f:
        for _ in range(count):
            damage, severity = make_damage_and_severity(rng)
            narrative = build_narrative(damage, rng)
            prompt = PROMPT_PREFIX + narrative
            completion_obj = {"damage": damage, "overall_severity": severity}
            completion_str = json.dumps(completion_obj, separators=(",", ":"))
            line = json.dumps({"prompt": prompt, "completion": completion_str})
            f.write(line + "\n")

    print(f"Wrote {count} examples to {out_path}")


if __name__ == "__main__":
    main()
