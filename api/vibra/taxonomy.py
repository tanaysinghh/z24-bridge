N_CLASSES = 17

CLASS_NAMES = [
    "Reference 1",
    "Settlement system installed",
    "Pier -20 mm",
    "Pier -40 mm",
    "Pier -80 mm",
    "Pier -95 mm",
    "Foundation tilt",
    "Reference 2",
    "Spalling 12 m2",
    "Spalling 24 m2",
    "Landslide",
    "Concrete hinge failure",
    "2 anchor heads failed",
    "4 anchor heads failed",
    "2 tendons ruptured",
    "4 tendons ruptured",
    "6 tendons ruptured",
]

GROUP_NAMES = ["Healthy", "Settlement", "Foundation", "Hinge", "Tendon"]

GROUP_MEMBERS = {
    "Healthy": [0, 1, 7],
    "Settlement": [2, 3, 4, 5],
    "Foundation": [6, 10],
    "Hinge": [8, 9, 11],
    "Tendon": [12, 13, 14, 15, 16],
}

GROUP_DESCRIPTIONS = {
    "Healthy": "No structural damage: both reference measurements and the state after the settlement system was installed",
    "Settlement": "Controlled lowering of the Koppigen pier by 20 to 95 mm",
    "Foundation": "Support and ground damage: foundation tilt after the pier was lifted, and the landslide at the abutment",
    "Hinge": "Concrete damage: soffit spalling and failure of the concrete hinge",
    "Tendon": "Post-tensioning system damage: failed anchor heads and ruptured tendons",
}

CLASS_TO_GROUP = [next(g for g, members in GROUP_MEMBERS.items() if c in members) for c in range(N_CLASSES)]
CLASS_TO_GROUP_INDEX = [GROUP_NAMES.index(g) for g in CLASS_TO_GROUP]

DSI = [
    0.00,
    0.02,
    0.20,
    0.30,
    0.45,
    0.55,
    0.50,
    0.05,
    0.35,
    0.45,
    0.55,
    0.65,
    0.70,
    0.75,
    0.80,
    0.90,
    1.00,
]

TIERS = [
    {
        "name": "Healthy",
        "lower": 0.0,
        "upper": 0.15,
        "action": "Routine monitoring. Keep the scheduled visual inspection cycle; no intervention needed.",
        "urgency": "Next scheduled inspection",
    },
    {
        "name": "Minor",
        "lower": 0.15,
        "upper": 0.40,
        "action": "Increase monitoring frequency and carry out a targeted visual inspection of the suspected damage location.",
        "urgency": "Within 6 months",
    },
    {
        "name": "Moderate",
        "lower": 0.40,
        "upper": 0.60,
        "action": "Detailed inspection with non-destructive testing, a structural assessment of load capacity and a repair plan.",
        "urgency": "Within 1 month",
    },
    {
        "name": "Severe",
        "lower": 0.60,
        "upper": 0.80,
        "action": "Urgent engineering inspection, consider load restrictions and plan strengthening of the affected members.",
        "urgency": "Within 1 week",
    },
    {
        "name": "Critical",
        "lower": 0.80,
        "upper": 1.0001,
        "action": "Restrict or close the bridge to traffic, inspect immediately and start emergency structural intervention.",
        "urgency": "Immediately",
    },
]


def tier_for(dsi):
    for k, tier in enumerate(TIERS):
        if tier["lower"] <= dsi < tier["upper"]:
            return k
    return len(TIERS) - 1


def setup_of_segment(segment_id):
    return (segment_id // 10) % 9 + 1
