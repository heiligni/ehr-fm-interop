from utility.codes import RACE_CODE

rules = [
    {"code": RACE_CODE + "UNKNOWN", "match_type": "full_match", "remove": True},
    {
        "code": RACE_CODE + "UNABLE TO OBTAIN",
        "match_type": "full_match",
        "remove": True,
    },
    {
        "code": RACE_CODE + "WHITE",
        "match_type": "start",
        "replace": RACE_CODE + "WHITE",
    },
    {"code": RACE_CODE + "PORTUGUESE", "match_type": "full_match", "remove": True},
    {
        "code": RACE_CODE + "ASIAN",
        "match_type": "start",
        "replace": RACE_CODE + "ASIAN",
    },
    {
        "code": RACE_CODE + "PATIENT DECLINED TO ANSWER",
        "match_type": "full_match",
        "remove": True,
    },
    {
        "code": RACE_CODE + "HISPANIC OR LATINO",
        "match_type": "full_match",
        "replace": RACE_CODE + "HISPANIC/LATINO",
    },
    {"code": RACE_CODE + "OTHER", "match_type": "full_match", "remove": True},
    {
        "code": RACE_CODE + "HISPANIC/LATINO",
        "match_type": "start",
        "replace": RACE_CODE + "HISPANIC/LATINO",
    },
    {
        "code": RACE_CODE + "BLACK",
        "match_type": "start",
        "replace": RACE_CODE + "BLACK/AFRICAN AMERICAN",
    },
]
