"""Pure Python implementations of the 10 pilot calculators' formulas.

This is `demo_med_multi`'s M4 step: instead of asking an LLM to do the final
arithmetic (which is exactly where DeMo-Med's earlier QTc pilot broke down -
the model can describe a formula correctly in prose and still botch applying
it), M2/M3 extract and verify structured entities, and this module computes
the answer directly from those entities. No LLM call, no arithmetic slip.

Every function takes a dict of entities (as produced by the demo_med_extract
prompt) and returns (value, unit). Raises KeyError/ValueError/ZeroDivisionError
on missing/invalid entities - run_experiment.py treats any exception here as
"formula execution failed" and abstains, rather than guessing.
"""

GENDER_COEFFICIENTS = {"male": 1.0, "female": 0.85}


def _sex(entities: dict) -> str:
    sex = str(entities["sex"]).strip().lower()
    if sex not in ("male", "female"):
        raise ValueError(f"unrecognized sex: {sex!r}")
    return sex


def ideal_body_weight_kg(sex: str, height_cm: float) -> float:
    """Devine formula. Verified against MedCalc-Bench ground_truth_explanation."""
    height_in = height_cm * 0.393701
    base = 50.0 if sex == "male" else 45.5
    return base + 2.3 * (height_in - 60)


def cockcroft_gault(entities: dict):
    age = float(entities["age"])
    sex = _sex(entities)
    height_cm = float(entities["height_cm"])
    weight_kg = float(entities["weight_kg"])
    creatinine_mg_dl = float(entities["serum_creatinine_mg_dl"])

    height_m = height_cm / 100
    bmi = weight_kg / (height_m**2)
    ibw = ideal_body_weight_kg(sex, height_cm)

    if bmi < 18.5:  # underweight
        adjusted_weight = weight_kg
    elif bmi < 25:  # normal
        adjusted_weight = min(ibw, weight_kg)
    else:  # overweight/obese
        adjusted_weight = ibw + 0.4 * (weight_kg - ibw)

    gender_coefficient = GENDER_COEFFICIENTS[sex]
    crcl = ((140 - age) * adjusted_weight * gender_coefficient) / (72 * creatinine_mg_dl)
    return crcl, "mL/min"


def bmi(entities: dict):
    weight_kg = float(entities["weight_kg"])
    height_cm = float(entities["height_cm"])
    height_m = height_cm / 100
    return weight_kg / (height_m**2), "kg/m^2"


def mean_arterial_pressure(entities: dict):
    sbp = float(entities["systolic_bp_mmhg"])
    dbp = float(entities["diastolic_bp_mmhg"])
    return dbp + (sbp - dbp) / 3, "mmHg"


def ideal_body_weight(entities: dict):
    sex = _sex(entities)
    height_cm = float(entities["height_cm"])
    return ideal_body_weight_kg(sex, height_cm), "kg"


def qtc_fridericia(entities: dict):
    qt_ms = float(entities["qt_interval_ms"])
    hr_bpm = float(entities["heart_rate_bpm"])
    rr_sec = 60 / hr_bpm
    return qt_ms / (rr_sec ** (1 / 3)), "ms"


def perc_rule(entities: dict):
    criteria = [
        float(entities["age"]) >= 50,
        float(entities["heart_rate_bpm"]) >= 100,
        float(entities["spo2_percent"]) < 95,
        bool(entities["unilateral_leg_swelling"]),
        bool(entities["hemoptysis"]),
        bool(entities["recent_surgery_or_trauma"]),
        bool(entities["prior_pe_or_dvt"]),
        bool(entities["hormone_use"]),
    ]
    return sum(criteria), "criteria met"


def sirs_criteria(entities: dict):
    temperature_c = float(entities["temperature_c"])
    heart_rate_bpm = float(entities["heart_rate_bpm"])
    respiratory_rate = float(entities["respiratory_rate"])
    paco2_mmhg = entities.get("paco2_mmhg")
    wbc_count = float(entities["wbc_count"])
    band_percent = entities.get("band_percent")

    temp_met = temperature_c > 38 or temperature_c < 36
    hr_met = heart_rate_bpm > 90
    resp_met = respiratory_rate > 20 or (paco2_mmhg is not None and float(paco2_mmhg) < 32)
    wbc_met = (
        wbc_count > 12000
        or wbc_count < 4000
        or (band_percent is not None and float(band_percent) > 10)
    )
    return sum([temp_met, hr_met, resp_met, wbc_met]), "criteria met"


def curb_65(entities: dict):
    confusion = bool(entities["confusion_present"])
    bun_mg_dl = float(entities["bun_mg_dl"])
    respiratory_rate = float(entities["respiratory_rate"])
    sbp = float(entities["systolic_bp_mmhg"])
    dbp = float(entities["diastolic_bp_mmhg"])
    age = float(entities["age"])

    criteria = [
        confusion,
        bun_mg_dl > 19,
        respiratory_rate >= 30,
        (sbp < 90) or (dbp <= 60),
        age >= 65,
    ]
    return sum(criteria), "score"


def heart_score(entities: dict):
    """M2/M3 already assign each category's 0/1/2 points (that part is clinical
    judgment, not arithmetic) - this just sums the 5 category scores."""
    points = [
        float(entities["history_points"]),
        float(entities["ekg_points"]),
        float(entities["age_points"]),
        float(entities["risk_factor_points"]),
        float(entities["troponin_points"]),
    ]
    for p in points:
        if not (0 <= p <= 2):
            raise ValueError(f"HEART sub-score out of range 0-2: {p}")
    return sum(points), "score"


def sofa_score(entities: dict):
    """Same pattern as HEART: M2/M3 grade each of the 6 organ systems 0-4, this sums them."""
    points = [
        float(entities["respiration_points"]),
        float(entities["coagulation_points"]),
        float(entities["liver_points"]),
        float(entities["cardiovascular_points"]),
        float(entities["cns_points"]),
        float(entities["renal_points"]),
    ]
    for p in points:
        if not (0 <= p <= 4):
            raise ValueError(f"SOFA sub-score out of range 0-4: {p}")
    return sum(points), "score"


# calculator_name (exact dataset string) -> executor function
EXECUTORS = {
    "Creatinine Clearance (Cockcroft-Gault Equation)": cockcroft_gault,
    "Body Mass Index (BMI)": bmi,
    "Mean Arterial Pressure (MAP)": mean_arterial_pressure,
    "Ideal Body Weight": ideal_body_weight,
    "QTc Fridericia Calculator": qtc_fridericia,
    "PERC Rule for Pulmonary Embolism": perc_rule,
    "SIRS Criteria": sirs_criteria,
    "CURB-65 Score for Pneumonia Severity": curb_65,
    "HEART Score for Major Cardiac Events": heart_score,
    "Sequential Organ Failure Assessment (SOFA) Score": sofa_score,
}


def execute(calculator_name: str, entities: dict):
    """Return (value, unit) for calculator_name given its extracted entities.
    Raises KeyError if there's no executor for calculator_name (caller should abstain)."""
    if calculator_name not in EXECUTORS:
        raise KeyError(f"no formula executor registered for {calculator_name!r}")
    return EXECUTORS[calculator_name](entities)
