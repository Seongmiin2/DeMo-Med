# Knowledge Cards

One JSON file per calculator, used as the "open-book" context for the
`open_book` and `demo_med` prompt methods.

Only build cards for calculators that actually appear in `data/pilot/pilot_20.csv`.
Do not try to pre-build cards for every calculator in MedCalc-Bench.

## Schema

```json
{
  "calculator_name": "string",
  "task_type": "equation-based | rule-based | scoring",
  "formula_or_rule": "string",
  "required_entities": ["..."],
  "unit_rules": {},
  "condition_rules": {},
  "output_format": {
    "value": "number | string",
    "unit": "string"
  }
}
```

See `sample_card.json` for a worked example (Cockcroft-Gault Creatinine Clearance).
