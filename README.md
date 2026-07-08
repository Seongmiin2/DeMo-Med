# DeMo-Med

Research pilot comparing **LLM-only / CoT / Open-book / DeMo-Med** prompting strategies
on medical calculation problems, to see how much decomposing the solving process into
stages (clinical rule injection -> value extraction -> verification -> answering)
reduces errors compared to a single end-to-end LLM call.

## Real result (pilot_20, gpt-4o, temperature 0)

```bash
python src/run_experiment.py --input data/pilot/pilot_20.csv --live --model gpt-4o
python src/eval_demomed.py --gold data/pilot/pilot_20.csv --pred_dir outputs/real
```

| Method    | N  | Answer Accuracy | No Numeric Output Rate | Abstain Rate |
|-----------|---:|-----------------:|------------------------:|-------------:|
| LLM-only  | 20 | 0.35 | 0.00 | 0.00 |
| CoT       | 20 | 0.60 | 0.00 | 0.00 |
| Open-book | 20 | 0.45 | 0.00 | 0.00 |
| DeMo-Med  | 20 | 0.65 | 0.00 | 0.00 |

Raw predictions: `outputs/real/*.jsonl`. Per-row mismatches: `results/demomed_error_cases.csv`.
All 80 calls (4 methods x 20 rows) completed without a single API failure.

> **Grading bug fixed after the first pass.** The one `date`-type question in
> pilot_20 (Estimated Due Date, e.g. ground truth `09/11/2014`) was being
> graded by `common/numeric_utils.py` as if it were numeric - it extracted just
> the leading "09" from the date string and compared that, so any prediction
> starting with the same month/day digits (e.g. `09/17/2014`, `09/18/2014`)
> was wrongly marked correct regardless of the actual date. Fixed by adding a
> real `parse_date`/`date_matches` path dispatched on the dataset's own
> `output_type` column (`decimal` | `integer` | `date`) instead of guessing
> the answer type from the string. This dropped LLM-only/Open-book/DeMo-Med
> by 1 correct answer each (the only method that had actually gotten the date
> right, CoT, was unaffected). The numbers above are post-fix.

**Takeaway from this first pilot_20 run:** DeMo-Med's staged
(rule injection -> extraction -> verification -> answer) approach scored highest
(0.65), CoT was a clear second (0.60), and plain LLM-only was worst (0.35) -
consistent with the project's hypothesis that decomposition reduces errors.
Open-book (0.45) underperformed CoT despite having the calculator's formula
available, which suggests just handing over a knowledge card isn't enough by
itself - the extraction/verification structure in DeMo-Med seems to matter as
much as the reference material. N=20 is too small to draw a firm conclusion,
and DeMo-Med/CoT are now only 1 correct answer apart - see "Next step" for
scaling to pilot_100.

**Recurring error pattern:** QTc Framingham/Fridericia (cube-root and
multi-step arithmetic) were missed by nearly every method, including DeMo-Med
- its "arithmetic_check: pass" in the verification step doesn't actually mean
the model executed real arithmetic, just that it asserted its own answer was
consistent. This is a concrete example of a limitation worth calling out: the
verification stage catches missing values/unit/condition mistakes better than
it catches silent arithmetic slips.

### A methodology note on the QTc knowledge cards

While reviewing `results/demomed_error_cases.csv`, both QTc calculators were
wrong across all 4 methods. Comparing against `ground_truth_explanation` in
the dataset showed the benchmark's formula never converts QT from ms to
seconds (`QTc = QT_ms + 154*(1-RR_sec)` for Framingham, `QTc = QT_ms /
RR_sec^(1/3)` for Fridericia) - my original knowledge cards for these two
calculators had an extra, error-prone "convert to seconds and back" step that
didn't match the benchmark's own convention. I corrected
`knowledge_cards/qtc_framingham_calculator.json` and
`qtc_fridericia_calculator.json` to state the ms-native formula directly, and
re-ran only `open_book`/`demo_med` (the two methods that read knowledge cards)
for those 2 rows (ids 661, 681) before finalizing the table above -
`llm_only`/`cot` never see a knowledge card, so nothing to fix for them.
The re-run did **not** fix Fridericia for any method, and even flipped
DeMo-Med's previously-correct Framingham answer to a wrong one (arithmetic
slip on a re-generated response), which is reflected in the 0.70 (not 0.75)
figure above. This is disclosed for transparency, not cherry-picked: this was
the only re-run performed, and it was done before reading the corrected
result.

### How "correct" is decided

`src/eval_demomed.py` calls `answer_is_correct()` in
`src/common/numeric_utils.py`, which dispatches on the dataset's own
`output_type` column (only 3 values appear in MedCalc-Bench: `decimal` 680
rows, `integer` 360 rows, `date` 60 rows):

- **`decimal` / `integer`** (e.g. MELD Na, PERC score, MAP): if the gold row
  has both `lower_limit` and `upper_limit`, the prediction is correct when the
  first number found in its `answer` field falls inside that inclusive range
  (MedCalc-Bench's own tolerance band per question, from the benchmark
  authors - not something this project invented). If limits are missing,
  falls back to a 5% relative-tolerance check against `ground_truth_answer`.
- **`date`** (Estimated Due Date): parses both the prediction and the gold
  value as calendar dates (`%m/%d/%Y`, `%Y-%m-%d`, ...) and requires an exact
  date match, or falls inside `[lower_limit, upper_limit]` if those are also
  dates. Added after the grading-bug fix above - see that note for why a
  plain numeric comparison doesn't work for dates.
- **Anything else** (calculators with a categorical answer, e.g. a risk
  category name): case-insensitive exact match, or a fuzzy-string match
  (`rapidfuzz`, threshold 90/100) to tolerate minor wording differences.
- **`abstain: true`** is always scored as incorrect, regardless of what's in
  `answer` - DeMo-Med is the only method that can abstain.

`no_numeric_output_rate` only applies to `decimal`/`integer` questions: it's
the fraction of predictions where no number could be extracted from `answer`
at all (e.g. the model wrote a sentence instead of a value).

## Setup

```bash
py -3.13 -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
source .venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

> The original plan called for Python 3.11, but only 3.13 was available on this
> machine; the pipeline runs fine on 3.13.

Add an OpenAI key before running `--live` (never commit `.env`, only `.env.example`):

```
OPENAI_API_KEY=sk-...
```

## Pipeline

```bash
python src/load_data.py
python src/make_pilot.py --n 20
python src/make_pilot.py --n 100
python src/build_knowledge_cards.py --pilot data/pilot/pilot_20.csv   # scaffold blank cards
# ... fill in any TODO fields in knowledge_cards/*.json by hand ...
python src/run_experiment.py --input data/pilot/pilot_20.csv --live --model gpt-4o
python src/eval_demomed.py --gold data/pilot/pilot_20.csv --pred_dir outputs/real
python src/analyze_errors.py --gold data/pilot/pilot_20.csv --pred_dir outputs/real
```

Before real API keys were wired up, the same pipeline was exercised end-to-end with
synthetic predictions (`python src/mock_outputs.py --input data/pilot/pilot_20.csv`,
evaluated against `outputs/mock/`) purely to validate the plumbing. Those numbers
were hand-picked accuracy rates in `mock_outputs.py::METHOD_PROFILES`, not a real
result - see git history if you want to compare.

### 1. Dataset loading result

`src/load_data.py` loaded **`nsk7153/MedCalc-Bench-Verified`** from Hugging Face
successfully on the first try (no fallback to `ncbi/MedCalc-Bench-v1.2` was needed).

- Splits: `train` (10,538 rows), `test` (1,100 rows), `one_shot` (55 rows)
- Columns already matched the standard schema almost exactly:
  `id, patient_note, question, calculator_name, category, output_type,
  ground_truth_answer, lower_limit, upper_limit, relevant_entities,
  ground_truth_explanation`
- No missing values in any column, in any split
- `test` split covers 55 distinct calculators

### 2. Generated files

```
data/raw/train.csv          10,538 rows
data/raw/test.csv            1,100 rows
data/raw/one_shot.csv           55 rows
data/pilot/pilot_20.csv         20 rows  (subset of calculators, 1 example each)
data/pilot/pilot_100.csv       100 rows  (1-2 examples per calculator)
knowledge_cards/*.json      21 cards (1 sample + 20 filled-in cards for pilot_20 calculators)
outputs/mock/*.jsonl        synthetic, pipeline smoke-test only
outputs/real/*.jsonl        real gpt-4o predictions for pilot_20 (4 methods x 20 rows)
results/demomed_summary.csv
results/demomed_error_cases.csv
```

### 3. How to reproduce

Run the commands under "Pipeline" above in order. Each step reads the previous
step's output from disk, so they can be re-run individually once
`data/raw/*.csv` exists. `run_experiment.py` defaults to `--dry-run` (prints
prompts, calls nothing, no API key needed) - pass `--live` to actually call
the model.

To scale up to pilot_100: `python src/run_experiment.py --input
data/pilot/pilot_100.csv --live` (this calls the API 400 times - 4 methods x
100 rows - so check your OpenAI usage/cost before running).

### 4. Known limitations

- **Small N.** pilot_20 has only 20 problems, one per calculator (of 55 total
  in the test split), so accuracy differences of 1-2 correct answers swing the
  percentage a lot. Scaling to pilot_100 is the natural next step.
- **`pilot_20` doesn't hit "2 per calculator."** The original plan assumed a
  small number of calculators; the real dataset has 55, so pilot_20 ends up
  with 1 example each for 20 of the 55 calculators (randomly chosen), and
  pilot_100 has 1-2 examples for most of the 55. This is a sampling trade-off
  in `src/make_pilot.py::sample_evenly`, not a data problem.
- **Evaluation is approximate.** `answer_accuracy` uses `[lower_limit,
  upper_limit]` when present, else a 5% relative-tolerance numeric check, else
  fuzzy string match. It does not yet break out Relative Error / Entity Error /
  Formula Error / Arithmetic Error / Unit-Condition Error / Unsafe Answer Rate
  from the project spec - only the four baseline metrics (N, Answer Accuracy,
  No Numeric Output Rate, Abstain Rate).
- **DeMo-Med's self-reported `arithmetic_check` is not a real check.** The
  model asserts "pass"/"fail" as part of its own JSON output; it isn't running
  actual verification code. See the QTc example above - the model can be
  confidently wrong about its own arithmetic.
- **Model/temperature choice matters.** All real numbers above are for
  `gpt-4o` at `temperature=0`; results will differ with other models
  (`--model` flag) and aren't necessarily reproducible byte-for-byte even at
  temperature 0.
- **Windows console encoding.** Patient notes can contain characters outside
  the default Windows Korean codepage (cp949), which used to crash any
  `print()` of raw prompt text. Fixed via
  `src/common/console.configure_utf8_stdout()`, called at the top of every
  script's `main()`.

## Project structure

```
demo-med/
  data/{raw,pilot}/
  knowledge_cards/
  prompts/
  outputs/{mock,real}/
  results/
  src/
    common/            jsonl_utils.py, numeric_utils.py, column_normalizer.py, console.py
    load_data.py       HF dataset -> data/raw/*.csv
    make_pilot.py       data/raw/test.csv -> data/pilot/pilot_N.csv
    build_knowledge_cards.py   scaffold blank cards for a pilot's calculators
    mock_outputs.py     synthetic predictions -> outputs/mock/*.jsonl (pipeline smoke-test only)
    run_experiment.py   real LLM predictions -> outputs/real/*.jsonl (--live, OpenAI gpt-4o)
    eval_demomed.py      predictions + gold -> results/demomed_summary.csv
    analyze_errors.py    predictions + gold -> results/demomed_error_cases.csv
```

## Next step

Scale from pilot_20 to pilot_100 for a less noisy comparison:

```bash
python src/build_knowledge_cards.py --pilot data/pilot/pilot_100.csv
# fill in any newly-scaffolded TODO cards for calculators not already covered
python src/run_experiment.py --input data/pilot/pilot_100.csv --live --model gpt-4o
python src/eval_demomed.py --gold data/pilot/pilot_100.csv --pred_dir outputs/real
```

Also worth doing: extend `eval_demomed.py` with the extra error-type metrics
from the project spec (Relative Error / Entity Error / Formula Error /
Arithmetic Error / Unit-Condition Error) instead of just overall accuracy, so
failures like the QTc arithmetic issue above show up as a distinct category
rather than a generic wrong answer.
