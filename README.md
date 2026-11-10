---
title: Maple Physio No-Show Check
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
python_version: "3.12"
pinned: false
---

# Maple Physio no-show check

A Python machine learning project that estimates whether a patient will miss a physiotherapy appointment and recommends whether the clinic should make a confirmation call. It combines a reproducible scikit-learn training pipeline, cost-based decisions, automated tests, and a Gradio interface.

## What it does

Maple Physio loses revenue when patients miss appointments. A receptionist can call patients the day before their visits. Under the project's business assumptions, each call costs **$8**, an uncalled no-show costs **$35**, and a successful call prevents that no-show.

The application is designed to show the **predicted no-show probability** and a plain-language **Call / Do not call** recommendation. It should also reject invalid bookings with a clear message and warn when travel distance is unavailable.

**Live Gradio app:** Link to be added when deployment is complete.

The recommendation uses the decision threshold saved alongside the trained model, rather than a threshold hardcoded into the interface.

## How to run it locally

### Prerequisites

- Git
- Conda (Miniconda or Anaconda)
- Internet access for the initial environment installation

The local Conda environment uses **Python 3.11**. The Hugging Face Space specifies **Python 3.12** in the settings block above.

### 1. Clone the repository

```bash
git clone https://github.com/arifelshafea/midterm-no-shows.git
cd midterm-no-shows
```

### 2. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate midterm-no-shows
```

The environment installs the pinned runtime and development dependencies. `requirements.txt` contains the runtime packages needed by the Space (`scikit-learn`, `pandas`, and `joblib`); `requirements-dev.txt` additionally installs Gradio, pytest, and Ruff.

### 3. Reproduce training and reports

```bash
python -m noshows.train
```

This regenerates the trained pipeline and its metadata in `models/`, plus the cross-validation and evaluation reports in `reports/`:

```text
models/no_show.joblib
models/model_info.json
reports/cv_results.csv
reports/metrics.json
```

### 4. Launch the interface

```bash
gradio app.py
```

*The app launch instructions will be confirmed after the interface is integrated.*

The interface will be available at the local URL printed in the terminal once `app.py` is implemented.

## Team

| Member | Main responsibilities |
|---|---|
| Arif Elshafea | Preprocessing, dataset split, model comparison, hyperparameter search, cost-based threshold, evaluation, leakage experiment, model artifacts, training tests |
| Devashree Paranjape | `Appointment` and `NoShowPredictor` classes, field validation, class and inference tests |
| Sara Aly | Gradio Blocks app, interface error/warning handling, deployment, and deployment tests |

**Devashree — contribution summary:** To be completed after the classes and related work are integrated.

**Sara — contribution summary:** To be completed after the interface and deployment work are integrated.

The team will combine individual contributions and complete the integration review and demonstration together.

## Decisions

### Data and split

`noshows/data.py` generates **4,000 synthetic appointments** reproducibly. `no_show = 1` means the patient missed the appointment; `no_show = 0` means they attended. The model uses a **stratified 75/25 split** with `random_state = 42`: 3,000 training rows and 1,000 held-out test rows. Model selection and threshold tuning use training data only.

### Feature selection: every dataset column

| Column | Use | Reason |
|---|---|---|
| `appointment_id` | Exclude | An arbitrary identifier, not a meaningful booking characteristic |
| `lead_days` | Include | Waiting time between booking and visit may influence attendance |
| `age` | Include | Attendance risk may differ across ages |
| `prior_no_shows` | Include | Past no-shows may be predictive of future no-shows |
| `hour` | Include | Early or late appointment times may affect attendance |
| `weekday` | Include | Attendance may vary by day of the week |
| `booking` | Include | Booking channels may relate to attendance behavior |
| `insurance` | Include | Payment/coverage categories may relate to attendance behavior |
| `treatment` | Include | Different treatments may have different attendance patterns |
| `first_visit` | Include | New and returning patients may behave differently |
| `distance_km` | Include | Travel distance may affect attendance; missing values are imputed |
| `fee_charged` | Exclude | Recorded **after** the appointment; including it would leak outcome information |
| `no_show` | Target, not feature | This is the value being predicted |

### Processing and model design

One reusable `build_pipeline()` function wraps any candidate classifier in the same preprocessing `ColumnTransformer`. Numeric features receive **median imputation and standard scaling**. Categorical features receive **one-hot encoding** with `handle_unknown="ignore"`. The same preprocessing design is used for model comparison, hyperparameter search, and final prediction.

Unknown categories do not crash the encoder, but a misspelled category should **not** be treated as valid input. The planned `Appointment` validation rejects unsupported categories. Conversely, a missing `distance_km` is an accepted input gap: the model imputes it, and the interface is designed to display a warning.

### Booking input rules

| Field | Accepted values | Why this limit is used |
|---|---|---|
| `lead_days` | Number, 0–120 | Negative lead time is impossible; 120 caps bookings far beyond the training range |
| `age` | Number, 12–110 | Prevents implausible or far-out-of-range ages |
| `prior_no_shows` | Number, 0–30 | Cannot be negative; limits extreme histories outside the training range |
| `hour` | Number, 8–17 | Matches the clinic's modeled appointment hours |
| `first_visit` | Exactly 0 or 1 | This is a binary indicator, not a continuous value |
| `distance_km` | Number, 0.1–150, **or missing** | Rejects invalid distances while allowing a known data gap with imputation |
| `weekday` | `mon`, `tue`, `wed`, `thu`, `fri` | Only these weekdays exist in the booking system/training categories |
| `booking` | `online`, `phone`, `front_desk` | Only supported booking methods should be scored |
| `insurance` | `private`, `provincial`, `self_pay` | Only supported insurance categories should be scored |
| `treatment` | `physio`, `massage`, `acupuncture` | Only modeled treatment categories should be scored |

The planned `Appointment` class will validate booking values, report invalid fields through `ValueError`, remain immutable, and produce a one-row table with the training feature names. The precise implementation will be documented after integration.

### Prediction and calling decision

The model produces a probability of a no-show. Calling decisions use the cost function:

```text
Total cost = ($8 × number of patients called)
           + ($35 × no-shows not called)
```

Training-only, out-of-fold predictions were evaluated at thresholds from **0.05 to 0.95**. The chosen threshold is **0.21**:

- Probability **below 0.21**: **Do not call**.
- Probability **at or above 0.21**: **Call**.

`models/model_info.json` stores the threshold, seed, feature lists, selected hyperparameters, scikit-learn version, and final test results. The planned `NoShowPredictor` class will load the saved model and metadata once and return both the predicted probability and the call decision for a validated `Appointment`. Its implementation and behavior will be documented after integration.

## Model comparison

We used five-fold stratified cross-validation **on the training set** to compare a majority-class baseline with three candidate classifiers, including a balanced model. Each cell below is **mean ± standard deviation** across the folds.

| Model | Precision | Recall | F1 | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| DummyClassifier | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 | 0.7383 ± 0.0000 |
| Logistic Regression | 0.5953 ± 0.0666 | 0.2153 ± 0.0236 | 0.3162 ± 0.0348 | 0.7129 ± 0.0222 | 0.7563 ± 0.0124 |
| Balanced Logistic Regression | 0.4149 ± 0.0168 | 0.6395 ± 0.0269 | 0.5028 ± 0.0139 | 0.7129 ± 0.0219 | 0.6690 ± 0.0163 |
| Random Forest | 0.5324 ± 0.0752 | 0.1809 ± 0.0303 | 0.2699 ± 0.0431 | 0.6788 ± 0.0227 | 0.7443 ± 0.0136 |

The dummy baseline obtains **73.83% accuracy** while finding **none** of the no-shows. Because most appointments are attended, accuracy on its own is misleading. Recall, precision, ROC-AUC, and ultimately the clinic's calling cost provide more useful evaluation.

### Chosen model and tuning

We chose **Logistic Regression** for its competitive ROC-AUC, simplicity, and probability output, which can be converted into a cost-aware recommendation. Although balanced Logistic Regression had stronger recall at its default cutoff, we optimized the calling threshold separately for the clinic's actual costs.

`GridSearchCV` used **five-fold stratified CV**, scoring=`roc_auc`, with two hyperparameters and eight combinations:

- `model__C`: 0.01, 0.1, 1.0, 10.0
- `model__fit_intercept`: `True`, `False`

Best settings: **`C = 0.1`, `fit_intercept = False`**.

| Comparison | Mean cross-validated ROC-AUC |
|---|---:|
| Untuned Logistic Regression | 0.7129 |
| Tuned Logistic Regression | 0.7131 |
| Difference | +0.0002 |
| Untuned model's cross-fold SD | 0.0222 |

The improvement (+0.0002) is much smaller than the fold-to-fold spread (0.0222); it is **not persuasive evidence** that tuning materially improved performance.

## Result

The final fitted pipeline was evaluated using the **1,000 held-out appointments**, after model and threshold selection.

### Classification performance

| Metric | Held-out test result |
|---|---:|
| Accuracy | 0.6190 |
| Precision | 0.3904 |
| Recall | 0.8092 |
| F1 | 0.5267 |
| ROC-AUC | 0.7599 |
| Call threshold | 0.21 |

At the selected threshold, the system identifies about **80.92%** of no-shows, with additional calls to some patients who would have attended.

### Business cost: three strategies

| Strategy | Cost on 1,000 test appointments |
|---|---:|
| Call nobody | $9,170 |
| Call everybody | $8,000 |
| **Model-based calling** | **$6,094** |

Under the scenario's cost assumptions, the model saves **$3,076 (33.5%)** versus calling nobody and **$1,906 (23.8%)** versus calling everybody. It beats **both** baselines on this test set.

**How much can we trust this difference?** These are estimates from one 1,000-row **synthetic** test sample, not guaranteed future cash savings. The training script uses 1,000 paired bootstrap resamples of the held-out test data to estimate uncertainty in the savings (95% percentile intervals):

- **Savings versus calling nobody:** **$2,234.33 to $3,869.28**.
- **Savings versus calling everybody:** **$1,433.98 to $2,376.08**.

These intervals describe sampling variation **under the supplied test data and cost assumptions**. They cannot capture real-world changes in patient behavior, call effectiveness, or clinic costs.

### Leakage check

We compared cross-validated ROC-AUC with and without `fee_charged`, which is recorded after an appointment:

| Features | Cross-validated ROC-AUC |
|---|---:|
| Booking-time features only (honest) | 0.7131 |
| Booking-time features **plus `fee_charged`** (leaking) | 0.9235 |

The larger leaking score cannot be trusted for decisions made before an appointment: the clinic does not yet know whether a missed-appointment fee will be charged. The shipped pipeline **does not use `fee_charged`**.

## Tests

The Training and Evaluation component has **34 passing tests** (last checked locally), including parametrized cases. These cover reproducible data, feature exclusions, stratified splitting, missing distance, hand-checked costs, both baseline extremes, threshold edge cases, comparison and tuning setup, out-of-fold predictions, test metrics, and saved report structure.

**Class and prediction tests (Devashree):** Coverage details will be added after integration.

**Application and deployment tests (Sara):** Coverage details will be added after integration.

The final group test total and fresh-clone results will be recorded after all components are combined.

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

Also confirm `python -m noshows.train` regenerates the model and reports from a fresh clone. `models/` must be committed, not excluded by `.gitignore`.

### Repository structure

```text
midterm-no-shows/
├── README.md
├── app.py                    # Gradio Blocks interface (planned)
├── noshows/
│   ├── __init__.py
│   ├── data.py               # original synthetic data generator
│   ├── pipeline.py           # features, split, preprocessing
│   ├── evaluation.py         # cost and threshold functions
│   ├── appointment.py        # validated booking class (planned)
│   ├── predictor.py          # model inference class (planned)
│   └── train.py              # reproducible training and reporting
├── models/
│   ├── no_show.joblib
│   └── model_info.json
├── reports/
│   ├── cv_results.csv
│   └── metrics.json
├── tests/
├── environment.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── .gitignore
```

*Planned files are shown for context; the structure will be updated after integration.*

## Limitations

- **Synthetic data:** The generator is useful for demonstrating a workflow, but its patterns may not represent real patients or clinics.
- **Unknown causes:** Illness, emergencies, transport delays, work conflicts, and changes after booking are not observed.
- **Imputed distance:** When travel distance is missing, the model substitutes a training median; this adds uncertainty.
- **Imperfect predictions:** A predicted probability is not a certainty, and both unnecessary calls and missed no-shows can occur.
- **Simplified costs:** The analysis assumes a call always prevents a no-show and uses fixed $8 and $35 costs. Real effectiveness and economics may differ.
- **Limited evaluation:** One 1,000-row synthetic test set does not establish real-world cost savings; ongoing monitoring and evaluation with real clinic data would be needed.
- **Human oversight:** The recommendation should support reception staff, not replace their judgment.

**Deployment and demo:** The public Space URL and, if applicable, a demo recording link will be included after group integration.
