# BPIC 2012 Dataset Summary

## Source

- **Dataset**: BPI Challenge 2012 (Version 1)
- **Repository**: 4TU.ResearchData
- **DOI**: [10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)
- **Citation**: van Dongen, B. F. (2012). BPI Challenge 2012. 4TU.ResearchData.

## File Integrity

- **File**: `BPI_Challenge_2012.xes`
- **Size**: 74,100,050 bytes
- **MD5**: `b815ef03ebae63407bc09de191a748f6`

---

## Event Log Contents

- **Timeframe**: October 1, 2011, to March 14, 2012 (16:04)
- **Total events**: 262,200
- **Cases** (loan applications): 13,087
- **Distinct activities**: 24
- **Originators**: 68 unique identifiers (employees or systems)
- **Lifecycle transitions**: three distinct values (`SCHEDULE`, `START`, `COMPLETE`) 

> Only `COMPLETE` cases are retained, removing about 37% of the raw event data

### Trace length distribution (post `COMPLETE`-only filter)

- **Minimum**: 3 events
- **Mean**: 12.6 events
- **Median**: 8 events
- **98th percentile**: 42 events (recorded as the selected prefix-length cap during encoding)
- **Maximum**: 96 events

### Case duration distribution

- **Mean**: 8.6 days
- **Median**: 0.8 days
- **95th percentile**: 31.3 days
- **Maximum**: ~137 days

---

## Event Types

Activities follow a three-prefix convention:

- **Application events** (`A_*`, 10 distinct, English labels): state changes of the application
- **Offer events** (`O_*`, seven distinct, English labels): state changes of an offer
- **Workflow events** (`W_*`, seven distinct, **Dutch labels**): manual work items by bank employees


The Dutch `W_*` labels require alignment to the BPIC 2017 vocabulary before any cross-dataset analysis. 
Alignment is performed in the encoding step (LLM-assisted, manual fallback).

The `lifecycle:transition` attribute resolves to three (3) distinct values. 
After the canonical `COMPLETE`-only filter, `lifecycle:transition` becomes constant and is dropped from the model 
input feature set.

---

## Application (Case) Attributes

Static attributes are attached to every event in a case. 
BPIC 2012 has only one informative case attribute, compared to BPIC 2017's three.

| Attribute           | Type       | Coverage | Notes                                                                |
|---------------------|------------|----------|----------------------------------------------------------------------|
| `case:AMOUNT_REQ`   | numeric    | 100%     | Median 10,000 EUR; max 99,999 EUR; one (1) zero-amount case in total |
| `case:REG_DATE`     | timestamp  | 100%     | Registration date; redundant with first event timestamp              |
| `case:concept:name` | identifier | 100%     | Application ID                                                       |

`LoanGoal`, `ApplicationType`, and the entire offer-attribute block 
(`CreditScore`, `MonthlyCost`, `OfferedAmount`, `NumberOfTerms`) **do not exist** in BPIC 2012. 
The encoder runs with `USE_OFFER_FEATURES=False` on this dataset.

---

## Offer Attributes

**None.** BPIC 2012 records offer events (`O_*` activities) but no offer-level attributes. 
The `offer_present` flag and forward-fill mechanism used on BPIC 2017 are skipped on BPIC 2012.

---

## Outcome Label Derivation

BPIC 2012 has no explicit outcome attribute. The multiclass outcome label is derived from the terminal application-event:

| Terminal event    | Cases      | Share      | Outcome class |
|-------------------|------------|------------|---------------|
| `A_DECLINED`      | 7,635      | 58.3%      | 1, declined   |
| `A_CANCELLED`     | 2,807      | 21.4%      | 0, cancelled  |
| `A_APPROVED`      | 2,246      | 17.2%      | 2, approved   |
| no terminal event | 399        | 3.0%       | dropped       |
| **Total**         | **13,087** | **100.0%** |               |

- **Effective n for outcome classification**: 12,688 after dropping truncation cases.
- **Outcome head**: three classes with `CrossEntropyLoss`. Canceled cases stay in the outcome target.

BPIC 2012 is declined-heavy, while BPIC 2017 is accepted-heavy. 
This class imbalance is handled as a multiclass training-policy question.

The 399 cases without a terminal event are concentrated on the log cutoff 
(last event between 2012-02-22 and 2012-03-14 16:04), indicating truncation rather than data error.

---

## Outcome-Label Alignment to BPIC 2017

The terminal-event vocabulary differs from BPIC 2017. The pipeline applies the following manual mapping 
for the multiclass outcome head:

| BPIC 2012     | BPIC 2017 equivalent | Role                                     |
|---------------|----------------------|------------------------------------------|
| `A_APPROVED`  | `O_Accepted`         | approved or accepted (class 2)           |
| `A_DECLINED`  | `A_Denied`           | declined or denied (class 1)             |
| `A_CANCELLED` | `A_Cancelled`        | cancelled (class 0)                      |

Full activity-vocabulary alignment (the `W_*` Dutch labels and the more granular `A_*` / `O_*` states) 
is performed in the encoding step.

---

## Known Data Quality Notes

- `lifecycle:transition` produces three values (`SCHEDULE`, `START`, `COMPLETE`). The canonical PPM treatment, established by Tax et al. (2017) and used by every subsequent benchmark, retains only `COMPLETE`.
- `AMOUNT_REQ = 0` occurs in exactly one case. Kept in the dataset; the IID partitioner assigns it deterministically, the non-IID partitioner routes it to Bank A.
- 399 cases without a terminal event are excluded before client partitioning. All are concentrated on the log cutoff, indicating truncation.
- Activity labels in the `W_*` subprocess are in Dutch and require alignment to the BPIC 2017 vocabulary before cross-dataset analysis.
- Distinct resources (68) is roughly half the BPIC 2017 figure (149). The smaller vocabulary lowers the embedding parameter count but carries less information per resource.