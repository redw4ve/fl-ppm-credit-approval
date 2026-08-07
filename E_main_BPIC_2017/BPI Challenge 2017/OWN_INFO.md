# BPIC 2017 Dataset Summary

## Source

- **Dataset**: BPI Challenge 2017 (Version 1)
- **Repository**: 4TU.ResearchData
- **DOI**: [10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b)
- **Citation**: van Dongen, B. F. (2017). BPI Challenge 2017. 4TU.ResearchData.

## File Integrity

- **File**: `BPI Challenge 2017.xes` (uncompressed)
- **Size**: 578,941,403 bytes
- **MD5**: `3b8eefc5a5981c48451af0513e1669d3`

---

## Event Log Contents

- **Timeframe**: January 1, 2016, until February 1, 2017 (15:11)
- **Total events**: 1,202,267
- **Cases** (loan applications): 31,509
- **Offers created**: 42,995
- **Distinct activities**: 26
- **Originators**: 149 unique identifiers (employees or systems)
- **Organizational groups / roles**: 1 / 1 (no information value, excluded from features)

### Trace length distribution

- **Minimum**: 10 events
- **Mean**: 38.2 events
- **Median**: 35 events
- **98th percentile**: 83 events (recorded as the selected prefix-length cap during encoding)
- **Maximum**: 180 events

### Case duration distribution

- **Mean**: 21.9 days
- **Median**: 19.1 days
- **95th percentile**: 42.5 days
- **Maximum**: ~286 days

---

## Event Types

Activities follow a three-prefix convention:

- **Application events** (`A_*`, 10 distinct, English labels): state changes of the application itself
- **Offer events** (`O_*`, seven distinct, English labels): state changes of an offer
- **Workflow events** (`W_*`, eight distinct, English labels): manual work items by bank employees

The `lifecycle:transition` attribute resolves to seven (7) distinct values. 
BPIC 2017 is not filtered by lifecycle, so the attribute stays informative and is kept in the model input feature set.

---

## Application (Case) Attributes

Static attributes are attached to every event in a case:

| Attribute              | Type                    | Coverage | Notes                                                     |
|------------------------|-------------------------|----------|-----------------------------------------------------------|
| `case:RequestedAmount` | numeric                 | 100%     | Median 12,500 EUR; max 450,000 EUR                        |
| `case:LoanGoal`        | categorical (14 levels) | 100%     | Dominated by *Car* (29.6%) and *Home improvement* (24.3%) |
| `case:ApplicationType` | categorical (2 levels)  | 100%     | 89% *New credit*, 11% *Limit raise*                       |
| `case:concept:name`    | identifier              | 100%     | Application ID                                            |

---

## Offer Attributes

Event-level attributes recorded only on offer-creation events (`O_Create_Offer`). 
Sparsity is 96.4% across the full log (1,159,272 of 1,202,267 events null), 
corresponding exactly to the 42,995 offer-creation events.

| Attribute                 | Used as encoder input | Notes                                    |
|---------------------------|-----------------------|------------------------------------------|
| `OfferID`                 | no                    | Identifier                               |
| `OfferedAmount`           | yes                   |                                          |
| `InitialWithdrawalAmount` | no                    |                                          |
| `FirstWithdrawalAmount`   | no                    |                                          |
| `NumberOfTerms`           | yes                   | Payback terms                            |
| `MonthlyCost`             | yes                   |                                          |
| `CreditScore`             | yes                   | Zero  mapped to NaN before normalization |
| `Selected`                | no                    | Boolean                                  |
| `Accepted`                | no                    | Boolean                                  |
| `org:resource`            | yes                   | Originator (149 distinct values)         |

A binary `offer_present` flag is added during encoding. 
It distinguishes prefixes that precede the first offer event from those that already include offer data.

---

## Outcome Label Derivation

BPIC 2017 has no explicit outcome attribute. The multiclass outcome label is derived from the final recognized 
decision event per case:

| Decision event     | Cases      | Share      | Outcome class |
|--------------------|------------|------------|---------------|
| `O_Accepted`       | 17,228     | 54.7%      | 2, accepted   |
| `A_Cancelled`      | 10,431     | 33.1%      | 0, cancelled  |
| `A_Denied`         | 3,752      | 11.9%      | 1, denied     |
| no decision event  | 98         | 0.3%       | dropped       |
| **Total**          | **31,509** | **100.0%** |               |

- **Effective n for outcome classification**: 31,411 after dropping truncation cases.
- **Outcome head**: three classes with `CrossEntropyLoss`. Cancelled cases stay in the outcome target.

`A_Accepted` is a mid-process activity, not a terminal state. 
`A_Pending` follows accepted offers and is not used as the positive label.

The 98 cases without a recognized decision event are concentrated on the log cutoff 
(last event between 2017-01-02 and 2017-02-01 14:11), indicating truncation rather than data error.

---

## Data Quality Notes

- `RequestedAmount = 0` occurs in a small subset of cases. Filtering rule documented in the preprocessing module.
- `CreditScore = 0` functions as a missing-data marker rather than a valid score (median and 25th percentile are both zero). Mapped to NaN before normalization.
- The `LoanGoal` value `Not speficied` contains a typographical error in the original log. Preserved unchanged for compatibility with prior BPIC 2017 benchmarks.
- Ninety-eight (98) cases without a recognized decision event are excluded before client partitioning. All are concentrated on the log cutoff, indicating truncation.