# E_04 Prefix Encoding

E_04 prepares the prefix representation facing the model for the BPIC 2017 main experiment and the BPIC 2012 ablation. 
It does not store full prefix tensors on the disk. Instead, it writes approved JSON recipes and compact metadata. 
Later training stages reload the processed Parquet files and use `PrefixDataset` to create padded tensors on demand.
Padded tensors are cached during training to provide faster access later, due to long on-demand loading times.

The stage has three roles:

- Define a shared feature and activity contract.
- Map BPIC source columns and activity labels into that contract.
- Build train-only vocabularies, input scalers, RT target scalers and run metadata for each experiment configuration.

## How to Run

Run the full encoding metadata and LLM side workflow from the E_04 folder:

```bash
cd E_prefix_encoding
./WORKFLOW_run_encoding.sh
```

> The LLM side experiments are off by default. Enable with RUN_LLM_EXPERIMENT=true and RUN_LLM_SCHEMA_EXPERIMENT=true.
> LLM schema generation is controlled by `RUN_LLM_SCHEMA_EXPERIMENT=true`.
> Export it in the same shell before the run with `export OPENAI_API_KEY="sk-..."`
> Alternatively, put that line in your `~/.zshrc` so it is set in every new shell.
> The workflow reads the variable directly and never stores the key.
> Enabling the switches without a key aborts the workflow with a clear error.

This command consumes the approved manual files under `mappings/`, validates them and writes metadata under 
`encoded_metadata/`. It does not recreate or overwrite the reviewed manual schema or dataset mapping.

To include the LLM schema and dataset mapping side experiments, enable both switches with the key present:

```bash
cd E_prefix_encoding
RUN_LLM_EXPERIMENT=true RUN_LLM_SCHEMA_EXPERIMENT=true ./WORKFLOW_run_encoding.sh
```

## Workflow Files

1. `04_0_extract_contract_context.py`
   - Optional privacy-support script.
   - Read local processed Parquet files and write schema summaries only.
   - Export column names, dtypes, missingness rates, activity labels and aggregate statistics.
   - Does not export event rows, case IDs, timestamp values, resource values or customer values.
   - Supports contract design without central access to raw data.

2. `04_1_contract.py`
   - Loads `mappings/MANUAL_contract.json`.
   - Validate fields, masks, reserved tokens and canonical activity labels.
   - Expose stable Python constants and tensor records for the later encoder and training code.

3. `04_2_create_canonical_schema.py`
   - Read the contract.
   - Manual mode writes an empty schema template to `mappings/llm_mapping/LLM_canonical_schema_template.json`.
   - LLM mode writes draft schemas under `mappings/llm_mapping/canonical_schemas/`.
   - Uses `mappings/llm_mapping/LLM_canonical_schema_targets.json` only as the LLM target brief.
   - The approved thesis schema is the manually reviewed `mappings/MANUAL_canonical_schemas.json`.

4. `04_3_create_dataset_mapping.py`
   - Require an approved canonical schema.
   - Map source Parquet columns into canonical fields.
   - Map raw activity labels into contract-approved canonical activity labels.
   - Support `manual`, `semantic` and `llm` modes.
   - Supports character-based and word-based semantic activity matching through `--semantic-activity-similarity`.
   - The approved thesis mapping is `mappings/MANUAL_dataset_mapping.json`.

5. `04_4_runner.py`
   - Require approved schema and mapping files.
   - Verify that the mapping was built from the current schema hash.
   - Run the metadata matrix for the selected schema profile.
   - Writes four compact JSON artifacts per run.

6. `04_5_encoding.py`
   - Provides the reusable runtime encoder.
   - Builds vocabularies, scalers and prefix indices.
   - Provides `PrefixDataset`, which creates padded prefix tensors on demand.
   - Imported by the runner and later training stages.

7. `04_6_analyze_llm_outputs.py`
   - Optional side-experiment script.
   - Compares LLM drafts against frozen manual references.
   - Writes review metrics and plots under `mappings/llm_mapping/llm_analysis/`.
   - Not used by the encoder, runner or training.

8. `04_7_decentralized_metadata_poc.py`
   - Optional privacy-oriented proof of concept (PoC).
   - Simulates a local metadata collection in the clients on one machine.
   - Rebuilds vocabularies, input scalers, RT target scalers and run counts from local aggregate statistics.
   - Compares the reconstructed metadata against the central E_04 runner metadata.

9. Training handoff
   - Later training stages reload the original processed Parquet files.
   - They load the approved schema, approved mapping and run metadata.
   - They use `PrefixDataset` from `04_5_encoding.py` to create tensors on demand.

## Approval Files

- `mappings/MANUAL_contract.json`: Editable federation contract. Defines the model-facing field universe, reserved tokens and canonical activity labels.
- `mappings/MANUAL_canonical_schemas.json`: Approved BPIC schema. Select active fields, dataset profiles and prefix caps from the contract.
- `mappings/MANUAL_dataset_mapping.json`: Approved BPIC feature and activity mapping. Map source columns and raw activity labels to shared canonical representation.
- `mappings/llm_mapping/LLM_canonical_schema_template.json`: Blank schema template carrying the contract catalog, with the profiles left unfilled. Not used by the runner.
- `mappings/llm_mapping/canonical_schemas/`: Contains optional LLM schema drafts. These drafts are never treated as approved inputs automatically.

The runner refuses unapproved files and stale schema hashes. 
This protects the encoded metadata from accidental use of outdated review files.

## Runner Outputs

Artifacts are saved under `E_prefix_encoding/encoded_metadata/`.

Each run folder contains:

- `<prefix>encoding_spec.json`: run configuration, counts, prefix cap and validation summary.
- `<prefix>vocabulary.json`: train-only categorical token indices.
- `<prefix>scaler.json`: train-only numeric means and standard deviations.
- `<prefix>mapping_report.json`: mapping hash, fallback count and unresolved labels.

The encoding spec stores `target_scalers.remaining_time`. 
The default target representation is `REMAINING_TIME_TRANSFORM=raw` and `REMAINING_TIME_SCALING=zscore`. 
Raw, median and z-score scaling remain available and `log1p` remains available through `REMAINING_TIME_TRANSFORM=log`.

## Run Matrix

`SCHEMA_PROFILE = "all"` runs the full matrix: 6 BPIC 2017 runs, 3 BPIC 2012 runs and 4 joint runs, 13 in total.

`SCHEMA_PROFILE = "bpic2017"` runs:

- `iid_3banks`
- `weak_3banks`
- `medium_3banks`
- `strong_3banks`
- `medium_5banks`
- `strong_5banks`

`SCHEMA_PROFILE = "bpic2012"` runs:

- `iid_3banks`
- `weak_3banks`
- `medium_3banks`

`SCHEMA_PROFILE = "joint"` runs:

- `iid_6banks`
- `weak_6banks`
- `medium_6banks`
- `medium_8banks`

## Encoding Rules

Vocabularies and scalers are fit only on the union of client train splits for the selected run. 
Validation and test data use the frozen train artifacts.

The RT target representation is also fit in E_04 on valid train prefixes only. 
`PrefixDataset` emits encoded `remaining_time_label` values.

Prefixes are padded to the configured cap. The cap limits generated prefix samples but does not truncate the source 
event stream. The outcome target is broadcast to every prefix. The final next activity target is `[END]`. 
The final decision prefix has `remaining_time_mask = 0`.

`PrefixDataset` emits `padding_mask`, `prefix_length`, `next_activity_mask` and `remaining_time_mask`. 
Later training uses these masks instead of dynamic batch lengths. Padding positions must not affect hidden-state 
selection, pooling or task losses.

BPIC 2017 offer state is forward-filled inside each prefix. A later offer overwrites the carried values. 
`CreditScore = 0` resets the carried score to missing. BPIC 2012 uses a profile without active offer tensors.

## Mapping Rules

No hard-coded reviewed activity mapping is stored in the scripts. The mapping lives in the corresponding JSON files.
Dataset mappings may only use canonical activity labels defined in the contract.

Canonical activity labels preserve the BPIC origin prefix:

- Application labels start with `A_`,
- Offer labels start with `O_`,
- Workflow labels start with `W_`.

Dataset mappings cannot cross these origin groups. Raw `A_*` labels map only to canonical `A_*` labels. 
Raw `O_*` labels map only to canonical `O_*` labels. Raw `W_*` labels map only to canonical `W_*` labels.

## LLM Side Experiment

The LLM side experiment evaluates whether an LLM can draft useful **dataset mappings** after the **contract** and 
**canonical schema** are already fixed. The thesis does not frame **contract** creation as an LLM task. The **contract** 
is manually defined because it fixes the shared semantic universe. The **canonical schema** can be LLM-supported, but it 
requires considerable business logic and is treated only as a supporting observation. **Dataset mapping** is the main 
experiment because it represents a realistic onboarding step for a new bank after a shared **contract** and approved 
**canonical schema** exist.

The **dataset mapping** is generated three times per prompt strategy and for two semantic scoring variants. 
`semantic_character` joins normalized label tokens and compares the resulting strings with `SequenceMatcher`. 
`semantic_word` compares the normalized token lists directly, so each word contributes one sequence item regardless of 
character length.

The three prompt strategies are:

- `strategy_1_baseline`: One prompt for the complete dataset mapping with little to no additional structure.
- `strategy_2_split_prompt`: Separate prompts for column mapping and activity mapping.
- `strategy_3_target_recipe`: Split prompt with stronger guidance for activity mapping as the classification task.

The dataset mapping report uses six method groups: 

1. Deterministic semantic character 
2. Deterministic semantic word 
3. LLM Strategy 1 with character seed 
4. LLM Strategy 1 with word seed 
5. LLM Strategy 2 split prompt
6. LLM Strategy 3 classification prompt. 

> Only deterministic semantic mode and LLM Strategy 1 are character-versus-word comparisons. 
> Strategy 2 and 3 are generated once per run because their split prompt does not receive a seeded semantic mapping.

For this side experiment, raw activity labels are discovered across train, validation and test parquets because activity 
mapping is schema harmonization metadata and does not influence training. The prompt does not include label frequencies, 
case counts, targets, timestamps or performance information. Train-only fitting for vocabularies, input scalers and RT
target scalers remain unchanged.

The side experiment writes:

- `mappings/llm_mapping/llm_analysis/04_06_llm_analysis_results.json`
- `mappings/llm_mapping/llm_analysis/04_06_llm_analysis_summary.txt`
- `mappings/llm_mapping/llm_analysis/04_06_llm_dataset_mapping_accuracy.png`
- `mappings/llm_mapping/llm_analysis/04_06_llm_dataset_mapping_errors.png`

> These files are evidence for the result of the side experiment. They are not encoder metadata.

## Privacy-Oriented Deployment and the Decentralized Metadata POC

The BPIC experiments run centrally because BPIC 2017 and BPIC 2012 are public benchmark logs and the banks are simulated 
partitions. A real federation should not require central access to raw event logs. `04_0_extract_contract_context.py` 
supports the stricter deployment narrative. It can be run inside the data owner boundary and writes only contract context:

- Column names and dtypes
- Missingness rates
- Raw activity label lists
- Lifecycle values
- Outcome counts
- Trace length summaries
- Numeric `count`, `sum` and `sum_squared` for the training split

The generated context files are stored under `decentralized_poc/contract_context/`. They document the privacy POC path, 
not the reviewed manual mapping inputs. The central contract engineer can use these summaries to define the federation 
contract. The contract is shared because all clients must agree on model-facing fields and canonical activity labels.

Canonical schemas can be created locally per dataset or bank. They may be shared with the server because they reveal 
only active contract fields and model compatibility information. They do not need to reveal raw column names, raw 
activity labels or event rows.

Dataset mappings are more sensitive because they connect raw local columns and raw activity labels to the contract. 
In a real deployment, these mappings can stay local. For BPIC, one central mapping is enough because all simulated 
banks are partitions of the same public source logs.

`04_7_decentralized_metadata_poc.py` checks whether the central step is only an implementation shortcut.

The script mirrors the federated learning setup on one machine:

- Each simulated bank is treated as a local client
- Each client reads only its own train, validation and test Parquet files
- Each client writes local aggregate statistics
- Secure-aggregation simulation masks additive client statistics before server summation
- The server side rebuilds global vocabularies, input scalers and POC-compatible target scalers from aggregate sums
- The script compares the reconstructed metadata with the central runner artifacts

The encoding workflow runs this POC by default (`RUN_DECENTRALIZED_POC=true`). 
To repeat it standalone once `encoded_metadata/` exists:

```bash
cd E_prefix_encoding
../fl-ppm/bin/python -B 04_7_decentralized_metadata_poc.py
```

It writes:

- `decentralized_poc/local_stats/<dataset>/<run>.json`
- `decentralized_poc/secure_aggregation_messages/<dataset>/<run>.json`
- `decentralized_poc/server_aggregation/<dataset>/<run>.json`
- `decentralized_poc/comparison_reports/<dataset>/<run>.json`
- `decentralized_poc/04_07_DECENTRALIZED_poc_summary.json`

The secure-aggregation logic is a simulation on one machine. It shows that additive metadata can be masked at client 
side and recovered only after summation. It does not implement real key exchange, dropout handling, authentication or 
differential privacy. It validates the encoding claim: global vocabularies, input scalers, z-score RT target scalers 
and run counts can be reconstructed from client statistics without raw event rows or prefix tensors on the server.

## Extension Points

For a new federation, edit `mappings/MANUAL_contract.json` when the model-facing field universe or canonical activity 
label universe changes. `04_1_contract.py` should change only when contract validation or tensor record structure changes. 
Then create a schema template, fill it manually or with LLM support, approve one schema and run dataset mapping.

For a new dataset in an existing federation, keep the contract unchanged when its columns and raw activities map to 
existing canonical fields and labels. Add the dataset input configuration, run dataset mapping, approve the mapping 
and rerun the metadata runner.