# Metadata
Model: gemini-2.0-flash-001
Temperature: 0.0 # Keep low for deterministic data transformation

# System Instructions
You are an expert data processor specializing in handling pharmaceutical information. You will receive prescription data and notes mixed together. Your task is to analyze the provided input/output examples, understand the transformation logic, and apply that logic to new input data to produce a JSON output conforming to the specified schema. **Pay extremely close attention to processing BOTH BIL and GEN notes correctly, aggregating notes by date, ensuring association logic correctly handles exact vs. base name mentions (using both summary and full note text), ensuring base-name-only notes associate broadly, and ensuring the output structure matches the schema precisely.**

# Controlled Output Schema
{
  "type": "array",
  "title": "PrescriptionNotesList",
  "description": "A list of prescriptions, each with its Rx number, drug name, and dated summaries of BIL and GEN notes.",
  "items": {
    "type": "object",
    "title": "PrescriptionEntry",
    "required": [
      "rxNumber",
      "drugName",
      "bilNotesSummary",
      "genNotesSummary"
    ],
    "properties": {
      "rxNumber": {
        "type": "string",
        "description": "The unique prescription number (Rx), exactly matching the input finalDisplayEntityId."
      },
      "drugName": {
        "type": "string",
        "description": "The name of the drug (e.g., 'HUMIRA 40MG SYRINGE (2/BOX)'), exactly matching the input medicineName."
      },
      "bilNotesSummary": {
        "type": "array",
        "description": "An array containing zero or more 'DatedNoteSummary' objects for BIL (billing, pricing, prior auth) notes, grouped and summarized by date. If no relevant BIL notes exist for the prescription, this MUST be an empty array `[]`. Sorted chronologically by noteDate.",
        "items": {
          "type": "object",
          "title": "DatedNoteSummary",
          "description": "A summary of notes for a specific date.",
          "required": [ "noteDate", "summaryText" ],
          "properties": {
            "noteDate": { "type": "string", "description": "The date (MM/DD/YYYY).", "pattern": "^(0[1-9]|1[0-2])\\/(0[1-9]|[12][0-9]|3[01])\\/\\d{4}$" },
            "summaryText": { "type": "string", "description": "Consolidated summary starting with 'MM/DD/YYYY: '." }
          }
        }
      },
      "genNotesSummary": {
        "type": "array",
        "description": "An array containing zero or more 'DatedNoteSummary' objects for GEN (general) notes, grouped and summarized by date. If no relevant GEN notes exist for the prescription, this MUST be an empty array `[]`. Sorted chronologically by noteDate.",
        "items": {
           "$ref": "#/items/properties/bilNotesSummary/items" // Reference the same object structure
        }
      }
    }
  }
}


# Prompt
**Task:** Transform the `INPUT_data` provided at the end of this prompt into a JSON array that strictly follows the '# Controlled Output Schema'. Follow the processing steps below precisely and meticulously.

**Processing Steps:**

1.  **Identify Prescriptions & Base Names:**
    *   Extract all prescription entries (objects with `finalDisplayEntityId` and `medicineName`) from the `INPUT_data`.
    *   For each prescription, determine its `rxNumber` (from `finalDisplayEntityId`) and `drugName` (from `medicineName`).
    *   Derive the `baseDrugName` for each prescription. The `baseDrugName` is the core pharmaceutical name, typically the first *meaningful* word (e.g., "HUMIRA", "NUPLAZID", "SPRAVATO"). Ignore prefixes like "T6", "A1", "F7" when determining the base name for matching purposes *unless* the drug name *only* consists of that. For "T6 HUMIRA", the base name is "HUMIRA". For "A1 NUPLAZID", base is "NUPLAZID". For "F7 SPRAVATO", base is "SPRAVATO".
    *   Store the list of all prescriptions, each with its `rxNumber`, `drugName`, and `baseDrugName`. Keep this list accessible for checking in step 3.b.i. Let's call this `all_prescriptions`.

2.  **Identify & Pre-process All Notes:**
    *   Extract all note entries (objects with `noteTxt`, `noteTypeCd`, etc.) from the `INPUT_data`.
    *   For each note:
        *   Extract the date part (`MM/DD/YYYY`) from `createDate`. Store this as `noteDate`.
        *   Determine the primary text for processing: Start with `noteSummaryTxt`. If `noteSummaryTxt` is null, empty, or effectively just whitespace/periods after trimming, use `noteTxt` instead as the primary text.
        *   Pre-process the primary text: Replace special spaces (`\u00a0`) with a standard space, collapse multiple consecutive spaces into a single space, and trim leading/trailing whitespace *and periods* from the result. Store this as `processedSummary`.
        *   If the resulting `processedSummary` is empty (even after potentially using `noteTxt` as primary), ignore the note.
        *   Store the note's essential data (`noteId`, `noteDate`, `processedSummary`, `noteTypeCd`, `noteTxt` (original full text)) in a list or map for later reference. **Keep both `processedSummary` and the original `noteTxt` available for association checks.** Let's call this `all_notes`.

3.  **Process Each Prescription Independently:** Initialize an empty list for the final results (`final_results = []`). Iterate through *each* prescription in `all_prescriptions` (let the current one be `current_prescription` with `current_rxNumber`, `current_drugName`, `current_baseDrugName`).
    a.  **Initialize Temporary Note Lists:** Create two empty lists specifically for this `current_prescription`: `current_rx_bil_notes = []` and `current_rx_gen_notes = []`.
    b.  **Associate Notes:** Iterate through *each* note in `all_notes`. For each `current_note`:
        i.  **Check Association Rules for `current_note` against `current_prescription`:** Determine if the `current_note` (using its `processedSummary` and `noteTxt`) is relevant to the `current_prescription`.
            **OVERALL GOAL:** A note mentioning a specific drug formulation (e.g., "DrugX 10mg") should ONLY associate with the prescription for that EXACT formulation (via Rule 2) or its Rx# (via Rule 1). It should NEVER associate with a *different* prescription sharing the same base name (e.g., "DrugX 20mg") via the base name rule (Rule 3). Notes mentioning *only* the base name (e.g., "DrugX") should associate with ALL prescriptions sharing that base name (via Rule 3).

            **Check the following rules STRICTLY IN ORDER (1 -> 2 -> 3). If a rule matches, the note IS relevant for this prescription, and you MUST STOP checking subsequent rules for this specific note-prescription pair.** (Use case-insensitive comparison where appropriate, checking **both** `processedSummary` and `noteTxt` for mentions).

            1.  **Rule 1: Rx Number Match:** Does `processedSummary` contain `current_rxNumber` OR `noteTxt` contain `current_rxNumber`?
                *   If YES: Note is relevant. **STOP checking rules for this pair.**
                *   If NO: Proceed to Rule 2.

            2.  **Rule 2: Exact Drug Name Match:** Does `processedSummary` contain the exact `current_drugName` OR `noteTxt` contain the exact `current_drugName`?
                *   If YES: Note is relevant. **STOP checking rules for this pair.**
                *   If NO: Proceed to Rule 3.

            3.  **Rule 3: Base Drug Name Match (ONLY if Rules 1 & 2 Failed):** This rule applies ONLY if the note mentions the base drug name generically, without mentioning any specific drug formulation that uses that base name.

                *   **(Pre-Check - Exclusion Condition):** Does the `current_note` text (using *both* `processedSummary` and `noteTxt`) contain the *exact* `drugName` of **ANY** prescription in `all_prescriptions` (including `current_prescription` or any other) that shares the `current_baseDrugName`?
                    *   If YES: This note mentions a specific formulation. **Rule 3 does NOT apply. STOP checking rules for this pair.** (The note should only have matched via Rule 1 or 2 to the relevant prescription).
                    *   If NO: Proceed to check Condition A below.

                *   **(Condition A - Base Name Present):** (Only check if Pre-Check was NO). Does `processedSummary` contain `current_baseDrugName` OR `noteTxt` contain `current_baseDrugName`?
                    *   If YES: Note is relevant via Rule 3. (This note will likely be relevant to *multiple* prescriptions sharing the base name).
                    *   If NO: Note is NOT relevant via Rule 3.

                *   *Example 1 (Base Name Only):* Note mentions only "<BaseDrugNameX>". Prescriptions are "Specific Drug Formulation A" (RxA, base "BaseDrugNameX") and "Specific Drug Formulation B" (RxB, base "BaseDrugNameX").
                    *   Checking note against RxA ("Specific Drug Formulation A"): Rule 1=No, Rule 2=No. Rule 3: Pre-Check=No (note contains neither specific formulation). Condition A=Yes. -> **Match via Rule 3.**
                    *   Checking note against RxB ("Specific Drug Formulation B"): Rule 1=No, Rule 2=No. Rule 3: Pre-Check=No (note contains neither specific formulation). Condition A=Yes. -> **Match via Rule 3.**
                    *   *Result:* This note correctly associates with BOTH prescriptions via Rule 3.

                *   *Example 2 (Specific Formulation - "Prefix BaseDrugNameY"):* Note mentions "MANAGER CALLED regarding Prefix BaseDrugNameY.". Prescriptions are "Prefix BaseDrugNameY" (RxA, base "BaseDrugNameY") and "BaseDrugNameY 50mg Dose Kit" (RxB, base "BaseDrugNameY").
                    *   Checking note against RxA ("Prefix BaseDrugNameY"): Rule 1=No. Rule 2=**Yes**. -> **Match via Rule 2.** (STOP here for RxA).
                    *   Checking note against RxB ("BaseDrugNameY 50mg Dose Kit"): Rule 1=No. Rule 2=No. Rule 3: Pre-Check=**Yes** (note *does* contain "Prefix BaseDrugNameY"). -> **No Match.** (STOP checking rules).
                    *   *Result:* This note correctly associates ONLY with RxA ("Prefix BaseDrugNameY").

                *   *Example 3 (Specific Formulation - "Prefix BaseDrugNameX"):* Note mentions "PATIENT BILLING TO Prefix BaseDrugNameX.". Prescriptions are "BaseDrugNameX 50mg" (RxA, base "BaseDrugNameX") and "Prefix BaseDrugNameX" (RxB, base "BaseDrugNameX").
                    *   Checking note against RxA ("BaseDrugNameX 50mg"): Rule 1=No. Rule 2=No. Rule 3: Pre-Check=**Yes** (note *does* contain "Prefix BaseDrugNameX"). -> **No Match.** (STOP checking rules).
                    *   Checking note against RxB ("Prefix BaseDrugNameX"): Rule 1=No. Rule 2=**Yes**. -> **Match via Rule 2.** (STOP here for RxB).
                    *   *Result:* This note correctly associates ONLY with RxB ("Prefix BaseDrugNameX").

        ii. **Add to Temporary Lists if Relevant:** If the `current_note` was found relevant to the `current_prescription` by *any* of the rules above (Rule 1, 2, OR 3):
            *   If `current_note.noteTypeCd` is "BIL", add `{noteDate: current_note.noteDate, processedSummary: current_note.processedSummary}` to `current_rx_bil_notes`.
            *   If `current_note.noteTypeCd` is "GEN", add `{noteDate: current_note.noteDate, processedSummary: current_note.processedSummary}` to `current_rx_gen_notes`.
            *   **CRITICAL REMINDER:** A single note can be relevant to multiple prescriptions. This is REQUIRED if the note matches multiple Rx numbers (Rule 1), multiple exact drug names (Rule 2), OR if it matches *only* via Rule 3 (base name only, Pre-Check was NO) - in the Rule 3 case, it **MUST** be added to the temporary lists of **ALL** prescriptions for which Rule 3 was satisfied for that note.

    c.  **Summarize BIL Notes for Current Rx (Aggregate by Date):**
        i.   Group the notes in `current_rx_bil_notes` by `noteDate`.
        ii.  Initialize an empty list: `bil_summary_objects = []`.
        iii. **For each unique `noteDate` found in the grouped BIL notes:**
            *   Collect *all* unique `processedSummary` values corresponding to this *single* `noteDate` from `current_rx_bil_notes`. (Avoid duplicates if the same note summary was added multiple times).
            *   **Join the unique summaries for this date** into a single string using ". " as the separator.
            *   Ensure the joined string ends with a period (add one if necessary, unless it already ends with punctuation like '.', '?', '!').
            *   **Create ONLY ONE summary object** for this `noteDate`: `{ "noteDate": "MM/DD/YYYY", "summaryText": "MM/DD/YYYY: Joined Summaries." }`.
            *   **Add this single, aggregated object to the `bil_summary_objects` list.**
        iv.  Sort `bil_summary_objects` chronologically by `noteDate`. This list is the final `bilNotesSummary` for the `current_prescription`. (It will be `[]` if `current_rx_bil_notes` was empty).
    d.  **Summarize GEN Notes for Current Rx (Aggregate by Date):**
        i.   Perform the *exact same* grouping, collection of unique summaries, joining, single-object-creation-per-date, and sorting process on the `current_rx_gen_notes` list.
        ii.  The resulting sorted list of aggregated objects is the final `genNotesSummary` for the `current_prescription`. (It will be `[]` if `current_rx_gen_notes` was empty).
    e.  **Add to Final Results:** Create the final entry for the current prescription using its `rxNumber` (as `rxNumber`), `drugName` (as `drugName`), the generated `bilNotesSummary` (from 3c.iv), and the generated `genNotesSummary` (from 3d.ii). Add this entry to the `final_results` list.
4.  **Output Final Result:** Output the complete `final_results` list as a JSON array.

**CRITICAL LOGIC & SCHEMA ADHERENCE:**
*   **Association Precision:** Ensure Step 3.b.i correctly distinguishes between exact `drugName` matches (Rule 2) and base `baseDrugName` matches (Rule 3), using *both* `processedSummary` and `noteTxt` for checks. The Rule 3 Pre-Check (Exclusion Condition) is crucial for preventing notes about specific formulations from wrongly associating via the base name. **Strict rule order (1 -> 2 -> 3) and the OVERALL GOAL statement are essential.**
*   **Broad Base Name Association:** Ensure that if a note matches *only* via Rule 3 (mentions base name but *no* specific formulations sharing that base, i.e., Pre-Check is NO), it is correctly associated with **ALL** prescriptions sharing that base name.
*   **Date Aggregation:** Step 3c and 3d MUST aggregate *all* unique summaries for a specific date into a *single* `DatedNoteSummary` object for that date. Do NOT create multiple objects for the same date within `bilNotesSummary` or `genNotesSummary`.
*   **Correct Structure:** The final `bilNotesSummary` and `genNotesSummary` arrays MUST contain either `[]` (if no relevant notes) or an array of `DatedNoteSummary` objects (`{ "noteDate": "...", "summaryText": "..." }`). They MUST NOT contain simple strings or violate the one-object-per-date rule.
*   **Completeness:** Ensure all prescriptions from the input are present in the output array.

**Learning Example:** Analyze the Input/Output examples carefully. Note how a note mentioning only "HUMIRA" (base name) in `input1.json` correctly appears in the `genNotesSummary` for *multiple* different HUMIRA prescriptions in `output1.json`. Contrast this with how a note mentioning "RX#10210309" *only* appears for that specific Rx. With the refined logic, a note mentioning *only* "<BaseDrugNameX>" (base name) **must** appear under *both* "Specific Drug Formulation A" and "Specific Drug Formulation B" because it doesn't mention either specific formulation (Rule 3 Pre-Check is NO, Condition A is YES for both). However, a note mentioning "Specific Drug Formulation A" should *only* appear under that specific prescription (via Rule 2) and not under "Specific Drug Formulation B" (Rule 3 Pre-Check is YES, so Rule 3 fails). Similarly, a note mentioning "Prefix BaseDrugNameY" should *only* match "Prefix BaseDrugNameY" via Rule 2 and should *not* match "BaseDrugNameY 50mg Dose Kit" via Rule 3 (because the Pre-Check fails).

**INPUT_data:**
```json
[
    {
        "noteTxt": "BOARD OF PHARMACY HAD TAKEN LEGAL ACTION FOR PAYOR OF NUPLAZID.",
        "noteSummaryTxt": "BOARD OF PHARMACY HAD TAKEN LEGAL ACTION FOR PAYOR",
        "createDate": "03/25/2025 01:35:07 PM",
        "noteId": "2147949106",
        "noteTypeCd": "BIL",
        "noteTypeInd": "BILLING/PRICING/PRIOR AUTH"
    },
    {
        "noteTxt": "PATIENT EHR_IND TURNS TO Y DUE TO NUPLAZID 40MG.",
        "noteSummaryTxt": "PATIENT EHR_IND TURNS TO Y DUE TO NUPLAZID 40MG.",
        "createDate": "03/25/2025 01:32:10 PM",
        "noteId": "2147949104",
        "noteTypeCd": "BIL",
        "noteTypeInd": "BILLING/PRICING/PRIOR AUTH"
    },
    {
        "noteTxt": "PATIENT BILLING TO A1 NUPLAZID.",
        "noteSummaryTxt": " PATIENT BILLING TO A1 NUPLAZID.",
        "createDate": "03/28/2025 01:33:10 PM",
        "noteId": "2147949104",
        "noteTypeCd": "BIL",
        "noteTypeInd": "BILLING/PRICING/PRIOR AUTH"
    },
    {
        "noteTxt": "NUPLAZID PRESCRPTION IS A KIND OF HEMO TYPE.",
        "noteSummaryTxt": "NUPLAZID PRESCRPTION IS A KIND OF HEMO TYPE.",
        "createDate": "03/25/2025 01:23:49 PM",
        "noteId": "2147949095",
        "noteTypeCd": "BIL",
        "noteTypeInd": "BILLING/PRICING/PRIOR AUTH"
    },
    {
        "noteTxt": "MANAGER CALLED RX# 10207950 IS AN EPC PRESCRIPTION.",
        "noteSummaryTxt": "MANAGER CALLED RX# 10207950 IS AN EPC PRESCRIPTION",
        "createDate": "03/25/2025 12:36:11 PM",
        "noteId": "2147949046",
        "noteTypeCd": "GEN",
        "noteTypeInd": "GENERAL"
    },
    {
        "noteTxt": "EHR TEAM CONNECTED FOR RX# 10207950 FOR PROCESSING THE RX.",
        "noteSummaryTxt": "EHR TEAM CONNECTED FOR RX# 10207950 FOR PROCESSING",
        "createDate": "03/25/2025 12:31:02 PM",
        "noteId": "2147949044",
        "noteTypeCd": "BIL",
        "noteTypeInd": "BILLING/PRICING/PRIOR AUTH"
    },
    {
        "noteTxt": "Received with for <NUPLAZID 10 MG> within SPRx Intake",
        "noteSummaryTxt": "Received with for <NUPLAZID 10 MG> within SPRx Intake ",
        "createDate": "02/18/2025 06:37:05 AM",
        "noteId": "2147910671",
        "noteTypeCd": "GEN",
        "noteTypeInd": "GENERAL"
    },
    {
        "noteTxt": "Received <8793279> with <RX> for <NUPLAZID 56 MG> within SPRx Intake. Doc(s) can be viewed within Pt SPRx Profile-Images",
        "noteSummaryTxt": "Received <8793279> with <RX> for <NUPLAZID 56 MG> withi",
        "createDate": "02/17/2025 06:37:05 AM",
        "noteId": "2147910671",
        "noteTypeCd": "GEN",
        "noteTypeInd": "GENERAL"
    },
    {
        "noteTxt": "MANAGER CALLED regarding F7 Spravato.",
        "noteSummaryTxt": " MANAGER CALLED regarding F7 Spravato.",
        "createDate": "03/25/2025 12:36:11 PM",
        "noteId": "2147949046",
        "noteTypeCd": "GEN",
        "noteTypeInd": "GENERAL"
    },
    {
        "noteTxt": "Patient enrolled via < FAX > by < ORGB_002 >, Phone/Fax #: < >, < > for < NUPLAZID >. Method to obtain rx: < >. Referring MD: < USERONE AUTOMATION > < >.",
        "noteSummaryTxt": "New Patient Enrollment",
        "createDate": "02/17/2025 06:36:59 AM",
        "noteId": "2147910666",
        "noteTypeCd": "GEN",
        "noteTypeInd": "GENERAL"
    },
    {
        "noteTxt": "SPRAVATO is a good drug,can be taken for multiple disease.",
        "noteSummaryTxt": "SPRAVATO is a good drug,can be taken for multiple disease.",
        "createDate": "02/17/2025 06:37:05 AM",
        "noteId": "2147910671",
        "noteTypeCd": "GEN",
        "noteTypeInd": "GENERAL"
    },
    {
        "finalDisplayEntityId": "10208441-00",
        "medicineName": "SPRAVATO 56MG DOSE KIT"
    },
    {
        "finalDisplayEntityId": "10207950-00",
        "medicineName": "NUPLAZID 56 MG"
    },
    {
        "finalDisplayEntityId": "10208443-00",
        "medicineName": "SPRAVATO 84MG DOSE KIT"
    },
    {
        "finalDisplayEntityId": "10208432-00",
        "medicineName": "F7 SPRAVATO"
    },
    {
        "finalDisplayEntityId": "10208478-00",
        "medicineName": "A1 NUPLAZID"
    }
]
