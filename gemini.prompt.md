# Metadata
Model: gemini-2.0-flash-001
Temperature: 0

# System Instructions
 You are an expert data processor specializing in handling pharmaceutical information. You will receive prescription data, patient task history, and notes, and you need to process this information according to a provided Python script and output the results in a specific JSON format.


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
        "description": "The unique prescription number (Rx)."
      },
      "drugName": {
        "type": "string",
        "description": "The name of the drug (e.g., 'HUMIRA 40MG SYRINGE (2/BOX)')."
      },
      "bilNotesSummary": {
        "type": "array",
        "description": "An array of BIL (billing, pricing, prior auth) note summaries, grouped and summarized by date. If no BIL notes exist for the prescription, this should be an empty array.",
        "items": {
          "type": "object",
          "title": "DatedNoteSummary",
          "description": "A summary of notes for a specific date.",
          "required": [
            "noteDate",
            "summaryText"
          ],
          "properties": {
            "noteDate": {
              "type": "string",
              "description": "The date for which the notes are summarized, formatted as 'MM/DD/YYYY'.",
              "pattern": "^(0[1-9]|1[0-2])\\/(0[1-9]|[12][0-9]|3[01])\\/\\d{4}$"
            },
            "summaryText": {
              "type": "string",
              "description": "A consolidated summary of all notes for the 'noteDate'. The summary should start with the date followed by a colon and the summarized text (e.g., 'MM/DD/YYYY: Summary of note 1. Summary of note 2.')."
            }
          }
        }
      },
      "genNotesSummary": {
        "type": "array",
        "description": "An array of GEN (general) note summaries, grouped and summarized by date. If no GEN notes exist for the prescription, this should be an empty array.",
        "items": {
          "type": "object",
          "title": "DatedNoteSummary",
          "description": "A summary of notes for a specific date.",
          "required": [
            "noteDate",
            "summaryText"
          ],
          "properties": {
            "noteDate": {
              "type": "string",
              "description": "The date for which the notes are summarized, formatted as 'MM/DD/YYYY'.",
              "pattern": "^(0[1-9]|1[0-2])\\/(0[1-9]|[12][0-9]|3[01])\\/\\d{4}$"
            },
            "summaryText": {
              "type": "string",
              "description": "A consolidated summary of all notes for the 'noteDate'. The summary should start with the date followed by a colon and the summarized text (e.g., 'MM/DD/YYYY: Summary of note 1. Summary of note 2.')."
            }
          }
        }
      }
    }
  }
}

# Prompt
You will be provided with a JSON array containing patient notes and prescription details mixed together (referred to as `INPUT_data` below). Your task is to process this array and generate a new JSON array according to the '# Controlled Output Schema' provided above, following these specific steps meticulously:

**Processing Steps:**

1.  **Identify Prescriptions:**
    *   Iterate through the `INPUT_data` array.
    *   Identify objects that define a prescription (have both `finalDisplayEntityId` and `medicineName` keys).
    *   Store each unique prescription with its `finalDisplayEntityId` as `rxNumber` and `medicineName` as `drugName`. Create a preliminary list of these prescriptions.

2.  **Identify and Pre-process Notes:**
    *   Identify objects in the `INPUT_data` array that represent notes (have `noteId`, `noteTxt`, `noteSummaryTxt`, `createDate`, `noteTypeCd`).
    *   For each note, perform the following pre-processing on its `noteTxt` and `noteSummaryTxt`:
        *   a. Replace any occurrences of the unicode non-breaking space (`\u00a0`) with a standard space (' ').
        *   b. Replace any occurrences of multiple consecutive standard spaces (including those resulting from step 2a) with a single standard space. (e.g., "word1  word2" becomes "word1 word2", `<   >` becomes `< >`, `Referring MD: <  > < >` becomes `Referring MD: < > < >`). Trim leading/trailing whitespace from the entire text.
    *   Store these fully pre-processed notes temporarily, keeping track of their original `noteId`, `noteTypeCd`, `createDate`, pre-processed `noteTxt`, and pre-processed `noteSummaryTxt`.
    *   Extract any Rx number (e.g., "RX#12345678") found within the pre-processed `noteTxt` or `noteSummaryTxt` for use in association step 3a.

3.  **Associate Notes with Prescriptions:**
    *   A note retains its original `noteTypeCd` (BIL or GEN) throughout this process, which determines its final placement in Step 4.
    *   For each prescription identified in Step 1, create an empty list to hold its associated notes.
    *   Iterate through *each* pre-processed note from Step 2. For *each* note, check against *all* prescriptions using the following logic in order:
        *   **a. RX Number Match:** Check if the note's pre-processed `noteTxt` or `noteSummaryTxt` contains an explicit RX number reference (e.g., "RX#10210309"). If it matches a prescription's `rxNumber` (ignoring any common suffix like `-00` for comparison), associate this note *only* with that single matching prescription. **If this rule matches for a note, do not apply subsequent rules (3b, 3c) to this note.**
        *   **b. Exact Drug Name Match:** If rule 3a did not apply *to this note*, check if the note's pre-processed `noteTxt` or `noteSummaryTxt` contains the *exact, case-insensitive* `drugName` of any prescription. If an exact match is found, associate this note *only* with that specific prescription. **If this rule matches for a note, do not apply subsequent rules (3c) to this note.**
        *   **c. Base Drug Name Match:** If rules 3a and 3b did not apply *to this note*, check if the note's pre-processed `noteTxt` or `noteSummaryTxt` contains a *base drug name* (e.g., 'HUMIRA', 'ENBREL', typically the part of the name before strength or form, identifiable from the set of `drugName`s in the input) (case-insensitive) *without* the more specific strength/formulation details. If such a base name is found, associate this note with *every* prescription whose `drugName` also contains that same base name (case-insensitive). A single note can be associated with multiple prescriptions via this rule.
        *   **d. No Match:** If a note doesn't match any prescription based on the above rules (3a, 3b, 3c) after checking all possibilities, discard it.

4.  **Group and Summarize Notes:**
    *   For each prescription object you are preparing for the final output:
        *   Initialize empty arrays for `bilNotesSummary` and `genNotesSummary`.
        *   Retrieve the final list of *all* notes associated with this specific prescription from Step 3.
        *   Create two temporary lists: one for BIL notes and one for GEN notes associated with this prescription (based on their retained `noteTypeCd`).
        *   **Process BIL Notes:**
            *   Group the notes in the temporary BIL list by date (using only the 'MM/DD/YYYY' part of `createDate`). **Important:** Sort the notes *within* each date group chronologically based on their full `createDate` timestamp before constructing the summary text.
            *   For each date group:
                1.  Create a single `DatedNoteSummary` object.
                2.  Set `noteDate` to the date string "MM/DD/YYYY".
                3.  Construct the `summaryText` string:
                    *   Start with the `noteDate` followed by a colon and a space (e.g., "MM/DD/YYYY: ").
                    *   Append the fully pre-processed **`noteTxt`** (from Step 2b) of the *first* note (chronologically) in this date group.
                    *   For *each subsequent* note within the *same date group*, append ". " (period and space) followed by its fully pre-processed **`noteTxt`**.
                    *   Allow standard JSON string escaping for characters like angle brackets (e.g., `<RX>` may become `&lt;RX&gt;` in the final JSON output if necessary).
                4.  Add this completed `DatedNoteSummary` object to the final `bilNotesSummary` array for the current prescription.
            *   Sort the final `bilNotesSummary` array chronologically by `noteDate`.
        *   **Process GEN Notes:**
            *   Perform the *exact same* grouping by date, sorting within date group, `summaryText` construction (using fully pre-processed **`noteTxt`**, date prefix, ". " separator, allowing standard JSON escaping), and final array population steps as described for BIL notes, but use the temporary GEN list and populate the final `genNotesSummary` array.
            *   **Important:** Ensure the final `genNotesSummary` array is also sorted chronologically by `noteDate`.

5.  **Specific Processing Clarifications (Reinforcement):**
    *   **Input Adherence:** The output `summaryText` MUST be constructed *only* from the pre-processed `noteTxt` values obtained in Step 2b from the `INPUT_data`. Do NOT add information that is not present in the input `noteTxt`. Do not manually shorten or alter the `noteTxt` beyond the pre-processing defined in Step 2.
    *   **Space Collapsing:** Ensure Step 2b correctly collapses all consecutive spaces to one, including around angle brackets (e.g., `< >` not `<  >`).
    *   **Concatenation:** Use exactly ". " (period, space) to separate concatenated `noteTxt` values within a single `summaryText` for a given date, as per Step 4.
    *   **Sorting:** Ensure final `bilNotesSummary` and `genNotesSummary` arrays are sorted chronologically by `noteDate`. Ensure notes *within* a single `summaryText` are concatenated in chronological order based on their full `createDate`.

6.  **Final Output:**
    *   Construct the final JSON array containing an object for each unique prescription identified in Step 1. Each object must have the keys `rxNumber`, `drugName`, `bilNotesSummary`, and `genNotesSummary`, populated according to the association, grouping, and summarization rules defined above. Ensure the output strictly adheres to the '# Controlled Output Schema'.


**INPUT_data:**
[
    {
      "noteTxt": "PATIENT WANTS TO TALK TO MD REGARDING RX#10210309",
      "noteSummaryTxt": "PATIENT WANTS TO TALK TO MD REGARDING RX#10210309",
      "createDate": "03/25/2025 07:31:49 PM",
      "noteId": "2147949351",
      "noteTypeCd": "BIL",
      "noteTypeInd": "BILLING/PRICING/PRIOR AUTH"
    },
    {
      "noteTxt": "NPS REPORTED FOR FEP PRESCRIPTIONS FOR HUMIRA",
      "noteSummaryTxt": "NPS REPORTED FOR FEP PRESCRIPTIONS FOR HUMIRA",
      "createDate": "03/21/2025 03:51:41 PM",
      "noteId": "2147945703",
      "noteTypeCd": "BIL",
      "noteTypeInd": "BILLING/PRICING/PRIOR AUTH"
    },
    {
      "noteTxt": "TOPIC : INSURANCE CELL PHONE NUMBER : 4379883916\\u00a0 EMAIL ADDRESS : Anup.Kumar2@CVSHealth.com\\u00a0 TEXT SENT ON:03/21/2025 01:39:46 PM\\u00a0 AKUMAR4(EMP) Sent :\\u00a0 SMS TO PATIENT : Test\\u00a0 EMAIL TO PATIENT : test\\u00a0 EMAIL SUBJECT LINE : test\\u00a0 SECURE MESSAGE BEHIND LINK : Insurance team want to inform the update in the covergae for drug HUMIRA PEN (2/BOX) (ABB)\\u00a0 BrandIND:CVS OrgId:1 DivisionId:1 ClientCD:CVS LOBCode:00 CarrierId: AccountNo: GroupNo:",
      "noteSummaryTxt": "<SM><INSURANCE(SM)>",
      "createDate": "03/21/2025 03:39:46 PM",
      "noteId": "2147945694",
      "noteTypeCd": "GEN",
      "noteTypeInd": "GENERAL"
    },
    {
      "noteTxt": "MD CALLED FOR HUMIRA PEN (2/BOX) (ABB)",
      "noteSummaryTxt": "MD CALLED FOR HUMIRA PEN (2/BOX) (ABB)",
      "createDate": "03/21/2025 01:36:33 PM",
      "noteId": "2147945570",
      "noteTypeCd": "GEN",
      "noteTypeInd": "GENERAL"
    },
    {
      "noteTxt": "MD CALLED FOR HUMIRA",
      "noteSummaryTxt": "MD CALLED FOR HUMIRA",
      "createDate": "03/20/2025 05:34:53 PM",
      "noteId": "2147944956",
      "noteTypeCd": "GEN",
      "noteTypeInd": "GENERAL"
    },
    {
      "noteTxt": "ADDED NOTES FOR HUMIRA",
      "noteSummaryTxt": "ADDED NOTES FOR HUMIRA",
      "createDate": "03/20/2025 10:48:33 AM",
      "noteId": "2147944391",
      "noteTypeCd": "GEN",
      "noteTypeInd": "GENERAL"
    },
    {
      "noteTxt": "GEN NOTE ADDED FOR T6 HUMIRA",
      "noteSummaryTxt": "GEN NOTE ADDED FOR T6 HUMIRA",
      "createDate": "03/20/2025 12:21:52 AM",
      "noteId": "2147944061",
      "noteTypeCd": "GEN",
      "noteTypeInd": "GENERAL"
    },
    {
      "noteTxt": "GEN NOTE ADDED FOR HUMIRA PEN(2/BOX) (ABB)",
      "noteSummaryTxt": "GEN NOTE ADDED FOR HUMIRA PEN(2/BOX) (ABB)",
      "createDate": "03/20/2025 12:21:11 AM",
      "noteId": "2147944059",
      "noteTypeCd": "GEN",
      "noteTypeInd": "GENERAL"
    },
    {
      "noteTxt": "GEN NOTE ADDED FOR HUMIRA 40 MG SYRINGE(2/BOX)",
      "noteSummaryTxt": "GEN NOTE ADDED FOR HUMIRA 40 MG SYRINGE(2/BOX)",
      "createDate": "03/20/2025 12:20:29 AM",
      "noteId": "2147944058",
      "noteTypeCd": "GEN",
      "noteTypeInd": "GENERAL"
    },
    {
      "noteTxt": "Received\\u00a0 with <RX> for <HUMIRA 20MG SYRINGE> within SPRx Intake.\\u00a0 Doc(s) can be viewed within Pt SPRx Profile-Images",
      "noteSummaryTxt": "Received\\u00a0 with <RX> for <HUMIRA 20MG SYR",
      "createDate": "03/19/2025 07:06:30 PM",
      "noteId": "2147943900",
      "noteTypeCd": "GEN",
      "noteTypeInd": "GENERAL"
    },
    {
      "noteTxt": "Patient enrolled via < CAREGIVER PHONE ENROLLMENT > by < QATST_R1 >, Phone/Fax #: <\\u00a0 >, <\\u00a0 > for < HUMIRA 20MG SYRINGE, TRUVADA >.\\u00a0 Method to obtain rx: <\\u00a0 >. Referring MD: <\\u00a0 \\u00a0> <\\u00a0 >.",
      "noteSummaryTxt": "New Patient Enrollment",
      "createDate": "03/19/2025 07:01:09 PM",
      "noteId": "2147943894",
      "noteTypeCd": "GEN",
      "noteTypeInd": "GENERAL"
    },
    {
      "finalDisplayEntityId": "10210308-00",
      "medicineName": "HUMIRA 40MG SYRINGE (2/BOX)"
    },
    {
      "finalDisplayEntityId": "10210310-00",
      "medicineName": "HUMIRA PEN (2/BOX) (ABB)"
    },
    {
      "finalDisplayEntityId": "10210311-00",
      "medicineName": "T6 HUMIRA"
    },
    {
      "finalDisplayEntityId": "10210309-00",
      "medicineName": "HUMIRA PEN (2/BOX)"
    }
  ]
