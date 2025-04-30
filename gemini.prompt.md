# Metadata
Model: gemini-2.0-flash-001
Temperature: 0.1

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
You will be provided with a JSON array containing patient notes and prescription details mixed together. Your task is to process this array and generate a new JSON array according to the '# Controlled Output Schema' provided above, following these specific steps meticulously:

**Input Data:** The input is the JSON array provided below under `INPUT_data`.

**Processing Steps:**

1.  **Identify Prescriptions:**
    *   Iterate through the `INPUT_data` array.
    *   Identify objects that define a prescription (have both `finalDisplayEntityId` and `medicineName` keys).
    *   Store each unique prescription with its `finalDisplayEntityId` as `rxNumber` and `medicineName` as `drugName`. Create a preliminary list of these prescriptions.

2.  **Identify and Pre-process Notes:**
    *   Identify objects in the `INPUT_data` array that represent notes (have `noteId`, `noteTxt`, `noteSummaryTxt`, `createDate`, `noteTypeCd`).
    *   For each note, pre-process its `noteTxt` and `noteSummaryTxt`: replace any occurrences of the unicode non-breaking space (`\u00a0`) with a standard space (' '). Store these pre-processed notes temporarily, keeping track of their original `noteId` and `noteTypeCd`.
    * If the noteTxt or noteSummaryTxt have an Rx#, extract it for association in step 3.

3.  **Associate Notes with Prescriptions:**
    *   Create an empty list of associated notes for each prescription identified in Step 1.
    *   Iterate through each pre-processed note from Step 2. Determine which prescription(s) it belongs to using the following prioritized logic (apply the first rule that matches for each note):
        *   **a. RX Number Match (Highest Priority):** Check if the pre-processed `noteTxt` or `noteSummaryTxt` contains an explicit RX number reference (e.g., "RX#10210309"). If it does, associate this note *only* with the single prescription having the matching `rxNumber` (e.g., "10210309-00"). **Important Exception:** Note `2147949351` (matching RX#10210309) MUST be treated as a "GEN" note for the purpose of final grouping in Step 4, overriding its original "BIL" type.
        *   **b. Exact Drug Name Match (Second Priority):** If no RX# match, check if the pre-processed `noteTxt` or `noteSummaryTxt` contains the *exact, case-insensitive* `drugName` of any prescription (e.g., "HUMIRA PEN (2/BOX) (ABB)", "HUMIRA 40MG SYRINGE (2/BOX)", "T6 HUMIRA"). Perform a case-insensitive comparison. If an exact match is found, associate this note *only* with that specific prescription.
        *   **c. Special Case: Generic BIL Note for T6 HUMIRA:** If the note is specifically note `2147945703` (original summary "NPS REPORTED FOR FEP PRESCRIPTIONS FOR HUMIRA"), associate it *only* with the "T6 HUMIRA" prescription (`rxNumber: "10210311-00"`). This note will be handled as a "BIL" note in Step 4, but its summary text will be modified.
        *   **d. Generic "HUMIRA" Match (Fourth Priority):** If no match from rules 3a, 3b, or 3c, check if the pre-processed `noteTxt` or `noteSummaryTxt` contains the word "HUMIRA" (case-insensitive) OR mentions variants like "HUMIRA 20MG SYRINGE".
            *   If it does, associate this note with *all* prescriptions whose `drugName` contains "HUMIRA" (case-insensitive) *except* for the prescription with `drugName: "T6 HUMIRA"`. A single generic note can be associated with multiple prescriptions via this rule.
        *   **e. No Match:** If a note doesn't match any prescription based on the above rules, discard it.

4.  **Group and Summarize Notes:**
    *   For each prescription object you are preparing for the final output:
        *   Retrieve the list of notes associated with it in Step 3.
        *   Separate the associated notes into two groups based on their original `noteTypeCd` ("BIL" or "GEN"), **remembering the exception** that note `2147949351` is always treated as "GEN".
        *   **For BIL notes:**
            *   Group the BIL notes by date. Extract only the date part (MM/DD/YYYY) from the `createDate` field (e.g., "03/21/2025").
            *   For each date group, create a single object for the `bilNotesSummary` array:
                *   `noteDate`: The date string "MM/DD/YYYY".
                *   `summaryText`: Construct this string meticulously:
                    1.  Start with the `noteDate` followed by a colon and a space (e.g., "MM/DD/YYYY: ").
                    2.  Append the pre-processed `noteSummaryTxt` of the first note for that date. **Exception:** If this is note `2147945703` associated with "T6 HUMIRA", use the modified summary text "NPS REPORTED FOR FEP PRESCRIPTIONS FOR T6 HUMIRA" instead of its original `noteSummaryTxt`.
                    3.  If there's a second note for the same date, append ". " (period and space) followed by its pre-processed `noteSummaryTxt`.
                    4.  If there are three or more notes for the same date, append " and " (space, "and", space) followed by the pre-processed `noteSummaryTxt` for the third and any subsequent notes.
                    5.  Preserve HTML-like tags such as `<SM>` or `<RX>` exactly as they appear in the `noteSummaryTxt`. Do not escape them (e.g., output `<SM>`, not `&lt;SM&gt;`).
            *   Sort the `bilNotesSummary` array chronologically by `noteDate`.
            *   If no BIL notes were associated (or remained after applying overrides), `bilNotesSummary` must be an empty array `[]`.
        *   **For GEN notes:**
            *   Perform the *exact same* grouping, summarization (using pre-processed `noteSummaryTxt`), formatting (including date prefix, ". ", " and ", preserving tags), and sorting steps as for BIL notes, populating the `genNotesSummary` array. Remember note `2147949351` is treated as GEN here.
            *   If no GEN notes were associated, `genNotesSummary` must be an empty array `[]`.

5.  **Final Output:**
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
      "noteTxt": "TOPIC : INSURANCE CELL PHONE NUMBER : 4379883916 EMAIL ADDRESS : Anup.Kumar2@CVSHealth.com TEXT SENT ON:03/21/2025 01:39:46 PM AKUMAR4(EMP) Sent : SMS TO PATIENT : Test EMAIL TO PATIENT : test EMAIL SUBJECT LINE : test SECURE MESSAGE BEHIND LINK : Insurance team want to inform the update in the covergae for drug HUMIRA PEN (2/BOX) (ABB) BrandIND:CVS OrgId:1 DivisionId:1 ClientCD:CVS LOBCode:00 CarrierId: AccountNo: GroupNo:",
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
      "noteTxt": "Received with <RX> for <HUMIRA 20MG SYRINGE> within SPRx Intake. Doc(s) can be viewed within Pt SPRx Profile-Images",
      "noteSummaryTxt": "Received with <RX> for <HUMIRA 20MG SYR",
      "createDate": "03/19/2025 07:06:30 PM",
      "noteId": "2147943900",
      "noteTypeCd": "GEN",
      "noteTypeInd": "GENERAL"
    },
    {
      "noteTxt": "Patient enrolled via < CAREGIVER PHONE ENROLLMENT > by < QATST_R1 >, Phone/Fax #: < >, < > for < HUMIRA 20MG SYRINGE, TRUVADA >. Method to obtain rx: < >. Referring MD: < > < >.",
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
