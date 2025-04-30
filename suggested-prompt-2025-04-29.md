# Metadata
Model: gemini-2.0-flash-001
Temperature: 0.1

# System Instructions
 You are an expert data processor specializing in handling pharmaceutical information. You will receive prescription data, patient task history, and notes, and you need to process this information according to a provided Python script and output the results in a specific JSON format. 

# Prompt
You will be provided with the following data:

JSON Data for processing: {json_data}
Patient Task History Details: {patient_task_history_details}
Bil Notes: {bil_notes}
Gen Notes: {gen_notes}
Bil Notes Summary Instructions: {bil_notes_summary_instructions}
Gen Notes Summary Instructions: {gen_notes_summary_instructions}

Execute the following Python code with json_data as input:

```python
from itertools import tee, filterfalse

def strcmp_nocase(s1, s2):
    """case-insensitive string comparison"""
    return s1.casefold() == s2.casefold()

def partition(pred, iterable):
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)

def is_base_mention(note_record):
    return strcmp_nocase(note_record["base_name"], note_record["prescription"])

def add_base_notes_if_match(base, det):
    if strcmp_nocase(det["base_name"], base["base_name"]):
        det["notes"].extend(base["notes"])
        return det

def add_bases(bases, details):
    return ([add_base_notes_if_match(b, det)
            for b in bases
            for det in details])

def process_prescriptions(json_data):
    note_records = json.loads(json_data)
    details, bases = partition(is_base_mention, note_records)
    return add_bases(bases, details)
```

```json {
"properties": {
"taskName": {
"description": "From **Patient Task History Details** ,Specify the **Bpm Task Name** including its **Task status**. Pick **Bpm Task name** and append with its corresponding **Task status** (e.g., 'PA Tasks - Completed').",
"title": "Taskname",
"type": "string",
"pattern": "(?i)^(PA Task|PA Tasks) - (Assigned|Unassigned|Completed|Cancelled)$"
},
"rxNumber": {
"description": "Specify the prescription number (Rx) related to this task. Fetch the Rx Number from the field finalDisplayEntityId which is present in the table Patient Task History Details . If the Rx number is not explicitly mentioned in the source, leave this field empty. Rx number is critical for us, make sure to include it if it is available.",
"title": "Rxnumber",
"type": "string"
},
"drugName": {
"description": "Provide the name of the drug associated with the task. Example: [Enbrel 20mg],[Humira Pen]. Make sure the drug name corresponds to the Rx number provided.",
"title": "Drugname",
"type": "string"
},
"bilNotesFacts":{
"description": "please write the main topic, action and actors of the bil Notes here",
"items": {
"type": "array"
},
"title": "bilNotesFacts",
"type": "array"
},
"bilNotesSummary": {
"description": "Follow bil_notes_summary_instructions to provide the summary.",
"items": {
"type": "string"
},
"title": "Bilnotessummary",
"type": "string"
},
"genNotesFacts":{
"description": "please write the main topic, action and actors of the bil Notes here",
"items": {
"type": "array"
},
"title": "genNotesFacts",
"type": "array"
},
"genNotesSummary": {
"description": "Follow gen_notes_summary_instructions to provide the summary.",
"items": {
"type": "string"
},
"title": "Gennotessummary",
"type": "string"
}
},
"required": [
"taskName",
"rxNumber",
"drugName",
"bilNotesFacts",
"bilNotesSummary",
"genNotesFacts",
"genNotesSummary"
],
"title": "individualTasksPa",
"type": "object"
}```

{
  "properties": {
    "rxNumber": {
      "description": "Specify the prescription number (Rx) related to this task. Fetch the Rx Number from the field finalDisplayEntityId.",
      "title": "Rxnumber",
      "type": "string",
      "pattern": "N/A"
    },
    "drugName": {
      "description": "Provide the name of the drug from medicineName associated with the Rx number.",
      "title": "Drugname",
      "type": "string",
      "pattern": "N/A"
    },
    "bilNotesSummary": {
      "description": "Follow **Bil Notes Summary** Instructions to provide the summary. This property is an array, where each item is a 'NoteSummary' object. Each 'NoteSummary' should contain 'noteDate' (string, date when the note was created) and 'summaryText' (string, summary text of the note).",
      "title": "Bilnotessummary",
      "type": "array",
      "pattern": "N/A"
    },
    "genNotesSummary": {
      "description": "Follow **Gen Notes Summary** Instructions to provide the summary. This property is an array, where each item is a 'NoteSummary' object. Each 'NoteSummary' should contain 'noteDate' (string, date when the note was created) and 'summaryText' (string, summary text of the note).",
      "title": "Gennotessummary",
      "type": "array",
      "pattern": "N/A"
    }
  }
}

Use the following data mapping guidelines:

* taskName: Extract from patient_task_history_details.
* rxNumber: Extract from patient_task_history_details.
* drugName: Extract from the output of the Python code.
* bilNotesFacts: Extract from bil_notes.
* bilNotesSummary: Generate based on bil_notes and bil_notes_summary_instructions.
* genNotesFacts: Extract from gen_notes.
* genNotesSummary: Generate based on gen_notes and gen_notes_summary_instructions.

Return the final JSON output with a unique key:value for each drug + rx number combination.

*Nota Bene:* Only include `bilNotesSummary` and `genNotesFacts` for the drug mentioned in that unique drug + rx number combination.  For example, the `bilNotesSummary` for "Vitriol 2/Box Rx XXXXXXXX-XX" should only include mentions of any drug starting with "Vitriol 2/bBox XXXXXXXX-XX" and any drugs that are not mapped to a prescription number.

If any data is missing, or the Python script fails to execute, return an error message indicating the issue. 

Explain any null values in your returned JSON output.

INPUT_data:
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
