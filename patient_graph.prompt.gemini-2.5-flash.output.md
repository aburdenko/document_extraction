# Gemini Output for: patient_graph.prompt.md
## Request Configuration
- **Model:** gemini-2.5-flash
- **System Instructions Provided:** Yes
- **Temperature:** 0.0
- **'# Ground Truth' Section Found:** Yes
- **'# RagEngine' Section Found:** Yes
- **RAG Tool Provided to Model:** Yes
- **'# Controlled Output Schema' Section Found:** Yes
- **'# Functions' Section Found:** No
- **Function Calling Active (Tools Provided):** No
- **JSON Output Mode Active (MIME Type):** True
- **Schema Parsed & Applied (for JSON Mode):** No
- **Safety Settings Applied:** [category: HARM_CATEGORY_HARASSMENT
threshold: BLOCK_MEDIUM_AND_ABOVE
, category: HARM_CATEGORY_HATE_SPEECH
threshold: BLOCK_MEDIUM_AND_ABOVE
, category: HARM_CATEGORY_SEXUALLY_EXPLICIT
threshold: BLOCK_MEDIUM_AND_ABOVE
, category: HARM_CATEGORY_DANGEROUS_CONTENT
threshold: BLOCK_MEDIUM_AND_ABOVE
]
- **Timestamp:** 2025-08-21 18:03:53.206907

## Usage Metadata (Primary Call)
- **Prompt Token Count:** 93
- **Candidates Token Count:** 1973
- **Total Token Count:** 4000
- **Time Taken:** 21.34 seconds
- **Estimated Cost:** $0.002001

## RAW OUTPUT

```json
{
  "allergies": [
    {
      "status": "No Known Allergies"
    }
  ],
  "care_plans": [
    {
      "activities": [
        "Dressing change management",
        "Behavior to prevent sun exposure",
        "Behavior to prevent infection"
      ],
      "name": "Burn care",
      "reason": "First degree burn",
      "stop_date": "2015-04-17"
    },
    {
      "activities": [
        "Rest, ice, compression and elevation treatment programme",
        "Stretching exercises"
      ],
      "name": "Physical therapy procedure",
      "reason": "Sprain of ankle",
      "stop_date": "2008-10-06"
    },
    {
      "activities": [
        "Antenatal education",
        "Antenatal risk assessment",
        "Antenatal blood tests"
      ],
      "name": "Routine antenatal care",
      "reason": "Normal pregnancy",
      "stop_date": "2007-11-13"
    }
  ],
  "conditions": [
    {
      "end_date": null,
      "name": "Stroke",
      "start_date": "2015-05-19"
    },
    {
      "end_date": "2015-05-15",
      "name": "First degree burn",
      "start_date": "2015-04-17"
    },
    {
      "end_date": "2015-04-28",
      "name": "Normal pregnancy",
      "start_date": "2015-04-07"
    },
    {
      "end_date": "2013-02-26",
      "name": "Blighted ovum",
      "start_date": "2013-02-19"
    },
    {
      "end_date": null,
      "name": "Miscarriage in first trimester",
      "start_date": "2013-02-19"
    },
    {
      "end_date": "2013-02-26",
      "name": "Normal pregnancy",
      "start_date": "2013-02-19"
    },
    {
      "end_date": "2012-01-16",
      "name": "Acute viral pharyngitis (disorder)",
      "start_date": "2012-01-04"
    },
    {
      "end_date": "2010-02-12",
      "name": "Acute viral pharyngitis (disorder)",
      "start_date": "2010-02-04"
    },
    {
      "end_date": "2008-10-20",
      "name": "Sprain of ankle",
      "start_date": "2008-10-06"
    },
    {
      "end_date": "2008-06-10",
      "name": "Antepartum eclampsia",
      "start_date": "2008-05-27"
    },
    {
      "end_date": "2008-06-10",
      "name": "Normal pregnancy",
      "start_date": "2007-11-13"
    },
    {
      "end_date": null,
      "name": "Body mass index 30+ - obesity (finding)",
      "start_date": "2000-08-15"
    }
  ],
  "medications": [
    {
      "dosage": "100 MG",
      "form": "Injection",
      "name": "Alteplase",
      "stop_date": "2015-05-19"
    },
    {
      "dosage": "75 MG",
      "form": "Oral Tablet",
      "name": "Clopidogrel",
      "stop_date": "2015-05-19"
    },
    {
      "dosage": "220 MG",
      "form": "Oral Tablet",
      "name": "Naproxen sodium",
      "stop_date": "2015-04-17"
    },
    {
      "dosage": "28 Day Pack",
      "name": "Jolivette",
      "stop_date": "2011-07-07"
    },
    {
      "dosage": "325 MG",
      "form": "Oral Tablet",
      "name": "Acetaminophen",
      "stop_date": "2008-10-06"
    },
    {
      "dosage": "28 Day Pack",
      "name": "Errin",
      "stop_date": "2006-05-02"
    }
  ],
  "patient": {
    "birth_date": "1985-06-11",
    "deceased_date": "2015-05-26",
    "ethnicity": "Mexican",
    "gender": "F",
    "marital_status": "M",
    "name": "Ana Luisa894 Montalvo564",
    "race": "Other"
  },
  "reports": [
    {
      "date": "2015-05-26",
      "details": [
        {
          "description": "Cause of Death [US Standard Certificate of Death] Stroke"
        }
      ],
      "type": "U.S. standard certificate of death - 2003 revision"
    },
    {
      "date": "2013-09-17",
      "details": [
        {
          "name": "Leukocytes [#/volume] in Blood by Automated count",
          "unit": "10*3/uL",
          "value": "9.5"
        },
        {
          "name": "Erythrocytes [#/volume] in Blood by Automated count",
          "unit": "10*6/uL",
          "value": "4.5"
        },
        {
          "name": "Hemoglobin [Mass/volume] in Blood",
          "unit": "g/dL",
          "value": "15.7"
        },
        {
          "name": "Hematocrit [Volume Fraction] of Blood by Automated count",
          "unit": "%",
          "value": "44.4"
        },
        {
          "name": "MCV [Entitic volume] by Automated count",
          "unit": "fL",
          "value": "83.1"
        },
        {
          "name": "MCH [Entitic mass] by Automated count",
          "unit": "pg",
          "value": "30.4"
        },
        {
          "name": "MCHC [Mass/volume] by Automated count",
          "unit": "g/dL",
          "value": "33.8"
        },
        {
          "name": "Erythrocyte distribution width [Entitic volume] by Automated count",
          "unit": "fL",
          "value": "42.0"
        },
        {
          "name": "Platelets [#/volume] in Blood by Automated count",
          "unit": "10*3/uL",
          "value": "238.2"
        },
        {
          "name": "Platelet distribution width [Entitic volume] in Blood by Automated count",
          "unit": "fL",
          "value": "180.8"
        },
        {
          "name": "Platelet mean volume [Entitic volume] in Blood by Automated count",
          "unit": "fL",
          "value": "10.0"
        }
      ],
      "type": "Complete blood count (hemogram) panel - Blood by Automated count"
    }
  ]
}
```


## Total Estimated Cost

**Total:** $0.002001
