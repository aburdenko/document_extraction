# Name
Patient Information Agent (Structured)

# Instruction
You are a helpful agent. Your goal is to solve the user's request by thinking step-by-step and using the available tools.
When you have the final answer, respond directly to the user without calling any more tools.

# Tools
```json
[
  {
    "type": "VertexAiRagRetrieval",
    "name": "retrieve_patient_documentation",
    "description": "Use this tool to retrieve documentation and reference materials for the patient from the RAG corpus.",
    "rag_corpus": "projects/kallogjeri-project-345114/locations/us-central1/ragCorpora/3458764513820540928"
  }
]
```

# Prompt
Please tell me the current diagnosis for patient ana luis montalvo.

# Ground Truth
```json
{
  "nodes": [
    {
      "id": "patient-ana_luisa894_montalvo564",
      "type": "Patient",
      "data": {
        "name": "Ana Luisa894 Montalvo564",
        "age": 64
      }
    },
    {
      "id": "condition-ischemic_stroke-2015-05-19",
      "type": "Condition",
      "data": {
        "diagnosis": "Ischemic Stroke",
        "date": "2015-05-19"
      }
    }
  ],
  "edges": [
    {
      "source": "patient-ana_luisa894_montalvo564",
      "target": "condition-ischemic_stroke-2015-05-19",
      "label": "HAS_CONDITION"
    }
  ]
}
```