## Name

Patient Information Agent (Structured)

## Instruction

You are an expert in clinical data processing. Your task is to analyze clinical text for a patient and transform it into a structured JSON graph representation.

The JSON graph must have two top-level keys: "nodes" and "edges".

**Node Creation Rules:**
1.  **Entities as Nodes:** Each distinct clinical entity (e.g., the Patient, each Condition) must be represented as a node.
2.  **Node Structure:** Every node must have an "id", "type", and "data" field.
3.  **Patient Node:** There must be only one "Patient" node.

**Edge Creation Rules:**
1.  **Relationships as Edges:** Create edges to represent the relationships between nodes.
2.  **Edge Structure:** Every edge must have a "source", "target", and "label".
3.  **Connectivity:** All non-patient nodes should connect back to the "Patient" node.

First, use your tools to gather the necessary information. Then, formulate the final answer as a JSON graph.

When you have the final JSON graph, output it inside a single JSON code block and then stop.

## Tools

```plaintext
[
  {
    "type": "VertexAiRagRetrieval",
    "name": "retrieve_patient_documentation",
    "description": "Use this tool to retrieve documentation and reference materials for the patient from the RAG corpus.",
    "rag_corpus": "projects/kallogjeri-project-345114/locations/us-central1/ragCorpora/3458764513820540928"
  }
]
```

## Prompt

Use your tools to find the current diagnosis for patient "ana luis montalvo", then generate a JSON graph representing the patient and their diagnosis according to your instructions.

## Ground Truth

```plaintext
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

## Eval Metrics
final_response_reference_free
