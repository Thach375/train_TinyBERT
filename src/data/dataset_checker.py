from datasets import load_dataset

ds = load_dataset("glue", "mnli")

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

for i, sample in enumerate(ds["validation_matched"].select(range(5))):
    print(f"--- Sample {i+1} ---")
    print(f"Premise:    {sample['premise']}")
    print(f"Hypothesis: {sample['hypothesis']}")
    print(f"Label:      {id2label[sample['label']]} ({sample['label']})")
    print()