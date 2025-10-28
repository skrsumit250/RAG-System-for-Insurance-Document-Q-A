import json
from tests.rouge_score import rouge_scorer
from src.retrieval import main

with open("tests/questions.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
# for data in test_data["elements"][:1]:
#     print(data["question"])
test_data = test_data["elements"]
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

results = []


def scoring():
    print("I am called")
    for item in test_data:
        question = item["question"]
        reference_answer = item["answer"]
        # print(len(reference_answer))
        ai_answer = main.ask_rag_pipeline(question)
        # print(len(ai_answer))
        scores = scorer.score(reference_answer, ai_answer)
        results.append(
            {
                "question": question,
                "reference": reference_answer,
                "generated": ai_answer,
                "rouge1_f": scores["rouge1"].fmeasure,
                "rouge2_f": scores["rouge2"].fmeasure,
                "rougeL_f": scores["rougeL"].fmeasure,
            }
        )


scoring()
avg_rouge1 = sum(r["rouge1_f"] for r in results) / len(results)
avg_rouge2 = sum(r["rouge2_f"] for r in results) / len(results)
avg_rougeL = sum(r["rougeL_f"] for r in results) / len(results)

print("\nAVERAGE ROUGE SCORES")
print(f"ROUGE-1: {avg_rouge1:.4f}")
print(f"ROUGE-2: {avg_rouge2:.4f}")
print(f"ROUGE-L: {avg_rougeL:.4f}")

with open("tests/rag_rouge_results.json", "w") as f:
    json.dump(results, f, indent=2)