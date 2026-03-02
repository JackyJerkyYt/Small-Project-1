import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Find questions common to both gold data JSONs (by index)")
    parser.add_argument("--chat", default="results/gold_data/chat_template/gold_data.json")
    parser.add_argument("--no-chat", dest="no_chat", default="results/gold_data/no_chat_template/gold_data.json")
    parser.add_argument("--output", default="results/gold_data/common_gold_data.json")
    args = parser.parse_args()

    with open(args.chat) as f:
        chat_data = json.load(f)
    with open(args.no_chat) as f:
        no_chat_data = json.load(f)

    chat_by_index = {item["index"]: item for item in chat_data["data"]}
    no_chat_by_index = {item["index"]: item for item in no_chat_data["data"]}

    common_indices = sorted(set(chat_by_index.keys()) & set(no_chat_by_index.keys()))

    common_entries = []
    for idx in common_indices:
        common_entries.append({
            "index": idx,
            "question": chat_by_index[idx]["question"],
            "gold_answer": chat_by_index[idx]["gold_answer"],
            "gold_value": chat_by_index[idx]["gold_value"],
            "chat_template": {
                "num_correct": chat_by_index[idx]["num_correct"],
                "num_rollouts": chat_by_index[idx]["num_rollouts"],
                "accuracy": chat_by_index[idx]["accuracy"],
                "rollouts": chat_by_index[idx]["rollouts"],
            },
            "no_chat_template": {
                "num_correct": no_chat_by_index[idx]["num_correct"],
                "num_rollouts": no_chat_by_index[idx]["num_rollouts"],
                "accuracy": no_chat_by_index[idx]["accuracy"],
                "rollouts": no_chat_by_index[idx]["rollouts"],
            },
        })

    output = {
        "description": "Questions appearing in both chat_template and no_chat_template gold data",
        "chat_template_config": chat_data["config"],
        "no_chat_template_config": no_chat_data["config"],
        "chat_template_total": chat_data["selected_questions"],
        "no_chat_template_total": no_chat_data["selected_questions"],
        "common_count": len(common_entries),
        "data": common_entries,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Chat template questions:      {chat_data['selected_questions']}")
    print(f"No chat template questions:   {no_chat_data['selected_questions']}")
    print(f"Common questions (by index):  {len(common_entries)}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
