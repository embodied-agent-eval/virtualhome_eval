import argparse
import os
from virtualhome_eval.agent_eval import agent_evaluation


def main():
    parser = argparse.ArgumentParser(description="VirtualHome Evaluation CLI")
    parser.add_argument(
        "--mode",
        choices=["generate_prompts", "evaluate_results"],
        default="generate_prompts",
        help="Mode of operation (default: generate_prompts)",
    )
    parser.add_argument(
        "--eval-type",
        choices=[
            "action_sequence",
            "transition_model",
            "goal_interpretation",
            "subgoal_decomposition",
        ],
        default="goal_interpretation",
        help="Type of evaluation (default: goal_interpretation)",
    )
    parser.add_argument(
        "--resource-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources"),
        help="Path to the resources directory",
    )
    parser.add_argument(
        "--llm-response-path",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "llm_response"
        ),
        help="Path to LLM response directory",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"),
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"),
        help="Path to the evaluation directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/",
        help="Path to the output directory (default: output/)",
    )
    parser.add_argument(
        "--dataset",
        choices=["virtualhome", "behavior"],
        default="virtualhome",
        help="The dataset to use (default: virtualhome)",
    )
    parser.add_argument(
        "--scene-id", type=int, default=1, help="The VirtualHome scene ID (default: 1)"
    )

    args = parser.parse_args()

    result = agent_evaluation(
        mode=args.mode,
        eval_type=args.eval_type,
        resource_dir=args.resource_dir,
        llm_response_path=args.llm_response_path,
        dataset_dir=args.dataset_dir,
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        scene_id=args.scene_id,
    )

    if result is not None:
        print("Evaluation Results:", result)


if __name__ == "__main__":
    main()
