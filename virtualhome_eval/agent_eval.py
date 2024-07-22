import argparse
import os
import os.path as osp
import json
import sys

from virtualhome_eval.evaluation.goal_interpretation.scripts.generate_prompts import (
    generate_prompts as goal_input_preparation,
)
from virtualhome_eval.evaluation.transition_modeling.scripts.generate_prompts import (
    generate_prompts as tm_input_preparation,
)
from virtualhome_eval.evaluation.action_sequence.scripts.generate_prompts import (
    generate_prompts as action_input_preparation,
)
from virtualhome_eval.evaluation.subgoal_decomposition.scripts.generate_prompts import (
    generate_prompts as subgoal_input_preparation,
)
from virtualhome_eval.evaluation.goal_interpretation.scripts.evaluate_results import (
    evaluate_results as goal_output_evaluation,
)
from virtualhome_eval.evaluation.transition_modeling.scripts.evaluate_results import (
    evaluate_results as tm_output_evaluation,
)
from virtualhome_eval.evaluation.action_sequence.scripts.evaluate_results import (
    evaluate_results as action_output_evaluation,
)
from virtualhome_eval.evaluation.subgoal_decomposition.scripts.evaluate_results import (
    evaluate_results as subgoal_output_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Agent evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="generate_prompts",
        help="generate_prompts, evaluate_results",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        # default="action_sequence",
        default="subgoal_decomposition",
        help="action_sequence, transition_model, goal_interpretation, subgoal_decomposition",
    )
    parser.add_argument(
        "--resource_dir",
        type=str,
        default="virtualhome_eval/resources/",
        help="resources directory",
    )
    parser.add_argument(
        "--llm_response_path",
        type=str,
        default="virtualhome_eval/llm_response/",
        help="your llm response path",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="virtualhome_eval/dataset/",
        help="dataset directory, necessary only when generating prompts",
    )
    parser.add_argument(
        "--dataset", type=str, default="virtualhome", help="virtualhome, behavior"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="virtualhome_eval/output/",
        help="output directory",
    )
    # virtualhoome
    parser.add_argument("--scene_id", type=int, default=1, help="virtualhome scene id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_type = args.eval_type
    mode = args.mode

    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    if mode == "generate_prompts":
        if eval_type == "action_sequence":
            action_input_preparation(args)
        elif eval_type == "transition_model":
            tm_input_preparation(args)
        elif eval_type == "goal_interpretation":
            goal_input_preparation(args)
        elif eval_type == "subgoal_decomposition":
            subgoal_input_preparation(args)
    elif mode == "evaluate_results":
        if eval_type == "action_sequence":
            all_results = action_output_evaluation(args)
        elif eval_type == "transition_model":
            all_results = tm_output_evaluation(args)
        elif eval_type == "goal_interpretation":
            all_results = goal_output_evaluation(args)
        elif eval_type == "subgoal_decomposition":
            summary, _ = subgoal_output_evaluation(args)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # save summary results and intermediate results
    if mode == "evaluate_results":
        print(f"All results saved to {output_dir}")
    elif mode == "generate_prompts":
        print(f"Prompts generated and saved to {output_dir}")
