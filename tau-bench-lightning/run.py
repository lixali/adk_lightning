# Copyright Sierra

import argparse
from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for the agent",
    )
    # Add our custom providers to the litellm provider list
    custom_providers = ["gemma", "google"]  # Our custom providers
    all_providers = list(provider_list) + custom_providers
    
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=all_providers,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=all_providers,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--user-temperature",
        type=float,
        default=1.0,
        help="The sampling temperature for the user model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--task-id", type=int, help="(Optional) run only a single specific task ID (convenience for --task-ids)")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, help="Path to a jsonlines file containing few shot displays")
    parser.add_argument("--custom-suffix", type=str, help="Custom suffix to add to output filenames (e.g., '_version_1')")
    args = parser.parse_args()
    
    # Handle --task-id flag by converting to --task-ids
    if args.task_id is not None:
        if args.task_ids is not None:
            raise ValueError("Cannot specify both --task-id and --task-ids. Use one or the other.")
        args.task_ids = [args.task_id]
    
    print(args)
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        user_temperature=args.user_temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        custom_suffix=args.custom_suffix,
    )


def main():
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
