## Welcome to the experiments runner!

### What is this?
An experiment runner that lets you quickly start evaluating blast on a variety of tasks. The runner reads a config file that specifies the tasks to run, the stages to run them in, and the settings for the experiment. See the [section](#get-started) below on how to get started!

### Get started
Make a copy of `config/demo-experiment-config.yaml` or `config/testing-experiment-config.yaml`. Then, replace the **tasks** section with actual task(s) you wish to run. 

Each task has the following fields:
```
  - id: # The unique identifier for the task
    initial_url: # Optional initial url
    goal: # The actual task definition
```
For convenience, all tasks from [agisdk-REAL](https://github.com/agi-inc/agisdk/tree/main/src/agisdk/REAL/browsergym/webclones/tasks) have already been imported and can be found in `tasks/agisdk/agisdk.yaml`.

Next, edit the **stages** section to configure the experiment stages. For example, to evaluate results under `baseline` and `data-parallelism`, you can set it as follows:
```
  - name: "baseline"
    description: "Baseline with all parallelism disabled"
    config:
      allow_parallelism:
        task: false
        data: false
        first_of_n: false
      max_parallelism_nesting_depth: 1
      llm_model: "openai:gpt-4.1"
      llm_model_mini: "openai:gpt-4.1-mini"
  
  - name: "data-parallelism"
    description: "Data parallelism enabled"
    config:
      allow_parallelism:
        task: false
        data: true
        first_of_n: false
      max_parallelism_nesting_depth: 1
      llm_model: "openai:gpt-4.1"
      llm_model_mini: "openai:gpt-4.1-mini"
```

Finally, edit the **settings** section to configure the experiment settings. `runs_per_stage` specifies the number of runs to repeat per stage. `logs_dir` and `output_dir` specify the directories to save the logs and results. For example, to run 50 runs per stage:
```
  runs_per_stage: 50
  logs_dir: "logs"
  output_dir: "results"
```

That's it! You can now run the experiment by running `python runner.py --config <path-to-your-config-file>`. 

When you run one of the tasks from the [agisdk-REAL](https://github.com/agi-inc/agisdk/tree/main/src/agisdk/REAL/browsergym/webclones/tasks), you can add the `--evaluate` flag to evaluate the results: `python runner.py --config <path-to-your-config-file> --evaluate`

### Import tasks
All tasks from [agisdk-REAL](https://github.com/agi-inc/agisdk/tree/main/src/agisdk/REAL/browsergym/webclones/tasks) have already been imported and can be found in `tasks/agisdk/agisdk.yaml`. To re-import them, clone the `agisdk` repo and run `python tasks/agisdk/import_real.py`.

### Evalute your own tasks
You can also set up evals for your tasks. Clone the `agisdk` repo, and make sure to install it in editable mode. You can then create a new task json in `src/agisdk/REAL/browsergym/webclones/tasks`. Your task json just needs to follow the same format as the other tasks.