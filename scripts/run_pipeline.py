import subprocess
import os
import argparse
import time
import yaml
from types import SimpleNamespace
from simple_slurm import Slurm

MAX_RETRIES = 5

default_args = {
    "num_procs": 1,
    "all": False,
    "test": False,
    "clean": False,
    "subsample": False,
    "pretrain": False,
    "ontology": False,
    "preprocess": False,
    "code_stats": False,
    "label": [],
    "split": False,
    "train_adapter": False,
    "eval": False,
    "slurm_config": None,
    "demo_mode": False,
    "reduced_size": None,
    "ws_name": None,
    "metadata_path": None,
    "sample_size": None,
    "prepare_pretrain": False,
}


def run_subprocess_and_log(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())

    rc = process.poll()
    return rc


def get_slurm(config_file, job_name, previous_job_id, slurm_args, experiment_name):
    print(f"Loading slurm config from {config_file}")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Merge config with slurm_args, slurm_args will overwrite config
    combined_config = {
        **config,
        **slurm_args,
        "output": f"$HOME/logs/{experiment_name}/%x_%A_%a.out",
        "error": f"$HOME/logs/{experiment_name}/%x_%A_%a.err",
    }

    job_name = job_name.replace(":", "_")

    if previous_job_id is not None:
        if isinstance(previous_job_id, list):
            dependency = {"afterok": ",".join(map(str, previous_job_id))}
        else:
            dependency = dict(afterok=previous_job_id)
        slurm = Slurm(
            **combined_config,
            job_name=f"{job_name}_{experiment_name}",
            dependency=dependency,
        )
    else:
        slurm = Slurm(**combined_config, job_name=f"{job_name}_{experiment_name}")
    return slurm


def execute(
    cmd,
    slurm_config,
    job_name,
    previous_job_id,
    slurm_args={},
    additional_cmds={},
    experiment_name="undefined_experiment",
):
    print(job_name)
    if slurm_config is None:
        retries = 0
        while retries < MAX_RETRIES:
            start_time = time.time()
            result = run_subprocess_and_log(cmd)
            end_time = time.time()
            execution_time = end_time - start_time

            hours, rem = divmod(execution_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Return code {result}")
            print(
                f"Finished execution in {int(hours)}h {int(minutes)}m {seconds:.2f}s\n\n"
            )

            if result == 0:
                return None
            elif result == -11:
                retries += 1
                print(f"Retrying... Attempt {retries} of {MAX_RETRIES}")
            else:
                exit(result)
    else:
        slurm = get_slurm(
            slurm_config, job_name, previous_job_id, slurm_args, experiment_name
        )
        # Make SLURM task fail if python script fails
        slurm.add_cmd("set -e")
        slurm.add_cmd(r"export HF_HOME=$TMPDIR")
        slurm.add_cmd('echo "TMP DIR: $TMPDIR"')
        slurm.add_cmd('echo "HF_HOME: $HF_HOME"')
        slurm.add_cmd("module load devel/python/3.12.3_intel_2023.1.0")
        if job_name.startswith("pretrain"):
            slurm.add_cmd("module load devel/cuda/12.4")

        slurm.add_cmd("source ~/thesis_env/bin/activate")
        if "before" in additional_cmds:
            for additional_cmd in additional_cmds["before"]:
                slurm.add_cmd(additional_cmd)
        slurm.add_cmd(" ".join(cmd))
        if "after" in additional_cmds:
            for additional_cmd in additional_cmds["after"]:
                slurm.add_cmd(additional_cmd)
        slurm.add_cmd("deactivate")
        print(slurm)
        return slurm.sbatch()


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        for key, value in default_args.items():
            if key not in config:
                config[key] = value
        config["config_path"] = config_path
        return SimpleNamespace(**config)


times = {
    "demo": {
        "test": "00:02:00",
        "clean": "00:09:00",
        "subsample": "00:05:00",
        "ontology": "00:05:00",
        "pretrain": "00:15:00",
        "prepare_pretrain": "00:05:00",
        "preprocess": "00:05:00",
        "label": "00:05:00",
        "split": "00:04:00",
        "featurize": "00:10:00",
        "train_adapter": "00:03:00",
        "evaluate": "00:03:00",
    },
    "prod": {
        "test": "00:01:00",
        "clean": "04:00:00",
        "subsample": "02:30:00",
        "ontology": "12:00:00",
        "pretrain": "48:00:00",
        "prepare_pretrain": "12:00:00",
        "preprocess": "02:00:00",
        "label": "02:00:00",
        "split": "00:15:00",
        "featurize": "20:00:00",
        "train_adapter": "06:30:00",
        "evaluate": "00:30:00",
    },
}


def get_args(demo_mode, task):
    if demo_mode:
        args = {"time": times["demo"][task]}
        if task == "featurize":
            args["partition"] = "dev_gpu_4_a100"
            args["gres"] = "gpu:1"
        if task == "pretrain":
            args["partition"] = "dev_gpu_4_a100"
            args["gres"] = "gpu:1"
        return args
    else:
        args = {"time": times["prod"][task]}
        if task == "featurize":
            args["partition"] = "gpu_4_a100"
            args["gres"] = "gpu:1"
        if task == "pretrain":
            args["partition"] = "gpu_4_a100"
            args["gres"] = "gpu:2"
        return args


def extract_stages(args):
    stages = []
    if args.all:
        return [
            "clean",
            "subsample",
            "ontology",
            "preptrain",
            "preprocess",
            "code_stats",
            "label",
            "split",
            "featurize",
            "train_adapter",
            "evaluate",
        ]
    if args.test:
        stages.append("test")
    if args.clean:
        stages.append("clean")
    if args.subsample:
        stages.append("subsample")
    if args.ontology:
        stages.append("ontology")
    if args.prepare_pretrain:
        stages.append("prepare_pretrain")
    if args.pretrain:
        stages.append("pretrain")
    if len(args.label) > 0:
        stages.append("label")
    if args.code_stats:
        stages.append("code_stats")
    if args.split:
        stages.append("split")
    if args.preprocess:
        stages.append("preprocess")
    if args.featurize:
        stages.append("featurize")
    if args.train_adapter:
        stages.append("train_adapter")
    if args.eval:
        stages.append("evaluate")
    return stages


def get_path(path, slurm):
    if slurm is not None:
        return os.path.join("$TMPDIR", path)
    else:
        return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pipeline")
    parser.add_argument(
        "--config-path",
        type=str,
        required=False,
        help="Instead of providing args to the parser you can read all arguments using a config file",
    )
    parser.add_argument(
        "--meds-path",
        type=str,
        help="Path to the meds data source",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--vocab-path",
        help="Path to the vocabulary",
        type=str,
    )
    parser.add_argument("--num-procs", type=int, default=default_args["num_procs"])
    parser.add_argument(
        "--test",
        action="store_true",
        help="Prints test to check whether the setup works",
    )
    parser.add_argument("--clean", action="store_true", help="Clean the data")
    parser.add_argument("--code-stats", action="store_true")
    parser.add_argument(
        "--ontology", action="store_true", help="Process vocabulary into ontology"
    )
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--featurize", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--subsample", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train-adapter", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--label", nargs="*", type=str, default=[])
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--sample-size",
        type=int,
        required=False,
        help="Sample size if subsampling is used",
    )
    parser.add_argument(
        "--slurm_config",
        type=str,
        help="If path to SLURM conf YAML is given, this script automatically creates SLURM jobs.",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Demo mode uses less SLURM resources for tasks to be selected faster.",
    )
    args = parser.parse_args()

    if args.config_path:
        args = read_config(args.config_path)
        experiment_name = os.path.splitext(os.path.basename(args.config_path))[0]
    else:
        experiment_name = "undefined_experiment"

    previous_job_id = None

    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)

    execute_tasks = extract_stages(args)
    slurm = args.slurm_config
    if slurm is not None:
        workspace_path = (
            subprocess.check_output("ws_find clmbr", shell=True).decode("utf-8").strip()
        )
        print(f"Workspace path: {workspace_path}")
    else:
        workspace_path = None

    print(execute_tasks)

    if "test" in execute_tasks:
        cmd = ["python", os.path.join(parent_directory, "test_script.py")]
        previous_job_id = execute(
            cmd,
            slurm,
            "test",
            previous_job_id,
            {**get_args(args.demo_mode, "test")},
            {},
            experiment_name,
        )

    if "clean" in execute_tasks:
        cmd = [
            "python",
            os.path.join(parent_directory, "clean.py"),
            "--input-path",
            get_path(args.meds_path, slurm),
            "--output-path",
            get_path(args.output_path, slurm),
            "--num-procs",
            str(args.num_procs),
        ]
        if args.reduced_size is not None:
            cmd.extend(["--reduced-size", str(args.reduced_size)])
        previous_job_id = execute(
            cmd,
            slurm,
            "clean",
            previous_job_id,
            {**get_args(args.demo_mode, "clean")},
            {
                "before": [
                    f"tar -C $TMPDIR/ -xvzf $(ws_find {args.ws_name})/{args.meds_path}.tgz"
                ],
                "after": [
                    f"tar -cvzf $TMPDIR/{args.output_path}/clean.tgz -C $TMPDIR/{args.output_path}/clean/ .",
                    f"rm -r $TMPDIR/{args.output_path}/clean",
                    f"rsync -av $TMPDIR/{args.output_path}/clean.tgz $(ws_find {args.ws_name})/{args.output_path}/",
                ],
            },
            experiment_name,
        )

    if "subsample" in execute_tasks:
        cmd = [
            "python",
            os.path.join(parent_directory, "cohort_selection.py"),
            "--input-path",
            get_path(os.path.join(args.output_path, "clean"), slurm),
            "--output-path",
            get_path(os.path.join(args.output_path, "cohort"), slurm),
            "--num-procs",
            str(args.num_procs),
            "--filter-function",
            args.filter_function,
        ]
        if args.sample_size is not None:
            cmd.extend(
                [
                    "--size",
                    str(args.sample_size),
                ]
            )
        previous_job_id = execute(
            cmd,
            slurm,
            "subsample",
            previous_job_id,
            {**get_args(args.demo_mode, "subsample")},
            {
                "before": [
                    f"mkdir -p $TMPDIR/{args.output_path}/clean",
                    f"tar -C $TMPDIR/{args.output_path}/clean -xvzf $(ws_find {args.ws_name})/{args.output_path}/clean.tgz",
                    "echo $(ls $TMPDIR)",
                ],
                "after": [
                    f"tar -cvzf $TMPDIR/{args.output_path}/cohort.tgz -C $TMPDIR/{args.output_path}/cohort/ .",
                    f"rm -r $TMPDIR/{args.output_path}/cohort",
                    f"rsync -av $TMPDIR/{args.output_path}/cohort.tgz $(ws_find {args.ws_name})/{args.output_path}/",
                ],
            },
            experiment_name,
        )

    if "split" in execute_tasks:
        # For slurm jobs the directory is not yet created but will be created
        if (
            os.path.exists(os.path.join(args.output_path, "cohort"))
            or "subsample" in execute_tasks
            or (
                workspace_path is not None
                and os.path.exists(
                    os.path.join(workspace_path, args.output_path, "cohort.tgz")
                )
            )
        ):
            input_path = os.path.join(args.output_path, "cohort")
        else:
            input_path = os.path.join(args.output_path, "clean")
        cmd = [
            "python",
            os.path.join(parent_directory, "split_data.py"),
            "--input-path",
            get_path(os.path.join(input_path), slurm),
            "--output-path",
            get_path(args.output_path, slurm),
            "--test-frac",
            str(args.test_frac),
            "--val-frac",
            str(args.val_frac),
        ]
        previous_job_id = execute(
            cmd,
            slurm,
            "split",
            previous_job_id,
            {**get_args(args.demo_mode, "split")},
            {
                "before": [
                    f"mkdir -p $TMPDIR/{input_path}",
                    f"tar -C $TMPDIR/{input_path}/ -xvzf $(ws_find {args.ws_name})/{input_path}.tgz",
                ],
                "after": [
                    f"rsync -avh $TMPDIR/{args.output_path}/splits/ $(ws_find {args.ws_name})/{args.output_path}/splits/"
                ],
            },
            experiment_name,
        )

    if "prepare_pretrain" in execute_tasks:
        if (
            os.path.exists(os.path.join(args.output_path, "cohort"))
            or "subsample" in execute_tasks
            or os.path.exists(
                os.path.join(workspace_path, args.output_path, "cohort.tgz")
            )
        ):
            input_path = os.path.join(args.output_path, "cohort")
        else:
            input_path = os.path.join(args.output_path, "clean")
        cmd = [
            "python",
            os.path.join(parent_directory, "prepare_pretraining.py"),
            "--data-dir",
            get_path(input_path, slurm),
            "--output-path",
            get_path(args.output_path, slurm),
            "--vocab-size",
            str(args.transformer["vocab_size"]),
            "--num-procs",
            str(args.num_procs),
            "--model-name",
            args.transformer["model"],
        ]
        after_prepare_pretrain = [
            f"rsync -avh $TMPDIR/{args.output_path}/fm/ $(ws_find {args.ws_name})/{args.output_path}/fm/",
            f"tar -cvzf $TMPDIR/{args.output_path}/train_batches.tgz -C $TMPDIR/{args.output_path}/train_batches/ .",
            f"rm -r $TMPDIR/{args.output_path}/train_batches",
            f"rsync -av $TMPDIR/{args.output_path}/train_batches.tgz $(ws_find {args.ws_name})/{args.output_path}/",
        ]
        if args.transformer["model"] == "CLMBR-T-lab":
            after_prepare_pretrain.extend(
                [
                    f"tar -cvzf $TMPDIR/{args.output_path}/lab_tokenization_data.tgz -C $TMPDIR/{args.output_path}/lab_tokenization_data/ .",
                    f"rm -r $TMPDIR/{args.output_path}/lab_tokenization_data",
                    f"rsync -av $TMPDIR/{args.output_path}/lab_tokenization_data.tgz $(ws_find {args.ws_name})/{args.output_path}/",
                ]
            )
        previous_job_id = execute(
            cmd,
            slurm,
            "prepare_pretrain",
            previous_job_id,
            {**get_args(args.demo_mode, "prepare_pretrain")},
            {
                "before": [
                    f"mkdir -p $TMPDIR/{input_path}",
                    f"tar -C $TMPDIR/{input_path}/ -xvzf $(ws_find {args.ws_name})/{input_path}.tgz",
                    f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/splits/ $TMPDIR/{args.output_path}/splits/",
                ],
                "after": after_prepare_pretrain,
            },
            experiment_name,
        )

    if "pretrain" in execute_tasks:
        pretrain_job_ids = []
        for learning_rate in args.transformer["learning_rate"]:
            cmd = [
                "python",
                os.path.join(parent_directory, "pretrain.py"),
                "--data-dir",
                get_path(os.path.join(args.output_path, "train_batches"), slurm),
                "--output-path",
                get_path(args.output_path, slurm),
                "--n-layers",
                str(args.transformer["n_layer"]),
                "--hidden-size",
                str(args.transformer["hidden_size"]),
                "--intermediate-size",
                str(args.transformer["intermediate_size"]),
                "--n-heads",
                str(args.transformer["n_head"]),
                "--vocab-size",
                str(args.transformer["vocab_size"]),
                "--num-procs",
                str(args.num_procs),
                "--learning-rate",
                str(learning_rate),
                "--model-name",
                args.transformer["model"],
            ]
            pretrain_job_id = execute(
                cmd,
                slurm,
                "pretrain_" + str(learning_rate),
                previous_job_id,
                {**get_args(args.demo_mode, "pretrain")},
                {
                    "before": [
                        f"mkdir -p $TMPDIR/{args.output_path}/fm",
                        f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/fm/ $TMPDIR/{args.output_path}/fm/",
                        f"mkdir -p $TMPDIR/{args.output_path}/train_batches",
                        f"tar -C $TMPDIR/{args.output_path}/train_batches/ -xvzf $(ws_find {args.ws_name})/{args.output_path}/train_batches.tgz",
                    ],
                    "after": [
                        f"rsync -avh $TMPDIR/{args.output_path}/fm/ $(ws_find {args.ws_name})/{args.output_path}/fm/",
                    ],
                },
                experiment_name,
            )
            pretrain_job_ids.append(pretrain_job_id)
        previous_job_id = pretrain_job_ids

    if "ontology" in execute_tasks:
        if os.path.exists(os.path.join(args.output_path, "cohort")) or (
            workspace_path is not None
            and os.path.exists(
                os.path.join(workspace_path, args.output_path, "cohort.tgz")
            )
        ):
            data_path = os.path.join(args.output_path, "cohort")
        else:
            data_path = os.path.join(args.output_path, "clean")
        if args.metadata_path is not None:
            metadata_path = get_path(args.metadata_path, slurm)
        else:
            metadata_path = get_path(
                os.path.join(args.meds_path, "metadata.json"), slurm
            )
        cmd = [
            "python",
            os.path.join(parent_directory, "load_ontology.py"),
            "--metadata-path",
            metadata_path,
            "--input-path",
            get_path(data_path, slurm),
            "--output-path",
            get_path(args.output_path, slurm),
            "--vocab-path",
            get_path(args.vocab_path, slurm),
            "--num-procs",
            str(args.num_procs),
        ]
        previous_job_id = execute(
            cmd,
            slurm,
            "ontology",
            previous_job_id,
            {**get_args(args.demo_mode, "ontology")},
            {
                "before": [
                    f"mkdir -p $TMPDIR/{data_path}",
                    f"tar -C $TMPDIR/{data_path} -xvzf $(ws_find {args.ws_name})/{data_path}.tgz",
                    f"tar -C $TMPDIR/ -xvzf $(ws_find {args.ws_name})/{args.meds_path}.tgz",
                    f"tar -C $TMPDIR/ -xvzf $(ws_find {args.ws_name})/{args.vocab_path}.tgz",
                    f"rsync -avh $(ws_find clmbr)/mapping-metadata $TMPDIR",
                ],
                "after": [
                    f"tar -cvzf $TMPDIR/{args.output_path}/ontology.pkl.tgz -C $TMPDIR/{args.output_path}/ ontology.pkl",
                    f"rm -r $TMPDIR/{args.output_path}/ontology.pkl",
                    f"rsync -av $TMPDIR/{args.output_path}/ontology.pkl.tgz $(ws_find {args.ws_name})/{args.output_path}/",
                ],
            },
            experiment_name,
        )

    if "preprocess" in execute_tasks:
        # For slurm jobs the directory is not yet created but will be created
        if args.transformer["model"] == "CLMBR-T-lab":
            input_path = os.path.join(args.output_path, "lab_tokenization_data")
        elif (
            os.path.exists(os.path.join(args.output_path, "cohort"))
            or "subsample" in execute_tasks
            or (
                workspace_path is not None
                and os.path.exists(
                    os.path.join(workspace_path, args.output_path, "cohort.tgz")
                )
            )
        ):
            input_path = os.path.join(args.output_path, "cohort")
        else:
            input_path = os.path.join(args.output_path, "clean")
        cmd = [
            "python",
            os.path.join(parent_directory, "preprocess.py"),
            "--input-path",
            get_path(input_path, slurm),
            "--output-path",
            get_path(args.output_path, slurm),
            "--num-procs",
            str(args.num_procs),
            "--model-name",
            args.transformer["model"],
        ]
        before_preprocess = [
            f"mkdir -p $TMPDIR/{input_path}",
            f"tar -C $TMPDIR/{input_path}/ -xvzf $(ws_find {args.ws_name})/{input_path}.tgz",
            f"tar -C $TMPDIR/{args.output_path}/ -xvzf $(ws_find {args.ws_name})/{args.output_path}/ontology.pkl.tgz",
        ]
        # More than model name in the transformer config
        if len(args.transformer) > 1:
            before_preprocess.extend(
                [
                    f"mkdir -p $TMPDIR/{args.output_path}/fm",
                    f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/fm/ $TMPDIR/{args.output_path}/fm/",
                ]
            )
        previous_job_id = execute(
            cmd,
            slurm,
            "preprocess",
            previous_job_id,
            {**get_args(args.demo_mode, "preprocess")},
            {
                "before": before_preprocess,
                "after": [
                    f"tar -cvzf $TMPDIR/{args.output_path}/preprocessed.tgz -C $TMPDIR/{args.output_path}/preprocessed/ .",
                    f"rm -r $TMPDIR/{args.output_path}/preprocessed",
                    f"rsync -av $TMPDIR/{args.output_path}/preprocessed.tgz $(ws_find {args.ws_name})/{args.output_path}/",
                ],
            },
            experiment_name,
        )

    if "code_stats" in execute_tasks:
        cmd = [
            "python",
            os.path.join(parent_directory, "create_code_stats.py"),
            "--output-path",
            args.output_path,
            "--num-procs",
            str(args.num_procs),
        ]
        execute(cmd, slurm, experiment_name=experiment_name)

    if "label" in execute_tasks:
        label_job_ids = []
        if args.all:
            # TODO add disease_labelers
            args.label = ["mortality", "long_los"]
        for labeler in args.label:
            labeler_parts = labeler.split(":")

            cmd = [
                "python",
                os.path.join(parent_directory, "label.py"),
                "--input-path",
                get_path(os.path.join(args.output_path, "preprocessed"), slurm),
                "--output-path",
                get_path(os.path.join(args.output_path), slurm),
                "--labeler",
                labeler_parts[0],
            ]
            if "random_admission" in labeler_parts:
                cmd.append("--random-admission-selection")
            if "biased_admission_types" in labeler_parts:
                cmd.append("--biased-admission-selection")
            label_job_id = execute(
                cmd,
                slurm,
                "label_" + labeler,
                previous_job_id,
                {**get_args(args.demo_mode, "label")},
                {
                    "before": [
                        f"mkdir -p $TMPDIR/{args.output_path}/preprocessed",
                        f"tar -C $TMPDIR/{args.output_path}/preprocessed -xvzf $(ws_find {args.ws_name})/{args.output_path}/preprocessed.tgz",
                    ],
                    "after": [
                        f"rsync -avh $TMPDIR/{args.output_path}/labels/ $(ws_find {args.ws_name})/{args.output_path}/labels/"
                    ],
                },
                experiment_name,
            )
            label_job_ids.append(label_job_id)

        cmd = [
            "python",
            os.path.join(parent_directory, "create_label_stats.py"),
            "--output-path",
            get_path(args.output_path, slurm),
        ]
        previous_job_id = execute(
            cmd,
            slurm,
            "label_stats",
            label_job_ids,
            {"time": "00:05:00"},
            {
                "before": [
                    f"mkdir -p $TMPDIR/{args.output_path}/labels",
                    f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/labels/ $TMPDIR/{args.output_path}/labels/",
                ],
                "after": [
                    f"rsync -avh $TMPDIR/{args.output_path}/labels/ $(ws_find {args.ws_name})/{args.output_path}/labels/"
                ],
            },
            experiment_name,
        )

    if "featurize" in execute_tasks:
        cmd = [
            "python",
            os.path.join(parent_directory, "featurize.py"),
            "--output-path",
            get_path(args.output_path, slurm),
            "--num-procs",
            str(args.num_procs),
            "--model-name",
            args.transformer["model"],
        ]
        before_featurize = [
            f"mkdir -p $TMPDIR/{args.output_path}/preprocessed",
            f"mkdir $TMPDIR/{args.output_path}/labels",
            f"tar -C $TMPDIR/{args.output_path}/preprocessed -xvzf $(ws_find {args.ws_name})/{args.output_path}/preprocessed.tgz",
            f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/labels/ $TMPDIR/{args.output_path}/labels/",
        ]
        if len(args.transformer) > 1:
            before_featurize.extend(
                [
                    f"mkdir -p $TMPDIR/{args.output_path}/fm",
                    f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/fm/ $TMPDIR/{args.output_path}/fm/",
                ]
            )
        previous_job_id = execute(
            cmd,
            slurm,
            "featurize",
            previous_job_id,
            {**get_args(args.demo_mode, "featurize")},
            {
                "before": before_featurize,
                "after": [
                    f"rsync -avh $TMPDIR/{args.output_path}/fm/ $(ws_find {args.ws_name})/{args.output_path}/fm/",
                    f"rsync -avh $TMPDIR/{args.output_path}/features/ $(ws_find {args.ws_name})/{args.output_path}/features/",
                ],
            },
            experiment_name,
        )

    if "train_adapter" in execute_tasks:
        cmd = [
            "python",
            os.path.join(parent_directory, "train_adapter.py"),
            "--output-path",
            get_path(args.output_path, slurm),
        ]
        previous_job_id = execute(
            cmd,
            slurm,
            "train_adapter",
            previous_job_id,
            {**get_args(args.demo_mode, "train_adapter")},
            {
                "before": [
                    f"mkdir -p $TMPDIR/{args.output_path}/features $TMPDIR/{args.output_path}/splits",
                    f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/features/ $TMPDIR/{args.output_path}/features/",
                    f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/splits/ $TMPDIR/{args.output_path}/splits/",
                ],
                "after": [
                    f"rsync -avh $TMPDIR/{args.output_path}/reg_model/ $(ws_find {args.ws_name})/{args.output_path}/reg_model/",
                    f"rsync -avh $TMPDIR/{args.output_path}/xgb_model/ $(ws_find {args.ws_name})/{args.output_path}/xgb_model/",
                ],
            },
            experiment_name,
        )

    if "evaluate" in execute_tasks:
        cmd = [
            "python",
            os.path.join(parent_directory, "evaluate.py"),
            "--output-path",
            get_path(args.output_path, slurm),
        ]
        execute(
            cmd,
            slurm,
            "evaluate",
            previous_job_id,
            {**get_args(args.demo_mode, "evaluate")},
            {
                "before": [
                    f"mkdir -p $TMPDIR/{args.output_path}/features $TMPDIR/{args.output_path}/splits $TMPDIR/{args.output_path}/reg_model $TMPDIR/{args.output_path}/xgb_model",
                    f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/reg_model/ $TMPDIR/{args.output_path}/reg_model/",
                    f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/xgb_model/ $TMPDIR/{args.output_path}/xgb_model/",
                    f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/features/ $TMPDIR/{args.output_path}/features/",
                    f"rsync -avh $(ws_find {args.ws_name})/{args.output_path}/splits/ $TMPDIR/{args.output_path}/splits/",
                ],
                "after": [
                    f"rsync -avh $TMPDIR/{args.output_path}/results/ $(ws_find {args.ws_name})/{args.output_path}/results/",
                ],
            },
            experiment_name,
        )
