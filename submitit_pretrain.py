# Copyright (c) OTiS.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae?tab=readme-ov-file
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import main_pretrain as trainer
import submitit


def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for OTiS pretrain", parents=[trainer_parser])
    parser.add_argument("--mem_per_task", default=96, type=int, help="RAM to request for each task (i.e. GPU)")
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=7200, type=int, help="Duration of the job in minutes (default: 7200min = 5days)")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    cwd = os.getcwd()
    if Path(f"{cwd}/slurm_output/").is_dir():
        p = Path(f"{cwd}/slurm_output/{user}/")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main_pretrain as trainer

        self._setup_gpu_args()
        trainer.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    kwargs = {}
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=int(args.mem_per_task * num_gpus_per_node),
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.num_workers,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="otis")

    args.dist_url = get_init_file().as_uri()
    
    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()