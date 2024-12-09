import wandb
import random
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('testing wandb', add_help=False)

    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_id', default='', type=str,
                    help='id of the current run')
    return parser
    

def main(args):
    print("here1")
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        id=args.wandb_id,
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
        },
        
    )

    # simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # log metrics to wandb
        wandb.log({"acc": acc, "loss": loss})

    print("here2")
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

    print("here3")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)