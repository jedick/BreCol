import argparse
from typing_extensions import override
from .._cli import CliSubCommand, subcommand

__doc__ = "Additional CLI tools for Weights & Biases."

@subcommand("generate_id", "Generate a random run ID.")
class GenerateId(CliSubCommand):
    @override
    def run(self, _: argparse.Namespace):
        import wandb
        print(wandb.util.runid.generate_id()) # type: ignore
