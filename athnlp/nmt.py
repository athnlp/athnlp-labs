# pylint: disable=no-self-use,invalid-name
from argparse import ArgumentParser
import json
import shutil
import sys

from allennlp.commands import main

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument('-c', "--config_file", type=str, default='athnlp/experiments/nmt_multi30k.jsonnet')
    argparse.add_argument('-m', "--model_path", default="/tmp/debugger_train")
    argparse.add_argument('-i', "--input_file", default="data/multi30k/val.lc.norm.tok.head-5.en.jsonl")
    argparse.add_argument("--predict", action='store_true')

    args = argparse.parse_args()
    config_file = args.config_file
    serialization_dir = args.model_path

    if args.predict:
        overrides = json.dumps({"model": {"visualize_attention": "false"}})

        sys.argv = [
            "allennlp",  # command name, not used by main
            "predict",
            "--predictor", "seq2seq",
            "--include-package", "athnlp",
            "-o", overrides,
            serialization_dir,
            args.input_file,
        ]
    else:
        # Training will fail if the serialization directory already
        # has stuff in it. If you are running the same training loop
        # over and over again for debugging purposes, it will.
        # Hence we wipe it out in advance.
        # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
        shutil.rmtree(serialization_dir, ignore_errors=True)

        # Use overrides to train on CPU.
        overrides = json.dumps({"trainer": {"cuda_device": -1}})

        # Assemble the command into sys.argv
        sys.argv = [
            "allennlp",  # command name, not used by main
            "train",
            config_file,
            "-s", serialization_dir,
            "--include-package", "athnlp",
            "-o", overrides,
        ]

    main()
