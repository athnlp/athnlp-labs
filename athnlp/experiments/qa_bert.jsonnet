{
    "dataset_reader": {
        "lazy": false,
        "type": "bert_squad",
        "tokenizer": {
            "word_splitter": {
                "type": "bert-basic-wordpiece",
                "pretrained_model": "resources/bert-base-uncased/vocab.txt"
            }
        },
        "token_indexers": {
            "bert": {
                "type": "bert-squad-indexer",
                "pretrained_model": "resources/bert-base-uncased/vocab.txt"
            }
        },
        "version_2": true,
        "max_sequence_length": 384,
        "question_length_limit": 64,
        "doc_stride": 128
    },
    "train_data_path": "data/squad/train.json",
    "validation_data_path": "data/squad/test.json",
    "model": {
        "type": "qa_bert",
        "bert_model": "resources/bert-base-uncased/",
        "dropout": 0.1
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 2
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0001
        },
        "validation_metric": "+f1",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 3,
        "grad_norm": 1.0,
        "patience": 5,
        "cuda_device": -1
    }
}