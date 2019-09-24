{
  "dataset_reader": {
    "type": "multi30k",
    "language_pairs": {
      "source": "en",
      "target": "fr"
    },
    "source_token_indexers": {
      "source_tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },
    "target_token_indexers": {
      "target_tokens": {
        "type": "single_id",
        "namespace": "target_tokens"
      }
    }
  },
  "train_data_path": "data/multi30k/val.lc.norm.tok.head-750",
  "validation_data_path": "data/multi30k/val.lc.norm.tok.head-250",
  "model": {
    "type": "nmt_seq2seq",
    "source_embedder": {
      "token_embedders": {
        "source_tokens": {
            "type": "embedding",
            "embedding_dim": 50,
            "trainable": true,
            "vocab_namespace": "source_tokens"
        }
      }
    },
    "target_namespace": "target_tokens",
//    "attention" : {
//      "type" : "dot_product"
//    },
    "encoder": {
      "type": "lstm",
      "input_size": 50,
      "hidden_size": 200,
      "num_layers": 1,
      "dropout": 0.3,
      "bidirectional": true
    },
    "decoder": {
      "type": "lstm",
      "input_size": 50,
      "hidden_size": 400
    },
    "max_decoding_steps": 15,
    "beam_size": 1
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [
      [
        "source_tokens",
        "num_tokens"
      ]
    ],
    "batch_size": 1
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 100,
    "patience": 10,
    "validation_metric": "-loss",
    "cuda_device": -1
  }
}
