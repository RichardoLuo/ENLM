{
    "vocabulary": {
        "type": "extended",
        "extend": false,
        "max_vocab_size": {"tokens": 33278},
    },
    "datasets_for_vocab_creation": ["train"],
    "dataset_reader": {
        "type": "enhanced-wikitext-entity-nlm",
    },
    "train_data_path": "enlm/tests/fixture/train.jsonl",
    "validation_data_path": "enlm/tests/fixture/valid.jsonl",
    "test_data_path":"enlm/tests/fixture/test.jsonl",
    "iterator": {
        "type": "split",
        "batch_size": 16,
        "sorting_keys": [
             ["tokens",
              "num_tokens"]
        ],
        "splitter": {
            "type": "random",
            "mean_split_size": 30,
            "min_split_size": 20,
            "max_split_size": 40,
            "splitting_keys": [
                "tokens",
                "entity_types",
                "entity_ids",
                "mention_lengths"
            ],
        },
    },
    "model": {
        "type": "entitynlm",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 256,
                    "trainable": true
                },
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": 256,
            "hidden_size": 256,
            "dropout": 0.5,
            "stateful": true
        },
        "embedding_dim": 256,
        "max_mention_length": 180,
        "max_embeddings": 1000,
        "tie_weights": true,
        "dropout_rate": 0.4,
        "variational_dropout_rate": 0.1
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": -1,
        "optimizer": {
            "type": "adam",
            "lr": 1e-3
        }
    }
}