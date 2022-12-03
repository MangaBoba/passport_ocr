common_config = {
    "data_dir": "/home/mangaboba/environ/passport_ocr/data",
    "img_width": 100,
    "img_height": 32,
    "map_to_seq_hidden": 64,
    "rnn_hidden": 256,
    "leaky_relu": False,
}

train_config = {
    "epochs": 100,
    "train_batch_size": 256,
    "eval_batch_size": 256,
    "lr": 0.0005,
    "cpu_workers": 4,
    "valid_max_iter": 100,
    "beam_size": 10,
    "checkpoints_dir": "checkpoints/",
}
train_config.update(common_config)
