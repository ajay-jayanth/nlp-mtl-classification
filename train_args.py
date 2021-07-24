from transformers import TrainingArguments

args = TrainingArguments(
    'Training Sequence',
    evaluation_strategy = "epoch",
    output_size=(7),
    per_device_train_batch_size=10,
    per_device_eval_batch_size=100,
    num_train_epochs=50,
    max_learning_rate=1e-2
)