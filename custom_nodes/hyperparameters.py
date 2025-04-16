from node_registry import register_node

@register_node(name="Hyperparameters", category="Config", tags={"run_early": True})
class HyperparamsNode:
    outputs = [
        {"name": "hyperparams", "type": "dict"}
    ]
    widgets = [
        {
            "type": "combo",
            "name": "optimizer",
            "value": "adam",
            "options": {"values": ["adam", "sgd", "rmsprop"]}
        },
        {
            "type": "combo",
            "name": "loss",
            "value": "cross_entropy",
            "options": {
                "values": [
                    "auto",
                    "cross_entropy",
                    "mse",
                    "bce",
                    "bce_with_logits",
                    "l1",
                    "smooth_l1",
                    "nll",
                    "hinge_embedding",
                    "cosine_embedding",
                    "ctc",
                    "huber"
                ]
            }
        },
        {
            "type": "number",
            "name": "learning_rate",
            "value": 0.001,
            "options": {"min": 0.00001, "max": 1.0, "step": 0.0001}
        },
        {
            "type": "number",
            "name": "epochs",
            "value": 5,
            "options": {"min": 1, "max": 100}
        },
        {
            "type": "number",
            "name": "early_stopping_patience",
            "value": 5,
            "options": {"min": 1, "max": 50}
        },
        {
            "type": "combo",
            "name": "use_early_stopping",
            "value": "off",
            "options": {"values": ["on", "off"]}
        },
        {
            "type": "combo",
            "name": "task_type",
            "value": "auto",
            "options": {"values": ["auto", "classification", "regression"]}
        }
    ]
    size = [220, 260]

    def __init__(self,
                 optimizer="adam",
                 loss="cross_entropy",
                 learning_rate=0.001,
                 epochs=5,
                 early_stopping_patience=5,
                 use_early_stopping="off",
                 task_type="auto"):
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_early_stopping = use_early_stopping
        self.task_type = task_type

    def build(self):
        return {
            "optimizer": self.optimizer,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "use_early_stopping": self.use_early_stopping,
            "task_type": self.task_type
        }
