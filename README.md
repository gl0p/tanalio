![image](https://github.com/user-attachments/assets/2495a418-2cef-4e50-a98b-0d6202dc0de8)

---

# 🧠 AI Builder - Visual Deep Learning Workflow UI

Welcome to **AI Builder**, a visual graph-based web application for building and running deep learning workflows — no boilerplate code needed. Inspired by ComfyUI, this app allows you to chain together custom nodes (like layers, preprocessors, trainers, and predictors) to build, train, and test deep learning models through an intuitive UI.

---

## 🚀 Features

- **Drag-and-drop Node Graph UI** – Create complex AI pipelines visually.
- **Live Socket Event System** – Trigger callbacks via UI elements like buttons and inputs.
- **Modular Node System** – Easily register and define new custom nodes.
- **CSV and Image Dataset Support** – Supports both CSV-based ML and COCO-style image tasks.
- **Dynamic Model Injection** – Input/output wiring and auto property syncing between nodes.
- **Training + Prediction Nodes** – Add `Trainer` and `Single Predict` nodes with real-time interactivity.
- **Autosave Graphs** – Automatically saves your work and allows you to reload at any time.
- **Custom Widget System** – Define widget-based inputs for any node (e.g., dropdowns, text inputs, buttons).


# 🧩 Creating Custom Nodes -  Define, Connect, and Build 

## ✅ Quick Overview

Each node is a self-contained class with:

- Inputs & Outputs (connectable ports)
- UI Widgets (editable settings)
- `build()` for nodes like data loaders, preprocessing, or evaluators
- `get_layer()` for Model nodes (because they're chained together)

---

## 1. @register_node Decorator

```python
@register_node(
    name="Dense Layer",
    category="Model",
    tags={"my_custom_tag": True}
)
class DenseLayer:
    ...
```

**Arguments:**
- `name` → how it appears in the UI
- `category` → places it in the right menu
- `tags` → tells the executor when to build this node (explained below)

---

## 2. Inputs & Outputs

```python
inputs = [{"name": "in", "type": "tensor"}]
outputs = [{"name": "out", "type": "tensor"}]
```

Used to wire connections. Types must match between output and input slots.

**Common types:**
- `"tensor"` → data tensors
- `"model"` / `"model_out"` → PyTorch models
- `"train"`, `"val"`, `"test"` → dataset loaders
- `"dict"` → generic config or params

---

## 3. Widgets (User Parameters)

```python
widgets = [
  {"type": "text", "name": "out_features", "value": "128"},
  {"type": "combo", "name": "activation", "value": "relu", "options": {"values": ["relu", "gelu"]}}
]
```

Widgets define user inputs in the UI.

---

## 4. Preprocessing & Config Nodes → use `build()`

If your node returns data, tensors, configs, etc., use:

```python
def build(self):
    return {
        "train": train_loader,
        "val": val_loader,
        "sample_tensor": example_tensor,
        "task_type": "classification"
    }
```

---

## 5. Model Nodes → use `get_layer()` (NOT `build()`)

Model layers like Dense, Conv2D, Dropout, etc., must inherit from `BaseModelNode`:

```python
from base_model_node import BaseModelNode

class DenseLayer(BaseModelNode):
    def get_layer(self):
        return nn.Linear(self.in_features, self.out_features)
```

**Why?**  Model nodes don’t build themselves directly. 

---

## 6. 🔁 `OutputLayer` walks backward and builds the model

The `OutputLayer` is the final model node. It’s responsible for:

- Walking backward through connected model nodes
- Calling `get_layer()` on each model node
- Assembling them into a `nn.Sequential(...)`

Internally:

```python
def build(self):
    self.model = nn.Sequential(
        previous_layer1.get_layer(),
        previous_layer2.get_layer(),
        ...,
        nn.Linear(..., out_features)
    )
```

This is why Model nodes do **not** use `build()` — they are **passively walked** and assembled by `OutputLayer`.

---

## 7. Node Build Timeline

| Phase              | Who Runs                                                 |
|--------------------|-----------------------------------------------------------|
| `run_early` nodes  | built first (data loaders, hyperparams, flatteners)       |
| `Model` nodes        | walked and assembled from `OutputLayer` via `get_layer()` |
| `TrainerNode`      | receives final model + dataloaders + hyperparams          |
| `run_after_train`  | run after training (evaluators, exporters, predictors)    |

---

## 8. Behavior Tags Summary

| Tag                  | Description                                                       |
|----------------------|-------------------------------------------------------------------|
| `run_early: True`    | build before model construction (data/config/preprocessing)       |
| `run_after_train: True` | build after training (evaluators, savers, predictors)          |
| `register_runtime: True` | track runtime nodes like Trainer                              |

---

## 9. 🔍 What Does `OutputLayer` Do?

It:
- Gets the `task_type` (e.g. regression/classification)
- Walks backward through all connected model nodes
- Collects layers via `get_layer()`
- Builds a full PyTorch `Sequential` model
- Sets `num_classes` or final activation as needed

---

## 10. Handling UI Widget Interactions → `on_widget_event`

Nodes can respond to user-triggered widget events (buttons, sliders, text) by defining:

```python
def on_widget_event(self, event_type, payload):
    ...
```

This is triggered automatically when a widget with a `"callback"` is used in the UI.

### Example: Button Widget with Callback

```python
widgets = [
  {"type": "button", "name": "Run Prediction", "value": "", "callback": "run_prediction"},
  {"type": "text", "name": "prediction", "value": "n/a"}
]
```

When the user clicks the button, this fires:

```python
on_widget_event("run_prediction", payload)
```

Inside `on_widget_event`, you can:
- Access the widget interaction `event_type`
- Use any values passed via `payload` (e.g. `input_features`, `clicked`)
- Trigger prediction, update state, or emit results

### Full Example

```python
def on_widget_event(self, event_type, payload):
    if event_type == "run_prediction":
        input_vals = payload.get("input_features", [])
        result = self.run_prediction(input_vals)
        socketio.emit("single_predict_result", {
            "node_id": self.graph_node_id,
            "result": result
        })
```

---

## UI Integration Summary

- Widgets with a `"callback"` automatically emit events
- The backend routes them to the correct node using `graph_node_id`
- Each node decides how to handle the event using `on_widget_event`

---


## 🤝 Let's Collaborate

Have ideas, feedback, or want to collaborate?

📬 **Email**: [arc.vortex@gmail.com](mailto:arc.vortex@gmail.com)  
💬 **Open an Issue**: [GitHub Issues](https://github.com/gl0p/tanalio/issues)

I’m always open to helping out, answering questions, or working together on cool ideas!


