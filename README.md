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

## 📦 Project Structure

```
├── app.py                   # Flask app entrypoint + API routes
├── graph_executor.py        # Core graph executor that builds and runs the node pipeline
├── node_registry.py         # Registry system for custom node classes
├── node_config.py           # Exports node metadata to frontend
├── base_model_node.py       # Abstract class for model layers
├── base_preprocess_node.py  # Abstract class for preprocessing nodes
├── load_all_nodes.py        # Dynamically loads all custom nodes
├── custom_nodes/            # Your custom node implementations go here
├── templates/index.html     # Main frontend page
├── datasets/                # Where your training/test data lives
└── saved_graphs/            # Where your graphs are autosaved
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/ai-builder
cd ai-builder

# Optional: Setup a virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## 🌐 Usage

1. Visit `http://localhost:5000` in your browser.
2. Drag nodes from the sidebar to the canvas.
3. Connect nodes together using compatible input/output ports.
4. Click buttons on nodes (e.g., “Run Prediction”) to trigger backend events.
5. Use the `Trainer` node to start training, and `Single Predict` to test your model.

---

## 🧩 Creating Custom Nodes

Just drop a Python file into the `custom_nodes/` folder using this structure:

```python
from node_registry import register_node
from base_model_node import BaseModelNode

@register_node(name="MyLayer", category="Model")
class MyLayerNode(BaseModelNode):
    widgets = [{"name": "units", "type": "int", "default": 64}]
    inputs = [{"name": "in", "type": "tensor"}]
    outputs = [{"name": "out", "type": "tensor"}]

    def get_layer(self):
        return nn.Linear(self.in_features, self.units)
```

It will automatically appear in the frontend!

---

## 📂 Dataset Support

- **CSV Datasets**: Place them under `datasets/csv/YourDataset/`. Must include a `.csv` file with headers.
- **Image Datasets**: Folders like `datasets/images/YourDataset/train`, `val`, `test`, with optional `annotations/` for COCO JSON.

---

## 🧠 How It Works

When a graph is submitted, `graph_executor.py` dynamically:

- Builds node instances
- Wires their inputs/outputs
- Executes build logic for data, models, training, and prediction

All node widgets emit frontend events to `on_widget_event()` in the Python class using a socket-based system.

---

