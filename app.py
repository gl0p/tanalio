from flask import Flask, render_template, jsonify, request
from sockets import socketio
import os
from utils.dataset_utils import validate_or_split_dataset, validate_or_split_csv_dataset, detect_image_format
from graph_executor import build_model_from_graph
from load_all_nodes import load_all_nodes
import json

app = Flask(__name__)
socketio.init_app(app)

# Load all nodes from the custom_nodes directory
NODE_REGISTRY = {}
GRAPH_DIR = "saved_graphs"
app.trainer_nodes = []
app.event_nodes = []  # for all nodes with event callbacks

def find_file_in_datasets(filename, base_dir="datasets"):
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None

def detect_annotation_tasks(annotation_dir):
    unique_tasks = set()
    for filename in os.listdir(annotation_dir):
        if filename.endswith(".json"):
            for key in ["instances", "captions", "keypoints"]:
                if key in filename:
                    unique_tasks.add(key)
    return sorted(list(unique_tasks))



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_nodes")
def get_nodes():
    from node_config import get_node_config
    return jsonify(get_node_config())


@app.route("/build_ai", methods=["POST"])
def build_ai():
    graph_json = request.get_json()
    model = build_model_from_graph(graph_json, app)
    print(model)

    return {"status": "model_built", "layers": str(model)}


@app.route("/validate_csv_dataset", methods=["POST"])
def validate_csv_dataset():
    import pandas as pd
    data = request.get_json()
    folder_name = data.get("folder_path")
    file_list = data.get("file_list", [])
    base_data_dir = os.path.abspath("datasets")
    # Correctly resolve nested folder under datasets/csv/
    csv_base_dir = os.path.join(base_data_dir, "csv")
    candidate_path = os.path.join(csv_base_dir, folder_name)
    # If direct path doesn't exist, try to resolve inside nested subfolders
    if not os.path.isdir(candidate_path):
        for sub in os.listdir(csv_base_dir):
            alt_path = os.path.join(csv_base_dir, sub, folder_name)
            if os.path.isdir(alt_path):
                candidate_path = alt_path
                print(f"✅ Resolved nested CSV folder: {candidate_path}")
                break

    if not os.path.isdir(candidate_path):
        return jsonify({"status": "error", "message": f"Folder not found: {candidate_path}"})

    folder_path = candidate_path

    print(f"📂 Validating CSV: {folder_path}")

    if not file_list:
        try:
            csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
        if not csv_files:
            return jsonify({"status": "error", "message": f"No CSV files in {folder_path}"})
        file_list = [csv_files[0]]

    filename = file_list[0]
    csv_path = os.path.join(folder_path, filename)
    result = validate_or_split_csv_dataset(folder_path, filename)
    try:
        df = pd.read_csv(csv_path, nrows=1)
        result["columns"] = list(df.columns)
    except Exception as e:
        result["columns"] = []
        print("⚠️ CSV header read error:", e)

    return jsonify({
        "status": result.get("status", "unknown"),
        "tensor_shape": result.get("tensor_shape"),
        "is_classification": result.get("is_classification"),
        "columns": result.get("columns", []),
        "format_type": "csv",
        "file_list": [filename]
    })


@app.route("/validate_image_dataset", methods=["POST"])
def validate_image_dataset():
    data = request.get_json()
    folder_name = data.get("folder_path")
    base_data_dir = os.path.abspath("datasets")
    folder_path = os.path.abspath(os.path.join(base_data_dir, folder_name))

    print(f"🖼️ Validating image dataset: {folder_path}")
    format_type = detect_image_format(folder_path)

    if format_type == "coco":
        annotation_dir = os.path.join(folder_path, "annotations")
        annotation_tasks = detect_annotation_tasks(annotation_dir)
        return jsonify({
            "status": "ok",
            "format": "coco",
            "annotation_tasks": annotation_tasks
        })

    result = validate_or_split_dataset(folder_path)
    return jsonify({
        "status": result.get("status", "unknown"),
        "tensor_shape": result.get("tensor_shape"),
        "is_classification": result.get("is_classification"),
        "columns": result.get("columns", []),
        "format_type": result.get("format_type", "unknown")
    })


@app.route("/autosave_graph", methods=["POST"])
def autosave_graph():
    graph_data = request.json
    # Strip heavy fields from Trainer and Predict nodes
    for node in graph_data.get("nodes", []):
        if node.get("title") in ["Trainer", "Single Predict"]:
            node.pop("losses", None)
            node.pop("epochs", None)
            node.pop("accuracies", None)
            node.pop("val_losses", None)
            node.pop("prediction", None)

    os.makedirs(GRAPH_DIR, exist_ok=True)
    with open(os.path.join(GRAPH_DIR, "autosave.json"), "w") as f:
        json.dump(graph_data, f, indent=2)
    return {"status": "saved"}

@app.route("/save_graph", methods=["POST"])
def save_graph():
    graph_data = request.json
    filename = request.args.get("filename", "autosave.json")
    save_path = os.path.join("saved_graphs", filename)

    os.makedirs("saved_graphs", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    return {"status": "ok", "saved_to": save_path}

@app.route("/load_graph", methods=["GET"])
def load_graph():
    with open(os.path.join(GRAPH_DIR, "autosave.json")) as f:
        content = f.read()
        try:
            graph = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON error: {e}")
            content = content.split("}\n{")[0] + "}"  # crude fix
            graph = json.loads(content)
        return graph


@app.route("/update_node_property", methods=["POST"])
def update_node_property():
    data = request.json
    # store property somewhere or log for frontend state re-sync
    return jsonify({"status": "ok"})

@socketio.on("pause_training", namespace="/")
def pause_training(data):
    node_id = data.get("node_id")
    print(f"⏸️ Pause requested for node_id={node_id}")
    for node in app.trainer_nodes:
        if node.graph_node_id == node_id:
            node.pause()
            break

@socketio.on("resume_training", namespace="/")
def resume_training(data):
    node_id = data.get("node_id")
    print(f"▶️ Resume requested for node_id={node_id}")
    for node in app.trainer_nodes:
        if node.graph_node_id == node_id:
            node.resume()
            break

@socketio.on("stop_training", namespace="/")
def stop_training(data):
    node_id = data.get("node_id")
    print(f"🛑 Stop requested for node_id={node_id}")
    for node in app.trainer_nodes:
        print(f"🔍 Checking node with id: {node.graph_node_id}")
        if node.graph_node_id == node_id:
            node.stop()
            break

@socketio.on("widget_event")
def handle_widget_event(data):
    node_id = data.get("node_id")
    event_type = data.get("event_type")
    payload = data.get("payload", {})
    print("📦 Current event_nodes:", [n.graph_node_id for n in app.event_nodes])
    print(f"📩 Widget event: node_id={node_id}, type={event_type}, payload={payload}")

    for node in app.event_nodes:
        if node.graph_node_id == node_id and hasattr(node, "on_widget_event"):
            try:
                node.on_widget_event(event_type, payload)
                print(f"✅ Event handled by node {node.title}")
            except Exception as e:
                print(f"❌ Error handling widget event: {e}")
                socketio.emit("toast", {
                    "message": f"❌ Error in '{event_type}': {str(e)}"
                })
            break


# @socketio.on("run_single_predict")
# def handle_single_predict(data):
#     node_id = data.get("node_id")
#     input_features = data.get("input_features")
#
#     print(f"📨 Received predict request for node_id={node_id} with input={input_features}")
#
#     found = False
#     for node in app.trainer_nodes:
#         if isinstance(node, SinglePredictNode) and node.graph_node_id == node_id:
#             found = True
#             if node.model is None:
#                 print(f"❌ No model linked to SinglePredictNode id={node_id}")
#                 socketio.emit("toast", {
#                     "message": "❌ Prediction failed: model not linked to this node."
#                 })
#                 return
#
#             node.set_user_input(input_features)
#             try:
#                 result = node.build(model_out=node.model)
#                 print(f"✅ Prediction result: {result}")
#                 socketio.emit("single_predict_result", {
#                     "node_id": node_id,
#                     "result": result
#                 })
#             except Exception as e:
#                 print(f"❌ Prediction error: {e}")
#                 socketio.emit("toast", {
#                     "message": f"❌ Prediction failed: {str(e)}"
#                 })
#             break
#
#     if not found:
#         print(f"❌ No matching SinglePredictNode found for node_id={node_id}")
#



if __name__ == "__main__":
    load_all_nodes()
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
