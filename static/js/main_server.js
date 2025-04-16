// -----------------------------
// ✅ 1. WebSocket connection
// -----------------------------
let socket;

function connectWebSocket() {
  socket = new WebSocket(`ws://${location.host}/ws`);

  socket.addEventListener("open", () => {
    console.log("✅ Connected to WebSocket");
  });

  socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    handleBackendEvent(data);
  });

  socket.addEventListener("close", () => {
    console.warn("❌ Disconnected from WebSocket");
    setTimeout(connectWebSocket, 1000);
  });
}

connectWebSocket();

function sendEvent(event_type, node_id, payload = {}) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ event_type, node_id, payload }));
  }
}

// -----------------------------
// ✅ 2. Canvas + Graph + Editor
// -----------------------------
const canvas = document.getElementById("graphCanvas");

canvas.tabIndex = 10;
canvas.style.outline = "none";
canvas.style.userSelect = "none";

const graph = new LGraph();
const editor = new LGraphCanvas(canvas, graph);
graph.onAfterChange = autoSaveGraph;

const originalConnectNodes = editor.connectNodes;
editor.connectNodes = function (nodeA, slotA, nodeB, slotB, type) {
  const link = originalConnectNodes.call(this, nodeA, slotA, nodeB, slotB, type);
  if (!nodeA || !nodeB) return link;

  if (typeof nodeB.onConnectionsChange === "function") {
    nodeB.onConnectionsChange(LiteGraph.INPUT, slotB, true, {
      id: link.id,
      origin_id: nodeA.id,
      target_id: nodeB.id
    });
  }
  return link;
};

editor.allow_interaction = true;
editor.allow_editing = true;
editor.show_info = true;
editor.setZoom(1, true);
editor.draw_grid = true;

editor.graph = graph;
graph.start();
editor.draw(true);

window.graph = graph;
window.editor = editor;

canvas.addEventListener("mousedown", () => canvas.focus());
canvas.addEventListener("contextmenu", e => e.preventDefault());

window.addEventListener("resize", () => {
  if (resizeCanvasToDisplaySize(canvas)) editor.draw(true);
});

function resizeCanvasToDisplaySize(canvas) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    return true;
  }
  return false;
}

resizeCanvasToDisplaySize(canvas);

// -----------------------------
// ✅ 3. Load Nodes from Backend
// -----------------------------
fetch("/get_nodes")
  .then(res => res.json())
  .then((nodes) => {
    LiteGraph.clearRegisteredTypes();

    nodes.forEach((nodeDef) => {
      const fullName = `${nodeDef.category}/${nodeDef.title}`;
      const NodeClass = function () {
        LiteGraph.LGraphNode.call(this);
        this.title = nodeDef.title;
        this.category = nodeDef.category;
        this.size = nodeDef.size || [240, 120];
        this.properties = {};
        this.serialize_widgets = true;

        if (nodeDef.inputs) {
          nodeDef.inputs.forEach((input) => this.addInput(input.name, input.type));
        }

        if (nodeDef.outputs) {
          nodeDef.outputs.forEach((output) => this.addOutput(output.name, output.type));
        }

        if (nodeDef.widgets) {
          nodeDef.widgets.forEach((widget) => {
            // ✅ Modular folder picker support
            if (widget.type === "folder_picker") {
              const btn = this.addWidget("button", "📂 Choose Folder", "", () => {
                const picker = document.createElement("input");
                picker.type = "file";
                picker.webkitdirectory = true;
                picker.style.display = "none";
                document.body.appendChild(picker);

                picker.onchange = () => {
                  const files = Array.from(picker.files);
                  const folderName = files[0]?.webkitRelativePath?.split("/")[0] || "";
                  this.properties[widget.name] = folderName;

                  if (widget.callback) {
                    sendEvent(widget.callback, this.id, {
                      folder_path: folderName,
                      file_list: files.map(f => f.name),
                    });
                  }

                  document.body.removeChild(picker);
                  this.setDirtyCanvas(true, true);
                };

                picker.click();
              });
              return; // skip default addWidget
            }

            // 🧩 Default widget behavior
            this.addWidget(widget.type, widget.name, widget.value, (value) => {
              this.properties[widget.name] = value;
              emitWidgetIfNeeded(this, widget, value);
            }, widget.options || {});
          });
        }

      };

      NodeClass.prototype = Object.create(LiteGraph.LGraphNode.prototype);
      NodeClass.prototype.constructor = NodeClass;
      NodeClass.title = nodeDef.title;
      LiteGraph.registerNodeType(fullName, NodeClass);
    });

    console.log("✅ All node types registered.");
  });

// -----------------------------
// ✅ 4. Widget Event Emitter
// -----------------------------
function emitWidgetIfNeeded(node, widget, value) {
  if (!widget.callback) return;

  const payload = {};

  if (widget.type === "button") {
    payload.clicked = true;
    payload.input_features = Object.entries(node.properties)
      .filter(([k]) => k.startsWith("feature_"))
      .map(([_, v]) => parseFloat(v) || 0);
  } else {
    payload.value = value;
  }

  sendEvent(widget.callback, node.id, payload);
}

// -----------------------------
// ✅ 5. Backend Event Handlers
// -----------------------------
function handleBackendEvent(data) {
  const { event_type } = data;

  if (event_type === "toast") showToast(data.message);

  if (event_type === "property_update") {
    applyNodeUpdate(data.node_id, data.property, data.value);
  }

  if (event_type === "single_predict_result") {
    const node = graph.getNodeById(data.node_id);
    const widget = node?.widgets?.find(w => w.name === "prediction");
    if (widget) widget.value = data.result;
  }

  if (event_type === "node_active" || event_type === "node_inactive") {
    const node = graph.getNodeById(data.node_id);
    if (node) {
      node.is_active = (event_type === "node_active");
      node.setDirtyCanvas(true, true);
    }
  }

  if (event_type === "node_error" || event_type === "node_clear_error") {
    const node = graph.getNodeById(data.node_id);
    if (node) {
      node.is_error = (event_type === "node_error");
      node.setDirtyCanvas(true, true);
    }
  }
}

function applyNodeUpdate(nodeId, property, value) {
  const node = graph.getNodeById(nodeId);
  if (!node) return;

  node.properties[property] = value;
  const widget = node.widgets?.find(w => w.name === property);
  if (widget) widget.value = value;
  node.setDirtyCanvas(true, true);
}

// -----------------------------
// ✅ 6. Toast Utility
// -----------------------------
function showToast(message, duration = 4000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");

  toast.innerText = message;
  toast.style = `
    background: #333; color: white; padding: 10px 15px;
    border-radius: 8px; margin-top: 10px;
    font-family: monospace; font-size: 13px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    transition: opacity 0.3s ease;
  `;
  container.appendChild(toast);

  setTimeout(() => toast.style.opacity = "0", duration - 500);
  setTimeout(() => container.removeChild(toast), duration);
}

// -----------------------------
// ✅ 7. AutoSave (optional)
// -----------------------------
function autoSaveGraph() {
  const json = graph.serialize();
  fetch("/autosave_graph", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(json)
  });
}

// -----------------------------
// ✅ 8. Toolbar Button Handlers
// -----------------------------

// SAVE: Trigger "Save As" dialog
document.getElementById("saveGraphBtn").addEventListener("click", () => {
  const json = graph.serialize();
  const filename = prompt("Save graph as:", "my_graph.json");
  if (!filename) return;

  fetch(`/save_graph?filename=${filename}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(json)
  })
    .then(res => res.json())
    .then(result => {
      console.log("✅ Graph saved:", result.saved_to);
      showToast("✅ Graph saved!");
    });
});

// LOAD: Trigger file input
document.getElementById("loadGraphBtn").addEventListener("click", () => {
  document.getElementById("importGraphInput").click();
});

// When user picks a file
document.getElementById("importGraphInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    const content = event.target.result;
    try {
      const graphData = JSON.parse(content);
      graph.configure(graphData);
      console.log("✅ Graph loaded.");
      showToast("✅ Graph loaded!");
    } catch (err) {
      alert("❌ Failed to load graph: " + err.message);
    }
  };
  reader.readAsText(file);
});

// PAUSE Trainer
document.getElementById("pauseBtn").addEventListener("click", () => {
  const trainer = graph._nodes.find(n => n.title === "Trainer");
  if (trainer) {
    sendEvent("pause_training", trainer.id);
    console.log("⏸ Pause button clicked");
  }
});

// RESUME Trainer
document.getElementById("resumeBtn").addEventListener("click", () => {
  const trainer = graph._nodes.find(n => n.title === "Trainer");
  if (trainer) {
    sendEvent("resume_training", trainer.id);
    console.log("▶️ Resume button clicked");
  }
});

// STOP Trainer
document.getElementById("stopBtn").addEventListener("click", () => {
  const trainer = graph._nodes.find(n => n.title === "Trainer");
  if (trainer) {
    sendEvent("stop_training", trainer.id);
    console.log("🛑 Stop button clicked");
  }
});

// -----------------------------
// ✅ 9. Build AI Button
// -----------------------------

document.getElementById("exportGraphBtn").addEventListener("click", () => {
  const json = graph.serialize();

  fetch("/build_ai", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(json)
  })
    .then(res => res.json())
    .then(data => {
      console.log("🚀 Build AI triggered!", data);
      showToast("✅ AI Build request sent!");
    })
    .catch(err => {
      console.error("❌ Build failed:", err);
      showToast("❌ Build failed: " + err.message);
    });

  // 🧹 Reset trainer graph state
  graph._nodes.forEach(node => {
    if (node.title === "Trainer") {
      node.losses = [];
      node.epochs = [];
      node.accuracy = [];
      node.accuracies = [];
      node.val_losses = [];
      node.setDirtyCanvas(true, true);
      debouncedRedraw?.();
      console.log("🔄 Trainer state reset");
    }
  });
});
