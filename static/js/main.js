const canvas = document.getElementById("graphCanvas");

// Force canvas to focus properly
canvas.tabIndex = 10;
canvas.style.outline = "none";
canvas.style.userSelect = "none";

// Set up graph and canvas
const graph = new LGraph();
graph.onAfterChange = autoSaveGraph;
const editor = new LGraphCanvas(canvas, graph); // ✅ THIS defines editor

const originalConnectNodes = editor.connectNodes;

editor.connectNodes = function(nodeA, slotA, nodeB, slotB, type) {
  const link = originalConnectNodes.call(this, nodeA, slotA, nodeB, slotB, type);

  if (!nodeA || !nodeB) return link;

  // Force trigger onConnectionsChange with proper format
  const outputSlot = nodeA.outputs?.[slotA];
  const inputSlot = nodeB.inputs?.[slotB];

  if (typeof nodeB.onConnectionsChange === "function") {
    nodeB.onConnectionsChange(LiteGraph.INPUT, slotB, true, {
      id: link.id,
      origin_id: nodeA.id,
      target_id: nodeB.id
    });
  }

  return link;
};

const tooltip = document.createElement("div");
tooltip.id = "nodeTooltip";
tooltip.style.position = "absolute";
tooltip.style.background = "rgba(30, 30, 30, 0.9)";
tooltip.style.color = "#fff";
tooltip.style.padding = "6px 10px";
tooltip.style.borderRadius = "6px";
tooltip.style.fontSize = "13px";
tooltip.style.pointerEvents = "none";
tooltip.style.whiteSpace = "pre-wrap";
tooltip.style.maxWidth = "300px";
tooltip.style.zIndex = "100";
tooltip.style.display = "none";
document.body.appendChild(tooltip);

editor.graph = graph;

graph.onNodeConnectionChange = function(changeType, linkInfo, inputSlot, outputSlot, linkId) {
  if (changeType === LiteGraph.INPUT) {
    const targetNode = graph.getNodeById(linkInfo.target_id);
    const sourceNode = graph.getNodeById(linkInfo.origin_id);

    if (!targetNode || !sourceNode) return;

    const sourceOut = sourceNode.properties?.["out_features"];
    if (typeof sourceOut === "undefined") return;

    if ("in_features" in targetNode.properties) {
      const isLocked = targetNode.properties?.["lock_in_features"];
      if (targetNode.title === "Dense Layer" && isLocked) return;

      targetNode.properties["in_features"] = sourceOut;

      setTimeout(() => {
        const inWidget = targetNode.widgets?.find(w => w.name === "in_features");
        if (inWidget) {
          inWidget.value = sourceOut;
          targetNode.setDirtyCanvas(true, true);
        }
      }, 0);

      console.log(`🔄 [First Connect] Synced ${targetNode.title} in_features to ${sourceOut} from ${sourceNode.title}`);
    }
  }
};


// Ensure full interaction
editor.allow_interaction = true;
editor.allow_editing = true;
editor.show_info = true;
editor.setZoom(1, true);
editor.draw_grid = true;

// Focus canvas on click (for dragging, keys, etc.)
canvas.addEventListener("mousedown", () => {
    canvas.focus();
});
canvas.addEventListener("contextmenu", e => e.preventDefault()); // prevent right-click menu bugs

let redrawTimeout;
function debouncedRedraw() {
  if (redrawTimeout) clearTimeout(redrawTimeout);
  redrawTimeout = setTimeout(() => {
    editor.draw(true);
  }, 100); // delay for batching
}

function emitWidgetEvent(nodeId, eventType, payload = {}) {
  socket.emit("widget_event", {
    node_id: nodeId,
    event_type: eventType, // ← like "run_prediction"
    payload: payload       // ← like { input_features: [...] }
  });
}
function emitWidgetIfNeeded(node, widget, value) {
  if (!widget.callback) return;

  const payload = {};

  if (widget.type === "button") {
    payload.clicked = true;

    // Collect input features from node properties
    const input_features = Object.entries(node.properties)
      .filter(([key]) => key.startsWith("feature_"))
      .map(([_, val]) => parseFloat(val) || 0);

    payload.input_features = input_features;

  } else {
    payload.value = value;
  }

  emitWidgetEvent(node.id, widget.callback, payload);
}


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

function showToast(message, duration = 10000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");

  toast.innerText = message;
    toast.style.background = "rgba(30, 30, 30, 0.95)"; // dark semi-transparent
    toast.style.border = "1px solid #ff4d4d";          // red border
    toast.style.color = "#ff4d4d";                     // red text for contrast
    toast.style.padding = "12px 16px";
    toast.style.marginTop = "10px";
    toast.style.borderRadius = "8px";
    toast.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.5)";
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s ease";
    toast.style.fontSize = "14px";
    toast.style.backdropFilter = "blur(6px)";


  container.appendChild(toast);

  // Fade in
  setTimeout(() => toast.style.opacity = "1", 100);

  // Fade out and remove
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => container.removeChild(toast), 500);
  }, duration);
}

window.addEventListener("resize", () => {
    if (resizeCanvasToDisplaySize(canvas)) {
        debouncedRedraw();
    }
});
window.addEventListener("load", () => {
    fetch("/load_graph")
        .then(res => res.json())
        .then(data => {
            if (data?.nodes) {
                graph.configure(data);
            }
        });
});

const socket = io(); // auto connects to current host
window.socket = socket;

socket.on("property_update", (data) => {
  const { node_id, property, value } = data;
  console.log("📡 Live update:", node_id, property, value);
  if (window.applyNodeUpdate) {
    window.applyNodeUpdate(node_id, property, value);
  }
});

socket.on("live_loss_update", (data) => {
  const chartImg = document.getElementById("loss_chart_img");
  if (chartImg && data.image) {
    chartImg.src = data.image;
  }
});

socket.on("toast", (data) => {
  showToast(data.message);
});

socket.on("test_accuracy_result", (data) => {
  const node = editor.graph.getNodeById(data.node_id);
  if (node && node.title === "Test Evaluator") {
    if (typeof data.accuracy === "number") {
      node.properties["test_accuracy"] = data.accuracy.toFixed(2) + "%";
      const accWidget = node.widgets?.find(w => w.name === "Test Accuracy");
      if (accWidget) accWidget.value = node.properties["test_accuracy"];
    }

    if (typeof data.loss === "number") {
      node.properties["test_loss"] = data.loss.toFixed(4);
      const lossWidget = node.widgets?.find(w => w.name === "Test Loss");
      if (lossWidget) lossWidget.value = node.properties["test_loss"];
    }

    debouncedRedraw();

  }
});


socket.on("loss_update", (data) => {
  graph._nodes.forEach(node => {
    if (node.title === "Trainer") {
      node.losses = node.losses || [];
      node.epochs = node.epochs || [];
      node.accuracies = node.accuracies || [];

      node.losses.push(data.loss);
      node.epochs.push(data.epoch);
      if (typeof data.accuracy === "number") {
        node.accuracies.push(data.accuracy);
      }

      node.setDirtyCanvas(true, true);
      debouncedRedraw();
    }
  });
});


socket.on("node_active", (data) => {
  console.log("🟢 ACTIVATING:", data.node_id);
  const node = editor.graph.getNodeById(data.node_id);
  if (node) {
    node.is_active = true;
    console.log("✅ Found node:", node.title);
    node.setDirtyCanvas(true, true);
  } else {
    console.warn("❌ No node found with id", data.node_id);
  }
});

socket.on("dynamic_widget_update", (data) => {
  const { node_id, feature_names } = data;
  const node = LiteGraph.getNodeById(node_id);
  if (!node || !Array.isArray(feature_names)) return;

  // 🧹 Remove only old dynamic feature widgets
  node.widgets = node.widgets.filter(w =>
    !w.name?.startsWith("feature_") &&
    w.name !== "Run Prediction" &&
    w.name !== "prediction"
  );

  // 🧠 Store feature names
  node.feature_names = feature_names;

  // ➕ Add new feature input fields
  feature_names.forEach((name) => {
    node.addWidget("number", `feature_${name}`, 0, (val) => {
      node.properties[`feature_${name}`] = val;
    });
  });

  // ➕ Re-add Run Prediction Button
  node.addWidget("button", "Run Prediction", "", () => {
    const features = feature_names.map((key) => {
      return parseFloat(node.properties[`feature_${key}`]) || 0;
    });

    socket.emit("run_single_predict", {
      node_id,
      input_features: features
    });

    console.log("📤 Sent updated prediction:", features);
  });

  // ➕ Add prediction result display (if missing)
  const predWidget = node.widgets.find(w => w.name === "prediction");
  if (!predWidget) {
    node.addWidget("text", "prediction", "n/a", () => {});
  }

  node.setDirtyCanvas(true, true);
});


socket.on("node_inactive", (data) => {
  const node = editor.graph.getNodeById(data.node_id);
  if (node) {
    node.is_active = false;
    node.setDirtyCanvas(true, true);
  }
});

window.applyNodeUpdate = function(nodeId, property, value) {
  const node = editor.graph._nodes.find(n => n.id === nodeId);
  if (!node) return;

  // ⛔ Optional safeguard for locked in_features
  if (property === "in_features" && node.properties?.lock_in_features) {
    console.log("🔒 Skipping update: lock_in_features is enabled on", node.title);
    return;
  }

  // ✅ Null check
  if (value == null) {
    console.warn(`⚠️ Skipping update: '${property}' is null for node '${node.title}'`);
    return;
  }

  node.properties[property] = value;

  const widget = node.widgets?.find(w => w.name === property);
  if (widget) {
    widget.value = typeof value === "string" ? value : value.toString?.();

    if (typeof node.onWidgetChanged === "function") {
      node.onWidgetChanged(property, value, widget);
    }
  }

  node.setDirtyCanvas(true, true);
  console.log(`✅ Synced [${property}] = ${value} for node [${node.title}]`);
};

socket.on("node_error", (data) => {
  const node = editor.graph.getNodeById(data.node_id);
  if (node) {
    node.is_error = true;
    node.error_message = data.message || "Error";
    node.setDirtyCanvas(true, true);
  }
});

socket.on("node_clear_error", (data) => {
  const node = editor.graph.getNodeById(data.node_id);
  if (node) {
    node.is_error = false;
    node.error_message = "";
    node.setDirtyCanvas(true, true);
  }
});


function autoSaveGraph() {
    const graphData = graph.serialize(); // ✅ Use the actual graph instance

    // 💣 Strip heavy sample tensor values
      graphData.nodes?.forEach(n => {
        if (n.title === "Load Images" && n.properties?.sample_tensor) {
          delete n.properties.sample_tensor;
        }
      });
    fetch("/autosave_graph", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(graphData)
    });
}


resizeCanvasToDisplaySize(canvas); // ⬅️ Sync actual vs display size

function drawSmoothCurve(ctx, points) {
  if (points.length < 2) return;

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);

  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i - 1] || points[i];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2] || p2;

    // Catmull-Rom to Bezier conversion
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;

    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  }

  ctx.stroke();
}

// Submit graph to server
document.getElementById("exportGraphBtn").addEventListener("click", () => {
    const json = graph.serialize();
    fetch("/build_ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(json)
    })
    .then(res => res.json())
    .then(data => {
        console.log("Graph submitted!", data);
    });
    graph._nodes.forEach(node => {
      if (node.title === "Trainer") {
        node.losses = [];
        node.epochs = [];
        node.accuracy = [];
        node.accuracies = [];
        node.val_losses = [];
        node.setDirtyCanvas(true, true);
        debouncedRedraw();
        console.log("🔄 Trainer state reset");
      }
    });

});

// Load and register nodes
fetch('/get_nodes')
  .then(res => res.json())
  .then(nodes => {
    LiteGraph.clearRegisteredTypes();

    nodes.forEach(nodeDef => {
      const fullName = `${nodeDef.category}/${nodeDef.title}`;

      // Dynamically define a class per node type
      const NodeClass = function() {
          // Call parent constructor
          LiteGraph.LGraphNode.call(this);

          this.title = nodeDef.title;
          this.category = nodeDef.category;
          this.resizable = true;
          this.serialize_widgets = true;
          this.size = nodeDef.size || [300, 250];
          this.properties = {};
          this.id = nodeDef.id;

          if (this.title === "Single Predict") {
              this.addOutput("prediction", "number");  // Optional output

              this._updateFeatureInputs = (columns = []) => {
                // 🧹 Clear all old feature values from properties
                Object.keys(this.properties).forEach(key => {
                  if (key.startsWith("feature_")) {
                    delete this.properties[key];
                  }
                });

                // Remove old widgets except for prediction
                this.widgets = this.widgets?.filter(w =>
                  !w.name?.startsWith("feature_") &&
                  w.name !== "Run Prediction" &&
                  w.name !== "prediction"
                ) || [];

                // Add input fields
                columns.forEach((col) => {
                  this.addWidget("text", `feature_${col}`, "0", (val) => {
                    this.properties[`feature_${col}`] = val;
                  });
                });

                // ✅ Add Run Prediction button with valid callback
                const runBtn = this.addWidget("button", "Run Prediction", "", () => {
                  const features = Object.entries(this.properties)
                    .filter(([k]) => k.startsWith("feature_"))
                    .map(([_, v]) => parseFloat(v) || 0);

                  socket.emit("run_single_predict", {
                    node_id: this.id,
                    input_features: features
                  });

                  console.log("📤 Sent prediction request with features:", features);
                });

                // ✅ Safe add prediction output (without warning)
                const predWidget = this.addWidget("text", "prediction", this.properties["prediction"] || "n/a", () => {});
                predWidget.value = this.properties["prediction"] || "n/a";

                this.setDirtyCanvas(true, true);
              };

              // Optional: show sample result
              this._showPrediction = (result) => {
                this.properties["prediction"] = result.toString();
                const predWidget = this.widgets?.find(w => w.name === "prediction");
                if (predWidget) {
                  predWidget.value = result.toString();
                  this.setDirtyCanvas(true, true);
                }
              };

              const _this = this;
              socket.on("single_predict_result", (data) => {
                if (data.node_id === _this.id) {
                  _this._showPrediction(data.result);
                }
              });
            }

          if (nodeDef.title === "Test Evaluator") {
              this.properties["test_accuracy"] = "n/a";
              this.properties["test_loss"] = "n/a";

              this.addWidget("text", "Test Accuracy", "n/a", () => {});
              this.addWidget("text", "Test Loss", "n/a", () => {});
              this.setDirtyCanvas(true, true);
            }


          if (nodeDef.title === "Trainer"){
            this.losses = [];
            this.epochs = [];
            this.accuracies = [];
            this.val_losses = [];

            // 👇 We'll use this inside the merged onDrawForeground
            this._drawTrainerChart = function(ctx) {
              const w = this.size[0];
              const h = this.size[1];

              const chartPadding = 10;
              const chartHeight = (h - 160) / 2;
              const chartWidth = w - 50;

              const yOffsetLoss = 110;
              const yOffsetAcc = yOffsetLoss + chartHeight + 40;

              // === Loss Chart ===
              ctx.fillStyle = "#000";
              ctx.fillRect(40, yOffsetLoss, chartWidth, chartHeight);

              ctx.strokeStyle = "#333";
              for (let i = 0; i <= 5; i++) {
                const y = yOffsetLoss + (chartHeight / 5) * i;
                ctx.beginPath();
                ctx.moveTo(40, y);
                ctx.lineTo(40 + chartWidth, y);
                ctx.stroke();
              }

              ctx.fillStyle = "#aaa";
              ctx.font = "14px monospace";
              ctx.textAlign = "left";
              ctx.fillText("Loss", w/2, yOffsetLoss - 10); // left side label

              if (this.epochs && this.epochs.length > 0) {
                const currentEpoch = this.epochs[this.epochs.length - 1];
                ctx.textAlign = "right";
                ctx.fillText(`Epoch ${currentEpoch}`, 40 + chartWidth, yOffsetLoss - 10); // right side
              }


              if (this.losses && this.losses.length > 1) {
                const maxLoss = Math.max(...this.losses);
                const minLoss = Math.min(...this.losses);
                for (let i = 0; i <= 5; i++) {
                  const y = yOffsetLoss + (chartHeight / 5) * i;
                  const value = (maxLoss - ((maxLoss - minLoss) / 5) * i).toFixed(2);
                  ctx.fillText(value, 35, y + 3);
                }

                ctx.strokeStyle = "orange";
                ctx.lineWidth = 2;

                const points = this.losses.map((val, i) => ({
                  x: 40 + (i / (this.losses.length - 1)) * chartWidth,
                  y: yOffsetLoss + chartHeight - ((val - minLoss) / (maxLoss - minLoss + 1e-5)) * chartHeight
                }));
                drawSmoothCurve(ctx, points);
                const lastLoss = this.losses[this.losses.length - 1];
                const x = 40 + ((this.losses.length - 1) / (this.losses.length - 1)) * chartWidth;
                const y = yOffsetLoss + chartHeight - ((lastLoss - minLoss) / (maxLoss - minLoss + 1e-5)) * chartHeight;

                ctx.fillStyle = "orange";
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fill();

                ctx.font = "13px monospace";
                ctx.fillStyle = "orange";
                ctx.textAlign = "left";
                ctx.fillText(`${lastLoss.toFixed(5)}`, x+8, y + 5); // ⬅️ marker label

              }


              // === Accuracy Chart ===
              ctx.fillStyle = "#000";
              ctx.fillRect(40, yOffsetAcc, chartWidth, chartHeight);

              ctx.strokeStyle = "#333";
              for (let i = 0; i <= 5; i++) {
                const y = yOffsetAcc + (chartHeight / 5) * i;
                ctx.beginPath();
                ctx.moveTo(40, y);
                ctx.lineTo(40 + chartWidth, y);
                ctx.stroke();
              }

              ctx.fillStyle = "#aaa";
              ctx.textAlign = "left";
              ctx.fillText("Accuracy", w / 2 - 10, yOffsetAcc - 8);

              if (this.accuracies && this.accuracies.length > 1) {
                for (let i = 0; i <= 5; i++) {
                  const y = yOffsetAcc + (chartHeight / 5) * i;
                  const value = (100 - 20 * i).toFixed(0) + "%";
                  ctx.fillText(value, 5, y + 3);
                }

                ctx.strokeStyle = "lightblue";
                ctx.lineWidth = 2;

                const points = this.accuracies.map((val, i) => ({
                  x: 40 + (i / (this.accuracies.length - 1)) * chartWidth,
                  y: yOffsetAcc + chartHeight - (val / 100) * chartHeight
                }));
                drawSmoothCurve(ctx, points);
                const lastAccuracy = this.accuracies[this.accuracies.length - 1];
                const x = 40 + ((this.accuracies.length - 1) / (this.accuracies.length - 1)) * chartWidth;
                const y = yOffsetAcc + chartHeight - (lastAccuracy / 100) * chartHeight;

                ctx.fillStyle = "lightblue";
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fill();

                ctx.font = "13px monospace";
                ctx.fillStyle = "lightblue";
                ctx.textAlign = "left";
                ctx.fillText(`${lastAccuracy.toFixed(1)}%`, x+8, y + 5); // ⬅️ marker label

              }

            };


          }
          const originalDraw = NodeClass.prototype.onDrawForeground;
            NodeClass.prototype.onDrawForeground = function(ctx) {
              // Call internal chart drawing if exists
              if (typeof this._drawTrainerChart === "function") {
                this._drawTrainerChart(ctx);
              }

              // Call other node-specific drawing if any
              if (typeof originalDraw === "function") {
                originalDraw.call(this, ctx);
              }
              // 🔴 Error border
              if (this.is_error) {
                const radius = 10;
                const w = this.size[0];
                const h = this.size[1];

                ctx.strokeStyle = "red";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(radius, 0);
                ctx.lineTo(w - radius, 0);
                ctx.quadraticCurveTo(w, 0, w, radius);
                ctx.lineTo(w, h - radius);
                ctx.quadraticCurveTo(w, h, w - radius, h);
                ctx.lineTo(radius, h);
                ctx.quadraticCurveTo(0, h, 0, h - radius);
                ctx.lineTo(0, radius);
                ctx.quadraticCurveTo(0, 0, radius, 0);
                ctx.closePath();
                ctx.stroke();
                return; // Skip green if errored
              }
              // ✅ Draw green outline if active
              if (this.is_active) {
                const radius = 10;
                const w = this.size[0];
                const h = this.size[1];

                ctx.strokeStyle = "limegreen";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(radius, 0);
                ctx.lineTo(w - radius, 0);
                ctx.quadraticCurveTo(w, 0, w, radius);
                ctx.lineTo(w, h - radius);
                ctx.quadraticCurveTo(w, h, w - radius, h);
                ctx.lineTo(radius, h);
                ctx.quadraticCurveTo(0, h, 0, h - radius);
                ctx.lineTo(0, radius);
                ctx.quadraticCurveTo(0, 0, radius, 0);
                ctx.closePath();
                ctx.stroke();

              }
            };


          // Add inputs
          if (nodeDef.inputs) {
            nodeDef.inputs.forEach(input => this.addInput(input.name, input.type));
          }

          // Add outputs
          if (nodeDef.outputs) {
            nodeDef.outputs.forEach(output => this.addOutput(output.name, output.type));
          }
          if (this.title === "AutoFlatten") {
              // Set a placeholder out_features (this could eventually be inferred on the backend)
              this.properties["out_features"] = 0;

              const outWidget = this.addWidget("text", "out_features", "0", () => {});
              outWidget.value = "0";
              debouncedRedraw()
            }


          // Add widgets
            if (nodeDef.widgets) {
              nodeDef.widgets.forEach(widget => {
                if (widget.type === "folder_picker") {
                  const buttonWidget = this.addWidget("button", "Choose Folder", "", () => {
                    const folderInput = document.getElementById("folderPicker");
                    folderInput.onchange = (event) => {
                      const files = Array.from(event.target.files);
                      const fullPath = files[0]?.webkitRelativePath?.split("/")[0] || "";
                      this.properties[widget.name] = fullPath;
                      this.properties["file_list"] = []; // empty or undefined
                      buttonWidget.name = `📂 ${fullPath}`;
                      buttonWidget.value = fullPath;

                      const endpoint = fullPath.includes("csv") ? "/validate_csv_dataset" : "/validate_image_dataset";
                        fetch(endpoint, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            folder_path: fullPath,
                            file_list: this.properties["file_list"],
                            target_column: this.properties["target_column"]
                          })
                        })

                      .then(res => res.json())
                      .then(result => {
                        console.log("✅ Folder validated", result);
                        if (result.file_list?.length) {
                            this.properties["file_list"] = result.file_list;
                            console.log("📁 Injected validated file list:", result.file_list);
                          }
                        if (result.annotation_tasks?.length) {
                          let taskWidget = this.widgets?.find(w => w.name === "annotation_task");

                          if (!taskWidget) {
                            taskWidget = this.addWidget("combo", "annotation_task", result.annotation_tasks[0], (val) => {
                              this.properties["annotation_task"] = val;
                            }, { values: result.annotation_tasks });
                            this.properties["annotation_task"] = result.annotation_tasks[0];
                            this.setDirtyCanvas(true, true);
                            console.log("🧩 Created annotation_task widget dynamically");
                          } else {
                            taskWidget.options.values = result.annotation_tasks;
                            taskWidget.value = result.annotation_tasks[0];
                            this.properties["annotation_task"] = taskWidget.value;
                            this.setDirtyCanvas(true, true);
                            console.log("🔁 Updated annotation_task values");
                          }
                        }


                        // 🔄 Inject columns into target_column widget
                        const targetWidget = this.widgets?.find(w => w.name === "target_column");
                        if (targetWidget && result.columns?.length) {
                          targetWidget.options.values = result.columns;
                          targetWidget.value = result.columns[result.columns.length - 1]; // default: last column
                          this.properties["target_column"] = targetWidget.value;
                          this.setDirtyCanvas(true, true);
                        }
                        // 🔮 Inject feature widgets into Single Predict nodes
                          const targetColumn = targetWidget?.value || result.columns?.at(-1);
                          const featureColumns = result.columns?.filter(col => col !== targetColumn) || [];

                          editor.graph._nodes.forEach(n => {
                            if (n.title === "Single Predict") {
                              // Clear old widgets
                              n.widgets = n.widgets?.filter(w => !w.name?.startsWith("feature_")) || [];

                              // Add one text widget per feature
                              featureColumns.forEach(col => {
                                n.addWidget("text", `feature_${col}`, "0", val => {
                                  n.properties[`feature_${col}`] = val;
                                });
                              });

                              // Add prediction result widget (if missing)
                              if (!n.widgets.find(w => w.name === "prediction")) {
                                n.addWidget("text", "prediction", "n/a");
                              }

                              n.setDirtyCanvas(true, true);
                            }
                          });
                      });
                    };
                    folderInput.click();
                  });
                }

                // 🧮 All Other Widgets
                else {
                  this.addWidget(
                  widget.type,
                  widget.name,
                  widget.value,
                  (value) => {
                    this.properties[widget.name] = value;
                     // ✨ Emit event for target_column or other widgets
                      if (widget.name === "target_column") {
                        emitWidgetEvent(this.id, "widget_event", { target_column: value });
                      }

                      // Optionally emit other widget events
                      emitWidgetIfNeeded(this, widget, value);
                    // Propagate out_features changes to connected nodes
                    if (widget.name === "out_features") {
                      const outputLinks = this.outputs?.[0]?.links || [];

                      outputLinks.forEach(linkId => {
                        const link = this.graph.links[linkId];
                        const targetNode = this.graph.getNodeById(link.target_id);
                        if (!targetNode) return;

                        const targetInFeat = targetNode.properties?.["in_features"];
                        const outFeatures = this.properties["out_features"];
                        if (outFeatures && targetInFeat !== outFeatures) {
                          if (targetNode.title === "Dense Layer" && targetNode.properties["lock_in_features"]) return;

                          targetNode.properties["in_features"] = outFeatures;

                          const inWidget = targetNode.widgets?.find(w => w.name === "in_features");
                          if (inWidget) inWidget.value = outFeatures;

                          targetNode.setDirtyCanvas(true, true);
                          console.log(`🔄 Updated ${targetNode.title} in_features to ${outFeatures}`);
                        }
                      });
                    }


                    // 🔁 Auto-revalidate when any param changes
                    if (
                      ["batch_size", "resize_width", "resize_height", "mean", "std"].includes(widget.name)
                      && this.properties["folder_path"]
                    ) {
                      const folderPath = this.properties["folder_path"] || "";
                      const endpoint = folderPath.includes("csv") ? "/validate_csv_dataset" : "/validate_image_dataset";
                      fetch(endpoint, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          folder_path: folderPath,
                          resize_width: this.properties["resize_width"],
                          resize_height: this.properties["resize_height"],
                          batch_size: this.properties["batch_size"],
                          mean: this.properties["mean"],
                          std: this.properties["std"]
                        })
                      })

                      .then(res => res.json())
                      .then(result => {
                        console.log("🔁 Re-validated due to widget change:", result);

                        // ✅ Update flatten if it exists
                        const flattenNode = editor.graph._nodes.find(n => n.title === "Flatten");
                        if (flattenNode) {
                          const tensorShape = result.tensor_shape;
                          const outFeatures = tensorShape ? tensorShape[0] * tensorShape[1] * tensorShape[2] : 0;

                          flattenNode.properties["out_features"] = outFeatures;
                          const widget = flattenNode.widgets?.find(w => w.name === "out_features");
                          if (widget) widget.value = outFeatures;
                          flattenNode.setDirtyCanvas(true, true);
                        }
                      });
                    }
                  },
                  widget.options || {}
                );

                }
              });
            }
            this.onConnectionsChange = function(type, slot, connected, link_info, input) {
              if (type !== LiteGraph.INPUT || !connected || !link_info) return;

              const link = this.graph.links[link_info.id];
              if (!link) return;

              const sourceNode = this.graph.getNodeById(link.origin_id);
              if (!sourceNode) return;

              let sourceOut = sourceNode.properties?.["out_features"];
              if (sourceOut === undefined || sourceOut === null || isNaN(Number(sourceOut))) return;

              sourceOut = Number(sourceOut); // ✅ Force to number
              console.log("🔌 Connection Event Fired:", {
                sourceNode: sourceNode.title,
                sourceOut,
                targetNode: this.title
              });
              // Special case: Flatten → Dense Layer
              if (sourceNode.title === "AutoFlatten" && this.title === "Dense Layer") {
                if (!this.properties["lock_in_features"]) {
                  this.properties["in_features"] = sourceOut;
                  const inFeatWidget = this.widgets?.find(w => w.name === "in_features");
                  if (inFeatWidget) inFeatWidget.value = sourceOut;
                  this.setDirtyCanvas(true, true);
                  console.log("🔁 Dense Layer updated from Flatten out_features:", sourceOut);
                }
                return;
              }
                if (this.title === "Concat") {
                  const existingInputs = this.inputs.length;
                  const isLastSlot = slot === existingInputs - 1;
                  if (connected && isLastSlot) {
                    const newIndex = existingInputs;
                    this.addInput("input_" + newIndex, "tensor");
                    this.setDirtyCanvas(true, true);
                    console.log(`➕ Added input_${newIndex} to Concat`);
                  }
                }
                if (this.title === "Add") {
                  const existingInputs = this.inputs.length;
                  const isLastSlot = slot === existingInputs - 1;
                  if (connected && isLastSlot) {
                    const newIndex = existingInputs;
                    this.addInput("input_" + newIndex, "tensor");
                    this.setDirtyCanvas(true, true);
                    console.log(`➕ Added input_${newIndex} to Concat`);
                  }
                }


              // Generic: Any source → target that supports in_features
              if ("in_features" in this.properties) {
                  const isLocked = this.properties["lock_in_features"];
                  if (this.title === "Dense Layer" && isLocked) return;

                  this.properties["in_features"] = sourceOut;

                  const inFeatWidget = this.widgets?.find(w => w.name === "in_features");
                  if (inFeatWidget) {
                    inFeatWidget.value = sourceOut;

                    // ⚡ Trigger internal widget update callback
                    if (this.onWidgetChanged) {
                      this.onWidgetChanged("in_features", sourceOut, inFeatWidget);
                    }
                  }

                  this.setDirtyCanvas(true, true);
                  console.log(`✅ Final Sync: ${this.title} in_features = ${sourceOut}`);
                }



            };



      };


      LiteGraph.registerNodeType(fullName, NodeClass);
      // 👇 Inherit from LGraphNode
      NodeClass.prototype = Object.create(LiteGraph.LGraphNode.prototype);
      NodeClass.prototype.constructor = NodeClass;

      NodeClass.title = nodeDef.title;
      NodeClass.category = nodeDef.category;
      // 🔁 Central green border handler
      const originalDraw = NodeClass.prototype.onDrawForeground;
      NodeClass.prototype.onDrawForeground = function(ctx) {
        if (typeof originalDraw === "function") {
          originalDraw.call(this, ctx);
        }
      };

      // ✅ Set static default size to apply on first drop
      if (nodeDef.title === "Trainer") {
        NodeClass.size = [300, 400];  // <--- THIS is what LiteGraph uses
      }
      if (nodeDef.title === "Hyperparameters") {
        NodeClass.size = [260, 220];
      }
      if (nodeDef.title === "Load CSV") {
        NodeClass.size = [260, 220];
      }
    });
  });

graph.onNodeConnectionChange = function(changeType, linkInfo, inputSlot, outputSlot, linkId) {
  if (changeType === LiteGraph.INPUT) {
    const targetNode = graph.getNodeById(linkInfo.target_id);
    if (targetNode?.onConnectionsChange) {
      targetNode.onConnectionsChange(LiteGraph.INPUT, inputSlot, true, linkInfo);
    }
  }
};

// Optional debug: log selected node
editor.onShowNodePanel = (node) => console.log("Selected node:", node);

// Start graph engine
graph.start();
editor.draw(true); // force first draw

window.graph = graph;
window.editor = editor;

graph._nodes.forEach(n => {
  if (n.title === "Trainer") {
    if (typeof n.onAdded === "function") n.onAdded();
    n.drawForeground = true; // ensure draw mode
    n.setDirtyCanvas(true, true);
  }
});

// SAVE: Trigger "Save As" dialog
document.getElementById("saveGraphBtn").addEventListener("click", () => {
    const json = graph.serialize();
    const filename = prompt("Save graph as:", "my_graph.json");
    if (!filename) return;

    fetch(`/save_graph?filename=${filename}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(json)
    }).then(res => res.json()).then(result => {
        console.log("✅ Graph saved:", result.saved_to);
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
        } catch (err) {
            alert("❌ Failed to load graph: " + err.message);
        }
    };
    reader.readAsText(file);
});

document.getElementById("pauseBtn").addEventListener("click", () => {
  const trainer = graph._nodes.find(n => n.title === "Trainer");
  if (trainer) {
    socket.emit("pause_training", { node_id: trainer.id });
    console.log("⏸ Pause button clicked");
  }
});

document.getElementById("resumeBtn").addEventListener("click", () => {
  const trainer = graph._nodes.find(n => n.title === "Trainer");
  if (trainer) {
    socket.emit("resume_training", { node_id: trainer.id });
    console.log("▶️ Resume button clicked");
  }
});

document.getElementById("stopBtn").addEventListener("click", () => {
  const trainer = graph._nodes.find(n => n.title === "Trainer");
  if (trainer) {
    socket.emit("stop_training", { node_id: trainer.id });
    console.log("🛑 Stop button clicked, sending node id", trainer.id);
  }
});


