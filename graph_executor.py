from node_registry import NODE_CLASS_MAP, NODE_TAGS
from sockets import socketio
import inspect

def inject_node_outputs(output, tgt, input_name, tgt_id):
    if not isinstance(output, dict):
        setattr(tgt, input_name, output)
        return

    # Special case for hyperparameters
    if input_name == "hyperparams":
        tgt.hyperparams = output
        print(f"⚙️ Injected hyperparams into {tgt.__class__.__name__} (node_id={tgt_id})")
        return

    val = output.get(input_name)

    # Don't inject raw strings for tensors
    if input_name not in ["train", "val", "test"] and isinstance(val, str):
        return

    if val is not None:
        setattr(tgt, input_name, val)

    # 🔁 Inject sample_tensor
    if "sample_tensor" in output and hasattr(tgt, "sample_tensor"):
        tgt.sample_tensor = output["sample_tensor"]
        print(f"📦 Injected sample_tensor into {tgt.__class__.__name__} (id={tgt_id})")

    # 🔁 Trigger shape handling
    if input_name == "in" and hasattr(tgt, "set_input_shape"):
        if "sample_tensor" in output:
            # print(f"📦 Passing shape to {tgt.__class__.__name__}: {output.get('sample_tensor')}")
            tgt.set_input_shape(output["sample_tensor"])  # ✅ actual tensor
        else:
            tgt.set_input_shape(val)  # fallback (e.g. if val is a tensor)

    # 🔁 Propagate out_features → in_features
    if "out_features" in output:
        out_feat = output["out_features"]
        if hasattr(tgt, "in_features"):
            tgt.in_features = out_feat
            socketio.emit("property_update", {
                "node_id": tgt_id,
                "property": "in_features",
                "value": out_feat
            })
        if hasattr(tgt, "input_size"):
            tgt.input_size = out_feat
            socketio.emit("property_update", {
                "node_id": tgt_id,
                "property": "input_size",
                "value": out_feat
            })
        print(f"🔄 Auto-updated in_features = {out_feat} on {tgt.__class__.__name__} (node_id={tgt_id})")

    # 🔁 num_classes → out_features
    if "num_classes" in output and hasattr(tgt, "out_features"):
        tgt.num_classes = output["num_classes"]
        socketio.emit("property_update", {
            "node_id": tgt_id,
            "property": "out_features",
            "value": output["num_classes"]
        })

    # 🔁 task_type
    if "task_type" in output and hasattr(tgt, "task_type"):
        tgt.task_type = output["task_type"]
        print(f"🧠 Injected task_type = {output['task_type']} into {tgt.__class__.__name__}")

    # 🔁 target_column → update downstream predictors
    if "columns" in output and "target_column" in output:
        input_features = [col for col in output["columns"] if col != output["target_column"]]
        for attr in ["set_feature_info", "feature_names"]:
            if hasattr(tgt, attr):
                try:
                    tgt.set_feature_info(
                        feature_names=input_features,
                        input_mean=output.get("input_mean"),
                        input_std=output.get("input_std"),
                        output_mean=output.get("output_mean"),
                        output_std=output.get("output_std"),
                    )
                    print(f"🧠 Sent input feature info to {tgt.__class__.__name__}")
                    break
                except Exception as e:
                    print(f"⚠️ Feature info injection failed: {e}")


def build_model_from_graph(graph_json, app):
    print("🧠 START OF GRAPH EXECUTOR")
    nodes = graph_json["nodes"]
    links = graph_json["links"]

    node_instances = {}
    node_map = {}
    link_map = {}

    # Build link map
    for link in links:
        link_id, src_id, src_slot, tgt_id, tgt_slot, _ = link
        link_map[link_id] = [src_id, src_slot, tgt_id, tgt_slot]

    # Instantiate nodes
    for node in nodes:
        node_type = node["type"].split("/")[-1]
        node_id = node["id"]
        props = node.get("properties", {})
        cls = NODE_CLASS_MAP.get(node_type)
        if not cls:
            print(f"❌ Unknown node type: {node_type}")
            socketio.emit("toast", {"message": f"⚠️ Missing class for {node_type}"})
            continue
        try:
            sig = inspect.signature(cls.__init__)
            valid_keys = list(sig.parameters.keys())[1:]
            filtered_props = {k: v for k, v in props.items() if k in valid_keys}
            instance = cls(**filtered_props)
            instance.graph_node_id = node_id
            node_instances[node_id] = instance
            node_map[node_id] = node

            app.graph_links = links  # Store full list of graph links
            app.graph_nodes = node_instances  # Optional, in case we need lookup

            # 💡 Add to event_nodes if it supports widget callbacks
            if hasattr(instance, "on_widget_event"):
                app.event_nodes.append(instance)

        except Exception as e:
            print(f"❌ Failed to instantiate {node_type}: {e}")
            socketio.emit("node_error", {"node_id": node_id})

    # Special context injection for OutputLayer
    for node_id, inst in node_instances.items():
        if inst.__class__.__name__ == "OutputLayer":
            inst.graph_nodes = node_instances
            inst.graph_nodes_data = node_map
            inst.links = link_map
            inst.final_node_id = node_id

    # Build connections
    connections = []
    for link_id, (src, _, tgt, tgt_slot) in link_map.items():
        try:
            input_name = node_map[tgt]["inputs"][tgt_slot]["name"]
            connections.append((node_instances[src], node_instances[tgt], input_name, tgt, tgt_slot))
        except Exception as e:
            print(f"❌ Skipping connection {src} → {tgt}: {e}")

    print(f"✅ MADE CONNECTIONS: {[(a.__class__.__name__, b.__class__.__name__, c) for a, b, c, _, _ in connections]}")

    # Build all Data/Config/Preprocessing nodes
    built_outputs = {}
    hyperparam_task_type = None

    for node_id, inst in node_instances.items():
        node_type = node_map[node_id]["type"].split("/")[-1]
        category = getattr(inst, "CATEGORY", "")
        print(f"Building > NODE TYPE: {node_type} with CATEGORY: {category}")

        node_type = node_map[node_id]["type"].split("/")[-1]
        tags = NODE_TAGS.get(node_type, {})
        print(f"🔖 Tags for {node_type}: {tags}")

        if tags.get("run_early", False):
            if hasattr(inst, "build"):
                out = inst.build()
                built_outputs[inst] = out

                if "task_type" in out:
                    hyperparam_task_type = out["task_type"]
                if "num_classes" in out:
                    for node in node_instances.values():
                        if node.__class__.__name__ == "OutputLayer":
                            node.num_classes = out["num_classes"]
                if "sample_tensor" in out:
                    for node in node_instances.values():
                        if NODE_TAGS.get(node_map[node.graph_node_id]["type"], {}).get("requires_sample_tensor"):
                            node.sample_tensor = out["sample_tensor"]

    # Inject task_type into OutputLayer
    for node in node_instances.values():
        if node.__class__.__name__ == "OutputLayer" and hyperparam_task_type:
            node.task_type = hyperparam_task_type

    # Wire connections
    for src, tgt, input_name, tgt_id, tgt_slot in connections:
        output = built_outputs.get(src)
        inject_node_outputs(output, tgt, input_name, tgt_id)
        if hasattr(tgt, "on_input_connected"):
            tgt.on_input_connected(input_index=tgt_slot, source_node=src)

    # Build model
    model = None
    for node in node_instances.values():
        if node.__class__.__name__ == "OutputLayer":
            model = node.build()

    # Train
    trained_model = None
    app.trainer_nodes.clear()
    for node in node_instances.values():
        if node.__class__.__name__ == "TrainerNode":
            node.model = model
            node.graph_nodes = node_instances
            app.trainer_nodes.append(node)
            print(f"✅ Registered TrainerNode id={node.graph_node_id} in app.trainer_nodes")
            trained_model = node.build()


    # Run all post-train nodes (Save, Export, Evaluators, etc.)
    for node_id, node in node_instances.items():
        node_type = node_map[node_id]["type"].split("/")[-1]
        tags = NODE_TAGS.get(node_type, {})
        print(f"🔖 Tags for {node_type}: {tags}")

        if tags.get("run_after_train", False):
            print(f"🧪 Running post-train node: {node.__class__.__name__}")

            if hasattr(node, "model"):
                node.model = trained_model

            try:
                # ✅ Safely inspect which args build() accepts
                build_sig = inspect.signature(node.build)
                kwargs = {}

                if "model_out" in build_sig.parameters:
                    kwargs["model_out"] = trained_model

                if "task_type" in build_sig.parameters and hasattr(node, "task_type"):
                    kwargs["task_type"] = node.task_type

                node.build(**kwargs)

            except Exception as e:
                print(f"❌ Error running post-train node {node.__class__.__name__}: {e}")
                socketio.emit("toast", {
                    "message": f"❌ {node.__class__.__name__} failed: {e}"
                })

    return trained_model


def find_downstream_predictors(source_id):
    from app import app

    if not hasattr(app, "graph_links") or not hasattr(app, "graph_nodes"):
        print("⚠️ Graph not yet built. No downstream predictors available.")
        return []

    visited = set()
    queue = [source_id]
    result = []

    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)

        for link in app.graph_links:
            _, origin_id, _, target_id, _, _ = link
            if origin_id == current_id:
                queue.append(target_id)
                target_node = app.graph_nodes.get(target_id)
                if (
                    target_node and
                    hasattr(target_node, "set_feature_info") and
                    target_node not in result
                ):
                    result.append(target_node)

    return result


