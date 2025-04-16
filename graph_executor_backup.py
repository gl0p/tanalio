from custom_nodes import test_evaluator_node, single_predict, add_node, residual_merge_node, concat_node, export_model_node, save_model_node, autocast, autopermute_node, autoreshape_node, autoflatten_node, dropout_node, batchnorm_node, conv2d_layer, separableconv2d_node, lstm_node, maxpool2d_layer, reshape, residual, hyper_param_node, trainer_node, load_json_node, load_image_node, load_csv_node, dense_node, output_layer_node  # import your custom nodes
import inspect
from sockets import socketio

NODE_CLASS_MAP = {
    # 📥 Data
    "Data/Load Images": load_image_node.LoadImages,
    "Data/Load CSV": load_csv_node.LoadCSV,
    "Data/Load JSON": load_json_node.LoadJSON,

    # 🧠 Model Layers
    "Model/Dense Layer": dense_node.DenseLayer,
    "Model/Output Layer": output_layer_node.OutputLayer,
    "Model/Conv2D": conv2d.Conv2DLayer,
    "Model/SeparableConv2D": separableconv2d_node.SeparableConv2D,
    "Model/LSTM": lstm_node.LSTM,
    "Model/Dropout": dropout_node.DropoutLayer,
    "Model/BatchNorm1d": batchnorm_node.BatchNorm1dLayer,
    "Model/Residual Block": residual.ResidualBlock,

    # 🔄 Preprocessing
    "Preprocessing/Reshape": reshape.ReshapeLayer,
    "Preprocessing/MaxPool2D": maxpool2d.MaxPool2DLayer,
    "Preprocessing/AutoFlatten": autoflatten_node.AutoFlatten,
    "Preprocessing/AutoReshape": autoreshape_node.AutoReshape,
    "Preprocessing/AutoPermute": autopermute_node.AutoPermute,
    "Preprocessing/AutoCast": autocast.AutoCast,

    # 🏋️ Training
    "Training/Trainer": trainer_node.TrainerNode,
    "Training/Hyperparameters": hyper_param_node.HyperparamsNode,

    # Utils
    "Utils/Save Model": save_model_node.SaveModelNode,
    "Utils/Export Model": export_model_node.ExportModelNode,
    "Utils/Single Predict": single_predict.SinglePredictNode,
    "Utils/Test Evaluator": test_evaluator_node.TestEvaluatorNode,


    # Operations
    "Operations/Concat": concat_node.ConcatNode,
    "Operations/Add": add_node.AddNode,
    "Operations/Residual Merge": residual_merge_node.ResidualMergeNode

}

def build_model_from_graph(graph_json, app):
    nodes = graph_json["nodes"]
    links = graph_json["links"]

    node_instances = {}
    node_map = {}  # maps id to full node data
    link_map = {}  # maps link_id to [source_node, output_slot, target_node, input_slot]

    # Build link map
    for link in links:
        link_id, origin_id, origin_slot, target_id, target_slot, data_type = link
        link_map[link_id] = [origin_id, origin_slot, target_id, target_slot]

    # Step 1: Create all node instances
    for node in nodes:
        node_type = node["type"]
        node_id = node["id"]
        properties = node.get("properties", {})

        if node_type in NODE_CLASS_MAP:
            cls = NODE_CLASS_MAP[node_type]
            sig = inspect.signature(cls.__init__)
            valid_keys = list(sig.parameters.keys())[1:]
            filtered_props = {k: v for k, v in properties.items() if k in valid_keys}

            try:
                instance = cls(**filtered_props)
                instance.graph_node_id = node_id
                node_instances[node_id] = instance
                node_map[node_id] = node
            except TypeError as e:
                socketio.emit("node_error", {"node_id": node_id})
                socketio.emit("toast", {
                    "message": f"⚠️ Please select a dataset."
                })
                print(f"❌ Failed to instantiate {node_type}: {e}")
                continue

        else:
            socketio.emit("toast", {
                "message": f"You need to import {node_type} in executor "
            })
            print(f"❌ Unknown node type: {node_type}")

    # Step 2: Handle special node behaviors
    for node_id, instance in node_instances.items():
        node_type = node_map[node_id]["type"]

        # ✅ Inject into Output Layer
        if isinstance(instance, output_layer_node.OutputLayer):
            instance.graph_nodes = node_instances
            instance.links = link_map
            instance.final_node_id = node_id
            instance.graph_nodes_data = node_map


    # Step 3: Wire outputs to inputs based on links
    connections = []

    for link_id, (source_id, source_slot, target_id, target_slot) in link_map.items():
        source_node = node_instances[source_id]
        target_node = node_instances[target_id]
        target_node_data = node_map[target_id]

        # Safely extract input name
        input_name = None
        try:
            input_name = target_node_data["inputs"][target_slot]["name"]
        except (IndexError, KeyError):
            continue

        connections.append((source_node, target_node, input_name, target_id))

    # ✅ Build only data-producing nodes first
    # ✅ Build only data-producing nodes first
    built_outputs = {}
    for node_id, instance in node_instances.items():
        # Look ahead to see if user set a manual task_type (regression/classification)
        hyperparam_task_type = None
        for node in node_instances.values():
            if isinstance(node, hyper_param_node.HyperparamsNode):
                hyperparam_task_type = getattr(node, "task_type", None)
                if hyperparam_task_type == "auto":
                    hyperparam_task_type = None
                break  # only one HyperparamsNode is expected

        # Build only if the node has outgoing connections (i.e. used as a source)
        is_source = any(source_id == node_id for (_, source_id, _, _, _, _) in links)
        if is_source and hasattr(instance, "build") and instance.CATEGORY.startswith("Data"):
            if isinstance(instance, load_csv_node.LoadCSV):
                built_outputs[instance] = instance.build(task_type_override=hyperparam_task_type)

                # ✅ Inject CSV-detected task_type into OutputLayer and HyperparamsNode if needed
                csv_task_type = built_outputs[instance].get("task_type")
                if csv_task_type and csv_task_type != "auto":
                    for node in node_instances.values():
                        if isinstance(node, hyper_param_node.HyperparamsNode) and getattr(node, "task_type",
                                                                                          "auto") == "auto":
                            print(f"🧠 [Auto] Injecting task_type = {csv_task_type} from CSV into HyperparamsNode")
                            node.task_type = csv_task_type
                        if isinstance(node, output_layer_node.OutputLayer):
                            print(f"🧠 [Auto] Injecting task_type = {csv_task_type} into OutputLayer")
                            node.task_type = csv_task_type

                # 🎯 Inject num_classes into OutputLayer early
                num_classes = built_outputs[instance].get("num_classes")
                if num_classes:
                    for node in node_instances.values():
                        if isinstance(node, output_layer_node.OutputLayer):
                            node.num_classes = num_classes
                            print(f"🎯 [Auto] Injected num_classes = {num_classes} into OutputLayer")

                # 🎯 Extract and send feature info to SinglePredictNode
                selected_target = node_map[node_id].get("properties", {}).get("target_column")
                if hasattr(instance, "get_property"):
                    selected_target = instance.get_property("target_column")

                columns = getattr(instance, "columns", [])
                feature_names = [col for col in columns if col != selected_target]
                print(f"SELECTED TARGET: {selected_target}")
                print(f"FEATURE NAMES: {feature_names}")

                norm_info = {
                    "input_mean": getattr(instance, "feature_mean", None),
                    "input_std": getattr(instance, "feature_std", None),
                    "output_mean": getattr(instance, "output_mean", None),
                    "output_std": getattr(instance, "output_std", None)
                }

                for node in node_instances.values():
                    if isinstance(node, single_predict.SinglePredictNode):
                        node.set_feature_info(
                            feature_names,
                            input_mean=norm_info["input_mean"],
                            input_std=norm_info["input_std"],
                            output_mean=norm_info["output_mean"],
                            output_std=norm_info["output_std"]
                        )
                        print(f"EMITTING FEATURE NAMES TO PREDICTOR: {feature_names} ")
                        socketio.emit("single_predict_columns", {
                            "node_id": node.graph_node_id,
                            "columns": feature_names
                        })

            elif isinstance(instance, load_image_node.LoadImages):
                # ✅ NEW: handle annotation_task from frontend
                annotation_task = node_map[node_id].get("properties", {}).get("annotation_task", "instances")
                instance.annotation_task = annotation_task
                instance.task_type = annotation_task
                built_outputs[instance] = instance.build()

                # 🧠 Inject task_type into HyperparamsNode and OutputLayer
                for node in node_instances.values():
                    if isinstance(node, hyper_param_node.HyperparamsNode) and getattr(node, "task_type",
                                                                                      "auto") == "auto":
                        node.task_type = annotation_task
                        print(f"🧠 [Auto] Injecting task_type = {annotation_task} from LoadImages into HyperparamsNode")
                    if isinstance(node, output_layer_node.OutputLayer):
                        node.task_type = annotation_task
                        print(f"🧠 [Auto] Injecting task_type = {annotation_task} into OutputLayer")

                # 🎯 Inject num_classes into OutputLayer and update UI
                num_classes = built_outputs[instance].get("num_classes")
                if num_classes:
                    for node in node_instances.values():
                        if isinstance(node, output_layer_node.OutputLayer):
                            node.num_classes = num_classes
                            print(f"🎯 Injected num_classes = {num_classes} into OutputLayer")

                            # 🔁 Emit UI sync
                            socketio.emit("property_update", {
                                "node_id": node.graph_node_id,
                                "property": "out_features",
                                "value": num_classes
                            })

            else:
                built_outputs[instance] = instance.build()

        # Explicitly build HyperparamsNode before linking
        if isinstance(instance, hyper_param_node.HyperparamsNode):
            built_outputs[instance] = instance.build()
            print(f"🔧 Built HyperparamsNode: {built_outputs[instance]}")

            # 🧠 Force-inject task_type into OutputLayer (even if not directly wired)
            task_type = built_outputs[instance].get("task_type")
            if task_type and task_type != "auto":
                for node in node_instances.values():
                    if isinstance(node, output_layer_node.OutputLayer):
                        setattr(node, "task_type", task_type)
                        print(f"🧠 [Force] Injected task_type = {task_type} into OutputLayer")

    print("🔌 Connections:")
    for s, t, i, id in connections:
        print(f"  {s.__class__.__name__} → {t.__class__.__name__}.{i} {id}")

    # ✅ Now wire everything
    for source_node, target_node, input_name, target_id in connections:
        if source_node not in built_outputs:
            continue  # Skip model nodes like Dense/Flatten — they'll be handled by OutputLayer
        output = built_outputs[source_node]
        print(f"🧵 Handling link from {source_node.__class__.__name__} to {target_node.__class__.__name__}.{input_name}")
        print(f"📦 Output keys from source: {list(output.keys())}")

        if isinstance(target_node, single_predict.SinglePredictNode) and "sample_tensor" in output:
            setattr(target_node, "sample_tensor", output["sample_tensor"])
            print(f"📦 Passed sample_tensor to SinglePredictNode")

        # 🔍 If it's a dict and contains out_features, auto-inject it
        if isinstance(output, dict):
            val = output.get(input_name)

            if isinstance(val, str):
                print(f"⚠️ Skipping string injection for {input_name}: {val}")
                continue

            if val is not None:
                setattr(target_node, input_name, val)

                if input_name == "in" and hasattr(target_node, "set_input_shape"):
                    target_node.set_input_shape(val)

                if hasattr(target_node, "on_input_connected"):
                    target_node.on_input_connected(input_index=target_slot, source_node=source_node)
                    print(f"🔌 Triggered on_input_connected on {target_node.__class__.__name__}")

            elif input_name == "in" and "sample_tensor" in output and hasattr(target_node, "set_input_shape"):
                target_node.set_input_shape(output["sample_tensor"])
                print(f"🧠 Fallback shape injected via sample_tensor into {target_node.__class__.__name__}")

            if input_name == "hyperparams":
                setattr(target_node, "hyperparams", output)
                print(f"🔧 Injected hyperparams into Trainer: {output}")
                continue

            if "out_features" in output:
                # optional widget sync here
                socketio.emit("property_update", {
                    "node_id": target_id,
                    "property": "in_features",
                    "value": output["out_features"]
                })
            if "task_type" in output:
                setattr(target_node, "task_type", output["task_type"])
                print(f"🧠 Injected task_type = {output['task_type']} into {target_node.__class__.__name__}")

            # 🎯 Inject num_classes into OutputLayer if available
            if "num_classes" in output:
                setattr(target_node, "num_classes", output["num_classes"])
                print(f"🎯 Injected num_classes = {output['num_classes']} into {target_node.__class__.__name__}")

            # ✅ Trigger input connection hook regardless
            if hasattr(target_node, "on_input_connected"):
                target_node.on_input_connected(input_index=target_slot, source_node=source_node)
                print(f"🔌 Triggered on_input_connected on {target_node.__class__.__name__}")

        else:
            setattr(target_node, input_name, output)
            print(
                f"🔗 Connected output of {source_node.__class__.__name__} to {target_node.__class__.__name__}.{input_name}")
            # Call on_input_connected if it exists
            if hasattr(target_node, "on_input_connected"):
                target_node.on_input_connected(input_index=target_slot, source_node=source_node)
                print(f"🔌 Triggered on_input_connected on {target_node.__class__.__name__}")

            # 🧠 Catch-all: if model tensor is passed to first preprocessing node
            if input_name == "in" and hasattr(target_node, "set_input_shape"):
                target_node.set_input_shape(output[input_name])
                print(f"🧠 Fallback: set_input_shape on {target_node.__class__.__name__}")
            elif isinstance(output, dict) and len(output) == 1:
                only_val = list(output.values())[0]
                if hasattr(target_node, "set_input_shape"):
                    print(f"📨 AutoInject single value into {target_node.__class__.__name__}")
                    target_node.set_input_shape(only_val)

    # 1. Build the model by triggering the Output Layer (to walk back and construct the full nn.Sequential)
    model = None
    for node in node_instances.values():
        if isinstance(node, output_layer_node.OutputLayer):
            model = node.build()

    # 2. Trigger the trainer, now that model and data are wired up
    trained_model = None

    for node in node_instances.values():
        if isinstance(node, trainer_node.TrainerNode):
            node.model = model  # Inject the built model into the trainer
            node.graph_nodes = node_instances
            # print(f"📦 Trainer received train: {node.train}")
            # print(f"📦 Trainer received val: {node.val}")
            trained_model = node.build()

    # 3. Save model if SaveModelNode exists
    for node in node_instances.values():
        if isinstance(node, save_model_node.SaveModelNode):
            node.model = trained_model
            node.build()

    # 4. Export model if ExportModelNode exists
    for node in node_instances.values():
        if isinstance(node, export_model_node.ExportModelNode):
            node.model = trained_model
            # Try to find the sample_tensor from LoadImages or any node that outputs it
            for source_node in built_outputs:
                output = built_outputs[source_node]
                if isinstance(output, dict) and "sample_tensor" in output:
                    node.sample_tensor = output["sample_tensor"]
                    break
            node.build()
    # 5. Register nodes for runtime interaction (Trainer + SinglePredict)
    app.trainer_nodes.clear()
    for node in node_instances.values():
        if isinstance(node, trainer_node.TrainerNode):
            app.trainer_nodes.append(node)
        if isinstance(node, single_predict.SinglePredictNode):
            node.model = trained_model  # ✅ link model to predictor
            app.trainer_nodes.append(node)
            print(f"🔁 Linked trained model to SinglePredictNode (id={node.graph_node_id})")

    # ✅ Trigger TestEvaluatorNode after training
    for node in node_instances.values():
        if isinstance(node, test_evaluator_node.TestEvaluatorNode):
            node.model = trained_model
            if hasattr(node, "test") and node.test is not None:
                print("🚀 Running TestEvaluatorNode with test data...")
                node.build()
            else:
                print("⚠️ Skipping TestEvaluatorNode: test data not available")

    return trained_model





