# load_all_nodes.py
import os, importlib

def load_all_nodes():
    node_folder = "custom_nodes"  # adjust to match your path
    for file in os.listdir(node_folder):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = file[:-3]
            importlib.import_module(f"{node_folder}.{module_name}")
