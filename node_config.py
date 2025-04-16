# node_config.py
from node_registry import NODE_CLASS_MAP, NODE_TAGS

def get_node_config():
    config = []
    for name, cls in NODE_CLASS_MAP.items():
        config.append({
            "name": name,
            "title": getattr(cls, "TITLE", name),
            "category": getattr(cls, "CATEGORY", "Uncategorized"),
            "class": cls.__name__,
            "widgets": getattr(cls, "widgets", []),
            "inputs": getattr(cls, "inputs", []),     # ✅ Add this
            "outputs": getattr(cls, "outputs", []),   # ✅ And this
            "description": NODE_TAGS.get(name, {}).get("description", ""),
            "size": getattr(cls, "size", [210, 120])
        })
    return config
