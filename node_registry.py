# node_registry.py
NODE_CLASS_MAP = {}
NODE_TAGS = {}  # optional per-node behavior metadata

def register_node(name=None, category=None, tags=None):
    def decorator(cls):
        node_name = name or cls.__name__
        cls.TITLE = node_name
        cls.CATEGORY = category or "Uncategorized"
        NODE_CLASS_MAP[node_name] = cls
        NODE_TAGS[node_name] = tags or {}
        return cls
    return decorator
