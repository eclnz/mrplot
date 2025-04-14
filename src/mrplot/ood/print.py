from typing import Dict, Any, List, Tuple, Optional, Callable, TypeVar, Set

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def print_tree(nodes: List[Tuple[int, str]], indent: str = "    ") -> None:
    """Print a tree structure from a list of (level, content) tuples.

    Args:
        nodes: List of tuples containing (indent_level, content)
        indent: The indentation string to use

    Example:
        nodes = [
            (0, "Root"),
            (1, "Child 1"),
            (2, "Grandchild 1"),
            (1, "Child 2"),
        ]
        print_tree(nodes)
    """
    if not nodes:
        return

    # First find the root nodes
    root_nodes = [i for i, (level, _) in enumerate(nodes) if level == 0]

    for root_idx, root_index in enumerate(root_nodes):
        root_level, root_content = nodes[root_index]
        print(root_content)

        # If this is the last root node, we need to handle it differently
        is_last_root = root_idx == len(root_nodes) - 1

        # Find all direct children of this root
        children_indices = []
        stop_idx = len(nodes) if is_last_root else root_nodes[root_idx + 1]
        for i in range(root_index + 1, stop_idx):
            if nodes[i][0] == 1:  # Direct child of root
                children_indices.append(i)

        # Print each child with appropriate prefix
        for child_idx, child_index in enumerate(children_indices):
            is_last_child = child_idx == len(children_indices) - 1
            prefix = "└── " if is_last_child else "├── "
            print(f"{indent}{prefix}{nodes[child_index][1]}")

            # Find all grandchildren of this child
            grandchildren_indices = []
            next_child_idx = (
                stop_idx
                if is_last_child or child_idx == len(children_indices) - 1
                else children_indices[child_idx + 1]
            )

            for i in range(child_index + 1, next_child_idx):
                if nodes[i][0] == 2:  # Grandchild (level 2)
                    grandchildren_indices.append(i)

            # Print each grandchild with appropriate prefix
            for gc_idx, gc_index in enumerate(grandchildren_indices):
                is_last_gc = gc_idx == len(grandchildren_indices) - 1
                if is_last_child:
                    prefix = "    └── " if is_last_gc else "    ├── "
                    print(f"{indent}{indent}{prefix}{nodes[gc_index][1]}")
                else:
                    prefix = "│   └── " if is_last_gc else "│   ├── "
                    print(f"{indent}{prefix}{nodes[gc_index][1]}")


def format_key_value(key: str, value: Any, separator: str = ": ") -> str:
    """Format a key-value pair as a string.

    Args:
        key: The key or label
        value: The value to display
        separator: The separator between key and value

    Returns:
        Formatted string like "key: value"
    """
    return f"{key}{separator}{value}"


def build_collection_tree(
    groups: Dict[K, Set[T]],
    group_formatter: Callable[[K, int], str],
    item_formatter: Optional[Callable[[T], List[str]]] = None,
    include_details: bool = False,
) -> List[Tuple[int, str]]:
    """Build a tree structure from a dictionary of groups containing collections.

    Args:
        groups: Dictionary mapping keys to sets of items
        group_formatter: Function to format group key and size into display string
        item_formatter: Optional function to format individual items into list of detail strings
        include_details: Whether to include item details in the tree

    Returns:
        List of (level, content) tuples representing the tree structure
    """
    nodes = []

    for key, items in groups.items():
        nodes.append((0, group_formatter(key, len(items))))

        if include_details and item_formatter:
            try:
                sorted_items = sorted(items)  # type: ignore
            except TypeError:
                sorted_items = list(items)

            for item in sorted_items:
                for detail in item_formatter(item):
                    nodes.append((1, detail))

    return nodes


def build_nested_tree(
    groups: Dict[K, Dict[V, Set[T]]],
    group_formatter: Callable[[K], str],
    subgroup_formatter: Callable[[V, int], str],
    item_formatter: Optional[Callable[[T], List[str]]] = None,
    include_details: bool = False,
) -> List[Tuple[int, str]]:
    """Build a tree structure from a nested dictionary of groups.

    Args:
        groups: Dictionary mapping keys to dictionaries of sets
        group_formatter: Function to format top-level group key
        subgroup_formatter: Function to format subgroup key and size
        item_formatter: Optional function to format individual items into list of detail strings
        include_details: Whether to include item details in the tree

    Returns:
        List of (level, content) tuples representing the tree structure
    """
    nodes = []

    for key, subgroups in groups.items():
        nodes.append((0, group_formatter(key)))

        for subkey, items in subgroups.items():
            nodes.append((1, subgroup_formatter(subkey, len(items))))

            if include_details and item_formatter:
                try:
                    sorted_items = sorted(items)  # type: ignore
                except TypeError:
                    sorted_items = list(items)

                for item in sorted_items:
                    for detail in item_formatter(item):
                        nodes.append((2, detail))

    return nodes
