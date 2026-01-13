"""Node management and control logic for the PowerSched environment."""

from src.config import CORES_PER_NODE


def adjust_nodes(action_type, action_magnitude, nodes, cores_available, env_print):
    """
    Adjust nodes based on action: turn nodes on or off.

    Args:
        action_type: 0=decrease, 1=maintain, 2=increase
        action_magnitude: Number of nodes to change
        nodes: Array of node states
        cores_available: Array of available cores per node
        env_print: Print function for logging

    Returns:
        Number of nodes changed
    """
    num_node_changes = 0

    # Adjust nodes based on action
    if action_type == 0:  # Decrease number of available nodes
        env_print(f"   >>> turning OFF up to {action_magnitude} nodes")
        nodes_modified = 0
        for i in range(len(nodes)):
            # Find idle nodes (no jobs running)
            if nodes[i] == 0 and cores_available[i] == CORES_PER_NODE:
                nodes[i] = -1  # Turn off
                cores_available[i] = 0  # No cores available on off nodes
                nodes_modified += 1
                num_node_changes += 1
                if nodes_modified == action_magnitude:
                    break
    elif action_type == 1:
        env_print(f"   >>> Not touching any nodes")
        pass  # maintain node count = do nothing
    elif action_type == 2:  # Increase number of available nodes
        env_print(f"   >>> turning ON up to {action_magnitude} nodes")
        nodes_modified = 0
        for i in range(len(nodes)):
            if nodes[i] == -1:  # Find off node
                nodes[i] = 0  # Turn on
                cores_available[i] = CORES_PER_NODE  # Reset cores to full availability
                nodes_modified += 1
                num_node_changes += 1
                if nodes_modified == action_magnitude:
                    break

    return num_node_changes
