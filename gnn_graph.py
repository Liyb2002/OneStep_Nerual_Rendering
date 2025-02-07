import torch
from torch_geometric.data import HeteroData

class StrokeGraph(HeteroData):
    def __init__(self, all_edges_matrix, intersection_matrix):
        super(StrokeGraph, self).__init__()

        num_strokes = all_edges_matrix.shape[0]

        # Convert to tensors
        all_edges_matrix = torch.tensor(all_edges_matrix, dtype=torch.float32)  # Shape: (num_strokes, 6)
        intersection_matrix = torch.tensor(intersection_matrix, dtype=torch.int64)  # Shape: (num_strokes, num_strokes)

        # Store stroke node features
        self['stroke'].x = all_edges_matrix  # Shape: (num_strokes, 6)

        # 1. Stroke order edges (0 → 1 → 2 → ... → num_strokes-1)
        order_edges = [[i, i + 1] for i in range(num_strokes - 1)]

        # 2. Intersection-based edges from intersection_matrix
        intersection_edges = [[src, dst] for src in range(num_strokes) for dst in range(num_strokes) 
                              if intersection_matrix[src, dst] == 1 and src != dst]  # No self-loops

        # Convert edges to tensors and store separately
        if order_edges:
            self['stroke', 'ordered_next', 'stroke'].edge_index = torch.tensor(order_edges, dtype=torch.long).t().contiguous()

        if intersection_edges:
            self['stroke', 'connected_to', 'stroke'].edge_index = torch.tensor(intersection_edges, dtype=torch.long).t().contiguous()

