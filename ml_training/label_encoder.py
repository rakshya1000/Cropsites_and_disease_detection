"""
Requirement 2 & 6: Assign numeric labels to every class, and encode labels.

Each unique (crop, class) pair gets its own integer label — e.g. "Healthy"
under Maize and "Healthy" under Potato are different labels, since they are
visually and biologically distinct classes even though they share a name.
With 7 crops and 29 total class folders, this produces 29 labels (0-28).
"""


class LabelEncoder:
    def __init__(self, samples):
        """
        samples: List[Tuple[filepath, crop_name, class_name]] as returned by
                 data_loader.scan_dataset()
        """
        combos = sorted({f"{crop}__{cls}" for _, crop, cls in samples})
        self.label_to_index = {label: idx for idx, label in enumerate(combos)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def encode(self, crop, class_name):
        """Returns the integer label for a given (crop, class) pair."""
        key = f"{crop}__{class_name}"
        return self.label_to_index[key]

    def decode(self, index):
        """Returns the "Crop__Class" string for a given integer label."""
        return self.index_to_label[index]

    @property
    def num_classes(self):
        return len(self.label_to_index)

    def as_dict(self):
        """Returns {index: "Crop__Class"} — useful for saving to JSON."""
        return dict(self.index_to_label)
