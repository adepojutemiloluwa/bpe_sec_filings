"""BPE merge representation."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Merge:
    """Represent a single BPE merge.

    Attributes:
        rank: Merge order (0 = first merge)
        left_id: Left token ID
        right_id: Right token ID
        new_token_id: Token ID for the merged token
        frequency: Frequency at time of merge
    """

    rank: int
    left_id: int
    right_id: int
    new_token_id: int
    frequency: int

    def to_dict(self) -> Dict:
        """Convert merge to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "rank": self.rank,
            "left": self.left_id,
            "right": self.right_id,
            "new_token_id": self.new_token_id,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Merge":
        """Create merge from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Merge instance.
        """
        return cls(
            rank=data["rank"],
            left_id=data["left"],
            right_id=data["right"],
            new_token_id=data["new_token_id"],
            frequency=data["frequency"],
        )


class MergeTable:
    """Manage the table of learned BPE merges."""

    def __init__(self):
        """Initialize merge table."""
        self.merges: List[Merge] = []
        self.pair_to_rank: Dict[tuple, int] = {}

    def add_merge(
        self,
        left_id: int,
        right_id: int,
        new_token_id: int,
        frequency: int,
    ) -> None:
        """Add a merge to the table.

        Args:
            left_id: Left token ID.
            right_id: Right token ID.
            new_token_id: New token ID.
            frequency: Pair frequency.
        """
        rank = len(self.merges)
        merge = Merge(rank, left_id, right_id, new_token_id, frequency)
        self.merges.append(merge)
        self.pair_to_rank[(left_id, right_id)] = rank

    def get_rank(self, left_id: int, right_id: int) -> int:
        """Get merge rank for a pair.

        Args:
            left_id: Left token ID.
            right_id: Right token ID.

        Returns:
            Merge rank, or -1 if pair not merged.
        """
        return self.pair_to_rank.get((left_id, right_id), -1)

    def get_merge(self, rank: int) -> Merge:
        """Get merge by rank.

        Args:
            rank: Merge rank.

        Returns:
            Merge instance.

        Raises:
            IndexError: If rank is invalid.
        """
        return self.merges[rank]

    def size(self) -> int:
        """Get number of merges.

        Returns:
            Number of merges.
        """
        return len(self.merges)

    def to_list(self) -> List[Dict]:
        """Convert merges to list of dictionaries.

        Returns:
            List of merge dictionaries.
        """
        return [merge.to_dict() for merge in self.merges]

    @classmethod
    def from_list(cls, data: List[Dict]) -> "MergeTable":
        """Load merge table from list.

        Args:
            data: List of merge dictionaries.

        Returns:
            MergeTable instance.
        """
        table = cls()
        for merge_data in data:
            merge = Merge.from_dict(merge_data)
            table.merges.append(merge)
            table.pair_to_rank[(merge.left_id, merge.right_id)] = merge.rank
        return table
