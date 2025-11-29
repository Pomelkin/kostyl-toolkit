from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseScheduler(ABC):
    """Base class for learning rate schedulers."""

    def state_dict(self) -> dict[str, Any]:
        """Get the state as a state dictionary."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["optimizer", "scheduler_values"]
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state from a state dictionary."""
        self.__dict__.update(state_dict)
        return

    def __getstate__(self) -> dict[str, Any]:
        """Get the state for pickling."""
        return self.state_dict()

    def __setstate__(self, state) -> None:
        """Set the state for unpickling."""
        self.load_state_dict(state)
        return

    @abstractmethod
    def step(self, it: int) -> None | float:
        """
        Update the scheduler state. This is a no-op for most schedulers.

        Args:
            it (int): The current iteration step.

        Returns:
            The updated value for the parameter, if applicable.

        """
        pass

    @abstractmethod
    def current_value(self) -> dict[str, float]:
        """Get the current value of the parameter being scheduled."""
        pass
