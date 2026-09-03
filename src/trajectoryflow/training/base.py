# std-lib imports
from abc import ABC, abstractmethod

# 3 party imports

# package imports
from trajectoryflow.models.base import BaseTrajectoryModel


class BaseTrainer(ABC):

    @abstractmethod
    def fit(self, model: BaseTrajectoryModel) -> None:
        pass