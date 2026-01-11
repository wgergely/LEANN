from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List

class BaseAnalysisProvider(ABC):
    """
    Abstract base class for analysis providers.
    Each provider implements language-specific dependency mapping and health checks.
    """

    @abstractmethod
    def bootstrap(self, project_root: Path, force: bool = False) -> bool:
        """
        Set up the analysis tool for the given project root.
        Returns True if successful.
        """
        pass

    @abstractmethod
    def get_file_context(self, abs_file_path: Path) -> Dict[str, Any]:
        """
        Return rich dependency metadata for a specific file.
        Expected keys: 'dependencies', 'dependents', 'closure', 'external', etc.
        """
        pass

    @abstractmethod
    def get_project_summary(self) -> Dict[str, Any]:
        """
        Return a high-level summary of the project's structure/health.
        """
        pass
