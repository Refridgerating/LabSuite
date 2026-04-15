"""Custom exceptions used across LabSuite."""


class LabSuiteError(Exception):
    """Base exception for application errors."""


class ParseError(LabSuiteError):
    """Raised when a raw data file cannot be parsed."""


class RecipeError(LabSuiteError):
    """Raised when a preprocessing recipe is invalid."""


class WorkflowError(LabSuiteError):
    """Raised when a workflow cannot complete successfully."""
