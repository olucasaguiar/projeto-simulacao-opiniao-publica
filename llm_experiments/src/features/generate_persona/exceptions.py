class IBGEDataError(Exception):
    """Base exception for IBGE data provider errors."""

    pass


class RaceDistributionError(IBGEDataError):
    """Raised when the race distribution data cannot be fetched or parsed."""

    pass


class InflationDataError(IBGEDataError):
    """Raised when the latest inflation rate cannot be fetched or parsed."""

    pass


class GenderDistributionError(IBGEDataError):
    """Raised when gender distribution data cannot be fetched or parsed."""

    pass


class RegionDistributionError(IBGEDataError):
    """Raised when region distribution data cannot be fetched or parsed."""

    pass


class UrbanRuralDistributionError(IBGEDataError):
    """Raised when urban/rural distribution data cannot be fetched or parsed."""

    pass


class ReligionDistributionError(IBGEDataError):
    """Raised when religion distribution data cannot be fetched or parsed."""

    pass


class EmploymentDistributionError(IBGEDataError):
    """Raised when employment distribution data cannot be fetched or parsed."""

    pass


class EducationDistributionError(IBGEDataError):
    """Raised when education distribution data cannot be fetched or parsed."""

    pass


class IncomeDistributionError(IBGEDataError):
    """Raised when income distribution data cannot be fetched or parsed."""

    pass


class HealthAssessmentDistributionError(IBGEDataError):
    """Raised when health assessment distribution data cannot be fetched or parsed."""

    pass


class ChronicDiseaseDistributionError(IBGEDataError):
    """Raised when chronic disease distribution data cannot be fetched or parsed."""

    pass


class MaritalStatusDistributionError(IBGEDataError):
    """Raised when marital status distribution data cannot be fetched or parsed."""

    pass
