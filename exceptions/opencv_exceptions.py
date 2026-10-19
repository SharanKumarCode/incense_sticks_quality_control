class InspectionError(Exception):
    pass

class NoStickDetectedError(InspectionError):
    pass

class InvalidGeometryError(InspectionError):
    pass

class AnalysisError(InspectionError):
    def __init__(self, analysis_name: str, message: str) -> None:
        self.analysis_name = analysis_name

        super().__init__(
            f"{analysis_name} analysis failed: {message}"
        )

