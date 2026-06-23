from typing import List, Dict, Union

from settings import settings


class SidraQueryBuilder:
    def __init__(self, agregado_id: Union[int, str]):
        self.agregado_id = str(agregado_id)
        self.variables: List[str] = []
        self.periods: List[str] = []
        self.locations: Dict[str, List[str]] = {}
        self.classifications: Dict[str, List[str]] = {}

    def select_variables(self, *var_ids: Union[int, str]):
        self.variables.extend(str(v) for v in var_ids)
        return self

    def select_periods(self, *periods: Union[int, str]):
        self.periods.extend(str(p) for p in periods)
        return self

    def select_locations(self, level: str, codes: List[str] = None):
        self.locations[level] = codes or [settings.sidra.defaults.all_sentinel]
        return self

    def select_classification(
        self, class_id: Union[int, str], categories: List[str] = None
    ):
        self.classifications[str(class_id)] = categories or [
            settings.sidra.defaults.all_sentinel
        ]
        return self

    def build_path(self) -> str:
        # Defaults if none provided
        periods_str = (
            "|".join(self.periods)
            if self.periods
            else settings.sidra.defaults.all_sentinel
        )
        variables_str = (
            "|".join(self.variables)
            if self.variables
            else settings.sidra.defaults.all_sentinel
        )

        path = f"/agregados/{self.agregado_id}/periodos/{periods_str}/variaveis/{variables_str}"

        # Build query parameters for localities and classifications
        query_params = []

        if self.locations:
            # Format: level[codes] e.g. N3[all] or N3[11,12]
            locs = []
            for level, codes in self.locations.items():
                codes_str = ",".join(codes)
                locs.append(f"{level}[{codes_str}]")
            query_params.append(f"localidades={'|'.join(locs)}")
        else:
            # Default to BR[all] if nothing is specified
            query_params.append(
                f"localidades=BR[{settings.sidra.defaults.all_sentinel}]"
            )

        if self.classifications:
            # Format: class_id[categories] e.g. 2[all] or 2[1,2]
            classes = []
            for class_id, categories in self.classifications.items():
                cats_str = ",".join(categories)
                classes.append(f"{class_id}[{cats_str}]")
            query_params.append(f"classificacao={'|'.join(classes)}")

        query_string = "&".join(query_params)
        return f"{path}?{query_string}"
