import httpx
from typing import List, Union
from .models import SidraMetadata, SidraVariableResponse
from .query_builder import SidraQueryBuilder

from settings import settings


class SidraClient:
    def __init__(
        self,
        base_url: str = settings.sidra.api.base_url,
        timeout: float = settings.sidra.api.timeout,
    ):
        self.base_url = base_url
        self.client = httpx.Client(timeout=timeout)

    def get_metadata(self, agregado_id: Union[int, str]) -> SidraMetadata:
        url = f"{self.base_url}/agregados/{agregado_id}/metadados"
        response = self.client.get(url)
        response.raise_for_status()
        return SidraMetadata.model_validate(response.json())

    def get_data(self, query: SidraQueryBuilder) -> List[SidraVariableResponse]:
        path = query.build_path()
        url = f"{self.base_url}{path}"
        response = self.client.get(url)
        response.raise_for_status()

        raw_data = response.json()
        if not raw_data or not isinstance(raw_data, list):
            return []

        return [SidraVariableResponse.model_validate(item) for item in raw_data]

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
