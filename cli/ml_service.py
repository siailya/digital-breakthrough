import asyncio

from fast_text_wrapper import FastTextWrapper


class MLService:
    def __init__(self, model_path):
        self.model_wrapper: FastTextWrapper = None
        self.load_fasttext_model(model_path)

    def load_fasttext_model(self, model_path: str):
        print("Model loading")
        self.model_wrapper = FastTextWrapper(
            model_path,
            town_index_path="data/additional_data/town_20230808.csv",
            district_index_path="data/additional_data/district_20230808.csv",
            street_abbv_index_path="data/additional_data/geonimtype_20230808.csv",
            town_abbv_index_path="data/additional_data/subrf_20230808.csv",
            building_index_path="data/additional_data/building_20230808.csv",
            area_type_index_path="data/additional_data/areatype_20230808.csv"
        )

    def _search_addresses(self, address: str, limit: int):
        predictions = self.model_wrapper.predict(address, limit)
        results = []

        for p in predictions:
            results.append({
                "target_building_id": p[0],
                "target_address": self.model_wrapper.get_address_from_id(p[0]).values[0],
                "confident": p[1]
            })

        return results

    def search_addresses(self, address: str, limit: int = 10):
        return self._search_addresses(address, limit)
