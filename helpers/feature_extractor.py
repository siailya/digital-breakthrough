from natasha import AddrExtractor, MorphVocab
from typing import Dict
import pandas as pd


NATASHA_TO_DOMAIN_TYPE_MAPPING = {
    "бульвар": "street",
    "город": "municipality_id",
    "деревня": "municipality_id",
    "дом": "house",
    "корпус": "corpus",
    "набережная": "street",
    "переулок": "street",
    "площадь": "street",
    "посёлок": "municipality_id",
    "проезд": "street",
    "проспект": "street",
    "район": "district_id",
    "село": "municipality_id",
    "строение": "structure",
    "улица": "street",  
    "шоссе": "street",
    "индекс": "post_index"
}


class FeatureExtractor:
    # Надо добавить лемматизацию Кировского -> Кировский
    # и работу с is_updated / is_actual
    def __init__(self,
                town_index_path: str,
                district_index_path: str,
                street_abbv_index_path: str,
                town_abbv_index_path: str
                ) -> None:
        
        morph_vocab = MorphVocab()
        self.extractor = AddrExtractor(morph=morph_vocab)      
        
        self.town_index = pd.read_csv(town_index_path)
        self.district_index = pd.read_csv(district_index_path)
        street_abbv_index = pd.read_csv(street_abbv_index_path).dropna()
        town_abbv_index = pd.read_csv(town_abbv_index_path).dropna()
        self.abbv_map = {row["short_name"]: row["name"] for ind, row in street_abbv_index.iterrows()}
        town_abbv_map = {row["short_name"]: row["name"] for ind, row in town_abbv_index.iterrows()}
        self.abbv_map.update(town_abbv_map)
        custom_map = {
            "г.": "город",
            "гор.": "город",
            "л.": "литера",
            "к.": "корпус",
            "д.": "дом"
        }
        self.abbv_map.update(custom_map)
    
    
    def resolve_abbv(self, text: str) -> str:
        for i in self.abbv_map:
            text = text.replace(i, self.abbv_map[i] + " ")
        return text
        
        
    def get_features(self, text: str) -> Dict:
        result = {
            "district_id": None,
            "house": None,
            "corpus": None,
            "liter": None,
            "municipality_id": None,
            "src_text": text,
            "street": None,
            "post_index": None
        }
        
        text = self.resolve_abbv(text)
        
        matches = self.extractor.find(text)
        
        if matches is None:
            return result
        
        for j in matches.fact.parts:
            
            if j.type is None:
                continue
            
            elif NATASHA_TO_DOMAIN_TYPE_MAPPING[j.type] == "municipality_id":
                
                if self.town_index['name'].str.contains(j.value).sum() > 0:
                    result[NATASHA_TO_DOMAIN_TYPE_MAPPING[j.type]] = self.town_index[self.town_index['name'].str.contains(j.value)]['id'].iloc[0]
                continue
                    
            elif NATASHA_TO_DOMAIN_TYPE_MAPPING[j.type] == "district_id":
                if self.district_index['name'].str.contains(j.value).sum() > 0:
                    result[NATASHA_TO_DOMAIN_TYPE_MAPPING[j.type]] = self.district_index[self.district_index['name'] == j.value]['id'].iloc[0] 
                continue
            
            result[NATASHA_TO_DOMAIN_TYPE_MAPPING[j.type]] = j.value 
            
        return result