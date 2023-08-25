from feature_extractor import FeatureExtractor
import pandas as pd


ext = FeatureExtractor(
    town_index_path="additional_data/town_20230808.csv",
    district_index_path="additional_data/district_20230808.csv",
    street_abbv_index_path="additional_data/geonimtype_20230808.csv",
    town_abbv_index_path="additional_data/subrf_20230808.csv"
)

a = ext.get_features("198325, г. Санкт-Петербург Кировский район дом 4 к.5")
b = ext.resolve_abbv("г. Санкт-Петербург Кировский район дом 4 к.5")
print(b)
print(a)