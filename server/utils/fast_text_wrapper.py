import os.path
import pickle
import warnings

import numpy as np
import pandas as pd
from gensim.models import FastText
from razdel import tokenize

from helpers.feature_extractor import FeatureExtractor

warnings.filterwarnings("ignore")


class FastTextWrapper:
    def __init__(self,
                 model_path: str,
                 town_index_path: str,
                 district_index_path: str,
                 street_abbv_index_path: str,
                 town_abbv_index_path: str,
                 building_index_path: str,
                 area_type_index_path: str
                 ) -> None:

        self.ext = FeatureExtractor(
            town_index_path=town_index_path,
            district_index_path=district_index_path,
            street_abbv_index_path=street_abbv_index_path,
            town_abbv_index_path=town_abbv_index_path,
            area_type_index_path=area_type_index_path
        )
        self.building = pd.read_csv(building_index_path)
        print("Buildings data loaded")
        self.building = self.building[self.building["is_actual"] == True]
        self.model = FastText.load(model_path)
        print("Model loaded")
        self.__build_index()

    def __build_index(self):
        self.vector_index = []

        if os.path.exists("model/vector_index_dump.bin"):
            print("Loading index from dump")
            self.vector_index = pickle.load(open("model/vector_index_dump.bin", "rb"))
            return

        print("Building index")

        for index, row in self.building.iterrows():
            tokens = list(tokenize(self.ext.clear_text(row['full_address'])))
            tokens = [_.text for _ in tokens]
            predict = np.array([self.model.wv[token] for token in tokens])
            predict = np.mean(predict, axis=0)
            predict = predict / np.linalg.norm(predict)
            self.vector_index.append((row['id'], predict))

        print("Dumping index to file")
        pickle.dump(self.vector_index, open("model/vector_index_dump.bin", "wb"))

    def _get_top_n(self, text: str, n: int = 10):
        vector = self.model.wv[text]
        tokens = list(tokenize(text))
        tokens = [_.text for _ in tokens]
        vector = np.array([self.model.wv[token] for token in tokens])
        vector = np.mean(vector, axis=0)
        vector = vector / np.linalg.norm(vector).reshape(1, -1)
        max_sym = -1
        best_id = 0
        step = 10000

        all_syms = np.ones((1, 1))

        for ind in range(0, len(self.vector_index), step):
            vectors = np.array([i[1] for i in self.vector_index[ind: ind + step]]).T
            syms = vector @ vectors
            all_syms = np.hstack([all_syms, syms])
            best_local_ind = np.argmax(syms)
            sym = syms[0][best_local_ind]
            if sym > max_sym:
                max_sym = sym
                best_id = self.vector_index[ind + best_local_ind][0]
        all_syms = all_syms[:, 1:]
        top_k = np.argpartition(all_syms[0], -n)[-n:].tolist()
        top_k = [(self.vector_index[i][0], all_syms[0][i]) for i in top_k]

        return best_id, max_sym, top_k

    def predict(self, text: str, n: int = 10):
        best_id, max_sym, top_k = self._get_top_n(self.ext.clear_text(text), 50)
        top_k.sort(key=lambda a: a[1], reverse=True)
        top_k_inds = [i[0] for i in top_k]
        map_global_to_local = {top_k_inds[i]: i for i in range(len(top_k_inds))}

        data_top_k_inds = self.building[self.building['id'].isin(top_k_inds)]
        data_top_k_inds[data_top_k_inds["liter"].notna()]["liter"] = data_top_k_inds[data_top_k_inds["liter"].notna()][
            "liter"].apply(lambda x: x.lower())
        features = self.ext.get_features(text)
        if features["structure"] is not None:
            data_top_k_inds["structure"] = data_top_k_inds["full_address"].apply(
                lambda x: self.ext.get_features(x)['structure'])
        for feature in features:
            if features[feature] is not None and \
                    feature not in ["src_text", "preproc_text", "street", "structure", "country"]:
                data_top_k_inds = data_top_k_inds[
                    (data_top_k_inds[feature] == features[feature]) | (data_top_k_inds[feature].isna())]
        if len(data_top_k_inds) != 0:
            data_top_k_inds['sim'] = data_top_k_inds['id'].apply(lambda x: top_k[map_global_to_local[x]][1])
            best_id = data_top_k_inds['id'].iloc[data_top_k_inds['sim'].argmax()]

            result = [(row["id"], row["sim"]) for ind, row in data_top_k_inds.iterrows()]
            result.sort(key=lambda a: a[1], reverse=True)

            return result[:n]

        return top_k[:n]

    def get_address_from_id(self, id: int):
        return self.building[self.building['id'] == id]['full_address']

# wrap = FastTextWrapper(
#     model_path="/work/hack/512_emb_fast_text.model",
#     town_index_path="/work/hack/additional_data/town_20230808.csv",
#     district_index_path="/work/hack/additional_data/district_20230808.csv",
#     street_abbv_index_path="/work/hack/additional_data/geonimtype_20230808.csv",
#     town_abbv_index_path="/work/hack/additional_data/subrf_20230808.csv",
#     building_index_path="/work/hack/additional_data/building_20230808.csv"
# )

# print(wrap.predict("Орджоникидзе, д.59"))
