import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import Counter
# from google.genai.errors import ServerError
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from summa.commons import build_graph as _build_graph
from summa.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank
from summa.preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from summa.summarizer import _set_graph_edge_weights, _add_scores_to_sentences, summarize
from thefuzz import fuzz
from tqdm import tqdm
from typing import Callable
from unionfind import unionfind
import argparse
import ast
import json
import numpy as np
import pandas as pd
import owlready2 as ol2
import re
import requests

from components.judgeLLM import JudgeLLM
from utils.data_utils import _safe_eval, _norm_triplet
from utils.logger import Logger
from utils.timer import Timer


class OntologyParams:
    SUBJECT_NAME_INDEX = 0
    SUBJECT_CATEGORY_INDEX = 1
    RELATION_NAME_INDEX = 2
    # Index 3 is skipped/padding in typical 6-element tuples
    OBJECT_NAME_INDEX = 4
    OBJECT_CATEGORY_INDEX = 5

def normalize_row_triplets(triplets_raw):
    t_list = _safe_eval(triplets_raw)

    if not isinstance(t_list, list) or len(t_list) == 0:
        return "[]"


    cleaned_list = []
    for t in t_list:
        d = _norm_triplet(t)
        if d:
            # Map normalized dict back to the 6-element tuple
            tuple_fmt = (
                d['sub'],
                d['sub_type'],
                d['rel'],
                None, # Index 3 is padding/unused
                d['obj'],
                d['obj_type']
            )
            cleaned_list.append(tuple_fmt)
        else:
            print(f"DEBUG DROP (Structure Fail): {t}")
    return str(cleaned_list)

class MetricsComputator:
    # RDF file contains predefined classes and relationship restrictions (e.g. allowed domains and ranges)
    ONTOLOGY_PATH = 'components/fin_ontology.rdf' #update this to your path
    EMBED_MODEL_ONTO_GLOBAL = SentenceTransformer('all-MiniLM-L6-v2')

    LOCAL_API_URL: str = "http://localhost:11434/api/generate"
    LOCAL_MODEL_NAME: str = "qwen2.5:14b"
    @staticmethod
    def generate_local_response(model_name: str, prompt: str) -> str:
        """Helper to call the local Ollama API."""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        response = requests.post(MetricsComputator.LOCAL_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def __init__(self, triplets_path_: str):
        self.triplets_path: Path       = Path(triplets_path_)

        Logger.info(f"Loading triplets from {self.triplets_path}...")
        try:
            self.triplets_df: pd.DataFrame = pd.read_csv(
                self.triplets_path,
                on_bad_lines='skip',
                engine='python'
            )

            if 'output_triplets' in self.triplets_df.columns:
                initial_count = len(self.triplets_df)

                Logger.info("Normalizing triplets and filtering invalid rows...")
                self.triplets_df['output_triplets'] = self.triplets_df['output_triplets'].apply(normalize_row_triplets)
                self.triplets_df = self.triplets_df[self.triplets_df['output_triplets'] != "[]"]
                final_count = len(self.triplets_df)
                Logger.info(f"Dropped {initial_count - final_count} rows due to invalid triplets. Remaining: {final_count}")
                self.triplets_df.reset_index(drop=True, inplace=True)

            else:
                Logger.error("Column 'output_triplets' not found in CSV.")

        except Exception as e:
            Logger.error(f"Failed to load or parse CSV: {e}")
            raise e

        self.output_path: Path = (
            self.triplets_path.parent
            / f"{self.triplets_path.stem}_JudgeLLM_metrics_computation{self.triplets_path.suffix}")

        self.world = ol2.World(filename='quadstore.sqlite3')
        self.ONTO_GLOBAL = self.world.get_ontology(MetricsComputator.ONTOLOGY_PATH).load()
        self.PROP_EMBEDDINGS_GLOBAL = MetricsComputator.get_embedded_object_properties(
            self.ONTO_GLOBAL, embed_model_func=MetricsComputator.EMBED_MODEL_ONTO_GLOBAL.encode)
        self.compute_metrics()

    def compute_metrics(self,
                        ignore_hc_uniqueness_: bool=False,
                        force_reprompt_ : bool=False,
                        verbose_ : bool=False):
        if "LLM judgments evaluation" not in self.triplets_df.columns:
            self.triplets_df["LLM judgments evaluation"] = ""
        if "Revised triplets" not in self.triplets_df.columns:
            self.triplets_df["Revised triplets"] = ""

        Logger.info(f"processing {len(self.triplets_df)} amount of triplets")
        for row_idx, triplets_row in tqdm(self.triplets_df.iterrows(), total=(len(self.triplets_df))):
            if len(str(triplets_row["LLM judgments evaluation"])) > 0 and not pd.isna(triplets_row["LLM judgments evaluation"]):
                Logger.info(f"Skipping already processed row {row_idx}")
                continue

            success = False
            try:
                (coverage_json, uniqueness_json,
                consistency_json, semantic_validity_json) = JudgeLLM.judge(triplets_row)

                self.triplets_df.at[row_idx, "LLM judgments evaluation"] = JudgeLLM.evaluate_llm_judgments(
                    coverage_json, uniqueness_json, semantic_validity_json, consistency_json)

                metric_dict = MetricsComputator.get_metric_dict(
                    self.ONTO_GLOBAL,
                    triplets_row,
                    self.PROP_EMBEDDINGS_GLOBAL,
                    MetricsComputator.EMBED_MODEL_ONTO_GLOBAL.encode,
                    coverage_json["LLM_Coverage_Score"],
                    uniqueness_json["LLM_Uniqueness_Score"],
                    consistency_json["LLM_Consistency_Score"],
                    semantic_validity_json["LLM_Semantic_Validity_Score"])

                flag_for_human_review = JudgeLLM.should_flag_for_human_review(metric_dict, ignore_hc_uniqueness_)

                if flag_for_human_review:
                    Logger.info(f"{triplets_row.get('ticker','UNK')} {triplets_row.get('title','DOC')} requires human review\n{self.triplets_df.iloc[row_idx]['output_triplets']=}\n{self.triplets_df.iloc[row_idx]['Revised triplets']}")
                    self.triplets_df.at[row_idx, "Revised triplets"] = "Require Human Review"
                else:
                    flag_for_reprompt, alignment_dict = JudgeLLM.get_handcrafted_and_llm_alignment_status(
                        metric_dict,
                        ignore_hc_uniqueness_=ignore_hc_uniqueness_,
                        force_reprompt_=force_reprompt_)

                    if flag_for_reprompt:
                        Logger.info(f"'{triplets_row.get('ticker','UNK')}' '{triplets_row.get('title','DOC')}' requires reprompt")

                        if not isinstance(alignment_dict, dict) or "is_coverage_sufficient" not in alignment_dict:
                            Logger.error(f"Row {row_idx}: alignment_dict is corrupt or missing key: {alignment_dict}")
                            raise TypeError(f"Corrupt alignment_dict: {alignment_dict}")

                        feedback_prompt = JudgeLLM.build_flagged_triplet_feedback_prompt(
                            triplets_row['text'],
                            triplets_row['output_triplets'],
                            alignment_dict,
                            triplets_row["LLM judgments evaluation"])
                        if verbose_:
                            Logger.info(f"Feedback prompt:\n{feedback_prompt}\n END Feedback prompt")

                        revised_triplets_text = MetricsComputator.generate_local_response(
                            MetricsComputator.LOCAL_MODEL_NAME,
                            feedback_prompt
                        )

                        try:
                            cleaned_json_text = JudgeLLM.clean_json_block(revised_triplets_text)
                            llm_response_dict = json.loads(cleaned_json_text)
                            triplets_string = llm_response_dict.get("Revised_Triplets")

                            if not triplets_string or not isinstance(triplets_string, str):
                                Logger.error(f"Row {row_idx}: 'Revised_Triplets' key missing or invalid in LLM response.")
                                self.triplets_df.at[row_idx, "Revised triplets"] = "Error: Missing 'Revised_Triplets' key"
                            else:
                                final_triplets_list = ast.literal_eval(triplets_string)
                                # Store as string representation to maintain consistency with other rows
                                self.triplets_df.at[row_idx, "Revised triplets"] = str(final_triplets_list)
                                Logger.info(f"revised triplets : \n{self.triplets_df.iloc[row_idx]['output_triplets']=}\n{self.triplets_df.iloc[row_idx]['Revised triplets']}")

                        except json.JSONDecodeError as e:
                            Logger.error(f"JSON Decoding Error on row {row_idx}: {e}")
                            self.triplets_df.at[row_idx, "Revised triplets"] = "JSON Error on Reprompt"

                        except (ValueError, SyntaxError) as ast_e:
                            Logger.error(f"Row {row_idx}: Failed to parse the 'Revised_Triplets' string: {ast_e}")
                            self.triplets_df.at[row_idx, "Revised triplets"] = "Error: Corrupt triplets string"

                        Logger.info(f"'{triplets_row.get('ticker','UNK')}' '{triplets_row.get('title','DOC')}' revised triplets updated.")
                    else:
                        Logger.info(f"'{triplets_row.get('ticker','UNK')}' '{triplets_row.get('title','DOC')}' is good enough")
                        self.triplets_df.at[row_idx, "Revised triplets"] = "Evaluation meets expectation"

                if row_idx % 10 == 0:
                    self.save_partial()

                success = True
            except requests.exceptions.RequestException as e:
                Logger.error(f"Ollama/Request Error on row {row_idx}: {e}. Check if Ollama is running.")
            except Exception as e:
                Logger.error(f"Unexpected error on row {row_idx}: {e}")

            if not success:
                Logger.error(f"Failed to process row {row_idx}. Skipping...")

        Logger.info("Finished judgment. Saving final csv to disk.")
        self.triplets_df.to_csv(self.output_path, index=False)
        Logger.info(f"Finished saving csv to disk at: {self.output_path}")

    def save_partial(self):
        self.triplets_df.to_csv(self.output_path, index=False)
        Logger.info(f"Saved partial progress to {self.output_path}")

    @staticmethod
    def _get_all_sentence_scores(text, language="english", split=False, scores=False, additional_stopwords=None, drop_last_sentence=False):
        if not isinstance(text, str):
            return [] if split else ""

        sentences = _clean_text_by_sentences(text, language, additional_stopwords)
        if drop_last_sentence and len(sentences) > 1:
            sentences = sentences[:-1]
        graph = _build_graph([sentence.token for sentence in sentences])
        _set_graph_edge_weights(graph)
        _remove_unreachable_nodes(graph)
        if len(graph.nodes()) == 0:
            return [] if split else ""
        pagerank_scores = _pagerank(graph)
        _add_scores_to_sentences(sentences, pagerank_scores)
        return sentences

    @staticmethod
    def get_topic_coverage_score(triplets_row, thresh=0.8):
        scale = 100
        # Use safe eval here as well in case normalization wasn't applied or data is mixed
        triplet_list = _safe_eval(triplets_row['output_triplets'])
        article_text = triplets_row['text']

        if not isinstance(article_text, str):
            return 0.0

        cleaned_sentences = _clean_text_by_sentences(article_text)
        cleaned_sentences = [sentence.token for sentence in cleaned_sentences]
        new_article = " ".join(cleaned_sentences)
        article_tokens = new_article.split(" ")
        unique_article_tokens = list(dict.fromkeys(article_tokens).keys())

        triplet_sentences = MetricsComputator.convert_triplet_list_to_sentences(triplet_list, second_idx=4)
        triplet_article = " ".join(triplet_sentences)
        cleaned_triplet_sentences = _clean_text_by_sentences(triplet_article)
        cleaned_triplet_sentences = [sentence.token for sentence in cleaned_triplet_sentences]
        new_triplet_article = " ".join(cleaned_triplet_sentences)
        triplet_tokens = new_triplet_article.split(" ")
        unique_triplet_tokens = list(dict.fromkeys(triplet_tokens).keys())

        if not unique_article_tokens:
            return 0.0

        valid_counts = 0
        for article_token in unique_article_tokens:
            for triplet_token in unique_triplet_tokens:
                ratio = fuzz.ratio(article_token, triplet_token) / scale
                if ratio >= thresh:
                    valid_counts += 1
                    break

        return valid_counts / len(unique_article_tokens)

    @staticmethod
    def create_metric_dict_shell():
        return {
            "hc_coverage": 0.0,
            "llm_coverage": 0.0,
            "hc_uniqueness": 0.0,
            "llm_uniqueness": 0.0,
            "hc_semantic": 0.0,
            "llm_semantic": 0.0,
            "entropy_ratio": 0.0
        }

    @staticmethod
    def gather_all_hc_metrics(onto_, triplets_row_, metric_dict_, prop_embeddings_, embedding_func_):
        entropy_ratio = MetricsComputator.get_entropy_ratio_for_row(triplets_row_)
        uniqueness_score, _, coverage_score = \
            MetricsComputator.get_metrics_for_one_row(triplets_row_)
        onto_score, _ = MetricsComputator.get_ontological_consistency_score(
            onto_, triplets_row_, prop_embeddings_, embedding_func_)

        metric_dict_["hc_coverage"] = coverage_score
        metric_dict_["hc_uniqueness"] = uniqueness_score
        metric_dict_["hc_semantic"] = onto_score
        metric_dict_["entropy_ratio"] = entropy_ratio

    @staticmethod
    def get_ontological_consistency_score(onto_, triplets_row_: pd.Series,
                                          prop_embeddings_, embedding_func_: Callable):
        (num_valid_triplets, num_total_triplets,
         included_valid_triplets, included_triplets) = \
             MetricsComputator.create_ontologies_from_triplets(onto_,
                                                               triplets_row_,
                                                               prop_embeddings_,
                                                               embedding_func_,
                                                               thresh_=0.75)
        if num_total_triplets == 0:
            final_score = 0.0
        else:
            final_score = num_valid_triplets / num_total_triplets

        subscore = None
        if included_triplets > 0:
            subscore = included_valid_triplets / included_triplets

        return final_score, subscore

    @staticmethod
    def get_metrics_for_one_row(df_row, first_idx=0, rel_idx=2, second_idx=4, drop_last_sentence=True):
        article = df_row['text']

        # Handle empty text cases
        if not isinstance(article, str) or not article.strip():
            return 0.0, 0.0, 0.0

        sentence_info = MetricsComputator.get_sentence_weights(article, drop_last_sentence=drop_last_sentence)
        _sentence_list = [info[0] for info in sentence_info]
        _sentence_weights = [info[1] for info in sentence_info]

        if len(_sentence_list) == 0:
            cleaned = _clean_text_by_sentences(article)
            if not cleaned:
                return 0.0, 0.0, 0.0
            _sentence_list = [sentence.text for sentence in cleaned]
            _sentence_weights = [1/len(_sentence_list) for _ in _sentence_list]

        triplets_data = _safe_eval(df_row['output_triplets'])
        _triplet_list = MetricsComputator.convert_triplet_list_to_sentences(
            triplets_data, first_idx, rel_idx, second_idx)

        embedded_article_sentences_test = MetricsComputator.embed_sentence_list(
            _sentence_list, MetricsComputator.EMBED_MODEL_ONTO_GLOBAL)
        embedded_triplet_sentences_test = MetricsComputator.embed_sentence_list(
            _triplet_list, MetricsComputator.EMBED_MODEL_ONTO_GLOBAL)

        uniqueness_score = MetricsComputator.calculate_uniqueness_metric(
            embedded_triplet_sentences_test, thresh=0.7)

        uniqueness_pair_score = MetricsComputator.calculate_uniqueness_metric(
            embedded_triplet_sentences_test, thresh=0.7, use_pairs=True)

        coverage_score = MetricsComputator.calculate_coverage_metric(
            embedded_article_sentences_test, embedded_triplet_sentences_test, _sentence_weights)

        return uniqueness_score, uniqueness_pair_score, coverage_score

    @staticmethod
    def get_sentence_weights(text, drop_last_sentence=False):
        sentences = MetricsComputator._get_all_sentence_scores(text, drop_last_sentence=drop_last_sentence)
        if not sentences:
            return []
        sum_scores = sum([x.score for x in sentences])
        res = []
        if sum_scores == 0:
            # Fallback for 0 scores
            weight = 1.0 / len(sentences)
            for s in sentences:
                res.append((s.text, weight))
        else:
            for i in range(len(sentences)):
                sentences[i].score /= sum_scores
                res.append( (sentences[i].text, sentences[i].score) )
        return res

    @staticmethod
    def calculate_coverage_metric(embedded_src_sentences, embedded_trip_sentences, sentence_weights):
        if not embedded_trip_sentences or not embedded_src_sentences:
            return 0.0

        coverage_score = 0.0
        for idx in range(len(embedded_src_sentences)):
            src_sentence = embedded_src_sentences[idx]
            max_sim = -1.0
            for trip_sentence in embedded_trip_sentences:
                sim = cosine_similarity(src_sentence.unsqueeze(0).cpu(), trip_sentence.unsqueeze(0).cpu())
                max_sim = max(max_sim, sim[0].item())

            norm_max_sim = (max_sim + 1) / 2
            coverage_score += norm_max_sim * sentence_weights[idx]
        return coverage_score

    @staticmethod
    def calculate_uniqueness_metric(embedded_trip_sentences, use_pairs=False, thresh=0.9):
        n = len(embedded_trip_sentences)
        if n == 0: return 0.0

        uf = unionfind(n)
        num_diff_pairs = 0
        for i in range(n):
            for j in range(i+1, n):
                trip_1 = embedded_trip_sentences[i]
                trip_2 = embedded_trip_sentences[j]
                sim = cosine_similarity(trip_1.unsqueeze(0).cpu(), trip_2.unsqueeze(0).cpu())
                sim_val = sim[0].item()
                if sim_val >= thresh:
                    uf.unite(i, j)
                else:
                    num_diff_pairs += 1

        unique_trips = len(uf.groups())
        total_pairs = n*(n-1) / 2

        if use_pairs:
            if total_pairs == 0:
                return 1.0
            return num_diff_pairs / total_pairs
        return unique_trips / n

    @staticmethod
    def get_metric_dict(onto_,
                        triplets_row_,
                        prop_embeddings_,
                        embedding_func_,
                        llm_coverage_score_,
                        llm_uniqueness_score_,
                        llm_consistency_score_,
                        llm_semantic_validity_score_):
        metric_dict = MetricsComputator.create_metric_dict_shell()
        MetricsComputator.gather_all_hc_metrics(onto_, triplets_row_, metric_dict, prop_embeddings_, embedding_func_)
        JudgeLLM.populate_llm_metrics(metric_dict,
                                      llm_coverage_score_,
                                      llm_uniqueness_score_,
                                      llm_consistency_score_,
                                      llm_semantic_validity_score_)
        return metric_dict

    @staticmethod
    def get_embedded_object_properties(onto, embed_model_func):
        res = []
        for prop in onto.object_properties():
            prop_name_cleaned = " ".join(prop.name.lower().split("_"))
            embedded_prop = embed_model_func(prop_name_cleaned)
            embed_tuple = (embedded_prop, prop.name)
            res.append(embed_tuple)
        return res

    @staticmethod
    def create_ontology_entity(onto, category, name):
        name = str(name)
        ent = None

        # Updated mapping: Maps both "Underscore" and "No-Underscore" inputs
        # to the specific Ontology Class (which we established has NO underscores)
        valid_cats = {
            "COMPANY": onto.COMPANY,
            "STOCKTICKER": onto.COMPANY,
            "STOCK_TICKER": onto.COMPANY,
            "DERIVATIVE": onto.DERIVATIVE,

            # Financial Entity
            "FINANCIALENTITY": onto.FINANCIALENTITY,
            "FINANCIAL_ENTITY": onto.FINANCIALENTITY,

            "CURRENCY": onto.CURRENCY,
            "BOND": onto.BOND,
            "PRODUCT": onto.PRODUCT,
            "EVENT": onto.EVENT,
            "SECTOR": onto.SECTOR,

            # Financial Instrument Info
            "FININSTRUMENTINFO": onto.FININSTRUMENTINFO,
            "FIN_INSTRUMENT_INFO": onto.FININSTRUMENTINFO,

            # Macro Indicator
            "MACROINDICATOR": onto.MACROINDICATOR,
            "MACRO_INDICATOR": onto.MACROINDICATOR,

            "CONCEPT": onto.CONCEPT,

            # Economic Indicator
            "ECONINDICATOR": onto.ECONINDICATOR,
            "ECON_INDICATOR": onto.ECONINDICATOR
        }

        cls_def = valid_cats.get(category)
        if cls_def:
            ent = cls_def(name)

        return ent

    @staticmethod
    def generate_dummy_embedding(text, dim=768, seed=None):
        if seed is None:
            seed = hash(text) % (2**32)
        rng = np.random.default_rng(seed)
        embedding = rng.uniform(low=-1.0, high=1.0, size=dim)
        embedding /= np.linalg.norm(embedding)
        return embedding

    @staticmethod
    def find_closest_prop_embedding(target_emb, embedding_list):
        all_embs = np.stack([emb for emb, _ in embedding_list])
        target_norm = target_emb / np.linalg.norm(target_emb)
        all_norms = all_embs / np.linalg.norm(all_embs, axis=1, keepdims=True)
        similarities = all_norms @ target_norm
        idx = np.argmax(similarities)
        return embedding_list[idx][1], similarities[idx]

    @staticmethod
    def get_closest_object_property(prop_embeddings_, embed_model_func_, relation_, thresh_):
        cleaned_relation = " ".join(relation_.lower().split("_"))
        relation_embedding = embed_model_func_(cleaned_relation)
        prop_name, sim_val = MetricsComputator.find_closest_prop_embedding(
            relation_embedding, prop_embeddings_)

        if sim_val >= thresh_:
            return prop_name, sim_val
        else:
            return relation_, None

    @staticmethod
    def is_triplet_domain_and_range_valid(subject, obj, domain, _range):
        in_domain = (len(domain) == 0)
        in_range = (len(_range) == 0)

        for _cls in subject.is_a:
            if _cls in domain:
                in_domain = True
                break
        for _cls in obj.is_a:
            if _cls in _range:
                in_range = True
                break

        return in_domain and in_range

    @staticmethod
    def create_ontologies_from_triplets(
        onto_, triplets_row_: pd.Series, prop_embeddings_, embedding_func_, thresh_):

        triplet_str = triplets_row_['output_triplets']
        triplet_list = _safe_eval(triplet_str)

        included_triplets = 0
        included_valid_triplets = 0
        prop_names = [prop.name for prop in onto_.object_properties()]
        num_valid_triplets = 0

        for triplet in triplet_list:
            if len(triplet) < 6: continue

            subject = MetricsComputator.create_ontology_entity(
                onto_,
                triplet[OntologyParams.SUBJECT_CATEGORY_INDEX],
                triplet[OntologyParams.SUBJECT_NAME_INDEX])
            obj = MetricsComputator.create_ontology_entity(
                onto_,
                triplet[OntologyParams.OBJECT_CATEGORY_INDEX],
                triplet[OntologyParams.OBJECT_NAME_INDEX])

            if subject == None or obj == None:
                continue

            relation_name = triplet[OntologyParams.RELATION_NAME_INDEX]

            prop_name, sim_val = MetricsComputator.get_closest_object_property(prop_embeddings_,
                                                                               embedding_func_,
                                                                               relation_name,
                                                                               thresh_)
            if prop_name not in prop_names:
                num_valid_triplets += 1
                continue

            included_triplets += 1

            # owlready2 property assignment
            try:
                prop_list = getattr(subject, prop_name)
                prop_list.append(obj)
                prop = onto_[prop_name]

                if MetricsComputator.is_triplet_domain_and_range_valid(
                    subject, obj, prop.domain, prop.range):
                    included_valid_triplets += 1
                    num_valid_triplets += 1
            except AttributeError:
                # Fallback if property doesn't exist on subject in the way we expect
                continue

        return num_valid_triplets, len(triplet_list), included_valid_triplets, included_triplets

    @staticmethod
    def get_entropy_ratio_for_row(df_row):
        triplet_list = _safe_eval(df_row['output_triplets'])
        article_text = df_row['text']

        if not isinstance(article_text, str):
            return 0.0

        cleaned_sentences = _clean_text_by_sentences(article_text)
        cleaned_sentences = [sentence.token for sentence in cleaned_sentences]
        new_article = " ".join(cleaned_sentences)

        triplet_sentences = MetricsComputator.convert_triplet_list_to_sentences(triplet_list, second_idx=4)
        triplet_article = " ".join(triplet_sentences)
        cleaned_triplet_sentences = _clean_text_by_sentences(triplet_article)
        cleaned_triplet_sentences = [sentence.token for sentence in cleaned_triplet_sentences]
        new_triplet_article = " ".join(cleaned_triplet_sentences)

        test_entropy_val, _ = MetricsComputator.calc_entropy(new_article)
        if test_entropy_val == 0:
            return 0.0

        triplet_entropy_val, _ = MetricsComputator.calc_entropy(new_triplet_article)

        return triplet_entropy_val / test_entropy_val

    @staticmethod
    def embed_sentence_list(sentence_list, model):
        if not sentence_list: return []
        embedding_list = []
        for sentence in sentence_list:
            embedding = model.encode(sentence, convert_to_tensor=True)
            embedding_list.append(embedding)
        return embedding_list

    @staticmethod
    def calc_entropy(cleaned_text, use_weights=False, weights=None):
        if not cleaned_text.strip():
            return 0.0, []

        tokens = cleaned_text.split(" ")
        token_ctr = Counter(tokens)
        count_arr = [count for count in token_ctr.values()]
        comb_arr = [(token, count) for token,count in zip(token_ctr.keys(), token_ctr.values())]

        total_counts = sum(count_arr)
        if total_counts == 0: return 0.0, comb_arr

        # Normalize to probability dist
        count_arr = [c / total_counts for c in count_arr]

        entropy_val = entropy(count_arr, base=2)
        return entropy_val, comb_arr

    @staticmethod
    def convert_triplet_list_to_sentences(triplet_list, first_idx=0, rel_idx=2, second_idx=3):
        sentence_list = []
        if not triplet_list: return []

        for triplet in triplet_list:
            if len(triplet) > max(first_idx, rel_idx, second_idx):
                sentence_list.append(MetricsComputator.convert_triplet_to_sentence(
                    triplet, first_idx, rel_idx, second_idx))

        return sentence_list

    @staticmethod
    def convert_triplet_to_sentence(triplet, first_idx, rel_idx, second_idx):
        entity_one = str(triplet[first_idx])
        relation = " ".join(str(triplet[rel_idx]).split("_"))
        entity_two = str(triplet[second_idx])

        return entity_one + " " + relation + " " + entity_two + "."


def parse_args():
    parser = argparse.ArgumentParser(description="JudgeLLM with specified model and data path.")

    parser.add_argument(
        "--triplets-path",
        type=str,
        help="Choose which triplets to execute JudgeLLM.",
        required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Logger.setup_file("metrics_computator")
    MetricsComputator(args.triplets_path)