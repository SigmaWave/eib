import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enum import Enum
from typing import Any, Dict, Tuple
import argparse
import json
import pandas as pd
import re
import requests
from tqdm import tqdm

from utils.logger import Logger


class JudgeLLM:
    class JudgeType(Enum):
        COVERAGE          = 0
        UNIQUENESS        = 1
        SEMANTIC_VALIDITY = 2
        CONSISTENCY       = 3

    JUDGE_MODEL_NAME: str = "qwen2.5:14b"
    LOCAL_API_URL: str    = "http://localhost:11434/api/generate"

    COMBINE_LLM_JUDGE_OUTPUTS_TEMPLATE_PATH: Path = \
      Path("components/combine_llm_judge_outputs_template.txt")
    FLAGGED_TRIPLET_FEEDBACK_PROMPT_PATH: Path    = \
      Path("components/flagged_triplet_feedback_prompt.txt")
    CONSISTENCY_PROMPT_PATH: Path       = Path("components/consistency_prompt.txt")
    COVERAGE_PROMPT_PATH: Path          = Path("components/coverage_prompt.txt")
    SEMANTIC_VALIDITY_PROMPT_PATH: Path = Path("components/semantic_validity_prompt.txt")
    UNIQUENESS_PROMPT_PATH: Path        = Path("components/uniqueness_prompt.txt")

    THRESHOLD_M: Dict[str, float] = {
        "hc_coverage_thresh"   : 0.75,
        "llm_coverage_thresh"  : 0.75,
        "hc_uniqueness_thresh" : 0.75,
        "llm_uniqueness_thresh": 0.75,
        "hc_semantic_thresh"   : 0.75,
        "llm_semantic_thresh"  : 0.75,
        "entropy_ratio_thresh" : 1.0}

    PARSE_JUDGE_OUTPUT_M: Dict[JudgeType, str] = {
        JudgeType.CONSISTENCY      : "LLM_Consistency_Score",
        JudgeType.COVERAGE         : "LLM_Coverage_Score",
        JudgeType.SEMANTIC_VALIDITY: "LLM_Semantic_Validity_Score",
        JudgeType.UNIQUENESS       : "LLM_Uniqueness_Score"}

    def __init__(self, triplets_path_: str):
        self.triplets_path: Path       = Path(triplets_path_)
        self.triplets_df: pd.DataFrame = pd.read_csv(self.triplets_path)
        self.judge_and_evaluate()

    @staticmethod
    def generate_local_response(model_name: str, prompt: str) -> str:
        payload = {"model": model_name, "prompt": prompt, "stream": False, "format": "json"}

        try:
            response = requests.post(JudgeLLM.LOCAL_API_URL, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes
            return response.json().get("response", "").strip()
        except requests.ConnectionError as e:
            Logger.error(f"Connection to Ollama failed: {e}. Is Ollama running?")
            raise # Re-raise the exception to stop the process
        except requests.HTTPError as e:
            Logger.error(f"Ollama API returned an error: {e}. Check model ID.")
            raise

    def judge_and_evaluate(self) -> None:
        for row_idx, triplets_row in tqdm(self.triplets_df.iterrows()):
            (coverage_json, uniqueness_json,
             consistency_json, semantic_validity_json) = JudgeLLM.judge(triplets_row)
            self.triplets_df.at[row_idx, "LLM judgments evaluation"] = JudgeLLM.evaluate_llm_judgments(
                coverage_json, uniqueness_json, semantic_validity_json, consistency_json)
        Logger.info("Finished judgment. Saving new csv to disk.")
        self.llm_judgments_evaluation_path = (
            self.triplets_path.parent
            / f"{self.triplets_path.stem}_JudgeLLM_judgments_evaluation{self.triplets_path.suffix}")
        self.triplets_df.dropna().to_csv(self.llm_judgments_evaluation_path)
        Logger.info(f"Finished saving csv to disk at: {self.llm_judgments_evaluation_path}")

    @staticmethod
    def judge(df_row_: pd.Series) -> Tuple[str, str, str, str]:
        (coverage_prompt, uniqueness_prompt,
         consistency_prompt, semantic_validity_prompt) = JudgeLLM.get_judge_prompts(df_row_)
        (coverage_judge_response, uniqueness_judge_response,
         consistency_judge_response, semantic_validity_judge_response) = \
             JudgeLLM.get_judge_responses_local(JudgeLLM.JUDGE_MODEL_NAME,
                                                coverage_prompt,
                                                uniqueness_prompt,
                                                consistency_prompt,
                                                semantic_validity_prompt)
        (coverage_json, uniqueness_json,
         consistency_json, semantic_validity_json) = JudgeLLM.extract_judge_json(
             coverage_judge_response, uniqueness_judge_response,
             consistency_judge_response, semantic_validity_judge_response)

        return coverage_json, uniqueness_json, consistency_json, semantic_validity_json

    @staticmethod
    def evaluate_llm_judgments(coverage_json_         : Dict[str, str],
                               uniqueness_json_       : Dict[str, str],
                               semantic_validity_json_: Dict[str, str],
                               consistency_json_      : Dict[str, str]):
        return (f"Judge LLM Coverage Score: {coverage_json_['LLM_Coverage_Score']}\n"
                f"Judge LLM Coverage Score Explanation: {coverage_json_['Explanation']}\n"
                 "\n"
                f"Judge LLM Uniqueness Score: {uniqueness_json_['LLM_Uniqueness_Score']}\n"
                f"Judge LLM Uniqueness Explanation: {uniqueness_json_['Explanation']}\n"
                 "\n"
                 "Judge LLM Semantic Validity Score:"
                f"{semantic_validity_json_['LLM_Semantic_Validity_Score']}\n"
                f"Judge LLM Semantic Validity Explanation:"
                f" {semantic_validity_json_['Explanation']}\n"
                "\n"
                f"Judge LLM Consistency Score: {consistency_json_['LLM_Consistency_Score']}\n"
                f"Judge LLM Consistency Explanation: {consistency_json_['Explanation']}\n")

    @staticmethod
    def get_judge_prompts(df_row_: pd.Series):
        article_text = df_row_['text']
        triplets = df_row_['output_triplets']

        coverage_prompt = JudgeLLM.build_coverage_prompt(article_text, triplets)
        uniqueness_prompt = JudgeLLM.build_uniqueness_prompt(triplets)
        consistency_prompt = JudgeLLM.build_consistency_prompt(triplets)
        semantic_validity_prompt = JudgeLLM.build_semantic_validity_prompt(article_text, triplets)

        return coverage_prompt, uniqueness_prompt, consistency_prompt, semantic_validity_prompt

    @staticmethod
    def get_judge_responses_local(model_name_: str,
                                  coverage_prompt_: str,
                                  uniqueness_prompt_: str,
                                  consistency_prompt_: str,
                                  semantic_validity_prompt_: str):
        prompts = {
            "coverage": coverage_prompt_,
            "uniqueness": uniqueness_prompt_,
            "consistency": consistency_prompt_,
            "semantic_validity": semantic_validity_prompt_
        }

        responses = {}
        for key, prompt in prompts.items():
            # You might want to wrap this in a try/except block specific to each call
            # to handle individual inference failures more gracefully, but for hardcoding
            # the local call, we'll let the error bubble up.
            responses[key] = JudgeLLM.generate_local_response(model_name_, prompt)

        return (responses["coverage"], responses["uniqueness"],
                responses["consistency"], responses["semantic_validity"])

    @staticmethod
    def get_judge_responses(client_,
                            coverage_prompt_: str,
                            uniqueness_prompt_: str,
                            consistency_prompt_: str,
                            semantic_validity_prompt_: str):
        coverage_judge_response = client_.models.generate_content(
                                model="gemini-2.5-flash", contents=coverage_prompt_)
        uniqueness_judge_response = client_.models.generate_content(
                                model="gemini-2.5-flash", contents=uniqueness_prompt_)
        consistency_judge_response = client_.models.generate_content(
                                model="gemini-2.5-flash", contents=consistency_prompt_)
        semantic_validity_judge_response = client_.models.generate_content(
                                model="gemini-2.5-flash", contents=semantic_validity_prompt_)

        return (coverage_judge_response, uniqueness_judge_response,
                consistency_judge_response, semantic_validity_judge_response)

    @staticmethod
    def extract_judge_json(
        coverage_judge_response_: str,
        uniqueness_judge_response_: str,
        consistency_judge_response_: str,
        semantic_validity_judge_response_: str):

        coverage_clean_text          = JudgeLLM.clean_json_block(
            coverage_judge_response_)
        uniqueness_clean_text        = JudgeLLM.clean_json_block(
            uniqueness_judge_response_)
        consistency_clean_text       = JudgeLLM.clean_json_block(
            consistency_judge_response_)
        semantic_validity_clean_text = JudgeLLM.clean_json_block(
            semantic_validity_judge_response_)

        _, _, coverage_json          = JudgeLLM.parse_judge_output(
            JudgeLLM.JudgeType.COVERAGE, coverage_clean_text)
        _, _, uniqueness_json        = JudgeLLM.parse_judge_output(
            JudgeLLM.JudgeType.UNIQUENESS, uniqueness_clean_text)
        _, _, consistency_json       = JudgeLLM.parse_judge_output(
            JudgeLLM.JudgeType.CONSISTENCY, consistency_clean_text)
        _, _, semantic_validity_json = JudgeLLM.parse_judge_output(
            JudgeLLM.JudgeType.SEMANTIC_VALIDITY, semantic_validity_clean_text)

        return coverage_json, uniqueness_json, consistency_json, semantic_validity_json

    @staticmethod
    def parse_judge_output(judge_type_: JudgeType,
                        judge_output_: str) -> Tuple[float, str, Dict[str, Any]]:
        try:
            # Pass the raw output to the improved cleaner first
            cleaned_json_string = JudgeLLM.clean_json_block(judge_output_)
            judge_json = json.loads(cleaned_json_string)
            explanation = judge_json["Explanation"]
            score = judge_json[JudgeLLM.PARSE_JUDGE_OUTPUT_M[judge_type_]]
            return score, explanation, judge_json
        except (json.JSONDecodeError, KeyError) as e:
            Logger.error(f"Failed to parse judge output for {judge_type_.name}. Error: {e}")
            Logger.error(f"Problematic output was: '{judge_output_[:200]}...'")
            raise e

    @staticmethod
    def get_llm_semantic_score(semantic_validity_score_: float, consistency_score_: float) -> float:
        return (semantic_validity_score_ + consistency_score_) / 2.0

    @staticmethod
    def should_flag_for_human_review(metric_dict_: Dict[str, float],
                                     ignore_hc_uniqueness_=False) -> True:
        if ((metric_dict_["hc_coverage"] <= JudgeLLM.THRESHOLD_M["hc_coverage_thresh"])
            and (metric_dict_["llm_coverage"] >= JudgeLLM.THRESHOLD_M["hc_coverage_thresh"])):
            return True

        if ((not ignore_hc_uniqueness_)
            and ((metric_dict_["hc_uniqueness"] <= JudgeLLM.THRESHOLD_M["hc_uniqueness_thresh"])
                  and (metric_dict_["llm_uniqueness"]
                       >= JudgeLLM.THRESHOLD_M["llm_uniqueness_thresh"]))):
            return True

        if ((metric_dict_["hc_semantic"] <= JudgeLLM.THRESHOLD_M["hc_semantic_thresh"])
            and (metric_dict_["llm_semantic"] >= JudgeLLM.THRESHOLD_M["llm_semantic_thresh"])):
            return True

        return False

    @staticmethod
    def populate_llm_metrics(metric_dict_: Dict[str, float],
                             llm_coverage_score_: float,
                             llm_uniqueness_score_: float,
                             llm_semantic_validity_score_: float,
                             llm_consistency_score_: float):
        metric_dict_["llm_coverage"] = llm_coverage_score_
        metric_dict_["llm_uniqueness"] = llm_uniqueness_score_
        metric_dict_["llm_semantic"] = JudgeLLM.get_llm_semantic_score(llm_semantic_validity_score_,
                                                                       llm_consistency_score_)

    @staticmethod
    def get_handcrafted_and_llm_alignment_status(metric_dict_: Dict[str, float],
                                                 ignore_hc_uniqueness_=False,
                                                 force_reprompt_=False):
        alignment_dict = {
          "is_coverage_sufficient": False,
          "is_unique": False,
          "is_semantically_valid": False,
          "is_entropy_ratio_good": False
        }
        flag_for_reprompt = False

        if ((metric_dict_["hc_coverage"] >= JudgeLLM.THRESHOLD_M["hc_coverage_thresh"])
            and (metric_dict_["llm_coverage"] >= JudgeLLM.THRESHOLD_M["hc_coverage_thresh"])):
            alignment_dict["is_coverage_sufficient"] = True
        else:
            """
            Cases:
            1. Both agree that the metric is below the threshold. We should definitely reprompt.
            2. LLM metric is less than the threshold, but the handcrafted one is above. Let's
            still reprompt.
            3. LLM metric is above the threshold, but the handcrafted one is below. This would
            ideally be caught by calling the should_flag_for_human_review function earlier.
            """
            flag_for_reprompt = True

        if not ignore_hc_uniqueness_:
            if ((metric_dict_["hc_uniqueness"] >= JudgeLLM.THRESHOLD_M["hc_uniqueness_thresh"])
                 and (metric_dict_["llm_uniqueness"]
                      >= JudgeLLM.THRESHOLD_M["llm_uniqueness_thresh"])):
                alignment_dict["is_unique"] = True
            else:
                flag_for_reprompt = True
        else:
            if metric_dict_["llm_uniqueness"] >= JudgeLLM.THRESHOLD_M["llm_uniqueness_thresh"]:
                alignment_dict["is_unique"] = True
            else:
                flag_for_reprompt = True

        if ((metric_dict_["hc_semantic"] >= JudgeLLM.THRESHOLD_M["hc_semantic_thresh"])
             and (metric_dict_["llm_semantic"] >= JudgeLLM.THRESHOLD_M["llm_semantic_thresh"])):
            alignment_dict["is_semantically_valid"] = True
        else:
            flag_for_reprompt = True

        if metric_dict_["entropy_ratio"] <= JudgeLLM.THRESHOLD_M["entropy_ratio_thresh"]:
            alignment_dict["is_entropy_ratio_good"] = True
        else:
            flag_for_reprompt = True

        if force_reprompt_:
            flag_for_reprompt = True

        return flag_for_reprompt, alignment_dict

    @staticmethod
    def build_coverage_prompt(article_text_: str, triplets_: str) -> str:
        with open(JudgeLLM.COVERAGE_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().format_map(locals())

    @staticmethod
    def build_flagged_triplet_feedback_prompt(
        article_text_: str,
        triplets_: str,
        alignment_dict_: Dict[str, str],
        judge_llm_output_: str) -> str:
        with open(JudgeLLM.FLAGGED_TRIPLET_FEEDBACK_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().format(
                article_text_=article_text_,
                triplets_=triplets_,
                judge_llm_output_=judge_llm_output_,
                **alignment_dict_
            )

    @staticmethod
    def build_uniqueness_prompt(triplets_: str) -> str:
        with open(JudgeLLM.UNIQUENESS_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().format_map(locals())

    @staticmethod
    def build_semantic_validity_prompt(article_text_: str, triplets_: str) -> str:
        with open(JudgeLLM.SEMANTIC_VALIDITY_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().format_map(locals())

    @staticmethod
    def build_consistency_prompt(triplets_: str) -> str:
        with open(JudgeLLM.CONSISTENCY_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().format_map(locals())

    @staticmethod
    def clean_json_block(text: str) -> str:
        # Find the first '{' and the last '}' to isolate the JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        # Fallback for simple markdown stripping if the above fails
        return text.strip().strip("```json").strip("```").strip()


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
    JudgeLLM(args.triplets_path)
