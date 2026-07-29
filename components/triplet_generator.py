# from google import genai
# import google.generativeai as genai  

import sys
from pathlib import Path

# Add project root to sys.path to enable imports from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from typing import Callable, Dict, List, Tuple, Any
import argparse
import ast
import boto3
import json
import os
import pandas as pd
import re

# from components.gemini_client import get_gemini_client  
from utils.logger import Logger
from utils.timer import Timer
from utils.pathlib_utils import ensure_dir

#MODIFS1
from anthropic import Anthropic


class TripletGenerator:
    MICRO_MODEL_ID: str   = "us.amazon.nova-micro-v1:0"
    LITE_MODEL_ID: str    = "us.amazon.nova-lite-v1:0"
    PRO_MODEL_ID: str     = "us.amazon.nova-pro-v1:0"
    PREMIER_MODEL_ID: str = "us.amazon.nova-premier-v1:0"
    TRIPLET_GENERATOR_PROMPT_PATH: Path = Path("components/tgp.txt")

    MODEL_ID_M: Dict[str, str]   = {"qwen2"           : "qwen2",
                                    "qwen2.5:14b"     : "qwen2.5:14b",
                                    "qwen2.5:32b"     : "qwen2.5:32b",
                                    "qwen2.5:latest"  : "qwen2.5:latest",
                                    "qwen3:4b"        : "qwen3:4b",
                                    "qwen3:14b"       : "qwen3:14b",
                                    "qwen3:30b"       : "qwen3:30b",
                                    "qwen3:latest"    : "qwen3:latest",
                                    "martain7r/finance-llama-8b:fp16": "martain7r/finance-llama-8b:fp16",
                                    "deepseek-r1:14b": "deepseek-r1:14b",
                                    "gemini-2.5-flash": "gemini-2.5-flash",
                                    "lite"            : LITE_MODEL_ID,
                                    "micro"           : MICRO_MODEL_ID,
                                    "pro"             : PRO_MODEL_ID,
                                    "premier"         : PREMIER_MODEL_ID,
                                    "claude-sonnet-5" : "claude-sonnet-5"}
    DATA_PATH_M: Dict[str, Path] = {"fnspid": Path("database/nasdaq_external_data.csv"),
                                    # "yf"    : Path("database/max_yahoo_data.csv"),
                                    "semi_cleaned_data": Path("database/nasdaq_semi_data_cleaned.csv")}
    
    EXECUTE_INFERENCE_M: Dict[str, Callable[[str, pd.DataFrame], None]]= {
        "qwen2": lambda model_id_, processed_df_, output_path_, text_col_:
            TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "qwen2.5:14b": lambda model_id_, processed_df_, output_path_, text_col_:
            TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "qwen2.5:32b": lambda model_id_, processed_df_, output_path_, text_col_:
            TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "qwen2.5:latest": lambda model_id_, processed_df_, output_path_, text_col_:
            TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "qwen3:4b": lambda model_id_, processed_df_, output_path_, text_col_:
            TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "qwen3:14b": lambda model_id_, processed_df_, output_path_, text_col_:
            TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "qwen3:30b": lambda model_id_, processed_df_, output_path_, text_col_:
            TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "martain7r/finance-llama-8b:fp16": lambda model_id_, processed_df_, output_path_, text_col_:
            TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "deepseek-r1:14b": lambda model_id_, processed_df_, output_path_, text_col_:
            TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "qwen3:latest": lambda model_id_, processed_df_, output_path_, text_col_:
            TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "gemini-2.5-flash": lambda model_id_, processed_df_, output_path_, text_col_:
         TripletGenerator._execute_inference_gemini(model_id_, processed_df_, output_path_, text_col_),
         "llama3.1:8b": lambda model_id_, processed_df_, output_path_, text_col_:
         TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
         "mistral:7b": lambda model_id_, processed_df_, output_path_, text_col_:
         TripletGenerator._execute_inference_local(model_id_, processed_df_, output_path_, text_col_),
        "lite"            : lambda model_id_, processed_df_, output_path_, text_col_:
         TripletGenerator._execute_inference_nova(model_id_, processed_df_, text_col_),
        "micro"           : lambda model_id_, processed_df_, output_path_, text_col_:
         TripletGenerator._execute_inference_nova(model_id_, processed_df_, text_col_),
        "pro"             : lambda model_id_, processed_df_, output_path_, text_col_:
         TripletGenerator._execute_inference_nova(model_id_, processed_df_, text_col_),
        "premier"         : lambda model_id_, processed_df_, output_path_, text_col_:
         TripletGenerator._execute_inference_nova(model_id_, processed_df_, text_col_),
         "claude-sonnet-5": lambda model_id_, processed_df_, output_path_, text_col_:
    TripletGenerator._execute_inference_claude(model_id_, processed_df_, output_path_, text_col_),}

    def __init__(self, model_name_: str, data_type_: str, text_column_: str = "text", start_date_: str = None, end_date_: str = None):
        self.model_id: str              = TripletGenerator.MODEL_ID_M[model_name_]
        self.text_column: str           = text_column_
        self.processed_df: pd.DataFrame = TripletGenerator.process_data(data_type_, text_column_, start_date_, end_date_)
        self.execute_inference()

    @staticmethod
    def process_data(data_type_: str, text_column_: str = "text", start_date_: str = None, end_date_: str = None) -> pd.DataFrame:
        read_options = {"encoding": "utf-8"}

        df = pd.read_csv(TripletGenerator.DATA_PATH_M[data_type_], **read_options)

        if data_type_ == "fnspid":
            df.rename(columns={"Article_title": "title", "Lsa_summary": "summary", "Url": "url",
                               "Date": "date", "Article": "text", "Stock_symbol": "ticker"}, inplace=True)
            df.drop(['Publisher', 'Author', 'Luhn_summary',
                     'Textrank_summary', 'Lexrank_summary'], axis='columns', inplace=True)
            Logger.info("Loading 'fnspid' dataset.")

        if data_type_ == "yf":
            df.rename(columns={'pub_date': 'date'}, inplace=True)

        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

        # Set default start date if not provided
        if start_date_ is None:
            start_date_ = "2024-01-01"
        start_date = pd.Timestamp(start_date_, tz="UTC")
        
        # Set end date if provided
        if end_date_ is None:
            df_filtered = df[df['date'] >= start_date].copy()
            Logger.info(f"--- Filtering Applied (From {start_date_} onwards & Semiconductors) ---")
        else:
            end_date = pd.Timestamp(end_date_, tz="UTC")
            df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
            Logger.info(f"--- Filtering Applied ({start_date_} to {end_date_} & Semiconductors) ---")

        SEMI_TICKERS = [
            'NVDA', 'AMD', 'INTC', 'QCOM', 'MU', 'TSM', 'TXN',
            'AMAT', 'ADI', 'MRVL', 'ON'
        ]

        if 'ticker' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['ticker'].isin(SEMI_TICKERS)].copy()

        Logger.info(f"Target Tickers: {SEMI_TICKERS}")
        Logger.info(f"Rows Remaining: {len(df_filtered)}")
        Logger.info(f"Using column for text content: '{text_column_}'")

        if 'ticker' in df_filtered.columns and not df_filtered.empty:
            df_filtered['period'] = df_filtered['date'].dt.to_period('M')

            ticker_stats = df_filtered.groupby('ticker').agg({
                'period': 'nunique',
                'title': 'count'
            }).reset_index()

            ticker_stats.rename(columns={'period': 'months_coverage', 'title': 'total_articles'}, inplace=True)
            ticker_stats.sort_values(by=['months_coverage', 'total_articles'], ascending=[False, False], inplace=True)

            Logger.info(f"\n--- Semiconductor Data Statistics ---")
            Logger.info(f"{'TICKER':<10} | {'MONTHS':<10} | {'TOTAL ARTICLES'}")
            Logger.info("-" * 40)

            for _, row in ticker_stats.iterrows():
                Logger.info(f"{row['ticker']:<10} | {row['months_coverage']:<10} | {row['total_articles']}")

            df_filtered.drop(columns=['period'], inplace=True)

        df = df_filtered
        print(f"{df.columns=}")
        # Ensure both text and summary columns exist and are filled
        if 'text' in df.columns:
            df['text'] = df['text'].fillna('')
        if 'summary' in df.columns:
            df['summary'] = df['summary'].fillna('')
        df['title'] = df['title'].fillna('')

        # Filter based on the selected column and title
        selected_col = text_column_ if text_column_ in df.columns else 'text'
        df = df[ (df[selected_col].str.len() > 50) | (df['title'].str.len() > 10) ].copy()

        Logger.info(f"--- Final Cleanup ---")
        Logger.info(f"Rows with valid text/title content: {len(df)}")

        print(f"{df.columns=}")
        df.sort_values(by='date', inplace=True)
        df.drop_duplicates(subset=['title'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['output_triplets'] = None
        Logger.info("Loaded and filtered CSV successfully.")
        return df

    def execute_inference(self) -> None:
        Logger.info("Finished inference. Saving new csv to disk.")
        output_path = ensure_dir(Path(f"output"))
        self.triplets_path: Path = output_path / f"triplets_{(args.model_name.replace('/', '_'))}_{args.data_type}_{self.text_column}.csv"
        TripletGenerator.EXECUTE_INFERENCE_M[self.model_id](self.model_id, self.processed_df, self.triplets_path, self.text_column)
        Logger.info(f"Finished saving csv to disk at: {self.triplets_path}")

    @staticmethod
    def _execute_inference_local(model_id_: str, processed_df_: pd.DataFrame, output_file: Path, text_col_: str = "text") -> None:
        Logger.info(f"Starting local inference with model: {model_id_}")
        TripletGenerator.local_do_inference(processed_df_, model_id_, output_file, text_col_)

    @staticmethod
    def _execute_inference_gemini(model_id_: str, processed_df_: pd.DataFrame, output_file: str, text_col_: str = "text") -> None:
        raise NotImplementedError("Gemini support requires components.gemini_client module. Please use local Ollama models instead.")
        # client = get_gemini_client()
        # TripletGenerator.gemini_do_inference(processed_df_, client, output_file, text_col_)

    @staticmethod
    def _execute_inference_claude(model_id_: str, processed_df_: pd.DataFrame, output_file: Path, text_col_: str = "text") -> None:
        client = Anthropic()  # reads ANTHROPIC_API_KEY from env automatically
        TripletGenerator.claude_do_inference(processed_df_, client, model_id_, output_file, text_col_)

    @staticmethod
    def claude_do_inference(processed_df_, client, model_id_: str, output_file: Path, text_col_: str = "text") -> None:
        num_inferences = len(processed_df_)
        for idx in tqdm(range(0, num_inferences)):
            row = processed_df_.iloc[idx]
            prompt = TripletGenerator.build_prompt(row, text_col_)
            try:
                response = client.messages.create(
                    model=model_id_,
                    max_tokens=1200,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = next((block.text for block in response.content if block.type == "text"), "")
                cleaned = TripletGenerator.sanitize_response(text)
                processed_df_.loc[idx, 'output_triplets'] = cleaned
            except Exception as e:
                Logger.info(f"Error executing inference for row {idx}: {e}")
                continue

            if (idx + 1) % 5 == 0:
                TripletGenerator._save_batch(processed_df_, max(0, idx - 4), idx + 1, output_file)

    @staticmethod
    def _execute_inference_nova(model_id_: str, processed_df_: pd.DataFrame, text_col_: str = "text") -> None:

        client = boto3.client("bedrock-runtime", region_name='us-east-2')

        Logger.info("Initialized client. Doing inference...")
        latency_ms, total_input_tokens, total_output_tokens = \
            TripletGenerator.nova_do_inference(processed_df_, client, model_id_, text_col_)
        Logger.info("Latency_ms:", latency_ms)
        Logger.info("Total input tokens:", total_input_tokens)
        Logger.info("Total output tokens:", total_output_tokens)

    @staticmethod
    def build_prompt(row: pd.Series, text_column_: str = "text") -> str:
        # Use the specified column if it exists, otherwise fall back to text
        column_to_use = text_column_ if text_column_ in row.index else "text"
        text_value = str(row[column_to_use]).strip()
        if text_value:
            return TripletGenerator._build_prompt(text_value)
        else:
            # Fall back to title + summary if selected column is empty
            fallback_text = str(row.get("title", "")) + " " + str(row.get("summary", ""))
            return TripletGenerator._build_prompt(fallback_text)

    @staticmethod
    def _build_prompt(input_text_: str) -> str:
        with open(TripletGenerator.TRIPLET_GENERATOR_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().format_map(locals())

    @staticmethod
    def gemini_do_inference(processed_df_: pd.DataFrame, genai_model_: Any, output_file: str, text_col_: str = "text") -> None:
        num_inferences = len(processed_df_)
        Logger.info(f"Processing {num_inferences} amount of articles.")
        last_saved = 0
        for idx in tqdm(range(4040, num_inferences)):
            with Timer("gemini inference"):
                prompt = TripletGenerator.build_prompt(processed_df_.iloc[idx], text_col_)

                response = TripletGenerator.sanitize_gemini_response(
                    genai_model_.models.generate_content(model="gemini-2.5-flash",
                                                        contents=prompt).text)
                processed_df_.loc[idx, 'output_triplets'] = response

                if (idx + 1) % 10 == 0 or (idx + 1) == num_inferences:
                    batch = processed_df_.iloc[last_saved:idx + 1]
                    header = not os.path.exists(output_file) if last_saved == 0 else False
                    batch.to_csv(output_file, mode='a', header=header, index=False)
                    Logger.info(f"{idx+1} processed to csv")
                    last_saved = idx + 1

    @staticmethod
    def sanitize_gemini_response(response_text: str):
        literal_eval_response: List[Tuple[str, ...]] = ast.literal_eval(response_text)
        return ", ".join(str(t) for t in literal_eval_response)

    @staticmethod
    def nova_do_inference(my_df, client, model_id_: str, text_col_: str = "text", debug_: bool=False) -> None:
        num_inferences = len(my_df)
        total_latency_ms = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        for idx in tqdm(range(0, num_inferences)):
            row = my_df.iloc[idx]
            prompt = TripletGenerator.build_prompt(row, text_col_)

            messages = [
                {"role": "user", "content": [{"text": prompt}]},
            ]

            inf_params = {"maxTokens": 1200, "topP": 0.1, "temperature": 0.0}

            model_response = client.converse(
                modelId=model_id_,
                messages=messages,
                inferenceConfig=inf_params)

            text = model_response["output"]["message"]["content"][0]["text"]
            my_df.loc['output_triplets', idx] = text

            total_latency_ms += model_response["metrics"]["latencyMs"]
            total_input_tokens += model_response["usage"]["inputTokens"]
            total_output_tokens += model_response["usage"]["outputTokens"]

            if debug_:
                Logger.info("\n[Full Response]")
                Logger.info(json.dumps(model_response, indent=2))

                Logger.info("\n[Response Content Text]")
                Logger.info(model_response["output"]["message"]["content"][0]["text"])

        return total_latency_ms, total_input_tokens, total_output_tokens

    @staticmethod
    def _save_batch(processed_df: pd.DataFrame, start_index: int, end_index: int, output_file: Path) -> None:
        batch = processed_df.iloc[start_index:end_index].copy()

        def is_valid(val):
            return val is not None and str(val).strip() != "" and str(val).lower() != "nan"

        original_count = len(batch)
        batch = batch[batch['output_triplets'].apply(is_valid)]
        filtered_count = len(batch)

        if filtered_count < original_count:
            Logger.info(f"Dropped {original_count - filtered_count} invalid/empty rows in this batch.")

        if batch.empty:
            return

        header = not os.path.exists(output_file)
        batch.to_csv(output_file, mode='a', header=header, index=False)
        Logger.info(f"Processed up to index {end_index - 1} and saved {len(batch)} rows to {output_file}.")

    @staticmethod
    def local_do_inference(processed_df_: pd.DataFrame, model_id_: str, output_file: Path, text_col_: str = "text") -> None:
        num_inferences = len(processed_df_)
        Logger.info(f"Processing {num_inferences} articles with local model {model_id_}.")

        existing_titles = set()
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                existing_df = pd.read_csv(output_file, usecols=['title'])
                existing_titles = set(existing_df['title'].astype(str).str.strip().values)
                Logger.info(f"Found {len(existing_titles)} already processed articles in {output_file}. These will be skipped.")
            except Exception as e:
                Logger.info(f"Warning: Could not read existing file for deduplication ({e}). Starting fresh.")

        last_saved_index = 0
        new_inferences_count = 0
        SAVE_INTERVAL_NEW_ITEMS = 5

        for idx in tqdm(range(0, num_inferences)):
            row = processed_df_.iloc[idx]
            title = str(row['title']).strip()

            if title in existing_titles:
                continue

            with Timer(f"{model_id_} inference"):
                prompt = TripletGenerator.build_prompt(row, text_col_)

                try:
                    response_text = TripletGenerator.generate_local_response(model_id_, prompt)
                    response = TripletGenerator.sanitize_response(response_text)
                    processed_df_.loc[idx, 'output_triplets'] = response
                    Logger.info(f"output_triplets (idx {idx}):\n{response}")

                    new_inferences_count += 1
                except Exception as e:
                    Logger.info(f"Error executing inference for row {idx}: {e}")
                    continue

            if new_inferences_count >= SAVE_INTERVAL_NEW_ITEMS:
                TripletGenerator._save_batch(processed_df_, last_saved_index, idx + 1, output_file)

                last_saved_index = idx + 1
                new_inferences_count = 0

        if last_saved_index < num_inferences:
            TripletGenerator._save_batch(processed_df_, last_saved_index, num_inferences, output_file)

        Logger.info("Finished processing all articles. All processed data is saved to CSV.")

    @staticmethod
    def generate_local_response(model_name: str, prompt: str) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    @staticmethod
    def sanitize_response(response_text: str) -> str:
        tuple_matches = re.findall(r'\((.*?)\)', response_text, re.DOTALL)

        cleaned_triplets = []

        for match in tuple_matches:
            element_pattern = r'(?:"([^"]*)"|\'([^\']*)\'|([^,\s]+))'
            elements = re.findall(element_pattern, match)

            raw_values = [e[0] or e[1] or e[2] for e in elements if any(e)]

            clean_values = []
            for val in raw_values:
                val_clean = re.sub(r'[^a-zA-Z0-9\s]', '', val).strip()
                clean_values.append(val_clean)

            if len(clean_values) == 6:
                t_str = "(" + ", ".join(f"'{v}'" for v in clean_values) + ")"
                cleaned_triplets.append(t_str)
        return "[" + ", ".join(cleaned_triplets) + "]"


def parse_args():
    parser = argparse.ArgumentParser(description="Triplet generation with specified model and data path.")

    parser.add_argument(
        "--model-name",
        type=str,
        choices=['qwen2', 'qwen2.5:14b', 'qwen2.5:30b', 'qwen2.5:latest', 'qwen3:4b', "qwen3:14b", 'qwen3:latest', 'gemini-2.5-flash', 'lite', 'micro', 'pro', 'premier', 'claude-sonnet-5'],
        default="qwen2.5:32b",
        help="Choose which model to run. Default is 'gemini-2.5-flash' from"
             " ['qwen2', 'qwen2.5:latest', 'qwen3:4b', 'qwen3:14b', 'qwen3:latest', 'llama3.1:8b', 'mistral:7b','gemini-2.5-flash', 'lite', 'micro', 'pro', 'premier']."
    )
    parser.add_argument(
        '--data-type',
        type=str,
        choices=["fnspid", "yf","semi_cleaned_data"],
        default="semi_cleaned_data",
        #help="Choose data type of the financial articles data. Default is yfinance articles data."
    )
    parser.add_argument(
        '--text-column',
        type=str,
        choices=['text', 'summary'],
        default='text',
        help="Choose which column to use for generating triplets. Default is 'text'."
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-02',
        help="Start date for filtering articles (format: YYYY-MM-DD). Default is '2024-01-01'."
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help="End date for filtering articles (format: YYYY-MM-DD). If not provided, no upper limit is applied."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Logger.setup_file("logs/triplet_generator")
    TripletGenerator(args.model_name, args.data_type, args.text_column, args.start_date, args.end_date)
``