import argparse
import json
import os

import requests

from ml_service import MLService


def local_mode(model_path, query, use_file):
    ml_service = MLService(model_path)

    if use_file:
        if not os.path.exists(query):
            print(f"File '{query}' does not exist.")
            return
        with open(query, 'r', encoding="u8") as file:
            data = file.read().replace("\r", "").split("\n")
            result_json = []

            for address in data:
                result_json.append({"query": address, "result": ml_service.search_addresses(query)})
    else:
        result_json = ml_service.search_addresses(query)

    print(json.dumps(result_json, indent=4, ensure_ascii=False))


def api_mode(api_path, query):
    url = f"{api_path}/search?query={query}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        result_json = response.json()
        print(json.dumps(result_json, indent=4, ensure_ascii=False))
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="AMM (Address Matching Machine) CLI")
    parser.add_argument("--mode", choices=["api", "local"], default="local", help="Select mode: api or local")

    parser.add_argument("--model-path", type=str, default="./model/best_fast_text.model", help="Path to the model (local mode only)")
    parser.add_argument("--file", action="store_true", help="Use a file as input (local mode only)")
    parser.add_argument("query", type=str, nargs="?", help="Query string or file path (local mode only)")

    parser.add_argument("--api-path", type=str, help="API endpoint (api mode only)")

    args = parser.parse_args()

    if args.mode == "local":
        if not args.model_path:
            parser.error("In local mode, --model-path is required.")
        if not args.query:
            parser.error("In local mode, a query string or file path is required.")
        local_mode(args.model_path, args.query, args.file)
    elif args.mode == "api":
        if not args.api_path:
            parser.error("In API mode, --api-path is required.")
        api_mode(args.api_path, args.query)


if __name__ == "__main__":
    main()
