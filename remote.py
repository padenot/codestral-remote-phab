#!/usr/bin/env python3

import sys
import os
from mistralai.client import MistralClient
from mistralai import Mistral, UserMessage, ToolMessage
import argparse
import requests
from retrying import retry
import json
import pprint as pp
import coloredlogs, logging
import functools
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import re
coloredlogs.install()

@retry(stop_max_attempt_number=3)  # Retry up to 3 times
def download_file(url, headers=()):
    logging.info(f"Downloading regular file {url} headers: {headers}")
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()  # Raise an exception if the request failed

    str = ""
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:  # Filter out keep-alive new chunks
          str += chunk.decode('utf-8')
    return str

def fetch_json(url):
    headers = {'Accept': 'application/json'}
    str = download_file(url, headers)
    return json.loads(str)

def source_files_from_symbol(symbol):
    try:
        logging.info(f"Attempting to find more info {symbol}")
        json = fetch_json(f"https://searchfox.org/mozilla-central/search?q={symbol}&path=&case=false&regexp=false")
        paths = []
        json = json['normal']
        # direct file hits
        if 'Files' in json:
            for file in json['Files']:
                paths.append(file["path"])
        else:
            # symbol name or whatever
            for i in json:
                for j in json[i]:
                    if 'path' in j:
                        paths.append(j['path'])
        path = list(set(paths))
        useful_source = ""
        paths = paths[:3] # the first few files ought to be enough
        for path in paths:
            # filter that out for now
            if "webidl" not in path:
                logging.info(f"Fetching file {path}")
                raw_file = download_file(f"https://hg.mozilla.org/mozilla-central/raw-file/tip/{path}")
                useful_source += f"Here's the source for {path}:\n\n"
                useful_source += "".join(raw_file)
                useful_source += "\n"

        return useful_source

    except Exception as e:
        logging.info(f"Couldn't find more info for {symbol}, {e}")
        return f"Couldn't find source for {symbol}, sorry! {e}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "source_files_from_symbol",
            "description": "Get more details or interesting Firefox source file from a programming symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "a programming language symbol",
                    }
                },
                "required": ["symbol"],
            },
        },
    }
]

names_to_functions = {
    'source_files_from_symbol': functools.partial(source_files_from_symbol),
}

my_parser = argparse.ArgumentParser(description='codestral-remote')

my_parser.add_argument('--diff', type=str, help='URL of diff', required=False)
my_parser.add_argument('--mamba', help='use a snake', action="store_true", required=False)
my_parser.add_argument('--large', help='L A R G E', action="store_true", required=False)
args = my_parser.parse_args()

model = "codestral-latest"
if args.mamba:
  model = "codestral-mamba-latest"
if args.large:
  model = "mistral-large-latest"

patch = ""

if args.diff is not None:
    patch = download_file(args.diff + "?download=true")
else: # mode stdin
    patch = "".join(sys.stdin.readlines())

with open('prompt.txt', 'r') as file:
    data = file.read()
    data = data.replace("HERE", patch)

messages = [{"role": "user", "content": data}]

api_key = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=api_key)

if not args.mamba:
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model = model,
        messages = messages,
        tools = tools,
        tool_choice = "any",
    )
    response

    messages.append(response.choices[0].message)

    import json
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    function_params = json.loads(tool_call.function.arguments)
    function_result = names_to_functions[function_name](**function_params)

    messages.append({"role":"tool", "name":function_name, "content":function_result, "tool_call_id":tool_call.id})

    import time
    time.sleep(1) # sometimes rate error without this

    response = client.chat.complete(
        model = model,
        messages = messages
    )
    print(response.choices[0].message.content)

else:
    # mamba doesn't support tooling. shove everything in the prompt
    regex = r"diff --git a\/(.*) b"
    matches = re.finditer(regex, data, re.MULTILINE)
    full_file = ""
    for matchNum, match in enumerate(matches, start=1):
        full_file = source_files_from_symbol(match.group(1))
        break

    data += "Here's the complete file on which the diff need to be applied if needed:\n"
    data += full_file

    messages = [
        UserMessage(role="user", content=data)
    ]

    # pp.pp(messages)

    response = client.chat.complete(
        model=model,
        messages=messages,
    )

    pp.pp(response)
    print(response.choices[0].message.content)
