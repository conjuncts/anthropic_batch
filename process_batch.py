import json
import time
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from tqdm import tqdm

# Load environment variables from a .env file
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize the Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)

def write_jsonl(file_path, data, mode='a'):
    with open(file_path, mode) as file:
        file.write(json.dumps(data) + '\n')

def get_completion(messages, model_name, temperature):
    """
    Convert into Anthropic API format, then call the API
    """
    system_prompts = []
    regular_messages = []
    for message in messages:
        if message['role'] == 'system':
            system_prompts.append(message['content'])
        else:
            regular_messages.append(message)
    response = client.messages.create(
        system=" ".join(system_prompts),
        temperature=temperature,
        model=model_name,
        max_tokens=4096,  # Adjust as necessary
        messages=regular_messages
    )
    return response

def process_batch_file(input_file, output_file=None, include_header=False, header_comments=None):
    """
    Given a JSONL file in OpenAI batch format (.jsonl), process each item in the file
    See https://platform.openai.com/docs/guides/batch
    
    :param input_file: str, path to the input JSONL file
    :param output_file: str, path to the output JSONL file (default: batch_{input_file}_output.jsonl)
    :param include_header: bool, whether to include a header in the output file (default: False)
    NOTE: This header is not normally part of the OpenAI batch format!
    :param header_comments: str, optional comments to include in the header (default: "output for {input_file}")
    """
    assert input_file.endswith(".jsonl"), "Input file must be a JSONL file"
    

    if output_file is None:
        input_dir, input_filename = os.path.split(input_file)
        input_filename = input_filename[:-6]  + "_output.jsonl"
        output_file = os.path.join(input_dir, "batch_" + input_filename)
        
        
    input_data = list(read_jsonl(input_file))
    if include_header:
        if header_comments is None:
            header_comments = f"output for {input_file}"
        header = {"header": header_comments, "file": input_file}
        write_jsonl(output_file, header, mode='w')

    for item in tqdm(input_data):
        custom_id = item.get("custom_id")
        body = item.get("body")

        if body is not None:
            messages = body["messages"]
            model_name = body["model"]
            temperature = body.get("temperature", 1.0)
            
            # Call the Anthropic API
            response = get_completion(messages, model_name=model_name, temperature=temperature)
            
            # Create the output structure
            if response.type == "message":
                output_item = {
                    "id": response.id,
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "request_id": None,
                        "body": {
                            "id": None,
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": response.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": response.content[0].text
                                    },
                                    "logprobs": None,
                                    "finish_reason": response.stop_reason,
                                }
                            ],
                            "usage": {
                                "prompt_tokens": response.usage.input_tokens,
                                "completion_tokens": response.usage.output_tokens,
                                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                            },
                            "system_fingerprint": None
                        }
                    },
                    "error": None
                }
            else:
                # errors are thrown as exceptions, so actually no need to handle them here
                # let the program fail, because otherwise we may repeatedly error out the API - bad
                output_item = {
                    "id": response.id,
                    "custom_id": custom_id,
                    "response": None,
                    "error": {
                        "status_code": response.error.type, # 
                        "message": response.message
                    }
                }
            write_jsonl(output_file, output_item)
                
                # Add a delay of 0.5 seconds
            time.sleep(0.5)

if __name__ == "__main__":
    input_file = "examples/toy_anthropic_batch.jsonl"  # Input file name
    process_batch_file(input_file, include_header=True)
