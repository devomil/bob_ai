#!/usr/bin/env python3
"""GPT4All CLI

A lightweight command-line interface (CLI) for GPT4All, allowing chat interactions in the terminal.
"""

import io
import sys
import importlib.metadata
from collections import namedtuple
from typing_extensions import Annotated
import typer
from gpt4all import GPT4All

MESSAGES = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi, how can I assist you?"},
]

SPECIAL_COMMANDS = {
    "/reset": lambda messages: messages.clear(),
    "/exit": lambda _: sys.exit(),
    "/clear": lambda _: print("\n" * 100),
    "/help": lambda _: print("Special commands: /reset, /exit, /help, /clear"),
}

VersionInfo = namedtuple('VersionInfo', ['major', 'minor', 'micro'])
VERSION_INFO = VersionInfo(1, 0, 2)
VERSION = '.'.join(map(str, VERSION_INFO))  

CLI_START_MESSAGE = f"""
    
 ██████  ██████  ████████ ██   ██  █████  ██      ██      
██       ██   ██    ██    ██   ██ ██   ██ ██      ██      
██   ███ ██████     ██    ███████ ███████ ██      ██      
██    ██ ██         ██         ██ ██   ██ ██      ██      
 ██████  ██         ██         ██ ██   ██ ███████ ███████ 
                                                          

Welcome to the GPT4All CLI! Version {VERSION}
Type /help for special commands.
"""

app = typer.Typer()

@app.command()
def repl(
    model: Annotated[str, typer.Option("--model", "-m", help="Model file to use")] = "mistral-7b-instruct-v0.1.Q4_0.gguf",
    n_threads: Annotated[int, typer.Option("--n-threads", "-t", help="Number of threads")] = None,
    device: Annotated[str, typer.Option("--device", "-d", help="Device (cpu, cuda, amd, etc.)")] = None,
):
    """Interactive CLI chat session with GPT4All."""
    
    gpt4all_instance = GPT4All(model, device=device)

    if n_threads is not None:
        gpt4all_instance.model.set_thread_count(n_threads)

    print(CLI_START_MESSAGE)

    use_new_loop = False
    try:
        version = importlib.metadata.version('gpt4all')
        if int(version.split('.')[0]) >= 1:
            use_new_loop = True
    except:
        pass

    if use_new_loop:
        _new_loop(gpt4all_instance)
    else:
        _old_loop(gpt4all_instance)


def _old_loop(gpt4all_instance):
    while True:
        message = input(" ⇢  ")

        if message in SPECIAL_COMMANDS:
            SPECIAL_COMMANDS[message](MESSAGES)
            continue

        MESSAGES.append({"role": "user", "content": message})

        full_response = gpt4all_instance.chat_completion(
            MESSAGES,
            n_past=0,
            n_predict=200,
            top_k=40,
            top_p=0.9,
            temp=0.9,
            n_batch=9,
            repeat_penalty=1.1,
            repeat_last_n=64,
            verbose=False,
            streaming=True,
        )

        MESSAGES.append(full_response.get("choices")[0].get("message"))
        print()


def _new_loop(gpt4all_instance):
    with gpt4all_instance.chat_session():
        while True:
            message = input(" ⇢  ")

            if message in SPECIAL_COMMANDS:
                SPECIAL_COMMANDS[message](MESSAGES)
                continue

            MESSAGES.append({"role": "user", "content": message})

            response_generator = gpt4all_instance.generate(
                message,
                max_tokens=200,
                temp=0.9,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.1,
                repeat_last_n=64,
                n_batch=9,
                streaming=True,
            )

            response = io.StringIO()
            for token in response_generator:
                print(token, end='', flush=True)
                response.write(token)

            response_message = {'role': 'assistant', 'content': response.getvalue()}
            response.close()
            gpt4all_instance.current_chat_session.append(response_message)
            MESSAGES.append(response_message)
            print()


@app.command()
def version():
    """Show GPT4All CLI version"""
    print(f"GPT4All CLI v{VERSION}")


if __name__ == "__main__":
    app()
