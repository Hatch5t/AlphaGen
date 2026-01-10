import argparse
import json
import os
from time import perf_counter, sleep

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ace_extensions import (
    get_alpha_recordset,
    get_datafield,
    get_stored_session,
    simulate_single_alpha,
)
from ace_lib import get_simulation_result_json
from alpha_utils import (
    copy,
    extract_datafields,
    fix_fastexpr,
    generate_pnl_chart,
    get_insample_context,
    strict_submissibility,
)

parser = argparse.ArgumentParser(
    description="Refine Alpha Expressions based on Static Simulation Settings and Iterative Refinement using Large Language Models."
)

parser.add_argument(
    "--alpha_id",
    "-id",
    type=str,
    help="Alpha ID",
)

args = parser.parse_args()
alpha_id = args.alpha_id

if not alpha_id:
    parser.error("Please input an Alpha ID.")


context = []
console = Console()
load_dotenv()

gemini_api_keys = os.getenv("GEMINI_API_KEYS").split(",")
gemini_api_key_id = 0


genai_client = genai.Client(api_key=gemini_api_keys[gemini_api_key_id])
brain_session = get_stored_session(duration=10800)


with open("model.json", "r") as f:
    model = json.load(f)

with open("config.json", "r") as f:
    config = json.load(f)


simulated_alpha = get_simulation_result_json(brain_session, alpha_id)
pnl = get_alpha_recordset(brain_session, alpha_id, "pnl")

generate_pnl_chart(config["pnl_chart"], pnl)
datafields = extract_datafields(simulated_alpha["regular"]["code"])
trial_alpha = copy(simulated_alpha)


print()
system_instruction_datafield = "Data Field Context:"
for datafield in datafields:
    datafield = get_datafield(brain_session, datafield)

    id = datafield["id"]
    type = datafield["type"]
    description = datafield["description"]

    if type == "VECTOR" and "VECTOR" not in config["operators"]:
        config["operators"].append("VECTOR")

    system_instruction_datafield += f"\n{id} ({type}): {description}"

if "Group" in config["operators"]:
    system_instruction_datafield += "\nGrouping Fields:\n"
    for group in config["groups"]:
        if (
            group == "currency" and simulated_alpha["settings"]["delay"] != 1
        ):  # Currency Grouping Field is only available for Delay 1 Alphas
            continue

        if group[0] == "!":
            continue

        system_instruction_datafield += group + ", "


console.print("Operator Category:", style="blue")
system_instruction_operators = "Operators Context:"
for operator_category in config["operators"]:
    if operator_category[0] == "!":
        continue

    system_instruction_operators += f"\n\n{operator_category} Operators:\n"

    with open(f"operators/{operator_category}.txt", "r") as f:
        system_instruction_operators += f.read()

    console.print(operator_category, end=", ", style="blue")
print()


system_instruction_warning = "System Warnings:\n"
with open("system_instructions/warnings.txt") as f:
    system_instruction_warning += f.read()


system_instructions = "\n\n".join(
    [
        system_instruction_operators,
        system_instruction_datafield,
        system_instruction_warning,
    ]
)


iteration_0 = "\n".join(
    [
        "Iteration #0",
        "Alpha Expression:",
        simulated_alpha["regular"]["code"],
        "",
        get_insample_context(simulated_alpha["is"]),
    ]
)
context.append(
    types.Content(role="user", parts=[types.Part.from_text(text=iteration_0)])
)


model_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level=model["thinking_level"]),
    temperature=model["temperature"],
    response_mime_type="application/json",
    response_schema=types.Schema(
        type=types.Type.OBJECT,
        description=model["structured_output"]["schema_description"],
        required=list(model["structured_output"]["schema"].keys()),
        properties={
            key: types.Schema(type=types.Type.STRING, description=desc)
            for key, desc in model["structured_output"]["schema"].items()
        },
    ),
    system_instruction=[types.Part.from_text(text=system_instructions)],
)


def get_model_output(genai_client, model, context, model_config):

    start = perf_counter()
    trial = 0
    with Progress(
        SpinnerColumn(),
        TextColumn(
            "[yellow]Attempt #{task.fields[trial]} | Retrieving Model Output...[/]"
        ),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(
            description="",
            total=None,
            trial=0,
        )

        while True:
            trial += 1
            progress.update(task, trial=trial)

            response = genai_client.models.generate_content(
                model=model,
                contents=context,
                config=model_config,
            )

            if response.parsed:
                break

            sleep(5)

    elapsed = perf_counter() - start
    console.log(
        f"[yellow]Attempt #{trial} | Model Output Retrieved[/] in {elapsed:.2f}s"
    )

    return response.parsed


def get_model_context(i, model_output):
    lines = [f"Iteration #{i + 1}"]
    for key, value in model_output.items():
        lines.append(f"{key}:")
        lines.append(str(value))
    return "\n".join(lines)


print()

console.print(f"Model: {model["model"]}", style="purple")
console.print(f"Temperature: {model["temperature"]}", style="purple")
console.print(f"Thinking Level: {model["thinking_level"]}", style="purple")

print()

console.print(system_instruction_datafield, style="blue")

print()

console.print(iteration_0, style="green")

print()


for i in range(config["iterations"]):

    while True:
        try:
            model_output = get_model_output(
                genai_client, model["model"], context, model_config
            )
            break

        except Exception as e:
            console.print(f"\nget_model_output: {e}", style="red")

            gemini_api_key_id = (gemini_api_key_id + 1) % len(gemini_api_keys)
            console.print(
                f"Switching GEMINI API Key ID...",
                style="yellow",
            )

            genai_client = genai.Client(api_key=gemini_api_keys[gemini_api_key_id])

            sleep(5)

    model_output["Alpha Expression"] = fix_fastexpr(model_output["Alpha Expression"])
    model_context = get_model_context(i, model_output)

    context.append(
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=model_context)],
        )
    )

    console.print(model_context, style="cyan")

    trial_alpha["regular"] = model_output["Alpha Expression"]

    start = perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:

        progress.add_task("[yellow]Simulating Alpha...[/]", total=None)
        simulation_result = simulate_single_alpha(brain_session, trial_alpha)

    alpha_id = simulation_result["id"]

    elapsed = perf_counter() - start
    console.log(f"[yellow]Simulated Alpha {alpha_id}[/] in {elapsed:.2f}s")

    insample = simulation_result["is"]
    checks = insample["checks"]

    pnl = get_alpha_recordset(brain_session, alpha_id, "pnl")
    generate_pnl_chart(config["pnl_chart"], pnl)

    iteration_context = get_insample_context(insample)

    context.append(
        types.Content(role="user", parts=[types.Part.from_text(text=iteration_context)])
    )
    console.print(iteration_context, style="green")

    print()
    print()

    submissibility = strict_submissibility(checks)

    if submissibility:
        console.print("Alpha Expression refined successfully!", style="purple")
        exit()
