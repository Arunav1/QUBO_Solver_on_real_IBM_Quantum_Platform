import json
import os

from openai import OpenAI

from .schemas import OptimizationModel


SYSTEM_PROMPT = """
You are an optimization modeling assistant for VeriQAOA.

Convert a user's natural-language optimization problem into a
structured binary optimization model.

Supported model format:
{
  "title": "Short descriptive title",
  "variables": ["x1", "x2"],
  "objective": {"x1": 3, "x2": 2},
  "sense": "min",
  "constraints": [
    {
      "name": "select_exactly_one",
      "coefficients": {"x1": 1, "x2": 1},
      "sense": "==",
      "rhs": 1
    }
  ]
}

A constraint "sense" is exactly one of: "==", "<=", ">=", "<", ">".

Rules:
1. All decision variables must be binary: 0 or 1.
2. Objective coefficients must be numeric.
3. The objective sense must be "min" or "max".
4. Supported constraint senses are:
   - "==" : equal to
   - "<=" : less than or equal to
   - ">=" : greater than or equal to
   - "<"  : strictly less than
   - ">"  : strictly greater than
5. Represent each constraint using numeric coefficients and a numeric rhs.
6. Use unique variable names and ensure every referenced variable exists.
7. Do not invent costs, capacities, coefficients, or constraints.
8. If essential information is missing or ambiguous, set
   "requires_clarification" to true and ask specific questions.
9. Do not silently assume a missing objective or constraint.
10. Include assumptions only when explicitly supported by the user's description.
11. For a budget that must not be exceeded, use "<=".
12. For a requirement to spend exactly a specified amount, use "==".
13. For a minimum required amount, use ">=".
14. Do not replace an inequality with an equality unless the user
    explicitly requires equality.
15. Use one binary variable per yes/no decision and no more than 20
    variables in total. Do not add helper or slack variables; the
    compiler creates those itself.
16. Return JSON only, with exactly these top-level keys:
{
  "model": {
    "title": "...",
    "variables": [],
    "objective": {},
    "sense": "min",
    "constraints": []
  },
  "assumptions": [],
  "requires_clarification": false,
  "clarification_questions": []
}
"""


def formulate_optimization_problem(description: str) -> dict:
    """Convert a natural-language problem into a validated optimization model."""

    description = description.strip()

    if not description:
        raise ValueError("Problem description cannot be empty.")

    # Read API configuration.
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Set it in your environment before using this endpoint."
        )

    client_args = {"api_key": api_key}

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_args["base_url"] = base_url

    client = OpenAI(**client_args)
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Ask the LLM to formulate the optimization problem.
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Formulate this optimization problem as JSON:\n\n"
                    + description
                ),
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The LLM returned an empty response.")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The LLM did not return valid JSON."
        ) from error

    # Validate the outer response structure.
    required_keys = {
        "model",
        "assumptions",
        "requires_clarification",
        "clarification_questions",
    }

    if not isinstance(result, dict):
        raise ValueError("The LLM response must be a JSON object.")

    missing_keys = required_keys - set(result.keys())

    if missing_keys:
        raise ValueError(
            f"The LLM response is missing required fields: "
            f"{sorted(missing_keys)}"
        )

    if not isinstance(result["requires_clarification"], bool):
        raise ValueError("requires_clarification must be a boolean.")

    if not isinstance(result["assumptions"], list):
        raise ValueError("assumptions must be a list.")

    if not isinstance(result["clarification_questions"], list):
        raise ValueError("clarification_questions must be a list.")

    # Do not compile an ambiguous problem.
    if result["requires_clarification"]:
        return {
            "formulated": False,
            "requires_clarification": True,
            "clarification_questions": result["clarification_questions"],
            "assumptions": result["assumptions"],
            "model": None,
        }

    if not isinstance(result["model"], dict):
        raise ValueError(
            "The LLM response must contain a model JSON object."
        )

    # Validate the generated model against the Pydantic schema.
    try:
        if hasattr(OptimizationModel, "model_validate"):
            validated_model = OptimizationModel.model_validate(
                result["model"]
            )
        else:
            validated_model = OptimizationModel.parse_obj(
                result["model"]
            )

        # Check duplicate variables and all referenced variable names.
        validated_model.validate_variables()

    except Exception as error:
        raise ValueError(
            f"The generated optimization model is invalid: {error}"
        ) from error

    # Explicitly validate constraint senses.
    # The schema should enforce this too, but this provides a clear error.
    supported_senses = {"==", "<=", ">=", "<", ">"}

    for constraint in validated_model.constraints:
        if constraint.sense not in supported_senses:
            raise ValueError(
                f"Unsupported constraint sense: {constraint.sense}"
            )

    # Return the normalized, validated model.
    model_data = (
        validated_model.model_dump()
        if hasattr(validated_model, "model_dump")
        else validated_model.dict()
    )

    return {
        "formulated": True,
        "requires_clarification": False,
        "clarification_questions": [],
        "assumptions": result["assumptions"],
        "model": model_data,
    }
