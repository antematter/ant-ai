import csv
import logging
import os
from logging import Handler


class UsageMetrics:
    """Class to store token usage details."""

    def __init__(
        self, total_tokens, prompt_tokens, completion_tokens, successful_requests
    ):
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.successful_requests = successful_requests


class TaskOutput:
    """Class to represent TaskOutput."""

    def __init__(self, description, expected_output, raw, agent, output_format):
        self.description = description
        self.expected_output = expected_output
        self.raw = raw
        self.agent = agent
        self.output_format = output_format


class CrewOutput:
    """Class to represent CrewOutput."""

    def __init__(self, raw, tasks_output, token_usage, pydantic=None, json_dict=None):
        self.raw = raw
        self.tasks_output = tasks_output
        self.token_usage = token_usage
        self.pydantic = pydantic
        self.json_dict = json_dict


class CSVHandler(Handler):
    """handler to log output to CSV, for headers and append behavior."""

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

        self.fieldnames = [
            "raw",
            "task_description",
            "expected_output",
            "agent",
            "output_format",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "successful_requests",
        ]

        self.ensure_headers_if_not_exists()

    def ensure_headers_if_not_exists(self):
        """Check if the file exists. If it doesn't, create it and add the header."""
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
            print(f"Headers written to {self.filename}")
        else:
            print(f"{self.filename} already exists, appending...")

    def emit(self, record):
        """Append log data to the CSV while ensuring headers aren't duplicated."""
        log_data = self.format(record)

        with open(self.filename, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(log_data)


class CrewOutputCSVFormatter(logging.Formatter):
    """ Formatter for the `CrewOutput` content for CSV file logging."""

    def format(self, record):
        """Extracts the important fields from CrewOutput and formats them for the CSV."""
        crew_output = record.crew_output

        task = crew_output.tasks_output[0] if crew_output.tasks_output else None

        log_data = {
            "raw": crew_output.raw or "N/A",
            "task_description": task.description if task else "N/A",
            "expected_output": task.expected_output if task else "N/A",
            "agent": task.agent if task else "N/A",
            "output_format": task.output_format if task else "N/A",
            "total_tokens": crew_output.token_usage.total_tokens,
            "prompt_tokens": crew_output.token_usage.prompt_tokens,
            "completion_tokens": crew_output.token_usage.completion_tokens,
            "successful_requests": crew_output.token_usage.successful_requests,
        }

        return log_data


def setup_logger(filename):
    """Sets up the logger with the CSVHandler and attaches a custom formatter."""
    logger = logging.getLogger("crew_csv_logger")

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    csv_handler = CSVHandler(filename)
    csv_formatter = CrewOutputCSVFormatter()
    csv_handler.setFormatter(csv_formatter)

    logger.addHandler(csv_handler)

    logger.propagate = False

    return logger



#logger.info('Logging CrewOutput Item', extra={'crew_output': PromptCrew})