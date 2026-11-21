from dotenv import load_dotenv

load_dotenv(override=True)

from llm_client import LLMClient, OllamaProvider
from usage_logger import UsageLogger


def main():

    usage_logger = UsageLogger()

    try:
        client = LLMClient(
            OllamaProvider(),
            usage_logger
        )

        prompt = input("Enter your question: ")

        answer = client.generate(prompt)

        print("\nLLM Response:")
        print(answer)

    finally:
        usage_logger.close()


if __name__ == "__main__":
    main()