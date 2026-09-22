import os
import requests

riddle = """I have keys but no locks.
I have space but no room.
You can enter, but you can't go inside.
What am I?"""

state = {
    "user_information": riddle
}

payload = {
    "model": "typesafe/jev-1.13",

    "state": state,

    "questions": {
        "answer": {
            "type": "choice",

            "instructions": """
            Find the answer to the riddle.
            Choose the candidate that best satisfies all
            three clues in the riddle.
            """,

            "criteria": {
                "keyboard": """
                findout based on riddle
                """,

                "mouse": """
                findout based on riddle
                """,

                "numbers": """
                findout based on riddle
                """
            }
        }
    }
}

response = requests.post(
    "https://openrouter.ai/api/alpha/decisions",
    headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json"
    },
    json=payload
)

response.raise_for_status()
print()
result = response.json()

print(result)