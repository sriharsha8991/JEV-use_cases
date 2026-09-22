
riddle = """I have keys but no locks.
I have space but no room.
You can enter, but you can't go inside.
What am I?"""


state = f"user information: {riddle}"

from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient() as client: 
    response = client.system_one(
        state=state,
        questions={
            "answer":Choice(
            instructions="find the answer from the given user information?",
            criteria={
                "keyboard":"computer unit system similar to the riddle",
                "mouse":"has similar composition of riddle",
                "numbers": "does the same"
            }
                    )
        }
        
    )
    print(response.answers["answer"].choice)