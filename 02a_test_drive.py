# test if the API works

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="http://api.yesapikey.com/v1"
)

try:
    response = client.chat.completions.create(
        model="gpt-5-2025-08-07",
        messages=[{"role": "user", "content": "Say 'Hello' if you can hear me."}],
    )
    print(response.choices[0].message.content)

except Exception as e:
    print(e)