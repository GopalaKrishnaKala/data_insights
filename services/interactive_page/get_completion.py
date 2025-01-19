from openai import OpenAI as OpenAIClient

def get_completion(prompt, client, model="gpt-4", temperature=0.1):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
