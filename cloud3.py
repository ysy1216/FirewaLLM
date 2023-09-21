#cloud3 gpt3.5
import openai

def cloud_model3(content):
  openai.api_key = "EnterYourKey"
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",  # gpt-3.5-turbo-0301、text-davinci-003
    messages=[
      {"role": "user", "content": content}
    ],
    temperature=0.7,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.5,
  )
  return response.choices[0].message.content
