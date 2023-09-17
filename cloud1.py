#cloud1 BaiDuModel
import requests
import json
def cloud_model1(question):
  #access_token
  url = "https://aip.baidubce.com/rpc/2.0/unit/service/chat?access_token=EnterYourAccess_Token"
  payload = json.dumps({
    "log_id":"1234567890",
    "version":"2.0",
    "service_id":"S96706",
    "session_id":"",
    "request":{
        "query":question,
        "user_id":"1234567890"
    },
    "dialog_state":{
        "contexts":{
            "SYS_REMEMBERED_SKILLS":[
                ""
            ]
        }
    }
})
  headers = {
    'Content-Type': 'text/plain'
  }

  response = requests.request("POST", url, headers=headers, data=payload)
  data = response.json()

  # Access and print "say" directly if it exists in the response
  say_content = data.get("result", {}).get("response_list", [{}])[0].get("action_list", [{}])[0].get("say", "")
  return say_content
