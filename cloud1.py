#cloud1 BaiDuModel
import requests
import json
from cosSIM import similarity

def cloud_model1(question):
    url = "https://aip.baidubce.com/rpc/2.0/unit/service/v3/chat?access_token=EnterYourToken"
    payload = json.dumps({
        "log_id": "1234567890",
        "version": "3.0",
        "service_id": "S99567",
        "session_id": "",
        "request": {
            "query": question,
            "terminal_id": "1234567890"
        },
        "dialog_state": {
            "contexts": {
                "SYS_REMEMBERED_SKILLS": [
                    ""
                ]
            }
        }
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    # Access and print "say" directly if it exists in the response
    response = data.get("result", {}).get("responses", [{}])[0].get("actions", [{}])[0].get("options", [{}])
    answer_list = []
    for i in range(0, 5):
        obj = response[i]
        res = obj.get("info").get("full_answer")
        answer_list.append(res)
    print("找到的相关信息:", answer_list)
    max_score = 0.0
    pos = -1
    for i in range(0, 5):
        cos_sim = similarity(question, answer_list[i])
        if max_score < cos_sim:
            max_score = cos_sim
            pos = i
    print("最终回复:", answer_list[pos])
    return answer_list[pos]

