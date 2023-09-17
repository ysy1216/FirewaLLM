#cloud2 TencentModel
import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tbp.v20190627 import tbp_client, models

def cloud_model2(user_input):
    try:
        # Instantiating an authentication object requires passing in the Tencent Cloud account SecretId and SecretKey to enter the parameters. Here, it is also necessary to pay attention to the confidentiality of the key pair
        #Code leakage may lead to the leakage of SecretId and SecretKey, and threaten the security of all resources under the account. The following code example is for reference only, and it is recommended to use the key in a more secure way
        SecretId = "EnterYourSecretId"
        SecretKey = "EnterYourSecretKey"
        cred = credential.Credential(SecretId, SecretKey)
        #Instantiate an HTTP option
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tbp.ap-guangzhou.tencentcloudapi.com"

        # Instantiating a client option
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # Instantiate the client object to request the product
        client = tbp_client.TbpClient(cred, "", clientProfile)

        #Instantiate a request object, and each interface will correspond to a request object
        req = models.TextProcessRequest()
        params = {
            "BotId": "58d826e3-73af-4ddb-887d-350475e88a9a",
            "BotEnv": "release",
            "TerminalId": "user1",
            "SessionAttributes": "True",
            "InputText": user_input,
            "PlatformType": "True",
            "PlatformId": "True"
        }
        req.from_json_string(json.dumps(params))

        #The returned resp is an instance of TextProcessResponse, corresponding to the request object
        resp = client.TextProcess(req)
        # Output the content in ResponseText
        response_text = json.loads(resp.to_json_string())["ResponseText"]
        return response_text

    except TencentCloudSDKException as err:
        return err