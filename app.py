from flask import Flask, render_template, request
from mark import fun_1
from cloud1 import cloud_model1
from cloud2 import cloud_model2
from cloud3 import cloud_model3
import os
app = Flask(__name__, static_url_path='/static')
# set env
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890" # os.environ["http_proxy"]  
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890" # os.environ["https_proxy"]  

@app.route('/')
def index():
    return render_template('index.html')


#submit and get masked information
@app.route("/get_mask", methods=['POST', 'GET'])
def get_mask():
    user_input = request.form['user_input']
    print("脱敏前", user_input)
    if user_input == "bye":
        return "Goodbye!"
    selected_sen_level = (int)(request.form["sen_level"])
    selected_tag = request.form["ask_tag"]
    masked_text = fun_1(user_input, selected_sen_level, selected_tag)
    print("脱敏后", masked_text)
    return masked_text
#get model response
@app.route("/get_response", methods=['POST','GET'])
def get_response():
    selected_cloud_model = request.form['selected_model']  # 
    masked_text = request.form['mask_info']
    if selected_cloud_model == 'model1':
        response = cloud_model1(masked_text)
    elif selected_cloud_model == 'model2':
        response = cloud_model2(masked_text)
    elif selected_cloud_model == 'model3':
        response = cloud_model3(masked_text)
    return response
if __name__ == '__main__':
    app.run(debug=True)