# FirewaLLM
By calling FirewaLLM, users can ensure the accuracy of the large model while greatly reducing the risk of privacy leakage when interacting with it.

With the development of large models, more and more users are using them for interactive Q&A. However, with the increasing volume of data, privacy issues have attracted widespread attention.So, we launched a privacy protection small model. Firstly, we will judge sensitive statements based on user questions, and secondly, we will handle sensitive statements accordingly. We guarantee that user privacy exposure may be greatly reduced when interacting with the model. Finally, we ensure the accuracy of large model interactions by restoring sensitive information. Experiments have shown that users interact with large models by calling small models. 

The FirewaLLM framework not only protects user data privacy, but also protects the accuracy of the interaction model.

# Usage

1. Train FirewallM to have the ability to recognize sensitive information.
'''python
python FirewaLLM/Judgment_sensitivity/train.py
'''
2. Open the FirewaLLM interface for interaction, and users can choose different large models for interaction.
'''python
python FirewaLLM/app.py
'''
3. The effect is as follows:
![image](https://github.com/ysy1216/FirewaLLM/blob/a4d47819027b5531141d15e967f67b7ed2684d5a/FirewaLLM_demo1.png)
4. It can be intuitively seen that when a user inputs sensitive information, FirewaLLM will perform different levels of sensitive information processing operations on the sensitive information.
5. In the end, FirewaLLM will perform sensitivity restoration processing, returning the results with the highest similarity between the answers of the large model and the problem, in order to protect privacy while ensuring the accuracy of the overall process.
