// Fold button click event listener
document.querySelector('.collapse-chat-btn').addEventListener('click', () => {
    document.querySelector('.chat-history').classList.toggle('collapsed');
    document.querySelector('.chat-panel').classList.toggle('expanded');
});


// Teenage mode
document.addEventListener('DOMContentLoaded', function() {
    const toggleButton = document.getElementById('toggleButton');
    const hiddenSL = document.getElementById('sensitivity-level');
    const modelSelect = document.getElementById('model-select');

    toggleButton.addEventListener('change', function() {
        if (toggleButton.checked) {
            hiddenSL.style.display = 'none'; 

            modelSelect.innerHTML = `
                <option value="new_model1">本地模型1</option>
                <option value="new_model2">本地模型2</option>
                <option value="new_model3">本地模型3</option>
            `;
        }

        else {
            hiddenSL.style.display = 'block'; 

            modelSelect.innerHTML = `
                <option value="model1">百度云模型</option>
                <option value="model2">腾讯云模型</option>
                <option value="model3" selected>gpt-3.5-turbo</option>
                <option value="model4">privategpt</option>
                <option value="model5">chatglm</option>
            `;
        }
    });
});


// Add message to chat-display
function appendMessage(userType, message, modelMessage = "") {
    const chatDisplay = document.getElementById("chat-display");
    const outsideMessageContainer = document.createElement("div")
    outsideMessageContainer.classList.add(userType === "user" ? "user-message-outside" : "chatbot-message-outside");
    const messageContainer = document.createElement("div");

    const messageContent = document.createElement("div");
    messageContent.classList.add("message-content");
    messageContent.innerHTML = message;

    const avatar = document.createElement("div");
    avatar.classList.add("avatar");
    const avatarImg = document.createElement("img");
    avatarImg.classList.add("avatar-img")

    avatarImg.src = userType === "user" ? "../static/images/user-avatar.svg" : "../static/images/bot-avatar.svg";
    avatar.appendChild(avatarImg);

    // Copy button
    const copyButton = document.createElement("button");
    copyButton.textContent = "复制";
    copyButton.classList.add("copy-button");
    copyButton.addEventListener("click", function() {
        copyToClipboard(message);
        copyButton.textContent = "已复制";
        setTimeout(function() {
            copyButton.textContent = "复制";
        }, 1500); 
    });

    // Hide model message button
    if (modelMessage !== "" && modelMessage.replace(/\s/g, '') != message.replace(/\s/g, '')) {
        const hideModelButton = document.createElement("button");
        hideModelButton.textContent = "隐藏模型消息";
        hideModelButton.classList.add("hide-model-button");
        hideModelButton.addEventListener("click", function() {
            outsideModelMessageContainer.style.display = outsideModelMessageContainer.style.display === "none" ? "flex" : "none";
            hideModelButton.textContent = outsideModelMessageContainer.style.display === "none" ? "展开模型消息" : "隐藏模型消息";
        });

        // Set different style class names based on user type
        messageContainer.classList.add(userType === "user" ? "user-message" : "chatbot-message");
        messageContainer.appendChild(avatar);
        messageContainer.appendChild(messageContent);
        messageContainer.appendChild(hideModelButton);
        messageContainer.appendChild(copyButton);
        outsideMessageContainer.appendChild(messageContainer)

        // Model message
        // width 100%
        const outsideModelMessageContainer = document.createElement("div");
        outsideModelMessageContainer.classList.add("model-message-outside");
        
        const modelMessageContainer = document.createElement("div");
        modelMessageContainer.classList.add("model-message");
        
        const modelAvatar = document.createElement("div");
        modelAvatar.classList.add("avatar");
        
        const modelMessageContent = document.createElement("div");
        modelMessageContent.classList.add("model-message-content");
        modelMessageContent.innerHTML = modelMessage;
      
        const modelCopyButton = document.createElement("button");
        modelCopyButton.textContent = "复制";
        modelCopyButton.classList.add("copy-button");
        modelCopyButton.addEventListener("click", function() {
            copyToClipboard(modelMessage);
            modelCopyButton.textContent = "已复制";
            setTimeout(function() {
                modelCopyButton.textContent = "复制";
            }, 1500); 
        });

        modelMessageContainer.appendChild(modelAvatar);
        modelMessageContainer.appendChild(modelMessageContent);
        modelMessageContainer.appendChild(modelCopyButton);
        outsideModelMessageContainer.appendChild(modelMessageContainer)


        chatDisplay.appendChild(outsideMessageContainer);
        chatDisplay.appendChild(outsideModelMessageContainer);
    }

    else {
        messageContainer.classList.add(userType === "user" ? "user-message" : "chatbot-message");
        messageContainer.appendChild(avatar);
        messageContainer.appendChild(messageContent);
        messageContainer.appendChild(copyButton);

        outsideMessageContainer.appendChild(messageContainer)
        chatDisplay.appendChild(outsideMessageContainer);
    }
}


function copyToClipboard(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
}

function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    const selectedLevel = document.getElementById("sensitivity-level").value;
    const selectedTag = document.getElementById("toggleButton").checked;
    document.getElementById("user-input").value = "";
    fetch("/get_mask", {
        method: "POST", 
        body: new URLSearchParams({
            "user_input": userInput, 
            "sen_level": selectedLevel,
            "ask_tag":selectedTag
        }),
        headers: {
            "Content-Type": "application/x-www-form-urlencoded" 
        }
    })
    .then(response => response.text()) 
    .then(data => {
        appendMessage("user", userInput, data);
        getMessage(data);
    })
    .catch(error => {
        console.error("Error:", error); 
    });
}

function getMessage(data){
    const selected_model = document.getElementById("model-select").value;
    fetch("/get_response", {
        method: "POST", 
        body: new URLSearchParams({
            "selected_model": selected_model,
            "mask_info": data
        }),
        headers: {
            "Content-Type": "application/x-www-form-urlencoded" 
        }
    })
    .then(response => response.text()) 
    .then(data => {
        appendMessage("bot", data);
    })
    .catch(error => {
        console.error("Error:", error); 
    });

}

document.getElementById('send-button').addEventListener('click', sendMessage);
// Enter+ctrl
document.getElementById('user-input').addEventListener('keydown', function (event) {
    if (event.key === "Enter" && !event.ctrlKey) {
        event.preventDefault();

        const startPos = this.selectionStart;
        const endPos = this.selectionEnd;
        this.value = this.value.substring(0, startPos) + "\n" + this.value.substring(endPos);

        this.selectionStart = startPos + 1;
        this.selectionEnd = startPos + 1;
    }
    else if (event.key === "Enter" && event.ctrlKey) {
        event.preventDefault();

        sendMessage();
    }
});


function clearChat() {
    var chatDisplay = document.getElementById("chat-display");
    var confirmation = confirm("记录清除后无法恢复，您确定要清除吗？");
    if (confirmation) {
      chatDisplay.innerHTML = ""; 
    }
  }


  function handleExport() {
    var chatDisplay = document.getElementById("chat-display");
    var chatContent = chatDisplay.innerText;
    var blob = new Blob([chatContent], { type: "text/plain" });

    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "chat_export.txt";
    a.textContent = "Download";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }