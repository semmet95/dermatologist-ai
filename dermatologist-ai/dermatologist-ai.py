from flask import Flask, jsonify, request
from PIL import Image
import requests
import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
from collections import OrderedDict

app = Flask(__name__)
@app.route('/dermatologistai', methods=['POST'])
def make_predict():
    # downloading and saving the image
    data=request.get_json(force=True)
    imgurl=data["url"]
    #print("could not fetch image url", imgurl)
    img_data = requests.get(imgurl).content
    file_type=imgurl[imgurl.rfind('.'):]
    img_filename_downloaded = 'downloaded'+file_type
    with open(img_filename_downloaded, 'wb') as handler:
        handler.write(img_data)

    image = Image.open(img_filename_downloaded)
    t1 = transforms.Resize(256)
    t2 = transforms.CenterCrop(224)
    t3 = transforms.ToTensor()
    #t4 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #image = t4(t3(t2(t1(image))))
    image = t3(t2(t1(image)))
    image = image.unsqueeze(0)

    # making predictions
    output = model(image)
    
    output = F.softmax(output, dim=1)
    print('output is ', output)
    return jsonify({"melanoma":float(output[0][0]), "nevus":float(output[0][1]), "seborrheic keratosis":float(output[0][2])})

@app.before_first_request
def loadthemodel():
    #loading the model here to ensure it's loaded only once
    global model
    model = models.resnet152(pretrained=True)
    classifier_name, old_classifier = model._modules.popitem()
    classifier_input_size = old_classifier.in_features
    classifier = nn.Sequential(OrderedDict([('drop1', nn.Dropout(p=0.4)),
                                           ('fc1', nn.Linear(classifier_input_size, 3))
                                           ]))
    model.add_module(classifier_name, classifier)
    #laoding the best most accurate checkpoint
    state_dict = torch.load(os.getcwd()+'/model_dermatologist_ai_84.pt', map_location='cpu')
    model.load_state_dict(state_dict)

if __name__=="__main__":
    app.run()