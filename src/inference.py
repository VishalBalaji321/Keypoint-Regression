import torch
import numpy as np
import cv2
import os
import config
import time
from model import KeypointEfficientNet, KeypointResNet, KeypointCustom
import tqdm
from torch.utils.data import DataLoader
from dataset import train_data

def InferFrame(model, frame):
    t1 = time.time()
    with torch.no_grad():
        image = frame
        h, w, c = frame.shape
        if not (h == config.IMG_SIZE and w == config.IMG_SIZE):
            print(f"Warning: The Expected input shape of the image is ({config.IMG_SIZE}, {config.IMG_SIZE}) but received ({h}, {w})\nResizing taking place")
            image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        orig_frame = image.copy()
        #orig_h, orig_w, c = orig_frame.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0).to(config.DEVICE)
        image = image.half()

        outputs = model(image)
        outputs = outputs.cpu().detach().numpy()
        t2 = time.time()
        inference_fps = round((1/(t2 - t1)) , 2)

        outputs = outputs.reshape(-1, 2)
        keypoints = outputs
        for p in range(keypoints.shape[0]):
            cv2.circle(orig_frame, (int(keypoints[p, 0] * (config.IMG_SIZE / h)), int(keypoints[p, 1] * (config.IMG_SIZE / w))),
                        1, (0, 0, 255), -1, cv2.LINE_AA)
    
        return orig_frame, inference_fps

def InferDirectory(weights_path, input_path, output_path, modelName=config.CURRENT_MODEL):
    #img = cv2.imread(f'{config.ROOT_PATH}/test/Abdel_Aziz_Al-Hakim_10.jpg')
    #model = KeypointResNet(pretrained=False, requires_grad=False).to(config.DEVICE)

    #model = KeypointEfficientNet(pretrained=False, requires_grad=False, model_name='efficientnet-b2')
    model = KeypointCustom(isPretrained=False, requires_grad=False, model_name=modelName)
    model = model.return_loaded_model().to(config.DEVICE)

    #checkpoint = torch.load('../weights/resnet18_30_epochs.pth', map_location=torch.device('cpu'))
    #checkpoint = torch.load('../weights/efficientNet-b2_full_25_epochs.pth', map_location=torch.device('cpu'))
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cuda()
    model.half()

    for files in os.listdir(input_path):
        if files.endswith(".jpg") or files.endswith(".png") or files.endswith(".bmp"):
            img = cv2.imread(f'{input_path}/{files}')
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            
            final_img, inferenceFPS = InferFrame(model, img)
            print(inferenceFPS)
            
            cv2.imwrite(f'{output_path}/{files}', final_img)

def InferDataloader(model, inferBatchSize):
    test_data = DataLoader(
        train_data, 
        batch_size=inferBatchSize, 
        shuffle=True
    )

    model.eval()
    model.half()

    cone_counter = 1
    avg_fps = 0
    
    # calculate the number of batches
    num_batches = int(len(train_data)/test_data.batch_size)
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_data), total=num_batches):
            image = data['image']
            final_img, fps = InferFrame(model, image)
            
            if inferBatchSize == 1:
                cv2.imwrite(f'{config.OUTPUT_PATH}/{config.CURRENT_MODEL}/inference_fp16/cone_{cone_counter}.jpg', final_img)
            
            avg_fps += fps
            cone_counter += 1
    
    avg_fps = round(avg_fps / cone_counter, 2)


    return avg_fps