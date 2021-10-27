import torch
import numpy as np
import cv2
import os
import config
import time
from model import KeypointEfficientNet, KeypointResNet, KeypointCustom

def InferFrame(model, frame, size=224):
    with torch.no_grad():
        image = frame
        h, w, c = frame.shape
        if not (h == size and w == size):
            print(f"Warning: The Expected input shape of the image is ({size}, {size}) but received ({h}, {w})\nResizing taking place")
            image = cv2.resize(image, (size, size))
        orig_frame = image.copy()
        orig_h, orig_w, c = orig_frame.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0).to(config.DEVICE)
        outputs = model(image)
        outputs = outputs.cpu().detach().numpy()

        outputs = outputs.reshape(-1, 2)
        keypoints = outputs
        for p in range(keypoints.shape[0]):
            cv2.circle(orig_frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                        1, (0, 0, 255), -1, cv2.LINE_AA)
    
        return orig_frame


#img = cv2.imread(f'{config.ROOT_PATH}/test/Abdel_Aziz_Al-Hakim_10.jpg')
#model = KeypointResNet(pretrained=False, requires_grad=False).to(config.DEVICE)

#model = KeypointEfficientNet(pretrained=False, requires_grad=False, model_name='efficientnet-b2')
model = KeypointCustom(isPretrained=True, requires_grad=False)
model = model.return_loaded_model().to(config.DEVICE)

#checkpoint = torch.load('../weights/resnet18_30_epochs.pth', map_location=torch.device('cpu'))
#checkpoint = torch.load('../weights/efficientNet-b2_full_25_epochs.pth', map_location=torch.device('cpu'))
checkpoint = torch.load('../weights/efficientNetV2_10Epochs.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

TEST_IMAGES_DIR = f'{config.ROOT_PATH}/test'
for files in os.listdir(TEST_IMAGES_DIR):
    if files.endswith(".jpg") or files.endswith(".png") or files.endswith(".bmp"):
        img = cv2.imread(f'{TEST_IMAGES_DIR}/{files}')
        if not os.path.isdir(f'{config.OUTPUT_PATH}/test_inference_images_effnetV2'):
            os.makedirs(f'{config.OUTPUT_PATH}/test_inference_images_effnetV2')
        
        t1 = time.time()
        final_img = InferFrame(model, img)
        t2 = time.time()
        print(f"Inference Speed: {1/(t2 - t1):.2f}fps")
        cv2.imwrite(f'{config.OUTPUT_PATH}/test_inference_images_effnetV2/{files}', final_img)
#cv2.waitKey()

