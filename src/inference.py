import torch
import numpy as np
import cv2
import os
import config
import time
from model import KeypointCustom
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from dataset import train_data
from matplotlib import pyplot as plt
#from google.colab.patches import cv2_imshow # for image display

def InferFrame(model, frame):
    
    with torch.no_grad():
        image = np.transpose(frame, (1, 2, 0)).cpu().detach().numpy()
        
        exit()
        #image = frame

        #print(image.shape)
        #image = frame
        h, w, c = image.shape
        if not (h == config.IMG_SIZE and w == config.IMG_SIZE):
            print(f"Warning: The Expected input shape of the image is ({config.IMG_SIZE}, {config.IMG_SIZE}) but received ({h}, {w})\nResizing taking place")
            image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        orig_frame = image.copy()
        #orig_frame = image.clone().detach().numpy()
        #orig_frame = orig_frame.cpu().detach().numpy()

        #orig_h, orig_w, c = orig_frame.shape
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        #image = np.transpose(image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0).to(config.DEVICE)
        image = image.half()

        outputs = model(image)
        outputs = outputs.cpu().detach().numpy()

          
        outputs = outputs.reshape(-1, 2)
        keypoints = outputs
        for p in range(keypoints.shape[0]):
            cv2.circle(orig_frame, (int(keypoints[p, 0] * (config.IMG_SIZE / h)), int(keypoints[p, 1] * (config.IMG_SIZE / w))),
                        1, (0, 0, 255), -1, cv2.LINE_AA)

         

    return orig_frame
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
    
    infer_data = DataLoader(
        train_data, 
        batch_size=inferBatchSize, 
        shuffle=True
    )

    #model.eval()
    #model.cuda()
    #model.half()

    cone_counter = 0
    avg_fps = 0
    
    t_start = time.time()
    
    with torch.no_grad():
      for _, data in tqdm(enumerate(infer_data), total=len(infer_data)):
          batch_fps = 0.0
          batch_counter = 0
          
          image = data['image'].to(config.DEVICE)
          image = image.half()
          
          t1 = time.time()
          outputs = model(image)
          t2 = time.time()
          inference_fps = round((1/(t2 - t1)) , 3) 
          
          # plot the predicted validation keypoints after every...
          # ... predefined number of epochs
          #valid_keypoints_plot(image, outputs, keypoints, epoch)
          # detach the image, keypoints, and output tensors from GPU to CPU
          images = image.detach().cpu()
          outputs = outputs.detach().cpu().numpy()
          print(images.shape)
          print(outputs.shape)
          # just get a single datapoint from each batch
          for Image, output_keypoint in zip(images, outputs):
            print(Image.shape)
            print(output_keypoint.shape)
            
            img = np.array(Image, dtype='float32')
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(f'{config.OUTPUT_PATH}/test.jpg', img)
            exit()
            h, w, c = img.shape
            if not (h == config.IMG_SIZE and w == config.IMG_SIZE):
                print(f"Warning: The Expected input shape of the image is ({config.IMG_SIZE}, {config.IMG_SIZE}) but received ({h}, {w})\nResizing taking place")
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            # img = image[random_index]
            # output_keypoint = outputs[random_index]
            # orig_keypoint = orig_keypoints[random_index]
            
            # fig = plt.figure()
            # fig.add_subplot(111)
            
            # plt.imshow(img)
            
            output_keypoint = output_keypoint.reshape(-1, 2)

            for p in range(output_keypoint.shape[0]):
              cv2.circle(img, (int(output_keypoint[p, 0] * (config.IMG_SIZE / h)), int(output_keypoint[p, 1] * (config.IMG_SIZE / w))),
                          1, (0, 0, 255), -1, cv2.LINE_AA)
            # for p in range(output_keypoint.shape[0]):
            #     plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')
            # plt.canvas.draw()
            # Now we can save it to a numpy array.
            # img_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            # final_img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # plt.savefig(f"{config.OUTPUT_PATH}/{config.CURRENT_MODEL}/validation/val_epoch_{epoch}.png")
            # plt.close()

            if inferBatchSize == 1:
              cv2.imwrite(f'{config.OUTPUT_PATH}/efficientnet_b0/inference_fp16_batch_size_1/cone_{cone_counter}.jpg', img)

            batch_counter += 1

          batch_fps += inference_fps

      batch_fps = round(batch_fps / batch_counter, 3)
      avg_fps += batch_fps
      cone_counter += 1
    t_end = time.time()

    print(round(t_end - t_start, 2))
    avg_fps = round(avg_fps / cone_counter, 2)

    return avg_fps