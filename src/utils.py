import matplotlib.pyplot as plt
import numpy as np
import config
import random
import torch
import os

def valid_keypoints_plot(image, outputs, orig_keypoints, epoch):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    """

    # detach the image, keypoints, and output tensors from GPU to CPU
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()
    
    # just get a single datapoint from each batch
    random_index = random.randint(3, len(image)-2)
    img = image[random_index]
    output_keypoint = outputs[random_index]
    orig_keypoint = orig_keypoints[random_index]
    img = np.array(img, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    
    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoint = orig_keypoint.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')
        plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'b.')
    plt.savefig(f"{config.OUTPUT_PATH}/{config.CURRENT_MODEL}/validation/val_epoch_{epoch}.png")
    plt.close()

def dataset_keypoints_plot(data):
    """
    This function shows the image faces and keypoint plots that the model
    will actually see. This is a good way to validate that our dataset is in
    fact corrent and the faces align wiht the keypoint features. The plot 
    will be show just before training starts. Press `q` to quit the plot and
    start training.
    """
    # transform = A.Compose(
    #     [A.HorizontalFlip(p=1)],
    #     keypoint_params=A.KeypointParams(format='xy')
    # )
    
    plt.figure(figsize=(20, 20))
    for i in range(0, 40):
        sample = data[i]
        img = sample['image']
        img = np.array(img, dtype='float32')
        img = np.transpose(img, (1, 2, 0))
        
        keypoints = sample['keypoints']
        #print(img.shape)
        
        
        # transformed = transform(image=img, keypoints=keypoints)

        # plt.subplot(2, 4, i+1)
        # plt.imshow(transformed['image']) 

        # for j in range(len(transformed['keypoints'])):
        #     plt.plot(transformed['keypoints'][j][0], transformed['keypoints'][j][1], 'b.')

        plt.subplot(8, 5, i+1)
        plt.imshow(img)
        
        for j in range(len(keypoints)):
            plt.plot(keypoints[j, 0], keypoints[j, 1], '.r')

    plt.show()
    plt.close()

def accuracy(outputs_tensor, keypoints_tensor):
  # Coordinates of the four corners of the image
  CORNERS = torch.tensor([(0, 0), (0, config.IMG_SIZE), (config.IMG_SIZE, 0), (config.IMG_SIZE, config.IMG_SIZE)])

  avg_acc = 0.0
  num_batches = 0
  
  for pred, target in zip(outputs_tensor, keypoints_tensor):
    num_batches += 1
    avg_acc_img = 0.0
    num_point = 0
    
    for points in target.reshape(-1, 2):
        # Finding the farthest corner
        max_dist = 0
        for corners in CORNERS:
            euclid_distance = torch.sqrt(torch.pow(corners[0] - points[0], 2) + torch.pow(corners[1] - points[1], 2))
            if euclid_distance > max_dist:
                max_dist = euclid_distance
        
        dist_pred_target = torch.sqrt(torch.pow(pred.reshape(-1, 2)[num_point][0] - points[0], 2) + torch.pow(pred.reshape(-1, 2)[num_point][1] - points[1], 2))
  
        if max_dist > 0:
            avg_acc_img += (max_dist - dist_pred_target)/max_dist
        else:
            print("Points cant be on the corner !!!")
        
        num_point += 1
    avg_acc += avg_acc_img/num_point * 100
  avg_acc = avg_acc / num_batches

  return avg_acc


def save_model(validation_loss, full_model):
    # full_model -> Tuple containing 'model', 'optimizer', 'criterion', 'epoch'

    weights_location = f'{config.OUTPUT_PATH}/{config.CURRENT_MODEL}/weights'
    if not os.path.exists(weights_location):
        os.makedirs(weights_location)
    
    # For the first iteration
    if 'last.pth' not in os.listdir(weights_location) and 'best.pth' not in os.listdir(weights_location):
        weights_path = f'{weights_location}/last.pth'
        torch.save({
            'epoch': full_model[3],
            'model_state_dict': full_model[0].state_dict(),
            'optimizer_state_dict': full_model[1].state_dict(),
            'loss': full_model[2],
        }, weights_path)

    else:
        # Saving the best model for the least validation loss
        if validation_loss[-1] == min(validation_loss):
            if 'best.pth' in os.listdir(weights_location):
                os.remove(f'{weights_location}/best.pth')
            
            weights_path = f'{weights_location}/best.pth'
            torch.save({
                'epoch': full_model[3],
                'model_state_dict': full_model[0].state_dict(),
                'optimizer_state_dict': full_model[1].state_dict(),
                'loss': full_model[2],
            }, weights_path)
        
        if 'last.pth' in os.listdir(weights_location):
            os.remove(f'{weights_location}/last.pth')
            
        weights_path = f'{weights_location}/last.pth'
        torch.save({
            'epoch': full_model[3],
            'model_state_dict': full_model[0].state_dict(),
            'optimizer_state_dict': full_model[1].state_dict(),
            'loss': full_model[2],
        }, weights_path)