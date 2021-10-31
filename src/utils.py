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


# def plot_comparison_graph(data_1, data_2, label_1, label_2, y_label):
#     plt.figure(figsize=(10, 7))
#     plt.plot(data_1, color='orange', label=label_1)
#     plt.plot(data_2, color='red', label=label_2)
#     plt.xlabel('Epochs')
#     plt.ylabel(y_label)
#     plt.title(config.CURRENT_MODEL)
#     plt.legend()

#     plt.savefig(f"{config.OUTPUT_PATH}/{config.CURRENT_MODEL}/{y_label}.png")

def plot_comparison_graph(data):
    
    # Predetermining the plot parameters to set the fonts
    SMALL_SIZE = 13
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 30

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    n_rows = 2
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(45, 20))
    
    fig.suptitle(config.CURRENT_MODEL)
    for ax, data_points in zip(axs.flat, data):
        
        if data_points is not None:
            if len(data_points) == 3:
                colors = ['red', 'blue', 'orange', 'green', 'cyan', 'purple', 'black']
                
                for batchSize, Color, fps_points in zip(config.INFER_BATCH_SIZES, colors, data_points[0]):
                    ax.plot(batchSize, fps_points, marker='o', markerfacecolor=Color, markeredgecolor='black', markersize=15, label=f'BATCH SIZE: {batchSize}')
                    ax.text(batchSize - 1, fps_points - 7, str(fps_points))
                ax.set(xlabel='Batch size', ylabel=data_points[1], title=data_points[2])
                ax.legend(loc='lower right')

            else:
                #plt.figure(figsize=(10, 7))
                ax.plot(data_points[0][0], color='orange', label=data_points[1])
                ax.plot(data_points[0][1], color='red', label=data_points[2])
                ax.set(xlabel='Epochs', ylabel=data_points[3], title=data_points[3])
                

                if data_points[4] == 'min':
                    min_value_1 = min(data_points[0][0])
                    index_min_1 = data_points[0][0].index(min_value_1)
                    #ax.annotate(str(min_value_1), xy=(index_min_1, min_value_1), color='black', ms=20)

                    ax.plot(index_min_1, min_value_1, marker='o', markerfacecolor='orange', markeredgecolor='black')
                    ax.text(index_min_1+1, min_value_1+1, str(min_value_1))

                    min_value_2 = min(data_points[0][1])
                    index_min_2 = data_points[0][1].index(min_value_2)
                    #ax.annotate(str(min_value_2), xy=(index_min_2, min_value_2), color='black', ms=20)
                    ax.plot(index_min_2, min_value_2, marker='o', markerfacecolor='red', markeredgecolor='black')
                    ax.text(index_min_2+1, min_value_2+1, str(min_value_2))

                    ax.legend(loc='upper right')
                
                if data_points[4] == 'max':
                    min_value_1 = max(data_points[0][0])
                    index_min_1 = data_points[0][0].index(min_value_1)
                    #ax.annotate(str(min_value_1), xy=(index_min_1, min_value_1), color='black', ms=20)
                    ax.plot(index_min_1, min_value_1, marker='o', markerfacecolor='orange', markeredgecolor='black')
                    ax.text(index_min_1+1, min_value_1+1, str(min_value_1))

                    min_value_2 = max(data_points[0][1])
                    index_min_2 = data_points[0][1].index(min_value_2)
                    # ax.annotate(str(min_value_2), xy=(index_min_2, min_value_2), color='black', ms=20)
                    ax.plot(index_min_2, min_value_2, marker='o', markerfacecolor='red', markeredgecolor='black')
                    ax.text(index_min_2+1, min_value_2+1, str(min_value_2))

                    ax.legend(loc='lower right')
            
    fig.delaxes(axs[1, 3])
    
    plt.savefig(f"{config.OUTPUT_PATH}/{config.CURRENT_MODEL}/{config.CURRENT_MODEL}_model_analysis.png")
