import matplotlib.pyplot as plt
import numpy as np
import config
import random
import albumentations as A

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
    plt.savefig(f"{config.OUTPUT_PATH}/val_epoch_{epoch}.png")
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
    
    plt.figure(figsize=(10, 10))
    for i in range(0, 16):
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

        plt.subplot(4, 4, i+1)
        plt.imshow(img)
        
        for j in range(len(keypoints)):
            plt.plot(keypoints[j, 0], keypoints[j, 1], '.r')

    plt.show()
    plt.close()
