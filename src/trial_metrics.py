
import torch
import torch.nn as nn

pred = torch.tensor([5, 10, 20, 32])
target = torch.tensor([4, 12, 18, 36])
IMG_SIZE = 45

CORNERS = torch.tensor([(0, 0), (0, IMG_SIZE), (IMG_SIZE, 0), (IMG_SIZE, IMG_SIZE)])

#print(pred.size())
#dist = nn.functional.pairwise_distance(pred, CORNERS)
#print(dist)

avg_acc = 0.0
num_point = 0
# Finding the farthest corner
for points in target.reshape(-1, 2):
    max_dist = 0
    for corners in CORNERS:
        euclid_distance = torch.sqrt(torch.pow(corners[0] - points[0], 2) + torch.pow(corners[1] - points[1], 2))
        if euclid_distance > max_dist:
            max_dist = euclid_distance
    
    dist_pred_target = torch.sqrt(torch.pow(pred.reshape(-1, 2)[num_point][0] - points[0], 2) + torch.pow(pred.reshape(-1, 2)[num_point][1] - points[1], 2))
    print(dist_pred_target)
    print(max_dist)
    print()

    if max_dist > 0:
        avg_acc += (max_dist - dist_pred_target)/max_dist
    else:
        print("Max dist between target and predictions is 0 !!!")
    
    num_point += 1

print(avg_acc/num_point * 100)
