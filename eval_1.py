import torch
from BagData_bk_6eval import transform, test_dataloader, name_list
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.misc
from PIL import Image
from BagData import transform
from skimage import io,data,img_as_float,img_as_ubyte,img_as_uint,color

def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)
    model1 = torch.load('./trained_model/fcn_model_95_1.pt')
    model2 = torch.load('./trained_model/fcn_model_0_2.pt')
    model3 = torch.load('./trained_model/fcn_model_95_3.pt')
    model4 = torch.load('./trained_model/fcn_model_95_4.pt')
    model5 = torch.load('./trained_model/fcn_model_95_5.pt')
    model6 = torch.load('./trained_model/fcn_model_95_6.pt')
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model4 = model4.to(device)
    model5 = model5.to(device)
    model6 = model6.to(device)
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()

    # imgA = cv2.imread('E:/dataset/Cityview/test/images/exp_001744.png')
    # imgA = cv2.resize(imgA, (160, 160))
    # imgA = transform(imgA)
    # print(imgA.shape)
    # output = model(imgA)
    # output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
    # output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
    # plt.imshow(np.squeeze(output_np[0, 0, ...]), 'gray')

# print(len(test_dataloader))
    # for index, (bag, bag_msk) in enumerate(test_dataloader):
    #     print(index)
    #     print(bag.shape, bag_msk.shape)
    #     # print(bag[0:4, 0:3, :, :])
    #     # plt.imshow(bag_msk[0, 0, :, :], cmap='gray')
    #     # plt.show()
    #
    #     # bag.unsqueeze(0)
    #     # print(bag.shape)
    #     output = model(bag)
    #     output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
    #     # print(output.shape)
    #     output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
    #     print(output_np.shape)
    #
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(bag_msk[0, 0, :, :], 'gray')
    #     # plt.show()
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(np.squeeze(output_np[0, 0, ...]), 'gray')
    #     plt.show()
    #     break

    for index, (bag, bag_msk) in enumerate(test_dataloader):
        print(index)
        print(name_list[index])
        output1 = model1(bag)
        output1 = torch.sigmoid(output1)  # output.shape is torch.Size([4, 2, 160, 160])
        output_np1 = output1.cpu().detach().numpy().copy()  # output_np.shape =
        output2 = model2(bag)
        output2 = torch.sigmoid(output2)  # output.shape is torch.Size([4, 2, 160, 160])
        output_np2 = output2.cpu().detach().numpy().copy()  # output_np.shape =
        output3 = model3(bag)
        output3 = torch.sigmoid(output3)  # output.shape is torch.Size([4, 2, 160, 160])
        output_np3 = output3.cpu().detach().numpy().copy()  # output_np.shape =
        output4 = model4(bag)
        output4 = torch.sigmoid(output4)  # output.shape is torch.Size([4, 2, 160, 160])
        output_np4 = output4.cpu().detach().numpy().copy()  # output_np.shape =
        output5 = model5(bag)
        output5 = torch.sigmoid(output5)  # output.shape is torch.Size([4, 2, 160, 160])
        output_np5 = output5.cpu().detach().numpy().copy()  # output_np.shape =
        output6 = model6(bag)
        output6 = torch.sigmoid(output6)  # output.shape is torch.Size([4, 2, 160, 160])
        output_np6 = output6.cpu().detach().numpy().copy()  # output_np.shape =
        mask1 = output_np1[0,0, ...]
        mask2 = output_np2[0,0, ...]
        mask3 = output_np3[0,0, ...]
        mask4 = output_np4[0,0, ...]
        mask5 = output_np5[0,0, ...]
        mask6 = output_np6[0,0, ...]
        #print(mask6.shape)
        mask1[mask1 > 0.5] = 1
        mask1[mask1 < 0.5] = 0
        mask2[mask2 > 0.5] = 2
        mask2[mask2 < 0.5] = 0
        mask12 = mask1 + mask2
        mask12[mask12 > 2] = 2
        mask3[mask3 > 0.5] = 3
        mask3[mask3 < 0.5] = 0
        mask123 = mask12 + mask3
        mask123[mask123 > 3] = 3
        mask4[mask4 > 0.5] = 4
        mask4[mask4 < 0.5] = 0
        mask1234 = mask123 + mask4
        mask1234[mask1234 > 4] = 4
        mask5[mask5 > 0.5] = 5
        mask5[mask5 < 0.5] = 0
        mask12345 = mask1234 + mask5
        mask12345[mask12345 > 5] = 5
        mask6[mask6 > 0.5] = 6
        mask6[mask6 < 0.5] = 0
        mask123456 = mask12345 + mask6
        mask123456[mask123456 > 6] = 6
        #print(mask123456.shape)
        #print(bag_msk.shape)
        #mask123456 = mask123456.transpose(1, 2, 0)
        #print(mask123456.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(np.squeeze(mask123456), 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(bag_msk[0, :, :])
        #plt.show()
        #scipy.misc.imsave('E:/03_python_file/comp5421_TASK2/licv/test.png', mask6)
        #cv2.imwrite("filename.png", np.zeros(10,10))
        imgA = cv2.resize(mask123456, (800, 600))
        imgB = np.rint(imgA)
        img1 = np.zeros((600,800), np.uint8)
        #print(img1.shape)
        #print(img1.dtype.name)
        for i in range(0,600):
            for j in range(0,800):
                temp = imgB[i, j]
                #print(i)
                #print(j)
                #print(temp)
                if (temp == 1):
                    img1[i, j] = 1
                elif (temp == 2):
                    img1[i, j] = 2
                elif (temp == 3):
                    img1[i, j] = 3
                elif (temp == 4):
                    img1[i, j] = 4
                elif (temp == 5):
                    img1[i, j] = 5
                elif (temp == 6):
                    img1[i, j] = 6
                else:
                    img1[i, j] = 0
        #print(img1.dtype.name)

        io.imsave('/tmp/' + name_list[index] +'.png',img1)
        #cv2.imwrite('E:/03_python_file/comp5421_TASK2/licv/test.png', final_mask)
        # [os.path.join('E:/03_python_file/comp5421_TASK2/licv/' + name_list[index] + '_cv.png')]
        # plt.imshow(np.squeeze(output_np[0, 0, ...]), 'gray')
        # plt.show()
        # break


if __name__ == '__main__':
    main()
