import torch
from BagData_bk_6eval import test_dataloader, transform
import numpy as np
import matplotlib.pyplot as plt

def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)
    model1 = torch.load('./trained_model/fcn_model_95_1.pt')
    model2 = torch.load('./trained_model/fcn_model_0_2.pt')
    # model3 = torch.load('./trained_model/fcn_model_95_3.pt')
    # model4 = torch.load('./trained_model/fcn_model_95_4.pt')
    # model5 = torch.load('./trained_model/fcn_model_95_5.pt')
    # model6 = torch.load('./trained_model/fcn_model_95_6.pt')
    model1 = model1.to(device)
    model2 = model2.to(device)
    # model3 = model3.to(device)
    # model4 = model4.to(device)
    # model5 = model5.to(device)
    # model6 = model6.to(device)
    model1.eval()
    model2.eval()
    # model3.eval()
    # model4.eval()
    # model5.eval()
    # model6.eval()
    print(len(test_dataloader))
    for index, (bag, bag_msk) in enumerate(test_dataloader):
        print(index)
        print(bag.shape, bag_msk.shape)
        # print(bag[0:4, 0:3, :, :])
        # plt.imshow(bag_msk[0, 0, :, :], cmap='gray')
        # plt.show()
        # bag.unsqueeze(0)
        # print(bag.shape)
        output1 = model1(bag)
        output2 = model2(bag)
        #output3 = model3(bag)
        #output4 = model4(bag)
        #output5 = model5(bag)
        #output6 = model6(bag)
        output1 = torch.sigmoid(output1)  # output.shape is torch.Size([4, 2, 160, 160])
        output2 = torch.sigmoid(output2)  # output.shape is torch.Size([4, 2, 160, 160])
        #output3 = torch.sigmoid(output3)  # output.shape is torch.Size([4, 2, 160, 160])
        #output4 = torch.sigmoid(output4)  # output.shape is torch.Size([4, 2, 160, 160])
        #output5 = torch.sigmoid(output5)  # output.shape is torch.Size([4, 2, 160, 160])
        #output6 = torch.sigmoid(output6)  # output.shape is torch.Size([4, 2, 160, 160])
        # print(output.shape)
        output_np1 = output1.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        output_np2 = output2.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        #output_np3 = output3.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        #output_np4 = output4.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        #output_np5 = output5.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        #output_np6 = output6.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        print(output_np1.shape)

        plt.subplot(1, 3, 1)
        plt.imshow(bag_msk[0, 0, :, :], 'gray')
        # plt.show()
        plt.subplot(1, 2, 2)
        # im1 = np.squeeze(output_np1[0, 0, ...])
        # im2 = np.squeeze(output_np2[0, 0, ...])
        # im1[im1 > 0.5] = 1
        # im1[im1 < 0.5] = 0
        # print(im1)
        plt.imshow(np.squeeze(output_np1[0, 0, ...]), 'gray')
        plt.subplot(1, 2, 3)
        plt.imshow(np.squeeze(output_np2[0, 0, ...]), 'gray')
        plt.show()
        break
    # for index, (bag, bag_msk) in enumerate(test_dataloader):

        # bag = bag.to(device)
        # output = model(bag)
        # output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])


if __name__ == '__main__':
    main()
