import os
import numpy as np
from torch.utils.data import DataLoader
from util.DataLoader import FaceIdPoseDataset
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from torchvision import transforms

def Validation_MPIE(Model, epoch, device, save_dir, args, Error_Flag=False):

    print("Start Validating...")
    Model.eval()

    csv_file = args.csv_file
    data_place = args.data_place

    GalleryTestPose = ['{}'.format(args.analyze_pose)]
    GalleryPose = ['110', '120', '090', '080', '130', '140', '051']  # Face left
    ProbePose = ['110', '120', '090', '080', '130',
                 '140', '051', '050', '041', '190',
                 '200', '010', '240']

    Performance_Results = {'Epoch-{}'.format(epoch):[]}
    for idx in ProbePose: Performance_Results.setdefault('{}'.format(idx), [])

    if Error_Flag:
        Error_Results = {'Epoch-{}/Gallery-{}'.format(epoch, GalleryTestPose[0]): []}
        for idx in ProbePose: Error_Results.setdefault('{}'.format(idx), [])
        Error_Sample_List = {'ErrorSample': []}


    # Load augmented data
    transform = transforms.Compose([transforms.Resize([112, 112]),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    transformed_dataset = FaceIdPoseDataset(csv_file, data_place, transform=transform)
    dataloader = DataLoader(transformed_dataset, batch_size=32, shuffle=False)  # , num_workers=6)

    Features_List = {'Subject': [], 'Pose': [], 'Illum': [], 'Features': []}
    if not os.path.exists('{}/Features.npy'.format(save_dir)):
        counter = 0
        for i, batch_data in enumerate(dataloader):

            batch_image = batch_data[0].to(device)
            batchImageName = batch_data[1]

            Model_Feature = Model(batch_image)
            features = (Model_Feature.data).cpu().numpy()

            for ImgName, feas in zip(batchImageName, features):
                tmp = ImgName.split('/')
                Features_List['Subject'].append(int(tmp[2]))
                Features_List['Pose'].append(int(tmp[-1][10:13]))
                Features_List['Illum'].append(int(tmp[-1][14:16]))
                Features_List['Features'].append(feas.reshape(-1, len(feas)))
            counter += 32
            print(counter)
        np.save('{}/Features.npy'.format(save_dir), Features_List)

    # Multi-PIE error analysis
    if Error_Flag:
        Performance = np.zeros((20, len(ProbePose)))
        for g_idx, g_pose in enumerate(GalleryTestPose):
            if g_pose == '110' or g_pose == '120': g_find = np.where((np.array(Features_List['Illum']) == 5) & (np.array(Features_List['Pose']) == int(g_pose)))[0]
            else: g_find = np.where((np.array(Features_List['Illum']) == 7) & (np.array(Features_List['Pose']) == int(g_pose)))[0]
            g_feature = np.squeeze(np.array(Features_List['Features'])[g_find])
            g_subject = np.squeeze(np.array(Features_List['Subject'])[g_find])

            for p_idx, p_pose in enumerate(ProbePose):
                # probe index
                p_find = np.where(np.array(Features_List['Pose']) == int(p_pose))[0]
                p_feature = np.squeeze(np.array(Features_List['Features'])[p_find])
                p_subject = np.squeeze(np.array(Features_List['Subject'])[p_find])
                p_illumination = np.squeeze(np.array(Features_List['Illum'])[p_find])

                # compare gallery and probe
                similarity_temp = cosine_similarity(p_feature.reshape(-1, len(feas)), g_feature.reshape(-1, len(feas)))

                # find the predict gallery results
                g_predict = np.array([g_subject[np.argmax(simi_idx)] for simi_idx in similarity_temp])

                # if probe_subject not equals to predicted_gallery, Incorrect ! Error sample occurred.
                error_samples = np.where((p_subject - g_predict) != 0)[0]
                for err_idx in error_samples:
                    Performance[p_illumination[err_idx], p_idx] += 1
                    Error_Sample_List['ErrorSample'].append('{}_01_01_{}_{}.png'.format(p_subject[err_idx], p_pose, p_illumination[err_idx]))

        for illumn_idx in range(20):
            Error_Results['Epoch-{}/Gallery-{}'.format(epoch, GalleryTestPose[0])].append('Illumination-{}'.format(illumn_idx))
            for p_idx in range(len(ProbePose)):
                Error_Results['{}'.format(ProbePose[p_idx])].append(Performance[illumn_idx, p_idx])

        save_dir_analysis = './{}/ErrorAnalysis_MPIE.csv'.format(save_dir)
        temp = ['Epoch-{}/Gallery-{}'.format(epoch, GalleryTestPose[0])]
        for idx in ProbePose: temp.append(idx)
        Error_Results = pd.DataFrame.from_dict(Error_Results)
        Error_Results = Error_Results.reindex(columns=temp)
        if not os.path.exists(save_dir_analysis): Error_Results.to_csv(save_dir_analysis, index=False)
        else: Error_Results.to_csv(save_dir_analysis, mode='a', index=False)

        Error_Sample_List = pd.DataFrame.from_dict(Error_Sample_List)
        np.savetxt('{}/ErrorList_MPIE.txt'.format(save_dir), Error_Sample_List.ErrorSample, fmt='%s')

        return []

    # Multi-PIE testing
    Performance = np.zeros((len(GalleryPose), len(ProbePose)))
    for g_idx, g_pose in enumerate(GalleryPose):
        # 定義Gallery的角度與光源，pose-90 and pose75 use the Illum-5, others use the Illum-7
        if g_pose == '110' or g_pose == '120': g_find = np.where((np.array(Features_List['Illum']) == 5) & (np.array(Features_List['Pose'])==int(g_pose)))[0]
        else: g_find = np.where((np.array(Features_List['Illum']) == 7) & (np.array(Features_List['Pose'])==int(g_pose)))[0]
        g_feature = np.squeeze(np.array(Features_List['Features'])[g_find])
        g_subject = np.squeeze(np.array(Features_List['Subject'])[g_find])

        for p_idx, p_pose in enumerate(ProbePose):
            # probe index
            p_find = np.where(np.array(Features_List['Pose']) == int(p_pose))[0]
            p_feature = np.squeeze(np.array(Features_List['Features'])[p_find])
            p_subject = np.squeeze(np.array(Features_List['Subject'])[p_find])

            # compare gallery and probe
            similarity_temp = cosine_similarity(p_feature.reshape(-1, len(feas)), g_feature.reshape(-1, len(feas)))

            # find the predict gallery results
            g_predict = np.array([g_subject[np.argmax(simi_idx)] for simi_idx in similarity_temp])

            # if probe_subject==predicted_gallery, correct !
            correct_prediction = float('{:.4f}'.format((len(np.where((p_subject - g_predict)==0)[0]) / len(p_subject))))
            Performance[g_idx, p_idx] = correct_prediction

    for g_idx in range(len(GalleryPose)):
        Performance_Results['Epoch-{}'.format(epoch)].append(GalleryPose[g_idx])
        for p_idx in range(len(ProbePose)):
            Performance_Results['{}'.format(ProbePose[p_idx])].append(Performance[g_idx, p_idx])

    save_dir = './{}/Performance_MPIE.csv'.format(save_dir)
    temp = ['Epoch-{}'.format(epoch)]
    for idx in ProbePose: temp.append(idx)
    Performance_Results = pd.DataFrame.from_dict(Performance_Results)
    Performance_Results = Performance_Results.reindex(columns=temp)
    if not os.path.exists(save_dir):
        Performance_Results.to_csv(save_dir, index=False)
    else:
        Performance_Results.to_csv(save_dir, mode='a', index=False)

    Features_List = []
    return []
