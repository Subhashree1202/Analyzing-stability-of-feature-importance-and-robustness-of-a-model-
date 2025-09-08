import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torchmetrics
import SingleLayerMLP as slMLP
from ComputeShapley import ComputeShapley as shpl
from sklearn.model_selection import KFold

class ModelSetUp:
    def __init__(self):
        # trainIDs = (pd.read_csv('shaply/trainidx_adrenal.csv')).to_numpy()
        dataset = pd.read_csv("Datasets/adrenalTumorData_red.csv", header=None)
        #dataset=pd.read_csv('Datasets/diabetes.csv')#,header=None)
        #dataset=pd.read_csv('Datasets/ColoradoData_reduced.csv')#,header=None)
        scaler = StandardScaler()
        #weight_set = False
        # mtrx=pd.read_csv('shaply/omegaColorado_fold5.txt',header=None)
        data = dataset.iloc[:, :-1]  # .to_numpy()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        labels = (dataset.iloc[:, -1]).to_numpy()
        data = data.apply(pd.to_numeric, errors="coerce", downcast="float")

        self.feature_names = [names for names in dataset.columns.to_numpy()[:-1]]
        self.input_data = torch.tensor(
            data.to_numpy()
        )  # Batch size of 1 for simplicity
        self.labels = torch.tensor(labels)
        self.num_classes = np.unique(self.labels.numpy()).size

        # Example dimensions
        # input_dim = mtrx.shape[0]
        self.input_dim = len(self.feature_names)
        self.output_dim = self.num_classes

        # Assuming these are your predefined weights (you need to replace these with your actual weights)
        # init_weights_fc = torch.tensor(mtrx.to_numpy())
        #one_hot = torch.nn.functional.one_hot(self.labels).to(torch.float64)

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.input_data, self.labels, test_size=0.2)
        self.model = slMLP.SingleLayerSoftmax(self.input_dim, self.output_dim)

        self.criterion = (
            nn.CrossEntropyLoss()
        )  # Suitable for multi-class classification problems
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001
        )  # Adam optimizer with learning rate of 0.001

    def train_model(self, n, train_data, train_labels):
        # Training loop
        for epoch in range(n):  # Train for 10 epochs
            self.optimizer.zero_grad()  # Zero the gradients
            outputs = self.model(train_data)  # Forward pass
            # print(outputs,train_labels)

            loss = self.criterion(outputs, train_labels)  # Calculate loss

            loss.backward()  # Backward pass to compute gradients
            self.optimizer.step()  # Update model parameters using Adam

    def test_model(self, test_data, test_labels):
        # Forward pass to get predictions (raw scores)
        predictions_raw = self.model(test_data)
        # Convert raw scores into class indices (argmax along dimension 1, which represents classes)
        _, predicted_classes = torch.max(predictions_raw, dim=1)

        accuracy_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        accuracy = accuracy_metric(predicted_classes, test_labels)
        # print('test accuracy:')
        print(accuracy)

    def calc_CIP_Sap(self, wt_, sp_, fold):
        weight_layer1 = 0
        x1_ = 0
        if wt_:
            weight_layer1 = self.model.fc.weight.detach().numpy()
            np.savetxt(
                "output\WeightMatrix_AdrenalFold" + str(fold) + ".txt",
                weight_layer1,
                fmt="%s",
                delimiter=",",
            )

        if sp_:
            shpl_val = shpl()
            x1_ = shpl_val.shap_profile(
                feature_names=self.feature_names,
                input_data=self.input_data,
                model=self.model,
                criterion=self.criterion,
            )
            np.savetxt(
                "output\ShapeProfileFold" + str(fold) + ".txt",
                x1_,
                fmt="%s",
                delimiter=",",
            )

        return weight_layer1, x1_

    def train_inFold(self, fold, epoch, exec_num):
        calc_wt=[]
        calc_sp=[]
        if fold > 1:
            kf = KFold(n_splits=fold, shuffle=True)
            trainID_fold = []
            testID_fold = []
            fold_datasets = {}
            fold_labels = {}
            # trainIDs = (pd.read_csv('shaply/trainidx_adrenal.csv',header=None)).to_numpy()
            # for train_IDs in trainIDs:
            #    cv_result = (model,data[train_IDs-2],labels[train_IDs-2],cv=3)
            for fold, (trainID, testID) in enumerate(kf.split(self.input_data)):
                # traintest_indices[fold]=[trainID,testID]
                trainID_fold.append(trainID)
                testID_fold.append(trainID)
                # print([input_data[i] for i in trainID])
                fold_datasets[fold] = [
                    [self.input_data[i] for i in trainID],
                    [self.input_data[i] for i in testID],
                ]
                fold_labels[fold] = [
                    [self.labels[i] for i in trainID],
                    [self.labels[i] for i in testID],
                ]
                # print(fold_labels)
            for fold in range(len(fold_datasets.keys())):
                # print(model(torch.stack(fold_datasets[fold][0])))
                # print(weight_layer1)
                self.model.fc.weight.data=(torch.randn(self.model.fc.weight.shape))
                self.train_model(
                    epoch,
                    torch.stack(fold_datasets[fold][0]),
                    torch.stack(fold_labels[fold][0]),
                )
                self.test_model(
                    torch.stack(fold_datasets[fold][0]),
                    torch.stack(fold_labels[fold][0]),
                )
                wt_,sp_=(self.calc_CIP_Sap(True, True, fold))
                calc_wt.append(wt_)
                calc_sp.append(sp_)

                    #self.model = slMLP.SingleLayerSoftmax(self.input_dim, self.output_dim)
            #pd.DataFrame(trainID_fold).to_csv(
            #    "output\AdrenalFold5_trainIDs-wt", index=False, header=False
            #)
            #pd.DataFrame(trainID_fold).to_csv(
            #    "output\AdrenalFold5_testIDs-wt", index=False, header=False
            #)

        elif fold == 1:
            # print(weight_layer1,mtrx,weight_layer2)
            self.train_model(epoch, self.X_train, self.y_train)
            self.test_model(self.X_valid, self.y_valid)
            wt_,sp_=(self.calc_CIP_Sap(True, True, exec_num))
            calc_wt.append(wt_)
            calc_sp.append(sp_)

        return calc_wt, calc_sp
        # return wght,sap



