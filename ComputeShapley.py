import itertools as it
import torch
from math import factorial as fctr


class ComputeShapley:

    def featureSet_Permtns(self,feature_i,feature_names):
        feature_set = []
        feature_name = feature_names.copy()
        del feature_name[feature_i]
        for i in range(len(feature_name)+1):
            feature_set += it.combinations(feature_name,i)
        return feature_set
    

    def featureSet_PermtnsValues(self,feature_set,n,feature_names,input_data):
        instance_set = []
        #label_set = []

        for i in (range(len(feature_set))):
            data1 = input_data[n].detach().numpy().copy()
            indx = []
            for j in range(len(feature_set[i])):
                #print(j)
                indx.append(feature_names.index(feature_set[i][j]))
            
            for k in range(len(feature_names)):
                if k not in indx:
                    data1[k] = 0
                    #print(k, data1)
            #print(i,data1)
            instance_set.append( data1)
            #label_set.append(labels[n])
        return instance_set
    
    def feature_setup(self,feature_i,n,feature_names,input_data):
        featureSet = self.featureSet_Permtns(feature_i,feature_names)
        instance_set = self.featureSet_PermtnsValues(featureSet,n,feature_names,input_data)
        featureSet_WithFeature = self.featureSet_PermtnsValues(featureSet,n,feature_names,input_data)
        for i in range(len(featureSet)):
            featureSet_WithFeature[i][feature_i] = input_data[n][feature_i]

        counts=[len(feature_names)-list(x).count(0) for x in instance_set]
        d=len(feature_names)-1
        prefactor = torch.Tensor([(fctr(x)*fctr(d-x)/fctr(d)) for x in counts if x>0])

        return featureSet, featureSet_WithFeature, instance_set, prefactor

    def shap_value(self,feature_i,n,feature_names,input_data,model,criterion):
        featureSet, featureSet_WithFeature, instance_set, prefactor = self.feature_setup(feature_i,n,feature_names,input_data)
        #print(instance_set)
        
        sum_wof=0
        permtns_pred = model(torch.tensor(instance_set)).detach()
        permtns_pred_withFeature = model(torch.tensor(featureSet_WithFeature)).detach()

        sum_wof = sum(prefactor*(criterion(permtns_pred, permtns_pred_withFeature)))
        sum_ = sum_wof/len(featureSet)
        
        return sum_
    

    def shap_profile(self,feature_names,input_data,model,criterion):
        shap=[0 for i in range(len(feature_names))]

        for i in range (len(feature_names)):
            for j in range (len(input_data)):
                shap[i]+=abs(self.shap_value(i,j,feature_names,input_data,model,criterion))
            shap[i]=shap[i]/len(input_data)
        return(shap)

