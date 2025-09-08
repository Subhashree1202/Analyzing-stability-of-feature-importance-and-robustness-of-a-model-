import pandas as pd
from scipy import stats
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from MLPConfig import ModelSetUp


class Visualize:
    def computeWTandSH(self):
        mdlcnfg = ModelSetUp()
        cor_data_wt = []
        cor_data_sap = []
        for i in range(4):
            wt_, sp_ = mdlcnfg.train_inFold(5, 250, i)
            for i in range(len(wt_)):
                cip_wt_ = [
                    round(j / sum(np.sum(abs(wt_[i].dot(wt_[i].T)), axis=1)), 3)
                    for j in np.sum(abs(wt_[i].dot(wt_[i].T)), axis=1)
                ]
                cor_data_wt.append(cip_wt_)
                cor_data_sap.append(sp_[i])
        plt.plot(cor_data_wt)
        plt.show()
        plt.plot(cor_data_sap)
        plt.show()
        return cor_data_wt, cor_data_sap

    def spearman_corr(self):
        cor_data_wt, cor_data_sap = self.computeWTandSH()
        # Calculate Spearman correlation matrix
        corr_matrix1, _ = stats.spearmanr(cor_data_wt, axis=1)
        corr_matrix2, _ = stats.spearmanr(cor_data_sap, axis=1)
        print(corr_matrix1, corr_matrix2)
        return corr_matrix1, corr_matrix2
    
    def calc_affinity(self,corr_matrix1, corr_matrix2):   
        #corr_matrix1, corr_matrix2 = self.spearman_corr()
        # Train the Affinity Propagation algorithm
        af_w = AffinityPropagation(affinity="precomputed").fit(abs(corr_matrix1))
        cluster_labels_w = af_w.labels_
        cluster_centers_indices_wt = af_w.cluster_centers_indices_
        n_clusters_wt = len(cluster_centers_indices_wt)
        unique_clusters_w = np.unique(cluster_labels_w)

        plt.close("all")
        plt.figure(1)
        plt.clf()

        colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, len(unique_clusters_w))))

        for k, col in zip(range(n_clusters_wt), colors):
            class_members = cluster_labels_w == k
            cluster_center = corr_matrix1[cluster_centers_indices_wt[k]]
            plt.scatter(
                corr_matrix1[class_members, 0], corr_matrix1[class_members, 1], color=col["color"], marker="."
            )
            plt.scatter(
                cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o"
            )
            for x in corr_matrix1[class_members]:
                plt.plot(
                    [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"]
                )

        plt.title("Estimated number of clusters using Lambda profiles: %d" % n_clusters_wt)
        plt.show()



        af_sh = AffinityPropagation(affinity="precomputed").fit(abs(corr_matrix2))
        cluster_labels_sh = af_sh.labels_
        cluster_centers_indices = af_sh.cluster_centers_indices_
        n_clusters_ = len(cluster_centers_indices)
        unique_clusters_sh = np.unique(cluster_labels_sh)
        print(unique_clusters_w)
        print(unique_clusters_sh)
        # plt.scatter(data.data[:, 0], data.data[:, 1], c=cluster_labels)
        # plt.show()

        plt.close("all")
        plt.figure(1)
        plt.clf()

        colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, len(unique_clusters_sh))))

        for k, col in zip(range(n_clusters_), colors):
            class_members = cluster_labels_sh == k
            cluster_center = corr_matrix2[cluster_centers_indices[k]]
            plt.scatter(
                corr_matrix2[class_members, 0], corr_matrix2[class_members, 1], color=col["color"], marker="."
            )
            plt.scatter(
                cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o"
            )
            for x in corr_matrix2[class_members]:
                plt.plot(
                    [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"]
                )

        plt.title("Estimated number of clusters using Shapley profiles: %d" % n_clusters_)
        plt.show()



    def tsne_vis(self,corr_matrix1,corr_matrix2):
        #corr_matrix1, corr_matrix2 = self.spearman_corr()
        corr1 = np.ones_like(corr_matrix1) - corr_matrix1
        cor2 = (corr_matrix2 - (np.ones_like(corr_matrix2) - np.identity(len(corr_matrix2))) * 0.001)
        corr2 = np.ones_like(cor2) - cor2
        print(corr_matrix2)
        tsne = TSNE(n_components=2, perplexity=3, random_state=2)
        X_tsne = tsne.fit_transform(corr1)
        df_subset = pd.DataFrame()
        df_subset["tsne-2d-one"] = X_tsne[:, 0]
        df_subset["tsne-2d-two"] = X_tsne[:, 1]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            palette=sns.color_palette("hls", 10),
            data=df_subset,
            legend="full",
            alpha=0.3,
        )
        plt.show()
        X_tsne1 = tsne.fit_transform(corr2)
        df_subset = pd.DataFrame()
        df_subset["tsne-2d-one"] = X_tsne1[:, 0]
        df_subset["tsne-2d-two"] = X_tsne1[:, 1]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            palette=sns.color_palette("hls", 10),
            data=df_subset,
            legend="full",
            alpha=0.3,
        )
        plt.show()
    def confsnMtrx_vis(self):
        corr_matrix1, corr_matrix2 = self.spearman_corr()
        # Convert to DataFrame for better visualization
        corr_df_dt = pd.DataFrame(
            corr_matrix1,
            index=[i for i in range(len(corr_matrix1))],
            columns=[i for i in range(len(corr_matrix1))],
        )
        corr_df_sp = pd.DataFrame(
            corr_matrix2,
            index=[i for i in range(len(corr_matrix2))],
            columns=[i for i in range(len(corr_matrix2))],
        )
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df_sp, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Spearman Correlation Matrix shapley")
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df_dt, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Spearman Correlation Matrix Weight Matrix")
        plt.show()

        self.calc_affinity(corr_matrix1,corr_matrix2)
        #self.tsne_vis(corr_matrix1, corr_matrix2)

viz=Visualize()
viz.confsnMtrx_vis()
