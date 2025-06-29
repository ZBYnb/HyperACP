import numpy as np
from imblearn.over_sampling import ADASYN

def balance_classes(Dn, Dp, max_ratio=1.0, max_iterations=50, random_state=42):
    Dn_new = np.copy(Dn)  # 多数类
    Dp_new = np.copy(Dp)  # 少数类（将逐步扩充）
    n_minority_start = len(Dp)

    iteration = 0
    while iteration < max_iterations:
        new_Dn_indices = set(range(len(Dn_new)))
        Q = {}

        for si_idx, si in enumerate(Dn_new):
            distances = np.linalg.norm(Dp_new - si, axis=1)
            sorted_indices = np.argsort(distances)
            Li = sorted_indices[:n_minority_start]

            for idx in Li:
                sj = Dp_new[idx]
                sj_idx = tuple(sj)
                if sj_idx not in Q:
                    Q[sj_idx] = sj
                    new_Dn_indices.discard(si_idx)

        Dn_new = Dn_new[list(new_Dn_indices)]

        # 用 ADASYN 基于 Q 合成新少数类样本（并保留原有）
        if Q:
            Q_array = np.array(list(Q.values()))
            X = np.vstack([Dn_new, Q_array])
            y = np.array([0] * len(Dn_new) + [1] * len(Q_array))

            ada = ADASYN(random_state=random_state)
            X_res, y_res = ada.fit_resample(X, y)

            new_minority = X_res[y_res == 1]
            Dp_new = np.vstack([Dp_new, new_minority])  # ✅ 保留原始 Dp

        # 检查是否达到类别比例目标
        if len(Dn_new) / len(Dp_new) <= max_ratio:
            break
        iteration += 1

    return Dn_new, Dp_new