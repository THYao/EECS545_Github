class SVM_BGD:
    
    def __init__(self,eta=0.001,C=3,num_iter=100):
        self._eta = eta
        self._C = C
        self._num_iter = num_iter
        self._acc_list = np.zeros(num_iter)
    
    def alpha(self,j):
        return self._eta / (1 + j * self._eta)
    
    def fit(self,X,t):
        self._X = X
        self._t = t
        w = np.zeros(X.shape[1])
        b = 0
        for ix in range(self._num_iter):
            summation_1 = 0
            summation_2 = 0
            for inx,row in X.iterrows():
                row = row.values
                if (1-row.dot(w) + b )> 0 :
                    summation_1 += t.ix[inx].values[0] * row
                    summation_2 += t.ix[inx].values[0]
            w_grad = w + 3 * summation_1
            b_grad = 3 * summation_2
#             indicator_mask = np.multiply(np.array(t).flatten(),np.array(X.dot(w) + b ).flatten())
#             indicator_mask = (pd.Series((1- indicator_mask),index=X.index)>0)
#             X_subset = np.array(X[indicator_mask])
#             t_subset =  self.make_diagonal_matrix(X_subset,indicator_mask)
#             X_subset = X_subset.dot(t_subset)
#             w_grad = w + self._C * X_subset.sum(axis=0)
#             b_grad = self._C * (-1 * t[indicator_mask]).sum()
            w = w - self.alpha(ix) * w_grad
            b = b - self.alpha(ix) * b_grad
            self._wprimal = w
            self._bprimal = b
            self.log_acc(ix)

    def get_w(self):
        return self._wprimal
    
    def get_b(self):
        return self._bprimal
    
    def log_acc(self,ix):
        self._acc_list[ix] = self.get_acc()
    
    def get_acc(self):
        self._tst = pd.DataFrame(pd.Series((self._X.dot(self.get_w()) + self.get_b()),index=self._t.index))
        acc = ((self._t > 0) == (self._tst > 0 )).sum() / self._t.shape[0]
        return acc
        
    def get_acc_list(self):
        return self._acc_list
    
    def make_diagonal_matrix(self,X_sub,indicator_mask):
        if X_sub.shape[0]<self._X.shape[1]:
            n = X_sub.shape[0]
            q = X_sub.shape[1]
            td = np.diag(np.array(self._t).flatten())[:self._X.shape[1],:self._X.shape[1]]
            return td
        else:
            td = np.diag(np.array(self._t[indicator_mask]).flatten())[:self._X.shape[1],:self._X.shape[1]]
            return td