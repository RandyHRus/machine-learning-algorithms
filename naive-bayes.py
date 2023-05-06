def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n 


        groupcount = np.zeros(shape = (d,self.num_classes))
        for i in range(n):
            for j in range(d):
                if (X[i,j] == 1):
                    groupcount[j,y[i]] += 1
                
        for i in range(self.num_classes):
            for j in range(d):
                word = groupcount[j,i]/counts[i]
                groupcount[j,i] = word
        
        p_xy = groupcount
            
        self.p_y = p_y
        self.p_xy = p_xy
        self.not_p_xy = 1 - p_xy