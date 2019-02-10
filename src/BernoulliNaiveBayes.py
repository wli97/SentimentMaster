class BNB(object):

    def __init__(self, count_vect):
        self.sum = None
        self.probs = None
        self.feat = count_vect.get_feature_names()
        self.vect = count_vect

    def train(self, x, y):

        # log(P(y))
        N = float(len(y))
        self.sum = {0:N-sum(y),1:0.+sum(y)}

        """Compute log( P(X|Y) )
           Use Laplace smoothing
        """
        self.probs = {sign: {f: 1. for f in self.feat} for sign in self.sum}
        count = 0
        print("start")
        # Step through each document
        for review, sign in zip(x,y):
            count += 1
            if(count > 300): break
            for word in self.feat:
                if word in review:
                    self.probs[sign][word] += 1.0
        
        print("end")
        # Now, compute log probs
        for sign in self.probs:
            self.probs[sign] = {k: v / self.sum[sign]+2 for k, v in self.probs[sign].items()}

    def predict(self, review):
        """Make a prediction from text
        """
        words = review.lower().split()
        # Perform MAP estimation
        log_sum0 = math.log(self.sum[0]/(self.sum[0]+self.sum[1]))
        log_sum1 = math.log(self.sum[1]/(self.sum[0]+self.sum[1]))
        for f in self.feat:
            prob1 = self.probs[1][f]
            prob0 = self.probs[0][f]
            log_sum1 += math.log(prob1)
            log_sum0 += math.log(prob0)
        prediction = 1
        if log_sum1 < log_sum0: prediction = 0
        return prediction
