import math

class BNB():

    def __init__(self, count_vect):
        self.sum = None
        self.probs = None
        self.feat = count_vect.get_feature_names()
        self.vect = count_vect

    def train(self, x, y):
        
        N = float(len(y))
        self.sum = {0:N-sum(y),1:0.+sum(y)}

        # Initialize the count for both good/bad reviews with 1 for Laplace numerator
        self.probs = {sign: {f: 1. for f in self.feat} for sign in self.sum}
        
        # Step through each document and increment by 1 if feature found in document i
        for review, sign in zip(x,y):
            review = self.to_set(review)
            for word in self.feat:
                if word in review:
                    self.probs[sign][word] += 1.0
                    
        # Compute probs for each feature in feature set with Laplace smoothing
        for sign in self.probs:
            self.probs[sign] = {k: v/(self.sum[sign]+2) for k, v in self.probs[sign].items()}

    def predict(self, review):
        # Compute the sum of log probability for the given review
        words = self.to_set(review)
        log_sum0 = math.log(self.sum[0]/(self.sum[0]+self.sum[1]))
        log_sum1 = math.log(self.sum[1]/(self.sum[0]+self.sum[1]))
        for f in self.feat:
            prob1 = self.probs[1][f]
            prob0 = self.probs[0][f]
            if f not in words:
                prob1 = 1-prob1
                prob0 = 1-prob0
            log_sum1 += math.log(prob1)
            log_sum0 += math.log(prob0)
        # Compare if the likelihood is higher for y=0 or y=1 given x
        prediction = 1
        if log_sum1 < log_sum0: prediction = 0
        return prediction

    def to_set(self, review):
        return set([word.lower() for word in review.split()])
