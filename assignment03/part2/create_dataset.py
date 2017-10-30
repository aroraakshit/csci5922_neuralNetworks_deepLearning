import numpy as np

def familytree():
    def bitvec(ix,nbit):
        out = []
        for i in range(nbit):
            out.append((i==ix)+0.0)
        return np.array(out)

    names = [ "Christopher", "Andrew", "Arthur", "James", "Charles", "Colin", "Penelope", "Christine", "Victoria", "Jennifer", "Margaret", "Charlotte", "Roberto", "Pierro", "Emilio", "Marco", "Tomaso", "Alfonso", "Maria", "Francesca", "Lucia", "Angela", "Gina", "Sophia"]
    relations = [ "husband", "wife", "son", "daughter", "father", "mother", "brother", "sister", "nephew", "niece", "uncle", "aunt"]

    dataset = []
    with open('relations.txt','r') as f:
        for line in f:
            sline = line.split();
            p1 = names.index(sline[0])
            r = relations.index(sline[1])
            p2 = names.index(sline[2])
            d = [ sline[0]+'-'+sline[1]+'-'+sline[2],
                  bitvec(p1,len(names)),
                  bitvec(r,len(relations)),
                  bitvec(p2,len(names)) ]
            dataset.append(d)
    return dataset
#HEY ! btw, if I use Relu for training, what happens when i calculate accuracy?
'''
right now i am doing this z'this is for sigmoid right?" , relu is 0 if x<0 and x if x>0
so use 0 as your threshold
but since this is a binary classification, it makes more sense to use sigmoid (it'll squash the inpt between 0
and 1) or (tanh(x)+1.0)/2.0 or if tanh(x)>0 return 1 else

so, is "accuracy calculation" dependent on activation function used while training? y/n?
if it is, then is there any way I can see the mapping of accuracy calc with activation function? logistic regression, (which uses sigmoid), so u should get the same results. :|
take output , then train on logreg
since relu is the last layer
use if relu(x)==0, return 0 else return 1
network would've learnt to put a large netative output for -ve class and a large +ve output for +ve class

okay later! I need to spend more time on this.
i have been messing this up: P

output[output < 0.5] = 0.0
output[output >= 0.5] = 1.0
accuracy = 100 - (np.sum(np.abs(output - testing_data_y)) / len(points_testing) * 100)
return accuracy

__
'''

if __name__ == "__main__":
    print familytree()
