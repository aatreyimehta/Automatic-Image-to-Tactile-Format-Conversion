#importing python packages to begin with
import random
import numpy as np
from math import comb

#given a set of control points P, the function will generate point on curve at a given value of parameter x
def generate_bezier(x, P):
    n = P.shape[0]   #n is equal to number of control points
    b = 0   #variable b is initialized to 0

    for i in range(n):
        a = comb(n,i)*((1-x)**(n-i))*(x**i)
        b = b + a*P[i].reshape(1,-1)

    if b.shape[1] == 1:
        b = np.concatenate((x,b), axis=1)
    print('b',b)
    return b

if __name__ == "__main__":
    P_1D = 0.25
    CROSS = []
    NUM_SAMPLES = 50000   #number of samples to generate
    for i in range(NUM_SAMPLES):
        x = np.linspace(0, 1, 10000).reshape(-1,1)
        P = np.array([[random.randint(-20,20), random.randint(-20,20)] for i in range(random.randint(2,20))])
        
        if random.random() <= P_1D:
            P = P.reshape(-1,1).flatten()[::2]
        b = generate_bezier(x, P)
        print(f"{i+1}.npy generated")
        np.save(f"./points/{i+1}.npy", b)   #saves b as a numpy array in a file named "{i+1}.npy"
