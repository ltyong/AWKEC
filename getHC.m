function A = getHC(X,bound)

E = X;
E(X >= bound) = 0;
A = X - E;