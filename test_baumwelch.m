%% true value
T = [
    0.1 0.7 0.2;
    0.2 0.1 0.7;
    0.7 0.2 0.1];

emission = [
    0.9 0.1;
    0.6 0.4;
    0.1 0.9];

pi_i = [0.6, 0.3, 0.1];

[~, data] = msmgenerate(1000, T, emission, pi_i);

%% initial value
T0 = [
    0.1  0.1 0.8;
    0.8  0.1 0.1;
    0.3  0.2 0.5];

%% Baum-Welch
[T1, emission1, pi1_i] = msmbaumwelch(data, T0, emission, pi_i);

T
T0
T1

save test_baumwelch.mat;

